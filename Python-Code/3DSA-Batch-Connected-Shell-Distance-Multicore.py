import os
import math
import numpy as np
import xarray as xr
import scipy.ndimage
from pathlib import Path
import netCDF4 as nc
import cc3d
import gc
from scipy.spatial import cKDTree
import time
import sys
import multiprocessing

# --- Configurations ---
ql_dilation = 1
num_cores = 3  # Configured for 3 parallel worker processes
input_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir.mkdir(parents=True, exist_ok=True)

export_registry = {
    "true_shell_distance.nc": ("distance", "f4"),
    "true_shell_distance_vert.nc": ("distance", "f4"),
    "true_relative_shell_altitude.nc": ("relative_altitude", "f4"),
    "true_shell_distance_horz.nc": ("distance", "f4"),
}

# --- Multiprocessing Worker Function ---
def process_connected_ql_timestep(t, input_dir, grid_distance, nz, ny, nx, z_real, box_limits, expansion):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    start_time = time.time()
    
    # 1. Read single timestep locally
    with xr.open_dataset(input_dir / "ql_mask.nc", decode_times=False) as ds_ql:
        ql_raw = ds_ql.ql_mask.isel(time=t).values.astype(bool)
    with xr.open_dataset(input_dir / "shell_mask.nc", decode_times=False) as ds_shell:
        sub_outline = ds_shell.shell_mask.isel(time=t).values.astype(bool)

    # Initialize tracking targets
    local_true_dist = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_true_dist_vert = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_true_shell_relative_alt = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_true_dist_horz = np.full_like(ql_raw, np.nan, dtype=np.float32)

    # 2. Extract Connected Components Labeled Regions
    padded_ql_raw = np.pad(ql_raw, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    temp_ql_labels = cc3d.connected_components(padded_ql_raw, connectivity=6, periodic_boundary=True)
    ql_labels = temp_ql_labels[1:-1, :, :]
    
    initial_dilated_ql_labels = ql_labels.copy()
    padded_labels = np.pad(initial_dilated_ql_labels, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    padded_labels = np.pad(padded_labels, ((0, 0), (ql_dilation, ql_dilation), (ql_dilation, ql_dilation)), mode='wrap')
    
    for _ in range(ql_dilation):
        padded_labels = scipy.ndimage.grey_dilation(padded_labels, footprint=expansion)
    dilated_ql_labels = padded_labels[1:-1, ql_dilation:-ql_dilation, ql_dilation:-ql_dilation]

    shell_parent_ids = np.where(sub_outline, dilated_ql_labels, 0)

    # 3. Dilation Traversal loop
    while True:
        travel_mask = sub_outline & (shell_parent_ids == 0)
        if not np.any(travel_mask):
            break
        
        padded_shell_ids = np.pad(shell_parent_ids, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        padded_shell_ids = np.pad(padded_shell_ids, ((0, 0), (1, 1), (1, 1)), mode='wrap')
        padded_expanded = scipy.ndimage.grey_dilation(padded_shell_ids, footprint=expansion)
        ql_label_expanded = padded_expanded[1:-1, 1:-1, 1:-1]

        shell_parent_ids[travel_mask] = ql_label_expanded[travel_mask]

    active_cloud_ids = np.unique(shell_parent_ids)
    active_cloud_ids = active_cloud_ids[active_cloud_ids != 0]
    total_clouds = len(active_cloud_ids)

    # 4. Process Individual Cloud Geometry via Trees
    # We enforce workers=1 inside the sub-process so it doesn't fight over threads
    for idx, cloud_id in enumerate(active_cloud_ids):
        parent_cloud = (ql_labels == cloud_id)
        cloud_z_pts, cloud_y_pts, cloud_x_pts = np.where(parent_cloud)
        if len(cloud_x_pts) == 0:
            continue

        cloud_mask = (shell_parent_ids == cloud_id)
        shell_z_pts, shell_y_pts, shell_x_pts = np.where(cloud_mask)
        if len(shell_x_pts) == 0:
            continue
        
        cloud_coords = np.column_stack((
            z_real[cloud_z_pts],
            cloud_y_pts * grid_distance,
            cloud_x_pts * grid_distance    
        ))

        shell_coords = np.column_stack((
            z_real[shell_z_pts],
            shell_y_pts * grid_distance,
            shell_x_pts * grid_distance
        ))

        tree = cKDTree(cloud_coords, boxsize=box_limits)
        distances, closest_cloud_indices = tree.query(shell_coords, k=1, workers=1)

        valid_idx_mask = closest_cloud_indices < len(cloud_z_pts)
        local_true_dist[shell_z_pts, shell_y_pts, shell_x_pts] = distances

        matched_cloud_z = cloud_z_pts[closest_cloud_indices[valid_idx_mask]]
        matched_cloud_y = cloud_y_pts[closest_cloud_indices[valid_idx_mask]]
        matched_cloud_x = cloud_x_pts[closest_cloud_indices[valid_idx_mask]]

        dy = shell_y_pts - matched_cloud_y
        dx = shell_x_pts - matched_cloud_x
        dy = dy - ny * np.round(dy / ny)
        dx = dx - nx * np.round(dx / nx)

        local_true_dist_vert[shell_z_pts, shell_y_pts, shell_x_pts] = np.abs(z_real[shell_z_pts] - z_real[matched_cloud_z])
        local_true_shell_relative_alt[shell_z_pts, shell_y_pts, shell_x_pts] = z_real[shell_z_pts] - z_real[matched_cloud_z]
        local_true_dist_horz[shell_z_pts, shell_y_pts, shell_x_pts] = np.sqrt((dx * grid_distance)**2 + (dy * grid_distance)**2)

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t, {
        "true_shell_distance.nc": local_true_dist,
        "true_shell_distance_vert.nc": local_true_dist_vert,
        "true_relative_shell_altitude.nc": local_true_shell_relative_alt,
        "true_shell_distance_horz.nc": local_true_dist_horz,
        "duration": elapsed_str,
        "cloud_count": total_clouds
    }


# --- Execution Controller Guard ---
if __name__ == '__main__':
    print("Verifying target datasets...")
    path_ds_ql_mask = input_dir / "ql_mask.nc"
    path_ds_shell_mask = input_dir / "shell_mask.nc"
    
    if not path_ds_ql_mask.is_file() or not path_ds_shell_mask.is_file():
        print("❌ ERROR: Input simulation arrays are missing.", file=sys.stderr)
        sys.exit(1)

    # Establish environment parameters
    with xr.open_dataset(path_ds_ql_mask, decode_times=False) as ds_ql_mask:
        grid_distance = float(ds_ql_mask.x[1] - ds_ql_mask.x[0])
        num_times = int(ds_ql_mask.time.size)
        nz, ny, nx = ds_ql_mask.ql_mask.shape[1:]
        time_vals = ds_ql_mask.time.values
        z_vals = ds_ql_mask.z.values
        y_vals = ds_ql_mask.y.values
        x_vals = ds_ql_mask.x.values

    box_limits = np.array([999999.0, ny * grid_distance, nx * grid_distance])

    expansion = np.zeros((3,3,3), dtype=bool)
    expansion[1, 1, :] = True
    expansion[1, :, 1] = True
    expansion[:, 1, 1] = True

    # 1. Preallocate blank files
    print("Pre-allocating NetCDF file structures on disk...")
    for filename, (var_name, data_type) in export_registry.items():
        file_path = output_dir / filename
        with nc.Dataset(str(file_path), "w", format="NETCDF4") as f:
            f.createDimension("time", num_times)
            f.createDimension("z", nz)
            f.createDimension("y", ny)
            f.createDimension("x", nx)
            
            f.createVariable("time", "f8", ("time",))[:] = time_vals
            f.createVariable("z", "f4", ("z",))[:] = z_vals
            f.createVariable("y", "f4", ("y",))[:] = y_vals
            f.createVariable("x", "f4", ("x",))[:] = x_vals
            
            f.createVariable(var_name, data_type, ("time", "z", "y", "x"), 
                             zlib=True, complevel=4, chunksizes=(1, nz, ny, nx), fill_value=False)

    # 2. Setup Multi-core Pool Execution Structure
    print(f"Spawning Pool with {num_cores} active workers over {num_times} timesteps...")
    pool_args = [(t, input_dir, grid_distance, nz, ny, nx, z_vals, box_limits, expansion) for t in range(num_times)]
    
    open_files = {fname: nc.Dataset(str(output_dir / fname), "a") for fname in export_registry}

    with multiprocessing.Pool(processes=num_cores) as pool:
        # imap_unordered handles dynamic tracking efficiently
        for t, results in pool.starmap(process_connected_ql_timestep, pool_args):
            print(f"Timestep {t}/{num_times - 1} finished in ({results['duration']}) processing {results['cloud_count']} clouds. Committing to files...")
            
            # Safe Single-Threaded Write operations
            open_files["true_shell_distance.nc"].variables["distance"][t, :, :, :] = results["true_shell_distance.nc"]
            open_files["true_shell_distance_vert.nc"].variables["distance"][t, :, :, :] = results["true_shell_distance_vert.nc"]
            open_files["true_relative_shell_altitude.nc"].variables["relative_altitude"][t, :, :, :] = results["true_relative_shell_altitude.nc"]
            open_files["true_shell_distance_horz.nc"].variables["distance"][t, :, :, :] = results["true_shell_distance_horz.nc"]
            
            for open_file in open_files.values():
                open_file.sync()
            
            del results
            gc.collect()

    # 3. Flush and close handles
    for file_obj in open_files.values():
        file_obj.close()

    print("\n✅ All computation and exporting complete (Program is safe to close)")