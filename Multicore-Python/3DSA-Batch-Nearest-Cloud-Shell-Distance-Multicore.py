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
import json

# --- Configurations ---
num_cores = int(os.environ.get("CORE_COUNT", 1))  # Default to 1 core if not specified

# --- Setting up directories from config ---
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

if not CONFIG_PATH.is_file():
    print(f"❌ ERROR: Configuration file missing at: {CONFIG_PATH}", file=sys.stderr)
    sys.exit(1)

# Read json config file
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)

# Extract Paths
source_input_dir = Path(config_data["paths"]["source_input_dir"])
output_dir = Path(config_data["paths"]["output_dir"])
input_dir = output_dir  # Input directory matches output directory for script chain dependencies

# In case directory does not exist
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Initialization Success:")
print(f" -> Source Input Path: {source_input_dir}")
print(f" -> Output Path:       {output_dir}")
print(f" -> Active CPU Cores:  {num_cores}")
print("-" * 50)

export_registry = {
    "nearest_shell_distance.nc": ("distance", "f4"),
    "nearest_shell_distance_vert.nc": ("distance", "f4"),
    "nearest_relative_shell_altitude.nc": ("relative_altitude", "f4"),
    "nearest_shell_distance_horz.nc": ("distance", "f4"),
}

# --- Multiprocessing Worker Function ---
def process_connected_ql_timestep(t, input_dir, grid_distance, nz, ny, nx, z_real, box_limits):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    start_time = time.time()
    
    # 1. Read single timestep locally
    with xr.open_dataset(input_dir / "shell_mask.nc", decode_times=False) as ds_shell_mask:
        shell_mask = ds_shell_mask.shell_mask.isel(time=t).values.astype(np.uint32)
    with xr.open_dataset(input_dir / "cloud_mask.nc", decode_times=False) as ds_cloud_mask:
        cloud_mask = ds_cloud_mask.cloud_mask.isel(time=t).values.astype(np.uint32)

    # Initialize tracking targets
    local_nearest_shell_dist = np.full_like(shell_mask, np.nan, dtype=np.float32)
    local_nearest_shell_dist_vert = np.full_like(shell_mask, np.nan, dtype=np.float32)
    local_nearest_shell_relative_alt = np.full_like(shell_mask, np.nan, dtype=np.float32)
    local_nearest_shell_dist_horz = np.full_like(shell_mask, np.nan, dtype=np.float32)

    # Obtaining cloud surface (eroded to get the surface of clouds rather than compute all points)
    padded_cloud_mask = np.pad(cloud_mask, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    padded_cloud_mask = np.pad(padded_cloud_mask, ((0, 0), (1, 1), (1, 1)), mode='wrap')
    eroded_padded = scipy.ndimage.binary_erosion(padded_cloud_mask)
    eroded_cloud_mask = eroded_padded[1:-1, 1:-1, 1:-1]
    cloud_surface_mask = cloud_mask ^ eroded_cloud_mask

    # Obtaining distances
    all_cloud_z, all_cloud_y, all_cloud_x = np.where(cloud_surface_mask)
    shell_z_pts, shell_y_pts, shell_x_pts = np.where(shell_mask)

    if len(all_cloud_x) > 0 and len(shell_x_pts) > 0:
        # Stack coordinates using physical values for Z, grid meters for X/Y
        global_cloud_coords = np.column_stack((
            z_real[all_cloud_z],
            all_cloud_y * grid_distance,
            all_cloud_x * grid_distance
        ))
        
        global_shell_coords = np.column_stack((
           z_real[shell_z_pts],            
            shell_y_pts * grid_distance,
            shell_x_pts * grid_distance
        ))

        # Build tree and query shell coordinates
        global_tree = cKDTree(global_cloud_coords, boxsize=box_limits)
        global_distances, closest_surface_indices = global_tree.query(global_shell_coords, k=1, workers=-1)

        valid_idx_mask = closest_surface_indices < len(all_cloud_z)

        # Assign the true Euclidean distance
        local_nearest_shell_dist[shell_z_pts, shell_y_pts, shell_x_pts] = global_distances

        # Extract components for directional distance tracking
        matched_surface_z = all_cloud_z[closest_surface_indices[valid_idx_mask]]
        matched_surface_y = all_cloud_y[closest_surface_indices[valid_idx_mask]]
        matched_surface_x = all_cloud_x[closest_surface_indices[valid_idx_mask]]

        # Calculate directional differences
        physical_delta_z = z_real[shell_z_pts] - z_real[matched_surface_z]
        delta_y = shell_y_pts - matched_surface_y
        delta_x = shell_x_pts - matched_surface_x

        # Apply periodic wrapping adjustments horizontally
        delta_y = delta_y - ny * np.round(delta_y / ny)
        delta_x = delta_x - nx * np.round(delta_x / nx)

        # Store directional metrics
        local_nearest_shell_dist_vert[shell_z_pts, shell_y_pts, shell_x_pts] = np.abs(physical_delta_z)
        local_nearest_shell_relative_alt[shell_z_pts, shell_y_pts, shell_x_pts] = physical_delta_z
        local_nearest_shell_dist_horz[shell_z_pts, shell_y_pts, shell_x_pts] = np.sqrt(
            (delta_x * grid_distance)**2 + (delta_y * grid_distance)**2
        )


    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t, {
        "nearest_shell_distance.nc": local_nearest_shell_dist,
        "nearest_shell_distance_vert.nc": local_nearest_shell_dist_vert,
        "nearest_relative_shell_altitude.nc": local_nearest_shell_relative_alt,
        "nearest_shell_distance_horz.nc": local_nearest_shell_dist_horz,
        "duration": elapsed_str,

    }


# --- Execution Controller Guard ---
if __name__ == '__main__':
    print("Verifying target datasets...")
    path_ds_shell_labels = input_dir / "shell_labels.nc"
    path_ds_cloud_mask = input_dir / "cloud_mask.nc"
    
    if not path_ds_shell_labels.is_file():
        print(f"❌ ERROR: Missing required file: {path_ds_shell_labels}", file=sys.stderr)
        sys.exit(1)
    if not path_ds_cloud_mask.is_file():
        print(f"❌ ERROR: Missing required file: {path_ds_cloud_mask}", file=sys.stderr)
        sys.exit(1)

    # Establish environment parameters
    with xr.open_dataset(path_ds_cloud_mask, decode_times=False) as ds_cloud_mask:
        grid_distance = float(ds_cloud_mask.x[1] - ds_cloud_mask.x[0])
        num_times = int(ds_cloud_mask.time.size)
        nz, ny, nx = ds_cloud_mask.cloud_mask.shape[1:]
        time_vals = ds_cloud_mask.time.values
        z_vals = ds_cloud_mask.z.values
        y_vals = ds_cloud_mask.y.values
        x_vals = ds_cloud_mask.x.values

    box_limits = np.array([999999.0, ny * grid_distance, nx * grid_distance])

    # 1. Preallocate blank files
    open_files = {}
    print("Pre-allocating NetCDF file structures on disk...")
    for filename, (var_name, data_type) in export_registry.items():
        file_path = output_dir / filename
        f = nc.Dataset(str(file_path), "w", format="NETCDF4")
        open_files[filename] = f

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
    pool_args = [(t, input_dir, grid_distance, nz, ny, nx, z_vals, box_limits) for t in range(num_times)]
    
    #open_files = {fname: nc.Dataset(str(output_dir / fname), "a") for fname in export_registry}

    with multiprocessing.Pool(processes=num_cores) as pool:
        for t, results in pool.starmap(process_connected_ql_timestep, pool_args):
            print(f"Timestep {t}/{num_times - 1} finished in ({results['duration']}). Committing to files...")
            
            # Safe Single-Threaded Write operations
            open_files["nearest_shell_distance.nc"].variables["distance"][t, :, :, :] = results["nearest_shell_distance.nc"]
            open_files["nearest_shell_distance_vert.nc"].variables["distance"][t, :, :, :] = results["nearest_shell_distance_vert.nc"]
            open_files["nearest_relative_shell_altitude.nc"].variables["relative_altitude"][t, :, :, :] = results["nearest_relative_shell_altitude.nc"]
            open_files["nearest_shell_distance_horz.nc"].variables["distance"][t, :, :, :] = results["nearest_shell_distance_horz.nc"]
            
            for open_file in open_files.values():
                open_file.sync()
            
            del results
            gc.collect()

    # 3. Flush and close handles
    for file_obj in open_files.values():
        file_obj.close()

    print("\n✅ All computation and exporting complete (Program is safe to close)")