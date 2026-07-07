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

# ─── OVERRIDE SYSTEM TMPDIR WITH CONFIG PATH ──────────────────────────
custom_tmp_dir = output_dir / "tmp"
custom_tmp_dir.mkdir(parents=True, exist_ok=True)

os.environ["TMPDIR"] = str(custom_tmp_dir)
# ──────────────────────────────────────────────────────────────────────

print(f"Initialization Success:")
print(f" -> Source Input Path: {source_input_dir}")
print(f" -> Output Path:       {output_dir}")
print(f" -> Active CPU Cores:  {num_cores}")
print("-" * 50)

export_registry = {
    "true_shell_distance.nc": ("distance", "f4"),
    "true_shell_distance_vert.nc": ("distance", "f4"),
    "true_relative_shell_altitude.nc": ("relative_altitude", "f4"),
    "true_shell_distance_horz.nc": ("distance", "f4"),
}

# --- Multiprocessing Worker Function ---
def process_connected_ql_timestep(t, input_dir, grid_distance, nz, ny, nx, z_real, box_limits):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    start_time = time.time()
    
    # 1. Read single timestep locally
    with xr.open_dataset(input_dir / "shell_labels.nc", decode_times=False) as ds_shell_labels:
        shell_labels = ds_shell_labels.shell_labels.isel(time=t).values.astype(np.uint32)
    with xr.open_dataset(input_dir / "cloud_labels.nc", decode_times=False) as ds_cloud_labels:
        cloud_labels = ds_cloud_labels.cloud_labels.isel(time=t).values.astype(np.uint32)

    # Initialize tracking targets
    local_true_dist = np.full_like(shell_labels, np.nan, dtype=np.float32)
    local_true_dist_vert = np.full_like(shell_labels, np.nan, dtype=np.float32)
    local_true_shell_relative_alt = np.full_like(shell_labels, np.nan, dtype=np.float32)
    local_true_dist_horz = np.full_like(shell_labels, np.nan, dtype=np.float32)

    active_cloud_ids = np.unique(shell_labels)
    active_cloud_ids = active_cloud_ids[active_cloud_ids != 0]
    total_clouds = len(active_cloud_ids)

    # 2. Process Individual Cloud Geometry via Trees
    # We enforce workers=1 inside the sub-process so it doesn't fight over threads
    for idx, cloud_id in enumerate(active_cloud_ids):
        cloud_mask = (cloud_labels == cloud_id)
        cloud_z_pts, cloud_y_pts, cloud_x_pts = np.where(cloud_mask)
        if len(cloud_x_pts) == 0:
            continue

        shell_mask = (shell_labels == cloud_id)
        shell_z_pts, shell_y_pts, shell_x_pts = np.where(shell_mask)
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
    path_ds_shell_labels = input_dir / "shell_labels.nc"
    path_ds_cloud_labels = input_dir / "cloud_labels.nc"
    
    if not path_ds_shell_labels.is_file():
        print(f"❌ ERROR: Missing required file: {path_ds_shell_labels}", file=sys.stderr)
        sys.exit(1)
    if not path_ds_cloud_labels.is_file():
        print(f"❌ ERROR: Missing required file: {path_ds_cloud_labels}", file=sys.stderr)
        sys.exit(1)

    # Establish environment parameters
    with xr.open_dataset(path_ds_cloud_labels, decode_times=False) as ds_cloud_labels:
        grid_distance = float(ds_cloud_labels.x[1] - ds_cloud_labels.x[0])
        num_times = int(ds_cloud_labels.time.size)
        nz, ny, nx = ds_cloud_labels.cloud_labels.shape[1:]
        time_vals = ds_cloud_labels.time.values
        z_vals = ds_cloud_labels.z.values
        y_vals = ds_cloud_labels.y.values
        x_vals = ds_cloud_labels.x.values

    box_limits = np.array([999999.0, ny * grid_distance, nx * grid_distance])

    # 1. Preallocate blank files
    open_files = {}
    try:
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

        print("\n✅ All computation and exporting complete")
    except KeyboardInterrupt:
        print("\n⚠️ Job interrupted or cancelled via Slurm. Closing files safely...")
    finally:
        # This block ALWAYS runs, ensuring handles are dropped on normal exit OR scancel
        print("Flushing and closing all NetCDF file handles...")
        for filename, file_obj in open_files.items():
            try:
                file_obj.close()
                print(f" -> Closed: {filename}")
            except Exception as e:
                print(f" -> Error closing {filename}: {e}")

        try:
            import shutil
            if custom_tmp_dir.exists():
                shutil.rmtree(custom_tmp_dir)
                print("🧹 Cleaned up temporary buffer directory.")
        except Exception as e:
            print(f"⚠️ Could not automatically clean up tmp folder: {e}")

        print("\n✅ All file streams safely disconnected (Program is safe to close).")