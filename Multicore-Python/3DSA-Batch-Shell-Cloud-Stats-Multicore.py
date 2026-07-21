import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import math
import numpy as np
import xarray as xr
import scipy.ndimage
from pathlib import Path
import netCDF4 as nc
import scipy.linalg
import time
import sys
import multiprocessing
import json
import gc
import argparse

# =====================================================================
# GLOBAL CONFIGURATION & SHARED REGISTRY
# =====================================================================
EXPORT_REGISTRY = {
    "shell_base.nc": ("shell_base", "f4"),
    "shell_top.nc": ("shell_top", "f4"),
    "shell_depth.nc": ("shell_depth", "f4"),
    "cloud_base.nc": ("cloud_base", "f4"),
    "cloud_top.nc": ("cloud_top", "f4"),
    "cloud_depth.nc": ("cloud_depth", "f4"),
    "shallow_mask.nc": ("shallow_mask", "u1"),
    "congestus_mask.nc": ("congestus_mask", "u1"),
    "deep_mask.nc": ("deep_mask", "u1"),
    "distance_from_shell_top.nc": ("distance", "f4"),
    "distance_from_cloud_top.nc": ("distance", "f4"),
    "normalized_distance_from_cloud_base.nc": ("normalized_distance", "f4"),
    "normalized_distance_from_shell_base.nc": ("normalized_distance", "f4")
}

def get_valid_min(numpy_arr, mask):
    masked_numpy = numpy_arr[mask]
    valid_mask = ~np.isnan(masked_numpy)

    if np.any(valid_mask):
        filtered_numpy = masked_numpy[valid_mask]
        return filtered_numpy.min()
    else:
        return np.nan
    
def get_valid_max(numpy_arr, mask):
    masked_numpy = numpy_arr[mask]
    valid_mask = ~np.isnan(masked_numpy)

    if np.any(valid_mask):
        filtered_numpy = masked_numpy[valid_mask]
        return filtered_numpy.max()
    else:
        return np.nan

# --- Multiprocessing Worker Function ---
def process_timestep_worker(args):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    start_time = time.time()
    t, cfg = args

    paths = cfg["paths"]
    z_coordinates = cfg["z_coordinates"]

    #load datasets
    with  nc.Dataset(paths["cloud_mask"], "r", parallel=False) as ds_cloud_mask, \
         nc.Dataset(paths["combined_labels"], "r", parallel=False) as ds_combined_labels, \
         nc.Dataset(paths["cloud_labels"], "r", parallel=False) as ds_cloud_labels, \
         nc.Dataset(paths["shell_labels"], "r", parallel=False) as ds_shell_labels:

        combined_labels_slice = ds_combined_labels.variables["labels"][t, :, :, :]
        cloud_labels_slice = ds_cloud_labels.variables["cloud_labels"][t, :, :, :]
        shell_labels_slice = ds_shell_labels.variables["shell_labels"][t, :, :, :]
        cloud_mask_slice = ds_cloud_mask.variables["cloud_mask"][t, :, :, :]

    grid_shape = shell_labels_slice.shape

    #initialize local arrays for this timestep
    local_shell_base = np.full(grid_shape, np.nan, dtype=np.float32)
    local_shell_top = np.full(grid_shape, np.nan, dtype=np.float32)
    local_shell_depth = np.full(grid_shape, np.nan, dtype=np.float32)
    local_cloud_bottom = np.full(grid_shape, np.nan, dtype=np.float32)
    local_cloud_top = np.full(grid_shape, np.nan, dtype=np.float32)
    local_cloud_depth = np.full(grid_shape, np.nan, dtype=np.float32)
    
    local_dfst = np.full(grid_shape, np.nan, dtype=np.float32)
    local_dfct = np.full(grid_shape, np.nan, dtype=np.float32)
    local_ndfcb = np.full(grid_shape, np.nan, dtype=np.float32)
    local_ndfso = np.full(grid_shape, np.nan, dtype=np.float32)

    local_congestus_mask = np.zeros(grid_shape, dtype=np.uint8)
    local_deep_mask = np.zeros(grid_shape, dtype=np.uint8)
    local_shallow_mask = np.zeros(grid_shape, dtype=np.uint8)

    matching_labels = set(np.unique(cloud_labels_slice))
    matching_labels.discard(0)
    timestep_cloud_data = {}
    timestep_shell_data = {}
    if matching_labels:
        #Stats calc
        for obj_id in matching_labels:
            cloud_mask = (cloud_labels_slice == obj_id)

            cloud_z_indices, _, _ = np.where(cloud_mask)

            if cloud_z_indices.size > 0:
                min_z_cloud = z_coordinates[cloud_z_indices.min()]
                max_z_cloud = z_coordinates[cloud_z_indices.max()]
                cloud_depth = max_z_cloud - min_z_cloud

                local_cloud_bottom[cloud_mask] = min_z_cloud
                local_cloud_top[cloud_mask] = max_z_cloud
                local_cloud_depth[cloud_mask] = cloud_depth

                voxel_zs = z_coordinates[cloud_z_indices]
                local_dfct[cloud_mask] = max_z_cloud - voxel_zs
                if cloud_depth > 0:
                    local_ndfcb[cloud_mask] = (voxel_zs - min_z_cloud) / cloud_depth
                else:
                    local_ndfcb[cloud_mask] = 0.0
                
                if max_z_cloud > 5000: #cloud is deep
                    classification = "deep"
                    local_deep_mask[cloud_mask] = 1
                elif max_z_cloud > 2000: #cloud is congestus
                    classification = "congestus"
                    local_congestus_mask[cloud_mask] = 1
                else: #cloud is shallow
                    classification = "shallow"
                    local_shallow_mask[cloud_mask] = 1

                timestep_cloud_data[int(obj_id)] = {
                    "cloud_base": float(min_z_cloud) if not np.isnan(min_z_cloud) else None,
                    "cloud_top": float(max_z_cloud) if not np.isnan(max_z_cloud) else None,
                    "cloud_depth": float(cloud_depth) if not np.isnan(cloud_depth) else None,
                    "class": classification
                }
        
    matching_labels_shell = set(np.unique(shell_labels_slice))
    matching_labels_shell.discard(0)
    if matching_labels_shell:
        for shell_id in matching_labels_shell:
            combined_obj_mask = (combined_labels_slice == shell_id)
            shell_obj_mask = (shell_labels_slice == shell_id)
            shell_z_indices, _, _ = np.where(shell_obj_mask)
            combined_z_indicies, _, _ = np.where(combined_obj_mask)

            if shell_z_indices.size > 0:
                min_z_shell = z_coordinates[shell_z_indices.min()]
                max_z_shell = z_coordinates[shell_z_indices.max()]
                shell_depth = max_z_shell - min_z_shell

                local_shell_base[combined_obj_mask] = min_z_shell
                local_shell_top[combined_obj_mask] = max_z_shell
                local_shell_depth[combined_obj_mask] = shell_depth

                #Shell Distances
                shell_voxel_zs = z_coordinates[shell_z_indices]
                combined_voxel_zs = z_coordinates[combined_z_indicies]
                local_dfst[combined_obj_mask] = max_z_shell - combined_voxel_zs
                
                if shell_depth > 0: # only include depth > 0
                    local_ndfso[combined_obj_mask] = (combined_voxel_zs - min_z_shell) / shell_depth

                cloud_obj_mask = combined_obj_mask & (cloud_mask_slice.astype(bool))
                contained_classification = "free"

                #Apply nans if there is no cloud
                contained_cloud_depth = np.nan
                contained_max_z_cloud = np.nan
                contained_min_z_cloud = np.nan

                #get properties from contained clouds
                if(np.any(cloud_obj_mask)):
                    contained_min_z_cloud = get_valid_min(local_cloud_bottom, cloud_obj_mask)
                    contained_max_z_cloud = get_valid_max(local_cloud_top, cloud_obj_mask)

                    if not np.isnan(contained_max_z_cloud) and not np.isnan(contained_min_z_cloud):
                        contained_cloud_depth = contained_max_z_cloud - contained_min_z_cloud

                        if contained_max_z_cloud > 5000:
                            contained_classification = "deep"
                            local_deep_mask[shell_obj_mask] = 1
                        elif contained_max_z_cloud > 2000:
                            contained_classification = "congestus"
                            local_congestus_mask[shell_obj_mask] = 1
                        else:
                            contained_classification = "shallow"
                            local_shallow_mask[shell_obj_mask] = 1

                        #Cloud Distances
                        local_dfct[shell_obj_mask] = contained_max_z_cloud - shell_voxel_zs
                        if contained_cloud_depth > 0: #only include depth > 0
                            local_ndfcb[shell_obj_mask] = (shell_voxel_zs - contained_min_z_cloud) / contained_cloud_depth

                local_cloud_bottom[shell_obj_mask] = contained_min_z_cloud
                local_cloud_top[shell_obj_mask] = contained_max_z_cloud
                local_cloud_depth[shell_obj_mask] = contained_cloud_depth

                timestep_shell_data[int(shell_id)] = {
                    "lowest_cloud_base": float(contained_min_z_cloud) if not np.isnan(contained_min_z_cloud) else None,
                    "highest_cloud_top": float(contained_max_z_cloud) if not np.isnan(contained_max_z_cloud) else None,
                    "combined_cloud_depth": float(contained_cloud_depth) if not np.isnan(contained_cloud_depth) else None,
                    "shell_base": float(min_z_shell) if not np.isnan(min_z_shell) else None,
                    "shell_top": float(max_z_shell) if not np.isnan(max_z_shell) else None,
                    "shell_depth": float(shell_depth) if not np.isnan(shell_depth) else None,
                    "class": contained_classification
                }




    # --- Exporting ---
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t, {
        "shell_base.nc": local_shell_base,
        "shell_top.nc": local_shell_top,
        "shell_depth.nc": local_shell_depth,
        "cloud_base.nc": local_cloud_bottom,
        "cloud_top.nc": local_cloud_top,
        "cloud_depth.nc": local_cloud_depth,
        "distance_from_shell_top.nc": local_dfst,
        "distance_from_cloud_top.nc": local_dfct,
        "normalized_distance_from_cloud_base.nc": local_ndfcb,
        "normalized_distance_from_shell_base.nc": local_ndfso,
        "shallow_mask.nc": local_shallow_mask,
        "congestus_mask.nc": local_congestus_mask,
        "deep_mask.nc": local_deep_mask,
        "timestep_cloud_data": timestep_cloud_data,
        "timestep_shell_data": timestep_shell_data,
        "duration": elapsed_str,
    }


# --- Main Thread ---
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    # --- Configurations ---
    num_cores = int(os.environ.get("CORE_COUNT", 1))  # Default to 1 core if not specified

    parser = argparse.ArgumentParser(description="Process 3DSA pipeline for a specific data source.")
    parser.add_argument(
        "--data_source", 
        type=str, 
        required=True, 
        help="Key matching the data source configuration block in config.json"
    )
    args = parser.parse_args()

    # --- Setting up directories from config ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR / "config.json"

    if not CONFIG_PATH.is_file():
        print(f"❌ ERROR: Configuration file missing at: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read json config file
    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)

    #load config preset based on 
    source_key = args.data_source
    if source_key not in config_data["paths"]:
        print(f"❌ ERROR: Data source '{source_key}' not found in config.json", file=sys.stderr)
        sys.exit(1)

    # Extract Paths
    output_dir = Path(config_data["paths"][source_key]["output_dir"])
    input_dir = output_dir

    #in case directory does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── OVERRIDE SYSTEM TMPDIR WITH CONFIG PATH ──────────────────────────
    custom_tmp_dir = output_dir / "tmp"
    custom_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["TMPDIR"] = str(custom_tmp_dir)
    # ──────────────────────────────────────────────────────────────────────

    print(f"Initialization Success:")
    print(f" -> Input Path: {input_dir}")
    print(f" -> Output Path:       {output_dir}")
    print(f" -> Active CPU Cores:  {num_cores}")
    print("-" * 50)

    print("Checking file dependencies...")
    file_paths = {
        "cloud_mask": input_dir / "cloud_mask.nc",
        "combined_labels": input_dir / "combined_labels.nc",
        "cloud_labels": input_dir / "cloud_labels.nc",
        "shell_labels": input_dir / "shell_labels.nc",
    }
    #Check that files exist
    for name, path in file_paths.items():
        if not path.is_file():
            print(f"❌ ERROR: Missing target dependency: {path}", file=sys.stderr)
            sys.exit(1)

    # Global structure
    with xr.open_dataset(file_paths["shell_labels"], decode_times=False, engine="netcdf4") as ds_meta:
        num_times = int(ds_meta.time.size)
        nz, ny, nx = ds_meta.shell_labels.shape[1:]
        time_vals = ds_meta.time.compute().values
        z_vals = ds_meta.z.compute().values
        y_vals = ds_meta.y.compute().values
        x_vals = ds_meta.x.compute().values
        dx = float(ds_meta.x[1] - ds_meta.x[0])
        dy = float(ds_meta.y[1] - ds_meta.y[0])


    # --- Preallocate NetCDF file structures ---
    open_files = {}
    try:
        print("Pre-allocating NetCDF file structures on disk...")
        for filename, (var_name, data_type) in EXPORT_REGISTRY.items():
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
            
            var = f.createVariable(var_name, data_type, ("time", "z", "y", "x"), 
                                zlib=True, complevel=4, chunksizes=(1, nz, ny, nx))
            
            if data_type == "u1":
                var.setncattr("_Unsigned", "true")

        all_timesteps_shell_data = {}
        all_timesteps_cloud_data = {}
        # --- Start Worker Pool ---
        # Package arguments cleanly into a metadata dictionary
        worker_config = {
            "paths": {k: str(v) for k, v in file_paths.items()},
            "dx": dx,
            "dy": dy,
            "z_coordinates": z_vals
        }

        print(f"Spawning Pool with {num_cores} active workers over {num_times} timesteps...")
        pool_tasks = [(t, worker_config) for t in range(num_times)]

        with multiprocessing.Pool(processes=num_cores) as pool:
            for t_idx, payload in pool.imap_unordered(process_timestep_worker, pool_tasks):
                print(f"Timestep {t_idx}/{num_times - 1} finished in ({payload['duration']}). Committing to files...")
                
                for filename, data_array in payload.items():
                    if filename == "duration":
                        continue
                    if filename == "timestep_cloud_data":
                        all_timesteps_cloud_data[t_idx] = data_array
                        continue
                    if filename == "timestep_shell_data":
                        all_timesteps_shell_data[t_idx] = data_array
                        continue
                    var_key = EXPORT_REGISTRY[filename][0]
                    open_files[filename].variables[var_key][t_idx, :, :, :] = data_array
                    open_files[filename].sync()

                gc.collect()

        # Save nested structured metadata dictionary as a JSON file
        if all_timesteps_cloud_data:
            json_path = output_dir / "cloud_height_stats.json"
            with open(json_path, "w") as jf:
                json.dump(all_timesteps_cloud_data, jf, indent=4)
            print(f"✅ Successfully saved cloud object tracking metadata to:\n   {json_path}")
        else:
            print("\n⚠️ No cloud object tracking statistics were collected across the simulation.")

        if all_timesteps_shell_data:
            json_path = output_dir / "shell_height_stats.json"
            with open(json_path, "w") as jf:
                json.dump(all_timesteps_shell_data, jf, indent=4)
            print(f"✅ Successfully saved shell object tracking metadata to:\n   {json_path}")
        else:
            print("\n⚠️ No shell object tracking statistics were collected across the simulation.")

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