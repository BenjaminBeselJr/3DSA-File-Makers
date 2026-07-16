import os
import math
import numpy as np
import xarray as xr
import scipy.ndimage
from pathlib import Path
import netCDF4 as nc
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
    "slab_shell_entrainment.nc": ("entrainment", "f4"), #dimensions: t, z
    "slab_shell_label_entrainment.nc": ("entrainment", "f4"), #dimensions: t, z, y, x
    "slab_cloud_entrainment.nc": ("entrainment", "f4"), #dimensions: t, z
    "slab_cloud_label_entrainment.nc": ("entrainment", "f4"), #dimensions: t, z, y, x
    
}

# Physical Constants
dilation_const = 1
max_entrainment_magnitude = 1e30

def dilate_mask(source, dilateAmount):
    if np.any(source) and dilateAmount > 0:

        expansion = np.zeros((3,3,3), dtype=bool)
        expansion[1, 1, :] = True  # X axis
        expansion[1, :, 1] = True  # Y axis
        expansion[:, 1, 1] = True  # Z axis

        for _ in range(dilateAmount):
            padded = np.pad(source, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=False)
            padded = np.pad(padded, ((0, 0), (1, 1), (1, 1)), mode='wrap')
            dilated = scipy.ndimage.binary_dilation(padded, structure=expansion)
            unpadded_dilated = dilated[1:-1, 1:-1, 1:-1]
        return unpadded_dilated
    else:
        return source

def filter_e(eSet, t, maxEntrainmentMagnitude):
    sliced_E = eSet.isel(time=t).compute()
    return xr.where(abs(sliced_E) < maxEntrainmentMagnitude, sliced_E, np.nan).values

# --- Multiprocessing Worker Function ---
def process_timestep_worker(args):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    start_time = time.time()
    t_new, t, arrays, dilation_const = args

    cloud_labels = arrays["cloud_labels"]
    combined_labels = arrays["combined_labels"]
    shell_labels = arrays["shell_labels"]

    # pre-allocate sets
    nz, ny, nx = cloud_labels.shape #create shape needed for exports

    out_shell_label_ent = np.zeros((nz, ny, nx), dtype=np.float32)
    out_cloud_label_ent = np.zeros((nz, ny, nx), dtype=np.float32)
    total_shell_ent_profile = np.zeros(nz, dtype=np.float32)
    total_cloud_ent_profile = np.zeros(nz, dtype=np.float32)

    # ----------------------------------------------------------------------
    # Step 1 - Shell Entrainment (Accumulates Shell Fluxes near Cloud Edges)
    # ----------------------------------------------------------------------

    # -- Iterate through shell
    shell_list = np.unique(shell_labels)
    shell_list = shell_list[shell_list != 0]

    for label_i in shell_list:

        # obtain combined mask for that shell index
        label_mask = (combined_labels == label_i)

        # Iterate through each cloud
        contained_cloud_list = np.unique(cloud_labels[label_mask])
        contained_cloud_list = contained_cloud_list[contained_cloud_list != 0]

        for c_label_i in contained_cloud_list:
            current_cloud = (cloud_labels == c_label_i)

            # Dilate the cloud for overlap
            dilated_cloud = dilate_mask(current_cloud, dilation_const)

            if not np.any(dilated_cloud):
                continue

            e_x_mask = dilated_cloud | np.roll(dilated_cloud, shift=1, axis=2)
            e_y_mask = dilated_cloud | np.roll(dilated_cloud, shift=1, axis=1)
            e_z_mask = dilated_cloud | np.roll(dilated_cloud, shift=1, axis=0)
            e_z_mask[0, :, :] = dilated_cloud[0, :, :] # prevent rolling along boundary

            # sum x and y
            sum_x = np.sum(arrays["shell_e_x"] * e_x_mask, axis=(1, 2))
            sum_y = np.sum(arrays["shell_e_y"] * e_y_mask, axis=(1, 2))

            shell_target = (shell_labels == label_i)
            shell_above = shell_target
            shell_below = np.zeros_like(shell_target)
            shell_below[1:] = shell_target[:-1]

            case1_mask = e_z_mask & shell_above & ~shell_below # case 1: shell above but not below
            case2_mask = e_z_mask & shell_below & ~shell_above # case 2: shell below but not above

            # sum z
            sum_z_case1 = np.sum(arrays["shell_e_z"] * case1_mask, axis=(1, 2))
            sum_z_case2 = np.sum(arrays["shell_e_z"] * case2_mask, axis=(1, 2))

            sum_z = np.zeros(nz, dtype=np.float32)
            sum_z += sum_z_case1
            sum_z[:-1] += sum_z_case2[1:]  # Shift map back down safely

            # apply sums
            sum_total = (sum_x + sum_y + sum_z)
            broadcasted_sum = sum_total[:, np.newaxis, np.newaxis]

            out_shell_label_ent += broadcasted_sum * label_mask
            total_shell_ent_profile += sum_total
        
        if len(contained_cloud_list) > 0:
            del current_cloud, dilated_cloud, e_x_mask, e_y_mask, e_z_mask, case1_mask, case2_mask

    # ----------------------------------------------------------------------
    # Step 2 - Cloud Entrainment (Accumulates Cloud Fluxes near Shell Edges)
    # ----------------------------------------------------------------------
    cloud_list = np.unique(cloud_labels)
    cloud_list = cloud_list[cloud_list != 0]

    for c_label_i in cloud_list:
        cloud_target = (cloud_labels == c_label_i)

        # Finds the index of the very first True value in the cloud mask
        first_idx = np.argmax(cloud_target) 
        label_i = combined_labels.flat[first_idx]
        if label_i == 0:
            continue

        current_shell = (shell_labels == label_i)
        if not np.any(current_shell):
            continue

        dilated_shell = dilate_mask(current_shell, dilation_const)
        if not np.any(dilated_shell):
            continue

        e_x_mask = dilated_shell | np.roll(dilated_shell, shift=1, axis=2)
        e_y_mask = dilated_shell | np.roll(dilated_shell, shift=1, axis=1)
        e_z_mask = dilated_shell | np.roll(dilated_shell, shift=1, axis=0)
        e_z_mask[0, :, :] = dilated_shell[0, :, :] # prevent rolling along boundary

        # sum x,y
        sum_x = np.sum(arrays["cloud_e_x"] * e_x_mask, axis=(1, 2))
        sum_y = np.sum(arrays["cloud_e_y"] * e_y_mask, axis=(1, 2))

        cloud_above = cloud_target
        cloud_below = np.zeros_like(cloud_target)
        cloud_below[1:] = cloud_target[:-1]

        case1_mask = e_z_mask & cloud_above & ~cloud_below # case 1: shell above but not below
        case2_mask = e_z_mask & cloud_below & ~cloud_above # case 2: shell below but not above

        sum_z_case1 = np.sum(arrays["cloud_e_z"] * case1_mask, axis=(1, 2))
        sum_z_case2 = np.sum(arrays["cloud_e_z"] * case2_mask, axis=(1, 2))

        sum_z = np.zeros(nz, dtype=np.float32)
        sum_z += sum_z_case1
        sum_z[:-1] += sum_z_case2[1:]

        sum_total = (sum_x + sum_y + sum_z)
        broadcasted_sum = sum_total[:, np.newaxis, np.newaxis]

        out_cloud_label_ent += broadcasted_sum * cloud_target
        total_cloud_ent_profile += sum_total

        if 'dilated_shell' in locals():
            del cloud_target, current_shell, dilated_shell, e_x_mask, e_y_mask, e_z_mask, case1_mask, case2_mask
            
    # --- Exporting ---
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t_new, t, {
        "slab_shell_entrainment.nc": total_shell_ent_profile,
        "slab_shell_label_entrainment.nc": out_shell_label_ent,
        "slab_cloud_entrainment.nc": total_cloud_ent_profile,
        "slab_cloud_label_entrainment.nc": out_cloud_label_ent,
        "duration": elapsed_str,
    }


# --- Main Thread ---
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main_start_time = time.time()
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
    source_input_dir = Path(config_data["paths"][source_key]["source_input_dir"])
    output_dir = Path(config_data["paths"][source_key]["output_dir"])

    #in case directory does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── OVERRIDE SYSTEM TMPDIR WITH CONFIG PATH ──────────────────────────
    custom_tmp_dir = output_dir / "tmp"
    custom_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["TMPDIR"] = str(custom_tmp_dir)
    # ──────────────────────────────────────────────────────────────────────

    print(f"Initialization Success:")
    print(f" -> Source Input Path:    {source_input_dir}")
    print(f" -> Input & Output Path:  {output_dir}")
    print(f" -> Active CPU Cores:     {num_cores}")
    print("-" * 50)

    print("Checking file dependencies...")

    file_paths = {
        "netE": source_input_dir / "netE.nc",
        "cloud_labels": output_dir / "cloud_labels.nc",
        "combined_labels": output_dir / "combined_labels.nc",
        "shell_labels": output_dir / "shell_labels.nc",
    }

    #Check that files exist
    for name, path in file_paths.items():
        if not path.is_file():
            print(f"❌ ERROR: Missing target dependency: {path}", file=sys.stderr)
            sys.exit(1)

    # Global structure
    with xr.open_dataset(file_paths["netE"], decode_times=False, engine="netcdf4") as ds_meta:
        num_times = int(ds_meta.time.size)
        nz, ny, nx = ds_meta.netE_ql.shape[1:]
        
        #Switch slicing depending on source of data
        if source_key in ["SEUS", "RICO"]:
            active_timesteps = [t for t in range(83, num_times, 2)] #all odds except index 1
            print(f"✂️ {source_key} Config: Filtering for odd timesteps skipping index 1.")
        else:
            active_timesteps = list(range(num_times))

        time_vals = ds_meta.time.values[active_timesteps]
        num_output_times = len(active_timesteps)
        z_vals = ds_meta.z.values
        y_vals = ds_meta.y.values
        x_vals = ds_meta.x.values
        dx = float(ds_meta.x[1] - ds_meta.x[0])
        dy = float(ds_meta.y[1] - ds_meta.y[0])

    # --- Preallocate NetCDF file structures ---
    open_files = {}
    try:
        print("Pre-allocating NetCDF file structures on disk...")
        for filename, (var_name, data_type) in EXPORT_REGISTRY.items():
            file_path = output_dir / filename
            print(f" -> Creating file: {file_path}") 
            f = nc.Dataset(str(file_path), "w", format="NETCDF4")
            open_files[filename] = f

            # Base Shared Dimensions
            f.createDimension("time", num_output_times)
            f.createDimension("z", nz)
            f.createVariable("time", "f8", ("time",))[:] = time_vals
            f.createVariable("z", "f4", ("z",))[:] = z_vals
            
            # Check if this is a 1D Profile or a full 3D Spatial Grid
            if filename == "slab_shell_label_entrainment.nc" or filename == "slab_cloud_label_entrainment.nc":
                # Setup spatial dims for 3D outputs
                f.createDimension("y", ny)
                f.createDimension("x", nx)
                f.createVariable("y", "f4", ("y",))[:] = y_vals
                f.createVariable("x", "f4", ("x",))[:] = x_vals
                
                f.createVariable(var_name, data_type, ("time", "z", "y", "x"), 
                                    zlib=True, complevel=4, chunksizes=(1, nz, ny, nx))
            else:
                # Setup configuration specifically tailored for 1D arrays
                f.createVariable(var_name, data_type, ("time", "z"), 
                                    zlib=True, complevel=4, chunksizes=(1, nz))


        # --- Start Worker Pool ---
        # Package arguments cleanly into a metadata dictionary
        worker_config = {
            "paths": {k: str(v) for k, v in file_paths.items()},
            "dx": dx,
            "dy": dy,
            "dilation": dilation_const,
            "max_entrainment": max_entrainment_magnitude
        }

        print(f"Spawning Pool with {num_cores} active workers over {num_output_times} timesteps...")
        pool_tasks = []
        # Open files once in the main thread to read and pass raw arrays to workers
        with xr.open_dataset(file_paths["netE"], decode_times=False) as ds_e, \
             xr.open_dataset(file_paths["cloud_labels"], decode_times=False) as ds_cloud, \
             xr.open_dataset(file_paths["combined_labels"], decode_times=False) as ds_comb, \
             xr.open_dataset(file_paths["shell_labels"], decode_times=False) as ds_shell:

            # 2. Define a generator function to yield tasks ONE by ONE instead of pre-allocating a list
            def task_generator():
                for new_idx, t_original in enumerate(active_timesteps):
                    payload_arrays = {
                        "cloud_labels": ds_cloud.cloud_labels.isel(time=new_idx).values,
                        "combined_labels": ds_comb.labels.isel(time=new_idx).values,
                        "shell_labels": ds_shell.shell_labels.isel(time=new_idx).values,

                        "cloud_e_x": filter_e(ds_e.netE_flux_x_ql, t_original, max_entrainment_magnitude),
                        "cloud_e_y": filter_e(ds_e.netE_flux_y_ql, t_original, max_entrainment_magnitude),
                        "cloud_e_z": filter_e(ds_e.netE_flux_z_ql, t_original, max_entrainment_magnitude),

                        "shell_e_x": filter_e(ds_e.netE_flux_x_shell, t_original, max_entrainment_magnitude),
                        "shell_e_y": filter_e(ds_e.netE_flux_y_shell, t_original, max_entrainment_magnitude),
                        "shell_e_z": filter_e(ds_e.netE_flux_z_shell, t_original, max_entrainment_magnitude),
                    }
                    yield (new_idx, t_original, payload_arrays, dilation_const)

            # 3. Feed the generator directly to the pool
            with multiprocessing.Pool(processes=num_cores) as pool:
                # task_generator() ensures only 'num_cores' worth of timesteps are in memory at a given time
                for t_idx, t_original, payload in pool.imap_unordered(process_timestep_worker, task_generator()):
                    print(f"Timestep {t_idx}/{num_output_times - 1} (Original index: {t_original}) finished in ({payload['duration']}). Committing to files...")
                    for filename, data_array in payload.items():
                        if filename in ["duration"]:
                            continue
                        var_key = EXPORT_REGISTRY[filename][0]
                        
                        if filename == "slab_shell_label_entrainment.nc" or filename == "slab_cloud_label_entrainment.nc":
                            open_files[filename].variables[var_key][t_idx, :, :, :] = data_array
                        else:
                            open_files[filename].variables[var_key][t_idx, :] = data_array
                            
                        open_files[filename].sync()

                    gc.collect()

        main_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - main_start_time))
        print(f"\n✅ All computation and exporting complete in ({main_elapsed_str})")
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

        print("\n✅ All file streams safely disconnected (Program is safe to close)")