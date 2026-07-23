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
    # Entrainment
    # - Across other body
    "slab_shell_entrainment_cloud_boundary.nc": ("entrainment", "f4"), #dimensions: t, z
    "slab_cloud_entrainment_shell_boundary.nc": ("entrainment", "f4"), #dimensions: t, z
    "slab_shell_label_entrainment_cloud_boundary.nc": ("entrainment", "f4"), #dimensions: t, z, y, x
    "slab_cloud_label_entrainment_shell_boundary.nc": ("entrainment", "f4"), #dimensions: t, z, y, x
    # - Across open air
    "slab_shell_entrainment_air_boundary.nc": ("entrainment", "f4"), #dimensions: t, z
    "slab_cloud_entrainment_air_boundary.nc": ("entrainment", "f4"), #dimensions: t, z
    "slab_shell_label_entrainment_air_boundary.nc": ("entrainment", "f4"), #dimensions: t, z, y, x
    "slab_cloud_label_entrainment_air_boundary.nc": ("entrainment", "f4"), #dimensions: t, z, y, x

    # Detrainment
    # - Across other body
    "slab_shell_detrainment_cloud_boundary.nc": ("detrainment", "f4"), #dimensions: t, z
    "slab_cloud_detrainment_shell_boundary.nc": ("detrainment", "f4"), #dimensions: t, z
    "slab_shell_label_detrainment_cloud_boundary.nc": ("detrainment", "f4"), #dimensions: t, z, y, x
    "slab_cloud_label_detrainment_shell_boundary.nc": ("detrainment", "f4"), #dimensions: t, z, y, x
    # - Across open air
    "slab_shell_detrainment_air_boundary.nc": ("detrainment", "f4"), #dimensions: t, z
    "slab_cloud_detrainment_air_boundary.nc": ("detrainment", "f4"), #dimensions: t, z
    "slab_shell_label_detrainment_air_boundary.nc": ("detrainment", "f4"), #dimensions: t, z, y, x
    "slab_cloud_label_detrainment_air_boundary.nc": ("detrainment", "f4"), #dimensions: t, z, y, x
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

def get_intersection_entrainment(origin_body_mask, other_body_mask, nz, ny, nx, ne_x, ne_y, ne_z, dilationAmount):
    # Preallocate empty sets
    entrainment_3d = np.zeros((nz, ny, nx), dtype=np.float32)
    entrainment_profile = np.zeros(nz, dtype=np.float32)
    detrainment_3d = np.zeros((nz, ny, nx), dtype=np.float32)
    detrainment_profile = np.zeros(nz, dtype=np.float32)

    # Obtain entrainment and detrainment masks
    ent_x_mask = ne_x > 0
    ent_y_mask = ne_y > 0
    ent_z_mask = ne_z > 0
    det_x_mask = ne_x < 0
    det_y_mask = ne_y < 0
    det_z_mask = ne_z < 0

    # Dilate other body
    dilated_other_body = dilate_mask(other_body_mask, dilationAmount)

    if not np.any(dilated_other_body):
        return entrainment_3d, entrainment_profile, detrainment_3d, detrainment_profile

    e_x_mask = dilated_other_body | np.roll(dilated_other_body, shift=1, axis=2)
    e_y_mask = dilated_other_body | np.roll(dilated_other_body, shift=1, axis=1)
    e_z_mask = dilated_other_body | np.roll(dilated_other_body, shift=1, axis=0)
    e_z_mask[0, :, :] = dilated_other_body[0, :, :] # prevent rolling along boundary

    # sum x and y
    sum_e_x = np.nansum(ne_x * ent_x_mask * e_x_mask, axis=(1, 2))
    sum_e_y = np.nansum(ne_y * ent_y_mask * e_y_mask, axis=(1, 2))
    sum_d_x = np.nansum(ne_x * det_x_mask * e_x_mask, axis=(1, 2))
    sum_d_y = np.nansum(ne_y * det_y_mask * e_y_mask, axis=(1, 2))

    origin_above = origin_body_mask
    origin_below = np.zeros_like(origin_body_mask)
    origin_below[1:] = origin_body_mask[:-1]

    case1_mask = e_z_mask & origin_above & ~origin_below # case 1: shell above but not below
    case2_mask = e_z_mask & origin_below & ~origin_above # case 2: shell below but not above

    # sum z
    sum_e_z_case1 = np.nansum(ne_z * ent_z_mask * case1_mask, axis=(1, 2))
    sum_e_z_case2 = np.nansum(ne_z * ent_z_mask * case2_mask, axis=(1, 2))
    sum_d_z_case1 = np.nansum(ne_z * det_z_mask * case1_mask, axis=(1, 2))
    sum_d_z_case2 = np.nansum(ne_z * det_z_mask * case2_mask, axis=(1, 2))

    sum_e_z = np.zeros(nz, dtype=np.float32)
    sum_d_z = np.zeros(nz, dtype=np.float32)

    sum_e_z += sum_e_z_case1
    sum_e_z[:-1] += sum_e_z_case2[1:]  # Shift map back down safely
    sum_d_z += sum_d_z_case1
    sum_d_z[:-1] += sum_d_z_case2[1:]  # Shift map back down safely

    # apply sums
    sum_e_total = (sum_e_x + sum_e_y + sum_e_z)
    sum_d_total = (sum_d_x + sum_d_y + sum_d_z)
    broadcasted_sum_e = sum_e_total[:, np.newaxis, np.newaxis]
    broadcasted_sum_d = sum_d_total[:, np.newaxis, np.newaxis]

    entrainment_3d += broadcasted_sum_e * origin_body_mask
    detrainment_3d += broadcasted_sum_d * origin_body_mask
    entrainment_profile += sum_e_total
    detrainment_profile += sum_d_total
    return entrainment_3d, entrainment_profile, detrainment_3d, detrainment_profile

# --- Multiprocessing Worker Function ---
def process_timestep_worker(task):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    start_time = time.time()
    t_val = task["t_val"]
    t_idx = task["t_idx"]
    paths = task["file_paths"]
    dilation_const = task["dilation_const"]
    max_ent_mag = task["max_entrainment"]

    with xr.open_dataset(paths["cloud_labels"], decode_times=False) as ds_cloud:
        cloud_labels = ds_cloud.cloud_labels.sel(time=t_val).values
        
    with xr.open_dataset(paths["combined_labels"], decode_times=False) as ds_comb:
        combined_labels = ds_comb.labels.sel(time=t_val).values
        
    with xr.open_dataset(paths["shell_labels"], decode_times=False) as ds_shell:
        shell_labels = ds_shell.shell_labels.sel(time=t_val).values

    # --- Pre-allocate sets ---
    nz, ny, nx = cloud_labels.shape

    # Entrainment
    # -Across other body
    out_shell_label_ent = np.zeros((nz, ny, nx), dtype=np.float32)
    out_cloud_label_ent = np.zeros((nz, ny, nx), dtype=np.float32)
    total_shell_ent_profile = np.zeros(nz, dtype=np.float32)
    total_cloud_ent_profile = np.zeros(nz, dtype=np.float32)
    # -Across open air
    out_shell_label_ent_air = np.zeros((nz, ny, nx), dtype=np.float32)
    out_cloud_label_ent_air = np.zeros((nz, ny, nx), dtype=np.float32)
    total_shell_ent_profile_air = np.zeros(nz, dtype=np.float32)
    total_cloud_ent_profile_air = np.zeros(nz, dtype=np.float32)

    # Detrainment 
    # -Across other body
    out_shell_label_det = np.zeros((nz, ny, nx), dtype=np.float32)
    out_cloud_label_det = np.zeros((nz, ny, nx), dtype=np.float32)
    total_shell_det_profile = np.zeros(nz, dtype=np.float32)
    total_cloud_det_profile = np.zeros(nz, dtype=np.float32)
    # -Across open air
    out_shell_label_det_air = np.zeros((nz, ny, nx), dtype=np.float32)
    out_cloud_label_det_air = np.zeros((nz, ny, nx), dtype=np.float32)
    total_shell_det_profile_air = np.zeros(nz, dtype=np.float32)
    total_cloud_det_profile_air = np.zeros(nz, dtype=np.float32)
    
    

    with xr.open_dataset(paths["netE"], decode_times=False) as ds_e:
        # Helper inline function to slice and filter on the fly
        def load_and_filter(var_name):
            sliced = ds_e[var_name].sel(time=t_val)
            return xr.where(abs(sliced) < max_ent_mag, sliced, np.nan).values

        cloud_ne_x = load_and_filter("netE_flux_x_ql")
        cloud_ne_y = load_and_filter("netE_flux_y_ql")
        cloud_ne_z = load_and_filter("netE_flux_z_ql")

        shell_ne_x = load_and_filter("netE_flux_x_shell")
        shell_ne_y = load_and_filter("netE_flux_y_shell")
        shell_ne_z = load_and_filter("netE_flux_z_shell")

    open_air_mask = combined_labels == 0
    # ----------------------------------------------------------------------
    # Step 1 - Shell Entrainment (Accumulates Shell Fluxes near Cloud/Air Edges)
    # ----------------------------------------------------------------------

    # -- Iterate through shell
    shell_list = np.unique(shell_labels)
    shell_list = shell_list[shell_list != 0]

    for label_i in shell_list:
        shell_target = (shell_labels == label_i)

        # --- 1A : Across Cloud ---
        # obtain combined mask for that shell index
        label_mask = (combined_labels == label_i)
        

        # Use contained clouds
        contained_cloud_list = np.unique(cloud_labels[label_mask])
        contained_cloud_list = contained_cloud_list[contained_cloud_list != 0]

        if len(contained_cloud_list) == 0:
            continue

        combined_cloud_mask = np.isin(cloud_labels, contained_cloud_list)

        ent3d, ent, det3d, det = get_intersection_entrainment(shell_target, combined_cloud_mask, nz, ny, nx, shell_ne_x, shell_ne_y, shell_ne_z, dilation_const)
        out_shell_label_ent += ent3d
        out_shell_label_det += det3d
        total_shell_ent_profile += ent
        total_shell_det_profile += det

        # --- 1B: Across open air ---
        ent3d, ent, det3d, det = get_intersection_entrainment(shell_target, open_air_mask, nz, ny, nx, shell_ne_x, shell_ne_y, shell_ne_z, dilation_const)
        out_shell_label_ent_air += ent3d
        out_shell_label_det_air += det3d
        total_shell_ent_profile_air += ent
        total_shell_det_profile_air += det
        
        

    # ----------------------------------------------------------------------
    # Step 2 - Cloud Entrainment (Accumulates Cloud Fluxes near Shell Edges)
    # ----------------------------------------------------------------------
    cloud_list = np.unique(cloud_labels)
    cloud_list = cloud_list[cloud_list != 0]

    for c_label_i in cloud_list:
        cloud_target = (cloud_labels == c_label_i)

        # --- 2A : Across shell ---
        # Finds the index of the very first True value in the cloud mask
        first_idx = np.argmax(cloud_target) 
        label_i = combined_labels.flat[first_idx]
        if label_i == 0:
            continue

        current_shell = (shell_labels == label_i)
        if not np.any(current_shell):
            continue

        ent3d, ent, det3d, det = get_intersection_entrainment(cloud_target, current_shell, nz, ny, nx, cloud_ne_x, cloud_ne_y, cloud_ne_z, dilation_const)

        out_cloud_label_ent += ent3d
        total_cloud_ent_profile += ent
        out_cloud_label_det += det3d
        total_cloud_det_profile += det

        # --- 2B : Across open air ---
        ent3d, ent, det3d, det = get_intersection_entrainment(cloud_target, open_air_mask, nz, ny, nx, cloud_ne_x, cloud_ne_y, cloud_ne_z, dilation_const)

        out_cloud_label_ent_air += ent3d
        total_cloud_ent_profile_air += ent
        out_cloud_label_det_air += det3d
        total_cloud_det_profile_air += det

            
    # --- Exporting ---
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t_idx, t_val, {
        # Across other body
        # -Entrainment
        "slab_shell_entrainment_cloud_boundary.nc": total_shell_ent_profile,
        "slab_shell_label_entrainment_cloud_boundary.nc": out_shell_label_ent,
        "slab_cloud_entrainment_shell_boundary.nc": total_cloud_ent_profile,
        "slab_cloud_label_entrainment_shell_boundary.nc": out_cloud_label_ent,

        # -Detrainment
        "slab_shell_detrainment_cloud_boundary.nc": total_shell_det_profile,
        "slab_shell_label_detrainment_cloud_boundary.nc": out_shell_label_det,
        "slab_cloud_detrainment_shell_boundary.nc": total_cloud_det_profile,
        "slab_cloud_label_detrainment_shell_boundary.nc": out_cloud_label_det,

        # Across open air
        # -Entrainment
        "slab_shell_entrainment_air_boundary.nc": total_shell_ent_profile_air,
        "slab_shell_label_entrainment_air_boundary.nc": out_shell_label_ent_air,
        "slab_cloud_entrainment_air_boundary.nc": total_cloud_ent_profile_air,
        "slab_cloud_label_entrainment_air_boundary.nc": out_cloud_label_ent_air,

        # -Detrainment
        "slab_shell_detrainment_air_boundary.nc": total_shell_det_profile_air,
        "slab_shell_label_detrainment_air_boundary.nc": out_shell_label_det_air,
        "slab_cloud_detrainment_air_boundary.nc": total_cloud_det_profile_air,
        "slab_cloud_label_detrainment_air_boundary.nc": out_cloud_label_det_air,

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
    if source_key in "SEUS":
        path_net_E = Path(config_data["paths"][source_key]["net_E_source_input_dir"])
    else:
        path_net_E = source_input_dir
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
        "netE": path_net_E / "netE.nc",
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
    with xr.open_dataset(file_paths["cloud_labels"], decode_times=False, engine="netcdf4") as ds_meta:
        nz, ny, nx = ds_meta.cloud_labels.shape[1:]

        all_time_vals = ds_meta.time.compute().values
        if source_key in ["SEUS", "RICO"]:
            if source_key in "SEUS":
                start_time = 154800
                step_delta = 7200
            elif source_key in "RICO":
                start_time = 3600
                step_delta = 3600
            target_times = [
                float(t_val) for t_val in all_time_vals
                if t_val >= start_time and (t_val - start_time) % step_delta == 0
            ]
        else:
            target_times = all_time_vals

        if not target_times:
            print("❌ ERROR: No physical times matched the selection criteria!", file=sys.stderr)
            sys.exit(1)

        num_output_times = len(target_times)
        time_vals = np.array(target_times)

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
            print(f" -> Creating file: {file_path}") 
            f = nc.Dataset(str(file_path), "w", format="NETCDF4")
            open_files[filename] = f

            # Base Shared Dimensions
            f.createDimension("time", num_output_times)
            f.createDimension("z", nz)
            f.createVariable("time", "f8", ("time",))[:] = time_vals
            f.createVariable("z", "f4", ("z",))[:] = z_vals

            # Check if this is a 1D Profile or a full 3D Spatial Grid
            if "_label_" in filename:
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
        def task_generator():
                for t_idx, t_val in enumerate(target_times):
                    yield {
                        "t_idx": t_idx,
                        "t_val": t_val,
                        "file_paths": {k: str(v) for k, v in file_paths.items()},
                        "dilation_const": dilation_const,
                        "max_entrainment": max_entrainment_magnitude
                    }

        with multiprocessing.Pool(processes=num_cores) as pool:
            # task_generator() ensures only 'num_cores' worth of timesteps are in memory at a given time
            for t_idx, t_val, payload in pool.imap_unordered(process_timestep_worker, task_generator()):
                print(f"Timestep {t_idx}/{num_output_times - 1} (Physical Time: {t_val:.1f}) finished in ({payload['duration']}). Committing to files...")
                for filename, data_array in payload.items():
                    if filename in ["duration"]:
                        continue
                    var_key = EXPORT_REGISTRY[filename][0]
                    
                    if "_label_" in filename:
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