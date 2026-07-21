import os
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
import cc3d
import argparse

# =====================================================================
# GLOBAL CONFIGURATION & SHARED REGISTRY
# =====================================================================
EXPORT_REGISTRY = {
    "ql_mask.nc": ("ql_mask", "u1"),
    "cloud_mask.nc": ("cloud_mask", "u1"),
    "shell_mask.nc": ("shell_mask", "u1"),
    "combined_mask.nc": ("mask", "u1"),
    "free_shell_mask.nc": ("shell_mask", "u1"),
    "cloud_labels.nc": ("cloud_labels", "u4"),
    "shell_labels.nc": ("shell_labels", "u4"),
    "combined_labels.nc": ("labels", "u4")
    
}

# Physical Constants
ql_threshold = 10**-5
shell_prop_lower_threshold = 0.2

def label(source):
    #source must be (z,y,x)
    #returns a labeled version wrapping across x,y

    padded_source = np.pad(source, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    padded_labeled_source = cc3d.connected_components(padded_source, connectivity=6, periodic_boundary=True)
    return padded_labeled_source[1:-1, :, :]

def strip_small_components(original):
    # ----- Input -----
    # original : numpy to be stripped of small components (must be z,y,x)
    # ----- Output -----
    # stripped_labels
    # stripped_mask

    if np.any(original):
        padded_pre_init_label = np.pad(original, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        padded_init_labels = cc3d.connected_components(padded_pre_init_label, connectivity=6, periodic_boundary=True)
        padded_init_labels = np.pad(padded_init_labels, ((0, 0), (1, 1), (1, 1)), mode='wrap')

        #getting ql labels that are on all axis at least two grid units thick
        valid_labels = []
        if np.any(padded_init_labels):
            # 2x2x2 box used to check against ql regions
            thick_core_footprint = np.ones((2, 2, 2), dtype=bool)

            # 2. Erode the mask. This obliterates all 1D lines, diagonals, and single-cell sheets
            binary_init_labels = (padded_init_labels > 0)
            eroded = scipy.ndimage.binary_erosion(binary_init_labels, structure=thick_core_footprint)

            # 3. Find which original labels possess a surviving "thick core"
            surviving_ids = np.unique(padded_init_labels[eroded])
            valid_labels = surviving_ids[surviving_ids != 0] # Drop background

            #Unpack and updating to the proper mask
            init_labels = padded_init_labels[1:-1, 1:-1, 1:-1]
            stripped_mask = np.isin(init_labels, valid_labels)
            stripped_labels = label(stripped_mask)

            return stripped_labels, stripped_mask
        else:
            return np.zeros_like(original), np.zeros_like(original)
    else:
        return np.zeros_like(original), np.zeros_like(original)

# --- Multiprocessing Worker Function ---
def process_timestep_worker(args):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    start_time = time.time()
    t_idx, t_val, cfg = args

    paths = cfg["paths"]

    shell_prop_lower_threshold = cfg["shell_prop_lower_threshold"]
    ql_threshold = cfg["ql_threshold"]

    #load datasets
    with xr.open_dataset(paths["ql"], decode_times=False, engine="netcdf4") as ds_ql, \
        xr.open_dataset(paths["shell"], decode_times=False, engine="netcdf4") as ds_shell_prop:

        ql_raw = (ds_ql.ql.sel(time=t_val).fillna(0) > ql_threshold).compute().values.astype(bool)
        shell_prop_raw = ds_shell_prop.sel(time=t_val)
        shell_prop_mask = ((shell_prop_raw.shell >= shell_prop_lower_threshold) & (shell_prop_raw.shell < 1)).compute().values.astype(bool)
        cloud_prop_mask = (shell_prop_raw.shell > 0.99).compute().values.astype(bool)
        
            
    #--- Step 1 : Obtain Filtered Cloud Mask ---
    #strip small parts of ql
    ql_labels, ql_mask = strip_small_components(ql_raw)

    #apply to cloud mask
    restricted_cloud_mask = ql_mask & cloud_prop_mask

    local_cloud_mask = restricted_cloud_mask
    local_cloud_labels = label(local_cloud_mask)

    local_shell_mask = shell_prop_mask & ~local_cloud_mask

    #--- Step 2: Obtain Shell Labels ---
    #make labels for combined object
    local_combined_mask = local_shell_mask | local_cloud_mask
    local_combined_labels = label(local_combined_mask)

    #strip to just the shell
    local_shell_labels = np.where(local_shell_mask, local_combined_labels, 0)
    
    #--- Step 3: Find Free Standing Shells ---
    #get cloud labels for combined objects
    connected_labels = np.where(local_cloud_mask, local_combined_labels, 0)

    #turn the labels into a list
    connected_ids = np.unique(connected_labels) #Don't drop background because we are negating this next

    #obtain free shell mask
    local_free_shell_mask = ~np.isin(local_shell_labels, connected_ids)

    #mask the labels
    local_free_shell_labels = np.where(local_free_shell_mask, local_shell_labels, 0)

    # --- Exporting ---
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t_idx, t_val, {
        "ql_mask.nc": ql_mask.astype(np.uint8),
        "shell_mask.nc": local_shell_mask.astype(np.uint8),
        "cloud_mask.nc": local_cloud_mask.astype(np.uint8),
        "combined_mask.nc": local_combined_mask.astype(np.uint8),
        "free_shell_mask.nc": local_free_shell_mask.astype(np.uint8),
        "shell_labels.nc": local_shell_labels,
        "cloud_labels.nc": local_cloud_labels.astype(np.uint32),
        "combined_labels.nc": local_combined_labels.astype(np.uint32),
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
    print(f" -> Source Input Path: {source_input_dir}")
    print(f" -> Output Path:       {output_dir}")
    print(f" -> Active CPU Cores:  {num_cores}")
    print("-" * 50)

    print("Checking file dependencies...")

    file_paths = {
        "ql": source_input_dir / "ql.nc",
        "shell": source_input_dir / "shell.nc",
        "netE": source_input_dir / "netE.nc"
    }

    has_shell_prop = True
    #Check that files exist
    for name, path in file_paths.items():
        if not path.is_file():
            print(f"❌ ERROR: Missing target dependency: {path}", file=sys.stderr)
            sys.exit(1)

    # Global structure
    with xr.open_dataset(file_paths["ql"], decode_times=False, engine="netcdf4") as ds_meta, \
        xr.open_dataset(file_paths["netE"], decode_times=False, engine="netcdf4").netE_flux_y_shell as ds_ex_e:
        nz, ny, nx = ds_meta.ql.shape[1:]

        all_time_vals = ds_ex_e.time.compute().values
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

        # ─── TEMPORARY DEBUG SLICE ──────────────────────────────────────────
        # Change [:3] to whatever number of test timesteps you want (e.g., [:5], [:1])
        #target_times = target_times[:3] 
        # ────────────────────────────────────────────────────────────────────

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
            f.createDimension("time", num_output_times)
            f.createDimension("z", nz)
            f.createDimension("y", ny)
            f.createDimension("x", nx)
            
            f.createVariable("time", "f8", ("time",))[:] = time_vals
            f.createVariable("z", "f4", ("z",))[:] = z_vals
            f.createVariable("y", "f4", ("y",))[:] = y_vals
            f.createVariable("x", "f4", ("x",))[:] = x_vals
            
            f.createVariable(var_name, data_type, ("time", "z", "y", "x"), 
                                zlib=True, complevel=4, chunksizes=(1, nz, ny, nx))


        # --- Start Worker Pool ---
        # Package arguments cleanly into a metadata dictionary
        worker_config = {
            "paths": {k: str(v) for k, v in file_paths.items()},
            "dx": dx,
            "dy": dy,
            "ql_threshold": ql_threshold,
            "shell_prop_lower_threshold": shell_prop_lower_threshold
        }

        print(f"Spawning Pool with {num_cores} active workers over {num_output_times} timesteps...")
        pool_tasks = [
            (t_idx, t_val, worker_config) 
            for t_idx, t_val in enumerate(target_times)
        ]

        with multiprocessing.Pool(processes=num_cores) as pool:
            for t_idx, t_val, payload in pool.imap_unordered(process_timestep_worker, pool_tasks):
                print(f"Timestep {t_idx}/{num_output_times - 1} (Physical Time: {t_val:.1f}) finished in ({payload['duration']}). Committing to files...")
                for filename, data_array in payload.items():
                    if filename in ["duration"]:
                        continue
                    var_key = EXPORT_REGISTRY[filename][0]
                    open_files[filename].variables[var_key][t_idx, :, :, :] = data_array
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