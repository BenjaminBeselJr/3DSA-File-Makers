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
    "w_mask.nc": ("w_mask", "u1"),
    "generated_cloud_mask.nc": ("cloud_mask", "u1"),
    "cloud_mask.nc": ("cloud_mask", "u1"),
    "gap_mask.nc": ("gap_mask", "u1"),
    "generated_shell_mask.nc": ("shell_mask", "u1"),
    "shell_mask.nc": ("shell_mask", "u1"),
    "shell_labels.nc": ("shell_labels", "u4"),
    "gap_labels.nc": ("gap_labels", "u4"),
    "generated_cloud_labels.nc": ("cloud_labels", "u4"),
    "cloud_labels.nc": ("cloud_labels", "u4"),
}

# Physical Constants
negative_w_threshold = -1
ql_threshold = 10**-5
ql_dilation = 1
shell_prop_lower_threshold = 0.2

def bleed_labels(original_labels, bleed_mask, expansion):
    iteration = 0
    label_workspace = original_labels.copy()
    if np.any(label_workspace) and np.any(bleed_mask):
        iteration = 0
        while True:
            # Pad for periodic boundaries on X/Y, constant on Z before dilating
            padded_flood = np.pad(label_workspace, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            padded_flood = np.pad(padded_flood, ((0, 0), (1, 1), (1, 1)), mode='wrap')

            padded_dilated = scipy.ndimage.grey_dilation(padded_flood, footprint=expansion)
            dilated_step = padded_dilated[1:-1, 1:-1, 1:-1]

            # Masking condition
            grow_mask = bleed_mask & (label_workspace == 0) & (dilated_step > 0)
            
            if not np.any(grow_mask):
                break
                
            label_workspace[grow_mask] = dilated_step[grow_mask]
            iteration += 1

    return iteration, label_workspace


# --- Multiprocessing Worker Function ---
def process_timestep_worker(args):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    start_time = time.time()
    t_new, t, cfg = args

    paths = cfg["paths"]
    dx, dy = cfg["dx"], cfg["dy"]

    ql_dilation = cfg["ql_dilation"]
    negative_w_threshold = cfg["negative_w_threshold"]
    ql_threshold = cfg["ql_threshold"]
    expansion = cfg["expansion"]
    has_shell_prop = cfg["has_shell_prop"]
    shell_prop_lower_threshold = cfg["shell_prop_lower_threshold"]

    #load datasets
    with xr.open_dataset(paths["ql"], decode_times=False, engine="netcdf4") as ds_ql, \
         xr.open_dataset(paths["w"], decode_times=False, engine="netcdf4") as ds_w:

        ql_raw = (ds_ql.ql.isel(time=t).fillna(0) > ql_threshold).compute().values.astype(bool)
        w_interpolated = ds_w.w.isel(time=t).rename({'zh': 'z'}).interp(z=ds_ql.z).fillna(0)
        w_mask = (w_interpolated < negative_w_threshold).compute().values.astype(bool)

    if has_shell_prop:
        with xr.open_dataset(paths["shell"], decode_times=False, engine="netcdf4") as ds_shell_prop:
            shell_prop_raw = ds_shell_prop.isel(time=t)
            shell_prop_mask = ((shell_prop_raw.shell >= shell_prop_lower_threshold) & (shell_prop_raw.shell < 1)).compute().values.astype(bool)
            cloud_prop_mask = (shell_prop_raw.shell == 1).compute().values.astype(bool)


    local_shell_mask = np.zeros_like(ql_raw, dtype=np.uint8)
    local_shell_labels = np.zeros_like(ql_raw, dtype=np.uint32)
    local_generated_shell_labels = np.zeros_like(ql_raw, dtype=np.uint32)
    local_generated_cloud_labels = np.zeros_like(ql_raw, dtype=np.uint32)

    if np.any(ql_raw):
        padded_ql_core = np.pad(ql_raw, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        padded_ql_labels = cc3d.connected_components(padded_ql_core, connectivity=6, periodic_boundary=True)
        local_generated_cloud_labels = padded_ql_labels[1:-1, :, :].astype(np.uint32)

    #getting labels that are on all axis at least two grid units thick
    valid_labels = []
    if np.any(local_generated_cloud_labels):
        # 1. Pad for periodic boundaries on X/Y before checking neighborhoods
        padded_labels = np.pad(local_generated_cloud_labels, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        padded_labels = np.pad(padded_labels, ((0, 0), (1, 1), (1, 1)), mode='wrap')

        # 2x2x2 box used to check against ql regions
        thick_core_footprint = np.ones((2, 2, 2), dtype=bool)

        # 2. Erode the mask. This obliterates all 1D lines, diagonals, and single-cell sheets
        binary_cloud = (padded_labels > 0)
        eroded_cloud = scipy.ndimage.binary_erosion(binary_cloud, structure=thick_core_footprint)
        eroded_cloud_unpadded = eroded_cloud[1:-1, 1:-1, 1:-1] # Unpad back to original domain size

        # 3. Find which original labels possess a surviving "thick core"
        surviving_ids = np.unique(local_generated_cloud_labels[eroded_cloud_unpadded])
        valid_labels = surviving_ids[surviving_ids != 0] # Drop background

    #updating to the proper mask
    local_generated_cloud_mask = np.isin(local_generated_cloud_labels, valid_labels)

    #update labels
    if np.any(local_generated_cloud_mask):
        padded_lcm = np.pad(local_generated_cloud_mask, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        padded_lcm_labels = cc3d.connected_components(padded_lcm, connectivity=6, periodic_boundary=True)
        local_generated_cloud_labels = padded_lcm_labels[1:-1, :, :].astype(np.uint32)

    flooded_labels = local_generated_cloud_labels.copy()
    
    if np.any(flooded_labels) and ql_dilation > 0:
        print(f" -> Pre-dilating clouds by {ql_dilation} step(s) to bridge gaps...")
        for _ in range(ql_dilation):
            padded_seed = np.pad(flooded_labels, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            padded_seed = np.pad(padded_seed, ((0, 0), (1, 1), (1, 1)), mode='wrap')
            padded_dilated_seed = scipy.ndimage.grey_dilation(padded_seed, footprint=expansion)
            flooded_labels = padded_dilated_seed[1:-1, 1:-1, 1:-1]

    dilated_cloud_labels = flooded_labels.copy()
    print(" -> Flooding cloud labels into the w mask...")
    iteration = 0
    iteration, flooded_labels = bleed_labels(flooded_labels, w_mask, expansion)
    
    #Making my shell
    shell_domain = w_mask & ~local_generated_cloud_mask
    local_generated_shell_labels = np.where(shell_domain, flooded_labels, 0).astype(np.uint32)
    local_generated_shell_mask = np.where(local_generated_shell_labels > 0, 1, 0).astype(np.uint8)

    #obtain gap between my shell and my cloud
    dilated_cloud_mask = np.where(dilated_cloud_labels > 0, True, False)
    gap_domain = dilated_cloud_mask & ~(shell_domain | local_generated_cloud_mask)
    local_gap_labels = np.where(gap_domain, dilated_cloud_labels, 0).astype(np.uint32)
    local_gap_mask = np.where(gap_domain, 1, 0).astype(np.uint8)

    #If it has the shell property, use it instead for the final masks and labels
    if has_shell_prop:

        merged_cloud_domain = local_generated_cloud_mask | cloud_prop_mask
        merged_shell_domain = shell_domain | shell_prop_mask

        #bleed labels
        cloud_bleed_i, bled_cloud_labels = bleed_labels(local_generated_cloud_labels, merged_cloud_domain, expansion)
        shell_bleed_i, bled_shell_labels = bleed_labels(local_generated_shell_labels, merged_shell_domain, expansion)

        #obtain new labels
        local_cloud_labels = np.where(cloud_prop_mask, bled_cloud_labels, 0).astype(np.uint32)
        local_shell_labels = np.where(shell_prop_mask, bled_shell_labels, 0).astype(np.uint32)

        local_cloud_mask = np.where(local_cloud_labels > 0, 1, 0).astype(np.uint8)    
        local_shell_mask = np.where(local_shell_labels > 0, 1, 0).astype(np.uint8)

        # --- Exporting ---
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        return t_new, t, {
            "ql_mask.nc": ql_raw.astype(np.uint8),
            "w_mask.nc": w_mask.astype(np.uint8),
            "original_shell_mask.nc": shell_prop_mask.astype(np.uint8),
            "generated_shell_mask.nc": local_generated_shell_mask,
            "shell_mask.nc": local_shell_mask,
            "gap_mask.nc": local_gap_mask,
            "original_cloud_mask.nc": cloud_prop_mask.astype(np.uint8),
            "generated_cloud_mask.nc": local_generated_cloud_mask.astype(np.uint8),
            "cloud_mask.nc": local_cloud_mask,
            "generated_shell_labels.nc": local_generated_shell_labels,
            "shell_labels.nc": local_shell_labels,
            "gap_labels.nc": local_gap_labels,
            "generated_cloud_labels.nc": local_generated_cloud_labels,
            "cloud_labels.nc": local_cloud_labels,
            "duration": elapsed_str,
            "iterations": iteration,
            "shell_bleed_i": shell_bleed_i,
            "cloud_bleed_i": cloud_bleed_i
        }

    else:
        local_cloud_labels = local_generated_cloud_labels
        local_shell_labels = local_generated_shell_labels
        local_cloud_mask = local_generated_cloud_mask
        local_shell_mask = local_generated_shell_mask

        # --- Exporting ---
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        return t_new, t, {
            "ql_mask.nc": ql_raw.astype(np.uint8),
            "w_mask.nc": w_mask.astype(np.uint8),
            "generated_shell_mask.nc": local_generated_shell_mask,
            "shell_mask.nc": local_shell_mask,
            "gap_mask.nc": local_gap_mask,
            "generated_cloud_mask.nc": local_generated_cloud_mask,
            "cloud_mask.nc": local_cloud_mask,
            "generated_shell_labels.nc": local_generated_shell_labels,
            "shell_labels.nc": local_shell_labels,
            "gap_labels.nc": local_gap_labels,
            "generated_cloud_labels.nc": local_generated_cloud_labels,
            "cloud_labels.nc": local_cloud_labels,
            "duration": elapsed_str,
            "iterations": iteration
        }


# --- Main Thread ---
if __name__ == '__main__':
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
        "w": source_input_dir / "w.nc",
        "shell": source_input_dir / "shell.nc"
    }

    has_shell_prop = True
    #Check that files exist
    for name, path in file_paths.items():
        if not path.is_file():
            if name in ["shell"]:
                has_shell_prop = False
                print(f"⚠️ WARNING: Missing shell property (target : {path}). Excluding exports using shell property")
            else:
                print(f"❌ ERROR: Missing target dependency: {path}", file=sys.stderr)
                sys.exit(1)

    if has_shell_prop:
        EXPORT_REGISTRY["original_shell_mask"] = ("shell_mask", "u1")
        EXPORT_REGISTRY["original_cloud_mask"] = ("cloud_mask", "u1")

    # Global structure
    with xr.open_dataset(file_paths["ql"], decode_times=False, engine="netcdf4") as ds_meta:
        num_times = int(ds_meta.time.size)
        nz, ny, nx = ds_meta.ql.shape[1:]
        
        #Switch slicing depending on source of data
        if source_key in ["SEUS", "RICO"]:
            active_timesteps = [t for t in range(3, num_times, 2)] #all odds except index 1
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

        expansion = np.zeros((3,3,3), dtype=bool)
        expansion[1, 1, :] = True  # X axis
        expansion[1, :, 1] = True  # Y axis
        expansion[:, 1, 1] = True  # Z axis

        # --- Start Worker Pool ---
        # Package arguments cleanly into a metadata dictionary
        worker_config = {
            "paths": {k: str(v) for k, v in file_paths.items()},
            "dx": dx,
            "dy": dy,
            "ql_dilation": ql_dilation,
            "ql_threshold": ql_threshold,
            "negative_w_threshold": negative_w_threshold,
            "expansion": expansion,
            "has_shell_prop" : has_shell_prop
        }

        print(f"Spawning Pool with {num_cores} active workers over {num_output_times} timesteps...")
        pool_tasks = [
            (new_idx, t_original, worker_config) 
            for new_idx, t_original in enumerate(active_timesteps)
        ]

        with multiprocessing.Pool(processes=num_cores) as pool:
            for t_idx, t_original, payload in pool.imap_unordered(process_timestep_worker, pool_tasks):
                if has_shell_prop:
                    print(f"Timestep {t_idx}/{num_output_times - 1} (Original index: {t_original}) finished in ({payload['duration']}) after {payload['iterations']} shell growth iterations, {payload['cloud_bleed_i']} cloud bleed iterations, and {payload['shell_bleed_i']} shell bleed iterations. Committing to files...")
                else:
                    print(f"Timestep {t_idx}/{num_output_times - 1} (Original index: {t_original}) finished in ({payload['duration']}) after {payload['iterations']} iterations. Committing to files...")
                
                for filename, data_array in payload.items():
                    if filename in ["duration", "iterations", "shell_bleed_i", "cloud_bleed_i"]:
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