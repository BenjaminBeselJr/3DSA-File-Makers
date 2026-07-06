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

# =====================================================================
# GLOBAL CONFIGURATION & SHARED REGISTRY
# =====================================================================
EXPORT_REGISTRY = {
    "ql_mask.nc": ("ql_mask", "u1"),
    "w_mask.nc": ("w_mask", "u1"),
    "cloud_mask.nc": ("cloud_mask", "u1"),
    "shell_mask.nc": ("shell_mask", "u1"),
    "shell_labels.nc": ("shell_labels", "u4"),
    "cloud_labels.nc": ("cloud_labels", "u4"),
    "shell_w.nc": ("w", "f4"),
}

# Physical Constants
negative_w_threshold = -0.5
ql_threshold = 10**-5
ql_dilation = 1


# --- Multiprocessing Worker Function ---
def process_timestep_worker(args):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    start_time = time.time()
    t, cfg = args

    paths = cfg["paths"]
    dx, dy = cfg["dx"], cfg["dy"]

    ql_dilation = cfg["ql_dilation"]
    negative_w_threshold = cfg["negative_w_threshold"]
    ql_threshold = cfg["ql_threshold"]
    expansion = cfg["expansion"]

    #load datasets
    with xr.open_dataset(paths["ql"], decode_times=False, engine="netcdf4") as ds_ql, \
         xr.open_dataset(paths["w"], decode_times=False, engine="netcdf4") as ds_w:

        ql_raw = (ds_ql.ql.isel(time=t).fillna(0) > ql_threshold).values.astype(bool)
        w_interpolated = ds_w.w.isel(time=t).rename({'zh': 'z'}).interp(z=ds_ql.z).fillna(0)
        w_mask = (w_interpolated < negative_w_threshold).values.astype(bool)

    local_outline_mask = np.zeros_like(ql_raw, dtype=np.uint8)
    local_shell_labels = np.zeros_like(ql_raw, dtype=np.uint32)
    local_cloud_labels = np.zeros_like(ql_raw, dtype=np.uint32)
    local_shell_w = np.full_like(ql_raw, np.nan, dtype=np.float32)

    if np.any(ql_raw):
        padded_ql_core = np.pad(ql_raw, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        padded_ql_labels = cc3d.connected_components(padded_ql_core, connectivity=6, periodic_boundary=True)
        local_cloud_labels = padded_ql_labels[1:-1, :, :].astype(np.uint32)

    #getting labels that are on all axis at least two grid units thick
    valid_labels = []
    if np.any(local_cloud_labels):
        # 1. Pad for periodic boundaries on X/Y before checking neighborhoods
        padded_labels = np.pad(local_cloud_labels, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        padded_labels = np.pad(padded_labels, ((0, 0), (1, 1), (1, 1)), mode='wrap')

        # 2x2x2 box used to check against ql regions
        thick_core_footprint = np.ones((2, 2, 2), dtype=bool)

        # 2. Erode the mask. This obliterates all 1D lines, diagonals, and single-cell sheets
        binary_cloud = (padded_labels > 0)
        eroded_cloud = scipy.ndimage.binary_erosion(binary_cloud, structure=thick_core_footprint)
        eroded_cloud_unpadded = eroded_cloud[1:-1, 1:-1, 1:-1] # Unpad back to original domain size

        # 3. Find which original labels possess a surviving "thick core"
        surviving_ids = np.unique(local_cloud_labels[eroded_cloud_unpadded])
        valid_labels = surviving_ids[surviving_ids != 0] # Drop background

    #updating to the proper mask
    local_cloud_mask = np.isin(local_cloud_labels, valid_labels)

    if np.any(local_cloud_mask):
        padded_lcm = np.pad(local_cloud_mask, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        padded_lcm_labels = cc3d.connected_components(padded_lcm, connectivity=6, periodic_boundary=True)
        local_cloud_labels = padded_lcm_labels[1:-1, :, :].astype(np.uint32)

    flooded_labels = local_cloud_labels.copy()
    
    if np.any(flooded_labels) and ql_dilation > 0:
        print(f" -> Pre-dilating clouds by {ql_dilation} step(s) to bridge gaps...")
        for _ in range(ql_dilation):
            padded_seed = np.pad(flooded_labels, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            padded_seed = np.pad(padded_seed, ((0, 0), (1, 1), (1, 1)), mode='wrap')
            padded_dilated_seed = scipy.ndimage.grey_dilation(padded_seed, footprint=expansion)
            flooded_labels = padded_dilated_seed[1:-1, 1:-1, 1:-1]

    iteration = 0
    if np.any(local_cloud_labels) and np.any(w_mask):
        print(" -> Flooding cloud labels into the w mask...")
        iteration = 0

        while True:
            # Pad for periodic boundaries on X/Y, constant on Z before dilating
            padded_flood = np.pad(flooded_labels, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            padded_flood = np.pad(padded_flood, ((0, 0), (1, 1), (1, 1)), mode='wrap')

            padded_dilated = scipy.ndimage.grey_dilation(padded_flood, footprint=expansion)
            dilated_step = padded_dilated[1:-1, 1:-1, 1:-1]

            # Masking condition
            grow_mask = w_mask & (flooded_labels == 0) & (dilated_step > 0)
            
            if not np.any(grow_mask):
                break
                
            flooded_labels[grow_mask] = dilated_step[grow_mask]
            iteration += 1
    
    shell_domain = w_mask & ~local_cloud_mask
    local_shell_labels = np.where(shell_domain, flooded_labels, 0).astype(np.uint32)
    local_outline_mask = np.where(local_shell_labels > 0, 1, 0).astype(np.uint8)

    #obtain shell w
    w_slice_physical = w_interpolated.values
    local_shell_w = np.where(local_outline_mask > 0, w_slice_physical, np.nan)

    # --- Exporting ---
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t, {
        "ql_mask.nc": ql_raw.astype(np.uint8),
        "w_mask.nc": w_mask.astype(np.uint8),
        "shell_mask.nc": local_outline_mask,
        "cloud_mask.nc": local_cloud_mask,
        "shell_labels.nc": local_shell_labels,
        "cloud_labels.nc": local_cloud_labels,
        "shell_w.nc": local_shell_w,
        "duration": elapsed_str,
        "iterations": iteration
    }


# --- Main Thread ---
if __name__ == '__main__':
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

    #in case directory does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initialization Success:")
    print(f" -> Source Input Path: {source_input_dir}")
    print(f" -> Output Path:       {output_dir}")
    print(f" -> Active CPU Cores:  {num_cores}")
    print("-" * 50)

    print("Checking file dependencies...")
    file_paths = {
        "ql": source_input_dir / "ql.nc",
        "w": source_input_dir / "w.nc",
    }
    #Check that files exist
    for name, path in file_paths.items():
        if not path.is_file():
            print(f"❌ ERROR: Missing target dependency: {path}", file=sys.stderr)
            sys.exit(1)

    # Global structure
    with xr.open_dataset(file_paths["ql"], decode_times=False, engine="netcdf4") as ds_meta:
        num_times = int(ds_meta.time.size)
        nz, ny, nx = ds_meta.ql.shape[1:]
        time_vals = ds_meta.time.values
        z_vals = ds_meta.z.values
        y_vals = ds_meta.y.values
        x_vals = ds_meta.x.values
        dx = float(ds_meta.x[1] - ds_meta.x[0])
        dy = float(ds_meta.y[1] - ds_meta.y[0])

    # --- Preallocate NetCDF file structures ---
    open_files = {}
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
        "expansion": expansion
    }

    print(f"Spawning Pool with {num_cores} active workers over {num_times} timesteps...")
    pool_tasks = [(t, worker_config) for t in range(num_times)]

    with multiprocessing.Pool(processes=num_cores) as pool:
        for t_idx, payload in pool.imap_unordered(process_timestep_worker, pool_tasks):
            print(f"Timestep {t_idx}/{num_times - 1} finished in ({payload['duration']}) after {payload['iterations']} iterations. Committing to files...")
            
            for filename, data_array in payload.items():
                if filename == "duration" or filename == "iterations":
                    continue
                var_key = EXPORT_REGISTRY[filename][0]
                open_files[filename].variables[var_key][t_idx, :, :, :] = data_array
                open_files[filename].sync()

            gc.collect()

    for file_obj in open_files.values():
        file_obj.close()

    print("\n✅ All computation and exporting complete (Program is safe to close)")