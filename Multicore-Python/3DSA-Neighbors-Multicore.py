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
    "free_air_neighbors.nc": "free_air_neighbors",
    "shell_neighbors.nc": "shell_neighbors",
    "cloud_neighbors.nc": "cloud_neighbors",
    "free_shell_neighbors.nc": "free_shell_neighbors",
    "shallow_neighbors.nc": "shallow_neighbors",
    "congestus_neighbors.nc": "congestus_neighbors",
    "deep_neighbors.nc": "deep_neighbors",
    "high_neighbors.nc": "high_neighbors"
}

def pad(source):
    """Pads Z with constant zeros, wraps X and Y periodically."""
    padded_source = np.pad(source, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    padded_wrapped_source = np.pad(padded_source, ((0, 0), (1, 1), (1, 1)), mode='wrap')
    return padded_wrapped_source

def compute_neighbor_bitmask(padded_matrix):
    """
    Computes a 6-direction adjacency bitmask.
    Bit 0 (1)  -> posX | Bit 1 (2)  -> negX
    Bit 2 (4)  -> posY | Bit 3 (8)  -> negY
    Bit 4 (16) -> posZ | Bit 5 (32) -> negZ
    """
    nz, ny, nx = padded_matrix.shape
    bitmask = np.zeros((nz-2, ny-2, nx-2), dtype=np.uint8)
    
    arr = padded_matrix.astype(np.uint8)
    
    # Shift and apply bitwise OR directly
    bitmask |= (arr[1:-1, 1:-1, 2:] << 0)   # posX
    bitmask |= (arr[1:-1, 1:-1, :-2] << 1)  # negX
    bitmask |= (arr[1:-1, 2:, 1:-1] << 2)   # posY
    bitmask |= (arr[1:-1, :-2, 1:-1] << 3)  # negY
    bitmask |= (arr[2:, 1:-1, 1:-1] << 4)   # posZ
    bitmask |= (arr[:-2, 1:-1, 1:-1] << 5)  # negZ
    
    return bitmask

# --- Multiprocessing Worker Function ---
def process_timestep_worker(args):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    start_time = time.time()
    t_new, t, cfg = args
    paths = cfg["paths"]


    # Load datasets
    with xr.open_dataset(paths["cloud_mask"], decode_times=False, engine="netcdf4") as ds:
        cloud_mask = ds.cloud_mask.isel(time=t).values.astype(bool)
        
    with xr.open_dataset(paths["shell_mask"], decode_times=False, engine="netcdf4") as ds:
        shell_mask = ds.shell_mask.isel(time=t).values.astype(bool)

    with xr.open_dataset(paths["free_shell_mask"], decode_times=False, engine="netcdf4") as ds:
        free_shell_mask = ds.shell_mask.isel(time=t).values.astype(bool)

    with xr.open_dataset(paths["shallow_mask"], decode_times=False, engine="netcdf4") as ds:
        shallow_mask = ds.shallow_mask.isel(time=t).values.astype(bool)

    with xr.open_dataset(paths["congestus_mask"], decode_times=False, engine="netcdf4") as ds:
        congestus_mask = ds.congestus_mask.isel(time=t).values.astype(bool)

    with xr.open_dataset(paths["deep_mask"], decode_times=False, engine="netcdf4") as ds:
        deep_mask = ds.deep_mask.isel(time=t).values.astype(bool)

    with xr.open_dataset(paths["high_mask"], decode_times=False, engine="netcdf4") as ds:
            high_mask = ds.high_mask.isel(time=t).values.astype(bool)

    free_air_mask = ~(cloud_mask | shell_mask)

    masks_to_process = {
        "free_air_neighbors.nc": free_air_mask,
        "cloud_neighbors.nc": cloud_mask,
        "shell_neighbors.nc": shell_mask,
        "free_shell_neighbors.nc": free_shell_mask,
        "shallow_neighbors.nc": shallow_mask,
        "congestus_neighbors.nc": congestus_mask,
        "deep_neighbors.nc": deep_mask,
        "high_neighbors.nc": high_mask
    }

    payload = {}
    for filename, mask_array in masks_to_process.items():
        payload[filename] = compute_neighbor_bitmask(pad(mask_array))
    del cloud_mask, shell_mask, free_shell_mask, shallow_mask, congestus_mask, deep_mask, high_mask, free_air_mask
    gc.collect()

    # --- Exporting ---
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t_new, t, payload, elapsed_str


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
    output_dir = Path(config_data["paths"][source_key]["output_dir"])
    default_fname = Path(config_data["paths"][source_key]["default_file_name"])

    #in case directory does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── OVERRIDE SYSTEM TMPDIR WITH CONFIG PATH ──────────────────────────
    custom_tmp_dir = output_dir / "tmp"
    custom_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["TMPDIR"] = str(custom_tmp_dir)
    # ──────────────────────────────────────────────────────────────────────

    print(f"Initialization Success:")
    print(f" -> Input & Output Path:   {output_dir}")
    print(f" -> Active CPU Cores:      {num_cores}")
    print("-" * 50)

    print("Checking file dependencies...")
    file_paths = {
        "shell_mask": output_dir / "shell_mask.nc",
        "cloud_mask": output_dir / "cloud_mask.nc",
        "free_shell_mask": output_dir / "free_shell_mask.nc",
        "shallow_mask": output_dir / "shallow_mask.nc",
        "congestus_mask": output_dir / "congestus_mask.nc",
        "deep_mask": output_dir / "deep_mask.nc",
        "high_mask": output_dir / "high_mask.nc"
    }
    #Check that files exist
    for name, path in file_paths.items():
        if not path.is_file():
            print(f"❌ ERROR: Missing target dependency: {path}", file=sys.stderr)
            sys.exit(1)

    # Global structure
    with xr.open_dataset(file_paths["shell_mask"], decode_times=False, engine="netcdf4") as ds_meta:
        num_times = int(ds_meta.time.size)
        nz, ny, nx = ds_meta.shell_mask.shape[1:]
        
        active_timesteps = list(range(num_times))

        time_vals = ds_meta.time.compute().values[active_timesteps]
        num_output_times = len(active_timesteps)

        z_vals = ds_meta.z.compute().values
        y_vals = ds_meta.y.compute().values
        x_vals = ds_meta.x.compute().values

    # --- Preallocate NetCDF file structures ---
    open_files = {}
    try:
        print("Pre-allocating NetCDF file structures on disk...")
        for filename, var_name in EXPORT_REGISTRY.items():
            file_path = output_dir / filename
            if file_path.exists():
                file_path.unlink()
                
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
            
            # Pack variables into highly compressed byte values
            v = f.createVariable(var_name, "u1", ("time", "z", "y", "x"), 
                                 zlib=True, complevel=4, chunksizes=(1, nz, ny, nx))
            v.description = "6-Directional neighborhood flag bitmask. Bit0:posX, Bit1:negX, Bit2:posY, Bit3:negY, Bit4:posZ, Bit5:negZ"

        # --- Start Worker Pool ---
        # Package arguments cleanly into a metadata dictionary
        worker_config = {"paths": {k: str(v) for k, v in file_paths.items()}}

        print(f"Spawning Pool with {num_cores} active workers over {num_output_times} timesteps...")
        pool_tasks = [(new_idx, t_original, worker_config) for new_idx, t_original in enumerate(active_timesteps)]


        with multiprocessing.Pool(processes=num_cores) as pool:
            for t_idx, t_original, worker_payload, duration in pool.imap_unordered(process_timestep_worker, pool_tasks):
                print(f"Timestep {t_idx}/{num_output_times - 1} (Original: {t_original}) completed in ({duration}). Syncing...")
                
                for filename, matrix_data in worker_payload.items():
                    var_key = EXPORT_REGISTRY[filename]
                    open_files[filename].variables[var_key][t_idx, :, :, :] = matrix_data
                    open_files[filename].sync()
                
                del worker_payload; gc.collect()

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

        print("\n✅ All file streams safely disconnected (Program is safe to close).")