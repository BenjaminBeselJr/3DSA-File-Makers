import os
import sys
import gc
import time
import json
from pathlib import Path
import numpy as np
import xarray as xr
import netCDF4 as nc
import multiprocessing
import argparse

# =====================================================================
# GLOBAL REGISTRIES
# =====================================================================
PHYSICAL_VARS = [
    "b", "w", "vpg", "vpg_b", "vpg_dn", "vpg_dl",
    "pi_b", "pi_dn", "pi_dl",
    "ke_b", "ke_b_eff", "ke_vpg", "ke_vpg_b",
    "ke_vpg_dn", "ke_vpg_dl", "ke_w", "b_eff"
]

MASK_KEYS = [
    "domain", "cloud", "shell", "shallow", "congestus", 
    "deep", "shallow_shell", "congestus_shell", "deep_shell"
]

# --- Multiprocessing Worker Function ---
def process_timestep_worker(args):
    """
    Worker task running on an isolated core. Computes 1D slab averages 
    across all physical variables and masks for a single timestep.
    """

    start_time = time.time()
    t, cfg = args
    paths = cfg["paths"]

    # Initialize a clean dictionary container for this timestep's profiles
    timestep_profiles = {v: {} for v in PHYSICAL_VARS}

    # 1. Load masks for this specific timestep
    with xr.open_dataset(paths["shell_mask"], decode_times=False) as ds_sm, \
         xr.open_dataset(paths["shallow_mask"], decode_times=False) as ds_shm, \
         xr.open_dataset(paths["congestus_mask"], decode_times=False) as ds_cm, \
         xr.open_dataset(paths["deep_mask"], decode_times=False) as ds_dm, \
         xr.open_dataset(paths["cloud_mask"], decode_times=False) as ds_qm:
         
        m_shell = ds_sm.shell_mask.isel(time=t).values.astype(bool)
        m_shallow = ds_shm.shallow_mask.isel(time=t).values.astype(bool)
        m_congestus = ds_cm.congestus_mask.isel(time=t).values.astype(bool)
        m_deep = ds_dm.deep_mask.isel(time=t).values.astype(bool)
        m_cloud = ds_qm.cloud_mask.isel(time=t).values.astype(bool)

    masks = {
        "domain": None,
        "cloud": m_cloud,
        "shell": m_shell,
        "shallow": m_shallow & m_cloud,
        "congestus": m_congestus & m_cloud,
        "deep": m_deep & m_cloud,
        "shallow_shell": m_shallow & m_shell,
        "congestus_shell": m_congestus & m_shell,
        "deep_shell": m_deep & m_shell
    }

    # 2. Iterate through and calculate values for each physics group
    for var_key in PHYSICAL_VARS:
        if var_key == "b_eff":
            with xr.open_dataset(paths["b"], decode_times=False) as ds_b, \
                 xr.open_dataset(paths["vpg_b"], decode_times=False) as ds_vb:
                raw_volume = ds_b["b"].isel(time=t).values + ds_vb["vpg_b"].isel(time=t).values
        elif var_key == "w":
            with xr.open_dataset(paths["w"], decode_times=False) as ds_w:
                raw_volume = ds_w["w"].isel(time=t).values
        else:
            with xr.open_dataset(paths[var_key], decode_times=False) as ds_var:
                raw_volume = ds_var[var_key].isel(time=t).values

        # Compute averages for each mask
        for m_key, mask_matrix in masks.items():
            if m_key == "domain":
                masked_volume = raw_volume
            else:
                masked_volume = np.where(mask_matrix, raw_volume, np.nan)
            
            # 1D profile calculation over horizontal dimensions
            timestep_profiles[var_key][m_key] = np.nanmean(masked_volume, axis=(1, 2)).astype(np.float32)

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t, timestep_profiles, elapsed_str

# --- Main Thread ---
if __name__ == '__main__':
    num_cores = int(os.environ.get("CORE_COUNT", 1))

    parser = argparse.ArgumentParser(description="Process 3DSA pipeline for a specific data source.")
    parser.add_index = parser.add_argument(
        "--data_source", 
        type=str, 
        required=True, 
        help="Key matching the data source configuration block in config.json"
    )
    args = parser.parse_args()

    # --- Setting up directories ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR / "config.json"

    if not CONFIG_PATH.is_file():
        print(f"❌ ERROR: Configuration file missing at: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)

    #load config preset based on 
    source_key = args.data_source
    if source_key not in config_data["paths"]:
        print(f"❌ ERROR: Data source '{source_key}' not found in config.json", file=sys.stderr)
        sys.exit(1)

    source_input_dir = Path(config_data["paths"][source_key]["source_input_dir"])
    output_dir = Path(config_data["paths"]["output_dir"])

    #in case directory does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── OVERRIDE SYSTEM TMPDIR WITH CONFIG PATH ──────────────────────────
    custom_tmp_dir = output_dir / "tmp"
    custom_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["TMPDIR"] = str(custom_tmp_dir)
    # ──────────────────────────────────────────────────────────────────────

    output_file = output_dir / "slab_averages_grouped.nc"

    # Define exact paths for files needed by workers
    file_registry = {v: output_dir / f"{v}.nc" for v in PHYSICAL_VARS if v not in ["b_eff", "w"]}
    file_registry["w"] = source_input_dir / "w.nc"
    file_registry["shell_mask"] = output_dir / "shell_mask.nc"
    file_registry["shallow_mask"] = output_dir / "shallow_mask.nc"
    file_registry["congestus_mask"] = output_dir / "congestus_mask.nc"
    file_registry["deep_mask"] = output_dir / "deep_mask.nc"
    file_registry["cloud_mask"] = output_dir / "cloud_mask.nc"

    # Verify everything exists
    for name, path in file_registry.items():
        if not path.is_file():
            print(f"❌ ERROR: Missing target dependency: {path}", file=sys.stderr)
            sys.exit(1)

    # Gather coordinate metadata
    with xr.open_dataset(file_registry["shell_mask"], decode_times=False) as ds_meta:
        num_times = int(ds_meta.time.size)
        nz = ds_meta.shell_mask.shape[1]
        time_vals = ds_meta.time.values
        z_vals = ds_meta.z.values

    # Reset output file to prevent mixing data with old runs
    if output_file.exists():
        output_file.unlink()

    root_nc = None

    try:
        # --- Preallocate structured NetCDF with Groups ---
        print("Pre-allocating Grouped NetCDF file on disk...")
        root_nc = nc.Dataset(str(output_file), "w", format="NETCDF4")

        group_handles = {}
        for var_key in PHYSICAL_VARS:
            grp = root_nc.createGroup(var_key)
            group_handles[var_key] = grp
            
            # Dimensions
            grp.createDimension("time", num_times)
            grp.createDimension("z", nz)
            
            # Coordinates
            grp.createVariable("time", "f8", ("time",))[:] = time_vals
            grp.createVariable("z", "f4", ("z",))[:] = z_vals
            
            # Profile Data Variables
            for m_key in MASK_KEYS:
                grp.createVariable(m_key, "f4", ("time", "z"), zlib=True, complevel=4)

        # --- Start Worker Pool ---
        worker_config = {"paths": {k: str(v) for k, v in file_registry.items()}}
        pool_tasks = [(t, worker_config) for t in range(num_times)]

        print(f"Spawning Pool with {num_cores} workers over {num_times} timesteps...")
        with multiprocessing.Pool(processes=num_cores) as pool:
            for t_idx, profiles, duration in pool.imap_unordered(process_timestep_worker, pool_tasks):
                print(f"Timestep {t_idx}/{num_times - 1} finished in ({duration}). Committing profiles...")
                
                # Stream the 1D profiles into their allocated netCDF slots
                for var_key, mask_dict in profiles.items():
                    grp = group_handles[var_key]
                    for m_key, profile_array in mask_dict.items():
                        grp.variables[m_key][t_idx, :] = profile_array
                
                root_nc.sync()
                gc.collect()

        root_nc.close()
        print("\n✅ All computation and exporting complete")
    except KeyboardInterrupt:
        print("\n⚠️ Job interrupted or cancelled via Slurm. Flushing and releasing main dataset lock...")
        
    finally:
        # This block ALWAYS runs, guaranteeing that the master lock drops
        if root_nc is not None:
            try:
                root_nc.close()
                print("✅ Root NetCDF dataset handle safely closed and file locks released.")
            except Exception as e:
                print(f"❌ Error encountered while closing root file handle: {e}")
        try:
            import shutil
            if custom_tmp_dir.exists():
                shutil.rmtree(custom_tmp_dir)
                print("🧹 Cleaned up temporary buffer directory.")
        except Exception as e:
            print(f"⚠️ Could not automatically clean up tmp folder: {e}")

        print("[Program is safe to close]")

