import os
import sys
import gc
import time
from pathlib import Path
import numpy as np
import xarray as xr
import multiprocessing
import json

# --- Multiprocessing Worker Function ---
def process_group_worker(args):
    """
    Worker task running on an isolated core. Computes time means for ALL masks 
    within a single physical variable group and returns a structured Dataset dictionary.
    """
    start_time = time.time()
    group_name, mask_keys, cfg = args
    input_path = cfg["input_path"]

    group_data = {}
    
    # Cleanly open the group, extract time means for all masks, and close it
    with xr.open_dataset(input_path, group=group_name, decode_times=False) as ds_group:
        for mask_type in mask_keys:
            if mask_type in ds_group:
                # Compute time mean profile over dimension 't'
                time_mean = ds_group[mask_type].mean(dim='time').compute()
                group_data[mask_type] = time_mean

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return group_name, group_data, elapsed_str


# --- Main Thread ---
if __name__ == '__main__':
    num_cores = int(os.environ.get("CORE_COUNT", 1))

    # --- Setting up directories from config ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR / "config.json"

    if not CONFIG_PATH.is_file():
        print(f"❌ ERROR: Configuration file missing at: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)

    output_dir = Path(config_data["paths"]["output_dir"])
    input_file = output_dir / "slab_averages_grouped.nc"
    output_file = output_dir / "time_averaged_slab_averages_grouped.nc"

    # ─── OVERRIDE SYSTEM TMPDIR WITH CONFIG PATH ──────────────────────────
    custom_tmp_dir = output_dir / "tmp"
    custom_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["TMPDIR"] = str(custom_tmp_dir)
    # ──────────────────────────────────────────────────────────────────────

    if not input_file.is_file():
        print(f"❌ ERROR: Source file missing at: {input_file}", file=sys.stderr)
        sys.exit(1)

    # All physical variables (groups) to process
    physical_vars = [
        "b", "w", "vpg", "vpg_b", "vpg_dn", "vpg_dl",
        "pi_b", "pi_dn", "pi_dl",
        "ke_b", "ke_b_eff", "ke_vpg", "ke_vpg_b",
        "ke_vpg_dn", "ke_vpg_dl", "ke_w", "b_eff"
    ]
    mask_keys = ["domain", "shell", "shallow", "congestus", "deep", "shallow_shell", "congestus_shell", "deep_shell"]

    worker_config = {"input_path": str(input_file)}
    pool_tasks = [(group, mask_keys, worker_config) for group in physical_vars]

    # Reset output file to prevent mixing data with old runs
    if output_file.exists():
        output_file.unlink()

    print(f"Spawning Pool with {num_cores} active workers over {len(physical_vars)} groups...")
    
    try:
        # Run parallel workers
        with multiprocessing.Pool(processes=num_cores) as pool:
            for group_name, group_data, duration in pool.imap_unordered(process_group_worker, pool_tasks):
                print(f"Group '{group_name}' compiled in ({duration}). Committing to netCDF...")
                
                # Combine individual mask DataArrays into a clean group dataset
                ds_group = xr.Dataset(group_data)
                
                # Append cleanly to the shared output file under its designated group name
                ds_group.to_netcdf(output_file, mode="a", group=group_name)
                
                gc.collect()

        print("\n✅ All computation and exporting complete")
    except KeyboardInterrupt:
        print("\n⚠️ Job interrupted or cancelled via Slurm. Stopping worker pool safely...")
        print("⚠️ Note: The last group being written may be incomplete or corrupt in the output file.")
    finally:
        try:
            import shutil
            if custom_tmp_dir.exists():
                shutil.rmtree(custom_tmp_dir)
                print("🧹 Cleaned up temporary buffer directory.")
        except Exception as e:
            print(f"⚠️ Could not automatically clean up tmp folder: {e}")
        print("[Program is safe to close]")