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
    "ke_vpg_dn", "ke_vpg_dl", "ke_w", "b_eff",
    "qt", "ql", "qv", "thl"
]

COMPUTED_VARS = ["b_eff", "qv"]
LOADED_VARS = [v for v in PHYSICAL_VARS if v not in COMPUTED_VARS]

MASK_KEYS = [
    "domain", "cloud", "shell", "shallow", "congestus", "deep", "high",
    "free_shell", "shallow_shell", "congestus_shell", "deep_shell", "high_shell"
]

DISTANCE_COORDS = {
    "dist_shell_top": "distance_from_shell_top",
    "dist_cloud_top": "distance_from_cloud_top",
    "norm_cloud_base": "normalized_distance_from_cloud_base",
    "norm_shell_base": "normalized_distance_from_shell_base"
}

NORMALIZED_STEP = 0.01

# Global dictionary per worker process to hold persistent dataset handles
worker_datasets = {}

def init_worker_process(paths_config):
    """
    Runs ONCE per worker core when the pool is spawned.
    Opens all netCDF datasets and keeps them open in worker memory.
    """
    global worker_datasets
    
    mask_keys = ["free_shell_mask", "shell_mask", "shallow_mask", "congestus_mask", "deep_mask", "cloud_mask", "high_mask"]
    dist_keys = list(DISTANCE_COORDS.values())
    
    all_keys = LOADED_VARS + mask_keys + dist_keys
    
    for k in all_keys:
        if k in paths_config:
            # Keep open and persistent across tasks on this core
            worker_datasets[k] = xr.open_dataset(paths_config[k], decode_times=False)

# --- Multiprocessing Worker Function ---
def process_timestep_worker(args):
    """
    Worker task running on an isolated core. Computes 1D slab averages 
    across all physical variables and masks for a single timestep.
    """

    start_time = time.time()
    t_idx, t_val, global_coord_values = args

    z_target = global_coord_values["z_grid"]

    # Storage structure: profiles[coord_type][var_key][m_key]
    timestep_profiles = {"geom_z": {v: {} for v in PHYSICAL_VARS}}
    for c_type in DISTANCE_COORDS.keys():
        timestep_profiles[c_type] = {v: {} for v in PHYSICAL_VARS}

    # 1. Access masks via persistent global datasets
    m_shell = worker_datasets["shell_mask"].shell_mask.sel(time=t_val).values.astype(bool)
    m_shallow = worker_datasets["shallow_mask"].shallow_mask.sel(time=t_val).values.astype(bool)
    m_congestus = worker_datasets["congestus_mask"].congestus_mask.sel(time=t_val).values.astype(bool)
    m_deep = worker_datasets["deep_mask"].deep_mask.sel(time=t_val).values.astype(bool)
    m_high = worker_datasets["high_mask"].high_mask.sel(time=t_val).values.astype(bool)
    m_free = worker_datasets["free_shell_mask"].shell_mask.sel(time=t_val).values.astype(bool)
    m_cloud = worker_datasets["cloud_mask"].cloud_mask.sel(time=t_val).values.astype(bool)

    masks = {
        "domain": None,
        "cloud": m_cloud,
        "shell": m_shell,
        "shallow": m_shallow & m_cloud,
        "congestus": m_congestus & m_cloud,
        "deep": m_deep & m_cloud,
        "high": m_high & m_cloud,
        "free_shell": m_free,
        "shallow_shell": m_shallow & m_shell,
        "congestus_shell": m_congestus & m_shell,
        "deep_shell": m_deep & m_shell,
        "high_shell": m_high & m_shell
    }

    # 2. Pre-load all 4 Distance Volume Coordinates up front
    # This ensures we don't repeatedly open distance files inside the loop either
    distance_volumes = {}
    for c_type, file_key in DISTANCE_COORDS.items():
        var_name = "normalized_distance" if "norm_" in c_type else "distance"
        distance_volumes[c_type] = worker_datasets[file_key][var_name].sel(time=t_val).values

    # 3. Compute
    for var_key in PHYSICAL_VARS:
        if var_key == "b_eff":
            v_matrix = worker_datasets["b"]["b"].sel(time=t_val).values + \
                           worker_datasets["vpg_b"]["vpg_b"].sel(time=t_val).values
        elif var_key == "qv":
            v_matrix = worker_datasets["qt"]["qt"].sel(time=t_val).values - \
                           worker_datasets["ql"]["ql"].sel(time=t_val).values
        elif var_key in ["w"]: # from source instead of computed
            w_slice = worker_datasets[var_key][var_key].sel(time=t_val)
            v_matrix = w_slice.rename({'zh': 'z'}).interp(z=z_target).values # interpolate zh to z
        else:
            v_matrix = worker_datasets[var_key][var_key].sel(time=t_val).values

        # --- Sub-step A: Geometric Z Profiles ---
        for m_key, mask_matrix in masks.items():
            if m_key == "domain":
                timestep_profiles["geom_z"][var_key][m_key] = np.nanmean(v_matrix, axis=(1, 2)).astype(np.float32)
            else:
                timestep_profiles["geom_z"][var_key][m_key] = np.nanmean(v_matrix, axis=(1, 2), where=mask_matrix).astype(np.float32)

        valid_data_mask = ~np.isnan(v_matrix)
        v_matrix_clean = np.where(valid_data_mask, v_matrix, 0.0)

        # --- Sub-step B: Inner Loop: Distance Coordinates ---
        for c_type in DISTANCE_COORDS.keys():
            coord_axis = global_coord_values[c_type]
            num_vals = len(coord_axis)
            dist_volume = distance_volumes[c_type]

            for m_key, mask_matrix in masks.items():
                base_valid = valid_data_mask if m_key == "domain" else (valid_data_mask & mask_matrix)

                flat_dist = dist_volume[base_valid]
                flat_vals = v_matrix_clean[base_valid]
                # filter out nans
                valid_dist_mask = ~np.isnan(flat_dist)
                flat_dist = flat_dist[valid_dist_mask]
                flat_vals = flat_vals[valid_dist_mask]

                dist_profile = np.full(num_vals, np.nan, dtype=np.float32)
                
                if flat_dist.size == 0:
                    timestep_profiles[c_type][var_key][m_key] = dist_profile
                    continue

                # --- Dual-Strategy Matrix Mapping ---
                if "norm_" in c_type:
                    # Continuous Floating Binning (Find which 0.01 increment bucket it falls into)
                    indices = np.round((flat_dist - coord_axis[0]) / 0.01).astype(np.int32)
                    matched = (indices >= 0) & (indices < num_vals)
                else:
                    # Non-Linear Grid Matching (Find the closest discrete grid level)
                    indices = np.searchsorted(coord_axis, flat_dist)
                    indices = np.clip(indices, 0, num_vals - 1)
                    
                    # Because coord_axis contains the exact physical grid values, 
                    # your raw points will match the unique list within float precision.
                    matched = np.abs(coord_axis[indices] - flat_dist) < 1e-3

                # strip invalid indexes
                bin_indices = indices[matched]
                flat_vals_matched = flat_vals[matched]

                if bin_indices.size == 0:
                    timestep_profiles[c_type][var_key][m_key] = dist_profile
                    continue

                counts = np.bincount(bin_indices, minlength=num_vals)
                sums = np.bincount(bin_indices, weights=flat_vals_matched, minlength=num_vals)

                # Compute the mean safely
                with np.errstate(divide='ignore', invalid='ignore'):
                    dist_profile = np.where(counts > 0, sums / counts, np.nan).astype(np.float32)
                        
                timestep_profiles[c_type][var_key][m_key] = dist_profile

    del v_matrix, v_matrix_clean, distance_volumes, masks, valid_data_mask
    gc.collect()

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t_idx, t_val, timestep_profiles, elapsed_str

# --- Main Thread ---
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main_start_time = time.time()

    # --- Configurations ---
    num_cores = int(os.environ.get("CORE_COUNT", 1))

    parser = argparse.ArgumentParser(description="Process 3DSA pipeline for a specific data source.")
    parser.add_argument(
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
    output_dir = Path(config_data["paths"][source_key]["output_dir"])

    #in case directory does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── OVERRIDE SYSTEM TMPDIR WITH CONFIG PATH ──────────────────────────
    custom_tmp_dir = output_dir / "tmp"
    custom_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["TMPDIR"] = str(custom_tmp_dir)
    # ──────────────────────────────────────────────────────────────────────

    output_file = output_dir / "slab_averages_grouped.nc"

    # Define exact paths for files needed by workers
    file_registry = {v: output_dir / f"{v}.nc" for v in PHYSICAL_VARS if v not in ["b_eff", "w", "qt", "ql", "qv", "thl"]}
    file_registry["w"] = source_input_dir / "w.nc"
    file_registry["qt"] = source_input_dir / "qt.nc"
    file_registry["ql"] = source_input_dir / "ql.nc"
    file_registry["thl"] = source_input_dir / "thl.nc"
    file_registry["shell_mask"] = output_dir / "shell_mask.nc"
    file_registry["shallow_mask"] = output_dir / "shallow_mask.nc"
    file_registry["congestus_mask"] = output_dir / "congestus_mask.nc"
    file_registry["deep_mask"] = output_dir / "deep_mask.nc"
    file_registry["high_mask"] = output_dir / "high_mask.nc"
    file_registry["free_shell_mask"] = output_dir / "free_shell_mask.nc"
    file_registry["cloud_mask"] = output_dir / "cloud_mask.nc"

    # Distances to also average over
    for c_type, f_name in DISTANCE_COORDS.items():
        file_registry[f_name] = output_dir / f"{f_name}.nc"

    # Verify everything exists
    for name, path in file_registry.items():
        if not path.is_file():
            print(f"❌ ERROR: Missing target dependency: {path}", file=sys.stderr)
            sys.exit(1)

    # Gather coordinate metadata
    with xr.open_dataset(file_registry["shell_mask"], decode_times=False) as ds_meta:
        nz = ds_meta.shell_mask.shape[1]
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

    # ─── CALCULATE GLOBAL DISTANCE RANGES FOR DIMENSIONS ───────────────────
    print("Scanning data to determine exact spatial coordinate ranges...")
    global_coord_values = {}
    global_coord_values["z_grid"] = z_vals
    
    for c_type, file_key in DISTANCE_COORDS.items():
        with xr.open_dataset(file_registry[file_key], decode_times=False) as ds_dist:
            # Load the unique values across all relevant active timesteps
            var_name = "normalized_distance" if "norm_" in c_type else "distance"
            subset = ds_dist[var_name].sel(time=time_vals).values
            
            if "norm_" in c_type:
                # Strategy A: Fractional binning from absolute min to max in 0.01 increments
                min_val = np.nanmin(subset)
                max_val = np.nanmax(subset)
                
                bin_min = np.floor(min_val / NORMALIZED_STEP) * NORMALIZED_STEP
                bin_max = np.ceil(max_val / NORMALIZED_STEP) * NORMALIZED_STEP
                
                global_coord_values[c_type] = np.arange(bin_min, bin_max + (NORMALIZED_STEP / 2.0), NORMALIZED_STEP, dtype=np.float32)
            else:
                # Strategy B: Extract the EXACT non-linear grid levels directly using unique
                # Drop NaNs before extracting unique values to keep the coordinate array clean
                valid_subset = subset[~np.isnan(subset)]
                raw_unique = np.unique(valid_subset).astype(np.float32)
                
                # Double-check tolerance spacing to filter out any minor float precision artifacts if needed
                global_coord_values[c_type] = raw_unique
                
            print(f"  ↳ Coordinate '{c_type}' range: [{global_coord_values[c_type][0]:.2f} to {global_coord_values[c_type][-1]:.2f}] | Total Bins/Levels: {len(global_coord_values[c_type])}")

    # Reset output file to prevent mixing data with old runs
    if output_file.exists():
        output_file.unlink()

    root_nc = None

    try:
        # --- Preallocate structured NetCDF with Groups ---
        print("Pre-allocating Grouped NetCDF file on disk...")
        root_nc = nc.Dataset(str(output_file), "w", format="NETCDF4")

        group_registry = {}
        all_coord_types = ["geom_z"] + list(DISTANCE_COORDS.keys())

        for c_type in all_coord_types:
            coord_grp = root_nc.createGroup(c_type)
            group_registry[c_type] = {}
            
            for var_key in PHYSICAL_VARS:
                phys_grp = coord_grp.createGroup(var_key)
                group_registry[c_type][var_key] = phys_grp
                
                phys_grp.createDimension("time", num_output_times)
                phys_grp.createVariable("time", "f8", ("time",))[:] = time_vals
                
                if c_type == "geom_z":
                    phys_grp.createDimension("z", nz)
                    phys_grp.createVariable("z", "f4", ("z",))[:] = z_vals
                    coord_dim_name = "z"
                else:
                    axis_vals = global_coord_values[c_type]
                    dim_name = f"exact_{c_type}"
                    phys_grp.createDimension(dim_name, len(axis_vals))
                    phys_grp.createVariable(dim_name, "f4", (dim_name,))[:] = axis_vals
                    coord_dim_name = dim_name
                
                for m_key in MASK_KEYS:
                    phys_grp.createVariable(m_key, "f4", ("time", coord_dim_name), zlib=True, complevel=4)

        # --- Start Worker Pool ---
        worker_config = {k: str(v) for k, v in file_registry.items()}
        print(f"Spawning Pool with {num_cores} active workers over {num_output_times} timesteps...")
        pool_tasks = [
            (t_idx, t_val, global_coord_values) 
            for t_idx, t_val in enumerate(target_times)
        ]

        with multiprocessing.Pool(processes=num_cores, initializer=init_worker_process, initargs=(worker_config,)) as pool:
            for t_idx, t_val, complex_profiles, duration in pool.imap_unordered(process_timestep_worker, pool_tasks):
                print(f"Timestep {t_idx}/{num_output_times - 1} finished in ({duration}). Committing data...")
                
                for c_type, phys_dict in complex_profiles.items():
                    for var_key, mask_dict in phys_dict.items():
                        phys_grp = group_registry[c_type][var_key]
                        for m_key, profile_array in mask_dict.items():
                            phys_grp.variables[m_key][t_idx, :] = profile_array
                
                root_nc.sync()
                gc.collect()

        root_nc.close()
        main_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - main_start_time))
        print(f"\n✅ All computation and exporting complete in ({main_elapsed_str})")
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

