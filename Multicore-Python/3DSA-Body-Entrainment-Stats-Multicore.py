import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import math
import numpy as np
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
    "slab_entrainment_stats.nc": ("entrainment", "f4"), # dimensions: t, z
}

GROUPS_STRUCTURE = {
    "Average": {
        "Domain": ["Cloud", "Shell"],
        "Shallow": ["Cloud", "Shell"],
        "Congestus": ["Cloud", "Shell"],
        "Deep": ["Cloud", "Shell"],
        "Free_shell": ["Shell"]
    },
    "Sum": {
        "Domain": ["Cloud", "Shell"],
        "Shallow": ["Cloud", "Shell"],
        "Congestus": ["Cloud", "Shell"],
        "Deep": ["Cloud", "Shell"],
        "Free_shell": ["Shell"]
    }
}

# --- Multiprocessing Worker Function ---
def process_timestep_worker(args):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    
    start_time = time.time()
    t_new, t, worker_config = args
    paths = worker_config["paths"]

    with nc.Dataset(paths["cloud_labels"], "r", parallel=False) as ds_cloud, \
         nc.Dataset(paths["shell_labels"], "r", parallel=False) as ds_shell, \
         nc.Dataset(paths["shallow_mask"], "r", parallel=False) as ds_shal, \
         nc.Dataset(paths["congestus_mask"], "r", parallel=False) as ds_cong, \
         nc.Dataset(paths["deep_mask"], "r", parallel=False) as ds_deep, \
         nc.Dataset(paths["free_shell_mask"], "r", parallel=False) as ds_free, \
         nc.Dataset(paths["shell_entrainment"], "r", parallel=False) as ds_shell_ent, \
         nc.Dataset(paths["cloud_entrainment"], "r", parallel=False) as ds_cloud_ent:

        cloud_labels = ds_cloud.variables["cloud_labels"][t_new, :, :, :]
        shell_labels = ds_shell.variables["shell_labels"][t_new, :, :, :]
        
        masks = {
            "Domain" : (cloud_labels > 0) | (shell_labels > 0),
            "Shallow": ds_shal.variables["shallow_mask"][t_new, :, :, :] > 0,
            "Congestus": ds_cong.variables["congestus_mask"][t_new, :, :, :] > 0,
            "Deep": ds_deep.variables["deep_mask"][t_new, :, :, :] > 0,
            "Free_shell": ds_free.variables["free_shell_mask"][t_new, :, :, :] > 0
        }

        shell_entrainment = ds_shell_ent.variables["entrainment"][t_new, :, :, :]
        cloud_entrainment = ds_cloud_ent.variables["entrainment"][t_new, :, :, :]

    nz, ny, nx = cloud_labels.shape


    counts = {
        "Domain":     {"Cloud": np.zeros(nz, dtype=np.float32), "Shell": np.zeros(nz, dtype=np.float32)},
        "Shallow":    {"Cloud": np.zeros(nz, dtype=np.float32), "Shell": np.zeros(nz, dtype=np.float32)},
        "Congestus":  {"Cloud": np.zeros(nz, dtype=np.float32), "Shell": np.zeros(nz, dtype=np.float32)},
        "Deep":       {"Cloud": np.zeros(nz, dtype=np.float32), "Shell": np.zeros(nz, dtype=np.float32)},
        "Free_shell": {"Shell": np.zeros(nz, dtype=np.float32)}
    }

    z_indices = np.arange(nz)

    # Initialize group structure
    results = {}
    for top in GROUPS_STRUCTURE.keys():
        results[top] = {}
        for mid, bottoms in GROUPS_STRUCTURE[top].items():
            results[top][mid] = {}
            for bot in bottoms:
                results[top][mid][bot] = np.zeros(nz, dtype=np.float32)

    # =====================================================================
    # 1. PROCESS SHELL LABELS (O(N_labels) complexity)
    # =====================================================================
    unique_shells = np.unique(shell_labels)
    unique_shells = unique_shells[unique_shells != 0] # drop background

    for label_id in unique_shells:
        label_locs = (shell_labels == label_id)
        
        # Determine intersecting categories
        intersecting_categories = []
        for cat_name, mask_3d in masks.items():
            if np.any(mask_3d & label_locs):
                intersecting_categories.append(cat_name)

        if not intersecting_categories:
            continue  # Label belongs to no specified category/mask

        # Get vertical footprint (voxel counts per level z)
        z_distribution = np.sum(label_locs, axis=(1, 2))
        active_mask = z_distribution > 0

        entrainment_profile = np.zeros(nz, dtype=np.float32)

        if np.any(active_mask):
            # Perform argmax ONLY on levels where the label is physically present
            active_z = z_indices[active_mask]
            flat_idx = np.argmax(label_locs[active_mask].reshape(len(active_z), -1), axis=1)
            y_indices = flat_idx // nx
            x_indices = flat_idx % nx

            # Assign values only to the active z levels
            entrainment_profile[active_mask] = shell_entrainment[active_z, y_indices, x_indices]

        
        sum_contribution = entrainment_profile * z_distribution

        for cat_name in intersecting_categories:
            results["Sum"][cat_name]["Shell"] += sum_contribution
            counts[cat_name]["Shell"] += z_distribution

    # =====================================================================
    # 2. PROCESS CLOUD LABELS (Skip Free_shell)
    # =====================================================================
    unique_clouds = np.unique(cloud_labels)
    unique_clouds = unique_clouds[unique_clouds != 0] # drop background

    for label_id in unique_clouds:
        label_locs = (cloud_labels == label_id)

        # Determine intersecting categories
        intersecting_categories = []
        for cat_name, mask_3d in masks.items():
            if cat_name == "Free_shell":
                continue
            if np.any(mask_3d & label_locs):
                intersecting_categories.append(cat_name)

        if not intersecting_categories:
            continue

        z_distribution = np.sum(label_locs, axis=(1, 2))
        active_mask = z_distribution > 0

        entrainment_profile = np.zeros(nz, dtype=np.float32)

        if np.any(active_mask):
            active_z = z_indices[active_mask]
            flat_idx = np.argmax(label_locs[active_mask].reshape(len(active_z), -1), axis=1)
            y_indices = flat_idx // nx
            x_indices = flat_idx % nx

            entrainment_profile[active_mask] = cloud_entrainment[active_z, y_indices, x_indices]
        
        sum_contribution = entrainment_profile * z_distribution

        for cat_name in intersecting_categories:
            results["Sum"][cat_name]["Cloud"] += sum_contribution
            counts[cat_name]["Cloud"] += z_distribution

    # =====================================================================
    # 3. COMPUTE AVERAGES
    # =====================================================================
    for cat_name, sub_dict in counts.items():
        for bot_name, voxel_counts in sub_dict.items():
            total_sum = results["Sum"][cat_name][bot_name]
            results["Average"][cat_name][bot_name] = np.divide(
                total_sum, 
                voxel_counts, 
                out=np.zeros_like(total_sum), 
                where=voxel_counts > 0
            )


    # --- Exporting ---
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t_new, t, {
        "slab_entrainment_stats.nc": results,
        "duration": elapsed_str,
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
    output_dir = Path(config_data["paths"][source_key]["output_dir"])

    #in case directory does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── OVERRIDE SYSTEM TMPDIR WITH CONFIG PATH ──────────────────────────
    custom_tmp_dir = output_dir / "tmp"
    custom_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["TMPDIR"] = str(custom_tmp_dir)
    # ──────────────────────────────────────────────────────────────────────

    print(f"Initialization Success:")
    print(f" -> Input & Output Path:  {output_dir}")
    print(f" -> Active CPU Cores:     {num_cores}")
    print("-" * 50)

    print("Checking file dependencies...")

    file_paths = {
        "shallow_mask": output_dir / "shallow_mask.nc",
        "congestus_mask": output_dir / "congestus_mask.nc",
        "deep_mask": output_dir / "deep_mask.nc",
        "free_shell_mask": output_dir / "free_shell_mask.nc", #strictly for sanity check
        "cloud_labels": output_dir / "cloud_labels.nc",
        "shell_labels": output_dir / "shell_labels.nc",
        "shell_entrainment": output_dir / "slab_shell_label_entrainment.nc",
        "cloud_entrainment": output_dir / "slab_cloud_label_entrainment.nc",
    }

    #Check that files exist
    for name, path in file_paths.items():
        if not path.is_file():
            print(f"❌ ERROR: Missing target dependency: {path}", file=sys.stderr)
            sys.exit(1)

    # Global structure
    with nc.Dataset(file_paths["shallow_mask"], "r") as ds_meta:
        num_times = len(ds_meta.dimensions["time"])
        # Dimensions typically go (time, z, y, x)
        nz = len(ds_meta.dimensions["z"])
        active_timesteps = list(range(num_times))

        time_vals = ds_meta.variables["time"][:]
        num_output_times = len(active_timesteps)
        z_vals = ds_meta.variables["z"][:]

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
            
            for top_grp_name, mid_structure in GROUPS_STRUCTURE.items():
                top_grp = f.createGroup(top_grp_name)
                for mid_grp_name, bottom_list in mid_structure.items():
                    mid_grp = top_grp.createGroup(mid_grp_name)
                    for bot_grp_name in bottom_list:
                        bot_grp = mid_grp.createGroup(bot_grp_name)
                        
                        # Create variable inside the leaf group: group/subgroup/subgroup/entrainment
                        bot_grp.createVariable(
                            var_name, data_type, ("time", "z"), 
                            zlib=True, complevel=4, chunksizes=(1, nz)
                        )


        # --- Start Worker Pool ---
        # Package arguments cleanly into a metadata dictionary
        worker_config = {
            "paths": {k: str(v) for k, v in file_paths.items()},
        }

        print(f"Spawning Pool with {num_cores} active workers over {num_output_times} timesteps...")
        pool_tasks = []
        def task_generator():
            for new_idx, t_original in enumerate(active_timesteps):
                yield (new_idx, t_original, worker_config)

        with multiprocessing.Pool(processes=num_cores) as pool:
            for t_new, t_original, payload in pool.imap_unordered(process_timestep_worker, task_generator()):
                print(f"Timestep {t_new}/{num_output_times - 1} finished in ({payload['duration']}). Committing...")
                
                f_stats = open_files["slab_entrainment_stats.nc"]
                stats_data = payload["slab_entrainment_stats.nc"]
                
                for top_grp_name, mid_structure in GROUPS_STRUCTURE.items():
                    for mid_grp_name, bottom_list in mid_structure.items():
                        for bot_grp_name in bottom_list:
                            profile_slice = stats_data[top_grp_name][mid_grp_name][bot_grp_name]
                            
                            # Standardized group path navigation
                            grp_path = f"/{top_grp_name}/{mid_grp_name}/{bot_grp_name}"
                            grp_var = f_stats[grp_path].variables["entrainment"]
                            grp_var[t_new, :] = profile_slice
                            
                f_stats.sync()
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