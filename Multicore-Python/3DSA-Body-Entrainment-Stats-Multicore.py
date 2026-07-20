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
    "slab_entrainment_stats.nc": ("e", "f4"), # dimensions: t, z
}

DISTANCE_COORDS = {
    "dist_shell_term": "distance_from_shell_termination",
    "dist_cloud_top": "distance_from_cloud_top",
    "norm_cloud_base": "normalized_distance_from_cloud_base",
    "norm_shell_origin": "normalized_distance_from_shell_origin"
}

NORMALIZED_STEP = 0.01

BODY_GROUP_STRUCTURE = {
    "Normal": {
        "Cloud": ["Entrainment_across_the_shell", "Detrainment_across_the_shell", "Net_E_across_the_shell", 
                  "Entrainment_across_open_air", "Detrainment_across_open_air", "Net_E_across_open_air", 
                  "Total_entrainment", "Total_detrainment", "Total_net_E", "Point_Count"],
        "Shell": ["Entrainment_across_clouds", "Detrainment_across_clouds", "Net_E_across_clouds", 
                  "Entrainment_across_open_air", "Detrainment_across_open_air", "Net_E_across_open_air", 
                  "Total_entrainment", "Total_detrainment", "Total_net_E", "Point_Count"],
    },
    "Free_shell": {
        "Shell": ["Entrainment_across_clouds", "Detrainment_across_clouds", "Net_E_across_clouds", 
                  "Entrainment_across_open_air", "Detrainment_across_open_air", "Net_E_across_open_air", 
                  "Total_entrainment", "Total_detrainment", "Total_net_E", "Point_Count"],
    },
}

SUB_GROUP_STRUCTURE = {
    "Domain": BODY_GROUP_STRUCTURE["Normal"].copy(),
    "Shallow": BODY_GROUP_STRUCTURE["Normal"].copy(),
    "Congestus": BODY_GROUP_STRUCTURE["Normal"].copy(),
    "Deep": BODY_GROUP_STRUCTURE["Normal"].copy(),
    "Free_shell": BODY_GROUP_STRUCTURE["Free_shell"].copy()
}

GROUPS_STRUCTURE = {
    "Sum": SUB_GROUP_STRUCTURE.copy(),
    "Distance_from_shell_termination": SUB_GROUP_STRUCTURE.copy(),
    "Distance_from_cloud_top": SUB_GROUP_STRUCTURE.copy(),
    "Normalized_distance_from_cloud_base": SUB_GROUP_STRUCTURE.copy(),
    "Normalized_distance_from_shell_origin": SUB_GROUP_STRUCTURE.copy()
}

COORD_MAP = {
    "Sum": "geom_z",
    "Distance_from_shell_termination": "dist_shell_term",
    "Distance_from_cloud_top": "dist_cloud_top",
    "Normalized_distance_from_cloud_base": "norm_cloud_base",
    "Normalized_distance_from_shell_origin": "norm_shell_origin"
}

def init_worker_process(paths_config, variable_config):
    """ Runs ONCE per worker core to pass file configurations. """
    global worker_paths, worker_vars
    worker_paths = paths_config
    worker_vars = variable_config

# --- Multiprocessing Worker Function ---
def process_timestep_worker(args):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    
    start_time = time.time()
    t_idx, t_val, global_coord_values = args

    data = {}
    for key, path in worker_paths.items():
        var_name = worker_vars[key]
        with nc.Dataset(path, "r") as ds:
            # Slices dataset at t_idx completely avoiding high memory xarray footprints
            data[key] = ds.variables[var_name][t_idx, :, :, :]

    cloud_labels = data["cloud_labels"]
    shell_labels = data["shell_labels"]

    masks = {
        "Domain" : (cloud_labels > 0) | (shell_labels > 0),
        "Shallow": data["shallow_mask"] > 0,
        "Congestus": data["congestus_mask"] > 0,
        "Deep": data["deep_mask"] > 0,
        "Free_shell": data["free_shell_mask"] > 0
    }

    s_ent_cloud = data["shell_entrainment_cloud"]
    s_ent_air   = data["shell_entrainment_air"]
    c_ent_shell = data["cloud_entrainment_shell"]
    c_ent_air   = data["cloud_entrainment_air"]

    s_det_cloud = data["shell_detrainment_cloud"]
    s_det_air   = data["shell_detrainment_air"]
    c_det_shell = data["cloud_detrainment_shell"]
    c_det_air   = data["cloud_detrainment_air"]

    nz, ny, nx = cloud_labels.shape
    z_indices = np.arange(nz)

    distance_volumes = {}
    for c_type, file_key in DISTANCE_COORDS.items():
        distance_volumes[c_type] = data[file_key]

    sum_accumulator = {}

    # Initialize group structure
    for top_name, mid_structure in GROUPS_STRUCTURE.items():
        c_type = COORD_MAP[top_name]
        dim_len = nz if c_type == "geom_z" else len(global_coord_values[c_type])
        
        sum_accumulator[top_name] = {}
        for mid_name, body_struct in mid_structure.items():
            sum_accumulator[top_name][mid_name] = {}
            for body_name, e_types in body_struct.items():
                sum_accumulator[top_name][mid_name][body_name] = {e: np.zeros(dim_len, dtype=np.float32) for e in e_types}

    def bin_flux_to_profile(flat_dist, flat_flux, c_type, num_vals, coord_axis):
        profile = np.zeros(num_vals, dtype=np.float32)
        valid_dist_mask = ~np.isnan(flat_dist)
        fd = flat_dist[valid_dist_mask]
        ff = flat_flux[valid_dist_mask]
        if fd.size == 0:
            return profile
        if "norm_" in c_type:
            indices = np.round((fd - coord_axis[0]) / NORMALIZED_STEP).astype(np.int32)
            matched = (indices >= 0) & (indices < num_vals)
        else:
            indices = np.searchsorted(coord_axis, fd)
            indices = np.clip(indices, 0, num_vals - 1)
            matched = np.abs(coord_axis[indices] - fd) < 1e-3
        
        if np.any(matched):
            profile = np.bincount(indices[matched], weights=ff[matched], minlength=num_vals).astype(np.float32)
        return profile

    # =====================================================================
    # 1. PROCESSING PIPELINE A: SHELLS
    # =====================================================================
    unique_shells = np.unique(shell_labels)
    unique_shells = unique_shells[unique_shells != 0] # drop background

    for label_id in unique_shells:
        label_locs = (shell_labels == label_id)
        
        # Determine intersecting categories
        intersecting_categories = [cat for cat, mask_3d in masks.items() if np.any(mask_3d & label_locs)]
        if not intersecting_categories:
            continue  # Label belongs to no specified category/mask

        # Get vertical footprint (voxel counts per level z)
        z_distribution = np.sum(label_locs, axis=(1, 2))
        active_mask = z_distribution > 0
        if not np.any(active_mask):
            continue

        # Extract argmax voxel values
        active_z = z_indices[active_mask]
        flat_idx = np.argmax(label_locs[active_mask].reshape(len(active_z), -1), axis=1)
        y_indices = flat_idx // nx
        x_indices = flat_idx % nx

        # 1D arrays matching length of active_z
        s_e_c = s_ent_cloud[active_z, y_indices, x_indices]
        s_e_a = s_ent_air[active_z, y_indices, x_indices]
        s_d_c = s_det_cloud[active_z, y_indices, x_indices]
        s_d_a = s_det_air[active_z, y_indices, x_indices]
        z_vols = z_distribution[active_mask]

        # -- Target A1: Geometric Vertical Space --
        for cat in intersecting_categories:

            # Across clouds
            sum_accumulator["Sum"][cat]["Shell"]["Entrainment_across_clouds"][active_z] += (s_e_c * z_vols)
            sum_accumulator["Sum"][cat]["Shell"]["Detrainment_across_clouds"][active_z] += (s_d_c * z_vols)
            sum_accumulator["Sum"][cat]["Shell"]["Net_E_across_clouds"][active_z] += (s_e_c * z_vols) + (s_d_c * z_vols)

            # Across open air
            sum_accumulator["Sum"][cat]["Shell"]["Entrainment_across_open_air"][active_z] += (s_e_a * z_vols)
            sum_accumulator["Sum"][cat]["Shell"]["Detrainment_across_open_air"][active_z] += (s_d_a * z_vols)
            sum_accumulator["Sum"][cat]["Shell"]["Net_E_across_open_air"][active_z] += (s_e_a * z_vols) + (s_d_a * z_vols)

            # Total
            sum_accumulator["Sum"][cat]["Shell"]["Total_entrainment"][active_z] += (s_e_c * z_vols) + (s_e_a * z_vols)
            sum_accumulator["Sum"][cat]["Shell"]["Total_detrainment"][active_z] += (s_d_c * z_vols) + (s_d_a * z_vols)
            sum_accumulator["Sum"][cat]["Shell"]["Total_net_E"][active_z] += (s_e_c * z_vols) + (s_d_c * z_vols) + (s_e_a * z_vols) + (s_d_a * z_vols)

            # Counts
            sum_accumulator["Sum"][cat]["Shell"]["Point_Count"][active_z] += z_vols

        # -- Target A2: Distance Boundary Mapping Spaces --
        for c_key, file_coord in DISTANCE_COORDS.items():
            top_grp_name = [k for k, v in COORD_MAP.items() if v == c_key][0]
            coord_axis = global_coord_values[c_key]
            num_vals = len(coord_axis)

            lbl_distances = distance_volumes[c_key][active_z, y_indices, x_indices]

            prof_ec = bin_flux_to_profile(lbl_distances, s_e_c * z_vols, c_key, num_vals, coord_axis)
            prof_dc = bin_flux_to_profile(lbl_distances, s_d_c * z_vols, c_key, num_vals, coord_axis)
            prof_ea = bin_flux_to_profile(lbl_distances, s_e_a * z_vols, c_key, num_vals, coord_axis)
            prof_da = bin_flux_to_profile(lbl_distances, s_d_a * z_vols, c_key, num_vals, coord_axis)
            prof_cnt = bin_flux_to_profile(lbl_distances, z_vols, c_key, num_vals, coord_axis)

            for cat in intersecting_categories:

                # Across clouds
                sum_accumulator[top_grp_name][cat]["Shell"]["Entrainment_across_clouds"] += prof_ec
                sum_accumulator[top_grp_name][cat]["Shell"]["Detrainment_across_clouds"] += prof_dc
                sum_accumulator[top_grp_name][cat]["Shell"]["Net_E_across_clouds"] += prof_ec + prof_dc

                # Across open air
                sum_accumulator[top_grp_name][cat]["Shell"]["Entrainment_across_open_air"] += prof_ea
                sum_accumulator[top_grp_name][cat]["Shell"]["Detrainment_across_open_air"] += prof_da
                sum_accumulator[top_grp_name][cat]["Shell"]["Net_E_across_open_air"] += prof_ea + prof_da

                # Total
                sum_accumulator[top_grp_name][cat]["Shell"]["Total_entrainment"] += prof_ec + prof_ea
                sum_accumulator[top_grp_name][cat]["Shell"]["Total_detrainment"] += prof_dc + prof_da
                sum_accumulator[top_grp_name][cat]["Shell"]["Total_net_E"] += prof_ec + prof_ea + prof_dc + prof_da

                # Counts
                sum_accumulator[top_grp_name][cat]["Shell"]["Point_Count"] += prof_cnt


    # =====================================================================
    # PROCESSING PIPELINE B: CLOUDS
    # =====================================================================
    unique_clouds = np.unique(cloud_labels)
    unique_clouds = unique_clouds[unique_clouds != 0] # drop background

    for label_id in unique_clouds:
        label_locs = (cloud_labels == label_id)
        intersecting_categories = [cat for cat, mask_3d in masks.items() if cat != "Free_shell" and np.any(mask_3d & label_locs)]
        if not intersecting_categories:
            continue

        z_distribution = np.sum(label_locs, axis=(1, 2))
        active_mask = z_distribution > 0
        if not np.any(active_mask):
            continue

        active_z = z_indices[active_mask]
        flat_idx = np.argmax(label_locs[active_mask].reshape(len(active_z), -1), axis=1)
        y_indices = flat_idx // nx
        x_indices = flat_idx % nx

        c_e_s = c_ent_shell[active_z, y_indices, x_indices]
        c_e_a = c_ent_air[active_z, y_indices, x_indices]
        c_d_s = c_det_shell[active_z, y_indices, x_indices]
        c_d_a = c_det_air[active_z, y_indices, x_indices]
        z_vols = z_distribution[active_mask]

        # -- Target B1: Geometric Vertical Space --
        for cat in intersecting_categories:
            # Across shell
            sum_accumulator["Sum"][cat]["Cloud"]["Entrainment_across_the_shell"][active_z] += (c_e_s * z_vols)
            sum_accumulator["Sum"][cat]["Cloud"]["Detrainment_across_the_shell"][active_z] += (c_d_s * z_vols)
            sum_accumulator["Sum"][cat]["Cloud"]["Net_E_across_the_shell"][active_z] += (c_e_s * z_vols) + (c_d_s * z_vols)

            # Across open air
            sum_accumulator["Sum"][cat]["Cloud"]["Entrainment_across_open_air"][active_z] += (c_e_a * z_vols)
            sum_accumulator["Sum"][cat]["Cloud"]["Detrainment_across_open_air"][active_z] += (c_d_a * z_vols)
            sum_accumulator["Sum"][cat]["Cloud"]["Net_E_across_open_air"][active_z] += (c_e_a * z_vols) + (c_d_a * z_vols)

            # Total
            sum_accumulator["Sum"][cat]["Cloud"]["Total_entrainment"][active_z] += (c_e_s * z_vols) + (c_e_a * z_vols)
            sum_accumulator["Sum"][cat]["Cloud"]["Total_detrainment"][active_z] += (c_d_s * z_vols) + (c_d_a * z_vols)
            sum_accumulator["Sum"][cat]["Cloud"]["Total_net_E"][active_z] += (c_e_s * z_vols) + (c_d_s * z_vols) + (c_e_a * z_vols) + (c_d_a * z_vols)

            # Counts
            sum_accumulator["Sum"][cat]["Cloud"]["Point_Count"][active_z] += z_vols

        # -- Target B2: Distance Boundary Mapping Spaces --
        for c_key, file_coord in DISTANCE_COORDS.items():
            top_grp_name = [k for k, v in COORD_MAP.items() if v == c_key][0]
            coord_axis = global_coord_values[c_key]
            num_vals = len(coord_axis)

            lbl_distances = distance_volumes[c_key][active_z, y_indices, x_indices]

            prof_es = bin_flux_to_profile(lbl_distances, c_e_s * z_vols, c_key, num_vals, coord_axis)
            prof_ds = bin_flux_to_profile(lbl_distances, c_d_s * z_vols, c_key, num_vals, coord_axis)
            prof_ea = bin_flux_to_profile(lbl_distances, c_e_a * z_vols, c_key, num_vals, coord_axis)
            prof_da = bin_flux_to_profile(lbl_distances, c_d_a * z_vols, c_key, num_vals, coord_axis)
            prof_cnt = bin_flux_to_profile(lbl_distances, z_vols, c_key, num_vals, coord_axis)

            for cat in intersecting_categories:
                # Across shell
                sum_accumulator[top_grp_name][cat]["Cloud"]["Entrainment_across_the_shell"] += prof_es
                sum_accumulator[top_grp_name][cat]["Cloud"]["Detrainment_across_the_shell"] += prof_ds
                sum_accumulator[top_grp_name][cat]["Cloud"]["Net_E_across_the_shell"] += prof_es + prof_ds

                # Across open air
                sum_accumulator[top_grp_name][cat]["Cloud"]["Entrainment_across_open_air"] += prof_ea
                sum_accumulator[top_grp_name][cat]["Cloud"]["Detrainment_across_open_air"] += prof_da
                sum_accumulator[top_grp_name][cat]["Cloud"]["Net_E_across_open_air"] += prof_ea + prof_da

                # Total
                sum_accumulator[top_grp_name][cat]["Cloud"]["Total_entrainment"] += prof_es + prof_ea
                sum_accumulator[top_grp_name][cat]["Cloud"]["Total_detrainment"] += prof_ds + prof_da
                sum_accumulator[top_grp_name][cat]["Cloud"]["Total_net_E"] += prof_es + prof_ds + prof_ea + prof_da

                # Count
                sum_accumulator[top_grp_name][cat]["Cloud"]["Point_Count"] += prof_cnt


    # --- Exporting ---
    results = {}
    for top_name, mid_structure in GROUPS_STRUCTURE.items():
        results[top_name] = {}
        for mid_name, body_struct in mid_structure.items():
            results[top_name][mid_name] = {}
            for body_name, e_types in body_struct.items():
                results[top_name][mid_name][body_name] = {}
                for e_type in e_types:
                    results[top_name][mid_name][body_name][e_type] = sum_accumulator[top_name][mid_name][body_name][e_type].astype(np.float32)

    del cloud_labels, shell_labels, masks, distance_volumes, sum_accumulator
    gc.collect()

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t_idx, t_val, results, elapsed_str


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
        #Other body
        "shell_entrainment_cloud": output_dir / "slab_shell_label_entrainment_cloud_boundary.nc",
        "cloud_entrainment_shell": output_dir / "slab_cloud_label_entrainment_shell_boundary.nc",
        "shell_detrainment_cloud": output_dir / "slab_shell_label_detrainment_cloud_boundary.nc",
        "cloud_detrainment_shell": output_dir / "slab_cloud_label_detrainment_shell_boundary.nc",

        # Air
        "shell_entrainment_air": output_dir / "slab_shell_label_entrainment_air_boundary.nc",
        "cloud_entrainment_air": output_dir / "slab_cloud_label_entrainment_air_boundary.nc",
        "shell_detrainment_air": output_dir / "slab_shell_label_detrainment_air_boundary.nc",
        "cloud_detrainment_air": output_dir / "slab_cloud_label_detrainment_air_boundary.nc",
    }

    file_var_names = {
        "shallow_mask": "shallow_mask",
        "congestus_mask": "congestus_mask",
        "deep_mask": "deep_mask",
        "free_shell_mask": "shell_mask", #strictly for sanity check
        "cloud_labels": "cloud_labels",
        "shell_labels": "shell_labels",
        "shell_entrainment_cloud": "entrainment",
        "cloud_entrainment_shell": "entrainment",
        "shell_detrainment_cloud": "detrainment",
        "cloud_detrainment_shell": "detrainment",
        "shell_entrainment_air": "entrainment",
        "cloud_entrainment_air": "entrainment",
        "shell_detrainment_air": "detrainment",
        "cloud_detrainment_air": "detrainment",
    }

    for c_type, file_key in DISTANCE_COORDS.items():
        file_paths[file_key] = output_dir / f"{file_key}.nc"
        if "norm_" in c_type:
            file_var_names[file_key] = "normalized_distance"
        else:
            file_var_names[file_key] = "distance"

    #Check that files exist
    for name, path in file_paths.items():
        if not path.is_file():
            print(f"❌ ERROR: Missing target dependency: {path}", file=sys.stderr)
            sys.exit(1)

    # Global structure
    with nc.Dataset(str(file_paths["shallow_mask"]), "r") as ds_meta:
        num_times = len(ds_meta.dimensions["time"])
        # Dimensions typically go (time, z, y, x)
        nz = len(ds_meta.dimensions["z"])
        active_timesteps = list(range(num_times))

        time_vals = ds_meta.variables["time"][:]
        num_output_times = len(active_timesteps)
        z_vals = ds_meta.variables["z"][:]

    print("Scanning data to determine spatial distance range bins...")
    global_coord_values = {"z_grid": z_vals}
    for c_type, file_key in DISTANCE_COORDS.items():
        with nc.Dataset(str(file_paths[file_key]), "r") as ds_dist:
            subset = ds_dist.variables[file_key][:]
            if "norm_" in c_type:
                bin_min = np.floor(np.nanmin(subset) / NORMALIZED_STEP) * NORMALIZED_STEP
                bin_max = np.ceil(np.nanmax(subset) / NORMALIZED_STEP) * NORMALIZED_STEP
                global_coord_values[c_type] = np.arange(bin_min, bin_max + (NORMALIZED_STEP / 2.0), NORMALIZED_STEP, dtype=np.float32)
            else:
                global_coord_values[c_type] = np.unique(subset[~np.isnan(subset)]).astype(np.float32)

    output_file = output_dir / "slab_entrainment_stats.nc"
    if output_file.exists():
        output_file.unlink()

    root_nc = None
    # --- Preallocate NetCDF file structures ---
    try:
        print("Pre-allocating NetCDF file structures on disk...")
        root_nc = nc.Dataset(str(output_file), "w", format="NETCDF4")
        data_type = EXPORT_REGISTRY["slab_entrainment_stats.nc"][1]

        for top_grp_name, mid_structure in GROUPS_STRUCTURE.items():
            top_grp = root_nc.createGroup(top_grp_name)
            c_type = COORD_MAP[top_grp_name]
            
            for mid_grp_name, body_list in mid_structure.items():
                mid_grp = top_grp.createGroup(mid_grp_name)
                
                for body_grp_name, e_type_list in body_list.items():
                    body_grp = mid_grp.createGroup(body_grp_name)
                    
                    body_grp.createDimension("time", num_output_times)
                    body_grp.createVariable("time", "f8", ("time",))[:] = time_vals
                    
                    if c_type == "geom_z":
                        body_grp.createDimension("z", nz)
                        body_grp.createVariable("z", "f4", ("z",))[:] = z_vals
                        coord_dim_name = "z"
                        axis_len = nz
                    else:
                        axis_vals = global_coord_values[c_type]
                        dim_name = f"exact_{c_type}"
                        body_grp.createDimension(dim_name, len(axis_vals))
                        body_grp.createVariable(dim_name, "f4", (dim_name,))[:] = axis_vals
                        coord_dim_name = dim_name
                        axis_len = len(axis_vals)
                        
                    for e_type_name in e_type_list:
                        body_grp.createVariable(
                            e_type_name, data_type, ("time", coord_dim_name),
                            zlib=True, complevel=4, chunksizes=(1, axis_len)
                        )


        # --- Start Worker Pool ---
        worker_config = {k: str(v) for k, v in file_paths.items()}
        print(f"Spawning Pool with {num_cores} active workers over {num_output_times} timesteps...")
        pool_tasks = [(t_idx, t_val, global_coord_values) for t_idx, t_val in enumerate(time_vals)]

        with multiprocessing.Pool(processes=num_cores, initializer=init_worker_process, initargs=(worker_config, file_var_names)) as pool:
            for t_idx, t_val, complex_profiles, duration in pool.imap_unordered(process_timestep_worker, pool_tasks):
                print(f"Timestep {t_idx}/{num_output_times - 1} finished in ({duration}). Committing data...")
                
                for top_grp_name, mid_structure in GROUPS_STRUCTURE.items():
                    for mid_grp_name, body_structure in mid_structure.items():
                        for body_grp_name, e_type_list in body_structure.items():
                            grp_path = f"/{top_grp_name}/{mid_grp_name}/{body_grp_name}"
                            target_grp = root_nc[grp_path]
                            
                            for e_type_name in e_type_list:
                                profile_array = complex_profiles[top_grp_name][mid_grp_name][body_grp_name][e_type_name]
                                target_grp.variables[e_type_name][t_idx, :] = profile_array
                
                root_nc.sync()
                gc.collect()

        root_nc.close()
        main_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - main_start_time))
        print(f"\n✅ All computation and exporting complete in ({main_elapsed_str})")
    except KeyboardInterrupt:
        print("\n⚠️ Job interrupted or cancelled via Slurm. Closing files safely...")
    finally:
        # This block ALWAYS runs, ensuring handles are dropped on normal exit OR scancel
        print("Flushing and closing all NetCDF file handles...")
        if root_nc is not None:
            try:
                root_nc.close()
                print("✅ Root NetCDF dataset handle safely disconnected.")
            except Exception as e:
                print(f"❌ Error encountered while closing root handle: {e}")

        try:
            import shutil
            if custom_tmp_dir.exists():
                shutil.rmtree(custom_tmp_dir)
                print("🧹 Cleaned up temporary buffer directory.")
        except Exception as e:
            print(f"⚠️ Could not automatically clean up tmp folder: {e}")

        print("\n✅ All file streams safely disconnected (Program is safe to close)")