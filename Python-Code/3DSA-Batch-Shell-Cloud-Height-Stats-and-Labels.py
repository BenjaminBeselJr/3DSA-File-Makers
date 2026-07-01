import os
import math
import numpy as np
import xarray as xr
import scipy.ndimage
from pathlib import Path
import netCDF4 as nc
import cc3d
import gc
import sys
import pandas as pd
import json

# Input
input_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")

#Output
output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir.mkdir(parents=True, exist_ok=True)

#Check that files exist
path_ds_shell_labels = Path(input_dir / "shell_labels.nc")
if not path_ds_shell_labels.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_shell_labels}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_cloud_labels = Path(input_dir / "cloud_labels.nc")
if not path_ds_cloud_labels.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_cloud_labels}\nEnding program early.", file=sys.stderr)
    sys.exit(1)

#Loading datasets
ds_cloud_labels = xr.open_dataset(path_ds_cloud_labels, decode_times=False)
ds_shell_labels = xr.open_dataset(path_ds_shell_labels, decode_times=False)


num_times = int(ds_cloud_labels.time.size)
nz, ny, nx = ds_cloud_labels.cloud_labels.shape[1:]
z_coordinates = ds_cloud_labels.z.values

#Export vars
#u1 means uint8 and f4 means float32
export_registry = {
    "shell_origin.nc": ("shell_origin", "f4"),
    "shell_termination.nc": ("shell_termination", "f4"),
    "shell_depth.nc": ("shell_depth", "f4"),
    "cloud_base.nc": ("cloud_base", "f4"),
    "cloud_top.nc": ("cloud_top", "f4"),
    "cloud_depth.nc": ("cloud_depth", "f4"),
    "shallow_mask.nc": ("shallow_mask", "u1"),
    "congestus_mask.nc": ("congestus_mask", "u1"),
    "deep_mask.nc": ("deep_mask", "u1")
}
#create blank datasets on memory
print("Pre-allocating NetCDF file structures on disk...")
for filename, (var_name, data_type) in export_registry.items():
    file_path = str(output_dir / filename)
    with nc.Dataset(file_path, "w", format="NETCDF4") as f:
        f.createDimension("time", num_times)
        f.createDimension("z", nz)
        f.createDimension("y", ny)
        f.createDimension("x", nx)
        
        t_v = f.createVariable("time", "f8", ("time",))
        z_v = f.createVariable("z", "f4", ("z",))
        y_v = f.createVariable("y", "f4", ("y",))
        x_v = f.createVariable("x", "f4", ("x",))
        
        t_v[:] = ds_cloud_labels.time.values
        z_v[:] = ds_cloud_labels.z.values
        y_v[:] = ds_cloud_labels.y.values
        x_v[:] = ds_cloud_labels.x.values
        
        f.createVariable(var_name, data_type, ("time", "z", "y", "x"), zlib=True, complevel=4, chunksizes=(1, nz, ny, nx), fill_value=False)
#open blank arrays created earlier
open_files = {fname: nc.Dataset(str(output_dir / fname), "a") for fname in export_registry}


all_timesteps_data = {}
for t in range(num_times):
    print(f"\n--- (Cloud/Shell Stats) Processing Timestep {t}/{num_times - 1} ---")
    #Slices
    cloud_labels_slice = ds_cloud_labels.cloud_labels[t, :, :, :].values
    shell_labels_slice = ds_shell_labels.shell_labels[t, :, :, :].values

    #initialize local arrays for this timestep
    local_shell_origin = np.zeros_like(shell_labels_slice, dtype=np.float32)
    local_shell_termination = np.zeros_like(shell_labels_slice, dtype=np.float32)
    local_shell_depth = np.zeros_like(shell_labels_slice, dtype=np.float32)
    local_cloud_bottom = np.zeros_like(shell_labels_slice, dtype=np.float32)
    local_cloud_top = np.zeros_like(shell_labels_slice, dtype=np.float32)
    local_cloud_depth = np.zeros_like(shell_labels_slice, dtype=np.float32)
    local_congestus_mask = np.zeros_like(shell_labels_slice, dtype=bool)
    local_deep_mask = np.zeros_like(shell_labels_slice, dtype=bool)
    local_shallow_mask = np.zeros_like(shell_labels_slice, dtype=bool)
    
    matching_labels = set(cloud_labels_slice)
    matching_labels.discard(0)

    if matching_labels:
        all_timesteps_data[t] = {}

    #Stats calc
    for obj_id in matching_labels:
        cloud_obj_slice = np.where(cloud_labels_slice == obj_id)
        shell_obj_slice = np.where(shell_labels_slice == obj_id)

        combined_obj_mask = (cloud_labels_slice == obj_id) | (shell_labels_slice == obj_id)

        # Initialize default values for this object
        min_z_shell = np.nan
        max_z_shell = np.nan
        shell_depth = np.nan
        min_z_cloud = np.nan
        max_z_cloud = np.nan
        cloud_depth = np.nan
        classification = "shallow"

        if shell_obj_slice[0].size > 0:
            min_z_shell = z_coordinates[shell_obj_slice[0].min()]
            max_z_shell = z_coordinates[shell_obj_slice[0].max()]
            shell_depth = max_z_shell - min_z_shell

            local_shell_origin[combined_obj_mask] = min_z_shell
            local_shell_termination[combined_obj_mask] = max_z_shell
            local_shell_depth[combined_obj_mask] = shell_depth

        if cloud_obj_slice[0].size > 0:
            min_z_cloud = z_coordinates[cloud_obj_slice[0].min()]
            max_z_cloud = z_coordinates[cloud_obj_slice[0].max()]
            cloud_depth = max_z_cloud - min_z_cloud

            local_cloud_bottom[combined_obj_mask] = min_z_cloud
            local_cloud_top[combined_obj_mask] = max_z_cloud
            local_cloud_depth[combined_obj_mask] = cloud_depth  
            
            if max_z_cloud > 5000: #cloud is deep
                classification = "deep"
                local_deep_mask[combined_obj_mask] = True
            elif max_z_cloud > 2000: #cloud is congestus
                classification = "congestus"
                local_congestus_mask[combined_obj_mask] = True
            else: #cloud is shallow
                classification = "shallow"
                local_shallow_mask[combined_obj_mask] = True

        all_timesteps_data[t][int(obj_id)] = {
            "cloud_base": float(min_z_cloud) if not np.isnan(min_z_cloud) else None,
            "cloud_top": float(max_z_cloud) if not np.isnan(max_z_cloud) else None,
            "cloud_depth": float(cloud_depth) if not np.isnan(cloud_depth) else None,
            "shell_origin": float(min_z_shell) if not np.isnan(min_z_shell) else None,
            "shell_termination": float(max_z_shell) if not np.isnan(max_z_shell) else None,
            "shell_depth": float(shell_depth) if not np.isnan(shell_depth) else None,
            "class": classification
        }

    print(f"Labels and Stats computed in timestep {t}.")

    # Commit changes cleanly to file storage system
    open_files["shell_origin.nc"].variables["shell_origin"][t, :, :, :] = local_shell_origin
    open_files["shell_termination.nc"].variables["shell_termination"][t, :, :, :] = local_shell_termination
    open_files["shell_depth.nc"].variables["shell_depth"][t, :, :, :] = local_shell_depth
    open_files["cloud_base.nc"].variables["cloud_base"][t, :, :, :] = local_cloud_bottom
    open_files["cloud_top.nc"].variables["cloud_top"][t, :, :, :] = local_cloud_top
    open_files["cloud_depth.nc"].variables["cloud_depth"][t, :, :, :] = local_cloud_depth
    open_files["congestus_mask.nc"].variables["congestus_mask"][t, :, :, :] = local_congestus_mask
    open_files["deep_mask.nc"].variables["deep_mask"][t, :, :, :] = local_deep_mask
    open_files["shallow_mask.nc"].variables["shallow_mask"][t, :, :, :] = local_shallow_mask

    for open_file in open_files.values():
        open_file.sync()

    # Memory optimization collection sweep
    gc.collect()


# Save nested structured metadata dictionary as a JSON file
if all_timesteps_data:
    json_path = output_dir / "shell_cloud_height_stats.json"
    with open(json_path, "w") as jf:
        json.dump(all_timesteps_data, jf, indent=4)
    print(f"✅ Successfully saved nested object tracking metadata to:\n   {json_path}")
else:
    print("\n⚠️ No object tracking statistics were collected across the simulation.")
    

#closing files
for filename, file_obj in open_files.items():
    file_obj.close()

#Computation complete
print("\nAll computation and exporting complete (Program is safe to close)")