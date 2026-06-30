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

# Input
input_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")

#Output
output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir.mkdir(parents=True, exist_ok=True)

#Check that files exist
path_ds_ql_mask = Path(input_dir / "ql_mask.nc")
if not path_ds_ql_mask.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_ql_mask}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_w_mask = Path(input_dir / "w_mask.nc")
if not path_ds_w_mask.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_w_mask}\nEnding program early.", file=sys.stderr)
    sys.exit(1)

#Loading datasets
ds_ql_mask = xr.open_dataset(path_ds_ql_mask, decode_times=False)
ds_w_mask = xr.open_dataset(path_ds_w_mask, decode_times=False)


num_times = int(ds_ql_mask.time.size)
nz, ny, nx = ds_ql_mask.ql_mask.shape[1:]
z_coordinates = ds_ql_mask.z.values

#Export vars
#u1 means uint8 and f4 means float32
export_registry = {
    # 3D Grid Vars (time, z, y, x)
    "shell_origin.nc": ("shell_origin", "f4"),
    "shell_termination.nc": ("shell_termination", "f4"),
    "cloud_base.nc": ("cloud_base", "f4"),
    "cloud_top.nc": ("cloud_top", "f4"),
    "shell_labels.nc": ("shell_labels", "u4"),
    "cloud_labels.nc": ("cloud_labels", "u4"),
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
        
        t_v[:] = ds_ql_mask.time.values
        z_v[:] = ds_ql_mask.z.values
        y_v[:] = ds_ql_mask.y.values
        x_v[:] = ds_ql_mask.x.values
        
        f.createVariable(var_name, data_type, ("time", "z", "y", "x"), zlib=True, complevel=4, chunksizes=(1, nz, ny, nx), fill_value=False)
#open blank arrays created earlier
open_files = {fname: nc.Dataset(str(output_dir / fname), "a") for fname in export_registry}

#-Creating array for expansion
expansion = np.zeros((3,3,3), dtype=bool)
expansion[1, 1, :] = True  # X axis
expansion[1, :, 1] = True  # Y axis
expansion[:, 1, 1] = True  # Z axis

all_timesteps_data = []
for t in range(num_times):
    print(f"\n--- (Cloud/Shell Labels and Height Stats) Processing Timestep {t}/{num_times - 1} ---")
    #Slices
    ql_raw = ds_ql_mask.ql_mask.isel(time=t).values.astype(bool)
    w_slice = ds_w_mask.w_mask.isel(time=t).values.astype(bool)

    local_shell_origin = np.full((nz,ny,nx), np.nan, dtype=np.float32)
    local_shell_termination = np.full((nz,ny,nx), np.nan, dtype=np.float32)

    local_shell_cloud_bottom = np.full((nz,ny,nx), np.nan, dtype=np.float32)
    local_shell_cloud_top = np.full((nz,ny,nx), np.nan, dtype=np.float32)

    local_congestus_mask = np.full((nz,ny,nx), False, dtype=bool)
    local_deep_mask = np.full((nz,ny,nx), False, dtype=bool)

    #Step 1 - Recreating the shell labels
    # Dilated ql serving working space for shell expansion and intersection detection
    padded_ql = np.pad(ql_raw, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0) #padded z with 0's to prevent 
    padded_ql = np.pad(padded_ql, ((0, 0), (1, 1), (1, 1)), mode='wrap')
    padded_current = scipy.ndimage.binary_dilation(padded_ql, structure=expansion)
    current = padded_current[1:-1, 1:-1, 1:-1]

    padded_w = np.pad(w_slice, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    padded_labels = cc3d.connected_components(padded_w, connectivity=6, periodic_boundary=True) #labels every region in w
    labels = padded_labels[1:-1, :, :]

    num_features = np.max(labels)
    if num_features == 0:
        print(f"No objects found in timestep {t}. Skipping.")
        continue

    #-get w regions that intersect with ql
    matching_labels = set(labels[current])
    matching_labels.discard(0)

    matching_labels_ql = set(labels[ql_raw])
    matching_labels_ql.discard(0)

    shell_origin_dict       = {}
    shell_termination_dict  = {}
    cloud_bottom_dict       = {}
    cloud_top_dict          = {}

    if not matching_labels:
        print(f"No valid convective features connected to cloud in timestep {t}. Skipping.")
        continue

    
    valid_shell_labels = np.where(current, labels, 0)
    valid_ql_labels = np.where(ql_raw, labels, 0)
    slices = scipy.ndimage.find_objects(valid_shell_labels)
    slices_ql = scipy.ndimage.find_objects(valid_ql_labels)

    #Stats calc
    for obj_id in matching_labels:
        obj_slice = slices[obj_id - 1] if obj_id - 1 < len(slices) else None

        if obj_slice is not None:
            z_slice = obj_slice[0]
            min_z_phys = z_coordinates[z_slice.start]
            max_z_phys = z_coordinates[z_slice.stop - 1]

            box_mask = valid_shell_labels[obj_slice] == obj_id

            local_shell_origin[obj_slice][box_mask] = min_z_phys
            local_shell_termination[obj_slice][box_mask] = max_z_phys

            shell_origin_dict[obj_id - 1] = min_z_phys
            shell_termination_dict[obj_id - 1] = max_z_phys

            #cloud min and max
            ql_obj_slice = slices_ql[obj_id - 1] if obj_id - 1 < len(slices_ql) else None
            if ql_obj_slice is not None:
                z_ql_slice = ql_obj_slice[0]
                min_z_ql = z_coordinates[z_ql_slice.start]
                max_z_ql = z_coordinates[z_ql_slice.stop - 1]
                
                if(max_z_ql > 5000):
                    local_deep_mask[obj_slice][box_mask] = True
                elif(max_z_ql > 2000):
                    local_congestus_mask[obj_slice][box_mask] = True

                local_shell_cloud_bottom[obj_slice][box_mask] = min_z_ql
                local_shell_cloud_top[obj_slice][box_mask] = max_z_ql

                cloud_bottom_dict[obj_id - 1]  = min_z_ql
                cloud_top_dict[obj_id - 1]     = max_z_ql

    for obj_id in shell_origin_dict.keys():
        
        # Build a single flat row matching database formatting
        row = {
            "time_idx": t,
            "object_id": obj_id,
            "shell_origin": shell_origin_dict.get(obj_id, np.nan),
            "shell_termination": shell_termination_dict.get(obj_id, np.nan),
            "cloud_bottom": cloud_bottom_dict.get(obj_id, np.nan),
            "cloud_top": cloud_top_dict.get(obj_id, np.nan)
        }
        
        # Append this flat dictionary to our global tracker list
        all_timesteps_data.append(row)

    print(f"Labels and Stats computed in timestep {t}.")

    # Commit changes cleanly to file storage system
    open_files["shell_origin.nc"].variables["shell_origin"][t, :, :, :] = local_shell_origin
    open_files["shell_termination.nc"].variables["shell_termination"][t, :, :, :] = local_shell_termination
    open_files["cloud_base.nc"].variables["cloud_base"][t, :, :, :] = local_shell_cloud_bottom
    open_files["cloud_top.nc"].variables["cloud_top"][t, :, :, :] = local_shell_cloud_top
    open_files["shell_labels.nc"].variables["shell_labels"][t, :, :, :] = valid_shell_labels
    open_files["cloud_labels.nc"].variables["cloud_labels"][t, :, :, :] = valid_ql_labels
    open_files["congestus_mask.nc"].variables["congestus_mask"][t, :, :, :] = local_congestus_mask
    open_files["deep_mask.nc"].variables["deep_mask"][t, :, :, :] = local_deep_mask

    for open_file in open_files.values():
        open_file.sync()

    # Memory optimization collection sweep
    gc.collect()


#Export lists
if all_timesteps_data:
    print("\nConverting object-tracking metadata to a Pandas DataFrame...")
    df_tracking = pd.DataFrame(all_timesteps_data)
    
    # Save as a clean, standard CSV file
    csv_path = output_dir / "shell_cloud_height_stats.csv"
    df_tracking.to_csv(csv_path, index=False)
    print(f"✅ Successfully saved per-object metrics to:\n   {csv_path}")
else:
    print("\n⚠️ No object tracking statistics were collected across the simulation.")
    

#closing files
for filename, file_obj in open_files.items():
    file_obj.close()

#Computation complete
print("\nAll computation and exporting complete (Program is safe to close)")