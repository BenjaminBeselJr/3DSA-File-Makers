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

negative_w_threshold = -0.25

# Input
input_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")

#Output
output_directory = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_directory.mkdir(parents=True, exist_ok=True)

#Check that files exist
path_ds_ql_mask = Path(input_dir / "ql_mask.nc")
if not path_ds_ql_mask.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_ql_mask}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_w_mask = Path(input_dir / "w_mask.nc")
if not path_ds_w_mask.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_w_mask}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_shell_mask = Path(input_dir / "shell_mask.nc")
if not path_ds_shell_mask.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_shell_mask}\nEnding program early.", file=sys.stderr)
    sys.exit(1)

#Loading datasets
ds_ql_mask = xr.open_dataset(path_ds_ql_mask, decode_times=False)
ds_w_mask = xr.open_dataset(path_ds_w_mask, decode_times=False)
ds_shell_mask = xr.open_dataset(path_ds_shell_mask, decode_times=False)

#Making the shell
num_times = int(ds_shell_mask.time.size)
nz, ny, nx = ds_shell_mask.shell_mask.shape[1:]
z_coordinates = ds_shell_mask.z.values

# Pre-allocating blank file template on disk using NetCDF4 API to save system memory
output_filename = output_directory / "shell_depth.nc"
print("Pre-allocating output file structure on disk...")
with nc.Dataset(str(output_filename), "w", format="NETCDF4") as f:
    f.createDimension("time", num_times)
    f.createDimension("z", nz)
    f.createDimension("y", ny)
    f.createDimension("x", nx)
    
    t_v = f.createVariable("time", "f8", ("time",))
    z_v = f.createVariable("z", "f4", ("z",))
    y_v = f.createVariable("y", "f4", ("y",))
    x_v = f.createVariable("x", "f4", ("x",))
    
    t_v[:] = ds_shell_mask.time.values
    z_v[:] = ds_shell_mask.z.values
    y_v[:] = ds_shell_mask.y.values
    x_v[:] = ds_shell_mask.x.values
    
    depth_var = f.createVariable("shell_depth", "f4", ("time", "z", "y", "x"), zlib=True, complevel=4, chunksizes=(1, nz, ny, nx), fill_value=np.nan)
    depth_var.units = "meters"
    depth_var.long_name = "Total vertical thickness of individual shell"

# Open the new structure to start append streaming
nc_out = nc.Dataset(str(output_filename), "a")

#-Creating array for expansion
expansion = np.zeros((3,3,3), dtype=bool)
expansion[1, 1, :] = True  # X axis
expansion[1, :, 1] = True  # Y axis
expansion[:, 1, 1] = True  # Z axis

grid_shape = ds_shell_mask.shell_mask.shape


for t in range(num_times):
    print(f"\n--- Processing Timestep {t}/{num_times - 1} ---")
    #Slices
    ql_raw = ds_ql_mask.ql_mask.isel(time=t).values.astype(bool)
    w_slice = ds_w_mask.w_mask.isel(time=t).values.astype(bool)
    outline_mask = ds_shell_mask.shell_mask.isel(time=t).values.astype(bool)

    local_shell_depth = np.full((nz,ny,nx), np.nan, dtype=np.float32)

    #Step 1 - Recreating the shell lables
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
        print(f"No objects found in timestep {t}. Writing empty layer matrix.")
        nc_out.variables["shell_depth"][t, :, :, :] = local_shell_depth
        nc_out.sync()
        continue

    #-get w regions that intersect with ql
    matching_labels = set(labels[current])
    matching_labels.discard(0)

    if not matching_labels:
        print(f"No valid convective features connected to cloud in timestep {t}.")
        nc_out.variables["shell_depth"][t, :, :, :] = local_shell_depth
        nc_out.sync()
        continue
    
    #Shell Depth
    valid_shell_labels = np.where(outline_mask, labels, 0)
    slices = scipy.ndimage.find_objects(valid_shell_labels)

    for obj_id in matching_labels:
        obj_slice = slices[obj_id - 1] if obj_id - 1 < len(slices) else None

        if obj_slice is not None:
            z_slice = obj_slice[0]
            min_z_phys = z_coordinates[z_slice.start]
            max_z_phys = z_coordinates[z_slice.stop - 1]
            obj_depth = max_z_phys - min_z_phys

            box_mask = valid_shell_labels[obj_slice] == obj_id

            local_shell_depth[obj_slice][box_mask] = obj_depth

    print(f"Shell depths constructed in timestep {t}.")

    # Commit changes cleanly to file storage system
    nc_out.variables["shell_depth"][t, :, :, :] = local_shell_depth
    nc_out.sync()
    
    # Memory optimization collection sweep
    del ql_raw, w_slice, outline_mask, local_shell_depth, padded_ql, padded_current, current, padded_w, padded_labels, labels, valid_shell_labels, slices
    gc.collect()


nc_out.close()

print(f"Successfully exported dataset to {output_filename}")