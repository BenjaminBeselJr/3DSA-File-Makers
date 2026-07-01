import os
import math
import numpy as np
import xarray as xr
import scipy.ndimage
from pathlib import Path
import netCDF4 as nc
import cc3d
import gc
from scipy.spatial import cKDTree
import time
import sys

#constants
negative_w_threshold = -0.25
ql_dilation = 1

#Input
source_input_dir = Path("/mnt/stor-pool-01/projects/heus/EUREC4A_Eulerian/Feb_1st_12day_cdnc70_nudge/")

#Output
output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir.mkdir(parents=True, exist_ok=True)

print("Imports and constants complete")
#Check that files exist
path_ds_ql = Path(source_input_dir / "ql.nc")
if not path_ds_ql.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_ql}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_w = Path(source_input_dir / "w.nc")
if not path_ds_w.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_w}\nEnding program early.", file=sys.stderr)
    sys.exit(1)

#Loading datasets
ds_ql = xr.open_dataset(path_ds_ql, decode_times=False,chunks={'time': 1})
ds_w = xr.open_dataset(path_ds_w, decode_times=False,chunks={'time': 1})
ds_w = ds_w.rename({'zh':'z'}).interp(z=ds_ql.z)

print("Dataset opening complete")
num_times = int(ds_ql.time.size)
nz, ny, nx = ds_ql.ql.shape[1:]
coords = ds_ql.coords
#Export vars
#u1 means uint8 and f4 means float32
export_registry = {
    "ql_mask.nc": ("ql_mask", "u1"),
    "w_mask.nc": ("w_mask", "u1"),
    "shell_mask.nc": ("shell_mask", "u1"),
    "shell_labels.nc": ("shell_labels", "u4"),
    "cloud_labels.nc": ("cloud_labels", "u4"),
    "shell_w.nc": ("w", "f4"),
}
#create blank datasets on memory
print("Pre-allocating NetCDF file structures on disk...")
open_files = {}
for filename, (var_name, data_type) in export_registry.items():
    file_path = str(output_dir / filename)
    f = nc.Dataset(file_path, "w", format="NETCDF4")
    open_files[filename] = f
    f.createDimension("time", num_times)
    f.createDimension("z", nz)
    f.createDimension("y", ny)
    f.createDimension("x", nx)
    
    t_v = f.createVariable("time", "f8", ("time",))
    z_v = f.createVariable("z", "f4", ("z",))
    y_v = f.createVariable("y", "f4", ("y",))
    x_v = f.createVariable("x", "f4", ("x",))
    
    t_v[:] = ds_ql.time.values
    z_v[:] = ds_ql.z.values
    y_v[:] = ds_ql.y.values
    x_v[:] = ds_ql.x.values
    
    f.createVariable(var_name, data_type, ("time", "z", "y", "x"), zlib=True, complevel=4, chunksizes=(1, nz, ny, nx), fill_value=False)

#Creating masks and shell w

#Prepare grid indicies
z_grid, y_grid, x_grid = np.indices((nz, ny, nx))

#-Creating array for expansion
expansion = np.zeros((3,3,3), dtype=bool)
expansion[1, 1, :] = True  # X axis
expansion[1, :, 1] = True  # Y axis
expansion[:, 1, 1] = True  # Z axis


for t in range(num_times):
#for t in [6]:
    print(f"\n--- Processing Timestep {t}/{num_times - 1} ---")
    # Get this time slice of ql and w
    ql_raw = (ds_ql.ql.isel(time=t).fillna(0) > 0).values.astype(bool)
    w_slice = (ds_w.w.isel(time=t).fillna(0) < negative_w_threshold).values.astype(bool)

    open_files["ql_mask.nc"].variables["ql_mask"][t, :, :, :] = np.where(ql_raw, 1, 0).astype(np.uint8)
    open_files["w_mask.nc"].variables["w_mask"][t, :, :, :] = np.where(w_slice, 1, 0).astype(np.uint8)

    # Initialize temporary 3D spatial trackers for ONLY this single timestep
    local_outline_mask = np.zeros_like(ql_raw, dtype=np.uint8)
    local_shell_labels = np.zeros_like(ql_raw, dtype=np.uint32)
    local_cloud_labels = np.zeros_like(ql_raw, dtype=np.uint32)
    local_shell_w = np.full_like(ql_raw, np.nan, dtype=np.float32)
    
    #Step 1 - Getting the shell
    # Dilated ql serving working space for shell expansion and intersection detection
    if np.any(ql_raw):
        padded_ql_core = np.pad(ql_raw, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        padded_ql_labels = cc3d.connected_components(padded_ql_core, connectivity=6, periodic_boundary=True)
        local_cloud_labels = padded_ql_labels[1:-1, :, :].astype(np.uint32)

    flooded_labels = local_cloud_labels.copy()

    if np.any(flooded_labels) and ql_dilation > 0:
        print(f" -> Pre-dilating clouds by {ql_dilation} step(s) to bridge gaps...")
        for _ in range(ql_dilation):
            padded_seed = np.pad(flooded_labels, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            padded_seed = np.pad(padded_seed, ((0, 0), (1, 1), (1, 1)), mode='wrap')
            padded_dilated_seed = scipy.ndimage.grey_dilation(padded_seed, footprint=expansion)
            flooded_labels = padded_dilated_seed[1:-1, 1:-1, 1:-1]

    if np.any(local_cloud_labels) and np.any(w_slice):
        print(" -> Flooding cloud labels into the w mask...")
        iteration = 0

        while True:
            # Pad for periodic boundaries on X/Y, constant on Z before dilating
            padded_flood = np.pad(flooded_labels, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            padded_flood = np.pad(padded_flood, ((0, 0), (1, 1), (1, 1)), mode='wrap')

            padded_dilated = scipy.ndimage.grey_dilation(padded_flood, footprint=expansion)
            dilated_step = padded_dilated[1:-1, 1:-1, 1:-1]

            # Masking condition
            grow_mask = w_slice & (flooded_labels == 0) & (dilated_step > 0)
            
            if not np.any(grow_mask):
                break
                
            flooded_labels[grow_mask] = dilated_step[grow_mask]
            iteration += 1

        print(f" -> Flooding finished after {iteration} iterations.")
    
    #old code
    """
    num_features = np.max(labels)
    if num_features == 0:
        print(f"No downdraft objects found in timestep {t}. Skipping calculations.")
    else:
        #-get w regions that intersect with ql
        matching_labels = set(labels[current])
        matching_labels.discard(0)

        if not matching_labels:
            print(f"No matching labels found in timestep {t}. Skipping calculations.")
            continue
        else:
            #-select w regions that connect to ql
            converged_mask = np.isin(labels, list(matching_labels))

            #-create outline and apply it to the mask
            sub_outline = converged_mask & ~ql_raw
            local_outline_mask = sub_outline.astype(np.uint8)

            w_slice_physical = ds_w.w.isel(time=t).values
            local_shell_w = np.where(sub_outline, w_slice_physical, np.nan)
            print(f"Shell constructed in timestep {t}.")

    """

    shell_domain = w_slice & ~ql_raw
    local_shell_labels = np.where(shell_domain, flooded_labels, 0).astype(np.uint32)
    local_outline_mask = np.where(local_shell_labels > 0, 1, 0).astype(np.uint8)

    #obtain shell w
    w_slice_physical = ds_w.w.isel(time=t).values
    local_shell_w = np.where(local_outline_mask > 0, w_slice_physical, np.nan)

    #Computation complete
    print("Computation complete")

    #Exporting
    print(f"Exporting timestep {t} datasets...")
    open_files["shell_mask.nc"].variables["shell_mask"][t, :, :, :] = local_outline_mask
    open_files["shell_labels.nc"].variables["shell_labels"][t, :, :, :] = local_shell_labels
    open_files["cloud_labels.nc"].variables["cloud_labels"][t, :, :, :] = local_cloud_labels
    open_files["shell_w.nc"].variables["w"][t, :, :, :] = local_shell_w

    for open_file in open_files.values():
        open_file.sync()
        
    gc.collect()

#closing files
for filename, file_obj in open_files.items():
    file_obj.close()

#Computation complete
print("\nAll computation and exporting complete (Program is safe to close)")