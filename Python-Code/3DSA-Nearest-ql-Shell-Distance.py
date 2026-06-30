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
ql_dilation = 1

#Input
input_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")

#Output
output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir.mkdir(parents=True, exist_ok=True)

print("Imports and constants complete")

#Open datasets

#Check that files exist
path_ds_ql_mask = Path(input_dir / "ql_mask.nc")
if not path_ds_ql_mask.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_ql_mask}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_shell_mask = Path(input_dir / "shell_mask.nc")
if not path_ds_shell_mask.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_shell_mask}\nEnding program early.", file=sys.stderr)
    sys.exit(1)

#Loading datasets
ds_ql_mask = xr.open_dataset(path_ds_ql_mask, decode_times=False)
ds_shell_mask = xr.open_dataset(path_ds_shell_mask, decode_times=False)

#find index meter distances
grid_distance = float(ds_ql_mask.x[1] - ds_ql_mask.x[0]) #x/y index distance (m)



print("Dataset opening complete")

num_times = int(ds_ql_mask.time.size)
nz, ny, nx = ds_ql_mask.ql_mask.shape[1:]
coords = ds_ql_mask.coords

#Export vars
#u1 means uint8 and f4 means float32
export_registry = {
    "shell_distance.nc": ("distance", "f4"),
    "shell_distance_vert.nc": ("distance", "f4"),
    "relative_shell_altitude.nc": ("relative_altitude", "f4"),
    "shell_distance_horz.nc": ("distance", "f4"),
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

#Creating shell mask and shell distances
#open blank arrays created earlier
open_files = {fname: nc.Dataset(str(output_dir / fname), "a") for fname in export_registry}

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
    ql_raw = ds_ql_mask.ql_mask.isel(time=t).values.astype(bool)
    sub_outline = ds_shell_mask.shell_mask.isel(time=t).values.astype(bool)

    # Initialize temporary 3D spatial trackers for ONLY this single timestep
    
    local_shell_dist = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_shell_dist_vert = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_shell_dist_horz = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_shell_relative_alt = np.full_like(ql_raw, np.nan, dtype=np.float32)

    #Step 3 - Nearest ql shell distance
    padded_ql_raw = np.pad(ql_raw, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    padded_ql_raw = np.pad(padded_ql_raw, ((0, 0), (1, 1), (1, 1)), mode='wrap')
    eroded_padded = scipy.ndimage.binary_erosion(padded_ql_raw)
    eroded_ql = eroded_padded[1:-1, 1:-1, 1:-1]
    
    cloud_surface_mask = ql_raw ^ eroded_ql
    all_cloud_z, all_cloud_y, all_cloud_x = np.where(cloud_surface_mask)
    shell_z_pts, shell_y_pts, shell_x_pts = np.where(sub_outline)

    num_shell_pts = len(shell_x_pts)

    if len(all_cloud_x) > 0 and len(shell_x_pts) > 0:
        # 1. Stack globally using (X, Y, Z) order to match box_limits
        z_real = ds_ql_mask.z.values
        global_cloud_coords = np.column_stack((
            z_real[all_cloud_z],
            all_cloud_y * grid_distance,
            all_cloud_x * grid_distance
        ))
        
        global_shell_coords = np.column_stack((
           z_real[shell_z_pts],            
            shell_y_pts * grid_distance,
            shell_x_pts * grid_distance
        ))

        # 2. Build the tree once for the entire cloud surface domain
        box_limits = np.array([999999.0, ny * grid_distance, nx * grid_distance])
        global_tree = cKDTree(global_cloud_coords, boxsize=box_limits)

        #3. Query all shell points instantly
        global_distances, closest_surface_indices = global_tree.query(global_shell_coords, k=1, workers=-1)

        valid_idx_mask = closest_surface_indices < len(all_cloud_z)

        # 4. Assign the true Euclidean distance
        local_shell_dist[shell_z_pts, shell_y_pts, shell_x_pts] = global_distances

        # 5. Extract components for directional distance tracking
        matched_surface_z = all_cloud_z[closest_surface_indices[valid_idx_mask]]
        matched_surface_y = all_cloud_y[closest_surface_indices[valid_idx_mask]]
        matched_surface_x = all_cloud_x[closest_surface_indices[valid_idx_mask]]

        # Calculate directional differences
        dz = shell_z_pts - matched_surface_z
        dy = shell_y_pts - matched_surface_y
        dx = shell_x_pts - matched_surface_x

        # Apply periodic wrapping adjustments horizontally
        dy = dy - ny * np.round(dy / ny)
        dx = dx - nx * np.round(dx / nx)

        # Store directional metrics
        local_shell_dist_vert[shell_z_pts, shell_y_pts, shell_x_pts] = np.abs(z_real[shell_z_pts] - z_real[matched_surface_z])
        local_shell_relative_alt[shell_z_pts, shell_y_pts, shell_x_pts] = z_real[shell_z_pts] - z_real[matched_surface_z]
        local_shell_dist_horz[shell_z_pts, shell_y_pts, shell_x_pts] = np.sqrt(
            (dx * grid_distance)**2 + (dy * grid_distance)**2
        )

        # Cleanup Tree Memory
        del global_cloud_coords, global_shell_coords, global_tree, global_distances, closest_surface_indices

    del all_cloud_z, all_cloud_y, all_cloud_x, shell_z_pts, shell_y_pts, shell_x_pts, cloud_surface_mask, eroded_padded, eroded_ql
    gc.collect()
    print(f"Nearest ql distance created in timestep {t}.")
    #Computation complete
    print("Computation complete")

    #Exporting
    print(f"Exporting timestep {t} datasets...")
    open_files["shell_distance.nc"].variables["distance"][t, :, :, :] = local_shell_dist
    open_files["shell_distance_vert.nc"].variables["distance"][t, :, :, :] = local_shell_dist_vert
    open_files["relative_shell_altitude.nc"].variables["relative_altitude"][t, :, :, :] = local_shell_relative_alt
    open_files["shell_distance_horz.nc"].variables["distance"][t, :, :, :] = local_shell_dist_horz
    for open_file in open_files.values():
        open_file.sync()
    gc.collect()

#closing files
for filename, file_obj in open_files.items():
    file_obj.close()

#Computation complete
print("\nAll computation and exporting complete (Program is safe to close)")