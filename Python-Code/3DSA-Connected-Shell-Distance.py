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
    "true_shell_distance.nc": ("distance", "f4"),
    "true_shell_distance_vert.nc": ("distance", "f4"),
    "true_relative_shell_altitude.nc": ("relative_altitude", "f4"),
    "true_shell_distance_horz.nc": ("distance", "f4"),
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
    local_true_dist = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_true_dist_vert = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_true_shell_relative_alt = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_true_dist_horz = np.full_like(ql_raw, np.nan, dtype=np.float32)


    #Step 2 - Getting shell distances for connected ql
    #-finding ql labeled regions
    
    padded_ql_raw = np.pad(ql_raw, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    temp_ql_labels = cc3d.connected_components(padded_ql_raw, connectivity=6, periodic_boundary=True)
    ql_labels = temp_ql_labels[1:-1, :, :]
    initial_dilated_ql_labels = ql_labels.copy()
    padded_labels = np.pad(initial_dilated_ql_labels, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    padded_labels = np.pad(padded_labels, ((0, 0), (ql_dilation, ql_dilation), (ql_dilation, ql_dilation)), mode='wrap')
    for i_dil in range(ql_dilation):
        padded_labels = scipy.ndimage.grey_dilation(padded_labels, footprint=expansion)
    dilated_ql_labels = padded_labels[1:-1, ql_dilation:-ql_dilation, ql_dilation:-ql_dilation]

    shell_parent_ids = np.where(sub_outline, dilated_ql_labels, 0)

    while True:
        travel_mask = sub_outline & (shell_parent_ids == 0) #where dilation still needs to occur

        if not np.any(travel_mask):
            break
        
        #expand ql (padded on z)
        padded_shell_ids = np.pad(shell_parent_ids, ((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        padded_shell_ids = np.pad(padded_shell_ids, ((0, 0), (1, 1), (1, 1)), mode='wrap')
        padded_expanded = scipy.ndimage.grey_dilation(padded_shell_ids, footprint=expansion)
        ql_label_expanded = padded_expanded[1:-1, 1:-1, 1:-1]

        #apply the expansion
        shell_parent_ids[travel_mask] = ql_label_expanded[travel_mask] 

    active_cloud_ids = np.unique(shell_parent_ids)
    active_cloud_ids = active_cloud_ids[active_cloud_ids != 0] #remove 0's

    total_clouds = len(active_cloud_ids)
    print(f"Calculating spatial distances using optimized k-d Trees for {len(active_cloud_ids)} cloud objects...")

    # Start tracking elapsed time for the cloud calculation loop
    start_time = time.time()
    
    for idx, cloud_id in enumerate(active_cloud_ids):
        # 1. Isolate the parent cloud coordinates
        parent_cloud = (ql_labels == cloud_id)
        cloud_z_pts, cloud_y_pts, cloud_x_pts = np.where(parent_cloud)
        if len(cloud_x_pts) == 0:
            continue

        # 2. Isolate the corresponding shell coordinates
        cloud_mask = (shell_parent_ids == cloud_id)
        shell_z_pts, shell_y_pts, shell_x_pts = np.where(cloud_mask)
        if len(shell_x_pts) == 0:
            continue
        
        z_real = ds_ql_mask.z.values

        # 3. Convert physical grid metrics to scaled spatial coordinates
        cloud_coords = np.column_stack((
            z_real[cloud_z_pts],
            cloud_y_pts * grid_distance,
            cloud_x_pts * grid_distance    
        ))

        shell_coords = np.column_stack((
            z_real[shell_z_pts],
            shell_y_pts * grid_distance,
            shell_x_pts * grid_distance
        ))


        # 4. Build a spatial k-d Tree for the parent cloud points
        # To handle domain wrapping (periodic boundaries), we pass 'boxsize'
        box_limits = np.array([999999.0, ny * grid_distance, nx * grid_distance]) # inf on Z implies non-periodic vertically
        tree = cKDTree(cloud_coords, boxsize=box_limits)

        # 5. Query the tree for the nearest cloud pixel for all shell pixels simultaneously
        # 'distances' returns true Euclidean distance; 'indices' maps back to the closest cloud point
        distances, closest_cloud_indices = tree.query(shell_coords, k=1, workers=-1)

        valid_idx_mask = closest_cloud_indices < len(cloud_z_pts)

        # 6. Assign true Euclidean distance directly
        local_true_dist[shell_z_pts, shell_y_pts, shell_x_pts] = distances

        # 7. Extract components for vertical and horizontal distance tracking
        # Get the matching cloud coordinates that won the proximity check
        matched_cloud_z = cloud_z_pts[closest_cloud_indices[valid_idx_mask]]
        matched_cloud_y = cloud_y_pts[closest_cloud_indices[valid_idx_mask]]
        matched_cloud_x = cloud_x_pts[closest_cloud_indices[valid_idx_mask]]

        # Calculate directional differences
        dy = shell_y_pts - matched_cloud_y
        dx = shell_x_pts - matched_cloud_x

        # Apply periodic wrapping adjustments to horizontal differences manually
        dy = dy - ny * np.round(dy / ny)
        dx = dx - nx * np.round(dx / nx)

        # Store directional metrics
        local_true_dist_vert[shell_z_pts, shell_y_pts, shell_x_pts] = np.abs(z_real[shell_z_pts] - z_real[matched_cloud_z])
        local_true_shell_relative_alt[shell_z_pts, shell_y_pts, shell_x_pts] = z_real[shell_z_pts] - z_real[matched_cloud_z]
        local_true_dist_horz[shell_z_pts, shell_y_pts, shell_x_pts] = np.sqrt((dx * grid_distance)**2 + (dy * grid_distance)**2)

        # Memory Cleanup
        del parent_cloud, cloud_mask, cloud_coords, shell_coords, tree, distances, closest_cloud_indices
        if idx % 10 == 0:
            gc.collect()

        # =====================================================================
        # PROGRESS BAR & TIME TRACKING PRODUCTION
        # =====================================================================
        completed = idx + 1
        pct = (completed / total_clouds) * 100
        elapsed = time.time() - start_time
        
        # Build a visual terminal loading bar (20 characters wide)
        bar_length = 20
        filled_length = int(round(bar_length * completed / float(total_clouds)))
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Format strings cleanly to display hours, minutes, and seconds
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        
        # Print progress dynamically on a single refreshing line
        print(f"\rProgress: |{bar}| {pct:.1f}% ({completed}/{total_clouds} Clouds) | Elapsed: {elapsed_str}", end="", flush=True)
    print()
    print(f"Connected ql distance created in timestep {t}.")

    #Computation complete
    print("Computation complete")

    #Exporting
    print(f"Exporting timestep {t} datasets...")
    open_files["true_shell_distance.nc"].variables["distance"][t, :, :, :] = local_true_dist
    open_files["true_shell_distance_vert.nc"].variables["distance"][t, :, :, :] = local_true_dist_vert
    open_files["true_relative_shell_altitude.nc"].variables["relative_altitude"][t, :, :, :] = local_true_shell_relative_alt
    open_files["true_shell_distance_horz.nc"].variables["distance"][t, :, :, :] = local_true_dist_horz

    for open_file in open_files.values():
        open_file.sync()
    gc.collect()

#closing files
for filename, file_obj in open_files.items():
    file_obj.close()

#Computation complete
print("\nAll computation and exporting complete (Program is safe to close)")