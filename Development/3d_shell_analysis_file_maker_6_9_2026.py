import sys
sys.path.insert(0, "/mnt/stor-pool-01/users/2821011/.local/lib/python3.9/site-packages")
sys.path.insert(0, "/usr/local/lib64/python3.9/site-packages")
import os
import math
import numpy as np
import xarray as xr
import scipy.ndimage
from pathlib import Path
import netCDF4 as nc
import cc3d
import gc

#constants
negative_w_threshold = -0.25
grid_distance = 200
ql_dilation = 1

#Output
output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir.mkdir(parents=True, exist_ok=True)

print("Imports and constants complete")
#Open datasets
ds_ql = xr.open_dataset("/mnt/stor-pool-01/projects/heus/EUREC4A_Eulerian/Feb_1st_12day_cdnc70_nudge/ql.nc", decode_times=False,chunks={'time': 1})
ds_w = xr.open_dataset("/mnt/stor-pool-01/projects/heus/EUREC4A_Eulerian/Feb_1st_12day_cdnc70_nudge/w.nc", decode_times=False,chunks={'time': 1})
ds_w = ds_w.rename({'zh':'z'}).interp(z=ds_ql.z)

#find index meter distances
grid_distance = float(ds_ql.x[1] - ds_ql.x[0]) #x/y index distance (m)
z_dist = float(ds_ql.z[1] - ds_ql.z[0]) #z index distance (m)



print("Dataset opening complete")
#crop to 25% width of x and y
x_cutoff = int(len(ds_ql.x) * 0.25)
y_cutoff = int(len(ds_ql.y) * 0.25)

ds_ql = ds_ql.isel(x=slice(0, x_cutoff), y=slice(0, y_cutoff))
ds_w = ds_w.isel(x=slice(0, x_cutoff), y=slice(0, y_cutoff))
num_times = int(ds_ql.time.size)
nz, ny, nx = ds_ql.ql.shape[1:]
coords = ds_ql.coords
#Export vars
#u1 means uint8 and f4 means float32
export_registry = {
    "ql_mask.nc": ("ql_mask", "u1"),
    "w_mask.nc": ("w_mask", "u1"),
    "shell_mask.nc": ("shell_mask", "u1"),
    "shell_w.nc": ("w", "f4"),
    "shell_distance.nc": ("distance", "f4"),
    "shell_distance_vert.nc": ("distance", "f4"),
    "shell_distance_horz.nc": ("distance", "f4"),
    "true_shell_distance.nc": ("distance", "f4"),
    "true_shell_distance_vert.nc": ("distance", "f4"),
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
        
        t_v[:] = ds_ql.time.values
        z_v[:] = ds_ql.z.values
        y_v[:] = ds_ql.y.values
        x_v[:] = ds_ql.x.values
        
        f.createVariable(var_name, data_type, ("time", "z", "y", "x"), zlib=True, complevel=4, fill_value=False)
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
    print(f"\n--- Processing Timestep {t}/{num_times - 1} ---")
    # Get this time slice of ql and w
    ql_raw = (ds_ql.ql.isel(time=t).fillna(0) > 0).values.astype(bool)
    w_slice = (ds_w.w.isel(time=t).fillna(0) < negative_w_threshold).values.astype(bool)

    ql_export = np.where(ql_raw, 1, 0).astype(np.uint8)
    w_export = np.where(w_slice, 1, 0).astype(np.uint8)

    open_files["ql_mask.nc"].variables["ql_mask"][t, :, :, :] = ql_export
    open_files["w_mask.nc"].variables["w_mask"][t, :, :, :] = w_export

    open_files["ql_mask.nc"].sync()
    open_files["w_mask.nc"].sync()

    # Initialize temporary 3D spatial trackers for ONLY this single timestep
    local_outline_mask = np.zeros_like(ql_raw, dtype=np.uint8)
    local_shell_w = np.full_like(ql_raw, np.nan, dtype=np.float32)
    
    local_shell_dist = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_shell_dist_vert = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_shell_dist_horz = np.full_like(ql_raw, np.nan, dtype=np.float32)
    
    local_true_dist = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_true_dist_vert = np.full_like(ql_raw, np.nan, dtype=np.float32)
    local_true_dist_horz = np.full_like(ql_raw, np.nan, dtype=np.float32)

    #Step 1 - Getting the shell
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
        print(f"No objects found in timestep {t}. Skipping calculations.")
        continue

    #-get w regions that intersect with ql
    matching_labels = set(labels[current])
    matching_labels.discard(0)

    if not matching_labels:
        continue

    #-select w regions that connect to ql
    converged_mask = np.isin(labels, list(matching_labels))

    #-create outline and apply it to the mask
    sub_outline = converged_mask & ~ql_raw
    local_outline_mask = sub_outline.astype(np.uint8)

    w_slice_physical = ds_w.w.isel(time=t).values
    local_shell_w = np.where(sub_outline, w_slice_physical, np.nan)
    print(f"Shell constructed in timestep {t}.")

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

    for idx, cloud_id in enumerate(active_cloud_ids):
        parent_cloud = (ql_labels == cloud_id)
        cloud_z_pts, cloud_y_pts, cloud_x_pts = np.where(parent_cloud)
        if len(cloud_x_pts) == 0:
            continue

        cloud_mask = (shell_parent_ids == cloud_id)
        shell_z_pts, shell_y_pts, shell_x_pts = np.where(cloud_mask)
        if len(shell_x_pts) == 0:
            continue

        sz = shell_z_pts[:, np.newaxis]
        sy = shell_y_pts[:, np.newaxis]
        sx = shell_x_pts[:, np.newaxis]

        cz = cloud_z_pts[np.newaxis, :]
        cy = cloud_y_pts[np.newaxis, :]
        cx = cloud_x_pts[np.newaxis, :]

        dlt_z = sz - cz
        dlt_y = sy - cy
        dlt_x = sx - cx

        #wrapping
        dlt_y = dlt_y - ny * np.round(dlt_y / ny)
        dlt_x = dlt_x - nx * np.round(dlt_x / nx)

        dlt_z_m_sq = (dlt_z * z_dist) ** 2
        dlt_y_m_sq = (dlt_y * grid_distance) ** 2
        dlt_x_m_sq = (dlt_x * grid_distance) ** 2

        dist_matrix_sq = dlt_x_m_sq + dlt_y_m_sq + dlt_z_m_sq

        #finds the closest cloud to use for distance
        min_idx = np.argmin(dist_matrix_sq, axis=1) 
        row_indices = np.arange(len(shell_x_pts))

        local_true_dist[shell_z_pts, shell_y_pts, shell_x_pts] = np.sqrt(dist_matrix_sq[row_indices, min_idx])
        
        local_true_dist_vert[shell_z_pts, shell_y_pts, shell_x_pts] = np.abs(dlt_z[row_indices, min_idx]) * z_dist
        
        local_true_dist_horz[shell_z_pts, shell_y_pts, shell_x_pts] = np.sqrt(
            dlt_x_m_sq[row_indices, min_idx] + dlt_y_m_sq[row_indices, min_idx]
        )

        del parent_cloud, cloud_z_pts, cloud_y_pts, cloud_x_pts, cloud_mask, shell_z_pts, shell_y_pts, shell_x_pts
        del sz, sy, sx, cz, cy, cx, dlt_z, dlt_y, dlt_x, dlt_z_m_sq, dlt_y_m_sq, dlt_x_m_sq, dist_matrix_sq, min_idx, row_indices
        
        if idx % 50 == 0:
            gc.collect()
    print(f"Connected ql distance created in timestep {t}.")

    #Step 3 - Nearest ql shell distance
    padded_ql_raw = np.pad(ql_raw, ((1, 1), (1, 1), (1, 1)), mode='wrap')
    eroded_padded = scipy.ndimage.binary_erosion(padded_ql_raw)
    eroded_ql = eroded_padded[1:-1, 1:-1, 1:-1]
    
    cloud_surface_mask = ql_raw ^ eroded_ql
    all_cloud_z, all_cloud_y, all_cloud_x = np.where(cloud_surface_mask)
    shell_z_pts, shell_y_pts, shell_x_pts = np.where(sub_outline)

    num_shell_pts = len(shell_x_pts)

    if len(all_cloud_x) > 0 and len(shell_x_pts) > 0:

        cz = all_cloud_z[np.newaxis, :]
        cy = all_cloud_y[np.newaxis, :]
        cx = all_cloud_x[np.newaxis, :]

        chunk_size = 15000
        for start_idx in range(0, num_shell_pts, chunk_size):
            end_idx = min(start_idx + chunk_size, num_shell_pts)

            chunk_z = shell_z_pts[start_idx:end_idx]
            chunk_y = shell_y_pts[start_idx:end_idx]
            chunk_x = shell_x_pts[start_idx:end_idx]

            sz = chunk_z[:, np.newaxis]
            sy = chunk_y[:, np.newaxis]
            sx = chunk_x[:, np.newaxis]

            dlt_z = sz - cz
            dlt_y = sy - cy
            dlt_x = sx - cx

            dlt_y = dlt_y - ny * np.round(dlt_y / ny)
            dlt_x = dlt_x - nx * np.round(dlt_x / nx)

            dlt_z_m_sq = (dlt_z * z_dist) ** 2
            dlt_y_m_sq = (dlt_y * grid_distance) ** 2
            dlt_x_m_sq = (dlt_x * grid_distance) ** 2
            
            dist_matrix_sq = dlt_x_m_sq + dlt_y_m_sq + dlt_z_m_sq

            min_idx = np.argmin(dist_matrix_sq, axis=1)
            row_indices = np.arange(len(chunk_x))

            local_shell_dist[chunk_z, chunk_y, chunk_x] = np.sqrt(dist_matrix_sq[row_indices, min_idx])
            
            local_shell_dist_vert[chunk_z, chunk_y, chunk_x] = np.abs(dlt_z[row_indices, min_idx]) * z_dist

            local_shell_dist_horz[chunk_z, chunk_y, chunk_x] = np.sqrt(
                dlt_x_m_sq[row_indices, min_idx] + dlt_y_m_sq[row_indices, min_idx]
            )
            del sz, sy, sx, dlt_z, dlt_y, dlt_x, dlt_z_m_sq, dlt_y_m_sq, dlt_x_m_sq, dist_matrix_sq, min_idx, row_indices
        del cz, cy, cx

    del all_cloud_z, all_cloud_y, all_cloud_x, shell_z_pts, shell_y_pts, shell_x_pts, cloud_surface_mask, eroded_padded, eroded_ql
    gc.collect()
    print(f"Nearest ql distance created in timestep {t}.")
    #Computation complete
    print("Computation complete")

    #Exporting
    print(f"Exporting timestep {t} datasets...")
    open_files["shell_mask.nc"].variables["shell_mask"][t, :, :, :] = local_outline_mask
    open_files["shell_w.nc"].variables["w"][t, :, :, :] = local_shell_w
    open_files["shell_distance.nc"].variables["distance"][t, :, :, :] = local_shell_dist
    open_files["shell_distance_vert.nc"].variables["distance"][t, :, :, :] = local_shell_dist_vert
    open_files["shell_distance_horz.nc"].variables["distance"][t, :, :, :] = local_shell_dist_horz
    open_files["true_shell_distance.nc"].variables["distance"][t, :, :, :] = local_true_dist
    open_files["true_shell_distance_vert.nc"].variables["distance"][t, :, :, :] = local_true_dist_vert
    open_files["true_shell_distance_horz.nc"].variables["distance"][t, :, :, :] = local_true_dist_horz

#closing files
for filename, file_obj in open_files.items():
    file_obj.close()

#Computation complete
print("\nAll computation and exporting complete (Program is safe to close)")
