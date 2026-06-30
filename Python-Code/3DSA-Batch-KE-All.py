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

# Input
input_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
source_input_dir = Path("/mnt/stor-pool-01/projects/heus/EUREC4A_Eulerian/Feb_1st_12day_cdnc70_nudge/")

#Output
output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir.mkdir(parents=True, exist_ok=True)

#Check that files exist
path_ds_ql_mask = Path(input_dir / "ql_mask.nc")
if not path_ds_ql_mask.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_ql_mask}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_w = Path(source_input_dir / "w.nc")
if not path_ds_w.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_w}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_vpg = Path(input_dir / "vpg.nc")
if not path_ds_vpg.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_vpg}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_b = Path(input_dir / "b.nc")
if not path_ds_b.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_b}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_vpg_b = Path(input_dir / "vpg_b.nc")
if not path_ds_vpg_b.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_vpg_b}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_vpg_dn = Path(input_dir / "vpg_dn.nc")
if not path_ds_vpg_dn.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_vpg_dn}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_vpg_dl = Path(input_dir / "vpg_dl.nc")
if not path_ds_vpg_dl.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_vpg_dl}\nEnding program early.", file=sys.stderr)
    sys.exit(1)

#Loading datasets
ds_ql_mask = xr.open_dataset(path_ds_ql_mask, decode_times=False,chunks={'time': 1})
ds_w = xr.open_dataset(path_ds_w, decode_times=False,chunks={'time': 1})
ds_w = ds_w.rename({'zh':'z'}).interp(z=ds_ql_mask.z)
ds_vpg = xr.open_dataset(path_ds_vpg, decode_times=False, chunks={'time': 1})
ds_b = xr.open_dataset(path_ds_b, decode_times=False, chunks={'time': 1})
ds_vpg_b = xr.open_dataset(path_ds_vpg_b, decode_times=False, chunks={'time': 1})
ds_vpg_dn = xr.open_dataset(path_ds_vpg_dn, decode_times=False, chunks={'time': 1})
ds_vpg_dl = xr.open_dataset(path_ds_vpg_dl, decode_times=False, chunks={'time': 1})


num_times = int(ds_ql_mask.time.size)
nz, ny, nx = ds_ql_mask.ql_mask.shape[1:]
z_coordinates = ds_ql_mask.z.values

#Export vars
#u1 means uint8 and f4 means float32
export_registry = {
    "ke_w.nc": ("ke_w", "f4"),
    "ke_vpg.nc": ("ke_vpg", "f4"),
    "ke_b.nc": ("ke_b", "f4"),
    "ke_vpg_b.nc": ("ke_vpg_b", "f4"),
    "ke_vpg_dn.nc": ("ke_vpg_dn", "f4"),
    "ke_vpg_dl.nc": ("ke_vpg_dl", "f4"),
    "ke_b_eff.nc": ("ke_b_eff", "f4")
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

#KE computation function
def compute_bounded_kinetic_energy(da, z_base, template_ds):
    """
    Computes cumulative kinetic energy from a specific lower bound 'z_base' 
    to the top of the domain, and realigns the output to match a template dataset.
    
    Parameters:
    -----------
    da : xr.DataArray
        The 3D spatial input field (e.g., buoyancy, VPG) for a single timestep.
    z_base : float
        The altitude value of the lower integration bound (e.g., cloud base).
    template_ds : xr.Dataset or xr.DataArray
        The full-grid template used to reindex the dimensions back to the complete domain.
    """
    # 1. Slice the variable from the lower bound to the top of the atmosphere
    da_bounded = da.sel(z=slice(z_base, None))
    
    # 2. Cumulative integration along the vertical axis
    ke_bounded = da_bounded.cumulative_integrate("z")
    
    # 3. Shape back into the regular full-grid dimension, filling below z_base with 0.0
    ke_full = ke_bounded.reindex_like(template_ds, fill_value=0.0)
    
    return ke_full


for t in range(num_times):
    print(f"\n--- (KE) Processing Timestep {t}/{num_times - 1} ---")

    #Getting zi (arbitrary z) at cloud base
    ql_mask_slice = ds_ql_mask.isel(time=t)
    has_nonzero_per_layer = (ql_mask_slice.ql_mask > 0).any(dim=["x", "y"])
    smallest_z_idx = has_nonzero_per_layer.argmax(dim="z").compute().item()
    arb_z =ql_mask_slice.z.isel(z=smallest_z_idx).compute().item()
    print(f"(KE) zi = {arb_z} in Timestep {t}/{num_times - 1}")

    #Perform slicing
    w_slice = ds_w.w.isel(time=t)
    vpg_slice = ds_vpg.vpg.isel(time=t)
    b_slice = ds_b.b.isel(time=t)
    vpg_b_slice = ds_vpg_b.vpg_b.isel(time=t)
    b_eff_slice = vpg_b_slice + b_slice #calculated
    vpg_dn_slice = ds_vpg_dn.vpg_dn.isel(time=t)
    vpg_dl_slice = ds_vpg_dl.vpg_dl.isel(time=t)

    #Convert to KE
    reindex_template = ds_b.isel(time=t)

    ke_w = (w_slice ** 2) / 2
    ke_vpg = compute_bounded_kinetic_energy(vpg_slice, arb_z, template_ds=reindex_template)
    ke_b = compute_bounded_kinetic_energy(b_slice, arb_z, template_ds=reindex_template)
    ke_vpg_b = compute_bounded_kinetic_energy(vpg_b_slice, arb_z, template_ds=reindex_template)
    ke_b_eff = compute_bounded_kinetic_energy(b_eff_slice, arb_z, template_ds=reindex_template)
    ke_vpg_dn = compute_bounded_kinetic_energy(vpg_dn_slice, arb_z, template_ds=reindex_template)
    ke_vpg_dl = compute_bounded_kinetic_energy(vpg_dl_slice, arb_z, template_ds=reindex_template)

    print(f"KE computed in timestep {t}.")

    # Commit changes cleanly to file storage system
    open_files["ke_w.nc"].variables["ke_w"][t, :, :, :] = ke_w.compute().values
    open_files["ke_vpg.nc"].variables["ke_vpg"][t, :, :, :] = ke_vpg.compute().values
    open_files["ke_b.nc"].variables["ke_b"][t, :, :, :] = ke_b.compute().values
    open_files["ke_vpg_b.nc"].variables["ke_vpg_b"][t, :, :, :] = ke_vpg_b.compute().values
    open_files["ke_b_eff.nc"].variables["ke_b_eff"][t, :, :, :] = ke_b_eff.compute().values
    open_files["ke_vpg_dn.nc"].variables["ke_vpg_dn"][t, :, :, :] = ke_vpg_dn.compute().values
    open_files["ke_vpg_dl.nc"].variables["ke_vpg_dl"][t, :, :, :] = ke_vpg_dl.compute().values

    for open_file in open_files.values():
        open_file.sync()
    
    # Memory optimization collection sweep
    gc.collect()


#closing files
for filename, file_obj in open_files.items():
    file_obj.close()

#Computation complete
print("\nAll computation and exporting complete (Program is safe to close)")