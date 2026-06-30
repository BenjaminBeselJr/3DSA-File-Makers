import sys
import gc
import numpy as np
import xarray as xr
import netCDF4 as nc
from pathlib import Path

# --- Configurations ---
input_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
source_input_dir = Path("/mnt/stor-pool-01/projects/heus/EUREC4A_Eulerian/Feb_1st_12day_cdnc70_nudge/")
output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")

path_ds_ql_mask = input_dir / "ql_mask.nc"
path_ds_w = source_input_dir / "w.nc"

if not path_ds_ql_mask.is_file() or not path_ds_w.is_file():
    print("❌ ERROR: Input files missing.", file=sys.stderr)
    sys.exit(1)

# Loading tracking metadata and w source
ds_ql_mask = xr.open_dataset(path_ds_ql_mask, decode_times=False, chunks={'time': 1})
ds_w = xr.open_dataset(path_ds_w, decode_times=False, chunks={'time': 1})
ds_w = ds_w.rename({'zh': 'z'}).interp(z=ds_ql_mask.z)

num_times = int(ds_ql_mask.time.size)
nz, ny, nx = ds_ql_mask.ql_mask.shape[1:]

# 1. Pre-allocate ONLY the ke_w.nc file
ke_w_file = output_dir / "ke_w.nc"
print(f"Pre-allocating fresh {ke_w_file.name} structure...")

with nc.Dataset(str(ke_w_file), "w", format="NETCDF4") as f:
    f.createDimension("time", num_times)
    f.createDimension("z", nz)
    f.createDimension("y", ny)
    f.createDimension("x", nx)
    
    f.createVariable("time", "f8", ("time",))[:] = ds_ql_mask.time.values
    f.createVariable("z", "f4", ("z",))[:] = ds_ql_mask.z.values
    f.createVariable("y", "f4", ("y",))[:] = ds_ql_mask.y.values
    f.createVariable("x", "f4", ("x",))[:] = ds_ql_mask.x.values
    
    f.createVariable("ke_w", "f4", ("time", "z", "y", "x"), zlib=True, complevel=4, chunksizes=(1, nz, ny, nx), fill_value=False)

# 2. Open in append mode to fill values
nc_file = nc.Dataset(str(ke_w_file), "a")

print("Populating corrected ke_w fields...")
for t in range(num_times):
    print(f"Processing Timestep {t}/{num_times - 1}...")
    
    # Direct algebraic extraction
    w_slice = ds_w.w.isel(time=t).values
    ke_w_slice = (w_slice ** 2) / 2.0
    
    # Save directly to disk array
    nc_file.variables["ke_w"][t, :, :, :] = ke_w_slice.astype(np.float32)
    nc_file.sync()
    gc.collect()

nc_file.close()
print("\n✅ Fast-fix complete! ke_w.nc has been successfully updated with w^2/2.")