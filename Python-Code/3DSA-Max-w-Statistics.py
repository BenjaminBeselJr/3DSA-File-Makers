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

#Input
source_input_dir = Path("/mnt/stor-pool-01/projects/heus/EUREC4A_Eulerian/Feb_1st_12day_cdnc70_nudge/")

#Output
output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir.mkdir(parents=True, exist_ok=True)

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
ds_w = ds_w.rename({'zh':'z'}).interp(z=ds_ql.z).load()

print("Dataset opening complete")

num_times = int(ds_ql.time.size)
nz, ny, nx = ds_ql.ql.shape[1:]
coords = ds_ql.coords
z_coordinates = ds_ql.z.values

#allocate arrays
placeholder_tz = np.full((num_times, nz), np.nan, dtype=np.float32)
max_horz_w = xr.DataArray(
    data=placeholder_tz,
    dims=("t", "z"),
    coords={"t": ds_ql.time.values, "z": z_coordinates},
    name="max_w"
)

for t in range(num_times):
    print(f"Processing Timestep {t}/{num_times - 1}...")
    
    # Load the full 3D W volume into memory for this single timestep
    w_3d_volume = ds_w.w.isel(time=t).values
    
    # Instantly calculate the max across both horizontal dimensions (Y and X axes are index 1 and 2)
    # This leaves us with a 1D array of length nz
    local_max_z_profile = np.nanmax(w_3d_volume, axis=(1, 2))
    
    # Store the entire profile for this timestep in one shot
    max_horz_w[t, :] = local_max_z_profile

print("Loop complete. Calculating global time-averaged statistics...")

# Vectorized operation: Average across the time dimension ('t') for all Z profiles at once
max_time_averaged_w = max_horz_w.mean(dim='t')

# --- Save outputs to a combined dataset or separate files ---
ds_output = xr.Dataset({
    "max_w_profile": max_horz_w,               # Dimensions: (t, z)
    "time_averaged_max_w": max_time_averaged_w # Dimensions: (z,)
})

print(f"Exporting dataset...")

output_file = output_dir / "max_w_statistics.nc"
ds_output.to_netcdf(output_file)

print("\nAll computation and exporting complete (Program is safe to close)")