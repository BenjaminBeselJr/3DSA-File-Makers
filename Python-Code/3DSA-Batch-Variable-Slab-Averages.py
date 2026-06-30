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

# --- Configurations ---
input_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
source_input_dir = Path("/mnt/stor-pool-01/projects/heus/EUREC4A_Eulerian/Feb_1st_12day_cdnc70_nudge/")

output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Define every dataset to load dynamically {filename: dictionary_key}
input_registry = {
    "b.nc": "b", "vpg.nc": "vpg", "vpg_b.nc": "vpg_b", "vpg_dn.nc": "vpg_dn", "vpg_dl.nc": "vpg_dl",
    "pi_b.nc": "pi_b", "pi_dn.nc": "pi_dn", "pi_dl.nc": "pi_dl",
    "ke_b.nc": "ke_b", "ke_b_eff.nc": "ke_b_eff", "ke_vpg.nc": "ke_vpg", "ke_vpg_b.nc": "ke_vpg_b",
    "ke_vpg_dn.nc": "ke_vpg_dn", "ke_vpg_dl.nc": "ke_vpg_dl", "ke_w.nc": "ke_w",
    "shell_mask.nc": "shell_mask", "congestus_mask.nc": "congestus_mask", "deep_mask.nc": "deep_mask",
}

# Define your mask calculations combinations
mask_keys = ["domain", "shell", "congestus", "deep", "congestus_shell", "deep_shell"]

loaded_datasets = {}
print("Verifying and loading required input datasets...")
for filename, var_name in input_registry.items():
    file_path = input_dir / filename
    if not file_path.is_file():
        print(f"❌ ERROR: Dataset missing: {file_path}", file=sys.stderr)
        sys.exit(1)
    loaded_datasets[var_name] = xr.open_dataset(file_path, decode_times=False, chunks={'time': 1})

# Load and interpolate 'w' explicitly from its distinct path
path_ds_w = source_input_dir / "w.nc"
if not path_ds_w.is_file():
    print(f"❌ ERROR: Dataset missing: {path_ds_w}", file=sys.stderr)
    sys.exit(1)

print("Interpolating w onto the core coordinate grid...")
ds_w_raw = xr.open_dataset(path_ds_w, decode_times=False, chunks={'time': 1})
loaded_datasets["w"] = ds_w_raw.rename({'zh': 'z'}).interp(z=loaded_datasets["shell_mask"].z).load()

# 2. Extract spatial metadata bounds
ds_meta = loaded_datasets["shell_mask"]
num_times = int(ds_meta.time.size)
nz, ny, nx = ds_meta.shell_mask.shape[1:]
z_coordinates = ds_meta.z.values
time_values = ds_meta.time.values

# 3. Dynamic Registry of physical variables and their internal variable names
physical_vars = {
    "b": "b", "w": "w", "vpg": "vpg", "vpg_b": "vpg_b", "vpg_dn": "vpg_dn", "vpg_dl": "vpg_dl",
    "pi_b": "pi_b", "pi_dn": "pi_dn", "pi_dl": "pi_dl",
    "ke_b": "ke_b", "ke_b_eff": "ke_b_eff", "ke_vpg": "ke_vpg", "ke_vpg_b": "ke_vpg_b",
    "ke_vpg_dn": "ke_vpg_dn", "ke_vpg_dl": "ke_vpg_dl", "ke_w": "ke_w", "b_eff": "b_eff"
}

# Initialize a nested tracking structure: { variable_group: { mask_name: DataArray } }
output_groups = {var_key: {} for var_key in physical_vars.keys()}

for var_key in physical_vars.keys():
    for m_key in mask_keys:
        # The internal variable name inside the group is just the mask name (tidy!)
        output_groups[var_key][m_key] = xr.DataArray(
            data=np.full((num_times, nz), np.nan, dtype=np.float32),
            dims=("t", "z"),
            coords={"t": time_values, "z": z_coordinates},
            name=m_key
        )

# --- Computation Loop ---
for t in range(num_times):
    print(f"Processing Timestep {t}/{num_times - 1}...")
    
    # Extract structural NumPy boolean mask arrays for this timestep instantly
    m_shell = loaded_datasets["shell_mask"].shell_mask.isel(time=t).values.astype(bool)
    m_congestus = loaded_datasets["congestus_mask"].congestus_mask.isel(time=t).values.astype(bool)
    m_deep = loaded_datasets["deep_mask"].deep_mask.isel(time=t).values.astype(bool)
    
    # Pre-calculate intersected mask matrices natively in NumPy
    masks = {
        "domain": None,  # No masking required
        "shell": m_shell,
        "congestus": m_congestus,
        "deep": m_deep,
        "congestus_shell": m_congestus & m_shell,
        "deep_shell": m_deep & m_shell
    }

    # Iterate over every physics variable for this specific single timestep chunk
    for var_key, internal_name in physical_vars.items():
        # Load raw 3D data slice cleanly into RAM
        if var_key == "b_eff": #b eff is not in a dataset and calculated separately
            # Extract both dependencies directly from loaded_datasets for timestep t
            vol_b = loaded_datasets["b"]["b"].isel(time=t).values
            vol_vpg_b = loaded_datasets["vpg_b"]["vpg_b"].isel(time=t).values
            raw_volume = vol_vpg_b + vol_b
        else:
            # Load raw 3D data slice cleanly from disk into RAM
            raw_volume = loaded_datasets[var_key][internal_name].isel(time=t).values
        
        for m_key, mask_matrix in masks.items():
            if m_key == "domain":
                masked_volume = raw_volume
            else:
                # Fill unmasked elements with NaN so they are skipped during averaging
                masked_volume = np.where(mask_matrix, raw_volume, np.nan)
            
            # Compute slab averages over the horizontal axes (1, 2)
            # Swap np.nanmean to np.nanmax here if you prefer slab maximums
            slab_profile = np.nanmean(masked_volume, axis=(1, 2))
            
            # Save results into our tracking registry dictionary group
            output_groups[var_key][m_key][t, :] = slab_profile

    gc.collect()
print(f"Exporting dataset...")

# Combine the complete dictionary of arrays into a centralized dataset
output_file = output_dir / "slab_averages_grouped.nc"

# To avoid conflicts, if an old run file exists, remove it before starting clean group writes
if output_file.exists():
    output_file.unlink()

# Loop through each physical variable and write it as its own isolated group
for var_key, mask_arrays in output_groups.items():
    print(f"Writing group: '{var_key}'...")
    
    # Convert this specific variable's masks into a tidy standalone dataset
    ds_group = xr.Dataset(mask_arrays)
    
    # Write to NetCDF under a specific group path
    ds_group.to_netcdf(output_file, mode="a", group=var_key)

print("\nAll computation and exporting complete (Program is safe to close)")