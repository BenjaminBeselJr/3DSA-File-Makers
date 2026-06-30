import sys
import gc
from pathlib import Path
import numpy as np
import xarray as xr

# --- Configurations ---
input_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir.mkdir(parents=True, exist_ok=True)

input_file_name = "slab_averages_grouped.nc"
file_path = input_dir / input_file_name

if not file_path.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {file_path}\nEnding program early.", file=sys.stderr)
    sys.exit(1)

# --- Metadata Extraction ---
# Instead of opening a heavy 3D file, we read the vertical coordinates 
# directly from the first group of your existing slab averages file.
print("Extracting vertical grid coordinates from slab averages...")
with xr.open_dataset(file_path, group="b", decode_times=False) as ds_meta:
    z_coordinates = ds_meta.z.values
    nz = len(z_coordinates)

# Iteration list setup
physical_vars = {
    "b": "b", "w": "w", "vpg": "vpg", "vpg_b": "vpg_b", "vpg_dn": "vpg_dn", "vpg_dl": "vpg_dl",
    "pi_b": "pi_b", "pi_dn": "pi_dn", "pi_dl": "pi_dl",
    "ke_b": "ke_b", "ke_b_eff": "ke_b_eff", "ke_vpg": "ke_vpg", "ke_vpg_b": "ke_vpg_b",
    "ke_vpg_dn": "ke_vpg_dn", "ke_vpg_dl": "ke_vpg_dl", "ke_w": "ke_w", "b_eff": "b_eff"
}

mask_keys = ["domain", "shell", "congestus", "deep", "congestus_shell", "deep_shell"]

# Initialize a nested tracking structure: { variable_group: { mask_name: DataArray } }
output_groups = {var_key: {} for var_key in physical_vars.keys()}

for var_key in physical_vars.keys():
    for m_key in mask_keys:
        output_groups[var_key][m_key] = xr.DataArray(
            data=np.full((nz), np.nan, dtype=np.float32),
            dims=("z"),
            coords={"z": z_coordinates},
            name=m_key
        )

i = 1
arr_length = len(physical_vars) # Fixed .length() syntax error
for group_name, internal_name in physical_vars.items():
    print(f"Processing {group_name} ({i}/{arr_length})")
    
    # Context manager cleanly opens and closes the group dataset
    with xr.open_dataset(file_path, group=group_name, decode_times=False) as plotted_group:
        for mask_type in mask_keys:
            print(f"    {group_name} ({i}/{arr_length}): Averaging on {mask_type}")
            
            # Compute time mean and assign via .values
            time_mean = plotted_group[mask_type].mean(dim='t')
            output_groups[group_name][mask_type].values = time_mean.values

    i += 1
    gc.collect()

print("\nExporting dataset...")
output_file = output_dir / "time_averaged_slab_averages_grouped.nc"

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