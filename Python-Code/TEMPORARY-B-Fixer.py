import os
import sys
import gc
import numpy as np
import xarray as xr
from pathlib import Path

# =====================================================================
# CONFIGURATIONS
# =====================================================================
input_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
source_input_dir = Path("/mnt/stor-pool-01/projects/heus/EUREC4A_Eulerian/Feb_1st_12day_cdnc70_nudge/")
output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")

# Target downstream stats files that need group modifications
slab_file = output_dir / "slab_averages_grouped.nc"
time_avg_file = output_dir / "time_averaged_slab_averages_grouped.nc"

if not slab_file.is_file():
    print(f"❌ ERROR: Target file missing: {slab_file}", file=sys.stderr)
    sys.exit(1)

print("Initialization Success. Validating 3D source and mask fields...")
path_b = output_dir / "b.nc"
path_vpg_b = output_dir / "vpg_b.nc"
path_w = source_input_dir / "w.nc"
path_pi_b = input_dir / "pi_b.nc"  # Loading existing pi_b.nc directly

# Individual mask file mappings
mask_files = {
    "shell": input_dir / "shell_mask.nc",
    "congestus": input_dir / "congestus_mask.nc",
    "deep": input_dir / "deep_mask.nc"
}

# Verify physical variables
for p in [path_b, path_vpg_b, path_w, path_pi_b]:
    if not p.is_file():
        print(f"❌ ERROR: Missing source 3D file: {p}", file=sys.stderr)
        sys.exit(1)

# Verify foundational mask files
for name, p in mask_files.items():
    if not p.is_file():
        print(f"❌ ERROR: Missing target mask file for '{name}': {p}", file=sys.stderr)
        sys.exit(1)

# =====================================================================
# STEP 1: COMPUTE VECTORIZED SLAB AVERAGES
# =====================================================================
print("\n--- Step 1: Processing 3D variables to update Slab Averages ---")

# Open primary datasets with single-timestep chunk profiles
ds_b = xr.open_dataset(path_b, decode_times=False, chunks={'time': 1})
ds_vpg_b = xr.open_dataset(path_vpg_b, decode_times=False, chunks={'time': 1})
ds_w_raw = xr.open_dataset(path_w, decode_times=False, chunks={'time': 1})
ds_pi_b = xr.open_dataset(path_pi_b, decode_times=False, chunks={'time': 1})

# Open mask datasets individually
ds_masks = {name: xr.open_dataset(p, decode_times=False, chunks={'time': 1}) for name, p in mask_files.items()}

# Correct vertical alignment for w (interp from half-levels 'zh' to centers 'z')
print(" -> Interpolating vertical coordinates for w field...")
ds_w = ds_w_raw.rename({'zh': 'z'}).interp(z=ds_b.z)

num_times = int(ds_b.time.size)
z_coords = ds_b.z.values
nz = len(z_coords)

mask_keys = ["domain", "shell", "congestus", "deep", "congestus_shell", "deep_shell"]
target_vars = ["b", "vpg_b", "b_eff", "ke_w", "pi_b"]

# Preallocate result dictionaries
slab_results = {v: {m: np.zeros((num_times, nz), dtype=np.float32) for m in mask_keys} for v in target_vars}

for t in range(num_times):
    print(f" -> Calculating vectorized spatial averages for Timestep {t + 1}/{num_times}...")
    
    # Extract 3D core fields
    b_3d = ds_b.b.isel(time=t).values
    vpg_b_3d = ds_vpg_b.vpg_b.isel(time=t).values
    w_3d = ds_w.w.isel(time=t).values
    pi_b_3d = ds_pi_b.pi_b.isel(time=t).values  # Pulled straight from the dataset
    
    # Algebra derivations
    b_eff_3d = b_3d + vpg_b_3d
    ke_w_3d = (w_3d ** 2) / 2.0
    
    # Extract boolean tracking arrays directly from their respective datasets
    m_shell = ds_masks["shell"].shell.isel(time=t).values > 0
    m_congestus = ds_masks["congestus"].congestus.isel(time=t).values > 0
    m_deep = ds_masks["deep"].deep.isel(time=t).values > 0
    
    # Reconstruct mask dictionary structure matching old workflow
    masks = {
        "domain": None,  # No masking required
        "shell": m_shell,
        "congestus": m_congestus,
        "deep": m_deep,
        "congestus_shell": m_congestus & m_shell,
        "deep_shell": m_deep & m_shell
    }

    # Map variables to their corresponding raw volume data
    raw_volumes = {
        "b": b_3d,
        "vpg_b": vpg_b_3d,
        "b_eff": b_eff_3d,
        "ke_w": ke_w_3d,
        "pi_b": pi_b_3d
    }

    # Fully vectorized spatial mask calculation across all height layers simultaneously
    for var_key in target_vars:
        raw_volume = raw_volumes[var_key]
        
        for m_key in mask_keys:
            mask_matrix = masks[m_key]
            
            if m_key == "domain" or mask_matrix is None:
                # Optimized domain mean along spatial dimensions (axis 1 and 2)
                slab_profile = np.mean(raw_volume, axis=(1, 2))
            else:
                # Mask elements with NaN so they are cleanly skipped during vectorized averaging
                masked_volume = np.where(mask_matrix, raw_volume, np.nan)
                slab_profile = np.nanmean(masked_volume, axis=(1, 2))
            
            # Save directly into our preallocated arrays
            slab_results[var_key][m_key][t, :] = slab_profile

# Explicit cleanup of file tracking blocks
ds_b.close(); ds_vpg_b.close(); ds_w_raw.close(); ds_w.close(); ds_pi_b.close()
for m_ds in ds_masks.values():
    m_ds.close()
gc.collect()

# =====================================================================
# STEP 2: OVERWRITE GROUP RESULTS IN SLAB_AVERAGES_GROUPED.NC
# =====================================================================
print(f"\n--- Step 2: Committing groups directly to {slab_file.name} ---")

for var_key in target_vars:
    print(f" -> Injecting group data for variable path: '{var_key}'...")
    
    group_ds = xr.Dataset(
        data_vars={m: (["t", "z"], slab_results[var_key][m]) for m in mask_keys},
        coords={"t": np.arange(num_times), "z": z_coords}
    )
    group_ds.to_netcdf(slab_file, mode="a", group=var_key)

# =====================================================================
# STEP 3: RE-CALCULATE TIME AVERAGES & COMPOSE TARGET GROUPS
# =====================================================================
if time_avg_file.is_file():
    print(f"\n--- Step 3: Updating downstream profiles in {time_avg_file.name} ---")
    
    for var_key in target_vars:
        print(f" -> Re-averaging over time dimensions for group: '{var_key}'...")
        
        time_avg_data = {}
        for m in mask_keys:
            # nanmean handles tracking clouds that might disappear at specific times
            mean_profile = np.nanmean(slab_results[var_key][m], axis=0)
            time_avg_data[m] = xr.DataArray(
                data=mean_profile.astype(np.float32),
                dims=("z"),
                coords={"z": z_coords},
                name=m
            )
            
        ds_time_group = xr.Dataset(time_avg_data)
        ds_time_group.to_netcdf(time_avg_file, mode="a", group=var_key)

print("\n✅ Fast-fix complete! All statistics including pi_b have been successfully compiled.")