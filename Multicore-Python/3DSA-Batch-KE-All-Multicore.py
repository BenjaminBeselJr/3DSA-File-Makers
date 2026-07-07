import os
import math
import numpy as np
import xarray as xr
import scipy.ndimage
from pathlib import Path
import netCDF4 as nc
import scipy.linalg
import time
import sys
import multiprocessing
import json
import gc
import metpy
from metpy.units import units
import metpy.calc as mpcalc

# =====================================================================
# GLOBAL CONFIGURATION & SHARED REGISTRY
# =====================================================================
EXPORT_REGISTRY = {
    "ke_w.nc": ("ke_w", "f4"),
    "ke_vpg.nc": ("ke_vpg", "f4"),
    "ke_b.nc": ("ke_b", "f4"),
    "ke_vpg_b.nc": ("ke_vpg_b", "f4"),
    "ke_vpg_dn.nc": ("ke_vpg_dn", "f4"),
    "ke_vpg_dl.nc": ("ke_vpg_dl", "f4"),
    "ke_b_eff.nc": ("ke_b_eff", "f4")
}

# --- Multiprocessing Worker Function ---
def process_timestep_worker(args):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    start_time = time.time()
    t, cfg = args

    paths = cfg["paths"]
    dx, dy = cfg["dx"], cfg["dy"]



    #load datasets
    with xr.open_dataset(paths["cloud_mask"], decode_times=False, engine="netcdf4") as ds_cloud_mask, \
         xr.open_dataset(paths["w"], decode_times=False, engine="netcdf4") as ds_w, \
         xr.open_dataset(paths["vpg"], decode_times=False, engine="netcdf4") as ds_vpg, \
         xr.open_dataset(paths["b"], decode_times=False, engine="netcdf4") as ds_b, \
         xr.open_dataset(paths["vpg_b"], decode_times=False, engine="netcdf4") as ds_vpg_b, \
         xr.open_dataset(paths["vpg_dn"], decode_times=False, engine="netcdf4") as ds_vpg_dn, \
         xr.open_dataset(paths["vpg_dl"], decode_times=False, engine="netcdf4") as ds_vpg_dl:

        z_coords = ds_cloud_mask.z.values

        cloud_mask_slice = ds_cloud_mask.cloud_mask.isel(time=t).compute()
        w_slice = ds_w.w.isel(time=t).rename({'zh':'z'}).interp(z=z_coords).compute()
        vpg_slice = ds_vpg.vpg.isel(time=t).compute()
        b_slice = ds_b.b.isel(time=t).compute()
        vpg_b_slice = ds_vpg_b.vpg_b.isel(time=t).compute()
        vpg_dn_slice = ds_vpg_dn.vpg_dn.isel(time=t).compute()
        vpg_dl_slice = ds_vpg_dl.vpg_dl.isel(time=t).compute()
        b_eff_slice = vpg_b_slice + b_slice

    has_nonzero_per_layer = (cloud_mask_slice > 0).any(dim=["x", "y"])

    # Filter the actual Z coordinates to only keep the ones where clouds exist
    cloud_z_coords = cloud_mask_slice.z.where(has_nonzero_per_layer, drop=True)


    if cloud_z_coords.size > 0:
        # Get the mathematical minimum Z value (safeguards against backwards arrays)
        arb_z = cloud_z_coords.min().compute().item()
        zi_msg= f"zi (Cloud Base) = {arb_z}"
    else:
        arb_z = 2000 #fallback value of 2km
        zi_msg= f"zi = 2km (Assuming since Cloud Base DNE)"

    reindex_template = b_slice

    #Compute KE
    ke_w = (w_slice ** 2) / 2
    ke_vpg = compute_bounded_kinetic_energy(vpg_slice, arb_z, template_ds=reindex_template)
    ke_b = compute_bounded_kinetic_energy(b_slice, arb_z, template_ds=reindex_template)
    ke_vpg_b = compute_bounded_kinetic_energy(vpg_b_slice, arb_z, template_ds=reindex_template)
    ke_b_eff = compute_bounded_kinetic_energy(b_eff_slice, arb_z, template_ds=reindex_template)
    ke_vpg_dn = compute_bounded_kinetic_energy(vpg_dn_slice, arb_z, template_ds=reindex_template)
    ke_vpg_dl = compute_bounded_kinetic_energy(vpg_dl_slice, arb_z, template_ds=reindex_template)

    # --- Exporting ---
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t, {
        "duration": elapsed_str,
        "zi_msg": zi_msg,
        "ke_w.nc": ke_w.values,
        "ke_vpg.nc": ke_vpg.values,
        "ke_b.nc": ke_b.values,
        "ke_vpg_b.nc": ke_vpg_b.values,
        "ke_vpg_dn.nc": ke_vpg_dn.values,
        "ke_vpg_dl.nc": ke_vpg_dl.values,
        "ke_b_eff.nc": ke_b_eff.values
        
    }

# --- KE computation function ---
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

# --- Main Thread ---
if __name__ == '__main__':
    # --- Configurations ---
    num_cores = int(os.environ.get("CORE_COUNT", 1))  # Default to 1 core if not specified

    # --- Setting up directories from config ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR / "config.json"

    if not CONFIG_PATH.is_file():
        print(f"❌ ERROR: Configuration file missing at: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read json config file
    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)

    # Extract Paths
    source_input_dir = Path(config_data["paths"]["source_input_dir"])
    output_dir = Path(config_data["paths"]["output_dir"])
    input_dir = output_dir # same as output_dir

    #in case directory does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── OVERRIDE SYSTEM TMPDIR WITH CONFIG PATH ──────────────────────────
    custom_tmp_dir = output_dir / "tmp"
    custom_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["TMPDIR"] = str(custom_tmp_dir)
    # ──────────────────────────────────────────────────────────────────────

    print(f"Initialization Success:")
    print(f" -> Source Input Path: {source_input_dir}")
    print(f" -> Regular Input Path:        {input_dir}")
    print(f" -> Output Path:       {output_dir}")
    print(f" -> Active CPU Cores:  {num_cores}")
    print("-" * 50)

    print("Checking file dependencies...")
    file_paths = {
        "cloud_mask": input_dir / "cloud_mask.nc",
        "w": source_input_dir / "w.nc",
        "vpg": input_dir / "vpg.nc",
        "b": input_dir / "b.nc",
        "vpg_b": input_dir / "vpg_b.nc",
        "vpg_dn": input_dir / "vpg_dn.nc",
        "vpg_dl": input_dir / "vpg_dl.nc",
    }
    #Check that files exist
    for name, path in file_paths.items():
        if not path.is_file():
            print(f"❌ ERROR: Missing target dependency: {path}", file=sys.stderr)
            sys.exit(1)

    # Global structure
    with xr.open_dataset(file_paths["cloud_mask"], decode_times=False, engine="netcdf4") as ds_meta:
        num_times = int(ds_meta.time.size)
        nz, ny, nx = ds_meta.cloud_mask.shape[1:]
        time_vals = ds_meta.time.values
        z_vals = ds_meta.z.values
        y_vals = ds_meta.y.values
        x_vals = ds_meta.x.values
        dx = float(ds_meta.x[1] - ds_meta.x[0])
        dy = float(ds_meta.y[1] - ds_meta.y[0])

    # --- Preallocate NetCDF file structures ---
    open_files = {}
    try:
        print("Pre-allocating NetCDF file structures on disk...")
        for filename, (var_name, data_type) in EXPORT_REGISTRY.items():
            file_path = output_dir / filename
            f = nc.Dataset(str(file_path), "w", format="NETCDF4")
            open_files[filename] = f
            f.createDimension("time", num_times)
            f.createDimension("z", nz)
            f.createDimension("y", ny)
            f.createDimension("x", nx)
            
            f.createVariable("time", "f8", ("time",))[:] = time_vals
            f.createVariable("z", "f4", ("z",))[:] = z_vals
            f.createVariable("y", "f4", ("y",))[:] = y_vals
            f.createVariable("x", "f4", ("x",))[:] = x_vals
            
            f.createVariable(var_name, data_type, ("time", "z", "y", "x"), 
                                zlib=True, complevel=4, chunksizes=(1, nz, ny, nx), fill_value=False)

        # --- Start Worker Pool ---
        # Package arguments cleanly into a metadata dictionary
        worker_config = {
            "paths": {k: str(v) for k, v in file_paths.items()},
            "dx": dx,
            "dy": dy,
        }

        print(f"Spawning Pool with {num_cores} active workers over {num_times} timesteps...")
        pool_tasks = [(t, worker_config) for t in range(num_times)]

        with multiprocessing.Pool(processes=num_cores) as pool:
            for t_idx, payload in pool.imap_unordered(process_timestep_worker, pool_tasks):
                print(f"Timestep {t_idx}/{num_times - 1} finished in ({payload['duration']}). Committing to files...")
                print(f" -> {payload['zi_msg']}")
                
                for filename, data_array in payload.items():
                    if filename == "duration" or filename == "zi_msg":
                        continue
                    var_key = EXPORT_REGISTRY[filename][0]
                    open_files[filename].variables[var_key][t_idx, :, :, :] = data_array
                    open_files[filename].sync()

                gc.collect()

        print("\n✅ All computation and exporting complete")
    except KeyboardInterrupt:
        print("\n⚠️ Job interrupted or cancelled via Slurm. Closing files safely...")
    finally:
        # This block ALWAYS runs, ensuring handles are dropped on normal exit OR scancel
        print("Flushing and closing all NetCDF file handles...")
        for filename, file_obj in open_files.items():
            try:
                file_obj.close()
                print(f" -> Closed: {filename}")
            except Exception as e:
                print(f" -> Error closing {filename}: {e}")

        try:
            import shutil
            if custom_tmp_dir.exists():
                shutil.rmtree(custom_tmp_dir)
                print("🧹 Cleaned up temporary buffer directory.")
        except Exception as e:
            print(f"⚠️ Could not automatically clean up tmp folder: {e}")

        print("\n✅ All file streams safely disconnected (Program is safe to close).")