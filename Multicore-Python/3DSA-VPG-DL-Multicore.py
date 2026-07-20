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
import argparse

# =====================================================================
# GLOBAL CONFIGURATION & SHARED REGISTRY
# =====================================================================
EXPORT_REGISTRY = {
    "pi_dl.nc": ("pi_dl", "f4"),
    "vpg_dl.nc": ("vpg_dl", "f4"),
}

# Physical Constants
L_v = 2.5 * (10**6)  # latent heat
c_pd = 1004          # specific heat
p_0 = 10**5          # reference pressure
R_d = 287
R_v = 461.5

# --- FFT Solver ---
def solve_generalized_poisson_3d(f, sigma, z_coords, dx, dy):
    """
    Solves the 3D generalized Poisson equation on a vertically stretched grid:
        grad dot (sigma(z) * grad(m)) = f(x, y, z)
    """
    nz, ny, nx = f.shape
    
    # 1. Horizontal Fourier Transform of the RHS source term
    f_hat = np.fft.rfft2(f, axes=(1, 2))
    
    # 2. Compute symmetric horizontal wavenumbers squared (kx^2 + ky^2)
    kx = np.fft.rfftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    k_sq = ky[:, np.newaxis]**2 + kx[np.newaxis, :]**2
    nkx = len(kx)
    
    # 3. Compute exact stretched grid metrics from true simulation centers
    dz_center = np.diff(z_coords)                 # Distance between centers (nz - 1)
    zh = 0.5 * (z_coords[:-1] + z_coords[1:])     # Reconstructed half-levels (nz - 1)
    dz_cell = np.diff(zh)                         # Thickness of interior cells (nz - 2)
    
    dz_cell_bottom = z_coords[1] - z_coords[0]
    dz_cell_top = z_coords[-1] - z_coords[-2]
    
    sigma_half = 0.5 * (sigma[:-1] + sigma[1:])   # Length: nz - 1
    
    # 4. Pre-compute the shared vertical operators (Built ONCE outside loops)
    interior_lower = sigma_half[:-1] / (dz_cell * dz_center[:-1])
    interior_upper = sigma_half[1:] / (dz_cell * dz_center[1:])
    
    # Allocate template for the base tridiagonal matrix
    ab_base = np.zeros((3, nz), dtype=complex)
    
    # Fill vertical terms for interior rows
    ab_base[2, :-2] = interior_lower               # Corrected lower diagonal shift
    ab_base[0, 2:]  = interior_upper               # Corrected upper diagonal shift
    ab_base[1, 1:-1] = -interior_upper - interior_lower
    
    # Fill vertical terms for boundary cap rows (Neumann boundary conditions)
    ab_base[1, 0]  = -sigma_half[0] / (dz_cell_bottom * dz_center[0])
    ab_base[0, 1]  =  sigma_half[0] / (dz_cell_bottom * dz_center[0])
    
    ab_base[1, -1] = -sigma_half[-1] / (dz_cell_top * dz_center[-1])
    ab_base[2, -2] =  sigma_half[-1] / (dz_cell_top * dz_center[-1])
    
    # Initialize container for the Fourier-space solution
    m_hat = np.zeros_like(f_hat, dtype=complex)
    
    # 5. Fast Execution Loop
    for r in range(ny):
        for c in range(nkx):
            ksq_rc = k_sq[r, c]
            
            if r == 0 and c == 0:
                m_hat[:, r, c] = 0.0
                continue
                
            # Copy the pre-computed base vertical structure 
            ab = ab_base.copy()
            
            # Layer the horizontal curvature decay on top of the main diagonal
            ab[1, :] -= sigma * ksq_rc
            
            # Instantaneous 1D tridiagonal solver execution
            m_hat[:, r, c] = scipy.linalg.solve_banded((1, 1), ab, f_hat[:, r, c])
            
    # 6. Transform back to real physical space
    m = np.fft.irfft2(m_hat, s=(ny, nx), axes=(1, 2))
    
    return m

# --- Multiprocessing Worker Function ---
def process_timestep_worker(args):
    """
    Worker task running on an isolated core. Computes connected-component components 
    and returns localized numpy matrices back to the orchestrator thread.
    """
    start_time = time.time()
    t_idx, t_val, cfg = args

    paths = cfg["paths"]
    dx, dy = cfg["dx"], cfg["dy"]

    init_th_profile = cfg["init_th_profile"]
    c_pd = cfg["c_pd"]
    rho_profile = cfg["rho_profile"]
    du0_dz = cfg["du0_dz"]
    dv0_dz = cfg["dv0_dz"]

    x_target = cfg["x_vals"]
    y_target = cfg["y_vals"]
    z_target = cfg["z_vals"]


    #load datasets
    with xr.open_dataset(paths["w"], decode_times=False, engine="netcdf4") as ds_w:
         
        w_slice = ds_w.w.sel(time=t_val)
        w_slice = w_slice.rename({'zh':'z'}).interp(z=z_target)

        w_vals = w_slice.values
        z_coords = w_slice.z.values

    rho_da = xr.DataArray(rho_profile, coords={'z': z_coords}, dims=['z'])

    #--Calculate pi DL--
    #Obtaining sigma (part of the left hand side)
    sigma_numpy = (c_pd * init_th_profile)
    if sigma_numpy.ndim > 1: #leakage protection
        sigma_numpy = sigma_numpy[0]

    #Obtaining f' (right hand side)
    dw_dx = w_slice.differentiate("x")
    dw_dy = w_slice.differentiate("y")

    f_DL = -rho_da * (dw_dx * du0_dz[:, np.newaxis, np.newaxis] + 
                  dw_dy * dv0_dz[:, np.newaxis, np.newaxis])

    f_DL_numpy = f_DL.compute().values
    f_DL_numpy = np.nan_to_num(f_DL_numpy, nan=0.0) #clean f

    #dx, dy, z vals
    dx = float(x_target[1] - x_target[0])
    dy = float(y_target[1] - y_target[0])
    z_numpy = w_slice.z.compute().values

    #Performing Fourier analysis to find pi'_b
    pi_DL_numpy = solve_generalized_poisson_3d(
        f=f_DL_numpy, 
        sigma=sigma_numpy, 
        z_coords=z_numpy, 
        dx=dx, 
        dy=dy
    )

    pi_DL_xr = xr.DataArray(
        pi_DL_numpy, 
        coords={"z": z_numpy, "y": y_target, "x": x_target}, 
        dims=["z", "y", "x"]
    )

    #VPG DL
    dpi_dz_numpy = pi_DL_xr.differentiate(coord="z").compute().values
    vpg_DL_numpy = -c_pd * init_th_profile[:, np.newaxis, np.newaxis] * dpi_dz_numpy


    # --- Exporting ---
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return t_idx, t_val, {
        "pi_dl.nc": pi_DL_numpy.astype(np.float32),
        "vpg_dl.nc": vpg_DL_numpy.astype(np.float32),
        "duration": elapsed_str,
    }


# --- Main Thread ---
if __name__ == '__main__':
    t_conds = 19
    multiprocessing.set_start_method('spawn', force=True)
    main_start_time = time.time()

    # --- Configurations ---
    num_cores = int(os.environ.get("CORE_COUNT", 1))  # Default to 1 core if not specified

    parser = argparse.ArgumentParser(description="Process 3DSA pipeline for a specific data source.")
    parser.add_argument(
        "--data_source", 
        type=str, 
        required=True, 
        help="Key matching the data source configuration block in config.json"
    )
    args = parser.parse_args()

    # --- Setting up directories from config ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR / "config.json"

    if not CONFIG_PATH.is_file():
        print(f"❌ ERROR: Configuration file missing at: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read json config file
    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)

    #load config preset based on 
    source_key = args.data_source
    if source_key not in config_data["paths"]:
        print(f"❌ ERROR: Data source '{source_key}' not found in config.json", file=sys.stderr)
        sys.exit(1)

    # Extract Paths
    source_input_dir = Path(config_data["paths"][source_key]["source_input_dir"])
    output_dir = Path(config_data["paths"][source_key]["output_dir"])
    default_fname = Path(config_data["paths"][source_key]["default_file_name"])

    #in case directory does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── OVERRIDE SYSTEM TMPDIR WITH CONFIG PATH ──────────────────────────
    custom_tmp_dir = output_dir / "tmp"
    custom_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["TMPDIR"] = str(custom_tmp_dir)
    # ──────────────────────────────────────────────────────────────────────

    print(f"Initialization Success:")
    print(f" -> Source Input Path: {source_input_dir}")
    print(f" -> Output Path:       {output_dir}")
    print(f" -> Active CPU Cores:  {num_cores}")
    print("-" * 50)

    print("Checking file dependencies...")
    file_paths = {
        "p": source_input_dir / "p.nc",
        "w": source_input_dir / "w.nc",
        "initial": source_input_dir / default_fname,
        "netE": source_input_dir / "netE.nc"
    }
    #Check that files exist
    for name, path in file_paths.items():
        if not path.is_file():
            print(f"❌ ERROR: Missing target dependency: {path}", file=sys.stderr)
            sys.exit(1)

    # Global structure
    with xr.open_dataset(file_paths["p"], decode_times=False, engine="netcdf4") as ds_meta, \
        xr.open_dataset(file_paths["netE"], decode_times=False, engine="netcdf4").netE_flux_y_shell as ds_ex_e:
        nz, ny, nx = ds_meta.p.shape[1:]

        all_time_vals = ds_ex_e.time.compute().values

        start_time = 154800
        step_delta = 7200

        
        target_times = [
            float(t_val) for t_val in all_time_vals
            if t_val >= start_time and (t_val - start_time) % step_delta == 0
        ]

        if not target_times:
            print("❌ ERROR: No physical times matched the selection criteria!", file=sys.stderr)
            sys.exit(1)

        num_output_times = len(target_times)
        time_vals = np.array(target_times)

        z_vals = ds_meta.z.compute().values
        y_vals = ds_meta.y.compute().values
        x_vals = ds_meta.x.compute().values
        dx = float(ds_meta.x[1] - ds_meta.x[0])
        dy = float(ds_meta.y[1] - ds_meta.y[0])


    # --- Setup Reference Profile States ---
    with xr.open_dataset(file_paths["initial"], group='thermo', decode_times=False) as ds_t, \
         xr.open_dataset(file_paths["initial"], group='default', decode_times=False) as ds_g:
         
        ds_initial_thermo = ds_t.isel(time=t_conds).compute()
        ds_initial_general = ds_g.isel(time=t_conds).compute()

    if source_key in ["SEUS", "RICO"]:
        rho_profile = ds_initial_general.rhoref.values
    else:
        rho_profile = ds_initial_thermo.rhoref.values
    u_o_da = ds_initial_general.u
    v_o_da = ds_initial_general.v
    
    #differentiate u0 and v0
    du0_dz_profile = u_o_da.differentiate("z").values
    dv0_dz_profile = v_o_da.differentiate("z").values

    #-create th-
    init_th_raw = mpcalc.potential_temperature(
        ds_initial_general.p.values * units.pascal, 
        ds_initial_thermo.T.values * units.kelvin
    )

    init_th_profile = np.asarray(init_th_raw.magnitude, dtype=np.float32)

    # Replace any top-of-atmosphere Infs or NaNs with standard neighboring numbers
    init_th_profile = np.nan_to_num(init_th_profile, nan=300.0, posinf=300.0, neginf=300.0)

    # If the top grid point is bad, backfill it with the layer just below it
    if not np.isfinite(init_th_profile[-1]):
        init_th_profile[-1] = init_th_profile[-2]

    # --- Preallocate NetCDF file structures ---
    open_files = {}
    try:
        print("Pre-allocating NetCDF file structures on disk...")
        for filename, (var_name, data_type) in EXPORT_REGISTRY.items():
            file_path = output_dir / filename
            f = nc.Dataset(str(file_path), "w", format="NETCDF4")
            open_files[filename] = f
            f.createDimension("time", num_output_times)
            f.createDimension("z", nz)
            f.createDimension("y", ny)
            f.createDimension("x", nx)
            
            f.createVariable("time", "f8", ("time",))[:] = time_vals
            f.createVariable("z", "f4", ("z",))[:] = z_vals
            f.createVariable("y", "f4", ("y",))[:] = y_vals
            f.createVariable("x", "f4", ("x",))[:] = x_vals
            
            f.createVariable(var_name, data_type, ("time", "z", "y", "x"), 
                                zlib=True, complevel=4, chunksizes=(1, nz, ny, nx))

        # --- Start Worker Pool ---
        # Package arguments cleanly into a metadata dictionary
        worker_config = {
            "paths": {k: str(v) for k, v in file_paths.items()},
            "dx": dx,
            "dy": dy,
            "x_vals": x_vals,
            "y_vals": y_vals,
            "z_vals": z_vals,
            "init_th_profile": init_th_profile,  # Raw 1D numpy array
            "rho_profile": rho_profile,           # Raw 1D numpy array
            "c_pd": c_pd,                          # Specific heat at constant pressure
            "du0_dz": du0_dz_profile,           
            "dv0_dz": dv0_dz_profile  
        }

        print(f"Spawning Pool with {num_cores} active workers over {num_output_times} timesteps...")
        pool_tasks = [
            (t_idx, t_val, worker_config) 
            for t_idx, t_val in enumerate(target_times)
        ]


        with multiprocessing.Pool(processes=num_cores) as pool:
            for t_idx, t_val, payload in pool.imap_unordered(process_timestep_worker, pool_tasks):
                print(f"Timestep {t_idx}/{num_output_times - 1} (Physical Time: {t_val:.1f}) finished in ({payload['duration']}). Committing to files...")
                
                for filename, data_array in payload.items():
                    if filename == "duration":
                        continue
                    var_key = EXPORT_REGISTRY[filename][0]
                    open_files[filename].variables[var_key][t_idx, :, :, :] = data_array
                    open_files[filename].sync()

                gc.collect()

        main_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - main_start_time))
        print(f"\n✅ All computation and exporting complete in ({main_elapsed_str})")
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