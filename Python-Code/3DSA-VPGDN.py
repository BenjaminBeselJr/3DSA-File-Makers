import os
import sys
import gc
from pathlib import Path
import numpy as np
import xarray as xr
import netCDF4 as nc
import scipy.linalg
import metpy
from metpy.units import units
import metpy.calc as mpcalc

#Constants
L_v = 2.5 * (10**6) #latent heat
c_pd = 1004 #specific heat
p_0 = 10**5 #reference pressure
R_d = 287
R_v = 461.5

t_conds = 18

#Input
source_input_dir = Path("/mnt/stor-pool-01/projects/heus/EUREC4A_Eulerian/Feb_1st_12day_cdnc70_nudge/")

#Output
output_dir = Path("/mnt/stor-pool-01/projects/heus/ShellAnalysis/full-area")
output_dir.mkdir(parents=True, exist_ok=True)

print("Imports and constants complete")

#Open datasets

#Check that files exist
path_ds_p = Path(source_input_dir / "p.nc")
if not path_ds_p.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_p}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_initial_conds = Path(source_input_dir / "eurec4a.default.0000000.nc")
if not path_ds_initial_conds.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_initial_conds}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_u = Path(source_input_dir / "u.nc")
if not path_ds_u.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_u}\nEnding program early.", file=sys.stderr)
    sys.exit(1)
path_ds_v = Path(source_input_dir / "v.nc")
if not path_ds_v.is_file():
    print(f"❌ ERROR: Simulation dataset not found at:\n   {path_ds_v}\nEnding program early.", file=sys.stderr)
    sys.exit(1)


#Loading datasets
ds_p = xr.open_dataset(path_ds_p, decode_times=False,chunks={'time': 1}) #pressure
ds_initial_thermo_conds = xr.open_dataset(path_ds_initial_conds,group='thermo', decode_times=False,chunks={'time': 1}).isel(time=t_conds) #initial conds
ds_initial_general_conds = xr.open_dataset(path_ds_initial_conds,group='default', decode_times=False,chunks={'time': 1}).isel(time=t_conds) #initial conds
ds_u = xr.open_dataset(path_ds_u, decode_times=False,chunks={'time': 1}).rename({'xh':'x'}).interp(x=ds_p.x)
ds_v = xr.open_dataset(path_ds_v, decode_times=False,chunks={'time': 1}).rename({'yh':'y'}).interp(y=ds_p.y)
print("Dataset opening complete")

num_times = int(ds_p.time.size)
nz, ny, nx = ds_p.p.shape[1:]
coords = ds_p.coords

#Export vars
#u1 means uint8 and f4 means float32
export_registry = {
    "pi_dn.nc": ("pi_dn", "f4"),
    "vpg_dn.nc": ("vpg_dn", "f4"),
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
        
        t_v[:] = ds_p.time.values
        z_v[:] = ds_p.z.values
        y_v[:] = ds_p.y.values
        x_v[:] = ds_p.x.values
        
        f.createVariable(var_name, data_type, ("time", "z", "y", "x"), zlib=True, complevel=4, chunksizes=(1, nz, ny, nx), fill_value=False)

#Function
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

#extract initial cond variables
rho_o = ds_initial_thermo_conds.rhoref
u_o = ds_initial_general_conds.u
v_o = ds_initial_general_conds.v

#-create th-
th_da = mpcalc.potential_temperature(
    ds_initial_general_conds.p.values * units.pascal, 
    ds_initial_thermo_conds.T.values * units.kelvin
).magnitude

# Replace any top-of-atmosphere Infs or NaNs with standard neighboring numbers
th_da = np.nan_to_num(th_da, nan=300.0, posinf=300.0, neginf=300.0)

# If the top grid point is bad, backfill it with the layer just below it
if not np.isfinite(th_da[-1]):
    th_da[-1] = th_da[-2]

th = xr.DataArray(th_da, coords={"z": ds_initial_thermo_conds.z}, dims=["z"])

#open the created files for writing
open_files = {fname: nc.Dataset(str(output_dir / fname), "a") for fname in export_registry}

for t in range(num_times):
    print(f"\n--- (DN) Processing Timestep {t}/{num_times - 1} ---")
    #slice inputs
    p_slice = ds_p.isel(time=t).p
    u_slice = ds_u.isel(time=t).u
    v_slice = ds_v.isel(time=t).v

    

    #--Calculate pi DN--
    #Obtaining sigma (part of the left hand side)
    sigma_numpy = (c_pd * th).compute().values
    if sigma_numpy.ndim > 1: #leakage protection
        sigma_numpy = sigma_numpy[0]

    #Big V' calc (V-Vo)
    #where V is (u,v) and Vo is (u_o, v_o)
    u_prime = u_slice - u_o
    v_prime = v_slice - v_o

    #Obtaining f' (right hand side)
    advection_x = u_prime * u_prime.differentiate("x") + v_prime * u_prime.differentiate("y")
    advection_y = u_prime * v_prime.differentiate("x") + v_prime * v_prime.differentiate("y")

    flux_x = rho_o * advection_x
    flux_y = rho_o * advection_y

    div_x = flux_x.differentiate("x")
    div_y = flux_y.differentiate("y")

    f_DN = -(div_x + div_y)

    f_DN_numpy = f_DN.compute().values
    f_DN_numpy = np.nan_to_num(f_DN_numpy, nan=0.0) #clean f

    #dx, dy, z vals
    dx = float(ds_p.x[1] - ds_p.x[0])
    dy = float(ds_p.y[1] - ds_p.y[0])
    z_numpy = ds_p.z.compute().values

    #Performing Fourier analysis to find pi'_b
    pi_DN_numpy = solve_generalized_poisson_3d(
        f=f_DN_numpy, 
        sigma=sigma_numpy, 
        z_coords=z_numpy, 
        dx=dx, 
        dy=dy
    )

    pi_DN_xr = xr.DataArray(
        pi_DN_numpy, 
        coords={"z": ds_p.z, "y": ds_p.y, "x": ds_p.x}, 
        dims=["z", "y", "x"]
    )

    # =====================================================================
    # VERIFICATION CHECKER FOR PI_DN (Identical Stencil Matching)
    # =====================================================================
    print(f"\n--- (DN) Running Spectral & Staggered Verification of pi DN at Timestep {t}/{num_times - 1} ---")

    # 1. Reconstruct Solver Horizontal Wavenumbers
    nx, ny = pi_DN_xr.sizes["x"], pi_DN_xr.sizes["y"]
    kx = np.fft.rfftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    k_sq = ky[:, np.newaxis]**2 + kx[np.newaxis, :]**2

    # 2. Compute Horizontal Laplacian Perfectly in Spectral Space
    pi_DN_hat = np.fft.rfft2(pi_DN_numpy, axes=(1, 2))
    sigma_3d = sigma_numpy[:, np.newaxis, np.newaxis]
    horizontal_laplacian_hat = -sigma_3d * k_sq[np.newaxis, :, :] * pi_DN_hat
    horizontal_laplacian_spectral = np.fft.irfft2(horizontal_laplacian_hat, s=(ny, nx), axes=(1, 2))

    # 3. Compute Vertical Divergence using Solver Staggered Grid Metrics
    dz_center = np.diff(z_numpy)
    zh = 0.5 * (z_numpy[:-1] + z_numpy[1:])
    dz_cell = np.diff(zh)
    sigma_half = 0.5 * (sigma_numpy[:-1] + sigma_numpy[1:])

    # Vertical Flux at half-levels (j + 1/2) -> Shape: (nz-1, ny, nx)
    flux_z_half = sigma_half[:, np.newaxis, np.newaxis] * (np.diff(pi_DN_xr.values, axis=0) / dz_center[:, np.newaxis, np.newaxis])

    # Flux Divergence at interior cell centers (j) -> Shape: (nz-2, ny, nx)
    diff_2_z_interior = np.diff(flux_z_half, axis=0) / dz_cell[:, np.newaxis, np.newaxis]

    # 4. Combine LHS Components & Extract Truth Target (Interior Rows 1 to nz-1)
    test_lhs_spectral = horizontal_laplacian_spectral[1:-1, :, :] + diff_2_z_interior
    truth_interior = f_DN_numpy[1:-1, :, :]

    # 5. Numerical Execution Verification
    correlation = np.corrcoef(test_lhs_spectral.flatten(), truth_interior.flatten())[0, 1]
    print(f"(DN) Pure Spectral-Aligned Spatial Correlation for Timestep {t}/{num_times - 1}: {correlation:.6f}")

    if np.allclose(test_lhs_spectral, truth_interior, rtol=1e-2, atol=1e-2):
        print(f"(DN) Verification of Timestep {t}/{num_times - 1}: SUCCESS - Solver approximately matches mathematical formulation")
    else:
        print(f"(DN) Verification of Timestep {t}/{num_times - 1}: FAILURE - Discrepancies found.")

    #VPG DN
    dpi_dz_xr = pi_DN_xr.differentiate(coord="z")
    vpg_DN_numpy = -c_pd * th.values[:, np.newaxis, np.newaxis] * dpi_dz_xr.compute().values

    #Computation complete
    print("DN Computation complete")

    #Exporting
    print(f"(DN) Exporting timestep {t} datasets...")

    open_files["pi_dn.nc"].variables["pi_dn"][t, :, :, :] = pi_DN_numpy
    open_files["vpg_dn.nc"].variables["vpg_dn"][t, :, :, :] = vpg_DN_numpy

    for open_file in open_files.values():
        open_file.sync()
        
    gc.collect()

#closing files
for filename, file_obj in open_files.items():
    file_obj.close()

#Computation complete
print("\nAll computation and exporting complete (Program is safe to close)")