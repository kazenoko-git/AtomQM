import numpy as np
from pyscf import gto, scf
import pyvista as pv

def compute(symbol='C', basis='cc-pVTZ', grid_n=50, grid_extent=3.0):
    """
    Compute electron density grid for an atom using Hartree-Fock.
    Returns: points (N,3), density (N,)
    """
    # Build PySCF atom
    mol = gto.Mole()
    mol.atom = f'{symbol} 0 0 0'
    mol.basis = basis
    mol.spin = 0
    mol.charge = 0
    mol.build()
    mf = scf.RHF(mol).run()
    # Grid in 3D Cartesian space
    lin = np.linspace(-grid_extent, grid_extent, grid_n)
    xg, yg, zg = np.meshgrid(lin, lin, lin, indexing='ij')
    points = np.vstack((xg.ravel(), yg.ravel(), zg.ravel())).T
    # Compute density at each point
    ao = mol.eval_gto('GTOval_sph', points)
    dm = mf.make_rdm1()
    densities = np.einsum('pi,ij,pj->p', ao, dm, ao)
    return points, densities

def plot(points, densities, symbol, n_samples=80000, noise=0.1):
    """
    Sample and plot points in 3D according to the provided electron density.
    """
    # Normalize and sample
    prob = np.clip(densities, 0, None)
    prob /= prob.sum()
    rng = np.random.default_rng(42)
    chosen = rng.choice(points.shape[0], size=n_samples, p=prob)
    cloud = points[chosen] + noise * rng.normal(size=(n_samples, 3))
    # Visualization
    cloud_points = pv.PolyData(cloud)
    plotter = pv.Plotter()
    plotter.add_mesh(cloud_points, color="navy", point_size=3, render_points_as_spheres=True)
    plotter.show_axes()
    plotter.background_color = 'white'
    plotter.add_text(f"Element - {symbol}\nPoints - {n_samples}", position='upper_left', font_size=13, color='black')
    plotter.show(title=f"Probability Density of {symbol} - AtomQM")

points, densities = compute(symbol='Fe')
plot(points, densities, "Fe")
