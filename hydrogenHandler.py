import numpy as np
import pyvista as pv
import math
from scipy.special import sph_harm, genlaguerre

def calculate_orbital_cloud(n, l, m, Z=1, N=200, lim=12, N_points=80000, noise=0.25):
    """Calculate a point cloud sampled according to hydrogen-like orbital probability density."""
    # Create grid
    x, y, z = np.meshgrid(
        np.linspace(-lim, lim, N),
        np.linspace(-lim, lim, N),
        np.linspace(-lim, lim, N),
        indexing='ij'
    )
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.divide(z, r, out=np.zeros_like(z), where=r!=0))
    phi = np.arctan2(y, x)

    # Hydrogen wavefunction
    rho = 2 * Z * r / n
    norm = np.sqrt((2 * Z / n)**3 * math.factorial(n-l-1)/(2*n*math.factorial(n+l)))
    laguerre = genlaguerre(n-l-1, 2*l+1)(rho)
    radial = norm * np.exp(-rho/2) * rho**l * laguerre
    angular = sph_harm(m, l, phi, theta)
    psi = radial * angular
    prob_density = np.abs(psi)**2
    prob_density /= prob_density.sum()  # Normalize for probability weighting

    # Sample points by density
    rng = np.random.default_rng(0)
    flat_prob = prob_density.ravel()
    flat_prob /= flat_prob.sum()
    indices = rng.choice(flat_prob.size, N_points, p=flat_prob)
    cloud = np.column_stack((x.ravel()[indices], y.ravel()[indices], z.ravel()[indices]))
    cloud += noise * rng.normal(size=cloud.shape)  # Add noise for aesthetics

    return cloud

def show_orbital_cloud_pyvista(cloud, n, l, m, Z, N_points):
    """Display the sampled orbital cloud using PyVista and overlay the orbital definition text."""
    point_cloud = pv.PolyData(cloud)
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color="navy", point_size=3, render_points_as_spheres=True)
    plotter.show_axes()
    plotter.background_color = 'white'
    if n == 0: subshell = "e"
    elif n == 1 and l == 0: subshell = "s"
    elif n == 2:
        if l == 1: subshell = "p"
        elif l == 0: subshell = "s"
        else: subshell = "e"
    elif n == 3:
        if l == 2: subshell = "d"
        elif l == 1: subshell = "p"
        elif l == 0: subshell = "s"
        else: subshell = "e"
    elif n >= 4:
        if l == 3: subshell = "f"
        elif l == 2: subshell = "d"
        elif l == 1: subshell = "p"
        elif l == 0: subshell = "s"
        else: subshell = "e"
    elif n > 8: n = subshell = "e"
    if l == 0: orbital = ""
    elif l == 1:
        if m < 0: orbital = "-x"
        elif m > 0: orbital = "-y"
        else: orbital = "-z"
    elif l == 2:
        if m == 0: orbital = "-z²"
        elif m == -1: orbital = "-yz/xz"
        elif m == 1: orbital = "-yz/xz2"
        elif m < -1: orbital = "-xy/x²y²"
        elif m > 1: orbital = "-x²y²"
    else: orbital = ""

    orbital_def = (
        f"Hydrogen\n"
        f"{n}{subshell}{orbital}, points={N_points}\n" if l < 3 else f"Hydrogen\n{n}{subshell}, m={m}, points={N_points}\n"
        "ψₙₗₘ(r,θ,φ) = Rₙₗ(r)·Yₗₘ(θ,φ)"
    )
    plotter.add_text(orbital_def, position='upper_left', font_size=13, color='black')
    plotter.show(title=f"Probability Density of Hydrogen {n}{subshell}{orbital} - AtomQM" if l < 3 else f"Probability Density of Hydrogen {n}{subshell} m={m} - AtomQM")

# Example usage
n, l, m = 6, 4, 0
Z = 1
N_points = 100000
cloud = calculate_orbital_cloud(n, l, m, Z=Z, N=500, lim=12, N_points=N_points)
show_orbital_cloud_pyvista(cloud, n, l, m, Z, N_points)
