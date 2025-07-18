import numpy as np
from scipy.special import genlaguerre, factorial
from scipy.special import lpmv  # Associated Legendre polynomial
import pyvista as pv
import asyncio
import platform
from tqdm import tqdm
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Physical constants (atomic units: a_0 = 1, hbar = 1, m_e = 1)
a_0 = 1.0  # Bohr radius


def radial_wavefunction(n, l, r):
    """Compute the radial part of the wavefunction R(r) for hydrogen-like atom."""
    rho = 2 * r / (n * a_0)
    laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    norm = np.sqrt((2 / (n * a_0)) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    radial = norm * np.exp(-r / (n * a_0)) * (2 * r / (n * a_0)) ** l * laguerre

    print(f"  - Radial function: n={n}, l={l}")
    print(f"    - Normalization constant: {norm:.2e}")
    print(f"    - Laguerre polynomial stats: min={np.min(laguerre):.2e}, max={np.max(laguerre):.2e}")

    return radial


def spherical_harmonic(l, m, theta, phi):
    """Compute spherical harmonic Y_l^m(θ, φ) manually."""
    norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - abs(m)) / factorial(l + abs(m)))
    cos_theta = np.cos(theta)
    P_lm = lpmv(abs(m), l, cos_theta)
    if m >= 0:
        phase = np.exp(1j * m * phi)
    else:
        phase = (-1) ** abs(m) * np.exp(1j * m * phi)
    Y = norm * P_lm * phase

    if l == 1 and m == 0:
        expected = np.sqrt(3 / (4 * np.pi)) * cos_theta
        print(
            f"  - Y_1^0 check: computed min={np.min(np.abs(Y)):.2e}, max={np.max(np.abs(Y)):.2e}, expected max={np.max(np.abs(expected)):.2e}")
        return expected
    return Y


def wavefunction(n, l, m, r, theta, phi):
    """Compute the full wavefunction ψ(r, θ, φ) = R(r) * Y_l^m(θ, φ)."""
    R = radial_wavefunction(n, l, r)
    Y = spherical_harmonic(l, m, theta, phi)
    psi = R * Y
    return psi


def generate_orbital_grid(n, l, m, grid_size=120, r_max=None):
    """Generate 3D grid of probability density |ψ|^2 with progress bar."""
    if r_max is None:
        r_max = 8.0 if n == 2 else max(10, n * 4 * a_0)
    print(f"Generating orbital grid for n={n}, l={l}, m={m}...")
    print(f"  - Using r_max={r_max:.2f}, grid_size={grid_size}")

    # Simulate progress with tqdm over grid creation steps
    with tqdm(total=4, desc="Processing Grid", unit="step") as pbar:
        x = np.linspace(-r_max, r_max, grid_size)
        pbar.update(1)
        y = np.linspace(-r_max, r_max, grid_size)
        pbar.update(1)
        z = np.linspace(-r_max, r_max, grid_size)
        pbar.update(1)
        X, Y, Z = np.meshgrid(x, y, z)
        pbar.update(1)

    # Convert to spherical coordinates
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    Theta = np.arccos(Z / (R + 1e-10))  # Avoid division by zero
    Phi = np.arctan2(Y, X)

    # Compute wavefunction with progress
    with tqdm(total=1, desc="Computing Wavefunction", unit="pass") as pbar:
        psi = wavefunction(n, l, m, R, Theta, Phi)
        pbar.update(1)

    # Debug: Test points
    test_r = np.array([0.1, 1.0, 4.0, 6.0])
    test_theta = np.array([0.0, np.pi / 2, 0.0, 0.0])
    test_phi = np.array([0.0, 0.0, 0.0, 0.0])
    test_psi = wavefunction(n, l, m, test_r, test_theta, test_phi)
    test_radial = radial_wavefunction(n, l, test_r)
    test_sph = spherical_harmonic(l, m, test_theta, test_phi)
    print(f"  - Test points (r, θ, φ): {list(zip(test_r, test_theta, test_phi))}")
    print(f"    - Test radial values: {test_radial}")
    print(f"    - Test spherical harmonic values: {test_sph}")
    print(f"    - Test wavefunction values: {test_psi}")

    print(f"  - Wavefunction stats: min={np.min(np.abs(psi)):.2e}, max={np.max(np.abs(psi)):.2e}")

    prob_density = np.abs(psi) ** 2

    return X, Y, Z, prob_density


def plot_orbital(n, l, m, grid_size=120, isovalue=None):
    """Plot 3D isosurface of orbital probability density using PyVista with enhanced visuals."""
    X, Y, Z, prob_density = generate_orbital_grid(n, l, m, grid_size)

    # Debug: Print probability density statistics
    max_density = np.max(prob_density)
    min_density = np.min(prob_density)
    non_zero_count = np.sum(prob_density > 1e-10)
    print(
        f"  - Probability density stats: min={min_density:.2e}, max={max_density:.2e}, non-zero points={non_zero_count}/{prob_density.size}")

    # Check if density is effectively zero
    if max_density < 1e-20:
        print(
            f"{Fore.RED}Error: Probability density is effectively zero for n={n}, l={l}, m={m}. Try increasing grid_size or adjusting r_max.{Style.RESET_ALL}")
        return

    # Create a PyVista structured grid
    grid = pv.StructuredGrid(X, Y, Z)
    grid["Probability Density"] = prob_density.flatten(order='F')

    # Determine isovalue dynamically
    if isovalue is None:
        non_zero_density = prob_density[prob_density > 1e-10]
        if non_zero_density.size > 0:
            isovalue = np.percentile(non_zero_density, 10)
        else:
            isovalue = max_density * 0.01
            print(
                f"{Fore.YELLOW}Warning: No non-zero density values found. Using fallback isovalue: {isovalue:.2e}{Style.RESET_ALL}")
        print(f"  - Using isovalue: {isovalue:.2e}")

    # Allow empty meshes
    pv.global_theme.allow_empty_mesh = True

    # Compute isosurface
    isosurface = grid.contour(isosurfaces=[isovalue])

    # Check if isosurface is empty
    if isosurface.n_points == 0:
        print(
            f"{Fore.YELLOW}Warning: Isosurface is empty for isovalue={isovalue:.2e}. Try reducing isovalue (e.g., 1e-7) or increasing grid_size.{Style.RESET_ALL}")
        return

    # Create a plotter with enhanced visuals
    plotter = pv.Plotter(window_size=[1600, 800])
    plotter.enable_lightkit()  # Enable default light kit
    plotter.set_background('white', top='lightgray')  # Clean background like falstad.com
    plotter.add_mesh(isosurface, cmap='RdYlBu', opacity=0.8, smooth_shading=True,
                     specular=0.5, diffuse=0.7, ambient=0.3)  # Vibrant colors, smooth shading
    plotter.add_scalar_bar(title='Probability Density', title_font_size=16, label_font_size=12,
                           color='black', n_labels=5, vertical=True)  # Enhanced scalar bar
    plotter.add_axes(xlabel='X (a₀)', ylabel='Y (a₀)', zlabel='Z (a₀)', line_width=2, labels_off=False)
    plotter.show_grid(color='gray', ticks='outside')  # Subtle grid
    plotter.add_text(f'Orbital: n={n}, l={l}, m={m}', font_size=16, color='black', position='upper_left')

    # Set camera for better view
    plotter.camera_position = 'xy'
    plotter.camera.azimuth = -45
    plotter.camera.elevation = 30
    plotter.camera.zoom(1.2)

    plotter.show()


async def main():
    n = 4
    l = 3
    m = 0
    if not (0 <= l < n and -l <= m <= l):
        raise ValueError("Invalid quantum numbers: Ensure 0 <= l < n and -l <= m <= l")

    plot_orbital(n, l, m, grid_size=250, isovalue=1e-4)


# Local execution (PyVista not supported in Pyodide)
if platform.system() == "Emscripten":
    print("PyVista is not supported in Pyodide. Please run this script in a local Python environment.")
else:
    if __name__ == "__main__":
        asyncio.run(main())
