import math, warnings
import numpy as np
from scipy.special import sph_harm, eval_genlaguerre, factorial
from pyscf import gto, scf

warnings.filterwarnings("ignore")

def autoSpin(nelec: int, user_spin: int | None) -> int:
    if user_spin is None:
        return nelec & 1
    if (nelec - user_spin) % 2:
        raise ValueError(
            f"Electron number {nelec} incompatible with spin {user_spin}. "
            "Remember mol.spin = 2S = Nα–Nβ."
        )
    return user_spin

def uniformGrid(extent=5.0, npts=120):
    lin = np.linspace(-extent, extent, npts)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    pts = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    spacing = (2 * extent) / (npts - 1)
    dims = (npts, npts, npts)
    origin = (-extent, -extent, -extent)
    return pts, dims, origin, (spacing, spacing, spacing)

def hydrogenDensity(n, l, m, Z=1, npts=150, extent=15.0):
    pts, dims, origin, spacing = uniformGrid(extent, npts)
    r = np.linalg.norm(pts, axis=1)
    th = np.arccos(np.clip(pts[:, 2] / r, -1, 1))
    ph = np.arctan2(pts[:, 1], pts[:, 0])
    rho = 2 * Z * r / n
    pref = math.sqrt((2 * Z / n) ** 3 * factorial(n - l - 1) /
                     (2 * n * factorial(n + l)))
    Rnl = pref * np.exp(-rho / 2) * rho ** l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    Ylm = sph_harm(m, l, ph, th)
    psi2 = np.abs(Rnl * Ylm) ** 2

    import pyvista as pv
    grid = pv.ImageData()
    grid.dimensions = dims
    grid.origin = origin
    grid.spacing = spacing
    grid.point_data["density"] = psi2.astype(np.float32).ravel(order='C')
    return grid

def pyscfMolecule(atom_string: str, basis='sto-3g', charge=0, spin: int | None = None):
    mol = gto.Mole()
    mol.atom, mol.basis, mol.charge = atom_string, basis, charge
    mol.build()
    mol.spin = autoSpin(mol.nelectron, spin)
    return mol

def hartreeFockGrid(mol, mo_index=0, mode='mo', npts=120, extent=None):
    mf = (scf.RHF if mol.spin == 0 else scf.UHF)(mol).run()
    pts, dims, origin, spacing = uniformGrid(extent or 6.0, npts)
    ao = mol.eval_gto("GTOval_sph", pts)
    if mode == 'mo':
        coeff = mf.mo_coeff if mol.spin == 0 else mf.mo_coeff[0]
        psi = ao @ coeff[:, mo_index]
        field = np.abs(psi) ** 2
    else:
        dm = mf.make_rdm1()
        field = np.einsum("pi,ij,pj->p", ao, dm, ao)

    import pyvista as pv
    grid = pv.ImageData()
    grid.dimensions = dims
    grid.origin = origin
    grid.spacing = spacing
    grid.point_data["density"] = field.astype(np.float32).ravel(order='C')
    return grid, mf

def loadSystem(kind: str, hydrogen_params=None, molecule_params=None, grid_kw=None):
    grid_kw = grid_kw or {}
    if kind == 'hydrogen':
        n, l, m = hydrogen_params
        grid = hydrogenDensity(n, l, m, **grid_kw)
        label = f"H atom  n={n} l={l} m={m}"
    elif kind == 'molecule':
        mol = pyscfMolecule(**molecule_params)
        grid, mf = hartreeFockGrid(mol, **grid_kw)
        label = f"{mol.atom}  |  RHF energy = {mf.e_tot:.6f} Ha"
    else:
        raise ValueError(f"Unknown kind '{kind}'")
    return grid, label

def extent4n(n, base_extent=15.0):
    """
    Returns a suitable grid extent for a given principal quantum number n.
    Scales as base_extent * n^1.5 * 1.5 for good viewing.
    """
    from math import pow
    return base_extent * pow(n, 1.5) * 1.5
