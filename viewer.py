import pyvista as pv
import handler as ql
import time, scipy
from handler import extent4n

n = 3  # choose your principal quantum number
extent = extent4n(n)

# Uncomment/Comment any of the below as needed

# Hydrogen Atom
grid, title = ql.loadSystem(
    kind='hydrogen',
    hydrogen_params=(n, 2, 0), # Change n,l,m values as necessary
    grid_kw=dict(npts=400, extent=extent) # Increased npts from 140 to 220
)

# Oxygen Atom
"""grid, title = ql.loadSystem(
    kind='molecule',
    molecule_params=dict(atom_string="O 0 0 0", basis="sto-3g", charge=0, spin=None),
    grid_kw=dict(npts=400, extent=extent) # Increased npts from 140 to 220
)"""

# Water molecule
"""grid, title = ql.loadSystem(
    kind='molecule',
    molecule_params=dict(
        atom_string="O 0 0 0; H 0.96 0 0; H -0.24 0.93 0",
        basis="sto-3g",
        charge=0,
        spin=None
    ),
    grid_kw=dict(mode='density', npts=160, extent=12)
)"""

# Sodium chloride salt
"""grid, title = ql.loadSystem(
    kind='molecule',
    molecule_params=dict(
        atom_string="Na 0 0 0; Cl 2.5 0 0",  # Diatomic, illustrative
        basis="sto-3g",
        charge=0,
        spin=None
    ),
    grid_kw=dict(mode='density', npts=160, extent=12)
)"""

# Carbon Monoxide molecule
"""grid, title = ql.load_system(
    kind='molecule',
    molecule_params=dict(atom_string="C 0 0 0; O 1.1 0 0", basis="sto-3g", charge=0, spin=None),
    grid_kw=dict(mode='mo', mo_index=0, npts=160, extent=10)
)"""

pl = pv.Plotter(window_size=(900, 700))

if "density" in list(grid.point_data):
    density = grid.point_data["density"].reshape(grid.dimensions, order='C')
    density_smooth = scipy.ndimage.gaussian_filter(density, sigma=1.0)  # sigma
    n_points = grid.GetNumberOfPoints()
    if density is not None and density.size != n_points:
        grid.point_data["density"] = density_smooth.ravel(order='C')[:n_points]
    grid.active_scalars_name = "density"
else:
    density = None

pl.add_volume(grid, scalars='density',
              cmap='magma', opacity='sigmoid',
              clim=[0, grid.point_data['density'].max() * 0.6])
pl.add_text(title, position='upper_left', font_size=13)

# Atom visualization omitted for brevity; keep as before if needed.

path = pl.generate_orbital_path(n_points=200, factor=2.5)
toggle = {'run': True}

def onSpace():
    toggle['run'] = not toggle['run']
pl.add_key_event('space', onSpace)
pl.add_text("SPACE = pause/resume", position='lower_left', font_size=10)

path_poly = pl.generate_orbital_path(n_points=180, factor=2.5)
path = path_poly.points  # array shape (n_points, 3)

def animateCamera(plotter, path, toggle):
    idx = 0
    while True:
        if toggle['run']:
            pos = path[idx % len(path)]
            plotter.camera_position = (pos, plotter.camera.focal_point, plotter.camera.up)
            plotter.update()
            time.sleep(0.05)
            idx += 1
        else:
            time.sleep(0.05)


import threading
animation_thread = threading.Thread(target=animateCamera, args=(pl, path, toggle), daemon=True)
animation_thread.start()

pl.show()
