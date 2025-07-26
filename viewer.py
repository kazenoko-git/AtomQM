import pyvista as pv
import handler as ql
import scipy
from handler import extent4n

n = 3  # Principal quantum number (for hydrogen examples)
extent = extent4n(n)

# Uncomment/Comment any below as needed

# Hydrogen Atom

grid, title = ql.loadSystem(
    kind='hydrogen',
    hydrogen_params=(n, 2, 0),  # Change n,l,m
    grid_kw=dict(npts=400, extent=extent)
)

# Oxygen Atom

# grid, title = ql.loadSystem(
#     kind='molecule',
#     molecule_params=dict(atom_string="O 0 0 0", basis="sto-3g", charge=0, spin=None),
#     grid_kw=dict(npts=160, extent=extent)
# )

# Water molecule

# grid, title = ql.loadSystem(
#     kind='molecule',
#     molecule_params=dict(
#         atom_string="O 0 0 0; H 0.96 0 0; H -0.24 0.93 0",
#         basis="sto-3g", charge=0, spin=None),
#     grid_kw=dict(mode='density', npts=160, extent=12)
# )

# Sodium chloride salt

# grid, title = ql.loadSystem(
#     kind='molecule',
#     molecule_params=dict(
#         atom_string="Na 0 0 0; Cl 2.5 0 0",
#         basis="sto-3g", charge=0, spin=None),
#     grid_kw=dict(mode='density', npts=160, extent=12)
# )

# Carbon monoxide

# grid, title = ql.loadSystem(
#     kind='molecule',
#     molecule_params=dict(
#         atom_string="C 0 0 0; O 1.1 0 0", basis="sto-3g", charge=0, spin=None),
#     grid_kw=dict(mode='mo', mo_index=0, npts=160, extent=10)
# )

pl = pv.Plotter(window_size=(1100, 900))

render_state = {
    'mode': 'volume',
    'iso_value': grid.point_data['density'].mean() * 5,
    'actor': None
}

def update_plot():
    pl.clear()
    if render_state['mode'] == 'volume':
        # Smoother blocks with 3D gaussian filter
        density = grid.point_data["density"].reshape(grid.dimensions, order='C')
        smooth = scipy.ndimage.gaussian_filter(density, sigma=1.4)
        grid.point_data["density"] = smooth.ravel(order='C')
        grid.active_scalars_name = "density"
        pl.add_volume(
            grid,
            scalars='density',
            cmap='plasma',                # Smooth, natural colors
            opacity='sigmoid',                # Soft edges (try 'sigmoid'/'linear'/geom')
            clim=[0, grid.point_data['density'].max() * 0.55],  # Lower clim for richer contrast
            blending='composite',
            shade=False                    # Enable shading for gradients
        )

    elif render_state['mode'] == 'isosurface':
        # Downsample grid for ultra-fast contour & smooth mesh
        from scipy.ndimage import zoom
        orig_density = grid.point_data['density'].reshape(grid.dimensions, order='C')
        factor = 0.3  # Downsample even at modest npts for speed
        density_lr = zoom(orig_density, factor, order=1)
        dims_lr = tuple(int(a * factor) for a in grid.dimensions)
        grid_lr = pv.ImageData()
        grid_lr.dimensions = dims_lr
        grid_lr.origin = grid.origin
        grid_lr.spacing = tuple(s / factor for s in grid.spacing)
        grid_lr.point_data['density'] = density_lr.ravel(order='C')
        grid_lr.active_scalars_name = 'density'
        iso = grid_lr.contour([render_state['iso_value']])

        pl.add_mesh(
            iso,
            color="deepskyblue",           # Brighter, more glassy highlight
            opacity=0.5,
            smooth_shading=True,           # Make surface "fluid"
            ambient=0.28,
            diffuse=0.69,
            specular=0.9,
            specular_power=40              # Brighter, sharper highlight
        )
    pl.add_text(title, position='upper_left', font_size=16)
    update_hud()
    pl.render()

def update_hud():
    pl.remove_actor('hud')
    iso_text = f", Iso-Value: {render_state['iso_value']:.2e}" if render_state['mode'] == 'isosurface' else ""
    hud_text = f"Mode: {render_state['mode'].upper()}{iso_text}\n" \
               f"Press 'M' to toggle mode\n" \
               f"Use UP/DOWN arrows to change iso-value"
    pl.add_text(hud_text, position='upper_right', font_size=13, name='hud')

def toggle_mode():
    render_state['mode'] = 'isosurface' if render_state['mode'] == 'volume' else 'volume'
    update_plot()

def adjust_iso_value(factor):
    if render_state['mode'] == 'isosurface':
        render_state['iso_value'] *= factor
        update_plot()

pl.add_key_event('m', toggle_mode)
pl.add_key_event('Up', lambda: adjust_iso_value(1.18))
pl.add_key_event('Down', lambda: adjust_iso_value(0.85))

pl.add_text(title, position='upper_left', font_size=15)
update_plot()

# Camera animation: safe main-thread timer callback
path_poly = pl.generate_orbital_path(n_points=180, factor=2.5)
path = path_poly.points
toggle = {'run': True}
frame = {'idx': 0}

def tick(_=None):
    if toggle['run']:
        pl.camera.position = path[frame['idx'] % len(path)]
        frame['idx'] += 1

pl.add_key_event('space', lambda: toggle.update(run=not toggle['run']))

# Use the best available timer or fall back to render callback
if hasattr(pl, "add_timer_callback"):
    pl.add_timer_callback(callback=tick, interval=40)
elif hasattr(pl, "add_on_render_callback"):
    pl.add_on_render_callback(tick)
else:
    print("No animation callback possible with this PyVista version.")

pl.add_text("SPACE = Pause/Resume Orbit", position='lower_left', font_size=11)

pl.show()
