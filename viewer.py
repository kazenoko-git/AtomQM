import math, warnings
import numpy as np
from scipy.special import sph_harm, eval_genlaguerre, factorial
from scipy.ndimage import zoom, gaussian_filter
from pyscf import gto, scf
import pyvista as pv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QSlider,
                             QCheckBox, QTabWidget, QLineEdit)
from PyQt5.QtCore import Qt, QTimer
from pyvistaqt import QtInteractor
import sys

warnings.filterwarnings("ignore")


# ==================== OPTIMIZED HANDLER ====================

def autoSpin(nelec: int, user_spin: int | None) -> int:
    if user_spin is None:
        return nelec & 1
    if (nelec - user_spin) % 2:
        raise ValueError(f"Electron number {nelec} incompatible with spin {user_spin}.")
    return user_spin


def uniformGrid(extent=5.0, npts=120):
    """Optimized grid generation with memory-efficient arrays"""
    lin = np.linspace(-extent, extent, npts, dtype=np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    pts = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    spacing = (2 * extent) / (npts - 1)
    dims = (npts, npts, npts)
    origin = (-extent, -extent, -extent)
    return pts, dims, origin, (spacing, spacing, spacing)


def hydrogenDensity(n, l, m, Z=1, npts=150, extent=15.0):
    """Optimized hydrogen orbital calculation with vectorization"""
    pts, dims, origin, spacing = uniformGrid(extent, npts)

    # Vectorized spherical coordinate calculation
    r = np.linalg.norm(pts, axis=1)
    r_safe = np.where(r < 1e-10, 1e-10, r)  # Avoid division by zero

    th = np.arccos(np.clip(pts[:, 2] / r_safe, -1, 1))
    ph = np.arctan2(pts[:, 1], pts[:, 0])

    # Optimized radial function
    rho = 2 * Z * r / n
    pref = math.sqrt((2 * Z / n) ** 3 * factorial(n - l - 1) /
                     (2 * n * factorial(n + l)))

    Rnl = pref * np.exp(-rho / 2) * rho ** l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    Ylm = sph_harm(m, l, ph, th)
    psi2 = np.abs(Rnl * Ylm) ** 2

    # Create PyVista grid
    grid = pv.ImageData()
    grid.dimensions = dims
    grid.origin = origin
    grid.spacing = spacing
    grid.point_data["density"] = psi2.astype(np.float32).ravel(order='C')
    return grid


def pyscfMolecule(atom_string: str, basis='sto-3g', charge=0, spin: int | None = None):
    """Create and build molecular system"""
    mol = gto.Mole()
    mol.atom, mol.basis, mol.charge = atom_string, basis, charge
    mol.build()
    mol.spin = autoSpin(mol.nelectron, spin)
    return mol


def hartreeFockGrid(mol, mo_index=0, mode='mo', npts=120, extent=None):
    """Optimized HF calculation with caching"""
    mf = (scf.RHF if mol.spin == 0 else scf.UHF)(mol).run(verbose=0)
    pts, dims, origin, spacing = uniformGrid(extent or 6.0, npts)

    ao = mol.eval_gto("GTOval_sph", pts)

    if mode == 'mo':
        coeff = mf.mo_coeff if mol.spin == 0 else mf.mo_coeff[0]
        psi = ao @ coeff[:, mo_index]
        field = np.abs(psi) ** 2
    else:
        dm = mf.make_rdm1()
        field = np.einsum("pi,ij,pj->p", ao, dm, ao, optimize=True)

    grid = pv.ImageData()
    grid.dimensions = dims
    grid.origin = origin
    grid.spacing = spacing
    grid.point_data["density"] = field.astype(np.float32).ravel(order='C')
    return grid, mf


def extent4n(n, base_extent=15.0):
    """Calculate appropriate extent for quantum number n"""
    return base_extent * (n ** 1.5) * 1.5


# ==================== GUI APPLICATION ====================

class QuantumVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Orbital Visualizer - Optimized")
        self.setGeometry(100, 100, 1600, 1000)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # PyVista viewer (optimized interactor)
        self.plotter = QtInteractor(self)
        main_layout.addWidget(self.plotter.interactor, stretch=3)

        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)

        # State variables
        self.grid = None
        self.render_mode = 'volume'
        self.iso_value = 1e-5
        self.animation_running = False
        self.current_system = 'hydrogen'

        # Animation
        self.orbit_path = None
        self.orbit_frame = 0
        self.orbit_timer = QTimer()
        self.orbit_timer.timeout.connect(self.animate_orbit)

        # Initial render
        self.update_visualization()

    def create_control_panel(self):
        """Create comprehensive control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tabs for different controls
        tabs = QTabWidget()

        # ===== SYSTEM TAB =====
        system_tab = QWidget()
        sys_layout = QVBoxLayout(system_tab)

        # System type selector
        sys_group = QGroupBox("System Type")
        sys_group_layout = QVBoxLayout()

        self.system_combo = QComboBox()
        self.system_combo.addItems(['Hydrogen Atom', 'Oxygen Atom', 'Water Molecule',
                                    'NaCl Salt', 'CO Molecule', 'Custom Molecule'])
        self.system_combo.currentTextChanged.connect(self.on_system_changed)
        sys_group_layout.addWidget(QLabel("Select System:"))
        sys_group_layout.addWidget(self.system_combo)
        sys_group.setLayout(sys_group_layout)
        sys_layout.addWidget(sys_group)

        # Hydrogen parameters
        self.h_group = QGroupBox("Hydrogen Orbital Parameters")
        h_layout = QVBoxLayout()

        h_layout.addWidget(QLabel("Principal (n):"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 10)
        self.n_spin.setValue(3)
        h_layout.addWidget(self.n_spin)

        h_layout.addWidget(QLabel("Angular (l):"))
        self.l_spin = QSpinBox()
        self.l_spin.setRange(0, 9)
        self.l_spin.setValue(2)
        h_layout.addWidget(self.l_spin)

        h_layout.addWidget(QLabel("Magnetic (m):"))
        self.m_spin = QSpinBox()
        self.m_spin.setRange(-9, 9)
        self.m_spin.setValue(0)
        h_layout.addWidget(self.m_spin)

        self.h_group.setLayout(h_layout)
        sys_layout.addWidget(self.h_group)

        # Molecule parameters
        self.mol_group = QGroupBox("Molecule Parameters")
        mol_layout = QVBoxLayout()

        mol_layout.addWidget(QLabel("Atom String:"))
        self.atom_input = QLineEdit("O 0 0 0")
        mol_layout.addWidget(self.atom_input)

        mol_layout.addWidget(QLabel("Basis Set:"))
        self.basis_combo = QComboBox()
        self.basis_combo.addItems(['sto-3g', '6-31g', 'cc-pvdz'])
        mol_layout.addWidget(self.basis_combo)

        mol_layout.addWidget(QLabel("Charge:"))
        self.charge_spin = QSpinBox()
        self.charge_spin.setRange(-5, 5)
        mol_layout.addWidget(self.charge_spin)

        self.mol_group.setLayout(mol_layout)
        self.mol_group.setVisible(False)
        sys_layout.addWidget(self.mol_group)

        sys_layout.addStretch()
        tabs.addTab(system_tab, "System")

        # ===== RENDERING TAB =====
        render_tab = QWidget()
        render_layout = QVBoxLayout(render_tab)

        # Resolution
        res_group = QGroupBox("Grid Resolution")
        res_layout = QVBoxLayout()

        res_layout.addWidget(QLabel("Grid Points (npts):"))
        self.npts_spin = QSpinBox()
        self.npts_spin.setRange(50, 500)
        self.npts_spin.setValue(150)
        self.npts_spin.setSingleStep(10)
        res_layout.addWidget(self.npts_spin)

        res_layout.addWidget(QLabel("Extent:"))
        self.extent_spin = QDoubleSpinBox()
        self.extent_spin.setRange(1.0, 50.0)
        self.extent_spin.setValue(15.0)
        self.extent_spin.setSingleStep(1.0)
        res_layout.addWidget(self.extent_spin)

        res_group.setLayout(res_layout)
        render_layout.addWidget(res_group)

        # Render mode
        mode_group = QGroupBox("Render Mode")
        mode_layout = QVBoxLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Volume', 'Isosurface'])
        self.mode_combo.currentTextChanged.connect(self.on_render_mode_changed)
        mode_layout.addWidget(self.mode_combo)

        mode_layout.addWidget(QLabel("Iso-value:"))
        self.iso_slider = QSlider(Qt.Horizontal)
        self.iso_slider.setRange(1, 100)
        self.iso_slider.setValue(50)
        self.iso_slider.valueChanged.connect(self.on_iso_changed)
        mode_layout.addWidget(self.iso_slider)

        self.iso_label = QLabel("5.00e-05")
        mode_layout.addWidget(self.iso_label)

        mode_group.setLayout(mode_layout)
        render_layout.addWidget(mode_group)

        # Visual quality
        quality_group = QGroupBox("Visual Quality")
        quality_layout = QVBoxLayout()

        self.smooth_check = QCheckBox("Smooth Rendering")
        self.smooth_check.setChecked(True)
        quality_layout.addWidget(self.smooth_check)

        self.downsample_check = QCheckBox("Downsample for Speed")
        self.downsample_check.setChecked(True)
        quality_layout.addWidget(self.downsample_check)

        quality_group.setLayout(quality_layout)
        render_layout.addWidget(quality_group)

        render_layout.addStretch()
        tabs.addTab(render_tab, "Rendering")

        # ===== ANIMATION TAB =====
        anim_tab = QWidget()
        anim_layout = QVBoxLayout(anim_tab)

        anim_group = QGroupBox("Camera Animation")
        anim_group_layout = QVBoxLayout()

        self.anim_btn = QPushButton("Start Orbit Animation")
        self.anim_btn.clicked.connect(self.toggle_animation)
        anim_group_layout.addWidget(self.anim_btn)

        anim_group_layout.addWidget(QLabel("Animation Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 200)
        self.speed_slider.setValue(40)
        anim_group_layout.addWidget(self.speed_slider)

        anim_group.setLayout(anim_group_layout)
        anim_layout.addWidget(anim_group)

        anim_layout.addStretch()
        tabs.addTab(anim_tab, "Animation")

        layout.addWidget(tabs)

        # Action buttons
        btn_layout = QVBoxLayout()

        update_btn = QPushButton("Update Visualization")
        update_btn.clicked.connect(self.update_visualization)
        update_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn_layout.addWidget(update_btn)

        reset_btn = QPushButton("Reset Camera")
        reset_btn.clicked.connect(self.reset_camera)
        btn_layout.addWidget(reset_btn)

        screenshot_btn = QPushButton("Save Screenshot")
        screenshot_btn.clicked.connect(self.save_screenshot)
        btn_layout.addWidget(screenshot_btn)

        layout.addLayout(btn_layout)

        return panel

    def on_system_changed(self, system_name):
        """Handle system type change"""
        self.current_system = system_name.lower().replace(' ', '_')

        # Update presets
        presets = {
            'hydrogen_atom': ('hydrogen', 3, 2, 0, 150, 15.0),
            'oxygen_atom': ('molecule', None, None, None, 160, 15.0),
            'water_molecule': ('molecule', None, None, None, 160, 12.0),
            'nacl_salt': ('molecule', None, None, None, 200, 12.0),
            'co_molecule': ('molecule', None, None, None, 160, 10.0),
        }

        if system_name == 'Hydrogen Atom':
            self.h_group.setVisible(True)
            self.mol_group.setVisible(False)
        else:
            self.h_group.setVisible(False)
            self.mol_group.setVisible(True)

    def on_render_mode_changed(self, mode):
        """Handle render mode change"""
        self.render_mode = mode.lower()
        self.update_visualization()

    def on_iso_changed(self, value):
        """Handle iso-value slider change"""
        if self.grid is not None:
            max_density = self.grid.point_data['density'].max()
            self.iso_value = (value / 100.0) * max_density
            self.iso_label.setText(f"{self.iso_value:.2e}")
            if self.render_mode == 'isosurface':
                self.render_visualization()

    def update_visualization(self):
        """Update the entire visualization"""
        try:
            # Get parameters
            npts = self.npts_spin.value()
            extent = self.extent_spin.value()

            # Load system
            if self.current_system == 'hydrogen_atom' or self.system_combo.currentText() == 'Hydrogen Atom':
                n = self.n_spin.value()
                l = self.l_spin.value()
                m = self.m_spin.value()

                if l >= n:
                    l = n - 1
                    self.l_spin.setValue(l)
                if abs(m) > l:
                    m = 0
                    self.m_spin.setValue(m)

                self.grid = hydrogenDensity(n, l, m, npts=npts, extent=extent)
                title = f"H Atom: n={n}, l={l}, m={m}"
            else:
                # Molecule presets
                atom_configs = {
                    'Oxygen Atom': "O 0 0 0",
                    'Water Molecule': "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0",
                    'NaCl Salt': "Na 0 0 0; Cl 2.5 0 0",
                    'CO Molecule': "C 0 0 0; O 1.1 0 0",
                    'Custom Molecule': self.atom_input.text()
                }

                atom_str = atom_configs.get(self.system_combo.currentText(), self.atom_input.text())
                mol = pyscfMolecule(atom_str, basis=self.basis_combo.currentText(),
                                    charge=self.charge_spin.value())
                self.grid, mf = hartreeFockGrid(mol, mode='density', npts=npts, extent=extent)
                title = f"{atom_str} | E = {mf.e_tot:.6f} Ha"

            # Update iso-value
            if self.grid is not None:
                mean_density = self.grid.point_data['density'].mean()
                self.iso_value = mean_density * 5

            # Render
            self.render_visualization()

            # Setup orbital path for animation
            self.orbit_path = self.plotter.generate_orbital_path(n_points=180, factor=2.5)
            self.orbit_frame = 0

        except Exception as e:
            print(f"Error updating visualization: {e}")

    def render_visualization(self):
        """Render the current grid"""
        if self.grid is None:
            return

        self.plotter.clear()

        if self.render_mode == 'volume':
            # Smooth volume rendering
            density = self.grid.point_data["density"].reshape(self.grid.dimensions, order='C')

            if self.smooth_check.isChecked():
                smooth = gaussian_filter(density, sigma=1.4)
                self.grid.point_data["density"] = smooth.ravel(order='C')

            self.grid.active_scalars_name = "density"
            self.plotter.add_volume(
                self.grid,
                scalars='density',
                cmap='plasma',
                opacity='sigmoid',
                clim=[0, self.grid.point_data['density'].max() * 0.55],
                blending='composite',
                shade=True
            )

        elif self.render_mode == 'isosurface':
            # Optimized isosurface with downsampling
            orig_density = self.grid.point_data['density'].reshape(self.grid.dimensions, order='C')

            if self.downsample_check.isChecked():
                factor = 0.3
                density_lr = zoom(orig_density, factor, order=1)
                dims_lr = tuple(int(a * factor) for a in self.grid.dimensions)
                grid_lr = pv.ImageData()
                grid_lr.dimensions = dims_lr
                grid_lr.origin = self.grid.origin
                grid_lr.spacing = tuple(s / factor for s in self.grid.spacing)
                grid_lr.point_data['density'] = density_lr.ravel(order='C')
                grid_lr.active_scalars_name = 'density'
                iso = grid_lr.contour([self.iso_value])
            else:
                self.grid.active_scalars_name = 'density'
                iso = self.grid.contour([self.iso_value])

            self.plotter.add_mesh(
                iso,
                color="deepskyblue",
                opacity=0.5,
                smooth_shading=True,
                ambient=0.28,
                diffuse=0.69,
                specular=0.9,
                specular_power=40
            )

        self.plotter.reset_camera()

    def toggle_animation(self):
        """Toggle orbit animation"""
        if self.animation_running:
            self.orbit_timer.stop()
            self.anim_btn.setText("Start Orbit Animation")
            self.animation_running = False
        else:
            interval = self.speed_slider.value()
            self.orbit_timer.start(interval)
            self.anim_btn.setText("Stop Orbit Animation")
            self.animation_running = True

    def animate_orbit(self):
        """Animate camera orbit"""
        if self.orbit_path is not None:
            points = self.orbit_path.points
            self.plotter.camera.position = points[self.orbit_frame % len(points)]
            self.orbit_frame += 1

    def reset_camera(self):
        """Reset camera to default position"""
        self.plotter.reset_camera()

    def save_screenshot(self):
        """Save current view as screenshot"""
        from datetime import datetime
        filename = f"quantum_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.plotter.screenshot(filename)
        print(f"Screenshot saved as {filename}")


# ==================== MAIN ====================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    window = QuantumVisualizer()
    window.show()
    sys.exit(app.exec_())
