"""3D aircraft visualization with VTK (GPU-accelerated) and Matplotlib fallback.

Provides embedded 3D viewing for the Tkinter GUI. Uses VTK via pyvista for
high-performance rendering with Phong shading. Falls back to Matplotlib 3D
if VTK/pyvista is not available.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any

import numpy as np

# Component color scheme (R, G, B) — no alpha needed for VTK
COMPONENT_COLORS = {
    "fuselage": (0.7, 0.7, 0.75),
    # Wing panels (inner/outer split)
    "wing_inner_right": (0.2, 0.35, 0.75),
    "wing_inner_left": (0.2, 0.35, 0.75),
    "wing_outer_right": (0.25, 0.45, 0.85),
    "wing_outer_left": (0.25, 0.45, 0.85),
    # Control surfaces
    "elevon_right": (0.7, 0.3, 0.3),
    "elevon_left": (0.7, 0.3, 0.3),
    "aileron_right": (0.8, 0.5, 0.2),
    "aileron_left": (0.8, 0.5, 0.2),
    # Legacy single-piece wing (backward compat)
    "wing_right": (0.2, 0.4, 0.8),
    "wing_left": (0.2, 0.4, 0.8),
    "vertical_stabilizer": (0.3, 0.7, 0.3),
    # V-tail fins
    "vtail_right": (0.3, 0.65, 0.35),
    "vtail_left": (0.3, 0.65, 0.35),
    # Intakes (side-mounted, chin, dorsal)
    "intake_right": (0.8, 0.3, 0.2),
    "intake_left": (0.8, 0.3, 0.2),
    "intake_chin": (0.8, 0.3, 0.2),
    "intake_dorsal": (0.75, 0.35, 0.25),
    "bld_dorsal": (0.6, 0.6, 0.6),
    "bld_right": (0.6, 0.6, 0.6),
    "bld_left": (0.6, 0.6, 0.6),
    "bld_chin": (0.6, 0.6, 0.6),
    "exhaust": (0.9, 0.6, 0.1),
    "wing_root_fairing_right": (0.5, 0.5, 0.8),
    "wing_root_fairing_left": (0.5, 0.5, 0.8),
    "strake_right": (0.4, 0.55, 0.85),
    "strake_left": (0.4, 0.55, 0.85),
    # Saw-tooth trailing edge (stealth)
    "sawtooth_te_right": (0.5, 0.5, 0.5),
    "sawtooth_te_left": (0.5, 0.5, 0.5),
}

# Check VTK availability
_HAS_VTK = False
try:
    import vtkmodules.all  # noqa: F401 — ensures VTK is importable
    _HAS_VTK = True
except Exception:
    try:
        import vtk as _vtk_test  # noqa: F401
        _HAS_VTK = True
    except Exception:
        pass

_HAS_PYVISTA = False
if _HAS_VTK:
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True  # We render to image, not to a separate window
        _HAS_PYVISTA = True
    except Exception:
        pass


def is_vtk_available() -> bool:
    """Return True if VTK+PyVista are available for GPU rendering."""
    return _HAS_PYVISTA


def _numpy_to_polydata(
    verts: np.ndarray, faces: np.ndarray
) -> "pv.PolyData":
    """Convert (vertices, faces) numpy arrays to a PyVista PolyData mesh."""
    if len(verts) == 0 or len(faces) == 0:
        return pv.PolyData()

    # Filter out-of-range face indices
    valid = np.all(faces < len(verts), axis=1) & np.all(faces >= 0, axis=1)
    faces_valid = faces[valid]

    if len(faces_valid) == 0:
        return pv.PolyData()

    # PyVista faces format: [n_pts, v0, v1, v2, n_pts, v0, v1, v2, ...]
    n_faces = len(faces_valid)
    pv_faces = np.column_stack([
        np.full(n_faces, 3, dtype=np.int64),
        faces_valid.astype(np.int64),
    ]).ravel()

    return pv.PolyData(verts.astype(np.float64), pv_faces)


class VTKAircraftViewer:
    """GPU-accelerated 3D aircraft viewer embedded in Tkinter via off-screen rendering.

    Renders the aircraft with PyVista/VTK off-screen, converts to an image,
    and displays it in a Tkinter Canvas with mouse interaction for
    rotation, pan, and zoom.
    """

    def __init__(self, parent: tk.Frame):
        self._parent = parent
        self._components: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._visibility: dict[str, bool] = {}
        self._render_mode = "solid"  # "solid", "wireframe", "solid_wire"
        self._bg_color = "dark"  # "dark" or "light"

        # Plotter (off-screen)
        self._plotter: pv.Plotter | None = None
        self._actors: dict[str, Any] = {}

        # Tkinter canvas for displaying rendered image
        self._canvas = tk.Canvas(parent, bg="#1e1e2e", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._photo_image: Any = None  # tk.PhotoImage reference

        # Camera state for mouse interaction
        self._cam_azimuth = -135.0
        self._cam_elevation = 25.0
        self._cam_distance = 30.0
        self._cam_focal = np.array([0.0, 0.0, 0.0])
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._last_mouse_x = 0
        self._last_mouse_y = 0

        # Bind mouse events
        self._canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self._canvas.bind("<B1-Motion>", self._on_mouse_drag_rotate)
        self._canvas.bind("<ButtonPress-3>", self._on_mouse_press)
        self._canvas.bind("<B3-Motion>", self._on_mouse_drag_pan)
        self._canvas.bind("<MouseWheel>", self._on_mouse_scroll)
        self._canvas.bind("<Button-4>", self._on_mouse_scroll)
        self._canvas.bind("<Button-5>", self._on_mouse_scroll)
        self._canvas.bind("<Configure>", self._on_resize)

        self._render_pending = False

    def set_components(
        self, components: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Set all aircraft components and render."""
        self._components = components
        for name in components:
            if name not in self._visibility:
                self._visibility[name] = True
        self._auto_camera()
        self._render()

    def update_component(
        self, name: str, verts: np.ndarray, faces: np.ndarray
    ) -> None:
        """Update a single component and re-render."""
        self._components[name] = (verts, faces)
        if name not in self._visibility:
            self._visibility[name] = True
        self._render()

    def set_visibility(self, name: str, visible: bool) -> None:
        """Toggle component visibility."""
        self._visibility[name] = visible
        self._render()

    def set_render_mode(self, mode: str) -> None:
        """Set render mode: 'solid', 'wireframe', or 'solid_wire'."""
        self._render_mode = mode
        self._render()

    def set_background(self, theme: str) -> None:
        """Set background: 'dark' or 'light'."""
        self._bg_color = theme
        bg = "#1e1e2e" if theme == "dark" else "#f0f0f0"
        self._canvas.configure(bg=bg)
        self._render()

    def set_view(self, preset: str) -> None:
        """Set camera to a preset view."""
        presets = {
            "top": (0.0, 90.0),
            "front": (0.0, 0.0),
            "side": (-90.0, 0.0),
            "iso": (-135.0, 25.0),
            "rear": (180.0, 10.0),
        }
        if preset in presets:
            self._cam_azimuth, self._cam_elevation = presets[preset]
            self._pan_x = 0.0
            self._pan_y = 0.0
            self._render()

    def reset_camera(self) -> None:
        """Reset camera to default position."""
        self._auto_camera()
        self._render()

    def _auto_camera(self) -> None:
        """Compute camera distance and focal point from component bounds."""
        all_pts = []
        for name, (v, f) in self._components.items():
            if len(v) > 0:
                all_pts.append(v)
        if all_pts:
            pts = np.vstack(all_pts)
            self._cam_focal = pts.mean(axis=0)
            extent = np.ptp(pts, axis=0).max()
            self._cam_distance = extent * 1.8
        self._cam_azimuth = -135.0
        self._cam_elevation = 25.0
        self._pan_x = 0.0
        self._pan_y = 0.0

    def _render(self) -> None:
        """Render scene off-screen and display in canvas."""
        if self._render_pending:
            return
        self._render_pending = True
        self._parent.after(16, self._do_render)  # ~60fps cap

    def _do_render(self) -> None:
        """Actually perform the render."""
        self._render_pending = False

        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w < 10 or h < 10:
            return

        try:
            self._render_vtk(w, h)
        except Exception as exc:
            # Fallback: show error text on canvas
            self._canvas.delete("all")
            self._canvas.create_text(
                w // 2, h // 2,
                text=f"Render hatasi: {exc}",
                fill="red", font=("TkDefaultFont", 10),
            )

    def _render_vtk(self, w: int, h: int) -> None:
        """Render with PyVista off-screen."""
        bg = (0.118, 0.118, 0.18) if self._bg_color == "dark" else (0.94, 0.94, 0.94)

        pl = pv.Plotter(off_screen=True, window_size=(w, h))
        pl.set_background(bg)

        has_meshes = False
        for name, (verts, faces) in self._components.items():
            if not self._visibility.get(name, True):
                continue
            if len(verts) == 0 or len(faces) == 0:
                continue

            mesh = _numpy_to_polydata(verts, faces)
            if mesh.n_points == 0:
                continue

            color = COMPONENT_COLORS.get(name, (0.5, 0.5, 0.5))
            has_meshes = True

            if self._render_mode == "wireframe":
                pl.add_mesh(mesh, color=color, style="wireframe", line_width=1)
            elif self._render_mode == "solid_wire":
                pl.add_mesh(mesh, color=color, opacity=0.9,
                            smooth_shading=True, show_edges=True,
                            edge_color=(0.15, 0.15, 0.15), line_width=0.3)
            else:  # solid
                pl.add_mesh(mesh, color=color, opacity=0.95,
                            smooth_shading=True)

        if not has_meshes:
            pl.close()
            self._canvas.delete("all")
            fg = "#aaa" if self._bg_color == "dark" else "#555"
            self._canvas.create_text(
                w // 2, h // 2,
                text="Henuz geometri olusturulmadi.\n'Olustur' butonuna basin.",
                fill=fg, font=("TkDefaultFont", 11), justify="center",
            )
            return

        # Camera setup
        az_rad = np.radians(self._cam_azimuth)
        el_rad = np.radians(self._cam_elevation)
        d = self._cam_distance

        cam_x = self._cam_focal[0] + d * np.cos(el_rad) * np.cos(az_rad) + self._pan_x
        cam_y = self._cam_focal[1] + d * np.cos(el_rad) * np.sin(az_rad) + self._pan_y
        cam_z = self._cam_focal[2] + d * np.sin(el_rad)

        focal = self._cam_focal.copy()
        focal[0] += self._pan_x
        focal[1] += self._pan_y

        pl.camera.position = (cam_x, cam_y, cam_z)
        pl.camera.focal_point = tuple(focal)
        pl.camera.up = (0, 0, 1)

        # Render to numpy image
        img = pl.screenshot(return_img=True)
        pl.close()

        # Convert numpy image to Tk PhotoImage
        self._display_image(img)

    def _display_image(self, img: np.ndarray) -> None:
        """Display a numpy RGB image on the Tkinter canvas."""
        h, w = img.shape[:2]

        # PPM format for tk.PhotoImage (no PIL dependency needed)
        header = f"P6\n{w} {h}\n255\n".encode()
        data = header + img[:, :, :3].tobytes()

        self._photo_image = tk.PhotoImage(data=data, width=w, height=h)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo_image)

    # -- Mouse interaction --
    def _on_mouse_press(self, event: Any) -> None:
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

    def _on_mouse_drag_rotate(self, event: Any) -> None:
        dx = event.x - self._last_mouse_x
        dy = event.y - self._last_mouse_y
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

        self._cam_azimuth += dx * 0.5
        self._cam_elevation = max(-89, min(89, self._cam_elevation + dy * 0.5))
        self._render()

    def _on_mouse_drag_pan(self, event: Any) -> None:
        dx = event.x - self._last_mouse_x
        dy = event.y - self._last_mouse_y
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

        scale = self._cam_distance * 0.002
        self._pan_x -= dx * scale
        self._pan_y += dy * scale
        self._render()

    def _on_mouse_scroll(self, event: Any) -> None:
        if event.num == 4:
            factor = 0.9
        elif event.num == 5:
            factor = 1.1
        elif event.delta > 0:
            factor = 0.9
        else:
            factor = 1.1

        self._cam_distance = max(1.0, self._cam_distance * factor)
        self._render()

    def _on_resize(self, _event: Any) -> None:
        if self._components:
            self._render()

    def get_mesh_stats(self) -> dict[str, tuple[int, int]]:
        """Return vertex/face counts per component."""
        stats = {}
        for name, (v, f) in self._components.items():
            stats[name] = (len(v), len(f))
        return stats

    def get_total_stats(self) -> tuple[int, int]:
        """Return total vertex and face counts."""
        tv = sum(len(v) for v, _ in self._components.values())
        tf = sum(len(f) for _, f in self._components.values())
        return tv, tf


# ---------------------------------------------------------------------------
# Matplotlib fallback (kept for environments without VTK)
# ---------------------------------------------------------------------------
_MPL_COLORS = {
    "fuselage": (0.7, 0.7, 0.75, 0.6),
    "wing_right": (0.2, 0.4, 0.8, 0.7),
    "wing_left": (0.2, 0.4, 0.8, 0.7),
    "vertical_stabilizer": (0.3, 0.7, 0.3, 0.7),
    "intake_right": (0.8, 0.3, 0.2, 0.7),
    "intake_left": (0.8, 0.3, 0.2, 0.7),
    "bld_right": (0.6, 0.6, 0.6, 0.7),
    "bld_left": (0.6, 0.6, 0.6, 0.7),
    "exhaust": (0.9, 0.6, 0.1, 0.7),
    "wing_root_fairing_right": (0.5, 0.5, 0.8, 0.7),
    "wing_root_fairing_left": (0.5, 0.5, 0.8, 0.7),
    "strake_right": (0.4, 0.55, 0.85, 0.7),
    "strake_left": (0.4, 0.55, 0.85, 0.7),
}


def plot_aircraft_3d(
    components: dict[str, tuple[np.ndarray, np.ndarray]],
    ax: Any = None,
    title: str = "Fighter Aircraft",
    max_triangles_per_component: int = 2000,
) -> Any:
    """Plot the aircraft in 3D using Matplotlib (fallback).

    Parameters
    ----------
    components : dict
        Mapping of component name to (vertices, faces).
    ax : matplotlib Axes3D, optional
        Existing 3D axes to plot on.
    title : str
        Plot title.
    max_triangles_per_component : int
        Limit triangles per component for performance.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    all_pts = []

    for name, (verts, faces) in components.items():
        if len(verts) == 0 or len(faces) == 0:
            continue

        color = _MPL_COLORS.get(name, (0.5, 0.5, 0.5, 0.5))

        if len(faces) > max_triangles_per_component:
            idx = np.linspace(0, len(faces) - 1, max_triangles_per_component, dtype=int)
            faces_sub = faces[idx]
        else:
            faces_sub = faces

        triangles = []
        for face in faces_sub:
            if np.all(face < len(verts)):
                tri = verts[face]
                triangles.append(tri)

        if triangles:
            poly = Poly3DCollection(
                triangles,
                alpha=color[3],
                facecolor=color[:3],
                edgecolor=(0.2, 0.2, 0.2, 0.15),
                linewidth=0.3,
            )
            ax.add_collection3d(poly)
            all_pts.append(verts)

    if all_pts:
        all_pts = np.vstack(all_pts)
        max_range = np.ptp(all_pts, axis=0).max() / 2.0
        mid = np.mean(all_pts, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel("X (m) - Flight direction")
    ax.set_ylabel("Y (m) - Spanwise")
    ax.set_zlabel("Z (m) - Vertical")
    ax.set_title(title)
    ax.view_init(elev=25, azim=-135)

    return fig


def create_embedded_canvas(parent_frame, components, figsize=(8, 6)):
    """Create a Matplotlib canvas embedded in a Tkinter frame (fallback)."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    plot_aircraft_3d(components, ax=ax)

    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    toolbar = NavigationToolbar2Tk(canvas, parent_frame)
    toolbar.update()

    return fig, canvas, toolbar
