"""3D aircraft visualization using Matplotlib's 3D projection.

Provides embedded 3D viewing for the Tkinter GUI without requiring
pythonocc's viewer.  Components are color-coded for clarity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Component color scheme
COMPONENT_COLORS = {
    "fuselage": (0.7, 0.7, 0.75, 0.6),
    "wing_right": (0.2, 0.4, 0.8, 0.7),
    "wing_left": (0.2, 0.4, 0.8, 0.7),
    "vertical_stabilizer": (0.3, 0.7, 0.3, 0.7),
    "intake_right": (0.8, 0.3, 0.2, 0.7),
    "intake_left": (0.8, 0.3, 0.2, 0.7),
    "exhaust": (0.9, 0.6, 0.1, 0.7),
}


def plot_aircraft_3d(
    components: dict[str, tuple[np.ndarray, np.ndarray]],
    ax: Any = None,
    title: str = "Fighter Aircraft",
    max_triangles_per_component: int = 2000,
) -> Any:
    """Plot the aircraft in 3D using Matplotlib.

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
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    all_pts = []

    for name, (verts, faces) in components.items():
        if len(verts) == 0 or len(faces) == 0:
            continue

        color = COMPONENT_COLORS.get(name, (0.5, 0.5, 0.5, 0.5))

        # Subsample faces if too many
        if len(faces) > max_triangles_per_component:
            idx = np.linspace(0, len(faces) - 1, max_triangles_per_component, dtype=int)
            faces_sub = faces[idx]
        else:
            faces_sub = faces

        # Build triangle collection
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
        # Set equal aspect ratio
        max_range = np.ptp(all_pts, axis=0).max() / 2.0
        mid = np.mean(all_pts, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel("X (m) - Flight direction")
    ax.set_ylabel("Y (m) - Spanwise")
    ax.set_zlabel("Z (m) - Vertical")
    ax.set_title(title)

    # Good viewing angle for aircraft
    ax.view_init(elev=25, azim=-135)

    return fig


def create_embedded_canvas(parent_frame, components, figsize=(8, 6)):
    """Create a Matplotlib canvas embedded in a Tkinter frame.

    Parameters
    ----------
    parent_frame : tk.Frame
        Parent Tkinter frame.
    components : dict
        Component mesh data.
    figsize : tuple
        Figure size.

    Returns
    -------
    tuple
        (figure, canvas, toolbar)
    """
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
