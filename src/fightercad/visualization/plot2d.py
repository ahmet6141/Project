"""2D plotting for aerodynamic analysis results.

Provides embedded Matplotlib plots for the GUI analysis panel:
drag polar, area distribution, Mach sweep, component breakdown,
planform outline, cross-section evolution, and side-view silhouette.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def plot_drag_polar(
    CL_range: np.ndarray,
    CD_values: np.ndarray,
    CL_design: float | None = None,
    ax: Any = None,
) -> Any:
    """Plot drag polar (CL vs CD).

    Parameters
    ----------
    CL_range : np.ndarray
        Array of CL values.
    CD_values : np.ndarray
        Corresponding CD values.
    CL_design : float, optional
        Design CL to highlight.
    ax : matplotlib Axes, optional
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    ax.plot(CD_values, CL_range, "b-", linewidth=2, label="Drag Polar")
    if CL_design is not None:
        idx = np.argmin(np.abs(CL_range - CL_design))
        ax.plot(CD_values[idx], CL_range[idx], "ro", markersize=8, label=f"Design CL={CL_design:.3f}")

    ax.set_xlabel("CD")
    ax.set_ylabel("CL")
    ax.set_title("Drag Polar")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_area_distribution(
    x: np.ndarray,
    actual_area: np.ndarray,
    ideal_area: np.ndarray,
    fuselage_area: np.ndarray | None = None,
    ax: Any = None,
) -> Any:
    """Plot cross-sectional area distribution vs Sears-Haack ideal.

    Parameters
    ----------
    x : np.ndarray
        Axial stations.
    actual_area : np.ndarray
        Total actual area.
    ideal_area : np.ndarray
        Ideal Sears-Haack area.
    fuselage_area : np.ndarray, optional
        Fuselage-only area.
    ax : matplotlib Axes, optional
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    ax.plot(x, actual_area, "b-", linewidth=2, label="Total (actual)")
    ax.plot(x, ideal_area, "r--", linewidth=1.5, label="Sears-Haack (ideal)")
    if fuselage_area is not None:
        ax.plot(x, fuselage_area, "g:", linewidth=1, label="Fuselage only")
    ax.fill_between(x, actual_area, ideal_area, alpha=0.15, color="red")

    ax.set_xlabel("X station (m)")
    ax.set_ylabel("Cross-sectional area (m²)")
    ax.set_title("Area Distribution (Area Rule)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_mach_sweep(
    mach_range: np.ndarray,
    LD_values: np.ndarray,
    CD_values: np.ndarray,
    mach_design: float | None = None,
    ax: Any = None,
) -> Any:
    """Plot L/D and CD vs Mach number.

    Parameters
    ----------
    mach_range : np.ndarray
    LD_values : np.ndarray
    CD_values : np.ndarray
    mach_design : float, optional
    ax : matplotlib Axes, optional
    """
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(5, 4))
    else:
        ax1 = ax
        fig = ax1.get_figure()

    color1 = "tab:blue"
    ax1.plot(mach_range, LD_values, "-", color=color1, linewidth=2, label="L/D")
    ax1.set_xlabel("Mach Number")
    ax1.set_ylabel("L/D", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.plot(mach_range, CD_values, "--", color=color2, linewidth=1.5, label="CD")
    ax2.set_ylabel("CD", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    if mach_design is not None:
        ax1.axvline(x=mach_design, color="gray", linestyle=":", alpha=0.7, label=f"Design M={mach_design}")

    ax1.set_title("Mach Sweep")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drag_breakdown(
    labels: list[str],
    values: list[float],
    ax: Any = None,
) -> Any:
    """Plot component drag breakdown as horizontal bar chart.

    Parameters
    ----------
    labels : list[str]
        Component names.
    values : list[float]
        Drag coefficient contributions.
    ax : matplotlib Axes, optional
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    colors = ["#4a90d9", "#50b050", "#d94a4a", "#d9a04a", "#888888"]
    total = sum(values) if sum(values) > 0 else 1.0
    pcts = [v / total * 100 for v in values]

    bars = ax.barh(labels, values, color=colors[:len(labels)], height=0.6)

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_width() + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("CD contribution")
    ax.set_title("Component Drag Breakdown")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_planform(
    assembler: Any,
    ax: Any = None,
) -> Any:
    """Plot top-view planform outline from assembled components.

    Shows fuselage body, wing LE/TE, stabilizers, and intake.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    colors = {
        "fuselage": "#4a90d9",
        "wing": "#50b050",
        "vtail": "#d94a4a",
        "intake": "#d9a04a",
        "exhaust": "#888888",
        "other": "#aaaaaa",
    }

    for name, (v, f) in assembler.components.items():
        if len(v) == 0:
            continue
        # Determine color
        if "fuselage" in name:
            c = colors["fuselage"]
        elif "wing" in name or "aileron" in name or "elevon" in name:
            c = colors["wing"]
        elif "vtail" in name or "stab" in name:
            c = colors["vtail"]
        elif "intake" in name:
            c = colors["intake"]
        elif "exhaust" in name:
            c = colors["exhaust"]
        else:
            c = colors["other"]

        # Plot convex hull projection (top view: x vs y)
        ax.scatter(v[:, 0], v[:, 1], s=0.1, c=c, alpha=0.3)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Planform (Top View)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cross_sections(
    fuselage_sections: list,
    stations_pct: list[float] | None = None,
    ax: Any = None,
) -> Any:
    """Plot fuselage cross-sections at several axial stations.

    Parameters
    ----------
    fuselage_sections : list[FuselageSection]
    stations_pct : list[float], optional
        Fractions of fuselage length to plot (default: [0.1, 0.25, 0.45, 0.6, 0.8]).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    if not fuselage_sections:
        return fig

    if stations_pct is None:
        stations_pct = [0.10, 0.25, 0.45, 0.60, 0.80]

    L = fuselage_sections[-1].x
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(stations_pct)))

    for i, pct in enumerate(stations_pct):
        target_x = L * pct
        # Find closest section
        best_sec = min(fuselage_sections, key=lambda s: abs(s.x - target_x))
        pts = best_sec.points
        y = pts[:, 1]
        z = pts[:, 2]
        # Close the loop
        y = np.append(y, y[0])
        z = np.append(z, z[0])
        ax.plot(y, z, color=colors[i], linewidth=1.5,
                label=f"x={best_sec.x:.1f}m ({pct*100:.0f}%)")

    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Cross-Section Evolution")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_side_view(
    assembler: Any,
    ax: Any = None,
) -> Any:
    """Plot side-view silhouette of the aircraft."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.get_figure()

    for name, (v, f) in assembler.components.items():
        if len(v) == 0:
            continue
        if "fuselage" in name:
            c = "#4a90d9"
        elif "wing" in name:
            c = "#50b050"
        else:
            c = "#aaaaaa"
        ax.scatter(v[:, 0], v[:, 2], s=0.1, c=c, alpha=0.3)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Side View")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
