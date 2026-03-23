"""Geometric primitive builders: airfoil profiles, curves, transformations.

This module provides pure-math geometry generation using numpy arrays.
When pythonocc-core is available, it also provides OCC wire/shape builders.
All coordinates are in meters unless noted otherwise.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Airfoil coordinate generators  (return Nx2 numpy arrays)
# ---------------------------------------------------------------------------

def biconvex_airfoil(
    chord: float,
    thickness_pct: float,
    le_radius_mm: float = 1.5,
    num_points: int = 80,
) -> np.ndarray:
    """Generate a biconvex (symmetric lens) supersonic airfoil.

    y = ±2·t_max·(x/c)·(1 - x/c)

    Parameters
    ----------
    chord : float
        Chord length in meters.
    thickness_pct : float
        Maximum thickness as percentage of chord (e.g. 5 for 5%).
    le_radius_mm : float
        Leading-edge radius in mm.  A small circular arc is blended at x=0.
    num_points : int
        Points per surface (upper + lower).

    Returns
    -------
    np.ndarray
        (N, 2) closed airfoil coordinates, starting and ending at trailing edge,
        going counter-clockwise (upper surface first, then lower reversed).
    """
    t_max = chord * thickness_pct / 100.0
    x = np.linspace(0, chord, num_points)
    y_half = 2.0 * t_max * (x / chord) * (1.0 - x / chord)

    # Apply tiny LE radius blending for the first few points
    le_r = le_radius_mm / 1000.0
    if le_r > 0:
        blend_zone = min(le_r * 4, chord * 0.05)
        mask = x < blend_zone
        if np.any(mask):
            t_local = x[mask] / blend_zone
            y_half[mask] = y_half[mask] * np.sqrt(t_local + 1e-12)

    upper = np.column_stack([x, y_half])
    lower = np.column_stack([x, -y_half])
    # Close the profile: upper TE -> LE -> lower LE -> TE
    coords = np.vstack([upper, lower[::-1]])
    return coords


def diamond_airfoil(
    chord: float,
    thickness_pct: float,
    peak_x_pct: float = 40.0,
    num_points: int = 80,
) -> np.ndarray:
    """Generate a diamond (double-wedge) supersonic airfoil.

    Parameters
    ----------
    chord : float
        Chord length in meters.
    thickness_pct : float
        Max thickness as percentage of chord.
    peak_x_pct : float
        Chordwise position of max thickness as percentage (e.g. 40).
    num_points : int
        Points per surface.
    """
    t_max = chord * thickness_pct / 100.0
    x_peak = chord * peak_x_pct / 100.0
    half_n = num_points // 2

    x_fwd = np.linspace(0, x_peak, half_n, endpoint=False)
    x_aft = np.linspace(x_peak, chord, num_points - half_n)
    x = np.concatenate([x_fwd, x_aft])

    y_half = np.where(
        x <= x_peak,
        t_max * x / x_peak,
        t_max * (chord - x) / (chord - x_peak),
    )

    upper = np.column_stack([x, y_half])
    lower = np.column_stack([x, -y_half])
    return np.vstack([upper, lower[::-1]])


def naca_4digit_symmetric(
    chord: float,
    thickness_pct: float,
    num_points: int = 80,
) -> np.ndarray:
    """Generate a symmetric NACA 4-digit airfoil (e.g. NACA 0004)."""
    t = thickness_pct / 100.0
    # Cosine spacing for better LE resolution
    beta = np.linspace(0, math.pi, num_points)
    x = chord * 0.5 * (1.0 - np.cos(beta))
    xc = x / chord

    yt = (t / 0.2) * chord * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1015 * xc**4
    )

    upper = np.column_stack([x, yt])
    lower = np.column_stack([x, -yt])
    return np.vstack([upper, lower[::-1]])


def get_airfoil(
    airfoil_type: str,
    chord: float,
    le_radius_mm: float = 1.5,
    num_points: int = 80,
) -> np.ndarray:
    """Get airfoil coordinates by type string.

    Type format: '{family}_{thickness_pct}' e.g. 'biconvex_5', 'diamond_4'.
    Special: 'naca64a004' maps to NACA 0004 thin symmetric.
    """
    if airfoil_type.startswith("biconvex_"):
        tpct = float(airfoil_type.split("_")[1])
        return biconvex_airfoil(chord, tpct, le_radius_mm, num_points)
    elif airfoil_type.startswith("diamond_"):
        tpct = float(airfoil_type.split("_")[1])
        return diamond_airfoil(chord, tpct, num_points=num_points)
    elif airfoil_type.startswith("naca"):
        tpct = float(airfoil_type[-2:]) / 10.0  # e.g. '04' -> 4.0
        return naca_4digit_symmetric(chord, tpct, num_points)
    else:
        # Fallback to biconvex 5%
        return biconvex_airfoil(chord, 5.0, le_radius_mm, num_points)


# ---------------------------------------------------------------------------
# Sears-Haack / nose profile radius distributions
# ---------------------------------------------------------------------------

def sears_haack_radius(
    x_stations: np.ndarray,
    length: float,
    max_radius: float,
    C: float = 0.0,
) -> np.ndarray:
    """Compute Haack-series body radius distribution.

    Parameters
    ----------
    x_stations : np.ndarray
        Axial stations from 0 to length.
    length : float
        Total nose length.
    max_radius : float
        Maximum body radius.
    C : float
        Haack series parameter (0=Sears-Haack, 1=Von Karman).

    Returns
    -------
    np.ndarray
        Radius at each station.
    """
    x_norm = np.clip(x_stations / length, 0, 1)
    theta = np.arccos(1.0 - 2.0 * x_norm)
    r = (max_radius / math.sqrt(math.pi)) * np.sqrt(
        theta - np.sin(2.0 * theta) / 2.0 + C * np.sin(theta) ** 3
    )
    return np.nan_to_num(r, 0.0)


def ogive_radius(
    x_stations: np.ndarray,
    length: float,
    max_radius: float,
) -> np.ndarray:
    """Tangent ogive nose profile."""
    rho = (max_radius**2 + length**2) / (2.0 * max_radius)
    r = np.sqrt(rho**2 - (length - x_stations) ** 2) + max_radius - rho
    return np.clip(r, 0, max_radius)


def conical_radius(
    x_stations: np.ndarray,
    length: float,
    max_radius: float,
) -> np.ndarray:
    """Simple conical nose profile."""
    return max_radius * np.clip(x_stations / length, 0, 1)


def power_series_radius(
    x_stations: np.ndarray,
    length: float,
    max_radius: float,
    n: float = 0.5,
) -> np.ndarray:
    """Power-series nose: r(x) = R * (x/L)^n."""
    return max_radius * np.clip(x_stations / length, 0, 1) ** n


def get_nose_profile(
    profile_type: str,
    x_stations: np.ndarray,
    length: float,
    max_radius: float,
    haack_C: float = 0.0,
) -> np.ndarray:
    """Get nose radius distribution by profile type."""
    if profile_type == "haack":
        return sears_haack_radius(x_stations, length, max_radius, haack_C)
    elif profile_type == "ogive":
        return ogive_radius(x_stations, length, max_radius)
    elif profile_type == "conical":
        return conical_radius(x_stations, length, max_radius)
    elif profile_type == "power_series":
        return power_series_radius(x_stations, length, max_radius)
    else:
        return sears_haack_radius(x_stations, length, max_radius, haack_C)


# ---------------------------------------------------------------------------
# Coordinate transformations
# ---------------------------------------------------------------------------

def translate_2d(coords: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate 2D coordinates."""
    return coords + np.array([dx, dy])


def rotate_2d(coords: np.ndarray, angle_deg: float, center: tuple[float, float] = (0, 0)) -> np.ndarray:
    """Rotate 2D coordinates around a center point."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    cx, cy = center
    shifted = coords - np.array([cx, cy])
    rotated = np.column_stack([
        shifted[:, 0] * c - shifted[:, 1] * s,
        shifted[:, 0] * s + shifted[:, 1] * c,
    ])
    return rotated + np.array([cx, cy])


def scale_2d(coords: np.ndarray, sx: float, sy: float | None = None) -> np.ndarray:
    """Scale 2D coordinates."""
    if sy is None:
        sy = sx
    return coords * np.array([sx, sy])


def airfoil_to_3d(
    coords_2d: np.ndarray,
    y_span: float,
    x_offset: float = 0.0,
    z_offset: float = 0.0,
) -> np.ndarray:
    """Convert 2D airfoil (x,z) to 3D (x, y, z) at a given spanwise station.

    Convention: x=longitudinal (flight direction), y=spanwise, z=vertical.
    """
    n = len(coords_2d)
    pts3d = np.zeros((n, 3))
    pts3d[:, 0] = coords_2d[:, 0] + x_offset  # x (chordwise -> longitudinal)
    pts3d[:, 1] = y_span  # y (spanwise)
    pts3d[:, 2] = coords_2d[:, 1] + z_offset  # z (thickness -> vertical)
    return pts3d
