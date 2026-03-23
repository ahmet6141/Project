"""Sears-Haack area rule analysis and fuselage waisting correction.

The area rule states that for minimum wave drag, the total cross-sectional
area distribution A(x) should follow a smooth Sears-Haack distribution.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import UnivariateSpline


def ideal_sears_haack_area(x: np.ndarray, length: float, max_area: float) -> np.ndarray:
    """Compute the ideal Sears-Haack area distribution.

    Parameters
    ----------
    x : np.ndarray
        Axial stations.
    length : float
        Total aircraft length.
    max_area : float
        Maximum cross-sectional area.

    Returns
    -------
    np.ndarray
        Ideal area at each station.
    """
    x_norm = np.clip(x / length, 0, 1)
    theta = np.arccos(1.0 - 2.0 * x_norm)
    # Sears-Haack area distribution
    A = max_area * (theta - np.sin(2.0 * theta) / 2.0) / math.pi
    return np.nan_to_num(A, 0.0)


def compute_wave_drag_coefficient(
    x: np.ndarray,
    area: np.ndarray,
    s_ref: float,
) -> float:
    """Estimate wave drag coefficient from area distribution.

    Uses the relation: CDw ∝ integral of A''(x)^2 dx

    Parameters
    ----------
    x : np.ndarray
        Axial stations.
    area : np.ndarray
        Cross-sectional area at each station.
    s_ref : float
        Reference wing area.

    Returns
    -------
    float
        Estimated wave drag coefficient.
    """
    L = x[-1] - x[0]
    if L <= 0 or s_ref <= 0:
        return 0.0

    # Fit a smooth spline to the area distribution
    spline = UnivariateSpline(x, area, s=0.001)
    # Second derivative
    x_fine = np.linspace(x[0], x[-1], 200)
    A_pp = spline.derivative(n=2)(x_fine)

    # CDw = -1/(2π) * integral of A''(x1)*A''(x2)*ln|x1-x2| dx1 dx2
    # Simplified: CDw ≈ (1/S_ref) * integral(A''(x)^2) dx / (4π)
    dx = x_fine[1] - x_fine[0]
    integral = np.sum(A_pp**2) * dx
    CDw = integral / (4.0 * math.pi * s_ref)
    return CDw


def compute_waisting_correction(
    x: np.ndarray,
    actual_area: np.ndarray,
    target_mach: float = 1.2,
    smoothing: int = 5,
) -> np.ndarray:
    """Compute fuselage radius correction for area-rule compliance.

    Parameters
    ----------
    x : np.ndarray
        Axial stations.
    actual_area : np.ndarray
        Current total cross-sectional area distribution.
    target_mach : float
        Design Mach number for area rule optimization.
    smoothing : int
        Number of smoothing iterations.

    Returns
    -------
    np.ndarray
        Radius delta (meters) to apply to the fuselage at each station.
        Negative values = waisting (reduce radius), positive = bulging.
    """
    L = x[-1] - x[0]
    max_area = np.max(actual_area)
    ideal = ideal_sears_haack_area(x, L, max_area)

    # Area difference: positive means we have too much area (need waisting)
    delta_area = actual_area - ideal

    # Convert area delta to radius delta: delta_r = delta_A / (2πr)
    # Use current average radius estimate (supports elliptical cross-sections
    # by using the geometric mean radius: r = sqrt(A/π))
    r_avg = np.sqrt(np.maximum(actual_area, 0.0) / math.pi + 1e-6)
    delta_r = -delta_area / (2.0 * math.pi * r_avg + 1e-6)

    # Smooth the correction
    for _ in range(smoothing):
        delta_r = np.convolve(delta_r, [0.25, 0.5, 0.25], mode="same")

    # Limit correction magnitude to ±30% of local radius
    max_corr = 0.3 * r_avg
    delta_r = np.clip(delta_r, -max_corr, max_corr)

    return delta_r


class AreaRuleAnalyzer:
    """Analyze and optimize the cross-sectional area distribution."""

    def __init__(self, target_mach: float = 1.2, smoothing: int = 5):
        self.target_mach = target_mach
        self.smoothing = smoothing
        self.x: np.ndarray | None = None
        self.actual_area: np.ndarray | None = None
        self.ideal_area: np.ndarray | None = None
        self.cdw: float = 0.0

    def analyze(
        self,
        x: np.ndarray,
        fuselage_area: np.ndarray,
        wing_area: np.ndarray | None = None,
        other_area: np.ndarray | None = None,
        s_ref: float = 28.0,
    ) -> dict:
        """Run area-rule analysis.

        Returns a dict with actual_area, ideal_area, CDw, and correction.
        """
        total = fuselage_area.copy()
        if wing_area is not None:
            total += wing_area
        if other_area is not None:
            total += other_area

        self.x = x
        self.actual_area = total
        self.ideal_area = ideal_sears_haack_area(x, x[-1] - x[0], np.max(total))
        self.cdw = compute_wave_drag_coefficient(x, total, s_ref)

        correction = compute_waisting_correction(x, total, self.target_mach, self.smoothing)

        return {
            "x": x,
            "actual_area": total,
            "ideal_area": self.ideal_area,
            "fuselage_area": fuselage_area,
            "CDw": self.cdw,
            "correction": correction,
        }
