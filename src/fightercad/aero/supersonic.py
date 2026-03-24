"""Supersonic aerodynamic theory formulas.

Implements Ackeret theory, Prandtl-Glauert correction, and related
supersonic flow relations for thin-wing analysis.
"""

from __future__ import annotations

import math

import numpy as np


def prandtl_glauert_factor(mach: float) -> float:
    """Prandtl-Glauert compressibility factor β = √|1 - M²|.

    For subsonic: β = √(1 - M²)
    For supersonic: β = √(M² - 1)
    """
    return math.sqrt(abs(1.0 - mach ** 2) + 1e-15)


def ackeret_cp(surface_angle_rad: float, mach: float) -> float:
    """Ackeret (linearized supersonic) pressure coefficient.

    Cp = 2·θ / √(M² - 1)

    Parameters
    ----------
    surface_angle_rad : float
        Local surface inclination angle (radians).
    mach : float
        Freestream Mach number (must be > 1).

    Returns
    -------
    float
        Pressure coefficient.
    """
    beta = math.sqrt(max(mach ** 2 - 1.0, 1e-6))
    return 2.0 * surface_angle_rad / beta


def supersonic_lift_curve_slope_2d(mach: float) -> float:
    """2D lift-curve slope for supersonic thin airfoil (per radian).

    CLα = 4 / √(M² - 1)
    """
    beta = math.sqrt(max(mach ** 2 - 1.0, 1e-6))
    return 4.0 / beta


def subsonic_lift_curve_slope_2d(mach: float) -> float:
    """2D lift-curve slope with Prandtl-Glauert correction (per radian).

    CLα = 2π / √(1 - M²)
    """
    beta = math.sqrt(max(1.0 - mach ** 2, 1e-6))
    return 2.0 * math.pi / beta


def finite_wing_lift_curve_slope(
    mach: float,
    aspect_ratio: float,
    sweep_le_deg: float,
) -> float:
    """Lift-curve slope for a finite wing (per radian).

    Uses Diederich's semi-empirical formula which works across
    subsonic and supersonic ranges:

    CLα = (2π·AR) / (2 + √(4 + AR²·β²·(1 + tan²(Λ_c/2)/β²)))

    where β² = |1 - M²| and Λ_c/2 is the half-chord sweep.
    """
    AR = aspect_ratio
    beta2 = abs(1.0 - mach ** 2) + 1e-6
    # Approximate half-chord sweep from LE sweep
    sweep_half = math.radians(sweep_le_deg) * 0.85  # rough approximation
    tan2_sweep = math.tan(sweep_half) ** 2

    numer = 2.0 * math.pi * AR
    denom = 2.0 + math.sqrt(4.0 + AR ** 2 * beta2 * (1.0 + tan2_sweep / beta2))
    return numer / denom


def oblique_shock_beta(mach: float, theta_deg: float, gamma: float = 1.4) -> float | None:
    """Solve the θ-β-M relation for the weak oblique shock angle.

    Parameters
    ----------
    mach : float
        Upstream Mach number.
    theta_deg : float
        Deflection (ramp) angle in degrees.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    float or None
        Shock angle β in degrees, or None if detached shock.
    """
    theta = math.radians(theta_deg)
    if theta <= 0 or mach <= 1.0:
        return None

    M2 = mach ** 2
    # Iterative solution using Newton's method on the θ-β-M relation
    # tan(θ) = 2·cot(β)·(M²·sin²(β) - 1) / (M²·(γ + cos(2β)) + 2)
    beta = math.radians(theta_deg + 10.0)  # initial guess

    for _ in range(50):
        sb = math.sin(beta)
        cb = math.cos(beta)
        sb2 = sb ** 2
        num = 2.0 * (cb / sb) * (M2 * sb2 - 1.0)
        den = M2 * (gamma + math.cos(2.0 * beta)) + 2.0
        tan_theta_calc = num / den
        f = math.tan(theta) - tan_theta_calc

        # Numerical derivative
        dbeta = 1e-6
        sb_d = math.sin(beta + dbeta)
        cb_d = math.cos(beta + dbeta)
        num_d = 2.0 * (cb_d / sb_d) * (M2 * sb_d ** 2 - 1.0)
        den_d = M2 * (gamma + math.cos(2.0 * (beta + dbeta))) + 2.0
        f_d = math.tan(theta) - num_d / den_d
        df = (f_d - f) / dbeta

        if abs(df) < 1e-12:
            break
        beta -= f / df
        beta = max(math.asin(1.0 / mach) + 1e-6, min(beta, math.pi / 2 - 1e-6))

        if abs(f) < 1e-10:
            break

    # Verify it's the weak solution
    if beta > math.pi / 2 or M2 * math.sin(beta) ** 2 < 1.0:
        return None

    return math.degrees(beta)


def mach_wave_angle(mach: float) -> float:
    """Mach wave angle μ = arcsin(1/M) in degrees."""
    if mach <= 1.0:
        return 90.0
    return math.degrees(math.asin(1.0 / mach))
