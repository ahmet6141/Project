"""Lift estimation for supersonic fighter aircraft."""

from __future__ import annotations

import math

from fightercad.parameters import AircraftParams
from fightercad.aero.supersonic import finite_wing_lift_curve_slope


def compute_lift_curve_slope(params: AircraftParams, mach: float) -> float:
    """Compute the wing lift-curve slope CLα (per radian)."""
    S = params.wing.area_m2
    AR = params.wing.span_m ** 2 / S if S > 0 else params.wing.aspect_ratio
    return finite_wing_lift_curve_slope(
        mach,
        AR,
        params.wing.leading_edge_sweep_deg,
    )


def compute_CL_for_level_flight(
    params: AircraftParams,
    mach: float,
    altitude_m: float,
    weight_N: float = 100000.0,
) -> float:
    """Compute CL required for level flight at given condition.

    Default weight ~10 tons (100 kN), typical for loaded fighter.
    """
    from fightercad.aero.atmosphere import dynamic_pressure
    q = dynamic_pressure(mach, altitude_m)
    S = params.wing.area_m2
    if q * S <= 0:
        return 0.0
    return weight_N / (q * S)


def compute_CLmax(params: AircraftParams, mach: float) -> float:
    """Estimate maximum lift coefficient.

    For delta wings at supersonic speeds, CLmax is limited.
    """
    # Subsonic CLmax for delta wings ~1.2-1.6
    # Reduces significantly at supersonic speeds
    if mach < 0.8:
        cl_max = 1.4
    elif mach < 1.0:
        cl_max = 1.4 - 0.4 * ((mach - 0.8) / 0.2)
    else:
        cl_max = 1.0 / math.sqrt(mach)

    # Sweep correction
    cos_sweep = math.cos(math.radians(params.wing.leading_edge_sweep_deg))
    cl_max *= cos_sweep ** 0.5

    return cl_max


def compute_alpha_for_CL(
    params: AircraftParams,
    mach: float,
    CL: float,
) -> float:
    """Compute angle of attack for given CL (degrees)."""
    CLa = compute_lift_curve_slope(params, mach)
    if CLa <= 0:
        return 0.0
    alpha_rad = CL / CLa
    return math.degrees(alpha_rad)
