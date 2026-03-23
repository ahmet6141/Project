"""Static stability estimation for fighter aircraft."""

from __future__ import annotations

import math
from dataclasses import dataclass

from fightercad.parameters import AircraftParams


@dataclass
class StabilityResult:
    """Static stability analysis results."""
    x_ac: float  # Aerodynamic center position from nose (m)
    x_cg: float  # Center of gravity position from nose (m)
    mac: float  # Mean aerodynamic chord (m)
    static_margin_pct: float  # Static margin as % MAC (positive = stable)
    neutral_point: float  # Neutral point position from nose (m)


def compute_mac(params: AircraftParams) -> float:
    """Compute mean aerodynamic chord (MAC) of the wing."""
    S = params.wing.area_m2
    span = params.wing.span_m
    lam = params.wing.taper_ratio
    c_root = 2.0 * S / (span * (1.0 + lam))
    mac = c_root * (2.0 / 3.0) * (1 + lam + lam ** 2) / (1 + lam)
    return mac


def compute_stability(params: AircraftParams, mach: float) -> StabilityResult:
    """Compute static longitudinal stability.

    Uses component buildup method for aerodynamic center estimation.
    """
    p = params
    L = p.fuselage.length_m
    S_ref = p.wing.area_m2

    mac = compute_mac(p)
    span = p.wing.span_m
    c_root = 2.0 * S_ref / (span * (1.0 + p.wing.taper_ratio))

    # Wing aerodynamic center position
    # For subsonic: ~25% MAC; for supersonic: shifts aft
    if mach < 1.0:
        x_ac_wing_local = 0.25 * mac
    else:
        # AC shifts to ~40-50% MAC at supersonic speeds
        x_ac_wing_local = (0.25 + 0.20 * min(mach - 1.0, 1.0)) * mac

    # Wing root LE position on fuselage
    wing_x = L * p.wing.wing_station_pct
    # MAC position (spanwise location of MAC)
    y_mac = (span / 6.0) * (1 + 2 * p.wing.taper_ratio) / (1 + p.wing.taper_ratio)
    sweep_rad = math.radians(p.wing.leading_edge_sweep_deg)
    x_mac_le = wing_x + y_mac * math.tan(sweep_rad)
    x_ac_wing = x_mac_le + x_ac_wing_local

    # Vertical stabilizer contribution (stabilizing)
    stab_x = L * 0.82  # stabilizer LE position
    stab_arm = stab_x - x_ac_wing  # moment arm from wing AC
    V_vstab = (p.vertical_stabilizer.area_m2 * stab_arm) / (S_ref * mac)

    # Fuselage destabilizing contribution
    # Volume coefficient
    S_fuse_side = L * p.fuselage.max_diameter_m * 0.8
    fuse_shift = -0.05 * S_fuse_side / S_ref  # destabilizing, forward shift

    # Overall aerodynamic center
    x_ac = x_ac_wing + fuse_shift * mac

    # Neutral point (AC of complete aircraft including tail)
    x_np = x_ac + V_vstab * mac * 0.3  # simplified tail contribution

    # CG estimation (rough, ~35% of fuselage length from nose for fighters)
    x_cg = L * 0.35

    # Static margin
    static_margin = (x_np - x_cg) / mac * 100.0  # percent MAC

    return StabilityResult(
        x_ac=x_ac,
        x_cg=x_cg,
        mac=mac,
        static_margin_pct=static_margin,
        neutral_point=x_np,
    )
