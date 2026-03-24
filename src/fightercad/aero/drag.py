"""Component drag buildup for supersonic fighter aircraft.

Computes zero-lift drag (CD0), induced drag (CDi), and wave drag (CDw)
using semi-empirical methods suitable for preliminary design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fightercad.parameters import AircraftParams
from fightercad.aero.atmosphere import reynolds_number, dynamic_pressure


@dataclass
class DragResult:
    """Complete drag breakdown."""
    CD0: float  # Zero-lift (friction + form) drag coefficient
    CDi: float  # Induced (lift-dependent) drag coefficient
    CDw: float  # Wave drag coefficient
    CD_total: float  # Total drag coefficient
    CL: float  # Lift coefficient at this condition

    # Component breakdown of CD0
    CD0_fuselage: float = 0.0
    CD0_wing: float = 0.0
    CD0_vstab: float = 0.0
    CD0_intake: float = 0.0
    CD0_other: float = 0.0

    @property
    def LD_ratio(self) -> float:
        """Lift-to-drag ratio."""
        if self.CD_total > 0:
            return self.CL / self.CD_total
        return 0.0


def turbulent_cf(reynolds: float) -> float:
    """Turbulent flat-plate skin friction coefficient (Schlichting)."""
    if reynolds < 1e3:
        return 0.01
    return 0.455 / (math.log10(reynolds) ** 2.58)


def fuselage_form_factor(fineness: float) -> float:
    """Fuselage form factor FF = 1 + 60/f³ + f/400."""
    f = max(fineness, 1.0)
    return 1.0 + 60.0 / f ** 3 + f / 400.0


def wing_form_factor(
    thickness_ratio: float,
    mach: float,
    sweep_deg: float,
    x_max_t: float = 0.4,
) -> float:
    """Wing form factor (Raymer method).

    FF = (1 + 0.6/x_c·(t/c) + 100·(t/c)⁴) · (1.34·M^0.18·cos(Λ)^0.28)
    """
    tc = max(thickness_ratio, 0.01)
    cos_sweep = math.cos(math.radians(sweep_deg))
    ff = (1.0 + 0.6 / max(x_max_t, 0.1) * tc + 100.0 * tc ** 4)
    ff *= 1.34 * max(mach, 0.1) ** 0.18 * max(cos_sweep, 0.1) ** 0.28
    return ff


def compute_drag(
    params: AircraftParams,
    mach: float,
    altitude_m: float,
    CL: float,
    CDw_area_rule: float = 0.0,
) -> DragResult:
    """Compute complete drag breakdown.

    Parameters
    ----------
    params : AircraftParams
        Aircraft configuration.
    mach : float
        Flight Mach number.
    altitude_m : float
        Altitude in meters.
    CL : float
        Lift coefficient.
    CDw_area_rule : float
        Wave drag from area-rule analysis (if available).

    Returns
    -------
    DragResult
        Complete drag breakdown.
    """
    p = params
    S_ref = p.wing.area_m2
    Q = 1.0  # interference factor

    # --- Fuselage ---
    L_fuse = p.fuselage.length_m
    d_fuse = p.fuselage.max_diameter_m
    S_wet_fuse = math.pi * d_fuse * L_fuse * 0.85  # 85% of cylinder
    fineness = L_fuse / d_fuse
    Re_fuse = reynolds_number(mach, altitude_m, L_fuse)
    Cf_fuse = turbulent_cf(Re_fuse)
    FF_fuse = fuselage_form_factor(fineness)
    CD0_fuse = Cf_fuse * FF_fuse * Q * S_wet_fuse / S_ref

    # --- Wing ---
    # Approximate wing wetted area ~ 2.0 * planform area (upper + lower)
    S_wet_wing = 2.0 * S_ref * 1.02  # 2% extra for thickness
    # Mean aerodynamic chord
    span = p.wing.span_m
    c_root = 2.0 * S_ref / (span * (1.0 + p.wing.taper_ratio))
    mac = c_root * (2.0 / 3.0) * (1 + p.wing.taper_ratio + p.wing.taper_ratio ** 2) / (1 + p.wing.taper_ratio)
    Re_wing = reynolds_number(mach, altitude_m, mac)
    Cf_wing = turbulent_cf(Re_wing)
    # Thickness ratio from airfoil name
    tc_pct = _parse_thickness(p.wing.root_airfoil)
    tc = tc_pct / 100.0
    FF_wing = wing_form_factor(tc, mach, p.wing.leading_edge_sweep_deg)
    CD0_wing = Cf_wing * FF_wing * Q * S_wet_wing / S_ref

    # --- Vertical stabilizer ---
    S_wet_vstab = 2.0 * p.vertical_stabilizer.area_m2 * 1.02
    tc_vstab = 0.04  # NACA 64A004
    span_vstab = math.sqrt(p.vertical_stabilizer.aspect_ratio * p.vertical_stabilizer.area_m2)
    c_vstab = p.vertical_stabilizer.area_m2 / span_vstab
    Re_vstab = reynolds_number(mach, altitude_m, c_vstab)
    Cf_vstab = turbulent_cf(Re_vstab)
    FF_vstab = wing_form_factor(tc_vstab, mach, p.vertical_stabilizer.sweep_deg)
    CD0_vstab = Cf_vstab * FF_vstab * Q * S_wet_vstab / S_ref

    # --- Intake ---
    S_wet_intake = 2.0 * p.intake.capture_area_m2 * 4.0  # rough duct wetted area
    Re_intake = reynolds_number(mach, altitude_m, 2.0)
    Cf_intake = turbulent_cf(Re_intake)
    CD0_intake = Cf_intake * 1.3 * S_wet_intake / S_ref  # FF~1.3 for ducts

    # --- Other (exhaust, misc) ---
    CD0_other = 0.002  # Miscellaneous items

    CD0 = CD0_fuse + CD0_wing + CD0_vstab + CD0_intake + CD0_other

    # --- Induced drag ---
    AR = p.wing.span_m ** 2 / S_ref if S_ref > 0 else p.wing.aspect_ratio
    # Oswald efficiency for delta wings (lower than conventional)
    e = 0.75 if p.wing.planform in ("delta", "cropped_delta") else 0.80
    if mach > 1.0:
        e *= 0.85  # Reduction at supersonic speeds
    CDi = CL ** 2 / (math.pi * AR * e) if AR > 0 else 0.0

    # --- Wave drag ---
    CDw = CDw_area_rule
    if CDw <= 0 and mach > 0.8:
        # Rough estimate if area-rule analysis not available
        if mach > 1.0:
            CDw = 0.015 * (mach - 0.8) ** 1.5  # Empirical rise
        else:
            CDw = 0.005 * ((mach - 0.8) / 0.2) ** 2  # Transonic rise

    # Wing volume wave drag contribution
    if mach > 1.0:
        CDw += 4.0 * tc ** 2 / math.sqrt(max(mach ** 2 - 1.0, 0.01))

    CD_total = CD0 + CDi + CDw

    return DragResult(
        CD0=CD0,
        CDi=CDi,
        CDw=CDw,
        CD_total=CD_total,
        CL=CL,
        CD0_fuselage=CD0_fuse,
        CD0_wing=CD0_wing,
        CD0_vstab=CD0_vstab,
        CD0_intake=CD0_intake,
        CD0_other=CD0_other,
    )


def _parse_thickness(airfoil_name: str) -> float:
    """Extract thickness percentage from airfoil name."""
    if "_" in airfoil_name:
        try:
            return float(airfoil_name.split("_")[-1])
        except ValueError:
            pass
    if airfoil_name.startswith("naca"):
        try:
            return float(airfoil_name[-2:])  # e.g. 'naca0012' -> 12.0%
        except ValueError:
            pass
    return 5.0  # default
