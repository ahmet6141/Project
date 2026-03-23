"""Dataclass definitions for all aircraft design parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Meta / flight condition
# ---------------------------------------------------------------------------
@dataclass
class MetaParams:
    name: str = "Baseline Supersonic Fighter"
    design_mach: float = 1.6
    design_altitude_m: float = 10000.0


# ---------------------------------------------------------------------------
# Fuselage
# ---------------------------------------------------------------------------
@dataclass
class FuselageParams:
    length_m: float = 15.0
    max_diameter_m: float = 1.8
    nose_fineness_ratio: float = 3.5
    nose_profile: Literal["haack", "conical", "ogive", "power_series"] = "haack"
    haack_C: float = 0.0  # 0=Sears-Haack, 0.333=LV-Haack, 1.0=Von Karman
    aft_taper_ratio: float = 0.6
    cockpit_station_pct: float = 0.15
    cross_section: Literal["circular", "elliptical", "rounded_rect"] = "elliptical"
    cross_section_aspect: float = 1.15  # width / height


# ---------------------------------------------------------------------------
# Wing
# ---------------------------------------------------------------------------
@dataclass
class WingParams:
    planform: Literal["delta", "cropped_delta", "double_delta"] = "cropped_delta"
    area_m2: float = 28.0
    aspect_ratio: float = 2.2
    taper_ratio: float = 0.10
    leading_edge_sweep_deg: float = 55.0
    trailing_edge_sweep_deg: float = -5.0
    dihedral_deg: float = -3.0  # negative = anhedral
    twist_deg: float = -2.0  # washout at tip
    root_airfoil: Literal[
        "biconvex_3", "biconvex_4", "biconvex_5",
        "diamond_3", "diamond_4", "diamond_5",
        "naca64a004",
    ] = "biconvex_5"
    tip_airfoil: Literal[
        "biconvex_3", "biconvex_4", "biconvex_5",
        "diamond_3", "diamond_4", "diamond_5",
        "naca64a004",
    ] = "biconvex_3"
    leading_edge_radius_mm: float = 1.5
    wing_station_pct: float = 0.45
    incidence_deg: float = 1.0
    # High-level thickness parameters
    thickness_to_chord_root: float = 0.05  # root t/c ratio (5%)
    thickness_to_chord_tip: float = 0.03  # tip t/c ratio (3%)


# ---------------------------------------------------------------------------
# Blending – wing/fuselage junction control
# ---------------------------------------------------------------------------
@dataclass
class BlendingParams:
    root_fillet_radius_mm: float = 20.0
    leading_edge_fillet_mm: float = 10.0
    trailing_edge_fillet_mm: float = 10.0
    blending_mode: Literal["sharp", "smooth", "strake"] = "smooth"
    strake_length_m: float = 0.0
    strake_sweep_deg: float = 70.0
    fairing_width_mm: float = 100.0
    stabilizer_root_fillet_mm: float = 30.0
    intake_fuselage_fillet_mm: float = 15.0


# ---------------------------------------------------------------------------
# Vertical stabilizer
# ---------------------------------------------------------------------------
@dataclass
class VerticalStabilizerParams:
    area_m2: float = 5.5
    aspect_ratio: float = 1.2
    sweep_deg: float = 50.0
    taper_ratio: float = 0.25
    cant_deg: float = 0.0  # 0 = vertical


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------
@dataclass
class IntakeParams:
    intake_type: Literal["side_mounted", "chin", "dorsal", "dsi"] = "side_mounted"
    capture_area_m2: float = 0.45
    ramp_angle_deg: float = 7.0
    lip_sweep_deg: float = 30.0
    station_pct: float = 0.30
    boundary_layer_diverter_mm: float = 80.0


# ---------------------------------------------------------------------------
# Exhaust
# ---------------------------------------------------------------------------
@dataclass
class ExhaustParams:
    nozzle_type: Literal["convergent_divergent", "convergent"] = "convergent_divergent"
    exit_diameter_m: float = 0.85
    throat_diameter_m: float = 0.65
    nozzle_length_m: float = 1.2


# ---------------------------------------------------------------------------
# Area rule
# ---------------------------------------------------------------------------
@dataclass
class AreaRuleParams:
    enabled: bool = True
    target_mach: float = 1.2
    waist_station_pct: float = 0.55
    smoothing_iterations: int = 5


# ---------------------------------------------------------------------------
# Future: internal structure placeholders (engine bay, fuel tanks, etc.)
# ---------------------------------------------------------------------------
@dataclass
class InternalStructureParams:
    """Placeholder for future updates: engine mounting bays, engine
    compartment, fuel tanks, weapons bay, landing gear bay, avionics bay."""
    engine_bay_station_pct: float = 0.55
    engine_bay_length_pct: float = 0.25
    fuel_tank_volume_pct: float = 0.30
    weapons_bay_enabled: bool = False
    # Additional internal structure parameters will be added in future versions


# ---------------------------------------------------------------------------
# Top-level aircraft parameters
# ---------------------------------------------------------------------------
@dataclass
class AircraftParams:
    meta: MetaParams = field(default_factory=MetaParams)
    fuselage: FuselageParams = field(default_factory=FuselageParams)
    wing: WingParams = field(default_factory=WingParams)
    blending: BlendingParams = field(default_factory=BlendingParams)
    vertical_stabilizer: VerticalStabilizerParams = field(
        default_factory=VerticalStabilizerParams
    )
    intake: IntakeParams = field(default_factory=IntakeParams)
    exhaust: ExhaustParams = field(default_factory=ExhaustParams)
    area_rule: AreaRuleParams = field(default_factory=AreaRuleParams)
    internal_structure: InternalStructureParams = field(
        default_factory=InternalStructureParams
    )
