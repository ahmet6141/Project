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
    cross_section_aspect: float = 1.15  # width / height at mid-body
    # Zone control
    cylindrical_end_pct: float = 0.75  # where cylindrical section ends
    aft_taper_power: float = 2.0  # taper curve exponent (1=linear, 2=quadratic)
    nose_body_blend_pct: float = 0.05  # nose-to-body smoothing zone length
    tail_closure_radius_pct: float = 0.15  # tail end radius (0=pointed, 1=blunt)
    # Variable cross-section aspect
    cross_section_aspect_nose: float = 1.0  # rounder nose
    cross_section_aspect_tail: float = 1.3  # flatter tail
    # Canopy
    canopy_enabled: bool = True
    canopy_height_mm: float = 180.0
    canopy_length_pct: float = 0.12
    # Dorsal spine
    dorsal_spine_height_mm: float = 0.0  # 0=disabled


# ---------------------------------------------------------------------------
# Wing
# ---------------------------------------------------------------------------
@dataclass
class WingParams:
    planform: Literal["delta", "cropped_delta", "double_delta"] = "cropped_delta"
    span_m: float = 7.85  # full wingspan (m), default ≈ √(2.2×28)
    area_m2: float = 28.0
    aspect_ratio: float = 2.2  # kept for backward compat; computed as span²/area in practice
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
    camber_root_pct: float = 0.0  # root camber (% of chord)
    camber_tip_pct: float = 0.0  # tip camber (% of chord)
    te_thickness_mm: float = 0.0  # blunt trailing edge thickness (0=sharp)
    # Inner/outer panel split
    inner_panel_span_pct: float = 0.35  # inner panel ends at 35% half-span
    inner_panel_sweep_deg: float = 60.0  # inner panel LE sweep (steeper)


# ---------------------------------------------------------------------------
# Control surfaces
# ---------------------------------------------------------------------------
@dataclass
class ControlSurfaceParams:
    elevon_chord_pct: float = 0.20  # trailing-edge fraction for elevon (inner panel)
    aileron_chord_pct: float = 0.25  # trailing-edge fraction for aileron (outer panel)
    elevon_span_pct: float = 0.60  # fraction of inner panel span covered by elevon
    aileron_span_pct: float = 0.50  # fraction of outer panel span covered by aileron
    deflection_deg: float = 0.0  # neutral position


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
    intake_type: Literal["side_mounted", "chin", "dorsal", "dsi"] = "chin"
    intake_count: int = 1  # 1 for chin/dorsal, 2 for side_mounted
    capture_area_m2: float = 0.45
    ramp_angle_deg: float = 7.0
    lip_sweep_deg: float = 30.0
    station_pct: float = 0.30
    boundary_layer_diverter_mm: float = 80.0
    capture_aspect_ratio: float = 1.5  # width/height of capture face
    duct_length_m: float = 2.5  # intake duct length
    lip_radius_mm: float = 15.0  # lip rounding radius


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
    control_surfaces: ControlSurfaceParams = field(
        default_factory=ControlSurfaceParams
    )
    intake: IntakeParams = field(default_factory=IntakeParams)
    exhaust: ExhaustParams = field(default_factory=ExhaustParams)
    area_rule: AreaRuleParams = field(default_factory=AreaRuleParams)
    internal_structure: InternalStructureParams = field(
        default_factory=InternalStructureParams
    )
