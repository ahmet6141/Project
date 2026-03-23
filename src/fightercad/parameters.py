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
    aircraft_type: Literal["fighter", "ucav"] = "fighter"
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
    # Saw-tooth (serrated) trailing edge for stealth
    sawtooth_te_enabled: bool = False
    sawtooth_depth_mm: float = 40.0  # tooth depth
    sawtooth_count: int = 8  # number of teeth per half-wing
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
    # V-tail configuration
    v_tail: bool = False  # True = dual canted fins (V-tail)
    v_tail_cant_deg: float = 35.0  # outward cant angle for each V-tail fin
    # Tailless option
    tailless: bool = False  # True = no vertical stabilizer at all


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


# ---------------------------------------------------------------------------
# UCAV preset factory
# ---------------------------------------------------------------------------

def create_ucav_params() -> AircraftParams:
    """Create professional-grade delta-wing UCAV parameter set.

    Based on nEUROn/X-47B/Taranis class UCAV design principles:
    - Blended wing-body, no cockpit
    - High LE sweep (55-65°), low aspect ratio
    - Dorsal intake, V-tail or tailless
    - Stealth features (saw-tooth TE, blunt TE)
    """
    return AircraftParams(
        meta=MetaParams(
            name="Delta-Wing UCAV",
            aircraft_type="ucav",
            design_mach=0.9,
            design_altitude_m=10000.0,
        ),
        fuselage=FuselageParams(
            length_m=11.0,
            max_diameter_m=1.4,
            nose_fineness_ratio=3.0,
            nose_profile="haack",
            haack_C=0.333,  # LV-Haack for smoother nose
            aft_taper_ratio=0.45,
            cockpit_station_pct=0.15,
            cross_section="elliptical",
            cross_section_aspect=1.25,  # wider than tall
            cylindrical_end_pct=0.65,  # shorter cylindrical section
            aft_taper_power=2.5,  # smoother tail contour
            nose_body_blend_pct=0.06,
            tail_closure_radius_pct=0.20,
            cross_section_aspect_nose=1.1,
            cross_section_aspect_tail=1.5,  # flat tail for nozzle
            canopy_enabled=False,  # no cockpit
            canopy_height_mm=0.0,
            canopy_length_pct=0.0,
            dorsal_spine_height_mm=0.0,  # smooth dorsal
        ),
        wing=WingParams(
            planform="cropped_delta",
            span_m=12.5,  # wide span for BWB
            area_m2=55.0,  # large wing area (BWB)
            aspect_ratio=2.8,
            taper_ratio=0.06,  # very tapered
            leading_edge_sweep_deg=55.0,
            trailing_edge_sweep_deg=25.0,  # positive TE sweep aligned toward LE angle
            dihedral_deg=-2.0,
            twist_deg=-3.0,
            root_airfoil="naca64a004",
            tip_airfoil="biconvex_3",
            leading_edge_radius_mm=3.0,  # slightly rounded LE
            wing_station_pct=0.35,  # forward wing position (BWB)
            incidence_deg=2.0,
            thickness_to_chord_root=0.12,  # thick root for BWB structure + payload
            thickness_to_chord_tip=0.09,  # moderate tip for wave drag balance
            camber_root_pct=1.5,  # slight camber for subsonic efficiency
            camber_tip_pct=0.0,
            te_thickness_mm=1.0,  # blunt TE for stealth
            sawtooth_te_enabled=True,  # serrated TE
            sawtooth_depth_mm=60.0,  # sized for X-band wavelength (~3cm)
            sawtooth_count=12,
            inner_panel_span_pct=0.30,
            inner_panel_sweep_deg=60.0,
        ),
        blending=BlendingParams(
            root_fillet_radius_mm=80.0,  # large fillet for BWB
            leading_edge_fillet_mm=30.0,
            trailing_edge_fillet_mm=20.0,
            blending_mode="smooth",
            strake_length_m=1.2,  # LEX for vortex lift
            strake_sweep_deg=72.0,
            fairing_width_mm=200.0,  # wide fairing for BWB blend
            stabilizer_root_fillet_mm=20.0,
            intake_fuselage_fillet_mm=25.0,
        ),
        vertical_stabilizer=VerticalStabilizerParams(
            area_m2=2.5,  # small V-tail
            aspect_ratio=1.0,
            sweep_deg=45.0,
            taper_ratio=0.30,
            cant_deg=0.0,
            v_tail=True,  # dual V-tail fins
            v_tail_cant_deg=40.0,
            tailless=False,
        ),
        control_surfaces=ControlSurfaceParams(
            elevon_chord_pct=0.25,  # larger elevons for pitch+roll
            aileron_chord_pct=0.20,
            elevon_span_pct=0.70,  # wide elevon coverage
            aileron_span_pct=0.40,
            deflection_deg=0.0,
        ),
        intake=IntakeParams(
            intake_type="dorsal",  # top-mounted for stealth
            intake_count=1,
            capture_area_m2=0.35,
            ramp_angle_deg=5.0,
            lip_sweep_deg=35.0,
            station_pct=0.25,
            boundary_layer_diverter_mm=60.0,
            capture_aspect_ratio=2.0,  # wide and flat
            duct_length_m=3.0,
            lip_radius_mm=20.0,
        ),
        exhaust=ExhaustParams(
            nozzle_type="convergent",  # simpler nozzle for subsonic UCAV
            exit_diameter_m=0.60,
            throat_diameter_m=0.55,
            nozzle_length_m=0.9,
        ),
        area_rule=AreaRuleParams(
            enabled=True,
            target_mach=0.9,
            waist_station_pct=0.50,
            smoothing_iterations=8,
        ),
        internal_structure=InternalStructureParams(
            engine_bay_station_pct=0.55,
            engine_bay_length_pct=0.20,
            fuel_tank_volume_pct=0.35,
            weapons_bay_enabled=True,
        ),
    )
