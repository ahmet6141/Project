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

    Calibrated to nEUROn-class strike UCAV design (9.5m length, 12.5m span):
    - BWB with smooth wing-body blend, no cockpit/canopy
    - 53° LE sweep cranked delta (SACCON/nEUROn standard)
    - Dorsal S-duct intake shielding engine face from ground radar
    - V-tail at 40° cant for combined pitch/yaw authority
    - Saw-tooth TE with edges aligned to LE sweep for RCS reduction
    - Reflex camber at root for tailless pitch trim (Cm0 > 0)

    Reference aircraft:
    - nEUROn: L=9.5m, span=12.5m, MTOW=7000kg, M=0.8
    - X-47B:  L=11.6m, span=18.9m, MTOW=20215kg, M=0.9+
    - Taranis: L=12.4m, span=10m, MTOW=8000kg, M=0.9
    - SACCON: L=12.3m, span=15.2m, LE sweep=53°
    """
    return AircraftParams(
        meta=MetaParams(
            name="Delta-Wing UCAV",
            aircraft_type="ucav",
            design_mach=0.85,  # high-subsonic cruise (nEUROn/Taranis regime)
            design_altitude_m=12000.0,  # 40,000 ft ceiling typical for strike UCAVs
        ),
        fuselage=FuselageParams(
            length_m=10.0,  # nEUROn=9.5m, Taranis=12.4m — mid-range
            max_diameter_m=1.5,  # ~15% of length (nEUROn body depth ~1.4m)
            nose_fineness_ratio=2.8,  # blunter nose than fighter (stealth chine shaping)
            nose_profile="ogive",  # tangent ogive: smooth curvature for low frontal RCS
            haack_C=0.0,
            aft_taper_ratio=0.35,  # aggressive tail narrowing for nozzle integration
            cockpit_station_pct=0.12,  # avionics bay position (no cockpit)
            cross_section="elliptical",
            cross_section_aspect=1.35,  # wide-flat body for BWB (width/height >1.3)
            cylindrical_end_pct=0.55,  # BWB: short cylindrical, long blend to tail
            aft_taper_power=2.0,  # quadratic taper (smooth, avoids sharp inflection)
            nose_body_blend_pct=0.08,  # wider blend zone for seamless BWB nose merge
            tail_closure_radius_pct=0.25,  # semi-blunt tail for flat exhaust nozzle
            cross_section_aspect_nose=1.15,  # slightly oval nose for chine line
            cross_section_aspect_tail=1.6,  # very flat tail (platypus-style exhaust)
            canopy_enabled=False,  # UCAV: no cockpit
            canopy_height_mm=0.0,
            canopy_length_pct=0.0,
            dorsal_spine_height_mm=0.0,  # smooth upper surface required for dorsal intake
        ),
        wing=WingParams(
            planform="cropped_delta",
            span_m=12.5,  # nEUROn=12.5m, gives L/span≈0.80
            area_m2=48.0,  # wing loading ~145 kg/m² at MTOW 7000kg
            aspect_ratio=3.25,  # AR=span²/area=156/48≈3.25 (SACCON≈3.1)
            taper_ratio=0.08,  # near-pointed tips (nEUROn ≈ 0.05-0.10)
            leading_edge_sweep_deg=53.0,  # SACCON standard, nEUROn ~55°
            trailing_edge_sweep_deg=20.0,  # aligned: TE edges at ~20° (planform alignment)
            dihedral_deg=-1.5,  # slight anhedral (typical BWB UCAV: -1 to -3°)
            twist_deg=-4.0,  # stronger washout for tailless pitch trim
            root_airfoil="naca64a004",  # modified supercritical-type base
            tip_airfoil="biconvex_3",  # thin supersonic section at tips
            leading_edge_radius_mm=4.0,  # rounder LE for subsonic efficiency (not fighter-sharp)
            wing_station_pct=0.30,  # forward wing root (BWB: wing starts early on body)
            incidence_deg=2.5,  # higher incidence for cruise lift without AoA
            thickness_to_chord_root=0.14,  # 14% root t/c (BWB centerbody volume)
            thickness_to_chord_tip=0.09,  # 9% tip (wave drag + structural minimum)
            camber_root_pct=2.0,  # reflex camber at root for Cm0>0 tailless trim
            camber_tip_pct=-0.5,  # negative (reflex) at tip for pitch trim
            te_thickness_mm=1.5,  # blunt TE: 0.5-1% chord for stealth/structural
            sawtooth_te_enabled=True,
            sawtooth_depth_mm=50.0,  # ~1.5× X-band wavelength for effective scattering
            sawtooth_count=14,  # teeth aligned with LE sweep angle
            inner_panel_span_pct=0.28,  # cranked delta break at ~28% semi-span (SACCON: 29%)
            inner_panel_sweep_deg=58.0,  # steeper inboard panel (nEUROn inboard ~55-60°)
        ),
        blending=BlendingParams(
            root_fillet_radius_mm=120.0,  # very large BWB blend radius
            leading_edge_fillet_mm=40.0,  # smooth LE junction (chine blending)
            trailing_edge_fillet_mm=25.0,
            blending_mode="smooth",
            strake_length_m=0.8,  # short LEX blended into forebody chine
            strake_sweep_deg=68.0,  # high sweep, blended (not protruding)
            fairing_width_mm=300.0,  # 300mm wide fairing for BWB blend zone
            stabilizer_root_fillet_mm=25.0,
            intake_fuselage_fillet_mm=30.0,  # smooth intake-body junction
        ),
        vertical_stabilizer=VerticalStabilizerParams(
            area_m2=1.8,  # ~3.75% of wing area per fin (small for stealth)
            aspect_ratio=1.1,  # low AR V-tail fins (stubby for RCS)
            sweep_deg=50.0,  # LE aligned with wing LE ±3° for RCS
            taper_ratio=0.35,  # moderate taper
            cant_deg=0.0,
            v_tail=True,
            v_tail_cant_deg=45.0,  # 45° from vertical (B-2/YF-23 range)
            tailless=False,
        ),
        control_surfaces=ControlSurfaceParams(
            elevon_chord_pct=0.25,  # 25% chord elevons (pitch + roll authority)
            aileron_chord_pct=0.20,  # 20% chord outboard ailerons
            elevon_span_pct=0.65,  # 65% of inboard panel span
            aileron_span_pct=0.45,  # 45% of outboard panel span
            deflection_deg=0.0,
        ),
        intake=IntakeParams(
            intake_type="dorsal",  # top-mounted: shields compressor from ground radar
            intake_count=1,
            capture_area_m2=0.32,  # sized for ~50kN class engine (M88/Adour)
            ramp_angle_deg=4.0,  # gentle compression for subsonic cruise
            lip_sweep_deg=40.0,  # swept lip aligned to planform edges
            station_pct=0.22,  # 22% of fuselage (behind avionics, ahead of wing root)
            boundary_layer_diverter_mm=50.0,  # thinner BLD for subsonic (less BL buildup)
            capture_aspect_ratio=2.5,  # very wide, flat dorsal opening
            duct_length_m=2.8,  # S-duct: sufficient length for 60-90° total turn
            lip_radius_mm=25.0,  # generous lip radius for subsonic (no shock interaction)
        ),
        exhaust=ExhaustParams(
            nozzle_type="convergent",  # subsonic UCAV: simple convergent nozzle
            exit_diameter_m=0.55,  # sized for engine class (~M88: 0.55m)
            throat_diameter_m=0.50,
            nozzle_length_m=1.0,  # short nozzle, shielded by aft fuselage
        ),
        area_rule=AreaRuleParams(
            enabled=True,
            target_mach=0.85,  # optimize for cruise Mach
            waist_station_pct=0.48,  # slightly forward of mid-body
            smoothing_iterations=10,  # more iterations for smoother distribution
        ),
        internal_structure=InternalStructureParams(
            engine_bay_station_pct=0.52,  # engine bay centered at 52% of length
            engine_bay_length_pct=0.22,  # 22% of fuselage for engine + accessories
            fuel_tank_volume_pct=0.30,  # 30% of internal volume for fuel
            weapons_bay_enabled=True,
        ),
    )
