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
            design_mach=0.85,  # high-subsonic cruise (nEUROn/Taranis class)
            design_altitude_m=12000.0,  # ~40,000 ft service ceiling
        ),
        fuselage=FuselageParams(
            length_m=10.0,  # nEUROn=9.5m, X-47B=11.6m, Taranis=11.35m
            max_diameter_m=1.2,  # height of flattened body (width=aspect*height)
            nose_fineness_ratio=3.5,  # 3.0-4.0 range (Crowell 1996)
            nose_profile="haack",  # Von Karman (C=0): smoothest, min wave drag
            haack_C=0.0,  # C=0: LD-Haack, smoothest profile for stealth
            aft_taper_ratio=0.35,  # narrows to 35% at nozzle (X-47B/nEUROn)
            cockpit_station_pct=0.12,  # avionics bay (no cockpit)
            cross_section="elliptical",
            cross_section_aspect=2.0,  # W:H=2:1 flattened (B-2/X-47B style)
            cylindrical_end_pct=0.45,  # no true cylinder on BWB; max area ~40-45%
            aft_taper_power=3.0,  # cubic: smooth 2nd derivative for area rule
            nose_body_blend_pct=0.08,  # wide blend for BWB nose merge
            tail_closure_radius_pct=0.30,  # semi-blunt for 2D slot exhaust
            cross_section_aspect_nose=1.5,  # nose: pointed to wide-flat transition
            cross_section_aspect_tail=2.5,  # very flat tail (platypus slot exhaust)
            canopy_enabled=False,
            canopy_height_mm=0.0,
            canopy_length_pct=0.0,
            dorsal_spine_height_mm=0.0,
        ),
        wing=WingParams(
            planform="cropped_delta",
            span_m=12.5,  # nEUROn=12.5m, L/span=0.80
            area_m2=48.0,  # wing loading ~145 kg/m2 at MTOW 7000kg (nEUROn~140)
            aspect_ratio=3.25,  # span2/area=156/48=3.25 (SACCON=3.1)
            taper_ratio=0.08,  # near-pointed tips (research: 0.08-0.15)
            leading_edge_sweep_deg=53.0,  # SACCON AVT-161 standard
            trailing_edge_sweep_deg=27.0,  # planform alignment (MULDICON: 26-30)
            dihedral_deg=0.0,  # 0 geometric (53 sweep gives ~5 effective dihedral)
            twist_deg=-4.0,  # 3-5 washout for tailless trim
            root_airfoil="naca64a004",  # SACCON uses NACA 64A-006 base
            tip_airfoil="biconvex_3",
            leading_edge_radius_mm=5.0,  # research: 0.2-0.5% chord; moderate
            wing_station_pct=0.25,  # BWB: wing root LE near nose (0-35% of length)
            incidence_deg=1.5,  # 0-2 for BWB (body camber provides cruise lift)
            thickness_to_chord_root=0.14,  # 14% root (BWB centerbody; research: 12-16%)
            thickness_to_chord_tip=0.07,  # 7% tip (research: 6-8%; NACA64A006=6%)
            camber_root_pct=2.0,  # reflex camber for Cm0>0 tailless trim
            camber_tip_pct=-1.0,  # negative reflex at tip for positive Cm0
            te_thickness_mm=1.5,  # blunt TE: 0.5-1% chord (stealth+composites)
            sawtooth_te_enabled=True,
            sawtooth_depth_mm=50.0,  # ~1.7x X-band wavelength (30mm)
            sawtooth_count=14,  # edges aligned to 53 LE sweep
            inner_panel_span_pct=0.29,  # SACCON: crank at 29% semi-span
            inner_panel_sweep_deg=53.0,  # SACCON: constant single LE angle
        ),
        blending=BlendingParams(
            root_fillet_radius_mm=150.0,  # massive BWB blend (no distinct fillet)
            leading_edge_fillet_mm=50.0,  # LE merges into nose without break
            trailing_edge_fillet_mm=30.0,
            blending_mode="smooth",
            strake_length_m=0.6,  # short LEX blended into forebody chine
            strake_sweep_deg=65.0,
            fairing_width_mm=400.0,  # 400mm BWB blend zone (25-35% semi-span)
            stabilizer_root_fillet_mm=30.0,
            intake_fuselage_fillet_mm=35.0,
        ),
        vertical_stabilizer=VerticalStabilizerParams(
            area_m2=4.5,  # 8-12% wing area per fin; 48*0.094=4.5m2
            aspect_ratio=1.2,  # low AR for stealth (research: 1.0-1.5)
            sweep_deg=53.0,  # match wing LE sweep for planform alignment
            taper_ratio=0.40,  # research: 0.3-0.5
            cant_deg=0.0,
            v_tail=True,
            v_tail_cant_deg=45.0,  # 45 from vertical (F-117~65, Kizilelma~30)
            tailless=False,
        ),
        control_surfaces=ControlSurfaceParams(
            elevon_chord_pct=0.25,  # 20-25% chord (research confirmed)
            aileron_chord_pct=0.22,  # 20-25% outboard
            elevon_span_pct=0.65,  # 50-70% inner panel TE
            aileron_span_pct=0.65,  # 60-80% outer panel TE (research: was 45%)
            deflection_deg=0.0,
        ),
        intake=IntakeParams(
            intake_type="dorsal",  # top-mounted S-duct (nEUROn, RQ-170)
            intake_count=1,
            capture_area_m2=0.25,  # 0.20-0.30m2 for Adour/F124; ratio~1.2-1.4
            ramp_angle_deg=5.0,  # gentle for subsonic (5-10)
            lip_sweep_deg=53.0,  # aligned to planform LE sweep
            station_pct=0.20,  # 15-25% of length (behind avionics, ahead of root)
            boundary_layer_diverter_mm=20.0,  # 10-30mm for subsonic BL
            capture_aspect_ratio=3.5,  # 3:1-5:1 W:H for dorsal (wide, flat)
            duct_length_m=2.8,  # S-duct L/D~5-6; engine D~0.5m
            lip_radius_mm=25.0,  # generous for subsonic (contraction ratio 1.2)
        ),
        exhaust=ExhaustParams(
            nozzle_type="convergent",  # subsonic UCAV: simple convergent
            exit_diameter_m=0.55,  # equiv circular; actual: slot ~0.7x0.08m
            throat_diameter_m=0.50,
            nozzle_length_m=1.2,  # serpentine for LOS blockage (2-3x engine D)
        ),
        area_rule=AreaRuleParams(
            enabled=True,
            target_mach=0.85,
            waist_station_pct=0.40,  # max area at 35-45% of length
            smoothing_iterations=12,  # high: smooth 2nd derivative requirement
        ),
        internal_structure=InternalStructureParams(
            engine_bay_station_pct=0.55,
            engine_bay_length_pct=0.22,
            fuel_tank_volume_pct=0.30,
            weapons_bay_enabled=True,
        ),
    )
