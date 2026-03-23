"""Top-level aircraft assembly orchestrator.

Combines all geometry components into a complete aircraft, applies
area-rule correction, and provides unified mesh output.
"""

from __future__ import annotations

import math

import numpy as np

from fightercad.parameters import AircraftParams
from fightercad.geometry.fuselage import FuselageBuilder
from fightercad.geometry.wing import WingBuilder
from fightercad.geometry.stabilizer import StabilizerBuilder
from fightercad.geometry.intake import IntakeBuilder
from fightercad.geometry.exhaust import ExhaustBuilder
from fightercad.geometry.area_rule import AreaRuleAnalyzer
from fightercad.geometry.blending import BlendingOperator


class AircraftAssembler:
    """Assemble a complete fighter aircraft from parameters.

    Each component is built independently, then positioned in aircraft
    coordinates and optionally area-rule-corrected.
    """

    def __init__(self, params: AircraftParams):
        self.params = params
        self.fuselage_builder: FuselageBuilder | None = None
        self.wing_builder: WingBuilder | None = None
        self.stab_builder: StabilizerBuilder | None = None
        self.intake_builder: IntakeBuilder | None = None
        self.exhaust_builder: ExhaustBuilder | None = None
        self.blending_op: BlendingOperator | None = None
        self.area_analyzer: AreaRuleAnalyzer | None = None
        self.area_rule_result: dict | None = None

        # Mesh components: dict of name -> (vertices, faces)
        self.components: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def build(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Build all aircraft components and return mesh dictionary.

        Returns
        -------
        dict[str, tuple[np.ndarray, np.ndarray]]
            Mapping of component name to (vertices, faces).
        """
        p = self.params
        L = p.fuselage.length_m

        # 1. Build fuselage (initial pass)
        self.fuselage_builder = FuselageBuilder(p.fuselage)
        self.fuselage_builder.build()

        # 2. Build wings
        self.wing_builder = WingBuilder(p.wing)
        right_secs, left_secs = self.wing_builder.build()

        # Position wings on fuselage
        wing_x_offset = L * p.wing.wing_station_pct
        fuse_r = p.fuselage.max_diameter_m / 2.0

        # 3. Area rule analysis (before final fuselage)
        if p.area_rule.enabled:
            self.area_analyzer = AreaRuleAnalyzer(
                p.area_rule.target_mach,
                p.area_rule.smoothing_iterations,
            )
            x_fuse, a_fuse = self.fuselage_builder.get_area_distribution()

            # Estimate wing area contribution at fuselage stations
            half_span, root_c, tip_c = self.wing_builder._compute_planform()
            a_wing = self._estimate_wing_area_distribution(x_fuse, wing_x_offset, root_c, p.wing)

            self.area_rule_result = self.area_analyzer.analyze(
                x_fuse, a_fuse, a_wing, s_ref=p.wing.area_m2,
            )

            # Rebuild fuselage with area-rule correction
            correction = self.area_rule_result["correction"]
            self.fuselage_builder.build(area_rule_correction=correction)

        # Get fuselage mesh
        fv, ff = self.fuselage_builder.get_mesh()
        self.components["fuselage"] = (fv, ff)

        # Wing meshes (offset to aircraft coordinates)
        for label, secs in [("wing_right", right_secs), ("wing_left", left_secs)]:
            verts = np.vstack([s.points_3d for s in secs])
            # Offset chordwise position
            verts[:, 0] += wing_x_offset
            # Offset vertically to fuselage surface
            sign = 1.0 if "right" in label else -1.0
            # Simple mesh from sections
            n_sec = len(secs)
            n_pts = len(secs[0].points_3d)
            faces = []
            for i in range(n_sec - 1):
                b0, b1 = i * n_pts, (i + 1) * n_pts
                for j in range(n_pts - 1):
                    faces.append([b0 + j, b0 + j + 1, b1 + j + 1])
                    faces.append([b0 + j, b1 + j + 1, b1 + j])
            self.components[label] = (verts, np.array(faces) if faces else np.zeros((0, 3), dtype=int))

        # 4. Vertical stabilizer
        self.stab_builder = StabilizerBuilder(p.vertical_stabilizer)
        stab_secs = self.stab_builder.build()
        sv, sf = self.stab_builder.get_mesh()
        # Position at rear of fuselage
        stab_x = L * 0.82
        sv[:, 0] += stab_x
        self.components["vertical_stabilizer"] = (sv, sf)

        # 5. Intakes
        self.intake_builder = IntakeBuilder(p.intake)
        right_intake, left_intake = self.intake_builder.build()
        intake_x = L * p.intake.station_pct
        for label, secs in [("intake_right", right_intake), ("intake_left", left_intake)]:
            v = np.vstack(secs)
            v[:, 0] += intake_x
            _, f = self.intake_builder.get_mesh(secs)
            self.components[label] = (v, f)

        # 6. Exhaust nozzle
        self.exhaust_builder = ExhaustBuilder(p.exhaust)
        self.exhaust_builder.build()
        ev, ef = self.exhaust_builder.get_mesh()
        ev[:, 0] += L - p.exhaust.nozzle_length_m
        self.components["exhaust"] = (ev, ef)

        # 7. Blending operator
        self.blending_op = BlendingOperator(p.blending)

        return self.components

    def _estimate_wing_area_distribution(
        self,
        x_stations: np.ndarray,
        wing_x_offset: float,
        root_chord: float,
        wing_params,
    ) -> np.ndarray:
        """Estimate wing cross-sectional area contribution at fuselage stations."""
        areas = np.zeros_like(x_stations)
        half_span, _, tip_chord = self.wing_builder._compute_planform()

        for i, x in enumerate(x_stations):
            x_local = x - wing_x_offset
            if 0 <= x_local <= root_chord:
                # Wing chord fraction at this station
                chord_frac = x_local / root_chord
                # Approximate local span and thickness
                local_span = half_span * 2.0  # full span
                t_pct = 5.0 / 100.0  # approximate thickness
                local_chord = root_chord * (1.0 - chord_frac * (1.0 - wing_params.taper_ratio))
                areas[i] = 2.0 * local_span * local_chord * t_pct * 0.5  # approximate

        return areas

    def get_combined_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Combine all component meshes into a single mesh.

        Returns
        -------
        tuple
            (all_vertices, all_faces)
        """
        if not self.components:
            self.build()

        all_verts = []
        all_faces = []
        offset = 0

        for name, (v, f) in self.components.items():
            all_verts.append(v)
            if len(f) > 0:
                all_faces.append(f + offset)
            offset += len(v)

        verts = np.vstack(all_verts) if all_verts else np.zeros((0, 3))
        faces = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=int)
        return verts, faces

    def get_component_names(self) -> list[str]:
        """Return list of component names."""
        return list(self.components.keys())
