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
        # Find fuselage radius at wing station for proper root placement
        fuse_rh = fuse_r * (math.sqrt(p.fuselage.cross_section_aspect)
                            if p.fuselage.cross_section != "circular" else 1.0)
        fuse_rv = fuse_r / (math.sqrt(p.fuselage.cross_section_aspect)
                            if p.fuselage.cross_section != "circular" else 1.0)

        for label, secs in [("wing_right", right_secs), ("wing_left", left_secs)]:
            verts = np.vstack([s.points_3d for s in secs])
            # Offset chordwise position
            verts[:, 0] += wing_x_offset
            # Offset wing root spanwise to fuselage surface
            side = 1.0 if "right" in label else -1.0
            root_y_offset = fuse_rh * side
            verts[:, 1] += root_y_offset
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

        # 7. Blending — wing-fuselage fairing
        self.blending_op = BlendingOperator(p.blending)

        for label, secs, side in [
            ("wing_root_fairing_right", right_secs, 1.0),
            ("wing_root_fairing_left", left_secs, -1.0),
        ]:
            if len(secs) > 0:
                # Wing root airfoil points (innermost section)
                wing_root_pts = secs[0].points_3d.copy()
                wing_root_pts[:, 0] += wing_x_offset
                wing_root_pts[:, 1] += fuse_rh * side

                fv, ff = self.blending_op.build_wing_fuselage_fairing(
                    wing_root_sections=wing_root_pts,
                    fuselage_sections=self.fuselage_builder.sections,
                    wing_x_offset=wing_x_offset,
                    fuselage_radius_h=fuse_rh,
                    fuselage_radius_v=fuse_rv,
                    side=side,
                )
                if len(fv) > 0:
                    self.components[label] = (fv, ff)

        # Strake (if enabled)
        for label, side in [("strake_right", 1.0), ("strake_left", -1.0)]:
            sv, sf = self.blending_op.build_strake(
                fuselage_length=L,
                wing_root_x=wing_x_offset,
                fuselage_radius=fuse_rh,
                side=side,
            )
            if len(sv) > 0:
                self.components[label] = (sv, sf)

        # 4. Vertical stabilizer
        self.stab_builder = StabilizerBuilder(p.vertical_stabilizer)
        stab_secs = self.stab_builder.build()
        sv, sf = self.stab_builder.get_mesh()
        # Position at rear of fuselage
        stab_x = L * 0.82
        sv[:, 0] += stab_x
        self.components["vertical_stabilizer"] = (sv, sf)

        # 5. Intakes (fuselage-aware positioning)
        self.intake_builder = IntakeBuilder(p.intake)
        intake_x = L * p.intake.station_pct
        fuse_r_at_intake = self._get_fuselage_radius_at(intake_x)
        right_intake, left_intake = self.intake_builder.build(
            fuselage_radius=fuse_r_at_intake,
        )
        for label, secs, side in [
            ("intake_right", right_intake, 1.0),
            ("intake_left", left_intake, -1.0),
        ]:
            iv, i_f = self.intake_builder.get_mesh(secs)
            iv[:, 0] += intake_x
            self.components[label] = (iv, i_f)

        # BLD splitter plates
        for label, side in [("bld_right", 1.0), ("bld_left", -1.0)]:
            bv, bf = self.intake_builder.build_bld_plate(
                side=side, fuselage_radius=fuse_r_at_intake,
            )
            if len(bv) > 0:
                bv[:, 0] += intake_x
                self.components[label] = (bv, bf)

        # 6. Exhaust nozzle
        self.exhaust_builder = ExhaustBuilder(p.exhaust)
        self.exhaust_builder.build()
        ev, ef = self.exhaust_builder.get_mesh()
        ev[:, 0] += L - p.exhaust.nozzle_length_m
        self.components["exhaust"] = (ev, ef)

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

    def _get_fuselage_radius_at(self, x: float) -> float:
        """Get fuselage horizontal half-width at a given x station.

        Interpolates from built fuselage sections.
        """
        if not self.fuselage_builder or not self.fuselage_builder.sections:
            return self.params.fuselage.max_diameter_m / 2.0

        sections = self.fuselage_builder.sections
        if x <= sections[0].x:
            return sections[0].radius_h
        if x >= sections[-1].x:
            return sections[-1].radius_h

        for j in range(len(sections) - 1):
            s0, s1 = sections[j], sections[j + 1]
            if s0.x <= x <= s1.x:
                frac = (x - s0.x) / (s1.x - s0.x) if (s1.x - s0.x) > 0 else 0.0
                return s0.radius_h + frac * (s1.radius_h - s0.radius_h)

        return sections[-1].radius_h

    def get_combined_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Combine all component meshes into a single welded mesh.

        Performs vertex welding: vertices within tolerance (1e-4 m) are
        merged so that adjacent components share boundary vertices.

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

        if not all_verts:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        verts = np.vstack(all_verts)
        faces = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=int)

        # Vertex welding: merge coincident vertices
        tolerance = 1e-4
        n = len(verts)
        if n > 0 and len(faces) > 0:
            # Quantize to grid for fast duplicate detection
            quantized = np.round(verts / tolerance).astype(np.int64)
            # Use structured array for unique detection
            dtype = np.dtype([('x', np.int64), ('y', np.int64), ('z', np.int64)])
            structured = np.empty(n, dtype=dtype)
            structured['x'] = quantized[:, 0]
            structured['y'] = quantized[:, 1]
            structured['z'] = quantized[:, 2]
            _, inverse = np.unique(structured, return_inverse=True)
            # Remap face indices
            faces = inverse[faces]
            # Compact vertices (keep unique only)
            unique_mask = np.zeros(n, dtype=bool)
            for new_idx in range(inverse.max() + 1):
                first = np.where(inverse == new_idx)[0][0]
                unique_mask[first] = True
            # Build old→new index mapping
            new_indices = np.cumsum(unique_mask) - 1
            remap = new_indices[inverse]
            verts = verts[unique_mask]
            faces = remap[faces]

        return verts, faces

    def get_component_names(self) -> list[str]:
        """Return list of component names."""
        return list(self.components.keys())
