"""Top-level aircraft assembly orchestrator.

Combines all geometry components into a complete aircraft, applies
area-rule correction, and provides unified mesh output.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger("fightercad")

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
        logger.info("Aircraft build started — %s (L=%.1fm)", p.meta.name, L)

        # 1. Build fuselage (initial pass)
        # Pass wing info for BWB lateral extension
        p.fuselage._wing_station_pct = p.wing.wing_station_pct
        if p.fuselage.body_wing_inner_span_m <= 0 and p.fuselage.body_wing_blend_ratio > 0:
            # Auto-compute BWB inner span from wing geometry
            half_span = p.wing.span_m / 2.0
            inner_span = half_span * p.wing.inner_panel_span_pct
            p.fuselage.body_wing_inner_span_m = inner_span
        self.fuselage_builder = FuselageBuilder(p.fuselage)
        self.fuselage_builder.build()
        logger.info("  Fuselage built: %d sections, nose=%s",
                     len(self.fuselage_builder.sections), p.fuselage.nose_profile)

        # 2. Build wings (panelled)
        self.wing_builder = WingBuilder(p.wing)

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

            half_span, root_c, tip_c = self.wing_builder._compute_planform()
            a_wing = self._estimate_wing_area_distribution(x_fuse, wing_x_offset, root_c, p.wing)

            self.area_rule_result = self.area_analyzer.analyze(
                x_fuse, a_fuse, a_wing, s_ref=p.wing.area_m2,
            )

            correction = self.area_rule_result["correction"]
            self.fuselage_builder.build(area_rule_correction=correction)

        # Get fuselage mesh
        fv, ff = self.fuselage_builder.get_mesh()
        self.components["fuselage"] = (fv, ff)
        logger.info("  Fuselage mesh: %d verts, %d faces", len(fv), len(ff))

        # Fuselage radii at wing station — use actual section data
        # Find the MAXIMUM fuselage rh across the entire wing chord range
        # to ensure the wing root is never inside the fuselage
        fuse_rh_base = fuse_r * (math.sqrt(p.fuselage.cross_section_aspect)
                                 if p.fuselage.cross_section != "circular" else 1.0)
        fuse_rv_base = fuse_r / (math.sqrt(p.fuselage.cross_section_aspect)
                                 if p.fuselage.cross_section != "circular" else 1.0)
        fuse_rh = fuse_rh_base
        fuse_rv = fuse_rv_base
        if self.fuselage_builder.sections:
            half_span, root_c_tmp, _ = WingBuilder(p.wing)._compute_planform()
            wing_x_end = wing_x_offset + root_c_tmp
            relevant = [s for s in self.fuselage_builder.sections
                        if wing_x_offset - 0.1 <= s.x <= wing_x_end + 0.1]
            if relevant:
                fuse_rh = max(s.radius_h for s in relevant)
                fuse_rv = max(s.radius_v for s in relevant)

        # Build panelled wings with control surfaces
        logger.info("  Wing: span=%.1fm, sweep=%.0f°, airfoil=%s→%s",
                     p.wing.span_m, p.wing.leading_edge_sweep_deg,
                     p.wing.root_airfoil, p.wing.tip_airfoil)
        self._build_wing_panels(wing_x_offset, fuse_rh, fuse_rv)

        # 4. Vertical stabilizer / V-tail / tailless
        logger.info("  Stabilizer: tailless=%s, v_tail=%s",
                     p.vertical_stabilizer.tailless, p.vertical_stabilizer.v_tail)
        self._build_stabilizers(L)

        # 5. Intakes
        logger.info("  Intake: type=%s, count=%d", p.intake.intake_type, p.intake.intake_count)
        self._build_intakes(L, fuse_rh, fuse_rv)

        # 6. Exhaust nozzle
        logger.info("  Exhaust: type=%s, exit_d=%.2fm", p.exhaust.nozzle_type, p.exhaust.exit_diameter_m)
        self.exhaust_builder = ExhaustBuilder(p.exhaust)
        self.exhaust_builder.build()
        ev, ef = self.exhaust_builder.get_mesh()
        ev[:, 0] += L - p.exhaust.nozzle_length_m
        self.components["exhaust"] = (ev, ef)

        total_v = sum(len(v) for v, _ in self.components.values())
        total_f = sum(len(f) for _, f in self.components.values())
        logger.info("Build complete: %d components, %d verts, %d faces",
                     len(self.components), total_v, total_f)
        return self.components

    def _build_wing_panels(
        self, wing_x_offset: float, fuse_rh: float, fuse_rv: float
    ) -> None:
        """Build inner/outer wing panels with control surfaces for both sides.

        The wing builder generates sections from y=0 to y=half_span.
        We scale spanwise coordinates so that after adding the fuselage
        offset, the total half-span from centerline equals span_m/2.
        This ensures correct total wingspan.
        """
        p = self.params
        cs = p.control_surfaces
        half_span = p.wing.span_m / 2.0
        # Scale factor: wing should extend from fuse_rh to half_span
        y_scale = (half_span - fuse_rh) / half_span if half_span > fuse_rh else 1.0

        self.blending_op = BlendingOperator(p.blending)

        for side_label, side in [("right", 1.0), ("left", -1.0)]:
            inner_secs, outer_secs = self.wing_builder.build_half_wing_panels(side=side)

            # Split control surfaces
            inner_main, elevon_secs = self.wing_builder.split_control_surface(
                inner_secs, cs.elevon_chord_pct, cs.elevon_span_pct,
            )
            outer_main, aileron_secs = self.wing_builder.split_control_surface(
                outer_secs, cs.aileron_chord_pct, cs.aileron_span_pct,
            )

            # Generate meshes and position in aircraft coordinates
            # Scale y so that tip ends at half_span from centerline
            for comp_name, secs in [
                (f"wing_inner_{side_label}", inner_main),
                (f"wing_outer_{side_label}", outer_main),
                (f"elevon_{side_label}", elevon_secs),
                (f"aileron_{side_label}", aileron_secs),
            ]:
                if not secs:
                    continue
                verts, faces = self.wing_builder.get_panel_mesh(secs)
                if len(verts) == 0:
                    continue
                verts[:, 0] += wing_x_offset
                # Scale y so panel spans from fuse_rh to half_span
                verts[:, 1] = verts[:, 1] * y_scale + fuse_rh * side
                self.components[comp_name] = (verts, faces)
                logger.debug("    %s: %d verts, %d faces", comp_name, len(verts), len(faces))

            # Saw-tooth trailing edge (stealth feature)
            if p.wing.sawtooth_te_enabled:
                all_secs = outer_main + (aileron_secs if aileron_secs else [])
                if all_secs:
                    st_v, st_f = self.wing_builder.build_sawtooth_te(
                        all_secs, p.wing.sawtooth_depth_mm,
                        p.wing.sawtooth_count, side,
                    )
                    if len(st_v) > 0:
                        st_v[:, 0] += wing_x_offset
                        st_v[:, 1] = st_v[:, 1] * y_scale + fuse_rh * side
                        self.components[f"sawtooth_te_{side_label}"] = (st_v, st_f)

            # Wing-fuselage fairing (use inner panel root section)
            if inner_main:
                wing_root_pts = inner_main[0].points_3d.copy()
                wing_root_pts[:, 0] += wing_x_offset
                wing_root_pts[:, 1] = wing_root_pts[:, 1] * y_scale + fuse_rh * side

                fv, ff = self.blending_op.build_wing_fuselage_fairing(
                    wing_root_sections=wing_root_pts,
                    fuselage_sections=self.fuselage_builder.sections,
                    wing_x_offset=wing_x_offset,
                    fuselage_radius_h=fuse_rh,
                    fuselage_radius_v=fuse_rv,
                    side=side,
                )
                if len(fv) > 0:
                    self.components[f"wing_root_fairing_{side_label}"] = (fv, ff)

        # Strakes
        for label, side in [("strake_right", 1.0), ("strake_left", -1.0)]:
            sv, sf = self.blending_op.build_strake(
                fuselage_length=self.params.fuselage.length_m,
                wing_root_x=wing_x_offset,
                fuselage_radius=fuse_rh,
                side=side,
            )
            if len(sv) > 0:
                self.components[label] = (sv, sf)

    def _build_stabilizers(self, fuselage_length: float) -> None:
        """Build vertical stabilizer, V-tail, or skip if tailless.

        Stabilizer root is offset to the fuselage surface and a root
        fillet is generated for a smooth, CFD-ready junction.
        """
        p = self.params
        stab_p = p.vertical_stabilizer
        stab_x = fuselage_length * 0.82

        if stab_p.tailless or stab_p.area_m2 <= 0:
            return  # tailless configuration

        # Get fuselage radii at stabilizer station
        fuse_rh_stab = self._get_fuselage_radius_at(stab_x)
        fuse_rv_stab = self._get_fuselage_radius_v_at(stab_x)
        fillet_mm = p.blending.stabilizer_root_fillet_mm

        if stab_p.v_tail:
            # Dual V-tail fins
            for label, side in [("vtail_right", 1.0), ("vtail_left", -1.0)]:
                builder = StabilizerBuilder(stab_p)
                builder.build(
                    side=side,
                    fuselage_radius_h=fuse_rh_stab,
                    fuselage_radius_v=fuse_rv_stab,
                )
                sv, sf = builder.get_mesh()
                if len(sv) > 0:
                    sv[:, 0] += stab_x
                    self.components[label] = (sv, sf)
                # Root fillet
                fv, ff = builder.build_root_fillet(fillet_mm=fillet_mm)
                if len(fv) > 0:
                    fv[:, 0] += stab_x
                    self.components[f"{label}_fillet"] = (fv, ff)
        else:
            # Single vertical stabilizer (with optional cant)
            self.stab_builder = StabilizerBuilder(stab_p)
            self.stab_builder.build(
                side=0.0,
                fuselage_radius_h=fuse_rh_stab,
                fuselage_radius_v=fuse_rv_stab,
            )
            sv, sf = self.stab_builder.get_mesh()
            if len(sv) > 0:
                sv[:, 0] += stab_x
                self.components["vertical_stabilizer"] = (sv, sf)
            # Root fillet
            fv, ff = self.stab_builder.build_root_fillet(fillet_mm=fillet_mm)
            if len(fv) > 0:
                fv[:, 0] += stab_x
                self.components["vstab_fillet"] = (fv, ff)

    def _build_intakes(
        self, fuselage_length: float, fuse_rh: float, fuse_rv: float
    ) -> None:
        """Build intake(s) based on intake_type (chin, dorsal, or side_mounted)."""
        p = self.params
        self.intake_builder = IntakeBuilder(p.intake)
        intake_x = fuselage_length * p.intake.station_pct
        fuse_r_at_intake = self._get_fuselage_radius_at(intake_x)
        fuse_rv_at_intake = self._get_fuselage_radius_v_at(intake_x)

        fillet_mm = p.blending.intake_fuselage_fillet_mm

        if p.intake.intake_type in ("chin", "dorsal"):
            # Single centerline intake (chin = below, dorsal = above)
            comp_suffix = p.intake.intake_type  # "chin" or "dorsal"
            secs = self.intake_builder.build_single(
                side=0.0,
                fuselage_radius=fuse_r_at_intake,
                fuselage_radius_v=fuse_rv_at_intake,
            )
            iv, i_f = self.intake_builder.get_mesh(secs)
            iv[:, 0] += intake_x
            self.components[f"intake_{comp_suffix}"] = (iv, i_f)

            # BLD plate
            bv, bf = self.intake_builder.build_bld_plate(
                side=0.0,
                fuselage_radius=fuse_r_at_intake,
                fuselage_radius_v=fuse_rv_at_intake,
            )
            if len(bv) > 0:
                bv[:, 0] += intake_x
                self.components[f"bld_{comp_suffix}"] = (bv, bf)

            # Intake-fuselage collar (watertight transition)
            cv, cf = self.intake_builder.build_intake_collar(
                fuselage_radius=fuse_r_at_intake,
                fuselage_radius_v=fuse_rv_at_intake,
                fillet_mm=fillet_mm,
            )
            if len(cv) > 0:
                cv[:, 0] += intake_x
                self.components[f"intake_collar_{comp_suffix}"] = (cv, cf)
        else:
            # Dual side-mounted intakes
            right_intake, left_intake = self.intake_builder.build(
                fuselage_radius=fuse_r_at_intake,
            )
            for label, secs, side in [
                ("intake_right", right_intake, 1.0),
                ("intake_left", left_intake, -1.0),
            ]:
                if secs is None:
                    continue
                iv, i_f = self.intake_builder.get_mesh(secs)
                iv[:, 0] += intake_x
                self.components[label] = (iv, i_f)

                # Collar for each side intake
                self.intake_builder.section_points = secs
                cv, cf = self.intake_builder.build_intake_collar(
                    fuselage_radius=fuse_r_at_intake,
                    fuselage_radius_v=fuse_rv_at_intake,
                    fillet_mm=fillet_mm,
                )
                if len(cv) > 0:
                    cv[:, 0] += intake_x
                    self.components[f"intake_collar_{label.split('_')[1]}"] = (cv, cf)

            for label, side in [("bld_right", 1.0), ("bld_left", -1.0)]:
                bv, bf = self.intake_builder.build_bld_plate(
                    side=side, fuselage_radius=fuse_r_at_intake,
                )
                if len(bv) > 0:
                    bv[:, 0] += intake_x
                    self.components[label] = (bv, bf)

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

        # Use actual wing thickness parameters for area estimation
        tc_root = wing_params.thickness_to_chord_root
        tc_tip = wing_params.thickness_to_chord_tip

        for i, x in enumerate(x_stations):
            x_local = x - wing_x_offset
            if 0 <= x_local <= root_chord and root_chord > 0:
                chord_frac = x_local / root_chord
                local_chord = root_chord * (1.0 - chord_frac * (1.0 - wing_params.taper_ratio))
                # Interpolate t/c from root to tip based on chordwise position
                tc_local = tc_root * (1.0 - chord_frac) + tc_tip * chord_frac
                # Wing cross-section area at this x: A(x) ≈ ∫ t(y) dy
                # For a linearly-tapered wing: A ≈ 2/3 * span * chord * t/c
                areas[i] = 2.0 * half_span * local_chord * tc_local * (2.0 / 3.0)

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

    def _get_fuselage_radius_v_at(self, x: float) -> float:
        """Get fuselage vertical half-height at a given x station."""
        if not self.fuselage_builder or not self.fuselage_builder.sections:
            return self.params.fuselage.max_diameter_m / 2.0

        sections = self.fuselage_builder.sections
        if x <= sections[0].x:
            return sections[0].radius_v
        if x >= sections[-1].x:
            return sections[-1].radius_v

        for j in range(len(sections) - 1):
            s0, s1 = sections[j], sections[j + 1]
            if s0.x <= x <= s1.x:
                frac = (x - s0.x) / (s1.x - s0.x) if (s1.x - s0.x) > 0 else 0.0
                return s0.radius_v + frac * (s1.radius_v - s0.radius_v)

        return sections[-1].radius_v

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
            # Fix winding per-component using component centroid
            if len(f) > 0 and len(v) > 0:
                centroid = v.mean(axis=0)
                v0 = v[f[:, 0]]
                v1 = v[f[:, 1]]
                v2 = v[f[:, 2]]
                normals = np.cross(v1 - v0, v2 - v0)
                face_centers = (v0 + v1 + v2) / 3.0
                outward = face_centers - centroid
                dots = np.sum(normals * outward, axis=1)
                flip_mask = dots < 0
                f = f.copy()
                f[flip_mask, 1], f[flip_mask, 2] = (
                    f[flip_mask, 2].copy(), f[flip_mask, 1].copy(),
                )
            all_verts.append(v)
            if len(f) > 0:
                all_faces.append(f + offset)
            offset += len(v)

        if not all_verts:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        verts = np.vstack(all_verts)
        faces = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=int)

        # Vertex welding: merge coincident vertices
        # Adaptive tolerance: 0.5mm base, scaled by model size for robustness
        model_extent = verts.max(axis=0) - verts.min(axis=0)
        model_scale = max(model_extent.max(), 1.0)
        tolerance = max(5e-4, model_scale * 1e-4)  # 0.5mm or 0.01% of model size
        n = len(verts)
        if n > 0 and len(faces) > 0:
            quantized = np.round(verts / tolerance).astype(np.int64)
            dtype = np.dtype([('x', np.int64), ('y', np.int64), ('z', np.int64)])
            structured = np.empty(n, dtype=dtype)
            structured['x'] = quantized[:, 0]
            structured['y'] = quantized[:, 1]
            structured['z'] = quantized[:, 2]
            _, inverse = np.unique(structured, return_inverse=True)
            faces = inverse[faces]
            unique_mask = np.zeros(n, dtype=bool)
            for new_idx in range(inverse.max() + 1):
                first = np.where(inverse == new_idx)[0][0]
                unique_mask[first] = True
            new_indices = np.cumsum(unique_mask) - 1
            remap = new_indices[inverse]
            verts = verts[unique_mask]
            faces = remap[faces]

            # Remove degenerate triangles (collapsed edges)
            valid = (faces[:, 0] != faces[:, 1]) & \
                    (faces[:, 1] != faces[:, 2]) & \
                    (faces[:, 0] != faces[:, 2])
            faces = faces[valid]

        return verts, faces

    def get_component_names(self) -> list[str]:
        """Return list of component names."""
        return list(self.components.keys())

    def get_mesh_quality_report(self) -> dict:
        """Compute mesh quality metrics for the combined mesh.

        Returns a dict with:
        - total_vertices, total_faces
        - degenerate_count: triangles with zero area
        - min/max/avg_aspect_ratio: triangle aspect ratios
        - max_skewness: worst triangle skewness (0=equilateral, 1=degenerate)
        - flipped_normals: count of inconsistently oriented triangles
        """
        verts, faces = self.get_combined_mesh()
        n_faces = len(faces)
        if n_faces == 0:
            return {"total_vertices": len(verts), "total_faces": 0}

        # Compute triangle edge lengths and areas
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)

        # Triangle areas via cross product
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        # Aspect ratio: longest edge / (2 * area / longest edge) = longest² / (2*area)
        longest = np.maximum(e0, np.maximum(e1, e2))
        shortest = np.minimum(e0, np.minimum(e1, e2))
        aspect_ratio = np.where(shortest > 1e-12, longest / shortest, 999.0)

        # Skewness: 1 - (2 * area * sqrt(3)) / (3 * max_edge²)
        # 0 = equilateral, 1 = degenerate
        ideal_area = (math.sqrt(3.0) / 4.0) * longest ** 2
        skewness = np.where(ideal_area > 1e-12, 1.0 - areas / ideal_area, 1.0)
        skewness = np.clip(skewness, 0.0, 1.0)

        # Degenerate triangles (near-zero area)
        degenerate = np.sum(areas < 1e-10)

        # Normal consistency check: per-component centroid-based check
        # For multi-component assemblies, use component centroids
        flipped = 0
        for name, (cv, cf) in self.components.items():
            if len(cf) == 0 or len(cv) == 0:
                continue
            centroid = cv.mean(axis=0)
            cv0 = cv[cf[:, 0]]
            cv1 = cv[cf[:, 1]]
            cv2 = cv[cf[:, 2]]
            c_normals = np.cross(cv1 - cv0, cv2 - cv0)
            c_centers = (cv0 + cv1 + cv2) / 3.0
            c_outward = c_centers - centroid
            c_dots = np.sum(c_normals * c_outward, axis=1)
            flipped += int(np.sum(c_dots < 0))

        return {
            "total_vertices": len(verts),
            "total_faces": n_faces,
            "degenerate_count": int(degenerate),
            "min_aspect_ratio": float(aspect_ratio.min()),
            "max_aspect_ratio": float(aspect_ratio.max()),
            "avg_aspect_ratio": float(aspect_ratio.mean()),
            "max_skewness": float(skewness.max()),
            "avg_skewness": float(skewness.mean()),
            "flipped_normals": flipped,
        }
