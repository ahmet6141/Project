"""Fuselage geometry generation with Sears-Haack and other nose profiles.

Generates a lofted fuselage body as a list of cross-section rings (3D points).
Supports variable cross-section aspect ratio, canopy bump, dorsal spine,
smooth aft taper, and cosine-distributed sections for high-quality output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fightercad.parameters import FuselageParams
from fightercad.geometry.primitives import get_nose_profile


@dataclass
class FuselageSection:
    """A single cross-section of the fuselage."""
    x: float  # axial position (m)
    radius_h: float  # horizontal half-width (m)
    radius_v: float  # vertical half-height (m)
    points: np.ndarray  # (N, 3) ring of points


def _smoothstep(t: float) -> float:
    """Hermite smooth-step: 3t² - 2t³ for t in [0,1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class FuselageBuilder:
    """Build fuselage geometry from parameters.

    The fuselage is divided into three zones:
      1. Nose: from tip to max-diameter station (governed by nose profile)
      2. Cylindrical: constant max diameter section
      3. Aft taper: smooth power-curve reduction to tail
    """

    def __init__(self, params: FuselageParams, n_sections: int = 100, n_ring: int = 80):
        self.p = params
        self.n_sections = n_sections
        self.n_ring = n_ring
        self.sections: list[FuselageSection] = []

    def _build_x_stations(self) -> np.ndarray:
        """Build cosine-clustered axial stations (denser at nose and tail)."""
        p = self.p
        L = p.length_m
        nose_len = p.nose_fineness_ratio * p.max_diameter_m
        cyl_end = L * p.cylindrical_end_pct

        # Allocate sections: 30% nose, 40% body, 30% tail
        n_nose = max(8, int(self.n_sections * 0.30))
        n_tail = max(8, int(self.n_sections * 0.30))
        n_body = self.n_sections - n_nose - n_tail

        # Nose: cosine clustering (denser near tip)
        t_nose = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi, n_nose)))
        x_nose = t_nose * nose_len

        # Body: uniform
        x_body = np.linspace(nose_len, cyl_end, n_body + 2)[1:-1]

        # Tail: cosine clustering (denser near tail end)
        t_tail = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi, n_tail)))
        x_tail = cyl_end + t_tail * (L - cyl_end)

        x_all = np.unique(np.concatenate([x_nose, x_body, x_tail]))
        # Ensure first and last stations are exactly 0 and L
        if x_all[0] > 0:
            x_all = np.concatenate([[0.0], x_all])
        if x_all[-1] < L:
            x_all = np.concatenate([x_all, [L]])
        return x_all

    def _compute_radius(self, x: float, R: float, nose_len: float) -> float:
        """Compute fuselage radius at axial position x."""
        p = self.p
        L = p.length_m
        cyl_end = L * p.cylindrical_end_pct
        blend_len = L * p.nose_body_blend_pct

        if x <= nose_len - blend_len:
            # Pure nose zone
            r_arr = get_nose_profile(
                p.nose_profile, np.array([x]), nose_len, R, p.haack_C,
            )
            return float(r_arr[0])
        elif x <= nose_len + blend_len:
            # Nose-to-body blending zone
            r_nose = float(get_nose_profile(
                p.nose_profile, np.array([x]), nose_len, R, p.haack_C,
            )[0])
            t_blend = (x - (nose_len - blend_len)) / (2.0 * blend_len) if blend_len > 0 else 1.0
            return r_nose * (1.0 - _smoothstep(t_blend)) + R * _smoothstep(t_blend)
        elif x <= cyl_end:
            # Cylindrical zone
            return R
        else:
            # Aft taper (power-curve): R at cyl_end → R*aft_taper_ratio at tail
            t = (x - cyl_end) / (L - cyl_end) if (L - cyl_end) > 0 else 1.0
            t_smooth = t ** p.aft_taper_power
            r_target = R * p.aft_taper_ratio
            r_base = R * (1.0 - t_smooth) + r_target * t_smooth
            # Tail closure: blend to smaller radius in last 10% for smooth tip
            if t > 0.9:
                t_close = (t - 0.9) / 0.1
                r_end = r_target * p.tail_closure_radius_pct
                r_end = max(r_end, 0.005)
                r_base = r_base * (1.0 - t_close) + r_end * t_close
            return r_base

    def _compute_aspect(self, x: float) -> float:
        """Compute cross-section aspect ratio at axial position x (varies along length)."""
        p = self.p
        L = p.length_m
        nose_len = p.nose_fineness_ratio * p.max_diameter_m
        cyl_end = L * p.cylindrical_end_pct

        if x <= nose_len:
            # Nose: interpolate from nose aspect to body aspect
            t = x / nose_len if nose_len > 0 else 1.0
            return p.cross_section_aspect_nose + _smoothstep(t) * (p.cross_section_aspect - p.cross_section_aspect_nose)
        elif x <= cyl_end:
            return p.cross_section_aspect
        else:
            # Tail: interpolate to tail aspect
            t = (x - cyl_end) / (L - cyl_end) if (L - cyl_end) > 0 else 1.0
            return p.cross_section_aspect + _smoothstep(t) * (p.cross_section_aspect_tail - p.cross_section_aspect)

    def _compute_bwb_extension(self, x: float, rh_base: float) -> float:
        """Compute BWB lateral body extension at axial position x.

        For BWB designs, the fuselage widens significantly at the wing station
        to create a smooth body-wing blending zone. Returns additional half-width.
        """
        p = self.p
        if p.body_wing_blend_ratio <= 0:
            return 0.0

        L = p.length_m
        # Wing region: from wing_station to ~80% of fuselage
        # (BWB body widens where the wing root would be)
        wing_start_pct = getattr(p, '_wing_station_pct', 0.25)
        wing_end_pct = min(wing_start_pct + 0.55, 0.85)
        wing_start = L * wing_start_pct
        wing_end = L * wing_end_pct

        if x < wing_start or x > wing_end:
            return 0.0

        # Target half-width: inner panel span
        target_hw = p.body_wing_inner_span_m if p.body_wing_inner_span_m > 0 else rh_base * 2.0

        # Bell-shaped extension: smooth rise and fall
        t = (x - wing_start) / (wing_end - wing_start)
        # Peak at 30% of wing region (where max chord is)
        peak = 0.30
        if t <= peak:
            s = t / peak
            # Quintic smoothstep for C2 continuity
            envelope = 6 * s**5 - 15 * s**4 + 10 * s**3
        else:
            s = (t - peak) / (1.0 - peak)
            envelope = 1.0 - (6 * s**5 - 15 * s**4 + 10 * s**3)

        extension = (target_hw - rh_base) * envelope * p.body_wing_blend_ratio
        return max(0.0, extension)

    def _apply_canopy(
        self, pts: np.ndarray, x: float, theta: np.ndarray
    ) -> np.ndarray:
        """Apply canopy bump to upper surface of cross-section points."""
        p = self.p
        if not p.canopy_enabled or p.canopy_height_mm <= 0:
            return pts
        L = p.length_m
        canopy_start = L * p.cockpit_station_pct
        canopy_end = canopy_start + L * p.canopy_length_pct

        if x < canopy_start or x > canopy_end or canopy_end <= canopy_start:
            return pts

        t_canopy = (x - canopy_start) / (canopy_end - canopy_start)
        bump = (p.canopy_height_mm / 1000.0) * math.sin(math.pi * t_canopy)

        # Apply only to upper half (sin(theta) > 0), weighted by sin(theta)
        sin_t = np.sin(theta)
        upper_mask = sin_t > 0
        pts[upper_mask, 2] += bump * sin_t[upper_mask]

        return pts

    def _apply_dorsal_spine(
        self, pts: np.ndarray, x: float, theta: np.ndarray
    ) -> np.ndarray:
        """Apply dorsal spine height to top of fuselage."""
        p = self.p
        if p.dorsal_spine_height_mm <= 0:
            return pts
        L = p.length_m
        # Spine runs from 30% to 85% of fuselage
        spine_start = L * 0.30
        spine_end = L * 0.85
        if x < spine_start or x > spine_end:
            return pts

        t = (x - spine_start) / (spine_end - spine_start)
        # Peak at ~50% of spine, smooth sin distribution
        spine_h = (p.dorsal_spine_height_mm / 1000.0) * math.sin(math.pi * t)

        # Apply to top sector (narrow band around theta=π/2)
        sin_t = np.sin(theta)
        top_mask = sin_t > 0.7  # top ~45° sector
        weight = np.clip((sin_t[top_mask] - 0.7) / 0.3, 0, 1)
        pts[top_mask, 2] += spine_h * weight

        return pts

    def build(self, area_rule_correction: np.ndarray | None = None) -> list[FuselageSection]:
        """Generate fuselage cross-sections.

        Parameters
        ----------
        area_rule_correction : np.ndarray, optional
            Per-section radius delta for area-rule waisting.

        Returns
        -------
        list[FuselageSection]
            Ordered list of cross-sections from nose to tail.
        """
        p = self.p
        R = p.max_diameter_m / 2.0
        L = p.length_m
        nose_len = p.nose_fineness_ratio * p.max_diameter_m

        # Build axial stations (cosine-clustered)
        x_all = self._build_x_stations()
        n_actual = len(x_all)

        # Compute radius at each station
        radii = np.array([self._compute_radius(x, R, nose_len) for x in x_all])

        # Apply area-rule correction if provided
        if area_rule_correction is not None:
            n_corr = len(area_rule_correction)
            if n_corr == n_actual:
                radii = np.maximum(radii + area_rule_correction, 0.005)
            else:
                corr_interp = np.interp(
                    np.linspace(0, 1, n_actual),
                    np.linspace(0, 1, n_corr),
                    area_rule_correction,
                )
                radii = np.maximum(radii + corr_interp, 0.005)

        # Generate cross-section rings
        self.sections = []
        theta = np.linspace(0, 2 * math.pi, self.n_ring, endpoint=False)

        for i, x in enumerate(x_all):
            r = radii[i]
            aspect = self._compute_aspect(x)

            if p.cross_section != "circular":
                rh = r * math.sqrt(aspect)
                rv = r / math.sqrt(aspect)
            else:
                rh = r
                rv = r

            # BWB lateral extension (widens body at wing station)
            bwb_ext = self._compute_bwb_extension(x, rh)
            rh += bwb_ext

            if p.cross_section == "rounded_rect":
                n_exp = 3.0
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                denom = (np.abs(cos_t) ** n_exp + np.abs(sin_t) ** n_exp) ** (1.0 / n_exp)
                py = rh * cos_t / denom
                pz = rv * sin_t / denom
            else:
                py = rh * np.cos(theta)
                pz = rv * np.sin(theta)

            pts = np.column_stack([np.full(self.n_ring, x), py, pz])

            # Apply canopy bump and dorsal spine
            pts = self._apply_canopy(pts, x, theta)
            pts = self._apply_dorsal_spine(pts, x, theta)

            self.sections.append(FuselageSection(x=x, radius_h=rh, radius_v=rv, points=pts))

        return self.sections

    def get_area_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_stations, cross_section_areas) for area-rule analysis."""
        if not self.sections:
            self.build()
        x = np.array([s.x for s in self.sections])
        areas = np.array([math.pi * s.radius_h * s.radius_v for s in self.sections])
        return x, areas

    def get_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a triangle mesh (vertices, faces) from sections.

        Uses proper nose tip and tail closure based on profile geometry.

        Returns
        -------
        vertices : np.ndarray
            (V, 3) vertex positions.
        faces : np.ndarray
            (F, 3) triangle face indices.
        """
        if not self.sections:
            self.build()

        n_sec = len(self.sections)
        n_ring = self.n_ring
        verts = np.vstack([s.points for s in self.sections])

        faces = []
        for i in range(n_sec - 1):
            base_curr = i * n_ring
            base_next = (i + 1) * n_ring
            for j in range(n_ring):
                j1 = (j + 1) % n_ring
                v0 = base_curr + j
                v1 = base_curr + j1
                v2 = base_next + j1
                v3 = base_next + j
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        # Nose tip: use actual nose tip point [0, 0, 0] for pointed nose
        # or centroid for blunt nose
        first_sec = self.sections[0]
        if first_sec.radius_h < 0.01 and first_sec.radius_v < 0.01:
            # Nearly pointed: use geometric tip
            tip = np.array([[first_sec.x, 0.0, 0.0]])
        else:
            tip = np.mean(first_sec.points, axis=0, keepdims=True)
        tip_idx = len(verts)
        verts = np.vstack([verts, tip])
        for j in range(n_ring):
            j1 = (j + 1) % n_ring
            faces.append([tip_idx, j, j1])

        # Tail closure
        last_sec = self.sections[-1]
        if last_sec.radius_h < 0.01 and last_sec.radius_v < 0.01:
            tail = np.array([[last_sec.x, 0.0, 0.0]])
        else:
            tail = np.mean(last_sec.points, axis=0, keepdims=True)
        tail_idx = len(verts)
        verts = np.vstack([verts, tail])
        base_last = (n_sec - 1) * n_ring
        for j in range(n_ring):
            j1 = (j + 1) % n_ring
            faces.append([tail_idx, base_last + j1, base_last + j])

        return verts, np.array(faces)
