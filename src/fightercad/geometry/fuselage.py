"""Fuselage geometry generation with Sears-Haack and other nose profiles.

Generates a lofted fuselage body as a list of cross-section rings (3D points).
When pythonocc is available, produces TopoDS_Shape via BRepOffsetAPI_ThruSections.
Without pythonocc, returns mesh data (vertices, faces) for visualization.
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


class FuselageBuilder:
    """Build fuselage geometry from parameters.

    The fuselage is divided into three zones:
      1. Nose: from tip to max-diameter station (governed by nose profile)
      2. Cylindrical: constant max diameter section
      3. Aft taper: from max diameter to tail
    """

    def __init__(self, params: FuselageParams, n_sections: int = 40, n_ring: int = 36):
        self.p = params
        self.n_sections = n_sections
        self.n_ring = n_ring
        self.sections: list[FuselageSection] = []

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

        # Axial stations
        x_all = np.linspace(0, L, self.n_sections)

        # Compute radius at each station
        radii = np.zeros(self.n_sections)
        for i, x in enumerate(x_all):
            if x <= nose_len:
                # Nose zone
                r_arr = get_nose_profile(
                    p.nose_profile,
                    np.array([x]),
                    nose_len,
                    R,
                    p.haack_C,
                )
                radii[i] = r_arr[0]
            elif x <= L * 0.75:
                # Cylindrical zone (max radius)
                radii[i] = R
            else:
                # Aft taper zone
                t = (x - L * 0.75) / (L * 0.25)
                radii[i] = R * (1.0 - t * (1.0 - p.aft_taper_ratio))

        # Apply area-rule correction if provided
        if area_rule_correction is not None:
            n_corr = len(area_rule_correction)
            if n_corr == self.n_sections:
                radii = np.maximum(radii + area_rule_correction, 0.01)
            else:
                corr_interp = np.interp(
                    np.linspace(0, 1, self.n_sections),
                    np.linspace(0, 1, n_corr),
                    area_rule_correction,
                )
                radii = np.maximum(radii + corr_interp, 0.01)

        # Generate cross-section rings
        self.sections = []
        theta = np.linspace(0, 2 * math.pi, self.n_ring, endpoint=False)
        aspect = p.cross_section_aspect

        for i, x in enumerate(x_all):
            r = radii[i]
            rh = r * math.sqrt(aspect) if p.cross_section != "circular" else r
            rv = r / math.sqrt(aspect) if p.cross_section != "circular" else r

            if p.cross_section == "rounded_rect":
                # Superellipse for rounded rectangle: |y/a|^n + |z/b|^n = 1
                n_exp = 3.0
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                denom = (np.abs(cos_t) ** n_exp + np.abs(sin_t) ** n_exp) ** (1.0 / n_exp)
                py = rh * cos_t / denom
                pz = rv * sin_t / denom
            else:
                # Circular or elliptical
                py = rh * np.cos(theta)
                pz = rv * np.sin(theta)

            pts = np.column_stack([
                np.full(self.n_ring, x),
                py,
                pz,
            ])
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

        # Close nose tip
        tip = np.mean(self.sections[0].points, axis=0, keepdims=True)
        tip_idx = len(verts)
        verts = np.vstack([verts, tip])
        for j in range(n_ring):
            j1 = (j + 1) % n_ring
            faces.append([tip_idx, j, j1])

        # Close tail
        tail = np.mean(self.sections[-1].points, axis=0, keepdims=True)
        tail_idx = len(verts)
        verts = np.vstack([verts, tail])
        base_last = (n_sec - 1) * n_ring
        for j in range(n_ring):
            j1 = (j + 1) % n_ring
            faces.append([tail_idx, base_last + j1, base_last + j])

        return verts, np.array(faces)
