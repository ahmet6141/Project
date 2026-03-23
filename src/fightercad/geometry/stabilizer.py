"""Vertical stabilizer geometry generation."""

from __future__ import annotations

import math

import numpy as np

from fightercad.parameters import VerticalStabilizerParams
from fightercad.geometry.primitives import get_airfoil, airfoil_to_3d


class StabilizerBuilder:
    """Build vertical stabilizer geometry.

    The stabilizer uses the same lofting approach as the wing but oriented
    vertically (z-axis instead of y-axis).
    """

    def __init__(self, params: VerticalStabilizerParams, n_sections: int = 6, n_af_pts: int = 40):
        self.p = params
        self.n_sections = n_sections
        self.n_af = n_af_pts
        self.section_points: list[np.ndarray] = []

    def build(self) -> list[np.ndarray]:
        """Generate stabilizer sections.

        Returns
        -------
        list[np.ndarray]
            List of (N, 3) arrays, one per spanwise (vertical) station.
        """
        p = self.p
        span = math.sqrt(p.aspect_ratio * p.area_m2)
        root_chord = 2.0 * p.area_m2 / (span * (1.0 + p.taper_ratio))
        tip_chord = root_chord * p.taper_ratio

        sweep_rad = math.radians(p.sweep_deg)
        cant_rad = math.radians(p.cant_deg)

        z_stations = np.linspace(0, span, self.n_sections)
        self.section_points = []

        for z in z_stations:
            frac = z / span
            chord = root_chord * (1.0 - frac) + tip_chord * frac
            x_le = z * math.tan(sweep_rad)

            # Use a thin symmetric airfoil for the stabilizer
            af_2d = get_airfoil("naca64a004", chord, le_radius_mm=2.0, num_points=self.n_af)

            # Build 3D points: x = chordwise, y = cant offset, z = vertical
            n = len(af_2d)
            pts = np.zeros((n, 3))
            pts[:, 0] = af_2d[:, 0] + x_le  # chordwise + sweep
            pts[:, 1] = z * math.sin(cant_rad)  # cant lateral offset
            pts[:, 2] = z * math.cos(cant_rad)  # vertical
            # Thickness is in the y-direction for vertical stabilizer
            pts[:, 1] += af_2d[:, 1]

            self.section_points.append(pts)

        return self.section_points

    def get_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate triangle mesh."""
        if not self.section_points:
            self.build()

        n_sec = len(self.section_points)
        n_pts = len(self.section_points[0])
        verts = np.vstack(self.section_points)

        faces = []
        for i in range(n_sec - 1):
            b0 = i * n_pts
            b1 = (i + 1) * n_pts
            for j in range(n_pts - 1):
                faces.append([b0 + j, b0 + j + 1, b1 + j + 1])
                faces.append([b0 + j, b1 + j + 1, b1 + j])

        return verts, np.array(faces) if faces else (verts, np.zeros((0, 3), dtype=int))
