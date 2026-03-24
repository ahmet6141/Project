"""Convergent-divergent exhaust nozzle geometry generation."""

from __future__ import annotations

import math

import numpy as np

from fightercad.parameters import ExhaustParams


class ExhaustBuilder:
    """Build exhaust nozzle geometry.

    Creates a convergent-divergent (C-D) nozzle with smooth bell contour.
    """

    def __init__(self, params: ExhaustParams, n_sections: int = 12, n_ring: int = 24):
        self.p = params
        self.n_sections = n_sections
        self.n_ring = n_ring
        self.section_points: list[np.ndarray] = []

    def build(self) -> list[np.ndarray]:
        """Generate nozzle cross-sections.

        Returns
        -------
        list[np.ndarray]
            List of (N, 3) circular section rings from inlet to exit.
        """
        p = self.p
        r_inlet = p.exit_diameter_m / 2.0 * 1.2  # inlet slightly larger than exit
        r_throat = p.throat_diameter_m / 2.0
        r_exit = p.exit_diameter_m / 2.0
        L = p.nozzle_length_m
        if L <= 0:
            self.section_points = []
            return []

        is_cd = p.nozzle_type == "convergent_divergent"

        # Throat position: 40% for C-D, 100% for convergent-only
        throat_frac = 0.4 if is_cd else 1.0
        x_stations = np.linspace(0, L, self.n_sections)
        theta = np.linspace(0, 2 * math.pi, self.n_ring, endpoint=False)

        self.section_points = []
        for x in x_stations:
            t = x / L  # normalized position

            if t <= throat_frac:
                # Convergent section (cosine contour for smooth transition)
                s = t / throat_frac
                r_target = r_throat if is_cd else r_exit
                r = r_inlet + (r_target - r_inlet) * (0.5 - 0.5 * math.cos(math.pi * s))
            else:
                # Divergent section (only for convergent-divergent nozzles)
                s = (t - throat_frac) / (1.0 - throat_frac)
                r = r_throat + (r_exit - r_throat) * (s ** 0.8)

            py = r * np.cos(theta)
            pz = r * np.sin(theta)
            pts = np.column_stack([np.full(self.n_ring, x), py, pz])
            self.section_points.append(pts)

        return self.section_points

    def get_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate triangle mesh."""
        if not self.section_points:
            self.build()

        n_sec = len(self.section_points)
        n_ring = self.n_ring
        verts = np.vstack(self.section_points)

        faces = []
        for i in range(n_sec - 1):
            b0 = i * n_ring
            b1 = (i + 1) * n_ring
            for j in range(n_ring):
                j1 = (j + 1) % n_ring
                faces.append([b0 + j, b0 + j1, b1 + j1])
                faces.append([b0 + j, b1 + j1, b1 + j])

        return verts, np.array(faces)
