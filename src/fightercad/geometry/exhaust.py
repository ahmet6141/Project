"""Exhaust nozzle geometry generation — round and slot (stealth) shapes."""

from __future__ import annotations

import math

import numpy as np

from fightercad.parameters import ExhaustParams


class ExhaustBuilder:
    """Build exhaust nozzle geometry.

    Supports:
    - Round nozzle (convergent or convergent-divergent)
    - Slot nozzle (flat rectangular exit for stealth/IR reduction)

    Slot nozzle transitions from circular inlet to high-aspect-ratio
    rectangular exit using superellipse cross-sections.
    """

    def __init__(self, params: ExhaustParams, n_sections: int = 16, n_ring: int = 32):
        self.p = params
        self.n_sections = n_sections
        self.n_ring = n_ring
        self.section_points: list[np.ndarray] = []

    def build(self) -> list[np.ndarray]:
        """Generate nozzle cross-sections."""
        p = self.p
        L = p.nozzle_length_m
        if L <= 0:
            self.section_points = []
            return []

        if p.nozzle_shape == "slot":
            return self._build_slot()
        return self._build_round()

    def _build_round(self) -> list[np.ndarray]:
        """Generate round (axisymmetric) nozzle sections."""
        p = self.p
        r_inlet = p.exit_diameter_m / 2.0 * 1.2
        r_throat = p.throat_diameter_m / 2.0
        r_exit = p.exit_diameter_m / 2.0
        L = p.nozzle_length_m

        is_cd = p.nozzle_type == "convergent_divergent"
        throat_frac = 0.4 if is_cd else 1.0
        x_stations = np.linspace(0, L, self.n_sections)
        theta = np.linspace(0, 2 * math.pi, self.n_ring, endpoint=False)

        self.section_points = []
        for x in x_stations:
            t = x / L
            if t <= throat_frac:
                s = t / throat_frac
                r_target = r_throat if is_cd else r_exit
                r = r_inlet + (r_target - r_inlet) * (0.5 - 0.5 * math.cos(math.pi * s))
            else:
                s = (t - throat_frac) / (1.0 - throat_frac)
                r = r_throat + (r_exit - r_throat) * (s ** 0.8)

            py = r * np.cos(theta)
            pz = r * np.sin(theta)
            pts = np.column_stack([np.full(self.n_ring, x), py, pz])
            self.section_points.append(pts)

        return self.section_points

    def _build_slot(self) -> list[np.ndarray]:
        """Generate slot nozzle sections.

        Transitions from circular inlet to high-aspect-ratio rectangular exit
        using superellipse cross-sections. The superellipse exponent increases
        from 2.0 (circle) to ~3.5 (rounded rectangle) along the nozzle length.
        """
        p = self.p
        L = p.nozzle_length_m
        ar = p.slot_aspect_ratio  # exit width/height ratio

        # Equivalent areas: circular inlet → slot exit
        r_inlet = p.exit_diameter_m / 2.0 * 1.2
        r_exit = p.exit_diameter_m / 2.0

        # Slot exit dimensions: preserve cross-section area
        # Area_circle = π * r² → slot area ≈ w * h (approximately)
        # w = ar * h, so: π*r² ≈ ar * h² → h = r*√(π/ar), w = ar*h
        area_exit = math.pi * r_exit ** 2
        h_exit = math.sqrt(area_exit / ar)
        w_exit = ar * h_exit

        # Throat (convergent): smaller area
        r_throat = p.throat_diameter_m / 2.0
        area_throat = math.pi * r_throat ** 2
        h_throat = math.sqrt(area_throat / (ar * 0.6 + 0.4))  # less squashed at throat
        w_throat = (ar * 0.6 + 0.4) * h_throat

        is_cd = p.nozzle_type == "convergent_divergent"
        throat_frac = 0.4 if is_cd else 1.0

        x_stations = np.linspace(0, L, self.n_sections)
        theta = np.linspace(0, 2 * math.pi, self.n_ring, endpoint=False)

        self.section_points = []
        for x in x_stations:
            t = x / L  # normalized position

            # Compute local half-axes and superellipse exponent
            if t <= throat_frac:
                s = t / throat_frac
                s_smooth = 0.5 - 0.5 * math.cos(math.pi * s)
                if is_cd:
                    a = r_inlet + (w_throat / 2 - r_inlet) * s_smooth
                    b = r_inlet + (h_throat / 2 - r_inlet) * s_smooth
                else:
                    a = r_inlet + (w_exit / 2 - r_inlet) * s_smooth
                    b = r_inlet + (h_exit / 2 - r_inlet) * s_smooth
                # Exponent: 2.0 (circle) → 3.0 (rounded rect)
                n_exp = 2.0 + 1.0 * s_smooth
            else:
                s = (t - throat_frac) / (1.0 - throat_frac)
                s_smooth = s ** 0.8
                a = w_throat / 2 + (w_exit / 2 - w_throat / 2) * s_smooth
                b = h_throat / 2 + (h_exit / 2 - h_throat / 2) * s_smooth
                n_exp = 3.0 + 0.5 * s_smooth  # gets more rectangular

            # Superellipse ring: |y/a|^n + |z/b|^n = 1
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            denom = (np.abs(cos_t) ** n_exp + np.abs(sin_t) ** n_exp) ** (1.0 / n_exp)
            denom = np.maximum(denom, 1e-10)
            py = a * cos_t / denom
            pz = b * sin_t / denom

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
