"""Supersonic air intake geometry generation.

Supports side-mounted (F-16 style) intakes with compression ramp
and boundary-layer diverter.
"""

from __future__ import annotations

import math

import numpy as np

from fightercad.parameters import IntakeParams


class IntakeBuilder:
    """Build supersonic air intake geometry."""

    def __init__(self, params: IntakeParams, n_sections: int = 10, n_ring: int = 24):
        self.p = params
        self.n_sections = n_sections
        self.n_ring = n_ring
        self.section_points: list[np.ndarray] = []

    def _make_rect_section(
        self, width: float, height: float, x: float, y_offset: float, z_offset: float
    ) -> np.ndarray:
        """Create a rounded-rectangle cross-section ring."""
        theta = np.linspace(0, 2 * math.pi, self.n_ring, endpoint=False)
        # Superellipse for rounded rectangle
        n_exp = 2.5
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        denom = (np.abs(cos_t) ** n_exp + np.abs(sin_t) ** n_exp) ** (1.0 / n_exp)
        py = (width / 2) * cos_t / denom + y_offset
        pz = (height / 2) * sin_t / denom + z_offset
        return np.column_stack([np.full(self.n_ring, x), py, pz])

    def _make_circle_section(
        self, radius: float, x: float, y_offset: float, z_offset: float
    ) -> np.ndarray:
        """Create a circular cross-section ring."""
        theta = np.linspace(0, 2 * math.pi, self.n_ring, endpoint=False)
        py = radius * np.cos(theta) + y_offset
        pz = radius * np.sin(theta) + z_offset
        return np.column_stack([np.full(self.n_ring, x), py, pz])

    def build_single(self, side: float = 1.0) -> list[np.ndarray]:
        """Build one intake (left or right side).

        Parameters
        ----------
        side : float
            +1.0 for right, -1.0 for left.

        Returns
        -------
        list[np.ndarray]
            List of (N, 3) section rings from capture to engine face.
        """
        p = self.p
        capture_area = p.capture_area_m2
        # Capture cross-section dimensions (rectangular)
        aspect = 1.5  # width/height of capture
        h_cap = math.sqrt(capture_area / aspect)
        w_cap = capture_area / h_cap
        # Engine face (circular)
        r_engine = math.sqrt(capture_area / math.pi) * 0.9

        # Duct length (estimate)
        duct_length = 2.5  # meters, typical for fighter intake duct

        # Boundary layer diverter offset
        bld = p.boundary_layer_diverter_mm / 1000.0
        y_offset = (0.9 + bld) * side  # offset from centerline

        # Compression ramp angle
        ramp_rad = math.radians(p.ramp_angle_deg)

        sections = []
        x_stations = np.linspace(0, duct_length, self.n_sections)

        for i, dx in enumerate(x_stations):
            t = dx / duct_length  # 0 at capture, 1 at engine face

            # Transition from rectangle to circle
            if t < 0.3:
                # Rectangular capture zone with ramp
                ramp_z = -dx * math.tan(ramp_rad) * (1.0 - t / 0.3)
                sec = self._make_rect_section(
                    w_cap * (1.0 - 0.1 * t),
                    h_cap * (1.0 - 0.1 * t),
                    dx, y_offset, ramp_z,
                )
            else:
                # Transition to circular
                blend = (t - 0.3) / 0.7
                r_curr = r_engine * blend + math.sqrt(w_cap * h_cap / math.pi) * (1.0 - blend)
                y_curr = y_offset * (1.0 - blend * 0.3)  # duct curves inward
                sec = self._make_circle_section(r_curr, dx, y_curr, 0.0)

            sections.append(sec)

        self.section_points = sections
        return sections

    def build(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Build both intakes (left and right).

        Returns
        -------
        tuple
            (right_sections, left_sections)
        """
        right = self.build_single(side=1.0)
        left = self.build_single(side=-1.0)
        return right, left

    def get_mesh(self, sections: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate triangle mesh from intake sections."""
        if sections is None:
            sections = self.section_points
        if not sections:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        n_sec = len(sections)
        n_ring = self.n_ring
        verts = np.vstack(sections)

        faces = []
        for i in range(n_sec - 1):
            b0 = i * n_ring
            b1 = (i + 1) * n_ring
            for j in range(n_ring):
                j1 = (j + 1) % n_ring
                faces.append([b0 + j, b0 + j1, b1 + j1])
                faces.append([b0 + j, b1 + j1, b1 + j])

        return verts, np.array(faces)
