"""Supersonic air intake geometry generation.

Supports side-mounted (F-16 style) intakes with compression ramp,
boundary-layer diverter (splitter plate), and smooth superellipse
cross-section transition from rectangular capture to circular engine face.
"""

from __future__ import annotations

import math

import numpy as np

from fightercad.parameters import IntakeParams


class IntakeBuilder:
    """Build supersonic air intake geometry."""

    def __init__(self, params: IntakeParams, n_sections: int = 16, n_ring: int = 32):
        self.p = params
        self.n_sections = n_sections
        self.n_ring = n_ring
        self.section_points: list[np.ndarray] = []

    def _make_superellipse_section(
        self,
        a: float,
        b: float,
        n_exp: float,
        x: float,
        y_offset: float,
        z_offset: float,
    ) -> np.ndarray:
        """Create a superellipse cross-section ring.

        Parameters
        ----------
        a, b : float
            Semi-axes (horizontal, vertical).
        n_exp : float
            Superellipse exponent (2.0 = ellipse/circle, >2 = rounded rect).
        x : float
            Axial position.
        y_offset, z_offset : float
            Center offsets.
        """
        theta = np.linspace(0, 2 * math.pi, self.n_ring, endpoint=False)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        denom = (np.abs(cos_t) ** n_exp + np.abs(sin_t) ** n_exp) ** (1.0 / n_exp)
        py = a * cos_t / denom + y_offset
        pz = b * sin_t / denom + z_offset
        return np.column_stack([np.full(self.n_ring, x), py, pz])

    def build_single(
        self, side: float = 1.0, fuselage_radius: float = 0.9
    ) -> list[np.ndarray]:
        """Build one intake (left or right side).

        Parameters
        ----------
        side : float
            +1.0 for right, -1.0 for left.
        fuselage_radius : float
            Fuselage horizontal half-width at intake station (from aircraft.py).

        Returns
        -------
        list[np.ndarray]
            List of (N, 3) section rings from capture to engine face.
        """
        p = self.p
        capture_area = p.capture_area_m2
        aspect = p.capture_aspect_ratio
        duct_length = p.duct_length_m

        # Capture cross-section dimensions
        h_cap = math.sqrt(capture_area / aspect)
        w_cap = capture_area / h_cap

        # Engine face (circular, full area preserved)
        r_engine = math.sqrt(capture_area / math.pi)

        # Boundary layer diverter offset
        bld = p.boundary_layer_diverter_mm / 1000.0
        y_base = (fuselage_radius + bld) * side

        # Compression ramp
        ramp_rad = math.radians(p.ramp_angle_deg)
        ramp_amplitude = 0.3 * duct_length * math.tan(ramp_rad)

        sections = []
        x_stations = np.linspace(0, duct_length, self.n_sections)

        for i, dx in enumerate(x_stations):
            t = dx / duct_length  # 0 at capture, 1 at engine face

            # Smooth cosine parameter for all transitions
            t_smooth = 0.5 * (1.0 - math.cos(t * math.pi))

            # Superellipse exponent: 2.5 (rounded rect) → 2.0 (circle)
            n_exp = 2.5 - 0.5 * t_smooth

            # Cross-section dimensions: capture → engine
            a = w_cap / 2.0 * (1.0 - t_smooth) + r_engine * t_smooth
            b = h_cap / 2.0 * (1.0 - t_smooth) + r_engine * t_smooth

            # Gentle inward curve (max 15% toward centerline)
            y_curr = y_base * (1.0 - 0.15 * t_smooth)

            # Compression ramp: smooth fade using sin²
            ramp_z = -ramp_amplitude * math.sin(math.pi * t / 2.0) ** 2 * (1.0 - t)

            sec = self._make_superellipse_section(a, b, n_exp, dx, y_curr, ramp_z)
            sections.append(sec)

        self.section_points = sections
        return sections

    def build_bld_plate(
        self, side: float = 1.0, fuselage_radius: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build boundary layer diverter (splitter plate) geometry.

        Returns (vertices, faces) for a thin plate between fuselage and intake.
        """
        p = self.p
        bld = p.boundary_layer_diverter_mm / 1000.0
        if bld <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        h_cap = math.sqrt(p.capture_area_m2 / p.capture_aspect_ratio)
        plate_length = 0.35 * p.duct_length_m  # plate covers front 35% of duct
        half_h = h_cap / 2.0 * 0.9  # slightly smaller than capture

        y_inner = fuselage_radius * side
        y_outer = (fuselage_radius + bld) * side

        # 4 corners × 2 (front and back) = 8 vertices
        # Front face (x=0), Back face (x=plate_length)
        verts = np.array([
            # Front face
            [0.0, y_inner, -half_h],
            [0.0, y_inner, half_h],
            [0.0, y_outer, half_h],
            [0.0, y_outer, -half_h],
            # Back face
            [plate_length, y_inner, -half_h],
            [plate_length, y_inner, half_h],
            [plate_length, y_outer, half_h],
            [plate_length, y_outer, -half_h],
        ])

        # 6 faces × 2 triangles = 12 triangles (closed box)
        faces = np.array([
            # Outer face (facing away from fuselage)
            [3, 2, 6], [3, 6, 7],
            # Inner face (facing fuselage)
            [0, 4, 5], [0, 5, 1],
            # Top
            [1, 5, 6], [1, 6, 2],
            # Bottom
            [0, 3, 7], [0, 7, 4],
            # Front cap
            [0, 1, 2], [0, 2, 3],
            # Back cap
            [4, 7, 6], [4, 6, 5],
        ])

        return verts, faces

    def build(
        self, fuselage_radius: float = 0.9
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Build both intakes (left and right).

        Returns
        -------
        tuple
            (right_sections, left_sections)
        """
        right = self.build_single(side=1.0, fuselage_radius=fuselage_radius)
        left = self.build_single(side=-1.0, fuselage_radius=fuselage_radius)
        return right, left

    def get_mesh(
        self, sections: list[np.ndarray] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate triangle mesh from intake sections with endcaps."""
        if sections is None:
            sections = self.section_points
        if not sections:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        n_sec = len(sections)
        n_ring = self.n_ring
        verts = np.vstack(sections)

        faces = []
        # Duct surface: connect adjacent section rings
        for i in range(n_sec - 1):
            b0 = i * n_ring
            b1 = (i + 1) * n_ring
            for j in range(n_ring):
                j1 = (j + 1) % n_ring
                faces.append([b0 + j, b0 + j1, b1 + j1])
                faces.append([b0 + j, b1 + j1, b1 + j])

        # Inlet endcap (fan triangulation from centroid)
        inlet_center = sections[0].mean(axis=0)
        inlet_idx = len(verts)
        verts = np.vstack([verts, inlet_center.reshape(1, 3)])
        for j in range(n_ring):
            j1 = (j + 1) % n_ring
            faces.append([inlet_idx, j1, j])

        # Outlet endcap
        outlet_center = sections[-1].mean(axis=0)
        outlet_idx = len(verts)
        verts = np.vstack([verts, outlet_center.reshape(1, 3)])
        base_last = (n_sec - 1) * n_ring
        for j in range(n_ring):
            j1 = (j + 1) % n_ring
            faces.append([outlet_idx, base_last + j, base_last + j1])

        return verts, np.array(faces)
