"""Supersonic air intake geometry generation.

Supports side-mounted (F-16 style) intakes with compression ramp,
boundary-layer diverter (splitter plate), and smooth superellipse
cross-section transition from rectangular capture to circular engine face.
Also supports a single chin-mounted intake under the fuselage.
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
        self, side: float = 1.0, fuselage_radius: float = 0.9,
        fuselage_radius_v: float | None = None,
    ) -> list[np.ndarray]:
        """Build one intake (left, right, or chin).

        Parameters
        ----------
        side : float
            +1.0 for right, -1.0 for left, 0.0 for chin (centerline).
        fuselage_radius : float
            Fuselage horizontal half-width at intake station.
        fuselage_radius_v : float or None
            Fuselage vertical half-height at intake station (used for chin).

        Returns
        -------
        list[np.ndarray]
            List of (N, 3) section rings from capture to engine face.
        """
        if self.p.intake_type == "chin":
            rv = fuselage_radius_v if fuselage_radius_v is not None else fuselage_radius
            return self._build_chin(rv)
        return self._build_side_mounted(side, fuselage_radius)

    def _build_side_mounted(
        self, side: float, fuselage_radius: float
    ) -> list[np.ndarray]:
        """Build a side-mounted intake."""
        p = self.p
        capture_area = p.capture_area_m2
        aspect = p.capture_aspect_ratio
        duct_length = p.duct_length_m

        h_cap = math.sqrt(capture_area / aspect)
        w_cap = capture_area / h_cap
        r_engine = math.sqrt(capture_area / math.pi)

        bld = p.boundary_layer_diverter_mm / 1000.0
        y_base = (fuselage_radius + bld) * side

        ramp_rad = math.radians(p.ramp_angle_deg)
        ramp_amplitude = 0.3 * duct_length * math.tan(ramp_rad)

        sections = []
        x_stations = np.linspace(0, duct_length, self.n_sections)

        for i, dx in enumerate(x_stations):
            t = dx / duct_length
            t_smooth = 0.5 * (1.0 - math.cos(t * math.pi))

            n_exp = 2.5 - 0.5 * t_smooth
            a = w_cap / 2.0 * (1.0 - t_smooth) + r_engine * t_smooth
            b = h_cap / 2.0 * (1.0 - t_smooth) + r_engine * t_smooth

            y_curr = y_base * (1.0 - 0.15 * t_smooth)
            ramp_z = -ramp_amplitude * math.sin(math.pi * t / 2.0) ** 2 * (1.0 - t)

            sec = self._make_superellipse_section(a, b, n_exp, dx, y_curr, ramp_z)
            sections.append(sec)

        self.section_points = sections
        return sections

    def _build_chin(self, fuselage_radius_v: float) -> list[np.ndarray]:
        """Build a single chin-mounted intake under the fuselage.

        The intake is centered on the y=0 plane, positioned below the
        fuselage (negative z).  The duct curves upward toward the engine
        face at the aircraft centerline.
        """
        p = self.p
        capture_area = p.capture_area_m2
        aspect = p.capture_aspect_ratio
        duct_length = p.duct_length_m

        # Capture face: wider than tall (chin style)
        h_cap = math.sqrt(capture_area / aspect)
        w_cap = capture_area / h_cap

        # Engine face (circular)
        r_engine = math.sqrt(capture_area / math.pi)

        # BLD offset below fuselage
        bld = p.boundary_layer_diverter_mm / 1000.0
        z_base = -(fuselage_radius_v + bld)

        # Compression ramp
        ramp_rad = math.radians(p.ramp_angle_deg)
        ramp_amplitude = 0.25 * duct_length * math.tan(ramp_rad)

        sections = []
        x_stations = np.linspace(0, duct_length, self.n_sections)

        for i, dx in enumerate(x_stations):
            t = dx / duct_length
            t_smooth = 0.5 * (1.0 - math.cos(t * math.pi))

            # Superellipse exponent: 2.5 (rounded rect) → 2.0 (circle)
            n_exp = 2.5 - 0.5 * t_smooth

            # Cross-section transition: capture → engine face
            a = w_cap / 2.0 * (1.0 - t_smooth) + r_engine * t_smooth
            b = h_cap / 2.0 * (1.0 - t_smooth) + r_engine * t_smooth

            # Duct curves upward: z goes from z_base toward ~40% of z_base
            # (engine face closer to centerline)
            z_curr = z_base * (1.0 - 0.6 * t_smooth)

            # Compression ramp: upward bump (positive z) in forward section
            ramp_z = ramp_amplitude * math.sin(math.pi * t / 2.0) ** 2 * (1.0 - t)

            sec = self._make_superellipse_section(
                a, b, n_exp, dx, 0.0, z_curr + ramp_z
            )
            sections.append(sec)

        self.section_points = sections
        return sections

    def build_bld_plate(
        self, side: float = 1.0, fuselage_radius: float = 0.9,
        fuselage_radius_v: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build boundary layer diverter (splitter plate) geometry.

        Returns (vertices, faces) for a thin plate between fuselage and intake.
        """
        p = self.p
        bld = p.boundary_layer_diverter_mm / 1000.0
        if bld <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        h_cap = math.sqrt(p.capture_area_m2 / p.capture_aspect_ratio)
        w_cap = p.capture_area_m2 / h_cap
        plate_length = 0.35 * p.duct_length_m
        half_h = h_cap / 2.0 * 0.9

        if p.intake_type == "chin":
            return self._build_chin_bld_plate(
                fuselage_radius_v or fuselage_radius,
                bld, w_cap, half_h, plate_length,
            )

        # Side-mounted BLD plate (original)
        y_inner = fuselage_radius * side
        y_outer = (fuselage_radius + bld) * side

        verts = np.array([
            [0.0, y_inner, -half_h],
            [0.0, y_inner, half_h],
            [0.0, y_outer, half_h],
            [0.0, y_outer, -half_h],
            [plate_length, y_inner, -half_h],
            [plate_length, y_inner, half_h],
            [plate_length, y_outer, half_h],
            [plate_length, y_outer, -half_h],
        ])

        faces = np.array([
            [3, 2, 6], [3, 6, 7],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [0, 3, 7], [0, 7, 4],
            [0, 1, 2], [0, 2, 3],
            [4, 7, 6], [4, 6, 5],
        ])

        return verts, faces

    def _build_chin_bld_plate(
        self,
        fuselage_radius_v: float,
        bld: float,
        w_cap: float,
        half_h: float,
        plate_length: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build BLD plate for chin intake (horizontal plate under fuselage)."""
        half_w = w_cap / 2.0 * 0.9
        z_fuse = -fuselage_radius_v  # fuselage bottom
        z_intake = -(fuselage_radius_v + bld)  # intake top

        verts = np.array([
            # Front face
            [0.0, -half_w, z_fuse],
            [0.0, half_w, z_fuse],
            [0.0, half_w, z_intake],
            [0.0, -half_w, z_intake],
            # Back face
            [plate_length, -half_w, z_fuse],
            [plate_length, half_w, z_fuse],
            [plate_length, half_w, z_intake],
            [plate_length, -half_w, z_intake],
        ])

        faces = np.array([
            # Top (fuselage side)
            [0, 1, 5], [0, 5, 4],
            # Bottom (intake side)
            [3, 7, 6], [3, 6, 2],
            # Left
            [0, 4, 7], [0, 7, 3],
            # Right
            [1, 2, 6], [1, 6, 5],
            # Front cap
            [0, 3, 2], [0, 2, 1],
            # Back cap
            [4, 5, 6], [4, 6, 7],
        ])

        return verts, faces

    def build(
        self, fuselage_radius: float = 0.9,
        fuselage_radius_v: float | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
        """Build intake(s).

        For chin type: returns (chin_sections, None).
        For side_mounted: returns (right_sections, left_sections).
        """
        if self.p.intake_type == "chin":
            chin = self.build_single(
                side=0.0, fuselage_radius=fuselage_radius,
                fuselage_radius_v=fuselage_radius_v,
            )
            return chin, None

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
