"""Vertical stabilizer and V-tail geometry generation."""

from __future__ import annotations

import math

import numpy as np

from fightercad.parameters import VerticalStabilizerParams
from fightercad.geometry.primitives import get_airfoil, airfoil_to_3d


class StabilizerBuilder:
    """Build vertical stabilizer or V-tail geometry.

    Supports three configurations:
    - Single vertical fin (cant_deg = 0)
    - Single canted fin (cant_deg != 0)
    - V-tail: dual canted fins (v_tail = True)
    - Tailless: no stabilizer (tailless = True)
    """

    def __init__(self, params: VerticalStabilizerParams, n_sections: int = 10, n_af_pts: int = 80):
        self.p = params
        self.n_sections = n_sections
        self.n_af = n_af_pts
        self.section_points: list[np.ndarray] = []
        self._fuselage_radius_v: float = 0.0
        self._fuselage_radius_h: float = 0.0

    def build(
        self, side: float = 0.0,
        fuselage_radius_h: float = 0.0, fuselage_radius_v: float = 0.0,
    ) -> list[np.ndarray]:
        """Generate stabilizer sections.

        Parameters
        ----------
        side : float
            0.0 for single vertical, +1.0 for right V-tail fin,
            -1.0 for left V-tail fin.
        fuselage_radius_h : float
            Fuselage horizontal half-width at stabilizer station.
        fuselage_radius_v : float
            Fuselage vertical half-height at stabilizer station.

        Returns
        -------
        list[np.ndarray]
            List of (N, 3) arrays, one per spanwise (vertical) station.
        """
        p = self.p
        self._fuselage_radius_v = fuselage_radius_v
        self._fuselage_radius_h = fuselage_radius_h

        if p.tailless or p.area_m2 <= 0:
            self.section_points = []
            return []

        span = math.sqrt(p.aspect_ratio * p.area_m2)
        root_chord = 2.0 * p.area_m2 / (span * (1.0 + p.taper_ratio))
        tip_chord = root_chord * p.taper_ratio

        sweep_rad = math.radians(p.sweep_deg)

        # Determine cant angle (V-tail: both fins cant outward symmetrically)
        if p.v_tail and side != 0.0:
            cant_rad = math.radians(p.v_tail_cant_deg) * (1.0 if side > 0 else -1.0)
        else:
            cant_rad = math.radians(p.cant_deg)

        # Cosine spanwise distribution for better resolution at root/tip
        eta = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi / 2, self.n_sections)))
        z_stations = eta * span

        self.section_points = []

        for z in z_stations:
            frac = z / span if span > 0 else 0.0
            chord = root_chord * (1.0 - frac) + tip_chord * frac
            x_le = z * math.tan(sweep_rad)

            af_2d = get_airfoil("naca64a004", chord, le_radius_mm=2.0, num_points=self.n_af)

            n = len(af_2d)
            pts = np.zeros((n, 3))
            pts[:, 0] = af_2d[:, 0] + x_le
            # Apply cant rotation: y = span * sin(cant), z = span * cos(cant)
            pts[:, 1] = z * math.sin(cant_rad)
            pts[:, 2] = z * math.cos(cant_rad)
            pts[:, 1] += af_2d[:, 1]

            # Offset root to fuselage surface
            if fuselage_radius_v > 0:
                pts[:, 2] += fuselage_radius_v

            self.section_points.append(pts)

        return self.section_points

    def build_root_fillet(self, fillet_mm: float = 30.0, n_fillet: int = 6) -> tuple[np.ndarray, np.ndarray]:
        """Build a fillet collar at the stabilizer root-fuselage junction.

        Creates a smooth transition between the stabilizer root airfoil
        and the fuselage surface.

        Parameters
        ----------
        fillet_mm : float
            Fillet radius in mm.
        n_fillet : int
            Number of interpolation steps.
        """
        if not self.section_points or self._fuselage_radius_v <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        fillet_r = fillet_mm / 1000.0
        root_ring = self.section_points[0]
        n_pts = len(root_ring)

        # Project root airfoil onto fuselage surface
        fuse_ring = root_ring.copy()
        rv = self._fuselage_radius_v
        rh = self._fuselage_radius_h if self._fuselage_radius_h > 0 else rv

        for i in range(n_pts):
            y = fuse_ring[i, 1]
            # Project onto top of fuselage ellipse
            if abs(y) < rh * 0.99 and rh > 0:
                z_fuse = rv * math.sqrt(max(0, 1.0 - (y / rh) ** 2))
            else:
                z_fuse = 0.0
            fuse_ring[i, 2] = z_fuse

        # Interpolate fillet sections
        sections = []
        for k in range(n_fillet + 1):
            t = k / n_fillet
            t_smooth = 0.5 * (1.0 - math.cos(t * math.pi))
            blended = fuse_ring * (1.0 - t_smooth) + root_ring * t_smooth
            # Fillet bulge
            bulge = fillet_r * math.sin(t * math.pi)
            if bulge > 0:
                for i in range(n_pts):
                    nz = blended[i, 2] - rv * 0.5
                    ny = blended[i, 1]
                    nr = math.sqrt(ny * ny + nz * nz)
                    if nr > 1e-6:
                        blended[i, 1] += bulge * ny / nr * 0.2
                        blended[i, 2] += bulge * abs(nz) / nr * 0.2
            sections.append(blended)

        if len(sections) < 2:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        verts = np.vstack(sections)
        faces = []
        for i in range(len(sections) - 1):
            b0 = i * n_pts
            b1 = (i + 1) * n_pts
            for j in range(n_pts - 1):
                faces.append([b0 + j, b0 + j + 1, b1 + j + 1])
                faces.append([b0 + j, b1 + j + 1, b1 + j])

        return verts, np.array(faces)

    def get_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate triangle mesh."""
        if not self.section_points:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

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

        return (verts, np.array(faces)) if faces else (verts, np.zeros((0, 3), dtype=int))
