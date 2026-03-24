"""Component blending and wing-fuselage fairing geometry.

Generates mesh-based junction geometry between wing root and fuselage surface.
Uses actual fuselage section ring data for accurate surface matching and
quintic (C²) interpolation for curvature-continuous transitions.
"""

from __future__ import annotations

import math

import numpy as np

from fightercad.parameters import BlendingParams


def _quintic_smoothstep(t: float) -> float:
    """Quintic smoothstep: 6t⁵ - 15t⁴ + 10t³, C² continuous."""
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class BlendingOperator:
    """Build wing-fuselage junction (fairing) and strake geometry."""

    def __init__(self, params: BlendingParams):
        self.p = params

    def build_wing_fuselage_fairing(
        self,
        wing_root_sections: np.ndarray,
        fuselage_sections: list,
        wing_x_offset: float,
        fuselage_radius_h: float,
        fuselage_radius_v: float,
        side: float = 1.0,
        n_blend_steps: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a smooth fairing mesh between wing root and fuselage surface.

        Uses actual fuselage section ring data to find the real surface
        profile, then interpolates to the wing root with C² continuity.

        Parameters
        ----------
        wing_root_sections : np.ndarray (N, 3)
            Wing root airfoil points in aircraft coordinates.
        fuselage_sections : list[FuselageSection]
            Fuselage cross-sections with actual ring point data.
        wing_x_offset : float
            Wing root LE x-position on fuselage.
        fuselage_radius_h : float
            Fuselage horizontal half-width at wing root station (fallback).
        fuselage_radius_v : float
            Fuselage vertical half-height at wing root station (fallback).
        side : float
            +1.0 for right, -1.0 for left.
        n_blend_steps : int
            Number of intermediate fairing cross-sections.

        Returns
        -------
        verts : np.ndarray (V, 3)
        faces : np.ndarray (F, 3)
        """
        if self.p.blending_mode == "sharp":
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        fillet_r = self.p.root_fillet_radius_mm / 1000.0
        fairing_w = self.p.fairing_width_mm / 1000.0

        wing_pts = wing_root_sections.copy()
        n_pts = len(wing_pts)

        if n_pts < 4:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        # Get fuselage surface profile using actual ring data
        fuse_profile = self._fuselage_surface_profile(
            wing_pts, fuselage_sections, fuselage_radius_h, fuselage_radius_v, side,
        )

        # Generate blending sections from fuselage surface to wing root
        # with C² quintic interpolation and controlled fillet bulge
        blend_sections = []
        for i in range(n_blend_steps + 1):
            t = i / n_blend_steps

            # C² interpolation between fuselage and wing profiles
            t_smooth = _quintic_smoothstep(t)

            # Fillet bulge: sin²(πt) envelope — zero derivative at both ends
            # Scale by fairing width for proportional bulge
            bulge_env = math.sin(math.pi * t) ** 2
            # Reduce fillet for thin surfaces (wing-like shapes)
            wing_thickness = wing_pts[:, 2].max() - wing_pts[:, 2].min()
            fuse_height = fuse_profile[:, 2].max() - fuse_profile[:, 2].min()
            thickness_ratio = min(wing_thickness, fuse_height) / max(0.01, max(wing_thickness, fuse_height))
            # Wider fairing → more fillet, thinner profile → less fillet
            width_scale = fairing_w / max(0.05, fairing_w)
            fillet_offset = fillet_r * width_scale * bulge_env * (0.3 + 0.7 * thickness_ratio)

            # Interpolate between fuselage and wing profiles
            blended = fuse_profile * (1.0 - t_smooth) + wing_pts * t_smooth

            # Apply fillet bulge outward
            if fillet_offset > 0.001 and n_pts > 2:
                normals = self._compute_outward_normals(blended, side)
                blended += normals * fillet_offset

            blend_sections.append(blended)

        verts, faces = self._sections_to_mesh(blend_sections)
        return verts, faces

    def build_strake(
        self,
        fuselage_length: float,
        wing_root_x: float,
        fuselage_radius: float,
        side: float = 1.0,
        n_chord: int = 15,
        n_span: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate LEX/strake mesh if strake mode is active."""
        if self.p.blending_mode != "strake" or self.p.strake_length_m <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        sweep_rad = math.radians(self.p.strake_sweep_deg)
        L = self.p.strake_length_m

        x_start = wing_root_x - L
        x_end = wing_root_x

        x_arr = np.linspace(x_start, x_end, n_chord)
        frac = (x_arr - x_start) / L

        y_inner = fuselage_radius
        y_outer = fuselage_radius + frac * L * math.tan(math.pi / 2 - sweep_rad)

        thickness = 0.005 * L

        verts_list = []
        for i in range(n_chord):
            y_span = np.linspace(y_inner, y_outer[i], n_span) * side
            for j in range(n_span):
                verts_list.append([x_arr[i], y_span[j], thickness * 0.5])
            for j in range(n_span):
                verts_list.append([x_arr[i], y_span[j], -thickness * 0.5])

        verts = np.array(verts_list)
        n_per_section = n_span * 2

        faces = []
        for i in range(n_chord - 1):
            b0 = i * n_per_section
            b1 = (i + 1) * n_per_section
            for j in range(n_span - 1):
                v0, v1, v2, v3 = b0 + j, b0 + j + 1, b1 + j + 1, b1 + j
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])
            off = n_span
            for j in range(n_span - 1):
                v0 = b0 + off + j
                v1 = b0 + off + j + 1
                v2 = b1 + off + j + 1
                v3 = b1 + off + j
                faces.append([v0, v2, v1])
                faces.append([v0, v3, v2])

        return (verts, np.array(faces)) if faces else (verts, np.zeros((0, 3), dtype=int))

    def _fuselage_surface_profile(
        self,
        wing_pts: np.ndarray,
        fuselage_sections: list,
        radius_h: float,
        radius_v: float,
        side: float,
    ) -> np.ndarray:
        """Project wing root points onto the fuselage surface.

        Uses actual fuselage section ring data to find the real surface
        point at each chordwise station. Falls back to ellipse math
        only when ring data is unavailable.
        """
        profile = wing_pts.copy()
        n_pts = len(profile)

        if not fuselage_sections:
            # Fallback: ellipse projection
            return self._ellipse_projection(profile, radius_h, radius_v, side)

        # Build sorted lookup of sections by x
        sec_by_x = sorted(fuselage_sections, key=lambda s: s.x)
        sec_x_arr = np.array([s.x for s in sec_by_x])

        for i in range(n_pts):
            x_pt = profile[i, 0]
            z_target = profile[i, 2]

            # Find the two bracketing fuselage sections
            rh, rv = radius_h, radius_v
            y_surface = self._find_surface_y_from_sections(
                sec_by_x, sec_x_arr, x_pt, z_target, side, rh, rv,
            )

            profile[i, 1] = y_surface
            # Clamp z to fuselage envelope
            rv_local = self._get_rv_at_x(sec_by_x, sec_x_arr, x_pt, rv)
            profile[i, 2] = np.clip(z_target, -rv_local * 0.95, rv_local * 0.95)

        return profile

    def _find_surface_y_from_sections(
        self,
        sec_by_x: list,
        sec_x_arr: np.ndarray,
        x: float,
        z_target: float,
        side: float,
        rh_fallback: float,
        rv_fallback: float,
    ) -> float:
        """Find the fuselage surface y-coordinate at given (x, z).

        Samples actual ring points from the two nearest fuselage sections
        and interpolates to find the surface y at the target z.
        """
        n_sec = len(sec_by_x)
        if n_sec == 0:
            return self._ellipse_y(rh_fallback, rv_fallback, z_target, side)

        # Find bracketing sections
        idx = np.searchsorted(sec_x_arr, x)
        idx = max(1, min(idx, n_sec - 1))
        sec_a = sec_by_x[idx - 1]
        sec_b = sec_by_x[idx]

        # Interpolation fraction between sections
        dx = sec_b.x - sec_a.x
        frac = (x - sec_a.x) / dx if dx > 1e-10 else 0.5

        # Find y on each section's ring at the target z
        y_a = self._ring_y_at_z(sec_a.points, z_target, side)
        y_b = self._ring_y_at_z(sec_b.points, z_target, side)

        # Interpolate y between sections
        return y_a * (1.0 - frac) + y_b * frac

    @staticmethod
    def _ring_y_at_z(
        ring_pts: np.ndarray, z_target: float, side: float,
    ) -> float:
        """Find the y-coordinate on the fuselage ring at a given z.

        Searches the ring points on the correct side (left/right) and
        interpolates between the two closest points to z_target.
        """
        # Filter ring points on the correct side
        if side > 0:
            mask = ring_pts[:, 1] >= -0.001  # right side (y >= 0)
        else:
            mask = ring_pts[:, 1] <= 0.001  # left side (y <= 0)

        side_pts = ring_pts[mask]
        if len(side_pts) < 2:
            return 0.0

        # Sort by z to find the two points closest to z_target
        z_vals = side_pts[:, 2]
        y_vals = side_pts[:, 1]

        # Find the outermost (max |y|) point at each z level
        # We want the surface point, not interior points
        # For an elliptical/superellipse ring, |y| is maximized at z=0
        # and decreases toward top/bottom

        # Find closest z values
        z_diffs = np.abs(z_vals - z_target)
        sorted_idx = np.argsort(z_diffs)

        # Use the 2-4 closest points and pick the outermost
        n_near = min(4, len(sorted_idx))
        near_idx = sorted_idx[:n_near]
        near_y = y_vals[near_idx]

        # Return the outermost y (max |y|) — this is the surface
        if side > 0:
            return float(np.max(near_y))
        else:
            return float(np.min(near_y))

    def _get_rv_at_x(
        self, sec_by_x: list, sec_x_arr: np.ndarray, x: float, rv_fallback: float,
    ) -> float:
        """Get interpolated fuselage rv at x position."""
        n_sec = len(sec_by_x)
        if n_sec == 0:
            return rv_fallback
        idx = np.searchsorted(sec_x_arr, x)
        idx = max(1, min(idx, n_sec - 1))
        a, b = sec_by_x[idx - 1], sec_by_x[idx]
        dx = b.x - a.x
        frac = (x - a.x) / dx if dx > 1e-10 else 0.5
        return a.radius_v * (1.0 - frac) + b.radius_v * frac

    @staticmethod
    def _ellipse_y(rh: float, rv: float, z: float, side: float) -> float:
        """Compute y on ellipse at given z (fallback)."""
        z_clamped = max(-rv * 0.95, min(rv * 0.95, z))
        if rv > 0:
            y = rh * math.sqrt(max(0.0, 1.0 - (z_clamped / rv) ** 2))
        else:
            y = rh
        return y * side

    @staticmethod
    def _ellipse_projection(
        profile: np.ndarray, rh: float, rv: float, side: float,
    ) -> np.ndarray:
        """Fallback: project points onto ellipse surface."""
        for i in range(len(profile)):
            z = profile[i, 2]
            z_clamped = np.clip(z, -rv * 0.95, rv * 0.95)
            if rv > 0:
                y_fuse = rh * math.sqrt(max(0.0, 1.0 - (z_clamped / rv) ** 2))
            else:
                y_fuse = rh
            profile[i, 1] = y_fuse * side
            profile[i, 2] = z_clamped
        return profile

    def _compute_outward_normals(
        self, pts: np.ndarray, side: float,
    ) -> np.ndarray:
        """Compute approximate outward normals for a cross-section.

        Uses the tangent vector between adjacent points to compute
        perpendicular normals in the YZ plane, ensuring they point
        outward from the aircraft centerline.
        """
        n = len(pts)
        normals = np.zeros_like(pts)

        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            tangent = pts[next_i] - pts[prev_i]
            # Normal perpendicular to tangent in YZ plane
            normal = np.array([0.0, -tangent[2], tangent[1]])
            length = np.linalg.norm(normal)
            if length > 1e-10:
                normal /= length
            # Ensure outward direction (away from centerline)
            if normal[1] * side < 0:
                normal *= -1
            normals[i] = normal

        return normals

    def _sections_to_mesh(
        self, sections: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a list of cross-section arrays to a triangle mesh.

        Uses modular indexing to close each ring (last point → first point).
        """
        if len(sections) < 2:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        n_sec = len(sections)
        n_pts = len(sections[0])
        verts = np.vstack(sections)

        faces = []
        for i in range(n_sec - 1):
            b0 = i * n_pts
            b1 = (i + 1) * n_pts
            for j in range(n_pts):
                j1 = (j + 1) % n_pts
                v0 = b0 + j
                v1 = b0 + j1
                v2 = b1 + j1
                v3 = b1 + j
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        return (verts, np.array(faces)) if faces else (verts, np.zeros((0, 3), dtype=int))

    def get_blend_info(self) -> dict:
        """Return a summary of all blending parameters for reporting."""
        return {
            "root_fillet_mm": self.p.root_fillet_radius_mm,
            "le_fillet_mm": self.p.leading_edge_fillet_mm,
            "te_fillet_mm": self.p.trailing_edge_fillet_mm,
            "mode": self.p.blending_mode,
            "strake_length_m": self.p.strake_length_m,
            "strake_sweep_deg": self.p.strake_sweep_deg,
            "fairing_width_mm": self.p.fairing_width_mm,
            "stab_fillet_mm": self.p.stabilizer_root_fillet_mm,
            "intake_fillet_mm": self.p.intake_fuselage_fillet_mm,
        }
