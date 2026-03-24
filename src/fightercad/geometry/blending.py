"""Component blending and wing-fuselage fairing geometry.

Generates mesh-based junction geometry between wing root and fuselage surface.
The fairing adapts automatically when wing or fuselage dimensions change.
"""

from __future__ import annotations

import math

import numpy as np

from fightercad.parameters import BlendingParams


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
        n_blend_steps: int = 12,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a smooth fairing mesh between wing root and fuselage surface.

        Creates intermediate cross-sections that interpolate between the
        fuselage surface profile and the wing root airfoil, producing a
        smooth fillet transition.

        Parameters
        ----------
        wing_root_sections : np.ndarray (N, 3)
            Wing root airfoil points in aircraft coordinates.
        fuselage_sections : list[FuselageSection]
            Fuselage cross-sections (used to find local radius).
        wing_x_offset : float
            Wing root LE x-position on fuselage.
        fuselage_radius_h : float
            Fuselage horizontal half-width at wing root station.
        fuselage_radius_v : float
            Fuselage vertical half-height at wing root station.
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

        fillet_r = self.p.root_fillet_radius_mm / 1000.0  # mm → m
        fairing_w = self.p.fairing_width_mm / 1000.0  # mm → m

        # Wing root profile (the innermost airfoil section)
        wing_pts = wing_root_sections.copy()
        n_pts = len(wing_pts)

        # Fuselage surface profile at wing junction using real section data
        fuse_profile = self._fuselage_surface_profile(
            wing_pts, fuselage_sections, fuselage_radius_h, fuselage_radius_v, side,
        )

        # Generate blending sections from fuselage surface to wing root
        # Using quintic (C2) Hermite interpolation for curvature continuity
        width_scale = fairing_w / 0.1  # normalized to 100mm baseline
        blend_sections = []
        for i in range(n_blend_steps + 1):
            t = i / n_blend_steps

            # Quintic smoothstep for C² continuity: f(t) = 6t⁵ - 15t⁴ + 10t³
            # This ensures zero 1st and 2nd derivatives at endpoints
            t_smooth = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3

            # Fillet bulge: smooth bell curve with C2 onset/exit
            # Uses sin²(πt) envelope which has zero derivative at ends
            bulge_env = math.sin(math.pi * t) ** 2
            fillet_offset = fillet_r * max(1.0, width_scale) * bulge_env

            # Interpolate each point between fuselage and wing profiles
            blended = fuse_profile * (1.0 - t_smooth) + wing_pts * t_smooth

            # Apply fillet bulge (outward normal direction)
            if fillet_offset > 0 and n_pts > 2:
                normals = self._compute_outward_normals(blended, side)
                blended += normals * fillet_offset

            blend_sections.append(blended)

        # Build mesh from blend sections (with ring closure)
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
        """Generate LEX/strake mesh if strake mode is active.

        Returns (vertices, faces) or empty arrays if not in strake mode.
        """
        if self.p.blending_mode != "strake" or self.p.strake_length_m <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        sweep_rad = math.radians(self.p.strake_sweep_deg)
        L = self.p.strake_length_m

        x_start = wing_root_x - L
        x_end = wing_root_x

        # Strake planform
        x_arr = np.linspace(x_start, x_end, n_chord)
        frac = (x_arr - x_start) / L

        # LE sweep determines how far outboard the strake extends
        y_inner = fuselage_radius
        y_outer = fuselage_radius + frac * L * math.tan(math.pi / 2 - sweep_rad)

        # Thin strake with slight thickness
        thickness = 0.005 * L  # 0.5% of strake length

        verts_list = []
        for i in range(n_chord):
            y_span = np.linspace(y_inner, y_outer[i], n_span) * side
            for j in range(n_span):
                # Upper surface
                verts_list.append([x_arr[i], y_span[j], thickness * 0.5])
            for j in range(n_span):
                # Lower surface
                verts_list.append([x_arr[i], y_span[j], -thickness * 0.5])

        verts = np.array(verts_list)
        n_per_section = n_span * 2  # upper + lower

        faces = []
        for i in range(n_chord - 1):
            b0 = i * n_per_section
            b1 = (i + 1) * n_per_section
            # Upper surface
            for j in range(n_span - 1):
                v0, v1, v2, v3 = b0 + j, b0 + j + 1, b1 + j + 1, b1 + j
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])
            # Lower surface
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

        Uses actual fuselage section data when available to find the real
        radius at each chordwise position. Falls back to the provided
        radius_h/radius_v if sections are empty.
        """
        profile = wing_pts.copy()

        # Build a lookup of fuselage x → (rh, rv) from actual sections
        fuse_x_rh_rv = []
        if fuselage_sections:
            for sec in fuselage_sections:
                fuse_x_rh_rv.append((sec.x, sec.radius_h, sec.radius_v))
            fuse_x_rh_rv.sort(key=lambda t: t[0])

        for i in range(len(profile)):
            x_pt = profile[i, 0]
            z = profile[i, 2]

            # Get fuselage radius at this x-position
            rh, rv = radius_h, radius_v
            if fuse_x_rh_rv:
                rh, rv = self._interpolate_fuselage_radius(fuse_x_rh_rv, x_pt)

            # Clamp z to fuselage envelope
            z_clamped = np.clip(z, -rv * 0.95, rv * 0.95)
            # Compute y on the fuselage ellipse: (y/rh)^2 + (z/rv)^2 = 1
            if rv > 0:
                y_fuse = rh * math.sqrt(max(0, 1.0 - (z_clamped / rv) ** 2))
            else:
                y_fuse = rh
            profile[i, 1] = y_fuse * side
            profile[i, 2] = z_clamped

        return profile

    @staticmethod
    def _interpolate_fuselage_radius(
        fuse_data: list[tuple[float, float, float]], x: float
    ) -> tuple[float, float]:
        """Linearly interpolate fuselage rh, rv at a given x position."""
        if not fuse_data:
            return 0.0, 0.0
        if x <= fuse_data[0][0]:
            return fuse_data[0][1], fuse_data[0][2]
        if x >= fuse_data[-1][0]:
            return fuse_data[-1][1], fuse_data[-1][2]

        for j in range(len(fuse_data) - 1):
            x0, rh0, rv0 = fuse_data[j]
            x1, rh1, rv1 = fuse_data[j + 1]
            if x0 <= x <= x1:
                frac = (x - x0) / (x1 - x0) if (x1 - x0) > 0 else 0.0
                return rh0 + frac * (rh1 - rh0), rv0 + frac * (rv1 - rv0)

        return fuse_data[-1][1], fuse_data[-1][2]

    def _compute_outward_normals(
        self, pts: np.ndarray, side: float
    ) -> np.ndarray:
        """Compute approximate outward normals for a 2D-ish cross-section."""
        n = len(pts)
        normals = np.zeros_like(pts)

        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            tangent = pts[next_i] - pts[prev_i]
            # Normal in YZ plane (perpendicular to tangent)
            normal = np.array([0.0, -tangent[2], tangent[1]])
            length = np.linalg.norm(normal)
            if length > 1e-10:
                normal /= length
            # Ensure outward direction
            if normal[1] * side < 0:
                normal *= -1
            normals[i] = normal

        return normals

    def _sections_to_mesh(
        self, sections: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a list of cross-section arrays to a triangle mesh.

        Uses modular indexing to close the ring (last point connects to first).
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
                j1 = (j + 1) % n_pts  # ring closure
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
