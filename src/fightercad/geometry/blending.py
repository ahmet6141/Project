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
        n_blend_steps: int = 6,
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

        # Fuselage surface profile at wing junction
        # Create an elliptical cut on the fuselage at the wing root y-position
        fuse_profile = self._fuselage_surface_profile(
            wing_pts, fuselage_radius_h, fuselage_radius_v, side,
        )

        # Generate blending sections from fuselage surface to wing root
        # Parameter t: 0 = fuselage surface, 1 = wing root
        blend_sections = []
        for i in range(n_blend_steps + 1):
            t = i / n_blend_steps

            # Smooth interpolation (cosine ease for tangent continuity)
            t_smooth = 0.5 * (1.0 - math.cos(t * math.pi))

            # Fillet offset: maximum at t=0.5 (middle of blend), zero at ends
            fillet_offset = fillet_r * math.sin(t * math.pi)

            # Interpolate between fuselage profile and wing profile
            blended = fuse_profile * (1.0 - t_smooth) + wing_pts * t_smooth

            # Apply fillet bulge (outward normal direction)
            if fillet_offset > 0 and n_pts > 2:
                normals = self._compute_outward_normals(blended, side)
                blended += normals * fillet_offset

            # Spanwise position: from fuselage surface to wing root
            y_fuse = fuselage_radius_h * side
            y_wing = wing_pts[:, 1].mean()
            y_blend = y_fuse + (y_wing - y_fuse) * t_smooth

            # Adjust fairing width
            width_factor = 1.0 + fairing_w * math.sin(t * math.pi)
            blended[:, 1] = y_blend * width_factor

            blend_sections.append(blended)

        # Build mesh from blend sections
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

        return verts, np.array(faces) if faces else (verts, np.zeros((0, 3), dtype=int))

    def _fuselage_surface_profile(
        self,
        wing_pts: np.ndarray,
        radius_h: float,
        radius_v: float,
        side: float,
    ) -> np.ndarray:
        """Project wing root points onto the fuselage surface.

        Creates a profile on the fuselage skin that matches the chordwise
        extent of the wing root airfoil.
        """
        profile = wing_pts.copy()

        for i in range(len(profile)):
            # Map each wing point's z-coordinate to the fuselage surface
            # at the given x-position, using the elliptical cross-section
            z = profile[i, 2]
            # Clamp z to fuselage envelope
            z_clamped = np.clip(z, -radius_v * 0.95, radius_v * 0.95)
            # Compute y on the fuselage ellipse: (y/rh)^2 + (z/rv)^2 = 1
            if radius_v > 0:
                y_fuse = radius_h * math.sqrt(max(0, 1.0 - (z_clamped / radius_v) ** 2))
            else:
                y_fuse = radius_h
            profile[i, 1] = y_fuse * side
            profile[i, 2] = z_clamped

        return profile

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
        """Convert a list of cross-section arrays to a triangle mesh."""
        if len(sections) < 2:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        n_sec = len(sections)
        n_pts = len(sections[0])
        verts = np.vstack(sections)

        faces = []
        for i in range(n_sec - 1):
            b0 = i * n_pts
            b1 = (i + 1) * n_pts
            for j in range(n_pts - 1):
                v0 = b0 + j
                v1 = b0 + j + 1
                v2 = b1 + j + 1
                v3 = b1 + j
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        return verts, np.array(faces) if faces else (verts, np.zeros((0, 3), dtype=int))

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
