"""Delta/swept wing geometry generation.

Generates wing geometry as a set of spanwise airfoil sections, then connects
them to form a lofted surface.  Supports cropped delta, full delta, and
double-delta planforms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fightercad.parameters import WingParams
from fightercad.geometry.primitives import get_airfoil, airfoil_to_3d, rotate_2d


@dataclass
class WingSection:
    """A single spanwise station of the wing."""
    y_span: float  # spanwise position (m, 0 = root)
    chord: float  # local chord (m)
    x_le: float  # leading-edge x position (m, aircraft coords)
    z: float  # vertical offset (dihedral)
    twist_deg: float  # local twist angle
    points_3d: np.ndarray  # (N, 3) airfoil in aircraft coordinates


class WingBuilder:
    """Build wing geometry from parameters.

    Coordinate system: x = flight direction (nose = 0), y = spanwise (right),
    z = vertical (up).
    """

    def __init__(self, params: WingParams, n_span_sections: int = 8, n_airfoil_pts: int = 60):
        self.p = params
        self.n_span = n_span_sections
        self.n_af = n_airfoil_pts
        self.sections: list[WingSection] = []

    def _compute_planform(self) -> tuple[float, float, float]:
        """Compute half-span, root chord, tip chord from wing params.

        Uses span_m directly as the primary sizing parameter.
        AR is derived: AR = span² / area.
        """
        p = self.p
        span = p.span_m
        half_span = span / 2.0
        root_chord = 2.0 * p.area_m2 / (span * (1.0 + p.taper_ratio))
        tip_chord = root_chord * p.taper_ratio
        return half_span, root_chord, tip_chord

    def build_half_wing(self, side: float = 1.0) -> list[WingSection]:
        """Build one half of the wing (right side by default).

        Parameters
        ----------
        side : float
            +1.0 for right wing, -1.0 for left wing.

        Returns
        -------
        list[WingSection]
            Ordered sections from root to tip.
        """
        p = self.p
        half_span, root_chord, tip_chord = self._compute_planform()

        # Spanwise stations (cosine distribution for better tip resolution)
        eta = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi / 2, self.n_span)))
        y_stations = eta * half_span

        sweep_le_rad = math.radians(p.leading_edge_sweep_deg)
        dihedral_rad = math.radians(p.dihedral_deg)

        # Wing root leading edge position on fuselage
        # wing_station_pct is the fraction along fuselage length
        # We'll set this relative offset later in aircraft assembly
        x_root_le = 0.0  # local coordinate; assembly will translate

        sections = []
        for y in y_stations:
            frac = y / half_span  # 0 at root, 1 at tip

            # Interpolate chord
            chord = root_chord * (1.0 - frac) + tip_chord * frac

            # Leading edge x position (sweep)
            x_le = x_root_le + y * math.tan(sweep_le_rad)

            # Vertical offset (dihedral)
            z = y * math.tan(dihedral_rad)

            # Twist (linear distribution, 0 at root, full twist at tip)
            twist = p.twist_deg * frac

            # Interpolate airfoil type
            if frac < 0.5:
                af_type = p.root_airfoil
            else:
                af_type = p.tip_airfoil

            # Generate 2D airfoil
            af_2d = get_airfoil(af_type, chord, p.leading_edge_radius_mm, self.n_af)

            # Apply twist rotation around quarter-chord point
            qc_x = chord * 0.25
            if abs(twist) > 0.01:
                af_2d = rotate_2d(af_2d, twist, center=(qc_x, 0.0))

            # Convert to 3D
            pts_3d = airfoil_to_3d(af_2d, y * side, x_offset=x_le, z_offset=z)

            # Apply incidence angle at root (diminishes toward tip)
            if abs(p.incidence_deg) > 0.01:
                inc = p.incidence_deg * (1.0 - frac * 0.5)
                inc_rad = math.radians(inc)
                # Rotate around the LE point in the xz plane
                x_ref = pts_3d[0, 0]
                z_ref = pts_3d[0, 2]
                dx = pts_3d[:, 0] - x_ref
                dz = pts_3d[:, 2] - z_ref
                pts_3d[:, 0] = x_ref + dx * math.cos(inc_rad) + dz * math.sin(inc_rad)
                pts_3d[:, 2] = z_ref - dx * math.sin(inc_rad) + dz * math.cos(inc_rad)

            sections.append(WingSection(
                y_span=y * side,
                chord=chord,
                x_le=x_le,
                z=z,
                twist_deg=twist,
                points_3d=pts_3d,
            ))

        self.sections = sections
        return sections

    def build(self) -> tuple[list[WingSection], list[WingSection]]:
        """Build both wing halves.

        Returns
        -------
        tuple
            (right_sections, left_sections)
        """
        right = self.build_half_wing(side=1.0)
        left = self.build_half_wing(side=-1.0)
        return right, left

    def get_planform_outline(self) -> np.ndarray:
        """Return (N, 2) array of planform outline in top-view (x, y).

        Useful for 2D visualization.
        """
        half_span, root_chord, tip_chord = self._compute_planform()
        sweep_le = math.radians(self.p.leading_edge_sweep_deg)
        sweep_te = math.radians(self.p.trailing_edge_sweep_deg)

        # Right wing outline: root LE -> tip LE -> tip TE -> root TE
        pts = np.array([
            [0, 0],  # root LE
            [half_span * math.tan(sweep_le), half_span],  # tip LE
            [half_span * math.tan(sweep_le) + tip_chord, half_span],  # tip TE
            [root_chord, 0],  # root TE
        ])

        # Mirror for left wing
        left = pts.copy()
        left[:, 1] *= -1.0
        left = left[::-1]

        return np.vstack([pts, left])

    def get_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate triangle mesh for one wing half (right side).

        Returns
        -------
        vertices : np.ndarray (V, 3)
        faces : np.ndarray (F, 3)
        """
        if not self.sections:
            self.build_half_wing(side=1.0)

        sections = self.sections
        n_sec = len(sections)
        n_pts = len(sections[0].points_3d)
        verts = np.vstack([s.points_3d for s in sections])

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
