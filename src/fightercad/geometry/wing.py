"""Delta/swept wing geometry generation.

Generates wing geometry as a set of spanwise airfoil sections, then connects
them to form a lofted surface.  Supports cropped delta, full delta, and
double-delta planforms with inner/outer panel split and control surfaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fightercad.parameters import WingParams, ControlSurfaceParams
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

    def build_half_wing_panels(
        self, side: float = 1.0
    ) -> tuple[list[WingSection], list[WingSection]]:
        """Build one half-wing split into inner and outer panels.

        The inner panel uses ``inner_panel_sweep_deg`` and extends from root
        to ``inner_panel_span_pct`` of half-span.  The outer panel uses the
        standard ``leading_edge_sweep_deg`` from there to the tip.

        Parameters
        ----------
        side : float
            +1.0 for right, -1.0 for left.

        Returns
        -------
        tuple
            (inner_sections, outer_sections)
        """
        p = self.p
        half_span, root_chord, tip_chord = self._compute_planform()
        split_eta = p.inner_panel_span_pct  # e.g. 0.35

        inner_sweep_rad = math.radians(p.inner_panel_sweep_deg)
        outer_sweep_rad = math.radians(p.leading_edge_sweep_deg)
        dihedral_rad = math.radians(p.dihedral_deg)

        # Number of sections per panel (at least 4 each)
        n_inner = max(4, self.n_span // 2)
        n_outer = max(4, self.n_span - n_inner + 1)  # +1 for shared boundary

        # Inner panel stations
        eta_inner = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi / 2, n_inner)))
        y_inner = eta_inner * (half_span * split_eta)

        # Outer panel stations (from split point to tip)
        eta_outer = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi / 2, n_outer)))
        y_split = half_span * split_eta
        y_outer = y_split + eta_outer * (half_span - y_split)

        x_root_le = 0.0

        def _make_section(y: float) -> WingSection:
            frac = y / half_span

            chord = root_chord * (1.0 - frac) + tip_chord * frac

            # LE x: inner panel uses inner sweep, outer uses outer sweep
            if y <= y_split:
                x_le = x_root_le + y * math.tan(inner_sweep_rad)
            else:
                # Outer panel continues from inner panel's endpoint
                x_at_split = x_root_le + y_split * math.tan(inner_sweep_rad)
                x_le = x_at_split + (y - y_split) * math.tan(outer_sweep_rad)

            z = y * math.tan(dihedral_rad)
            twist = p.twist_deg * frac

            af_type = p.root_airfoil if frac < 0.5 else p.tip_airfoil
            af_2d = get_airfoil(af_type, chord, p.leading_edge_radius_mm, self.n_af)

            qc_x = chord * 0.25
            if abs(twist) > 0.01:
                af_2d = rotate_2d(af_2d, twist, center=(qc_x, 0.0))

            pts_3d = airfoil_to_3d(af_2d, y * side, x_offset=x_le, z_offset=z)

            if abs(p.incidence_deg) > 0.01:
                inc = p.incidence_deg * (1.0 - frac * 0.5)
                inc_rad = math.radians(inc)
                x_ref = pts_3d[0, 0]
                z_ref = pts_3d[0, 2]
                dx = pts_3d[:, 0] - x_ref
                dz = pts_3d[:, 2] - z_ref
                pts_3d[:, 0] = x_ref + dx * math.cos(inc_rad) + dz * math.sin(inc_rad)
                pts_3d[:, 2] = z_ref - dx * math.sin(inc_rad) + dz * math.cos(inc_rad)

            return WingSection(
                y_span=y * side, chord=chord, x_le=x_le,
                z=z, twist_deg=twist, points_3d=pts_3d,
            )

        inner_sections = [_make_section(y) for y in y_inner]
        outer_sections = [_make_section(y) for y in y_outer]

        self.sections = inner_sections + outer_sections[1:]  # avoid duplicate at split
        return inner_sections, outer_sections

    def split_control_surface(
        self,
        sections: list[WingSection],
        chord_pct: float,
        span_pct: float,
    ) -> tuple[list[WingSection], list[WingSection]]:
        """Split trailing-edge control surface from wing sections.

        Parameters
        ----------
        sections : list[WingSection]
            Panel sections (inner or outer).
        chord_pct : float
            Fraction of chord for the control surface (e.g. 0.20).
        span_pct : float
            Fraction of panel span covered by control surface (from outboard end).

        Returns
        -------
        tuple
            (main_sections, ctrl_sections) — main wing with TE removed,
            and control surface sections.
        """
        n_pts = len(sections[0].points_3d)
        # How many airfoil points belong to the trailing-edge portion
        # Airfoil points go: upper surface LE→TE then lower surface TE→LE
        # So TE region is near the middle of the point array
        half_n = n_pts // 2
        cut_idx = int(half_n * (1.0 - chord_pct))

        # Determine which sections get a control surface (outboard span_pct)
        n_sec = len(sections)
        ctrl_start = max(0, n_sec - int(n_sec * span_pct))

        main_sections = []
        ctrl_sections = []

        for i, sec in enumerate(sections):
            pts = sec.points_3d
            if i >= ctrl_start:
                # Split: upper surface [0:cut_idx] + bridge + lower [n-cut_idx:]
                # Upper TE region: [cut_idx:half_n]
                # Lower TE region: [half_n:n_pts-cut_idx]
                upper_main = pts[:cut_idx]
                upper_ctrl = pts[cut_idx:half_n]
                lower_ctrl = pts[half_n:n_pts - cut_idx]
                lower_main = pts[n_pts - cut_idx:]

                # Main wing: upper_main + lower_main (close the cut)
                main_pts = np.vstack([upper_main, lower_main])
                # Control surface: upper_ctrl + lower_ctrl
                ctrl_pts = np.vstack([upper_ctrl, lower_ctrl])

                main_sections.append(WingSection(
                    y_span=sec.y_span, chord=sec.chord * (1 - chord_pct),
                    x_le=sec.x_le, z=sec.z, twist_deg=sec.twist_deg,
                    points_3d=main_pts,
                ))
                ctrl_sections.append(WingSection(
                    y_span=sec.y_span, chord=sec.chord * chord_pct,
                    x_le=sec.x_le + sec.chord * (1 - chord_pct),
                    z=sec.z, twist_deg=sec.twist_deg,
                    points_3d=ctrl_pts,
                ))
            else:
                main_sections.append(sec)

        return main_sections, ctrl_sections

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

    @staticmethod
    def get_panel_mesh(sections: list[WingSection]) -> tuple[np.ndarray, np.ndarray]:
        """Generate triangle mesh from a list of wing sections.

        Works for any panel (inner, outer, control surface).

        Returns
        -------
        vertices : np.ndarray (V, 3)
        faces : np.ndarray (F, 3)
        """
        if not sections:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        n_sec = len(sections)
        n_pts = len(sections[0].points_3d)
        verts = np.vstack([s.points_3d for s in sections])

        faces = []
        for i in range(n_sec - 1):
            n_pts_i = len(sections[i].points_3d)
            n_pts_j = len(sections[i + 1].points_3d)
            b0 = sum(len(sections[k].points_3d) for k in range(i))
            b1 = b0 + n_pts_i
            n_common = min(n_pts_i, n_pts_j)
            for j in range(n_common - 1):
                v0 = b0 + j
                v1 = b0 + j + 1
                v2 = b1 + j + 1
                v3 = b1 + j
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        return verts, np.array(faces) if faces else (verts, np.zeros((0, 3), dtype=int))

    def get_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate triangle mesh for one wing half (right side).

        Returns
        -------
        vertices : np.ndarray (V, 3)
        faces : np.ndarray (F, 3)
        """
        if not self.sections:
            self.build_half_wing(side=1.0)

        return self.get_panel_mesh(self.sections)
