"""Delta/swept wing geometry generation.

Generates wing geometry as a set of spanwise airfoil sections, then connects
them to form a lofted surface.  Supports cropped delta, full delta, and
double-delta planforms with inner/outer panel split, control surfaces,
tip caps, and continuous airfoil interpolation.
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


def _smoothstep(t: float) -> float:
    """Hermite smooth-step for C¹ blending."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class WingBuilder:
    """Build wing geometry from parameters.

    Coordinate system: x = flight direction (nose = 0), y = spanwise (right),
    z = vertical (up).
    """

    def __init__(self, params: WingParams, n_span_sections: int = 20, n_airfoil_pts: int = 150):
        self.p = params
        self.n_span = n_span_sections
        self.n_af = n_airfoil_pts
        self.sections: list[WingSection] = []

    def _compute_planform(self) -> tuple[float, float, float]:
        """Compute half-span, root chord, tip chord from wing params."""
        p = self.p
        span = p.span_m
        half_span = span / 2.0
        root_chord = 2.0 * p.area_m2 / (span * (1.0 + p.taper_ratio))
        tip_chord = root_chord * p.taper_ratio
        return half_span, root_chord, tip_chord

    def _make_section(
        self, y: float, half_span: float, root_chord: float, tip_chord: float,
        sweep_rad: float, side: float, x_le_base: float = 0.0,
    ) -> WingSection:
        """Create a single wing section at spanwise position y.

        Chord is derived from LE and TE sweep angles so that
        ``trailing_edge_sweep_deg`` actually controls the trailing-edge
        geometry (not just the 2-D outline).
        """
        p = self.p
        frac = y / half_span if half_span > 0 else 0.0
        dihedral_rad = math.radians(p.dihedral_deg)

        x_le = x_le_base + y * math.tan(sweep_rad)
        # TE position governed by TE sweep: x_te(y) = root_chord + y*tan(te_sweep)
        te_sweep_rad = math.radians(p.trailing_edge_sweep_deg)
        x_te = root_chord + y * math.tan(te_sweep_rad)
        chord = max(x_te - x_le, tip_chord * 0.5)  # ensure minimum chord
        z = y * math.tan(dihedral_rad)
        twist = p.twist_deg * frac

        # Continuous thickness interpolation
        t_c = p.thickness_to_chord_root * (1.0 - frac) + p.thickness_to_chord_tip * frac
        camber = p.camber_root_pct * (1.0 - frac) + p.camber_tip_pct * frac

        # Blend airfoil type from root to tip when they differ
        if p.tip_airfoil != p.root_airfoil and frac > 0.3:
            # Transition zone: 30%-100% span → blend root→tip airfoil
            blend_t = (frac - 0.3) / 0.7
            af_root = get_airfoil(
                p.root_airfoil, chord, p.leading_edge_radius_mm, self.n_af,
                thickness_override=t_c, camber_pct=camber,
                te_thickness_mm=p.te_thickness_mm,
            )
            af_tip = get_airfoil(
                p.tip_airfoil, chord, p.leading_edge_radius_mm, self.n_af,
                thickness_override=t_c, camber_pct=camber,
                te_thickness_mm=p.te_thickness_mm,
            )
            # Ensure same point count for blending
            n_common = min(len(af_root), len(af_tip))
            af_2d = af_root[:n_common] * (1.0 - blend_t) + af_tip[:n_common] * blend_t
        else:
            af_2d = get_airfoil(
                p.root_airfoil, chord, p.leading_edge_radius_mm, self.n_af,
                thickness_override=t_c,
                camber_pct=camber,
                te_thickness_mm=p.te_thickness_mm,
            )

        # Apply twist around quarter-chord
        qc_x = chord * 0.25
        if abs(twist) > 0.01:
            af_2d = rotate_2d(af_2d, twist, center=(qc_x, 0.0))

        pts_3d = airfoil_to_3d(af_2d, y * side, x_offset=x_le, z_offset=z)

        # Apply incidence (diminishes toward tip)
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

    def build_half_wing(self, side: float = 1.0) -> list[WingSection]:
        """Build one half of the wing (right side by default)."""
        p = self.p
        half_span, root_chord, tip_chord = self._compute_planform()
        sweep_le_rad = math.radians(p.leading_edge_sweep_deg)

        # Cosine distribution for better tip resolution
        eta = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi / 2, self.n_span)))
        y_stations = eta * half_span

        sections = []
        for y in y_stations:
            sec = self._make_section(y, half_span, root_chord, tip_chord, sweep_le_rad, side)
            sections.append(sec)

        self.sections = sections
        return sections

    def build_half_wing_panels(
        self, side: float = 1.0
    ) -> tuple[list[WingSection], list[WingSection]]:
        """Build one half-wing split into inner and outer panels.

        Uses C¹ smooth blending at the panel junction to avoid visible kinks.

        Returns
        -------
        tuple
            (inner_sections, outer_sections)
        """
        p = self.p
        half_span, root_chord, tip_chord = self._compute_planform()
        split_eta = p.inner_panel_span_pct
        y_split = half_span * split_eta

        inner_sweep_rad = math.radians(p.inner_panel_sweep_deg)
        outer_sweep_rad = math.radians(p.leading_edge_sweep_deg)
        dihedral_rad = math.radians(p.dihedral_deg)

        # Blend zone: ±10% of half-span around split point
        blend_half = 0.10 * half_span

        n_inner = max(4, self.n_span // 2)
        n_outer = max(4, self.n_span - n_inner + 1)

        # Inner panel stations (cosine within inner span)
        eta_inner = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi / 2, n_inner)))
        y_inner = eta_inner * y_split

        # Outer panel stations
        eta_outer = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi / 2, n_outer)))
        y_outer = y_split + eta_outer * (half_span - y_split)

        def _get_sweep_rad(y: float) -> float:
            """Get effective sweep at spanwise position y with C¹ blending."""
            if y < y_split - blend_half:
                return inner_sweep_rad
            elif y > y_split + blend_half:
                return outer_sweep_rad
            else:
                t = (y - (y_split - blend_half)) / (2.0 * blend_half) if blend_half > 0 else 1.0
                return inner_sweep_rad * (1.0 - _smoothstep(t)) + outer_sweep_rad * _smoothstep(t)

        def _compute_x_le(y: float) -> float:
            """Compute LE x-position by integrating sweep incrementally."""
            # Numerical integration of tan(sweep(y')) from 0 to y
            n_steps = max(20, int(y / half_span * 100))
            y_pts = np.linspace(0, y, n_steps + 1)
            dy = y_pts[1] - y_pts[0] if n_steps > 0 else 0
            x_le = 0.0
            for yp in y_pts[:-1]:
                x_le += math.tan(_get_sweep_rad(yp + dy / 2)) * dy
            return x_le

        def _make_panel_section(y: float) -> WingSection:
            frac = y / half_span if half_span > 0 else 0.0
            x_le = _compute_x_le(y)
            # Chord from LE+TE sweep geometry
            te_sweep_rad = math.radians(p.trailing_edge_sweep_deg)
            x_te = root_chord + y * math.tan(te_sweep_rad)
            chord = max(x_te - x_le, tip_chord * 0.5)
            z = y * math.tan(dihedral_rad)
            twist = p.twist_deg * frac

            t_c = p.thickness_to_chord_root * (1.0 - frac) + p.thickness_to_chord_tip * frac
            camber = p.camber_root_pct * (1.0 - frac) + p.camber_tip_pct * frac

            # Blend airfoil type from root to tip when they differ
            if p.tip_airfoil != p.root_airfoil and frac > 0.3:
                blend_t = (frac - 0.3) / 0.7
                af_root = get_airfoil(
                    p.root_airfoil, chord, p.leading_edge_radius_mm, self.n_af,
                    thickness_override=t_c, camber_pct=camber,
                    te_thickness_mm=p.te_thickness_mm,
                )
                af_tip = get_airfoil(
                    p.tip_airfoil, chord, p.leading_edge_radius_mm, self.n_af,
                    thickness_override=t_c, camber_pct=camber,
                    te_thickness_mm=p.te_thickness_mm,
                )
                n_common = min(len(af_root), len(af_tip))
                af_2d = af_root[:n_common] * (1.0 - blend_t) + af_tip[:n_common] * blend_t
            else:
                af_2d = get_airfoil(
                    p.root_airfoil, chord, p.leading_edge_radius_mm, self.n_af,
                    thickness_override=t_c, camber_pct=camber,
                    te_thickness_mm=p.te_thickness_mm,
                )

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

        inner_sections = [_make_panel_section(y) for y in y_inner]
        outer_sections = [_make_panel_section(y) for y in y_outer]

        self.sections = inner_sections + outer_sections[1:]
        return inner_sections, outer_sections

    def split_control_surface(
        self,
        sections: list[WingSection],
        chord_pct: float,
        span_pct: float,
    ) -> tuple[list[WingSection], list[WingSection]]:
        """Split trailing-edge control surface from wing sections.

        All sections are split at the same chord fraction to maintain
        consistent point counts, but only outboard sections produce
        control surface geometry.

        Returns
        -------
        tuple
            (main_sections, ctrl_sections)
        """
        if not sections:
            return [], []

        n_pts = len(sections[0].points_3d)
        half_n = n_pts // 2
        cut_idx = int(half_n * (1.0 - chord_pct))

        n_sec = len(sections)
        ctrl_start = max(0, n_sec - int(n_sec * span_pct))

        main_sections = []
        ctrl_sections = []

        for i, sec in enumerate(sections):
            pts = sec.points_3d

            # Always split to maintain consistent point counts
            upper_main = pts[:cut_idx]
            lower_main = pts[n_pts - cut_idx:]
            main_pts = np.vstack([upper_main, lower_main])

            main_sections.append(WingSection(
                y_span=sec.y_span, chord=sec.chord * (1 - chord_pct),
                x_le=sec.x_le, z=sec.z, twist_deg=sec.twist_deg,
                points_3d=main_pts,
            ))

            if i >= ctrl_start:
                upper_ctrl = pts[cut_idx:half_n]
                lower_ctrl = pts[half_n:n_pts - cut_idx]
                ctrl_pts = np.vstack([upper_ctrl, lower_ctrl])

                ctrl_sections.append(WingSection(
                    y_span=sec.y_span, chord=sec.chord * chord_pct,
                    x_le=sec.x_le + sec.chord * (1 - chord_pct),
                    z=sec.z, twist_deg=sec.twist_deg,
                    points_3d=ctrl_pts,
                ))

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
        """Return (N, 2) array of planform outline in top-view (x, y)."""
        half_span, root_chord, tip_chord = self._compute_planform()
        sweep_le = math.radians(self.p.leading_edge_sweep_deg)
        sweep_te = math.radians(self.p.trailing_edge_sweep_deg)

        pts = np.array([
            [0, 0],
            [half_span * math.tan(sweep_le), half_span],
            [half_span * math.tan(sweep_le) + tip_chord, half_span],
            [root_chord, 0],
        ])

        left = pts.copy()
        left[:, 1] *= -1.0
        left = left[::-1]

        return np.vstack([pts, left])

    @staticmethod
    def get_panel_mesh(
        sections: list[WingSection], add_tip_cap: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate triangle mesh from a list of wing sections.

        Parameters
        ----------
        sections : list[WingSection]
            Wing sections to mesh.
        add_tip_cap : bool
            If True, add a fan-triangulated cap at the last (tip) section.

        Returns
        -------
        vertices : np.ndarray (V, 3)
        faces : np.ndarray (F, 3)
        """
        if not sections:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        n_sec = len(sections)
        verts = np.vstack([s.points_3d for s in sections])

        faces = []
        for i in range(n_sec - 1):
            n_pts_i = len(sections[i].points_3d)
            n_pts_j = len(sections[i + 1].points_3d)
            b0 = sum(len(sections[k].points_3d) for k in range(i))
            b1 = b0 + n_pts_i
            n_common = min(n_pts_i, n_pts_j)
            for j in range(n_common):
                j1 = (j + 1) % n_common
                v0 = b0 + j
                v1 = b0 + j1
                v2 = b1 + j1
                v3 = b1 + j
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        # Tip cap (fan triangulation from centroid)
        if add_tip_cap and n_sec >= 2:
            tip_pts = sections[-1].points_3d
            n_tip = len(tip_pts)
            base_tip = sum(len(sections[k].points_3d) for k in range(n_sec - 1))
            tip_center = tip_pts.mean(axis=0)
            center_idx = len(verts)
            verts = np.vstack([verts, tip_center.reshape(1, 3)])
            for j in range(n_tip):
                j1 = (j + 1) % n_tip
                faces.append([center_idx, base_tip + j, base_tip + j1])

        return (verts, np.array(faces)) if faces else (verts, np.zeros((0, 3), dtype=int))

    @staticmethod
    def build_sawtooth_te(
        sections: list[WingSection],
        depth_mm: float,
        n_teeth: int,
        side: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate saw-tooth (serrated) trailing edge geometry.

        Creates a thin zigzag strip along the trailing edge of the given
        wing sections for stealth shaping.

        Parameters
        ----------
        sections : list[WingSection]
            Wing sections (must have at least 2).
        depth_mm : float
            Tooth depth in mm (how far back the serration extends).
        n_teeth : int
            Number of teeth along the half-span.
        side : float
            +1.0 for right, -1.0 for left.

        Returns
        -------
        vertices : np.ndarray (V, 3)
        faces : np.ndarray (F, 3)
        """
        if len(sections) < 2 or n_teeth < 1 or depth_mm <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

        depth = depth_mm / 1000.0
        # Get TE positions from each section
        te_points = []
        for sec in sections:
            pts = sec.points_3d
            n_pts = len(pts)
            half_n = n_pts // 2
            # TE is at index half_n-1 (upper) and half_n (lower)
            # Average upper and lower TE for center line
            te_upper = pts[half_n - 1]
            te_lower = pts[half_n]
            te_center = (te_upper + te_lower) / 2.0
            te_points.append(te_center)

        te_points = np.array(te_points)

        # Interpolate TE line at tooth positions
        n_sec = len(te_points)
        t_teeth = np.linspace(0, 1, 2 * n_teeth + 1)
        interp_pts = np.zeros((len(t_teeth), 3))
        for dim in range(3):
            interp_pts[:, dim] = np.interp(
                t_teeth, np.linspace(0, 1, n_sec), te_points[:, dim]
            )

        # Build zigzag: even indices = base TE, odd indices = extended TE
        verts = []
        for i, pt in enumerate(interp_pts):
            if i % 2 == 0:
                # Base position (on the wing TE)
                verts.append(pt.copy())
            else:
                # Extended position (tooth tip, pushed aft)
                extended = pt.copy()
                extended[0] += depth  # extend in x (aft)
                verts.append(extended)

        verts = np.array(verts)

        # Thin strip: duplicate with slight z offset for thickness
        thickness = 0.001  # 1mm thick strip
        upper_verts = verts.copy()
        upper_verts[:, 2] += thickness / 2
        lower_verts = verts.copy()
        lower_verts[:, 2] -= thickness / 2

        all_verts = np.vstack([upper_verts, lower_verts])
        n_v = len(upper_verts)

        faces = []
        for i in range(n_v - 1):
            # Upper surface
            faces.append([i, i + 1, n_v + i + 1])
            faces.append([i, n_v + i + 1, n_v + i])

        return (all_verts, np.array(faces)) if faces else (all_verts, np.zeros((0, 3), dtype=int))

    def get_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate triangle mesh for one wing half (right side)."""
        if not self.sections:
            self.build_half_wing(side=1.0)

        return self.get_panel_mesh(self.sections)
