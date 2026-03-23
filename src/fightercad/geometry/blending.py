"""Component blending and fillet operations.

Manages the junction geometry between wing-fuselage, stabilizer-fuselage,
and intake-fuselage connections.  When pythonocc is available, applies
BRepFilletAPI operations.  Otherwise provides geometric blending data
for mesh-based visualization.
"""

from __future__ import annotations

import math

import numpy as np

from fightercad.parameters import BlendingParams


class BlendingOperator:
    """Apply blending/fillet operations between aircraft components."""

    def __init__(self, params: BlendingParams):
        self.p = params

    def compute_fillet_profile(
        self,
        radius_mm: float,
        n_points: int = 10,
    ) -> np.ndarray:
        """Generate a quarter-circle fillet profile.

        Returns (n_points, 2) array of local (u, v) coordinates
        for the fillet cross-section.
        """
        r = radius_mm / 1000.0  # convert to meters
        theta = np.linspace(0, math.pi / 2, n_points)
        u = r * (1.0 - np.cos(theta))
        v = r * (1.0 - np.sin(theta))
        return np.column_stack([u, v])

    def compute_strake_geometry(
        self,
        fuselage_length: float,
        wing_root_x: float,
        fuselage_radius: float,
    ) -> np.ndarray | None:
        """Generate LEX/strake planform points if strake mode is active.

        Returns (N, 3) points defining the strake outline, or None.
        """
        if self.p.blending_mode != "strake" or self.p.strake_length_m <= 0:
            return None

        sweep_rad = math.radians(self.p.strake_sweep_deg)
        L = self.p.strake_length_m

        # Strake starts ahead of wing root and merges into wing LE
        x_start = wing_root_x - L
        x_end = wing_root_x

        n_pts = 20
        x = np.linspace(x_start, x_end, n_pts)
        # Strake edge grows from fuselage to wing root
        frac = (x - x_start) / L
        y_edge = fuselage_radius + frac * L * math.tan(math.pi / 2 - sweep_rad)

        # Upper surface (thin strake, slight thickness)
        pts_upper = np.column_stack([x, y_edge, np.full(n_pts, 0.01)])
        pts_lower = np.column_stack([x, y_edge, np.full(n_pts, -0.01)])

        return np.vstack([pts_upper, pts_lower[::-1]])

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
