"""Tests for wing geometry builder."""

import math

import numpy as np

from fightercad.parameters import WingParams
from fightercad.geometry.wing import WingBuilder
from fightercad.geometry.primitives import get_airfoil


def test_wing_build():
    builder = WingBuilder(WingParams())
    right_secs, left_secs = builder.build()
    assert len(right_secs) > 0
    assert len(left_secs) > 0
    assert len(right_secs) == len(left_secs)


def test_planform_geometry():
    p = WingParams(area_m2=28.0, aspect_ratio=2.2, taper_ratio=0.10)
    builder = WingBuilder(p)
    half_span, root_c, tip_c = builder._compute_planform()

    span = 2.0 * half_span
    expected_span = math.sqrt(p.aspect_ratio * p.area_m2)
    assert abs(span - expected_span) < 0.01

    expected_root = 2.0 * p.area_m2 / (span * (1.0 + p.taper_ratio))
    assert abs(root_c - expected_root) < 0.01

    assert abs(tip_c - root_c * p.taper_ratio) < 0.01


def test_airfoil_types():
    airfoils = [
        "biconvex_3", "biconvex_4", "biconvex_5",
        "diamond_3", "diamond_4", "diamond_5",
        "naca64a004",
    ]
    for name in airfoils:
        coords = get_airfoil(name, chord=1.0)
        assert isinstance(coords, np.ndarray), f"Airfoil '{name}' did not return ndarray"
        assert coords.shape[1] == 2, f"Airfoil '{name}' should have 2 columns (x, y)"
        assert len(coords) >= 10, f"Airfoil '{name}' has too few points"


def test_wing_mesh():
    builder = WingBuilder(WingParams())
    builder.build()
    # Sections should have 3D points
    right_secs, left_secs = builder.build()
    for sec in right_secs:
        assert hasattr(sec, "points_3d")
        assert sec.points_3d.shape[1] == 3
