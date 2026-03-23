"""Tests for fuselage geometry builder."""

import numpy as np

from fightercad.parameters import FuselageParams
from fightercad.geometry.fuselage import FuselageBuilder


def test_fuselage_build():
    builder = FuselageBuilder(FuselageParams())
    builder.build()
    sections = builder.sections
    assert len(sections) > 0
    for sec in sections:
        assert sec.points.shape[1] == 3  # (N, 3) points


def test_fuselage_mesh():
    builder = FuselageBuilder(FuselageParams())
    builder.build()
    verts, faces = builder.get_mesh()
    assert isinstance(verts, np.ndarray)
    assert isinstance(faces, np.ndarray)
    assert verts.shape[1] == 3
    assert len(verts) > 0
    assert len(faces) > 0


def test_nose_profiles():
    for profile in ["haack", "conical", "ogive", "power_series"]:
        p = FuselageParams(nose_profile=profile)
        builder = FuselageBuilder(p)
        builder.build()
        verts, faces = builder.get_mesh()
        assert len(verts) > 0, f"Profile '{profile}' produced empty mesh"


def test_area_distribution():
    builder = FuselageBuilder(FuselageParams())
    builder.build()
    x, areas = builder.get_area_distribution()
    assert len(x) == len(areas)
    assert len(x) > 0
    assert np.all(areas >= 0), "Area distribution should be non-negative"
    # Area should start and end near zero
    assert areas[0] < areas[len(areas) // 2]


def test_elliptical_cross_section():
    p = FuselageParams(cross_section="elliptical", cross_section_aspect=1.3)
    builder = FuselageBuilder(p)
    builder.build()
    verts, faces = builder.get_mesh()
    assert len(verts) > 0
