"""Tests for export functionality."""

import tempfile
from pathlib import Path

import numpy as np

from fightercad.export.step_export import is_occ_available, export_stl_mesh


def test_occ_availability_check():
    # Should return bool without error
    result = is_occ_available()
    assert isinstance(result, bool)


def test_stl_export():
    # Create a simple triangle mesh (a tetrahedron)
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3],
    ])

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        path = Path(f.name)

    export_stl_mesh(vertices, faces, path)

    assert path.exists()
    assert path.stat().st_size > 0

    # STL binary: 80-byte header + 4-byte count + 50 bytes per triangle
    expected_size = 80 + 4 + 50 * len(faces)
    assert path.stat().st_size == expected_size

    path.unlink(missing_ok=True)


def test_stl_export_empty():
    vertices = np.zeros((0, 3))
    faces = np.zeros((0, 3), dtype=int)

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        path = Path(f.name)

    export_stl_mesh(vertices, faces, path)
    assert path.exists()
    # Should be just header + count (84 bytes)
    assert path.stat().st_size == 84

    path.unlink(missing_ok=True)
