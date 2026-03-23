"""STEP AP214 export for CATIA compatibility.

Requires pythonocc-core to be installed.  Provides a fallback STL-like
export when pythonocc is not available.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Try to import pythonocc
_HAS_OCC = False
try:
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Circ, gp_Vec
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
    )
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer
    _HAS_OCC = True
except ImportError:
    pass


def is_occ_available() -> bool:
    """Check if pythonocc-core is available."""
    return _HAS_OCC


def points_to_wire(points_3d: np.ndarray):
    """Convert (N, 3) points to an OCC wire (closed loop).

    Requires pythonocc-core.
    """
    if not _HAS_OCC:
        raise RuntimeError("pythonocc-core is required for STEP export")

    wire_builder = BRepBuilderAPI_MakeWire()
    n = len(points_3d)
    for i in range(n):
        j = (i + 1) % n
        p1 = gp_Pnt(float(points_3d[i, 0]), float(points_3d[i, 1]), float(points_3d[i, 2]))
        p2 = gp_Pnt(float(points_3d[j, 0]), float(points_3d[j, 1]), float(points_3d[j, 2]))
        edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
        wire_builder.Add(edge)
    return wire_builder.Wire()


def sections_to_solid(sections: list[np.ndarray], is_solid: bool = True):
    """Loft through cross-sections to create an OCC solid.

    Parameters
    ----------
    sections : list[np.ndarray]
        List of (N, 3) arrays, each representing a cross-section ring.
    is_solid : bool
        If True, create a solid; otherwise create a shell.
    """
    if not _HAS_OCC:
        raise RuntimeError("pythonocc-core is required for STEP export")

    loft = BRepOffsetAPI_ThruSections(is_solid, False)  # solid, not ruled
    for sec in sections:
        wire = points_to_wire(sec)
        loft.AddWire(wire)
    loft.Build()
    return loft.Shape()


def export_step(
    components: dict[str, list[np.ndarray]],
    filepath: str | Path,
    schema: str = "AP214",
) -> bool:
    """Export aircraft components to STEP AP214 format.

    Parameters
    ----------
    components : dict
        Mapping of component name to list of cross-section arrays.
        Each component will become a named solid in the STEP file.
    filepath : str or Path
        Output STEP file path.
    schema : str
        STEP schema (AP214 recommended for CATIA).

    Returns
    -------
    bool
        True if export succeeded.
    """
    if not _HAS_OCC:
        raise RuntimeError(
            "pythonocc-core is required for STEP export. "
            "Install via: conda install -c conda-forge pythonocc-core"
        )

    writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", schema)
    Interface_Static.SetCVal("write.step.product.name", "FighterCAD_Aircraft")
    Interface_Static.SetRVal("write.precision.val", 1e-6)

    for name, sections in components.items():
        if len(sections) < 2:
            continue
        try:
            shape = sections_to_solid(sections)
            # Validate
            analyzer = BRepCheck_Analyzer(shape)
            if not analyzer.IsValid():
                print(f"Warning: {name} geometry has validation issues")
            writer.Transfer(shape, STEPControl_AsIs)
        except Exception as e:
            print(f"Warning: could not process {name}: {e}")

    status = writer.Write(str(filepath))
    return status == 1  # IFSelect_RetDone


def export_stl_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    filepath: str | Path,
) -> None:
    """Export mesh as binary STL (fallback when pythonocc is not available).

    Parameters
    ----------
    vertices : np.ndarray (V, 3)
    faces : np.ndarray (F, 3)
    filepath : str or Path
    """
    filepath = Path(filepath)
    n_faces = len(faces)

    with open(filepath, "wb") as f:
        # 80-byte header
        f.write(b"\x00" * 80)
        # Number of triangles
        f.write(n_faces.to_bytes(4, "little"))

        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            # Compute normal
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal /= norm_len

            # Write: normal (3×float32), v0, v1, v2 (each 3×float32), attribute (uint16)
            for val in normal:
                f.write(np.float32(val).tobytes())
            for v in [v0, v1, v2]:
                for val in v:
                    f.write(np.float32(val).tobytes())
            f.write(b"\x00\x00")  # attribute byte count
