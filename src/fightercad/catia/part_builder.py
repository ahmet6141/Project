"""CATIA Part builder via COM automation.

Creates parametric geometry directly inside CATIA's Part Design workbench.
Each aircraft component is built as a separate Body in the specification tree.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fightercad.parameters import AircraftParams
from fightercad.catia.connection import get_catia, is_catia_available


class CATIAPartBuilder:
    """Build aircraft geometry directly in CATIA via COM."""

    def __init__(self):
        self.catia: Any = None
        self.part_doc: Any = None
        self.part: Any = None
        self.hsf: Any = None  # HybridShapeFactory
        self.parameters: Any = None

    def initialize(self) -> None:
        """Connect to CATIA and create a new Part document."""
        if not is_catia_available():
            raise RuntimeError("CATIA COM not available (requires Windows + pywin32 + running CATIA)")

        self.catia = get_catia()
        documents = self.catia.Documents
        self.part_doc = documents.Add("Part")
        self.part = self.part_doc.Part
        self.hsf = self.part.HybridShapeFactory
        self.parameters = self.part.Parameters

    def _add_parameter(self, name: str, value: float, unit: str = "mm") -> Any:
        """Add a named parameter to the CATIA part."""
        if unit == "deg":
            return self.parameters.CreateReal(name, value)
        elif unit == "mm":
            return self.parameters.CreateDimension(name, "LENGTH", value)
        else:
            return self.parameters.CreateReal(name, value)

    def set_aircraft_parameters(self, params: AircraftParams) -> None:
        """Register all aircraft parameters in CATIA's parameter tree."""
        p = params

        # Fuselage parameters
        self._add_parameter("Fuselage_Length", p.fuselage.length_m * 1000, "mm")
        self._add_parameter("Fuselage_MaxDiameter", p.fuselage.max_diameter_m * 1000, "mm")
        self._add_parameter("Fuselage_NoseFineness", p.fuselage.nose_fineness_ratio)
        self._add_parameter("Fuselage_AftTaper", p.fuselage.aft_taper_ratio)

        # Wing parameters
        self._add_parameter("Wing_Area", p.wing.area_m2 * 1e6, "mm")  # mm²
        self._add_parameter("Wing_AR", p.wing.aspect_ratio)
        self._add_parameter("Wing_TaperRatio", p.wing.taper_ratio)
        self._add_parameter("Wing_SweepLE", p.wing.leading_edge_sweep_deg, "deg")
        self._add_parameter("Wing_Dihedral", p.wing.dihedral_deg, "deg")
        self._add_parameter("Wing_Twist", p.wing.twist_deg, "deg")
        self._add_parameter("Wing_LERadius", p.wing.leading_edge_radius_mm, "mm")

        # Blending
        self._add_parameter("Blend_RootFillet", p.blending.root_fillet_radius_mm, "mm")
        self._add_parameter("Blend_LEFillet", p.blending.leading_edge_fillet_mm, "mm")
        self._add_parameter("Blend_TEFillet", p.blending.trailing_edge_fillet_mm, "mm")

        self.part.Update()

    def update_parameter(self, name: str, value: float) -> None:
        """Update a single CATIA parameter and refresh the part.

        This is the key method for live GUI updates:
        GUI slider changes → this method → CATIA geometry updates.
        """
        try:
            param = self.parameters.Item(name)
            param.Value = value
            self.part.Update()
        except Exception as e:
            print(f"Warning: could not update parameter {name}: {e}")

    def create_body(self, name: str) -> Any:
        """Create a new Body in the specification tree."""
        bodies = self.part.Bodies
        body = bodies.Add()
        body.Name = name
        return body

    def add_spline_from_points(
        self,
        points: np.ndarray,
        geo_set: Any,
        closed: bool = True,
    ) -> Any:
        """Create a CATIA spline from 3D points in a geometrical set.

        Parameters
        ----------
        points : np.ndarray (N, 3)
            3D points in mm.
        geo_set : CATIA HybridBody
            The geometrical set to add the spline to.
        closed : bool
            Whether to close the spline.
        """
        spline = self.hsf.AddNewSpline()
        spline.SetSplineType(0)  # 0 = basic spline
        spline.SetClosing(2 if closed else 0)  # 2 = closed

        for pt in points:
            catia_pt = self.hsf.AddNewPointCoord(
                float(pt[0]), float(pt[1]), float(pt[2])
            )
            geo_set.AppendHybridShape(catia_pt)
            spline.AddPoint(catia_pt)

        geo_set.AppendHybridShape(spline)
        return spline

    def add_loft(
        self,
        sections: list[Any],
        body: Any,
    ) -> Any:
        """Create a multi-section solid (loft) through section curves.

        Parameters
        ----------
        sections : list
            List of CATIA spline/wire references.
        body : CATIA Body
            The body to add the loft to.
        """
        shape_factory = self.part.ShapeFactory
        loft = shape_factory.AddNewLoft()
        for sec in sections:
            loft.AddSectionToLoft(sec, 1, None)
        self.part.Update()
        return loft

    def build_fuselage(
        self,
        sections_3d: list[np.ndarray],
    ) -> Any:
        """Build the fuselage as a lofted body in CATIA.

        Parameters
        ----------
        sections_3d : list[np.ndarray]
            List of (N, 3) cross-section arrays (in meters, will convert to mm).
        """
        body = self.create_body("Fuselage")
        geo_set = self.part.HybridBodies.Add()
        geo_set.Name = "Fuselage_Sections"

        splines = []
        for i, sec in enumerate(sections_3d):
            pts_mm = sec * 1000.0  # meters to mm
            spline = self.add_spline_from_points(pts_mm, geo_set)
            splines.append(spline)

        if len(splines) >= 2:
            self.add_loft(splines, body)

        self.part.Update()
        return body

    def build_wing(
        self,
        sections_3d: list[np.ndarray],
        name: str = "Wing_Right",
    ) -> Any:
        """Build a wing as a lofted body in CATIA."""
        body = self.create_body(name)
        geo_set = self.part.HybridBodies.Add()
        geo_set.Name = f"{name}_Sections"

        splines = []
        for sec in sections_3d:
            pts_mm = sec * 1000.0
            spline = self.add_spline_from_points(pts_mm, geo_set)
            splines.append(spline)

        if len(splines) >= 2:
            self.add_loft(splines, body)

        self.part.Update()
        return body

    def build_full_aircraft(
        self,
        assembler,
    ) -> None:
        """Build the complete aircraft in CATIA from an AircraftAssembler.

        Parameters
        ----------
        assembler : AircraftAssembler
            The assembled aircraft with built geometry.
        """
        self.initialize()
        self.set_aircraft_parameters(assembler.params)

        # Build fuselage
        if assembler.fuselage_builder and assembler.fuselage_builder.sections:
            fuse_sections = [s.points for s in assembler.fuselage_builder.sections]
            self.build_fuselage(fuse_sections)

        # Build wings
        if assembler.wing_builder and assembler.wing_builder.sections:
            # Right wing
            right_secs = [s.points_3d for s in assembler.wing_builder.build_half_wing(1.0)]
            self.build_wing(right_secs, "Wing_Right")
            # Left wing
            left_secs = [s.points_3d for s in assembler.wing_builder.build_half_wing(-1.0)]
            self.build_wing(left_secs, "Wing_Left")

        # Build stabilizer
        if assembler.stab_builder and assembler.stab_builder.section_points:
            self.build_wing(assembler.stab_builder.section_points, "Vertical_Stabilizer")

        # Build intakes
        if assembler.intake_builder:
            right_int, left_int = assembler.intake_builder.build()
            self.build_wing(right_int, "Intake_Right")
            self.build_wing(left_int, "Intake_Left")

        # Build exhaust
        if assembler.exhaust_builder and assembler.exhaust_builder.section_points:
            self.build_wing(assembler.exhaust_builder.section_points, "Exhaust_Nozzle")

        self.part.Update()
