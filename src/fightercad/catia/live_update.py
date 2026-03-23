"""Live CATIA parameter update from GUI.

Provides a bridge between the Tkinter GUI sliders and CATIA's parametric
model, enabling real-time geometry updates in CATIA as parameters change.
"""

from __future__ import annotations

from typing import Any

from fightercad.catia.connection import is_catia_available
from fightercad.catia.part_builder import CATIAPartBuilder


class CATIALiveUpdater:
    """Manages live parameter updates between GUI and CATIA.

    Usage:
        updater = CATIALiveUpdater()
        updater.connect(assembler)  # Initial build
        updater.update_param("Wing_SweepLE", 60.0)  # GUI slider change
    """

    def __init__(self):
        self.builder: CATIAPartBuilder | None = None
        self.connected: bool = False

    def is_available(self) -> bool:
        """Check if CATIA live update is possible."""
        return is_catia_available()

    def connect(self, assembler) -> bool:
        """Initialize CATIA connection and build the full aircraft.

        Parameters
        ----------
        assembler : AircraftAssembler
            Assembled aircraft to build in CATIA.

        Returns
        -------
        bool
            True if connection and build succeeded.
        """
        if not self.is_available():
            return False

        try:
            self.builder = CATIAPartBuilder()
            self.builder.build_full_aircraft(assembler)
            self.connected = True
            return True
        except Exception as e:
            print(f"CATIA connection failed: {e}")
            self.connected = False
            return False

    def update_param(self, name: str, value: float) -> bool:
        """Update a single parameter in CATIA.

        Parameters
        ----------
        name : str
            CATIA parameter name (e.g., 'Wing_SweepLE').
        value : float
            New parameter value.

        Returns
        -------
        bool
            True if update succeeded.
        """
        if not self.connected or self.builder is None:
            return False
        try:
            self.builder.update_parameter(name, value)
            return True
        except Exception as e:
            print(f"CATIA update failed for {name}: {e}")
            return False

    def rebuild(self, assembler) -> bool:
        """Completely rebuild the CATIA model (for major changes).

        Use this when a parameter change requires full geometry rebuild
        rather than just a parameter update.
        """
        if not self.is_available():
            return False
        try:
            self.builder = CATIAPartBuilder()
            self.builder.build_full_aircraft(assembler)
            self.connected = True
            return True
        except Exception as e:
            print(f"CATIA rebuild failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from CATIA."""
        self.builder = None
        self.connected = False


# Parameter name mapping: GUI field name -> CATIA parameter name
PARAM_MAP = {
    # Fuselage
    "fuselage.length_m": ("Fuselage_Length", 1000.0),  # m -> mm
    "fuselage.max_diameter_m": ("Fuselage_MaxDiameter", 1000.0),
    "fuselage.nose_fineness_ratio": ("Fuselage_NoseFineness", 1.0),
    "fuselage.aft_taper_ratio": ("Fuselage_AftTaper", 1.0),
    # Wing
    "wing.area_m2": ("Wing_Area", 1e6),  # m² -> mm²
    "wing.aspect_ratio": ("Wing_AR", 1.0),
    "wing.taper_ratio": ("Wing_TaperRatio", 1.0),
    "wing.leading_edge_sweep_deg": ("Wing_SweepLE", 1.0),
    "wing.dihedral_deg": ("Wing_Dihedral", 1.0),
    "wing.twist_deg": ("Wing_Twist", 1.0),
    "wing.leading_edge_radius_mm": ("Wing_LERadius", 1.0),
    # Blending
    "blending.root_fillet_radius_mm": ("Blend_RootFillet", 1.0),
    "blending.leading_edge_fillet_mm": ("Blend_LEFillet", 1.0),
    "blending.trailing_edge_fillet_mm": ("Blend_TEFillet", 1.0),
}
