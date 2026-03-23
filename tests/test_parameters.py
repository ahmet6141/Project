"""Tests for parameter definitions and YAML config round-trip."""

import tempfile
from pathlib import Path

from fightercad.parameters import (
    AircraftParams,
    FuselageParams,
    WingParams,
    MetaParams,
)
from fightercad.config import load_config, save_config


def test_default_params_creation():
    params = AircraftParams()
    assert params.meta.design_mach == 1.6
    assert params.fuselage.length_m == 15.0
    assert params.wing.area_m2 == 28.0
    assert params.wing.planform == "cropped_delta"
    assert params.exhaust.nozzle_type == "convergent_divergent"
    assert params.area_rule.enabled is True


def test_custom_params():
    params = AircraftParams(
        meta=MetaParams(design_mach=2.0),
        fuselage=FuselageParams(length_m=20.0),
    )
    assert params.meta.design_mach == 2.0
    assert params.fuselage.length_m == 20.0
    # Other sections keep defaults
    assert params.wing.area_m2 == 28.0


def test_load_save_config_roundtrip():
    original = AircraftParams(
        meta=MetaParams(design_mach=1.8, design_altitude_m=12000.0),
        fuselage=FuselageParams(length_m=18.0, max_diameter_m=2.0),
        wing=WingParams(area_m2=35.0, aspect_ratio=2.5),
    )
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        path = Path(f.name)

    save_config(original, path)
    loaded = load_config(path)

    assert loaded.meta.design_mach == 1.8
    assert loaded.meta.design_altitude_m == 12000.0
    assert loaded.fuselage.length_m == 18.0
    assert loaded.fuselage.max_diameter_m == 2.0
    assert loaded.wing.area_m2 == 35.0
    assert loaded.wing.aspect_ratio == 2.5

    path.unlink(missing_ok=True)


def test_load_default_config():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default_fighter.yaml"
    if config_path.exists():
        params = load_config(config_path)
        assert params.meta.design_mach > 0
        assert params.fuselage.length_m > 0
        assert params.wing.area_m2 > 0
