"""YAML configuration loader and parameter builder."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("fightercad")

from fightercad.parameters import (
    AircraftParams,
    AreaRuleParams,
    BlendingParams,
    ControlSurfaceParams,
    ExhaustParams,
    FuselageParams,
    IntakeParams,
    InternalStructureParams,
    MetaParams,
    VerticalStabilizerParams,
    WingParams,
    create_ucav_params,
)

_SECTION_MAP: dict[str, type] = {
    "meta": MetaParams,
    "fuselage": FuselageParams,
    "wing": WingParams,
    "blending": BlendingParams,
    "vertical_stabilizer": VerticalStabilizerParams,
    "control_surfaces": ControlSurfaceParams,
    "intake": IntakeParams,
    "exhaust": ExhaustParams,
    "area_rule": AreaRuleParams,
    "internal_structure": InternalStructureParams,
}

# Preset configurations
_PRESETS: dict[str, Any] = {
    "ucav": create_ucav_params,
}


def _build_section(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass, ignoring unknown keys."""
    valid = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid}
    return cls(**filtered)


def load_config(path: str | Path) -> AircraftParams:
    """Load aircraft parameters from a YAML file."""
    logger.info("Loading config from %s", path)
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    sections: dict[str, Any] = {}
    for key, cls in _SECTION_MAP.items():
        if key in raw:
            sections[key] = _build_section(cls, raw[key])
            logger.debug("  Loaded section: %s", key)
        else:
            sections[key] = cls()
            logger.debug("  Default section: %s", key)

    params = AircraftParams(**sections)
    logger.info("Config loaded: %s (type=%s, Mach=%.2f)",
                params.meta.name, params.meta.aircraft_type, params.meta.design_mach)
    return params


def save_config(params: AircraftParams, path: str | Path) -> None:
    """Save aircraft parameters to a YAML file."""
    data = dataclasses.asdict(params)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_preset(name: str) -> AircraftParams:
    """Load a built-in preset configuration by name.

    Available presets: 'ucav', 'fighter' (default).
    """
    if name in _PRESETS:
        logger.info("Loading preset: %s", name)
        return _PRESETS[name]()
    logger.info("Loading default fighter preset (no preset named '%s')", name)
    return AircraftParams()


def params_to_dict(params: AircraftParams) -> dict[str, Any]:
    """Convert parameters to a plain dictionary."""
    return dataclasses.asdict(params)
