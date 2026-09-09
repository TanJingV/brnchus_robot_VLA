"""Configuration loading with deterministic defaults and validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .models import KEYPOINT_COUNT


PACKAGE_DIR = Path(__file__).resolve().parent
VISUAL_ROOT = PACKAGE_DIR.parent
PROJECT_ROOT = VISUAL_ROOT.parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "default_config.json"


def _merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if path is not None:
        user_path = Path(path).expanduser().resolve()
        with user_path.open("r", encoding="utf-8") as stream:
            config = _merge(config, json.load(stream))
        config["config_path"] = str(user_path)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    markers = config["markers"]
    if len(markers) != KEYPOINT_COUNT:
        raise ValueError(f"Exactly {KEYPOINT_COUNT} marker definitions are required")
    arclengths = [float(marker["arclength_mm"]) for marker in markers]
    if arclengths != sorted(arclengths) or arclengths[0] != 0.0 or arclengths[-1] != 42.0:
        raise ValueError("Marker arclengths must be ordered from 0 to 42 mm")
    mapping = config["axes"]
    for key in ("zero_native", "scale_m_per_native", "neutral_tendon_lengths_m"):
        expected = 6 if key == "neutral_tendon_lengths_m" else 7
        if len(mapping[key]) != expected:
            raise ValueError(f"axes.{key} must contain {expected} values")
    transform = config["calibration"]["transform_base_from_camera"]
    if len(transform) != 4 or any(len(row) != 4 for row in transform):
        raise ValueError("calibration.transform_base_from_camera must be 4x4")
    mode = str(config["fusion"]["mode"])
    if mode not in ("d435_internal", "dual_view"):
        raise ValueError("fusion.mode must be d435_internal or dual_view")
    depth_view_space = str(config.get("ui", {}).get("depth_view_space", "native"))
    if depth_view_space not in ("native", "aligned"):
        raise ValueError("ui.depth_view_space must be native or aligned")
    side = config["side_camera"]
    side_transform = side["transform_base_from_camera"]
    if len(side_transform) != 4 or any(len(row) != 4 for row in side_transform):
        raise ValueError("side_camera.transform_base_from_camera must be 4x4")
    side_intrinsics = side["intrinsics"]
    for key in ("width", "height", "fx", "fy", "ppx", "ppy"):
        if key not in side_intrinsics:
            raise ValueError(f"side_camera.intrinsics.{key} is required")


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
    return target
