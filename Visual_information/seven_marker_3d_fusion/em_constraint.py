"""Coordinate-frame invariant constraints from two recorded NDI probes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace

import numpy as np

from .curve_model import project_material_keypoints
from .models import SevenMarkerResult


@dataclass(frozen=True)
class EmPairObservation:
    valid: bool
    first_port: int
    second_port: int
    chord_m: float
    relative_angle_rad: float


def extract_em_pair(em_row: dict, first_port: int = 10, second_port: int = 11) -> EmPairObservation:
    invalid = EmPairObservation(False, int(first_port), int(second_port), float("nan"), float("nan"))
    if not em_row:
        return invalid
    try:
        payload = em_row.get("all_tools_json", "[]")
        tools = json.loads(payload) if isinstance(payload, str) else list(payload)
        by_port = {int(tool["port_handle"]): tool for tool in tools if bool(tool.get("valid", True))}
        if first_port not in by_port or second_port not in by_port:
            available = sorted(by_port)
            if len(available) < 2:
                return invalid
            first_port, second_port = available[:2]
        first, second = by_port[int(first_port)], by_port[int(second_port)]
        first_position = np.asarray(first["translation_mm"], dtype=float) * 0.001
        second_position = np.asarray(second["translation_mm"], dtype=float) * 0.001
        first_transform = np.asarray(first["transform_row_major"], dtype=float).reshape(4, 4)
        second_transform = np.asarray(second["transform_row_major"], dtype=float).reshape(4, 4)
        relative_rotation = first_transform[:3, :3].T @ second_transform[:3, :3]
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
        chord = float(np.linalg.norm(second_position - first_position))
        if not np.isfinite(chord) or not 0.005 <= chord <= 0.080:
            return invalid
        return EmPairObservation(
            True, int(first_port), int(second_port), chord, math.acos(cosine)
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return invalid


class EmChordConstraintFusion:
    """Use only the frame-invariant probe separation, not uncalibrated EM XYZ."""

    def __init__(
        self,
        enabled: bool = True,
        first_port: int = 10,
        second_port: int = 11,
        gain: float = 0.92,
        endpoint_offset_m: float = 0.002,
    ) -> None:
        self.enabled = bool(enabled)
        self.first_port = int(first_port)
        self.second_port = int(second_port)
        self.gain = float(np.clip(gain, 0.0, 1.0))
        self.endpoint_offset_m = float(np.clip(endpoint_offset_m, -0.010, 0.010))
        self.last_observation = extract_em_pair({})

    def fuse(
        self,
        result: SevenMarkerResult,
        em_row: dict,
        transform_base_from_camera: np.ndarray,
    ) -> SevenMarkerResult:
        self.last_observation = extract_em_pair(em_row, self.first_port, self.second_port)
        if not self.enabled or not self.last_observation.valid:
            return result
        points = result.smoothed_camera_m.copy()
        if not np.isfinite(points[[0, 6]]).all():
            return result
        direction = points[6] - points[0]
        current = float(np.linalg.norm(direction))
        if current < 1e-7:
            return result
        # The probe coils are normally mounted slightly inside K0/K6.  The
        # straight calibration offset maps coil separation to material-end
        # separation; it is user-adjustable in the UI.
        target = float(np.clip(
            self.last_observation.chord_m + self.endpoint_offset_m, 0.010, 0.042
        ))
        # Reject a one-frame NDI jump instead of forcing the visual reconstruction
        # to follow an implausible discontinuity.
        if abs(target - current) > 0.035:
            return result
        unit = direction / current
        endpoint_delta = unit * (target - current) * self.gain
        measured = np.asarray(result.measured_valid, dtype=bool)
        first_mobility = 0.25 if measured[0] else 1.0
        second_mobility = 0.25 if measured[6] else 1.0
        total = first_mobility + second_mobility
        first_shift = -endpoint_delta * first_mobility / total
        second_shift = endpoint_delta * second_mobility / total
        for index, fraction in enumerate(np.linspace(0.0, 1.0, 7)):
            points[index] += (1.0 - fraction) * first_shift + fraction * second_shift
        points = project_material_keypoints(points, result.confidence, result.measured_valid)
        transform = np.asarray(transform_base_from_camera, dtype=float).reshape(4, 4)
        base = points @ transform[:3, :3].T + transform[:3, 3]
        return replace(result, smoothed_camera_m=points, base_m=base)
