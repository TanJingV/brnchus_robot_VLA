"""Data contracts local to the seven-marker reconstruction pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Visual_information.d435_tdcr_capture.models import KEYPOINT_COUNT


def _readonly(value, shape, dtype=float) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(shape).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ModalityMeasurement:
    point_camera_m: np.ndarray
    covariance_m2: np.ndarray
    confidence: float
    valid: bool
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_camera_m", _readonly(self.point_camera_m, (3,)))
        object.__setattr__(self, "covariance_m2", _readonly(self.covariance_m2, (3, 3)))


@dataclass(frozen=True)
class SevenMarkerResult:
    sequence: int
    capture_host_ns: int
    raw_camera_m: np.ndarray
    fused_camera_m: np.ndarray
    smoothed_camera_m: np.ndarray
    base_m: np.ndarray
    covariance_m2: np.ndarray
    measured_valid: np.ndarray
    predicted: np.ndarray
    confidence: np.ndarray
    color_pixels: np.ndarray
    left_pixels: np.ndarray
    right_pixels: np.ndarray
    sources: tuple[str, ...]
    rgb_depth_camera_m: np.ndarray
    ir_stereo_camera_m: np.ndarray
    processing_ms: float

    def __post_init__(self) -> None:
        for name in (
            "raw_camera_m", "fused_camera_m", "smoothed_camera_m", "base_m",
            "rgb_depth_camera_m", "ir_stereo_camera_m",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), (KEYPOINT_COUNT, 3)))
        object.__setattr__(self, "covariance_m2", _readonly(self.covariance_m2, (KEYPOINT_COUNT, 3, 3)))
        for name in ("measured_valid", "predicted"):
            object.__setattr__(self, name, _readonly(getattr(self, name), (KEYPOINT_COUNT,), bool))
        object.__setattr__(self, "confidence", _readonly(self.confidence, (KEYPOINT_COUNT,)))
        for name in ("color_pixels", "left_pixels", "right_pixels"):
            object.__setattr__(self, name, _readonly(getattr(self, name), (KEYPOINT_COUNT, 2)))
        if len(self.sources) != KEYPOINT_COUNT:
            raise ValueError(f"sources must contain {KEYPOINT_COUNT} entries")
