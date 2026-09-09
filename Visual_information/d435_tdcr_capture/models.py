"""Shared immutable data contracts for capture, replay and identification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

import numpy as np


KEYPOINT_COUNT = 7
KEYPOINT_ARCLENGTHS_M = np.asarray([0, 7, 14, 21, 28, 35, 42], dtype=float) / 1000.0


def _array(value: Any, shape: tuple[int, ...], *, dtype=float) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).reshape(shape).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CameraFrame:
    """One synchronized D435 frameset.

    Coordinates use the left infrared/depth optical frame: +X right, +Y down,
    +Z forward. Timestamps retain both device time and the mapped host clock.
    """

    sequence: int
    host_arrival_ns: int
    capture_host_ns: int
    device_timestamp_ms: float
    color_timestamp_ms: float
    depth_timestamp_ms: float
    infrared_timestamp_ms: float
    color_bgr: np.ndarray
    depth_m: np.ndarray
    infrared_left: np.ndarray
    infrared_right: np.ndarray
    color_intrinsics: Mapping[str, Any]
    left_intrinsics: Mapping[str, Any]
    right_intrinsics: Mapping[str, Any]
    transform_left_from_color: np.ndarray
    transform_right_from_left: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_depth_z16: np.ndarray | None = None
    depth_unit_m: float = 0.001
    native_depth_m: np.ndarray | None = None
    native_depth_color_bgr: np.ndarray | None = None
    depth_intrinsics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform_left_from_color", _array(self.transform_left_from_color, (4, 4)))
        object.__setattr__(self, "transform_right_from_left", _array(self.transform_right_from_left, (4, 4)))


@dataclass(frozen=True)
class RgbCameraFrame:
    """One frame from the auxiliary RGB camera.

    Ordinary USB cameras usually expose no hardware clock, so capture_host_ns
    is the host monotonic timestamp assigned by the background grabber.
    """

    sequence: int
    host_arrival_ns: int
    capture_host_ns: int
    image_bgr: np.ndarray
    intrinsics: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KeypointObservation:
    raw_camera_m: np.ndarray
    filtered_camera_m: np.ndarray
    base_m: np.ndarray
    covariance_m2: np.ndarray
    valid: np.ndarray
    predicted: np.ndarray
    confidence: np.ndarray
    color_pixels: np.ndarray
    left_pixels: np.ndarray
    right_pixels: np.ndarray
    source: Tuple[str, ...]
    color_confidence: np.ndarray = field(default_factory=lambda: np.zeros(KEYPOINT_COUNT))
    side_pixels: np.ndarray = field(default_factory=lambda: np.full((KEYPOINT_COUNT, 2), np.nan))
    side_reprojection_error_px: np.ndarray = field(default_factory=lambda: np.full(KEYPOINT_COUNT, np.nan))
    fusion_mode: str = "d435_internal"
    side_time_offset_ms: float = float("nan")

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_camera_m", _array(self.raw_camera_m, (KEYPOINT_COUNT, 3)))
        object.__setattr__(self, "filtered_camera_m", _array(self.filtered_camera_m, (KEYPOINT_COUNT, 3)))
        object.__setattr__(self, "base_m", _array(self.base_m, (KEYPOINT_COUNT, 3)))
        object.__setattr__(self, "covariance_m2", _array(self.covariance_m2, (KEYPOINT_COUNT, 3, 3)))
        object.__setattr__(self, "valid", _array(self.valid, (KEYPOINT_COUNT,), dtype=bool))
        object.__setattr__(self, "predicted", _array(self.predicted, (KEYPOINT_COUNT,), dtype=bool))
        object.__setattr__(self, "confidence", _array(self.confidence, (KEYPOINT_COUNT,)))
        object.__setattr__(self, "color_pixels", _array(self.color_pixels, (KEYPOINT_COUNT, 2)))
        object.__setattr__(self, "left_pixels", _array(self.left_pixels, (KEYPOINT_COUNT, 2)))
        object.__setattr__(self, "right_pixels", _array(self.right_pixels, (KEYPOINT_COUNT, 2)))
        object.__setattr__(self, "color_confidence", _array(self.color_confidence, (KEYPOINT_COUNT,)))
        object.__setattr__(self, "side_pixels", _array(self.side_pixels, (KEYPOINT_COUNT, 2)))
        object.__setattr__(self, "side_reprojection_error_px", _array(self.side_reprojection_error_px, (KEYPOINT_COUNT,)))
        if len(self.source) != KEYPOINT_COUNT:
            raise ValueError(f"source must contain {KEYPOINT_COUNT} entries")


def empty_keypoint_observation(
    fusion_mode: str = "d435_internal",
    source: str = "tracking_disabled",
) -> KeypointObservation:
    """Create an explicit all-invalid observation without inventing measurements."""

    return KeypointObservation(
        raw_camera_m=np.full((KEYPOINT_COUNT, 3), np.nan),
        filtered_camera_m=np.full((KEYPOINT_COUNT, 3), np.nan),
        base_m=np.full((KEYPOINT_COUNT, 3), np.nan),
        covariance_m2=np.full((KEYPOINT_COUNT, 3, 3), np.nan),
        valid=np.zeros(KEYPOINT_COUNT, dtype=bool),
        predicted=np.zeros(KEYPOINT_COUNT, dtype=bool),
        confidence=np.zeros(KEYPOINT_COUNT, dtype=float),
        color_pixels=np.full((KEYPOINT_COUNT, 2), np.nan),
        left_pixels=np.full((KEYPOINT_COUNT, 2), np.nan),
        right_pixels=np.full((KEYPOINT_COUNT, 2), np.nan),
        source=(source,) * KEYPOINT_COUNT,
        fusion_mode=fusion_mode,
    )


@dataclass(frozen=True)
class AxisSample:
    host_ns: int
    demand_native: np.ndarray
    measured_native: np.ndarray
    control_m: np.ndarray
    demand_valid: np.ndarray
    measured_valid: np.ndarray
    status: str = "ok"

    def __post_init__(self) -> None:
        for name, dtype in (
            ("demand_native", float),
            ("measured_native", float),
            ("control_m", float),
            ("demand_valid", bool),
            ("measured_valid", bool),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), (7,), dtype=dtype))


@dataclass(frozen=True)
class AlignedSample:
    frame: CameraFrame
    keypoints: KeypointObservation
    axes: AxisSample
    simulation_points_base_m: np.ndarray
    errors_mm: np.ndarray
    rmse_mm: float
    valid_count: int
    estimated_delay_ms: float = 0.0
    side_frame: RgbCameraFrame | None = None
    fusion_mode: str = "d435_internal"
    d435_tracking_roi: np.ndarray = field(default_factory=lambda: np.asarray([0.0, 0.0, 1.0, 1.0]))
    side_tracking_roi: np.ndarray = field(default_factory=lambda: np.asarray([0.0, 0.0, 1.0, 1.0]))

    def __post_init__(self) -> None:
        object.__setattr__(self, "simulation_points_base_m", _array(self.simulation_points_base_m, (KEYPOINT_COUNT, 3)))
        object.__setattr__(self, "errors_mm", _array(self.errors_mm, (KEYPOINT_COUNT,)))
        object.__setattr__(self, "d435_tracking_roi", _array(self.d435_tracking_roi, (4,)))
        object.__setattr__(self, "side_tracking_roi", _array(self.side_tracking_roi, (4,)))
