"""Uncertainty-aware fusion of aligned RGB-depth and D435 IR stereo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from Visual_information.d435_tdcr_capture.models import KEYPOINT_COUNT, CameraFrame, KeypointObservation

from .models import ModalityMeasurement


def _invalid(source: str) -> ModalityMeasurement:
    return ModalityMeasurement(
        np.full(3, np.nan), np.full((3, 3), np.nan), 0.0, False, source
    )


def _deproject(pixel: np.ndarray, depth_m: float, intrinsics: dict[str, Any]) -> np.ndarray:
    return np.asarray([
        (float(pixel[0]) - float(intrinsics["ppx"])) * depth_m / float(intrinsics["fx"]),
        (float(pixel[1]) - float(intrinsics["ppy"])) * depth_m / float(intrinsics["fy"]),
        depth_m,
    ])


def _transform(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    return matrix[:3, :3] @ point + matrix[:3, 3]


def _safe_inverse(covariance: np.ndarray) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    return (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T


def covariance_intersection(
    first: ModalityMeasurement,
    second: ModalityMeasurement,
) -> ModalityMeasurement:
    """Fuse correlated estimates without assuming statistical independence."""

    if not first.valid:
        return second
    if not second.valid:
        return first
    information_a = _safe_inverse(first.covariance_m2)
    information_b = _safe_inverse(second.covariance_m2)
    best = None
    for weight in np.linspace(0.0, 1.0, 21):
        information = weight * information_a + (1.0 - weight) * information_b
        covariance = _safe_inverse(information)
        score = float(np.linalg.slogdet(covariance)[1])
        if best is None or score < best[0]:
            mean = covariance @ (
                weight * information_a @ first.point_camera_m
                + (1.0 - weight) * information_b @ second.point_camera_m
            )
            best = (score, mean, covariance)
    confidence = 1.0 - (1.0 - first.confidence) * (1.0 - second.confidence)
    return ModalityMeasurement(
        best[1], best[2], float(np.clip(confidence, 0.0, 1.0)), True,
        "rgb_depth+ir_stereo_ci",
    )


@dataclass
class MeasurementBundle:
    fused_points_m: np.ndarray
    covariance_m2: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    sources: tuple[str, ...]
    depth_points_m: np.ndarray
    stereo_points_m: np.ndarray


class MultimodalMeasurementFuser:
    """Recover and robustly combine the two metric depth routes in a D435 frame."""

    def __init__(self, config: dict) -> None:
        self.camera = config["camera"]
        self.settings = config.get("seven_marker_fusion", {})

    def _depth_measurement(
        self,
        frame: CameraFrame,
        pixel: np.ndarray,
        reference_z_m: float,
        color_confidence: float,
    ) -> ModalityMeasurement:
        depth = np.asarray(frame.depth_m)
        if (
            depth.ndim != 2
            or depth.shape != frame.color_bgr.shape[:2]
            or not np.isfinite(pixel).all()
        ):
            return _invalid("rgb_aligned_depth")
        radius = int(self.settings.get("depth_roi_radius_px", self.camera.get("depth_roi_radius_px", 4)))
        u, v = np.rint(pixel).astype(int)
        y0, y1 = max(0, v - radius), min(depth.shape[0], v + radius + 1)
        x0, x1 = max(0, u - radius), min(depth.shape[1], u + radius + 1)
        values = depth[y0:y1, x0:x1].reshape(-1).astype(float)
        minimum = float(self.camera.get("depth_min_m", 0.18))
        maximum = float(self.camera.get("depth_max_m", 0.60))
        values = values[np.isfinite(values) & (values >= minimum) & (values <= maximum)]
        if len(values) < int(self.settings.get("minimum_depth_samples", 4)):
            return _invalid("rgb_aligned_depth")
        center = float(reference_z_m) if np.isfinite(reference_z_m) else float(np.median(values))
        gate = float(self.settings.get("depth_cluster_gate_m", 0.010))
        clustered = values[np.abs(values - center) <= gate]
        if len(clustered) >= 3:
            values = clustered
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_sigma = max(1.4826 * mad, float(self.settings.get("depth_noise_floor_m", 0.0007)))
        inliers = values[np.abs(values - median) <= max(3.5 * robust_sigma, 0.0015)]
        if len(inliers) >= 3:
            median = float(np.median(inliers))
            mad = float(np.median(np.abs(inliers - median)))
            robust_sigma = max(1.4826 * mad, float(self.settings.get("depth_noise_floor_m", 0.0007)))
            values = inliers
        point_color = _deproject(pixel, median, dict(frame.color_intrinsics))
        point_left = _transform(frame.transform_left_from_color, point_color)
        pixel_sigma = float(self.settings.get("rgb_pixel_sigma_px", 0.45))
        fx = max(float(frame.color_intrinsics["fx"]), 1.0)
        fy = max(float(frame.color_intrinsics["fy"]), 1.0)
        sigma_x = math.hypot(median * pixel_sigma / fx, robust_sigma * abs(point_color[0]) / max(median, 1e-6))
        sigma_y = math.hypot(median * pixel_sigma / fy, robust_sigma * abs(point_color[1]) / max(median, 1e-6))
        covariance_color = np.diag([
            max(sigma_x, 0.00020) ** 2,
            max(sigma_y, 0.00020) ** 2,
            robust_sigma**2,
        ])
        rotation = np.asarray(frame.transform_left_from_color)[:3, :3]
        covariance_left = rotation @ covariance_color @ rotation.T
        sample_quality = min(1.0, len(values) / max((2 * radius + 1) ** 2 * 0.45, 1.0))
        confidence = float(np.clip(color_confidence * (0.35 + 0.65 * sample_quality), 0.0, 1.0))
        return ModalityMeasurement(point_left, covariance_left, confidence, True, "rgb_aligned_depth")

    def _stereo_measurement(
        self,
        frame: CameraFrame,
        observation: KeypointObservation,
        index: int,
    ) -> ModalityMeasurement:
        left = observation.left_pixels[index]
        right = observation.right_pixels[index]
        if (
            observation.source[index] != "stereo_ir"
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
        ):
            return _invalid("ir_stereo")
        disparity = float(left[0] - right[0])
        vertical_error = abs(float(left[1] - right[1]))
        baseline = float(np.linalg.norm(np.asarray(frame.transform_right_from_left)[:3, 3]))
        fx = float(frame.left_intrinsics["fx"])
        if disparity <= 0.2 or baseline <= 1e-5 or vertical_error > float(
            self.camera.get("max_vertical_stereo_error_px", 2.5)
        ):
            return _invalid("ir_stereo")
        z = fx * baseline / disparity
        if not float(self.camera.get("depth_min_m", 0.18)) <= z <= float(self.camera.get("depth_max_m", 0.60)):
            return _invalid("ir_stereo")
        point = _deproject(left, z, dict(frame.left_intrinsics))
        quality = float(np.clip(observation.confidence[index], 0.05, 1.0))
        disparity_sigma = float(self.settings.get("stereo_disparity_sigma_px", 0.30)) / math.sqrt(quality)
        pixel_sigma = float(self.settings.get("ir_pixel_sigma_px", 0.30)) / math.sqrt(quality)
        sigma_z = max(0.00020, z * z * disparity_sigma / max(fx * baseline, 1e-9))
        sigma_xy = max(0.00012, z * pixel_sigma / max(fx, 1.0))
        covariance = np.diag([sigma_xy**2, sigma_xy**2, sigma_z**2])
        confidence = float(np.clip(quality * math.exp(-0.5 * vertical_error**2), 0.0, 1.0))
        return ModalityMeasurement(point, covariance, confidence, True, "ir_stereo")

    def _combine(
        self,
        depth: ModalityMeasurement,
        stereo: ModalityMeasurement,
        fallback: ModalityMeasurement,
    ) -> ModalityMeasurement:
        if depth.valid and stereo.valid:
            innovation = depth.point_camera_m - stereo.point_camera_m
            innovation_covariance = depth.covariance_m2 + stereo.covariance_m2
            mahalanobis = float(innovation @ _safe_inverse(innovation_covariance) @ innovation)
            euclidean = float(np.linalg.norm(innovation))
            if (
                mahalanobis <= float(self.settings.get("modality_mahalanobis_gate", 16.0))
                and euclidean <= float(self.settings.get("modality_distance_gate_m", 0.012))
            ):
                return covariance_intersection(depth, stereo)
            # Do not average across a foreground/background depth edge.  Prefer
            # the lower-uncertainty estimate but reduce its confidence.
            selected = min((depth, stereo), key=lambda item: float(np.trace(item.covariance_m2)))
            return ModalityMeasurement(
                selected.point_camera_m,
                selected.covariance_m2 * 2.5,
                selected.confidence * 0.55,
                True,
                selected.source + "_conflict_selected",
            )
        if depth.valid:
            return depth
        if stereo.valid:
            return stereo
        return fallback

    def _reject_nonphysical_chain_outliers(
        self,
        points: np.ndarray,
        covariance: np.ndarray,
        valid: np.ndarray,
        confidence: np.ndarray,
        sources: list[str],
    ) -> None:
        """Reject foreground/background depth jumps before shape smoothing.

        A 7 mm arc interval cannot have a Euclidean chord longer than 7 mm.
        We also compare every point to the local interpolation of its two
        neighbours.  The endpoint with lower confidence or larger covariance
        is removed, rather than letting the optimiser bend the entire robot to
        explain one transparent-wall depth sample.
        """

        maximum_chord = float(self.settings.get("chain_depth_maximum_chord_m", 0.0082))
        interpolation_gate = float(
            self.settings.get("chain_depth_interpolation_gate_m", 0.0060)
        )

        def reliability(index: int) -> float:
            trace = float(np.trace(covariance[index])) if np.isfinite(covariance[index]).all() else 1e-3
            return float(confidence[index]) / max(math.sqrt(max(trace, 1e-12)), 1e-6)

        for _ in range(3):
            changed = False
            for index in range(KEYPOINT_COUNT - 1):
                if not (valid[index] and valid[index + 1]):
                    continue
                distance = float(np.linalg.norm(points[index + 1] - points[index]))
                if distance <= maximum_chord:
                    continue
                reject = index if reliability(index) < reliability(index + 1) else index + 1
                valid[reject] = False
                points[reject] = np.nan
                covariance[reject] = np.nan
                confidence[reject] = 0.0
                sources[reject] += "_chain_outlier_rejected"
                changed = True
            if not changed:
                break
        for index in range(1, KEYPOINT_COUNT - 1):
            if not (valid[index - 1] and valid[index] and valid[index + 1]):
                continue
            expected = 0.5 * (points[index - 1] + points[index + 1])
            residual = float(np.linalg.norm(points[index] - expected))
            neighbour_span = float(np.linalg.norm(points[index + 1] - points[index - 1]))
            if residual > interpolation_gate and neighbour_span < 0.015:
                valid[index] = False
                points[index] = np.nan
                covariance[index] = np.nan
                confidence[index] = 0.0
                sources[index] += "_local_depth_outlier_rejected"

    def fuse(self, frame: CameraFrame, observation: KeypointObservation) -> MeasurementBundle:
        points = np.full((KEYPOINT_COUNT, 3), np.nan)
        covariance = np.full((KEYPOINT_COUNT, 3, 3), np.nan)
        valid = np.zeros(KEYPOINT_COUNT, dtype=bool)
        confidence = np.zeros(KEYPOINT_COUNT)
        depth_points = np.full_like(points, np.nan)
        stereo_points = np.full_like(points, np.nan)
        sources: list[str] = []
        for index in range(KEYPOINT_COUNT):
            raw = observation.raw_camera_m[index]
            fallback_valid = bool(observation.valid[index] and np.isfinite(raw).all())
            fallback_covariance = observation.covariance_m2[index]
            if not np.isfinite(fallback_covariance).all():
                fallback_covariance = np.eye(3) * 4e-6
            fallback = ModalityMeasurement(
                raw if fallback_valid else np.full(3, np.nan),
                fallback_covariance,
                float(observation.confidence[index]) if fallback_valid else 0.0,
                fallback_valid,
                observation.source[index] if fallback_valid else "invalid",
            )
            depth = self._depth_measurement(
                frame,
                observation.color_pixels[index],
                raw[2] if fallback_valid else math.nan,
                float(observation.color_confidence[index]),
            )
            stereo = self._stereo_measurement(frame, observation, index)
            combined = self._combine(depth, stereo, fallback)
            if depth.valid:
                depth_points[index] = depth.point_camera_m
            if stereo.valid:
                stereo_points[index] = stereo.point_camera_m
            if combined.valid:
                points[index] = combined.point_camera_m
                covariance[index] = combined.covariance_m2
                valid[index] = True
                confidence[index] = combined.confidence
            sources.append(combined.source)
        self._reject_nonphysical_chain_outliers(
            points, covariance, valid, confidence, sources
        )
        return MeasurementBundle(
            fused_points_m=points,
            covariance_m2=covariance,
            valid=valid,
            confidence=confidence,
            sources=tuple(sources),
            depth_points_m=depth_points,
            stereo_points_m=stereo_points,
        )
