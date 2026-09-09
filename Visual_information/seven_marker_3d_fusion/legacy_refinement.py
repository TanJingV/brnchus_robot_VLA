"""Use the full RGB-aligned depth neighbourhood to refine legacy recordings."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from Visual_information.d435_tdcr_capture.models import CameraFrame, KEYPOINT_COUNT

from .models import SevenMarkerResult


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _project_pixel(pixel: np.ndarray, depth_m: float, intrinsics: dict) -> np.ndarray:
    fx = float(intrinsics.get("fx", 1.0))
    fy = float(intrinsics.get("fy", 1.0))
    cx = float(intrinsics.get("ppx", intrinsics.get("cx", 0.0)))
    cy = float(intrinsics.get("ppy", intrinsics.get("cy", 0.0)))
    u, v = map(float, pixel)
    return np.asarray([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m])


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    output = np.full_like(points, np.nan)
    finite = np.isfinite(points).all(axis=1)
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    output[finite] = points[finite] @ matrix[:3, :3].T + matrix[:3, 3]
    return output


class LocalDepthCurveFusion:
    """Refine predicted markers with the depth field around the RGB curve.

    Black tubing and specular rings often contain no D435 disparity.  Reading a
    single pixel therefore fails.  For an already localised marker we instead
    inspect concentric neighbourhoods, reject samples inconsistent with the
    motor/temporal depth prior, and fit one smooth z(s) profile through all seven
    arc-length positions.  The observation stays marked as predicted because a
    neighbouring surface is weaker evidence than depth on the marker itself.
    """

    def __init__(
        self,
        enabled: bool = True,
        search_radius_px: int = 22,
        depth_gate_m: float = 0.018,
    ) -> None:
        self.enabled = bool(enabled)
        self.search_radius_px = max(6, int(search_radius_px))
        self.depth_gate_m = max(0.003, float(depth_gate_m))

    def _local_depth(
        self,
        depth_m: np.ndarray,
        pixel: np.ndarray,
        predicted_depth_m: float,
    ) -> tuple[float, float]:
        height, width = depth_m.shape[:2]
        u, v = np.rint(pixel).astype(int)
        radius = self.search_radius_px
        x0, x1 = max(0, u - radius), min(width, u + radius + 1)
        y0, y1 = max(0, v - radius), min(height, v + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return float("nan"), 0.0
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)
        # Skip the central 3 px, which are most likely the original disparity
        # hole, while retaining the tube edge and immediately adjacent surface.
        support = (distance >= 3.0) & (distance <= radius)
        values = np.asarray(depth_m[y0:y1, x0:x1], dtype=float)
        support &= np.isfinite(values) & (values > 0.04)
        if np.isfinite(predicted_depth_m):
            support &= np.abs(values - predicted_depth_m) <= self.depth_gate_m
        selected = values[support]
        selected_distance = distance[support]
        if len(selected) < 6:
            return float("nan"), 0.0
        weights = 1.0 / np.maximum(selected_distance, 1.0)
        center = _weighted_median(selected, weights)
        absolute = np.abs(selected - center)
        mad = _weighted_median(absolute, weights)
        robust = absolute <= max(0.0015, 3.5 * mad)
        if np.count_nonzero(robust) < 5:
            return float("nan"), 0.0
        center = _weighted_median(selected[robust], weights[robust])
        dispersion = float(np.median(np.abs(selected[robust] - center)))
        count_score = min(1.0, np.count_nonzero(robust) / 80.0)
        noise_score = float(np.exp(-dispersion / 0.004))
        return center, float(np.clip(0.15 + 0.70 * count_score * noise_score, 0.0, 0.85))

    @staticmethod
    def _smooth_depth_profile(
        current_points: np.ndarray,
        measured: np.ndarray,
        local_depth: np.ndarray,
        local_confidence: np.ndarray,
    ) -> np.ndarray:
        prior = current_points[:, 2].copy()
        target = prior.copy()
        weights = np.where(np.isfinite(prior), 0.75, 0.0)
        target[measured] = current_points[measured, 2]
        weights[measured] = 18.0
        local = np.isfinite(local_depth)
        target[local] = local_depth[local]
        weights[local] += 3.0 + 8.0 * local_confidence[local]
        finite = np.isfinite(target) & (weights > 0)
        if np.count_nonzero(finite) < 2:
            return prior
        if not np.all(finite):
            indices = np.arange(KEYPOINT_COUNT)
            target[~finite] = np.interp(indices[~finite], indices[finite], target[finite])
            weights[~finite] = 0.08
        second = np.zeros((KEYPOINT_COUNT - 2, KEYPOINT_COUNT))
        for row in range(KEYPOINT_COUNT - 2):
            second[row, row:row + 3] = (1.0, -2.0, 1.0)
        matrix = np.diag(weights) + 4.5 * (second.T @ second) + np.eye(KEYPOINT_COUNT) * 1e-8
        return np.linalg.solve(matrix, weights * target)

    def fuse(
        self,
        result: SevenMarkerResult,
        frame: CameraFrame,
        transform_base_from_camera: np.ndarray,
    ) -> SevenMarkerResult:
        if not self.enabled or frame.depth_m.size == 0:
            return result
        points = result.smoothed_camera_m.copy()
        measured = result.measured_valid.copy()
        predicted = result.predicted.copy()
        confidence = result.confidence.copy()
        covariance = result.covariance_m2.copy()
        sources = list(result.sources)
        local_depth = np.full(KEYPOINT_COUNT, np.nan)
        local_confidence = np.zeros(KEYPOINT_COUNT)
        for index in range(KEYPOINT_COUNT):
            if measured[index] or not np.isfinite(result.color_pixels[index]).all():
                continue
            prior_depth = points[index, 2] if np.isfinite(points[index]).all() else float("nan")
            local_depth[index], local_confidence[index] = self._local_depth(
                frame.depth_m, result.color_pixels[index], prior_depth
            )
        profile = self._smooth_depth_profile(points, measured, local_depth, local_confidence)
        for index in range(KEYPOINT_COUNT):
            if measured[index] or not np.isfinite(result.color_pixels[index]).all():
                continue
            if not np.isfinite(profile[index]):
                continue
            projected = _project_pixel(result.color_pixels[index], float(profile[index]), frame.color_intrinsics)
            had_prior = np.isfinite(points[index]).all()
            if had_prior:
                visual_weight = 0.72 if np.isfinite(local_depth[index]) else 0.48
                points[index] = visual_weight * projected + (1.0 - visual_weight) * points[index]
            else:
                points[index] = projected
            predicted[index] = True
            direct = np.isfinite(local_depth[index])
            confidence[index] = max(
                float(confidence[index]),
                0.28 + 0.22 * float(local_confidence[index]) if direct else 0.24,
            )
            sigma = 0.0030 if direct else 0.0055
            covariance[index] = np.eye(3) * sigma ** 2
            if direct and had_prior:
                sources[index] = "rgb_local_depth+motor_pcc"
            elif direct:
                sources[index] = "rgb_local_depth_curve"
            elif had_prior:
                sources[index] = "rgb_curve+motor_pcc"
            else:
                sources[index] = "rgb_depth_curve"
        return replace(
            result,
            smoothed_camera_m=points,
            base_m=_transform_points(transform_base_from_camera, points),
            covariance_m2=covariance,
            predicted=predicted,
            confidence=confidence,
            sources=tuple(sources),
        )
