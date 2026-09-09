"""D435 plus auxiliary RGB geometric fusion for the seven TDCR markers."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .models import CameraFrame, KEYPOINT_COUNT, KeypointObservation, RgbCameraFrame
from .tracking import (
    ConstantVelocityFilter,
    MarkerTracker,
    detect_color_bands,
    scaled_tracking_config,
)


D435_INTERNAL_MODE = "d435_internal"
DUAL_VIEW_MODE = "dual_view"
FUSION_MODES = (D435_INTERNAL_MODE, DUAL_VIEW_MODE)


def _camera_matrix(intrinsics: dict[str, Any]) -> np.ndarray:
    return np.asarray([
        [intrinsics["fx"], 0.0, intrinsics["ppx"]],
        [0.0, intrinsics["fy"], intrinsics["ppy"]],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def _transform(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ point + transform[:3, 3]


def _project_base(point_base: np.ndarray, transform_base_from_camera: np.ndarray, intrinsics: dict[str, Any]) -> np.ndarray:
    point = _transform(np.linalg.inv(transform_base_from_camera), point_base)
    if not np.isfinite(point).all() or point[2] <= 0:
        return np.full(2, np.nan)
    return np.asarray([
        intrinsics["fx"] * point[0] / point[2] + intrinsics["ppx"],
        intrinsics["fy"] * point[1] / point[2] + intrinsics["ppy"],
    ])


class SideViewFusion:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.fusion_config = config["fusion"]
        self.side_config = config["side_camera"]
        self.marker_config = config["markers"]
        self.tracking_config = config["tracking"]
        self.transform_base_from_d435 = np.asarray(
            config["calibration"]["transform_base_from_camera"], dtype=float
        )
        self.transform_base_from_side = np.asarray(
            self.side_config["transform_base_from_camera"], dtype=float
        )
        self.calibration_ready = bool(
            self.side_config.get("calibration_ready", False)
            and self.side_config.get("intrinsics_ready", False)
        )
        self.last_side_pixels = np.full((KEYPOINT_COUNT, 2), np.nan)
        self.last_continuum_segmentation = None
        self.filters = [ConstantVelocityFilter(self.tracking_config) for _ in range(KEYPOINT_COUNT)]
        self.latest_side_overlay: np.ndarray | None = None
        self.side_roi_normalized = (0.0, 0.0, 1.0, 1.0)
        self.side_color_prototypes = self._load_side_color_prototypes()

    def _load_side_color_prototypes(self) -> list[dict[str, float]]:
        saved = self.tracking_config.get("side_color_prototypes", {})
        result = []
        for marker in self.marker_config:
            bgr = np.asarray(marker["display_bgr"], dtype=np.uint8).reshape(1, 1, 3)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]
            value = saved.get(marker["name"], {}) if isinstance(saved, dict) else {}
            result.append({
                "hue": float(value.get("hue", hsv[0])),
                "lab_a": float(value.get("lab_a", lab[1])),
                "lab_b": float(value.get("lab_b", lab[2])),
            })
        return result

    def _save_side_color_prototypes(self) -> None:
        self.tracking_config["side_color_prototypes"] = {
            marker["name"]: {key: float(value) for key, value in prototype.items()}
            for marker, prototype in zip(self.marker_config, self.side_color_prototypes)
        }

    def reset(self) -> None:
        self.last_side_pixels[:] = np.nan
        self.last_continuum_segmentation = None
        self.filters = [ConstantVelocityFilter(self.tracking_config) for _ in range(KEYPOINT_COUNT)]

    def set_d435_transform(self, transform: np.ndarray) -> None:
        self.transform_base_from_d435 = np.asarray(transform, dtype=float).reshape(4, 4).copy()
        self.reset()

    def set_side_calibration(
        self,
        transform_base_from_side: np.ndarray,
        intrinsics: dict[str, Any] | None = None,
        ready: bool = True,
    ) -> None:
        transform = np.asarray(transform_base_from_side, dtype=float).reshape(4, 4)
        if not np.isfinite(transform).all():
            raise ValueError("侧相机外参包含非有限值")
        self.transform_base_from_side = transform.copy()
        self.side_config["transform_base_from_camera"] = transform.tolist()
        self.side_config["calibration_ready"] = bool(ready)
        if intrinsics is not None:
            self.side_config["intrinsics"] = dict(intrinsics)
            self.side_config["intrinsics_ready"] = True
        self.calibration_ready = bool(
            ready and self.side_config.get("intrinsics_ready", False)
        )
        self.reset()

    def set_side_roi(self, roi_normalized: tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = map(float, roi_normalized)
        x0, x1 = sorted((float(np.clip(x0, 0.0, 1.0)), float(np.clip(x1, 0.0, 1.0))))
        y0, y1 = sorted((float(np.clip(y0, 0.0, 1.0)), float(np.clip(y1, 0.0, 1.0))))
        if x1 - x0 < 1e-4 or y1 - y0 < 1e-4:
            raise ValueError("Side tracking ROI must have a non-zero area")
        self.side_roi_normalized = (x0, y0, x1, y1)

    def _apply_side_roi(self, mask: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.side_roi_normalized
        if x0 <= 0.0 and y0 <= 0.0 and x1 >= 1.0 and y1 >= 1.0:
            return mask
        height, width = mask.shape
        left, top = int(np.floor(x0 * width)), int(np.floor(y0 * height))
        right, bottom = int(np.ceil(x1 * width)), int(np.ceil(y1 * height))
        restricted = np.zeros_like(mask)
        restricted[max(0, top):min(height, bottom), max(0, left):min(width, right)] = mask[
            max(0, top):min(height, bottom), max(0, left):min(width, right)
        ]
        return restricted

    def _color_blob(
        self,
        hsv: np.ndarray,
        lab: np.ndarray,
        marker: dict,
        previous: np.ndarray,
        marker_index: int,
    ) -> tuple[np.ndarray, float]:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in marker["hsv_ranges"]:
            mask |= cv2.inRange(hsv, np.asarray(low, np.uint8), np.asarray(high, np.uint8))
        prototype = self.side_color_prototypes[marker_index]
        expected_hue = float(prototype["hue"])
        half_width = float(self.tracking_config.get("adaptive_hue_half_width", 12.0))
        hue_low, hue_high = expected_hue - half_width, expected_hue + half_width
        saturation = int(self.tracking_config.get("minimum_saturation", 55))
        value = int(self.tracking_config.get("minimum_value", 35))
        if hue_low < 0.0:
            adaptive = cv2.inRange(hsv, np.asarray([0, saturation, value], np.uint8), np.asarray([int(np.ceil(hue_high)), 255, 255], np.uint8))
            adaptive |= cv2.inRange(hsv, np.asarray([int(np.floor(180 + hue_low)), saturation, value], np.uint8), np.asarray([179, 255, 255], np.uint8))
        elif hue_high >= 180.0:
            adaptive = cv2.inRange(hsv, np.asarray([int(np.floor(hue_low)), saturation, value], np.uint8), np.asarray([179, 255, 255], np.uint8))
            adaptive |= cv2.inRange(hsv, np.asarray([0, saturation, value], np.uint8), np.asarray([int(np.ceil(hue_high - 180)), 255, 255], np.uint8))
        else:
            adaptive = cv2.inRange(hsv, np.asarray([int(np.floor(hue_low)), saturation, value], np.uint8), np.asarray([int(np.ceil(hue_high)), 255, 255], np.uint8))
        mask |= adaptive
        mask = self._apply_side_roi(mask)
        radius = int(self.tracking_config.get("morphology_radius_px", 1))
        if radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = self._apply_side_roi(mask)
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        minimum = float(self.tracking_config["minimum_blob_area_px"])
        maximum = float(self.tracking_config["maximum_blob_area_px"])
        hue_sigma = max(float(self.tracking_config.get("hue_sigma", 9.0)), 1e-3)
        lab_sigma = max(float(self.tracking_config.get("lab_chroma_sigma", 32.0)), 1e-3)
        temporal_sigma = max(float(self.tracking_config.get("temporal_sigma_px", 42.0)), 1e-3)
        maximum_jump = float(self.tracking_config.get("maximum_temporal_jump_px", 110.0))
        candidates = []
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if minimum <= area <= maximum:
                component = _labels == label
                component_hue = hsv[..., 0][component].astype(np.float32)
                hue_delta = np.abs(component_hue - expected_hue)
                hue_distance = np.minimum(hue_delta, 180.0 - hue_delta)
                lab_a = lab[..., 1][component].astype(np.float32)
                lab_b = lab[..., 2][component].astype(np.float32)
                lab_distance = np.sqrt(
                    (lab_a - float(prototype["lab_a"])) ** 2
                    + (lab_b - float(prototype["lab_b"])) ** 2
                )
                likelihood = np.exp(-0.5 * (hue_distance / hue_sigma) ** 2) * np.exp(-0.5 * (lab_distance / lab_sigma) ** 2)
                weights = likelihood * np.sqrt(np.clip(hsv[..., 1][component].astype(np.float32) / 255.0, 0.05, 1.0)) + 1e-6
                yy, xx = np.nonzero(component)
                center = np.asarray([np.sum(weights * xx) / np.sum(weights), np.sum(weights * yy) / np.sum(weights)], dtype=float)
                distance = float(np.linalg.norm(center - previous)) if np.isfinite(previous).all() else 0.0
                if np.isfinite(previous).all() and distance > maximum_jump:
                    continue
                temporal_quality = math.exp(-0.5 * (distance / temporal_sigma) ** 2) if np.isfinite(previous).all() else 1.0
                color_quality = float(np.clip(np.mean(likelihood), 0.0, 1.0))
                area_quality = float(np.clip(area / max(minimum * 8.0, 1.0), 0.15, 1.0))
                score = (0.25 + 0.75 * area_quality) * (0.10 + 0.90 * color_quality) ** 2 * (0.15 + 0.85 * temporal_quality)
                candidates.append((score, area, center, color_quality, temporal_quality))
        if not candidates:
            return np.full(2, np.nan), 0.0
        _score, area, center, color_quality, temporal_quality = max(candidates, key=lambda value: value[0])
        area_quality = float(np.clip(area / max(minimum * 8.0, 1.0), 0.15, 1.0))
        confidence = float(np.clip(area_quality * (0.25 + 0.75 * color_quality) * math.sqrt(max(temporal_quality, 0.0)), 0.0, 1.0))
        return center, confidence

    def detect_side(self, frame: RgbCameraFrame) -> tuple[np.ndarray, np.ndarray]:
        height, width = frame.image_bgr.shape[:2]
        maximum_width = max(
            320, int(self.tracking_config.get("processing_max_width_px", 416))
        )
        scale = min(1.0, maximum_width / float(width))
        processing_size = (int(round(width * scale)), int(round(height * scale)))
        image = (
            cv2.resize(frame.image_bgr, processing_size, interpolation=cv2.INTER_AREA)
            if scale < 0.999
            else frame.image_bgr
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        previous_pixels = self.last_side_pixels * scale if scale < 0.999 else self.last_side_pixels
        tracking_config = scaled_tracking_config(self.tracking_config, scale)
        pixels, confidence, _component_masks, _observed_colours, segmentation = detect_color_bands(
            hsv,
            lab,
            self.marker_config,
            self.side_color_prototypes,
            tracking_config,
            previous_pixels,
            self.side_roi_normalized,
        )
        self.last_continuum_segmentation = segmentation
        if scale < 0.999:
            pixels /= scale
        for index, pixel in enumerate(pixels):
            if np.isfinite(pixel).all():
                self.last_side_pixels[index] = pixel
        return pixels, confidence

    def calibrate_color_prototypes(self, frame: RgbCameraFrame, pixels: np.ndarray) -> int:
        hsv = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2LAB)
        calibrated = 0
        radius = 6
        for index, pixel in enumerate(pixels):
            if not np.isfinite(pixel).all():
                continue
            u, v = map(int, np.round(pixel))
            y0, y1 = max(0, v - radius), min(hsv.shape[0], v + radius + 1)
            x0, x1 = max(0, u - radius), min(hsv.shape[1], u + radius + 1)
            hp, lp = hsv[y0:y1, x0:x1], lab[y0:y1, x0:x1]
            selected = (hp[..., 1] >= int(self.tracking_config.get("minimum_saturation", 55))) & (hp[..., 2] >= int(self.tracking_config.get("minimum_value", 35)))
            if np.count_nonzero(selected) < 4:
                continue
            weights = hp[..., 1][selected].astype(float) / 255.0 + 0.05
            self.side_color_prototypes[index] = {
                "hue": MarkerTracker._circular_hue_mean(hp[..., 0][selected], weights),
                "lab_a": float(np.average(lp[..., 1][selected], weights=weights)),
                "lab_b": float(np.average(lp[..., 2][selected], weights=weights)),
            }
            calibrated += 1
        self._save_side_color_prototypes()
        return calibrated

    @staticmethod
    def _undistort(pixel: np.ndarray, intrinsics: dict[str, Any]) -> np.ndarray:
        result = cv2.undistortPoints(
            np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2),
            _camera_matrix(intrinsics),
            np.asarray(intrinsics.get("coeffs", [0.0] * 5), dtype=np.float64),
        )
        return result.reshape(2)

    def _triangulate(
        self,
        d435_pixel: np.ndarray,
        side_pixel: np.ndarray,
        d435_intrinsics: dict[str, Any],
        side_intrinsics: dict[str, Any],
        transform_base_from_d435_color: np.ndarray,
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        if not np.isfinite(d435_pixel).all() or not np.isfinite(side_pixel).all():
            return np.full(3, np.nan), math.nan, math.nan, np.full(2, np.nan)
        d435_from_base = np.linalg.inv(transform_base_from_d435_color)
        side_from_base = np.linalg.inv(self.transform_base_from_side)
        projection_d435 = d435_from_base[:3, :]
        projection_side = side_from_base[:3, :]
        point_d435 = self._undistort(d435_pixel, d435_intrinsics)
        point_side = self._undistort(side_pixel, side_intrinsics)
        homogeneous = cv2.triangulatePoints(
            projection_d435,
            projection_side,
            point_d435.reshape(2, 1),
            point_side.reshape(2, 1),
        ).reshape(4)
        if abs(float(homogeneous[3])) < 1e-12:
            return np.full(3, np.nan), math.nan, math.nan, np.full(2, np.nan)
        point_base = homogeneous[:3] / homogeneous[3]
        point_in_d435 = _transform(d435_from_base, point_base)
        point_in_side = _transform(side_from_base, point_base)
        if point_in_d435[2] <= 0 or point_in_side[2] <= 0:
            return np.full(3, np.nan), math.nan, math.nan, np.full(2, np.nan)
        projected_d435 = _project_base(point_base, transform_base_from_d435_color, d435_intrinsics)
        projected_side = _project_base(point_base, self.transform_base_from_side, side_intrinsics)
        reprojection = float(max(
            np.linalg.norm(projected_d435 - d435_pixel),
            np.linalg.norm(projected_side - side_pixel),
        ))
        ray_d435 = point_base - transform_base_from_d435_color[:3, 3]
        ray_side = point_base - self.transform_base_from_side[:3, 3]
        denominator = max(float(np.linalg.norm(ray_d435) * np.linalg.norm(ray_side)), 1e-12)
        cosine = float(np.clip(np.dot(ray_d435, ray_side) / denominator, -1.0, 1.0))
        ray_angle_deg = float(np.degrees(np.arccos(abs(cosine))))
        return point_base, reprojection, ray_angle_deg, projected_side

    @staticmethod
    def _precision_fuse(
        first: np.ndarray,
        first_covariance: np.ndarray,
        second: np.ndarray,
        second_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            p1 = np.linalg.inv(first_covariance)
            p2 = np.linalg.inv(second_covariance)
            covariance = np.linalg.inv(p1 + p2)
            point = covariance @ (p1 @ first + p2 @ second)
            return point, covariance
        except np.linalg.LinAlgError:
            return 0.5 * (first + second), 0.5 * (first_covariance + second_covariance)

    def fuse(
        self,
        frame: CameraFrame,
        internal: KeypointObservation,
        side_frame: RgbCameraFrame | None,
        side_detection: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> KeypointObservation:
        if side_frame is None:
            self.latest_side_overlay = None
            return KeypointObservation(
                raw_camera_m=internal.raw_camera_m,
                filtered_camera_m=internal.filtered_camera_m,
                base_m=internal.base_m,
                covariance_m2=internal.covariance_m2,
                valid=internal.valid,
                predicted=internal.predicted,
                confidence=internal.confidence,
                color_pixels=internal.color_pixels,
                left_pixels=internal.left_pixels,
                right_pixels=internal.right_pixels,
                source=tuple(f"{value}|side_missing" for value in internal.source),
                color_confidence=internal.color_confidence,
                fusion_mode=DUAL_VIEW_MODE,
            )

        side_pixels, side_confidence = (
            side_detection if side_detection is not None else self.detect_side(side_frame)
        )
        time_offset_ms = (side_frame.capture_host_ns - frame.capture_host_ns) * 1e-6
        side_overlay = side_frame.image_bgr.copy()
        segmentation = self.last_continuum_segmentation
        if segmentation is not None and segmentation.valid:
            body_mask = segmentation.body_mask
            if body_mask.shape != side_overlay.shape[:2]:
                body_mask = cv2.resize(
                    body_mask, side_overlay.shape[1::-1], interpolation=cv2.INTER_NEAREST
                )
            contours, _hierarchy = cv2.findContours(
                body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(side_overlay, contours, -1, (70, 220, 255), 1, cv2.LINE_AA)
            centerline = segmentation.centerline_xy.copy()
            if len(centerline):
                centerline[:, 0] *= side_overlay.shape[1] / float(segmentation.body_mask.shape[1])
                centerline[:, 1] *= side_overlay.shape[0] / float(segmentation.body_mask.shape[0])
                cv2.polylines(
                    side_overlay,
                    [np.round(centerline).astype(np.int32)],
                    False,
                    (70, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
        x0, y0, x1, y1 = self.side_roi_normalized
        if x0 > 0.0 or y0 > 0.0 or x1 < 1.0 or y1 < 1.0:
            height, width = side_overlay.shape[:2]
            corner0 = (int(round(x0 * width)), int(round(y0 * height)))
            corner1 = (int(round(x1 * width)) - 1, int(round(y1 * height)) - 1)
            cv2.rectangle(side_overlay, corner0, corner1, (255, 210, 70), 2, cv2.LINE_AA)
            cv2.putText(side_overlay, "SIDE TRACKING ROI", (corner0[0] + 7, max(20, corner0[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 210, 70), 1, cv2.LINE_AA)
        if not self.calibration_ready:
            for index, marker in enumerate(self.marker_config):
                if np.isfinite(side_pixels[index]).all():
                    center = tuple(int(round(v)) for v in side_pixels[index])
                    cv2.circle(side_overlay, center, 9, tuple(marker["display_bgr"]), 2, cv2.LINE_AA)
            cv2.putText(side_overlay, "side calibration required - D435 fallback", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 180, 255), 2, cv2.LINE_AA)
            self.latest_side_overlay = side_overlay
            return KeypointObservation(
                raw_camera_m=internal.raw_camera_m,
                filtered_camera_m=internal.filtered_camera_m,
                base_m=internal.base_m,
                covariance_m2=internal.covariance_m2,
                valid=internal.valid,
                predicted=internal.predicted,
                confidence=internal.confidence,
                color_pixels=internal.color_pixels,
                left_pixels=internal.left_pixels,
                right_pixels=internal.right_pixels,
                source=tuple(f"{value}|side_uncalibrated" for value in internal.source),
                color_confidence=internal.color_confidence,
                side_pixels=side_pixels,
                fusion_mode=DUAL_VIEW_MODE,
                side_time_offset_ms=float(time_offset_ms),
            )

        transform_base_from_color = self.transform_base_from_d435 @ frame.transform_left_from_color
        raw_left = np.full((KEYPOINT_COUNT, 3), np.nan)
        filtered_left = np.full_like(raw_left, np.nan)
        base = np.full_like(raw_left, np.nan)
        covariance_left = np.full((KEYPOINT_COUNT, 3, 3), np.nan)
        valid = np.zeros(KEYPOINT_COUNT, dtype=bool)
        predicted = np.zeros(KEYPOINT_COUNT, dtype=bool)
        confidence = np.zeros(KEYPOINT_COUNT, dtype=float)
        reprojection_errors = np.full(KEYPOINT_COUNT, np.nan)
        sources: list[str] = []
        rotation_base_from_left = self.transform_base_from_d435[:3, :3]
        left_from_base = np.linalg.inv(self.transform_base_from_d435)

        max_side_error = float(self.fusion_config["maximum_side_reprojection_error_px"])
        max_tri_error = float(self.fusion_config["maximum_triangulation_reprojection_error_px"])
        max_disagreement = float(self.fusion_config["maximum_position_disagreement_m"])
        min_angle = float(self.fusion_config["minimum_ray_angle_deg"])
        pixel_std = float(self.fusion_config["side_pixel_std_px"])

        for index, marker in enumerate(self.marker_config):
            internal_ok = bool(internal.valid[index] and np.isfinite(internal.raw_camera_m[index]).all())
            internal_base = _transform(self.transform_base_from_d435, internal.raw_camera_m[index]) if internal_ok else np.full(3, np.nan)
            internal_cov_base = (
                rotation_base_from_left @ internal.covariance_m2[index] @ rotation_base_from_left.T
                if internal_ok else np.full((3, 3), np.nan)
            )
            side_error = math.nan
            projected_internal_side = np.full(2, np.nan)
            if internal_ok and np.isfinite(side_pixels[index]).all():
                projected_internal_side = _project_base(
                    internal_base, self.transform_base_from_side, side_frame.intrinsics
                )
                if np.isfinite(projected_internal_side).all():
                    side_error = float(np.linalg.norm(projected_internal_side - side_pixels[index]))
            reprojection_errors[index] = side_error

            tri_base, tri_error, ray_angle, projected_tri_side = self._triangulate(
                internal.color_pixels[index],
                side_pixels[index],
                frame.color_intrinsics,
                side_frame.intrinsics,
                transform_base_from_color,
            )
            tri_ok = bool(
                np.isfinite(tri_base).all()
                and np.isfinite(tri_error)
                and tri_error <= max_tri_error
                and ray_angle >= min_angle
            )
            if tri_ok:
                range_m = 0.5 * (
                    np.linalg.norm(tri_base - transform_base_from_color[:3, 3])
                    + np.linalg.norm(tri_base - self.transform_base_from_side[:3, 3])
                )
                focal = max(min(float(frame.color_intrinsics["fx"]), float(side_frame.intrinsics["fx"])), 1.0)
                sigma = max(0.00025, range_m * pixel_std / (focal * max(math.sin(math.radians(ray_angle)), 0.05)))
                tri_cov_base = np.eye(3) * sigma**2
            else:
                tri_cov_base = np.full((3, 3), np.nan)

            if internal_ok and tri_ok:
                disagreement = float(np.linalg.norm(internal_base - tri_base))
                verified = np.isfinite(side_error) and side_error <= max_side_error and disagreement <= max_disagreement
                if verified:
                    chosen_base, chosen_cov_base = self._precision_fuse(
                        internal_base, internal_cov_base, tri_base, tri_cov_base
                    )
                    source = "dual_fused"
                    point_confidence = float(np.clip(0.5 * internal.confidence[index] + 0.5 * side_confidence[index], 0.0, 1.0))
                elif bool(self.fusion_config.get("prefer_triangulation_when_depth_disagrees", True)):
                    chosen_base, chosen_cov_base = tri_base, tri_cov_base
                    source = "dual_triangulated_depth_rejected"
                    point_confidence = float(np.clip(side_confidence[index] * 0.8, 0.0, 1.0))
                else:
                    chosen_base, chosen_cov_base = internal_base, internal_cov_base
                    source = "d435_internal_side_disagrees"
                    point_confidence = float(internal.confidence[index] * 0.55)
            elif tri_ok:
                chosen_base, chosen_cov_base = tri_base, tri_cov_base
                source = "dual_triangulated"
                point_confidence = float(np.clip(side_confidence[index] * 0.8, 0.0, 1.0))
            elif internal_ok:
                chosen_base, chosen_cov_base = internal_base, internal_cov_base
                source = "d435_fallback"
                point_confidence = float(internal.confidence[index] * (0.75 if np.isfinite(side_pixels[index]).all() else 0.9))
            else:
                chosen_base = np.full(3, np.nan)
                chosen_cov_base = np.eye(3) * 1e-4
                source = "invalid"
                point_confidence = 0.0

            is_valid = bool(np.isfinite(chosen_base).all() and point_confidence > 0.02)
            chosen_left = _transform(left_from_base, chosen_base) if is_valid else np.full(3, np.nan)
            chosen_cov_left = (
                left_from_base[:3, :3] @ chosen_cov_base @ left_from_base[:3, :3].T
                if is_valid else np.full((3, 3), np.nan)
            )
            filter_point, was_predicted = self.filters[index].update(
                chosen_left if is_valid else None,
                chosen_cov_left if is_valid else np.eye(3) * 1e-4,
                frame.capture_host_ns,
            )
            raw_left[index] = chosen_left
            filtered_left[index] = filter_point
            covariance_left[index] = chosen_cov_left
            valid[index] = is_valid
            predicted[index] = bool(was_predicted and np.isfinite(filter_point).all())
            confidence[index] = point_confidence
            if np.isfinite(filter_point).all():
                base[index] = _transform(self.transform_base_from_d435, filter_point)
            sources.append(source)

            if np.isfinite(side_pixels[index]).all():
                center = tuple(int(round(v)) for v in side_pixels[index])
                colour = tuple(int(value) for value in marker["display_bgr"])
                cv2.circle(side_overlay, center, 9, colour, 2, cv2.LINE_AA)
                cv2.putText(side_overlay, f"K{index} {source}", (center[0] + 10, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
                projection = projected_internal_side if np.isfinite(projected_internal_side).all() else projected_tri_side
                if np.isfinite(projection).all():
                    projected_center = tuple(int(round(v)) for v in projection)
                    cv2.drawMarker(side_overlay, projected_center, (255, 255, 255), cv2.MARKER_CROSS, 9, 1)
                    cv2.line(side_overlay, center, projected_center, (80, 80, 255), 1, cv2.LINE_AA)

        cv2.putText(
            side_overlay,
            f"dual view | dt {time_offset_ms:+.1f} ms | valid {np.count_nonzero(valid)}/7",
            (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA,
        )
        self.latest_side_overlay = side_overlay
        return KeypointObservation(
            raw_camera_m=raw_left,
            filtered_camera_m=filtered_left,
            base_m=base,
            covariance_m2=covariance_left,
            valid=valid,
            predicted=predicted,
            confidence=confidence,
            color_pixels=internal.color_pixels,
            left_pixels=internal.left_pixels,
            right_pixels=internal.right_pixels,
            source=tuple(sources),
            color_confidence=internal.color_confidence,
            side_pixels=side_pixels,
            side_reprojection_error_px=reprojection_errors,
            fusion_mode=DUAL_VIEW_MODE,
            side_time_offset_ms=float(time_offset_ms),
        )
