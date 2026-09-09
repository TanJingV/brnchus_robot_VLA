"""Multi-frame ChArUco camera-to-base calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .models import CameraFrame, RgbCameraFrame


def _camera_matrix(intrinsics: dict[str, Any]) -> np.ndarray:
    return np.asarray([
        [intrinsics["fx"], 0.0, intrinsics["ppx"]],
        [0.0, intrinsics["fy"], intrinsics["ppy"]],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def _pose_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float))[0]
    transform[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return transform


class CharucoBaseCalibrator:
    def __init__(self, config: dict) -> None:
        self.config = config
        board_cfg = config["charuco"]
        dictionary_id = getattr(cv2.aruco, board_cfg["dictionary"])
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard(
            (int(board_cfg["squares_x"]), int(board_cfg["squares_y"])),
            float(board_cfg["square_length_m"]),
            float(board_cfg["marker_length_m"]),
            self.dictionary,
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.transforms_base_from_left: list[np.ndarray] = []
        self.reprojection_errors_px: list[float] = []
        self.last_overlay: np.ndarray | None = None

    def reset(self) -> None:
        self.transforms_base_from_left.clear()
        self.reprojection_errors_px.clear()

    def add_frame(self, frame: CameraFrame) -> tuple[bool, str, np.ndarray]:
        return self._add_image(
            frame.color_bgr,
            frame.color_intrinsics,
            frame.transform_left_from_color,
        )

    def add_rgb_frame(self, frame: RgbCameraFrame) -> tuple[bool, str, np.ndarray]:
        return self._add_image(frame.image_bgr, frame.intrinsics, np.eye(4))

    def _add_image(
        self,
        image_bgr: np.ndarray,
        intrinsics: dict[str, Any],
        transform_camera_from_image: np.ndarray,
    ) -> tuple[bool, str, np.ndarray]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
        overlay = image_bgr.copy()
        if marker_ids is not None and len(marker_ids):
            cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
        if charuco_ids is None or len(charuco_ids) < 6:
            self.last_overlay = overlay
            return False, "ChArUco角点不足（至少需要6个）", overlay
        cv2.aruco.drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids)
        object_points, image_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
        object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        camera_matrix = _camera_matrix(intrinsics)
        distortion = np.asarray(intrinsics.get("coeffs", [0.0] * 5), dtype=float)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            self.last_overlay = overlay
            return False, "PnP求解失败", overlay
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        reprojection = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        rmse = float(np.sqrt(np.mean(reprojection**2)))
        maximum = float(self.config["maximum_reprojection_error_px"])
        if rmse > maximum:
            self.last_overlay = overlay
            return False, f"重投影RMSE {rmse:.3f}px 超过阈值 {maximum:.3f}px", overlay

        transform_color_from_board = _pose_matrix(rvec, tvec)
        transform_left_from_board = np.asarray(transform_camera_from_image, dtype=float) @ transform_color_from_board
        transform_base_from_board = np.asarray(self.config["transform_base_from_board"], dtype=float)
        transform_base_from_left = transform_base_from_board @ np.linalg.inv(transform_left_from_board)
        self.transforms_base_from_left.append(transform_base_from_left)
        self.reprojection_errors_px.append(rmse)
        cv2.drawFrameAxes(overlay, camera_matrix, distortion, rvec, tvec, 0.025)
        cv2.putText(overlay, f"accepted {len(self.transforms_base_from_left)} | RMSE {rmse:.3f}px", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 230, 40), 2, cv2.LINE_AA)
        self.last_overlay = overlay
        return True, f"已采集 {len(self.transforms_base_from_left)} 帧，RMSE {rmse:.3f}px", overlay

    def solve(self) -> dict[str, Any]:
        minimum = int(self.config["minimum_frames"])
        if len(self.transforms_base_from_left) < minimum:
            raise RuntimeError(f"标定帧不足：{len(self.transforms_base_from_left)}/{minimum}")
        transforms = np.asarray(self.transforms_base_from_left)
        translations = transforms[:, :3, 3]
        median_translation = np.median(translations, axis=0)
        distances = np.linalg.norm(translations - median_translation, axis=1)
        mad = float(np.median(np.abs(distances - np.median(distances))))
        threshold = max(0.0015, np.median(distances) + 3.5 * 1.4826 * mad)
        keep = distances <= threshold
        if np.count_nonzero(keep) < minimum:
            keep[:] = True
        rotations = Rotation.from_matrix(transforms[keep, :3, :3])
        mean_rotation = rotations.mean().as_matrix()
        result = np.eye(4, dtype=float)
        result[:3, :3] = mean_rotation
        result[:3, 3] = np.median(translations[keep], axis=0)
        translation_residual_mm = np.linalg.norm(translations[keep] - result[:3, 3], axis=1) * 1000.0
        rotation_residual_deg = np.degrees((Rotation.from_matrix(result[:3, :3]).inv() * rotations).magnitude())
        return {
            "ready": True,
            "transform_base_from_camera": result.tolist(),
            "accepted_frames": int(np.count_nonzero(keep)),
            "captured_frames": len(transforms),
            "mean_reprojection_error_px": float(np.mean(np.asarray(self.reprojection_errors_px)[keep])),
            "translation_repeatability_median_mm": float(np.median(translation_residual_mm)),
            "translation_repeatability_p95_mm": float(np.percentile(translation_residual_mm, 95)),
            "rotation_repeatability_median_deg": float(np.median(rotation_residual_deg)),
        }


class CharucoIntrinsicCalibrator:
    """Multi-pose intrinsic calibration for the ordinary RGB camera."""

    def __init__(self, config: dict) -> None:
        self.config = config
        board_cfg = config["charuco"]
        dictionary_id = getattr(cv2.aruco, board_cfg["dictionary"])
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard(
            (int(board_cfg["squares_x"]), int(board_cfg["squares_y"])),
            float(board_cfg["square_length_m"]),
            float(board_cfg["marker_length_m"]),
            dictionary,
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.object_points: list[np.ndarray] = []
        self.image_points: list[np.ndarray] = []
        self.image_size: tuple[int, int] | None = None
        self.last_overlay: np.ndarray | None = None

    def reset(self) -> None:
        self.object_points.clear()
        self.image_points.clear()
        self.image_size = None

    def add_frame(self, frame: RgbCameraFrame) -> tuple[bool, str, np.ndarray]:
        image = frame.image_bgr
        height, width = image.shape[:2]
        if self.image_size is not None and self.image_size != (width, height):
            return False, "侧相机分辨率发生变化，请重置内参标定", image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
        overlay = image.copy()
        if marker_ids is not None and len(marker_ids):
            cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
        if ids is None or len(ids) < 8:
            self.last_overlay = overlay
            return False, "侧相机内参标定需要至少8个ChArUco角点", overlay
        cv2.aruco.drawDetectedCornersCharuco(overlay, corners, ids)
        object_points, image_points = self.board.matchImagePoints(corners, ids)
        self.object_points.append(np.asarray(object_points, np.float32).reshape(-1, 3))
        self.image_points.append(np.asarray(image_points, np.float32).reshape(-1, 2))
        self.image_size = (width, height)
        cv2.putText(
            overlay, f"intrinsic frames {len(self.object_points)}", (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 230, 40), 2, cv2.LINE_AA,
        )
        self.last_overlay = overlay
        return True, f"侧相机内参已采集 {len(self.object_points)} 帧", overlay

    def solve(self) -> dict[str, Any]:
        minimum = max(10, int(self.config.get("minimum_frames", 15)))
        if len(self.object_points) < minimum or self.image_size is None:
            raise RuntimeError(f"侧相机内参标定帧不足：{len(self.object_points)}/{minimum}")
        rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.image_size,
            None,
            None,
            flags=cv2.CALIB_FIX_K3,
        )
        errors = []
        for object_points, image_points, rvec, tvec in zip(
            self.object_points, self.image_points, rvecs, tvecs
        ):
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
            error = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
            errors.append(float(np.sqrt(np.mean(error**2))))
        width, height = self.image_size
        intrinsics = {
            "width": int(width),
            "height": int(height),
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "ppx": float(camera_matrix[0, 2]),
            "ppy": float(camera_matrix[1, 2]),
            "coeffs": [float(value) for value in distortion.reshape(-1)[:5]],
            "model": "opencv",
        }
        return {
            "intrinsics_ready": True,
            "intrinsics": intrinsics,
            "captured_frames": len(self.object_points),
            "calibration_rms_px": float(rms),
            "mean_reprojection_error_px": float(np.mean(errors)),
            "p95_reprojection_error_px": float(np.percentile(errors, 95)),
        }


def save_calibration(result: dict[str, Any], path: str | Path) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    return target


def load_calibration(path: str | Path) -> dict[str, Any]:
    with Path(path).resolve().open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    transform = np.asarray(result["transform_base_from_camera"], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("标定文件中的变换矩阵无效")
    return result


def load_intrinsic_calibration(path: str | Path) -> dict[str, Any]:
    with Path(path).resolve().open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    intrinsics = result.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError("标定文件中缺少intrinsics")
    for key in ("width", "height", "fx", "fy", "ppx", "ppy"):
        if key not in intrinsics or not np.isfinite(float(intrinsics[key])):
            raise ValueError(f"侧相机内参字段无效: {key}")
    return result
