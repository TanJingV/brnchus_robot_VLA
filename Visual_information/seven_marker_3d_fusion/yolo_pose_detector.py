"""Learned TDCR terminal detector based on a custom seven-keypoint YOLO Pose model.

Generic COCO pose weights recognise human joints and cannot recognise K0--K6.
This wrapper therefore accepts only a project-specific model whose pose head
has exactly seven keypoints.  A missing/incompatible model fails closed: no
plausible-looking points are manufactured by colour or spacing rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

from Visual_information.d435_tdcr_capture.models import KEYPOINT_COUNT


VISUAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VISUAL_ROOT.parent
DEFAULT_MODEL = VISUAL_ROOT / "models" / "tdcr_yolo_pose" / "best.pt"


@dataclass(frozen=True)
class YoloPoseObservation:
    accepted: bool
    pixels: np.ndarray
    confidence: np.ndarray
    box_xyxy: np.ndarray
    object_confidence: float
    reason: str


def _empty(reason: str) -> YoloPoseObservation:
    return YoloPoseObservation(
        False,
        np.full((KEYPOINT_COUNT, 2), np.nan),
        np.zeros(KEYPOINT_COUNT),
        np.full(4, np.nan),
        0.0,
        reason,
    )


class TdcrYoloPoseDetector:
    """CUDA YOLO Pose inference with target-aware instance selection."""

    def __init__(self, tracking_config: dict | None = None) -> None:
        cfg = dict(tracking_config or {})
        configured = str(cfg.get("tdcr_yolo_pose_model", "")).strip()
        self.model_path = Path(configured).expanduser() if configured else DEFAULT_MODEL
        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path
        self.model_path = self.model_path.resolve()
        self.enabled = bool(cfg.get("tdcr_yolo_pose_enabled", True))
        self.strict = bool(cfg.get("tdcr_yolo_pose_strict", True))
        self.image_size = int(cfg.get("tdcr_yolo_pose_image_size", 960))
        self.box_threshold = float(cfg.get("tdcr_yolo_pose_box_confidence", 0.28))
        self.keypoint_threshold = float(cfg.get("tdcr_yolo_pose_keypoint_confidence", 0.30))
        self.minimum_keypoints = int(cfg.get("tdcr_yolo_pose_minimum_keypoints", 7))
        self.device = str(cfg.get("tdcr_yolo_pose_device", "0"))
        self.half = bool(cfg.get("tdcr_yolo_pose_half", True))
        self._model: Any = None
        self._load_error = ""
        self._target_roi: tuple[float, float, float, float] | None = None
        self._last_box = np.full(4, np.nan)
        self.last_observation = _empty("YOLO Pose not run")

    @property
    def ready(self) -> bool:
        return self.enabled and self.model_path.is_file() and not self._load_error

    @property
    def status(self) -> str:
        if not self.enabled:
            return "YOLO POSE DISABLED"
        if not self.model_path.is_file():
            return f"YOLO POSE MODEL MISSING: {self.model_path}"
        if self._load_error:
            return f"YOLO POSE LOAD ERROR: {self._load_error}"
        return "YOLO POSE READY" if self._model is not None else "YOLO POSE MODEL FOUND"

    def set_target_lock(self, roi_normalized) -> None:
        values = tuple(map(float, roi_normalized))
        self._target_roi = values  # validated by the existing target-lock UI
        self._last_box[:] = np.nan

    def clear_target_lock(self) -> None:
        self._target_roi = None
        self._last_box[:] = np.nan

    def reset(self) -> None:
        self._last_box[:] = np.nan
        self.last_observation = _empty("YOLO Pose reset")

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if not self.enabled or not self.model_path.is_file():
            return False
        try:
            config_root = VISUAL_ROOT / ".ultralytics"
            config_root.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
            os.environ.setdefault("XDG_CONFIG_HOME", str(config_root))
            os.environ.setdefault("HF_HOME", str(VISUAL_ROOT / ".hf"))
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path), task="pose")
            model_shape = getattr(getattr(self._model, "model", None), "kpt_shape", None)
            if model_shape is not None and int(model_shape[0]) != KEYPOINT_COUNT:
                raise ValueError(
                    f"pose head has {model_shape[0]} keypoints, expected {KEYPOINT_COUNT}"
                )
            self._load_error = ""
            return True
        except Exception as exc:
            self._model = None
            self._load_error = f"{type(exc).__name__}: {exc}"
            return False

    @staticmethod
    def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
        left = max(float(first[0]), float(second[0]))
        top = max(float(first[1]), float(second[1]))
        right = min(float(first[2]), float(second[2]))
        bottom = min(float(first[3]), float(second[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        area0 = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        area1 = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        return intersection / max(area0 + area1 - intersection, 1e-6)

    def detect(self, image_bgr: np.ndarray) -> YoloPoseObservation:
        image = np.asarray(image_bgr, dtype=np.uint8)
        if not self._ensure_model():
            self.last_observation = _empty(self.status)
            return self.last_observation
        try:
            results = self._model.predict(
                source=image,
                imgsz=self.image_size,
                conf=self.box_threshold,
                iou=0.55,
                device=self.device,
                half=self.half,
                max_det=8,
                verbose=False,
            )
            result = results[0]
            if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
                self.last_observation = _empty("YOLO Pose found no TDCR terminal")
                return self.last_observation
            boxes = result.boxes.xyxy.detach().float().cpu().numpy()
            box_confidence = result.boxes.conf.detach().float().cpu().numpy()
            keypoint_data = result.keypoints.data.detach().float().cpu().numpy()
            if keypoint_data.ndim != 3 or keypoint_data.shape[1] != KEYPOINT_COUNT:
                self.last_observation = _empty(
                    f"incompatible YOLO pose output {tuple(keypoint_data.shape)}"
                )
                return self.last_observation
            height, width = image.shape[:2]
            reference = self._last_box.copy()
            if not np.isfinite(reference).all() and self._target_roi is not None:
                x0, y0, x1, y1 = self._target_roi
                reference = np.asarray([x0 * width, y0 * height, x1 * width, y1 * height])
            scores = box_confidence.astype(float).copy()
            if np.isfinite(reference).all():
                scores += 1.5 * np.asarray([self._box_iou(box, reference) for box in boxes])
            selected = int(np.argmax(scores))
            points = keypoint_data[selected, :, :2].astype(float)
            confidence = (
                keypoint_data[selected, :, 2].astype(float)
                if keypoint_data.shape[2] >= 3
                else np.ones(KEYPOINT_COUNT)
            )
            visible = confidence >= self.keypoint_threshold
            if int(np.count_nonzero(visible)) < self.minimum_keypoints:
                self.last_observation = _empty(
                    f"YOLO Pose only confirmed {np.count_nonzero(visible)}/{KEYPOINT_COUNT} keypoints"
                )
                return self.last_observation
            if not np.isfinite(points).all():
                self.last_observation = _empty("YOLO Pose returned non-finite keypoints")
                return self.last_observation
            self._last_box = boxes[selected].copy()
            self.last_observation = YoloPoseObservation(
                True,
                points,
                np.clip(confidence, 0.0, 1.0),
                self._last_box.copy(),
                float(box_confidence[selected]),
                "custom seven-keypoint YOLO Pose",
            )
            return self.last_observation
        except Exception as exc:
            self.last_observation = _empty(f"YOLO Pose inference error: {type(exc).__name__}: {exc}")
            return self.last_observation
