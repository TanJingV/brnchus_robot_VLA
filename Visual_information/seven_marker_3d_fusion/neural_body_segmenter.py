"""SAM2.1 video-memory tracking for the physical TDCR terminal section.

The target is deliberately *not* discovered again on every frame.  The user
locks one RGB rectangle around the black terminal section and its seven rings.
That rectangle becomes object 1 in a streaming SAM2 video session.  Subsequent
frames are propagated by SAM2's memory and checked against an optical-flow
prediction, area continuity and the coarse black-body centreline.

This design has an important safety property: a lost target stays lost.  It is
never silently replaced by a lung reflection, transparent airway edge or drive
rail just because another object also looks like a thin dark line.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class NeuralBodyMaskResult:
    accepted: bool
    mask: np.ndarray
    confidence: float
    temporal_iou: float
    backend: str
    reason: str


def _empty(
    shape: tuple[int, int], reason: str, backend: str = "sam2.1-video"
) -> NeuralBodyMaskResult:
    return NeuralBodyMaskResult(
        accepted=False,
        mask=np.zeros(shape, dtype=np.uint8),
        confidence=0.0,
        temporal_iou=0.0,
        backend=backend,
        reason=reason,
    )


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b)) / max(union, 1)


def _sample_mask(mask: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.empty(0, dtype=bool)
    height, width = mask.shape
    x = np.clip(np.rint(points_xy[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(points_xy[:, 1]).astype(int), 0, height - 1)
    return mask[y, x] != 0


def _select_component(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Keep the component containing most reference points."""

    binary = np.asarray(mask != 0, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary * 255
    height, width = binary.shape
    points = np.asarray(reference, dtype=float).reshape(-1, 2)
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
    reference_labels = labels[y, x]
    best_label, best_score = 0, -1.0
    for label in range(1, count):
        hits = int(np.count_nonzero(reference_labels == label))
        area = int(stats[label, cv2.CC_STAT_AREA])
        score = 100000.0 * hits + math.sqrt(max(area, 1))
        if score > best_score:
            best_label, best_score = label, score
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


class NeuralContinuumSegmenter:
    """Track exactly one manually locked TDCR instance with SAM2.1 memory."""

    OBJECT_ID = 1

    def __init__(self, tracking_config: dict | None = None) -> None:
        cfg = dict(tracking_config or {})
        self.enabled = bool(cfg.get("body_neural_segmentation_enabled", True))
        self.required = bool(cfg.get("body_neural_segmentation_required", True))
        default_model = Path(__file__).resolve().parents[1] / "models" / "sam2.1-hiera-tiny"
        override = os.environ.get("TDCR_SAM_MODEL", "").strip()
        configured_model = Path(
            override or cfg.get("body_neural_model_path", str(default_model))
        ).expanduser()
        # Codex uses a lightweight worktree while the 156 MB SAM2 weights are
        # stored once in the user's real laboratory project.  Resolve both
        # locations so launching the same UI from either folder cannot
        # silently disable neural tracking and fall back to reflection-prone
        # foreground subtraction.
        candidates = [configured_model, default_model]
        laboratory_root = os.environ.get("BRNCHUS_ROBOT_VLA_ROOT", "").strip()
        if laboratory_root:
            candidates.append(
                Path(laboratory_root)
                / "Visual_information"
                / "models"
                / "sam2.1-hiera-tiny"
            )
        candidates.append(
            Path(r"E:\PHD\Laboratory\feibu\brnchus_robot_VLA")
            / "Visual_information"
            / "models"
            / "sam2.1-hiera-tiny"
        )
        # One-release migration fallback: permits verification before the
        # laboratory checkout has been reorganised, then becomes unused.
        candidates.append(
            Path(r"E:\PHD\Laboratory\feibu\brnchus_robot_VLA")
            / "models"
            / "sam2.1-hiera-tiny"
        )
        existing = next(
            (
                path.expanduser().resolve()
                for path in candidates
                if (path.expanduser() / "model.safetensors").is_file()
            ),
            configured_model.expanduser().resolve(),
        )
        self.model_path = existing
        self.requested_device = str(cfg.get("body_neural_device", "cuda")).lower()
        self.minimum_confidence = float(cfg.get("body_neural_minimum_confidence", 0.50))
        self.minimum_prompt_coverage = float(
            cfg.get("body_neural_minimum_prompt_coverage", 0.72)
        )
        self.minimum_temporal_iou = float(
            cfg.get("body_neural_minimum_temporal_iou", 0.18)
        )
        self.maximum_area_ratio = float(cfg.get("body_neural_maximum_area_ratio", 2.4))
        self.minimum_area_ratio = float(cfg.get("body_neural_minimum_area_ratio", 0.38))
        self.maximum_propagation_frames = max(
            0, int(cfg.get("body_neural_maximum_propagation_frames", 2))
        )
        self.inference_stride = max(
            1, int(cfg.get("body_neural_inference_stride", 3))
        )
        self.tracking_margin = float(cfg.get("body_target_tracking_margin", 0.65))

        self._model = None
        self._processor = None
        self._torch = None
        self._device = "cpu"
        self._dtype = None
        self._load_error = ""
        self._load_lock = threading.Lock()
        self._session = None
        self._session_shape: tuple[int, int] | None = None
        self._frame_index = 0
        self._physical_frame_index = 0
        self._target_lock_roi: tuple[float, float, float, float] | None = None
        configured_lock = cfg.get("body_target_lock_roi")
        if configured_lock is not None:
            self._target_lock_roi = self._validate_roi(configured_lock)
        self._last_mask: np.ndarray | None = None
        self._last_gray: np.ndarray | None = None
        self._lost_frames = 0
        self.last_backend = "disabled" if not self.enabled else "sam2.1-video"
        self.last_confidence = 0.0
        self.last_reason = "target not locked"

    @staticmethod
    def _validate_roi(values) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = map(float, values)
        x0, x1 = sorted((float(np.clip(x0, 0.0, 1.0)), float(np.clip(x1, 0.0, 1.0))))
        y0, y1 = sorted((float(np.clip(y0, 0.0, 1.0)), float(np.clip(y1, 0.0, 1.0))))
        if x1 - x0 < 0.02 or y1 - y0 < 0.015:
            raise ValueError("target lock rectangle is too small")
        return x0, y0, x1, y1

    @property
    def load_error(self) -> str:
        return self._load_error

    @property
    def model_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def target_locked(self) -> bool:
        return self._target_lock_roi is not None

    @property
    def target_lock_roi(self) -> tuple[float, float, float, float] | None:
        return self._target_lock_roi

    def set_target_lock(self, roi_normalized) -> None:
        self._target_lock_roi = self._validate_roi(roi_normalized)
        self.reset(preserve_lock=True)
        self.last_reason = "target locked; waiting for conditioning frame"

    def clear_target_lock(self) -> None:
        self._target_lock_roi = None
        self.reset(preserve_lock=False)
        self.last_reason = "target not locked"

    def reset(self, preserve_lock: bool = True) -> None:
        if not preserve_lock:
            self._target_lock_roi = None
        self._session = None
        self._session_shape = None
        self._frame_index = 0
        self._physical_frame_index = 0
        self._last_mask = None
        self._last_gray = None
        self._lost_frames = 0
        self.last_confidence = 0.0
        self.last_reason = "reset"

    def _load(self) -> bool:
        if not self.enabled:
            self._load_error = "neural segmentation disabled"
            return False
        if self.model_loaded:
            return True
        with self._load_lock:
            if self.model_loaded:
                return True
            if not (self.model_path / "model.safetensors").exists():
                self._load_error = f"SAM2.1 weights missing: {self.model_path}"
                return False
            try:
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                import torch
                from transformers import Sam2VideoModel, Sam2VideoProcessor
                from transformers.utils import logging as transformers_logging

                transformers_logging.set_verbosity_error()
                if self.requested_device == "cuda" and torch.cuda.is_available():
                    self._device = "cuda"
                    self._dtype = torch.float16
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                else:
                    self._device = "cpu"
                    self._dtype = torch.float32
                self._processor = Sam2VideoProcessor.from_pretrained(
                    self.model_path, local_files_only=True
                )
                self._model = Sam2VideoModel.from_pretrained(
                    self.model_path, local_files_only=True, dtype=self._dtype
                ).to(self._device).eval()
                self._torch = torch
                self.last_backend = f"sam2.1-video-{self._device}"
                self._load_error = ""
                return True
            except Exception as exc:
                self._model = None
                self._processor = None
                self._torch = None
                self._load_error = f"{type(exc).__name__}: {exc}"
                return False

    @staticmethod
    def _roi_pixels(
        shape: tuple[int, int], roi: tuple[float, float, float, float]
    ) -> tuple[int, int, int, int]:
        height, width = shape
        x0, y0, x1, y1 = roi
        left = int(np.clip(round(x0 * width), 0, width - 2))
        top = int(np.clip(round(y0 * height), 0, height - 2))
        right = int(np.clip(round(x1 * width), left + 2, width))
        bottom = int(np.clip(round(y1 * height), top + 2, height))
        return left, top, right, bottom

    def tracking_roi_normalized(
        self,
        shape: tuple[int, int],
        fallback: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Return the moving search box without changing target identity."""

        if self._last_mask is None or not np.any(self._last_mask):
            return self._target_lock_roi or fallback
        # The tracker runs on the detector's reduced RGB image.  Callers such
        # as the full-resolution overlay may pass a different shape; bounding
        # box pixels must always be normalised by the mask's own dimensions.
        height, width = self._last_mask.shape
        x, y, box_width, box_height = cv2.boundingRect(self._last_mask)
        margin_x = max(30, int(round(self.tracking_margin * box_width)))
        margin_y = max(24, int(round(1.4 * self.tracking_margin * box_height)))
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(width, x + box_width + margin_x)
        bottom = min(height, y + box_height + margin_y)
        return left / width, top / height, right / width, bottom / height

    @staticmethod
    def _flow_warp(
        previous_gray: np.ndarray | None,
        current_gray: np.ndarray,
        previous_mask: np.ndarray | None,
    ) -> np.ndarray | None:
        if (
            previous_gray is None
            or previous_mask is None
            or previous_gray.shape != current_gray.shape
            or not np.any(previous_mask)
        ):
            return previous_mask
        features = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=120,
            qualityLevel=0.01,
            minDistance=5,
            mask=cv2.dilate(previous_mask, np.ones((9, 9), dtype=np.uint8)),
            blockSize=5,
        )
        if features is None or len(features) < 6:
            return previous_mask.copy()
        flowed, status, error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            features,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
        )
        if flowed is None or status is None:
            return previous_mask.copy()
        source = features.reshape(-1, 2)
        target = flowed.reshape(-1, 2)
        valid = status.reshape(-1).astype(bool) & np.isfinite(target).all(axis=1)
        if error is not None:
            valid &= error.reshape(-1) < 35.0
        if np.count_nonzero(valid) < 5:
            return previous_mask.copy()
        transform, _ = cv2.estimateAffinePartial2D(
            source[valid], target[valid], method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if transform is None:
            displacement = np.median(target[valid] - source[valid], axis=0)
            transform = np.asarray(
                [[1.0, 0.0, displacement[0]], [0.0, 1.0, displacement[1]]], dtype=float
            )
        return cv2.warpAffine(
            previous_mask,
            transform,
            current_gray.shape[::-1],
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )

    @staticmethod
    def _prompt_points(path: np.ndarray, radius: float) -> tuple[list, list]:
        indices = np.unique(np.rint(np.linspace(0, len(path) - 1, 9)).astype(int))
        positive = path[indices].astype(float)
        negative: list[np.ndarray] = []
        offset = max(8.0, 3.0 * radius)
        for index in np.unique(np.rint(np.linspace(0, len(path) - 1, 4)).astype(int)):
            before = path[max(0, index - 4)]
            after = path[min(len(path) - 1, index + 4)]
            tangent = after - before
            tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
            normal = np.asarray([-tangent[1], tangent[0]])
            negative.extend((path[index] + offset * normal, path[index] - offset * normal))
        points = np.vstack((positive, np.asarray(negative))).astype(float).tolist()
        labels = ([1] * len(positive)) + ([0] * len(negative))
        return points, labels

    @staticmethod
    def _corridor(path: np.ndarray, radius: float, shape: tuple[int, int]) -> np.ndarray:
        corridor = np.zeros(shape, dtype=np.uint8)
        thickness = max(9, int(round(5.0 * radius)) | 1)
        cv2.polylines(
            corridor,
            [np.rint(path).astype(np.int32)],
            False,
            255,
            thickness,
            cv2.LINE_8,
        )
        return corridor

    def _initialise_session(
        self,
        shape: tuple[int, int],
        path: np.ndarray,
        radius: float,
    ) -> None:
        self._session = self._processor.init_video_session(
            video=None,
            inference_device=self._device,
            inference_state_device=self._device,
            processing_device=self._device,
            video_storage_device=self._device,
            dtype=self._dtype,
        )
        self._session_shape = shape
        self._frame_index = 0
        points, labels = self._prompt_points(path, radius)
        left, top, right, bottom = self._roi_pixels(shape, self._target_lock_roi)
        self._processor.add_inputs_to_inference_session(
            self._session,
            frame_idx=0,
            obj_ids=self.OBJECT_ID,
            input_points=[[points]],
            input_labels=[[labels]],
            input_boxes=[[[float(left), float(top), float(right), float(bottom)]]],
            original_size=shape,
        )

    def _preprocess_frame(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        processed = self._processor.video_processor(
            videos=[[rgb]], return_tensors="pt"
        ).pixel_values_videos[0, 0]
        return processed.to(self._device, dtype=self._dtype, non_blocking=True)

    def _propagate_or_lose(
        self,
        predicted: np.ndarray | None,
        path: np.ndarray,
        shape: tuple[int, int],
        reason: str,
    ) -> NeuralBodyMaskResult:
        self._lost_frames += 1
        if predicted is not None and self._lost_frames <= self.maximum_propagation_frames:
            coverage = float(np.mean(_sample_mask(predicted, path)))
            if coverage >= 0.45:
                self._last_mask = predicted.copy()
                confidence = max(0.20, self.last_confidence * 0.62)
                self.last_backend = "sam2.1-video-flow"
                self.last_confidence = confidence
                self.last_reason = f"temporary optical-flow propagation: {reason}"
                return NeuralBodyMaskResult(
                    True, predicted.copy(), confidence, 1.0, self.last_backend, self.last_reason
                )
        self.last_confidence = 0.0
        self.last_backend = "sam2.1-video-lost"
        self.last_reason = f"TARGET LOST: {reason}; relock required"
        return _empty(shape, self.last_reason, self.last_backend)

    def segment(
        self,
        image_bgr: np.ndarray,
        active_path_xy: np.ndarray,
        estimated_radius_px: float,
        roi: tuple[int, int, int, int] | None = None,
    ) -> NeuralBodyMaskResult:
        del roi  # the moving ROI is a detector optimisation, not a new SAM prompt
        image = np.asarray(image_bgr, dtype=np.uint8)
        height, width = image.shape[:2]
        path = np.asarray(active_path_xy, dtype=np.float32).reshape(-1, 2)
        if not self.target_locked:
            return _empty((height, width), "TARGET UNLOCKED: drag a box and click lock")
        if len(path) < 12 or not np.isfinite(path).all():
            return _empty((height, width), "locked target centreline unavailable")
        if not self._load():
            return _empty((height, width), self._load_error or "SAM2.1 unavailable")

        radius = float(np.clip(estimated_radius_px, 2.0, 18.0))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flow_prediction = self._flow_warp(self._last_gray, gray, self._last_mask)
        is_conditioning_frame = self._last_mask is None
        physical_frame_index = self._physical_frame_index
        self._physical_frame_index += 1
        if (
            not is_conditioning_frame
            and physical_frame_index % self.inference_stride != 0
            and flow_prediction is not None
            and np.any(flow_prediction)
        ):
            coverage = float(np.mean(_sample_mask(flow_prediction, path)))
            if coverage >= 0.40:
                self._last_mask = flow_prediction.copy()
                self._last_gray = gray.copy()
                self.last_backend = f"sam2.1-video-flow-{self._device}"
                self.last_confidence = max(0.35, self.last_confidence * 0.995)
                self.last_reason = (
                    f"30 Hz optical-flow propagation between SAM2 keyframes; "
                    f"coverage={coverage:.3f} stride={self.inference_stride}"
                )
                return NeuralBodyMaskResult(
                    True,
                    flow_prediction.copy(),
                    self.last_confidence,
                    1.0,
                    self.last_backend,
                    self.last_reason,
                )
        if self._session is None or self._session_shape != (height, width):
            self._initialise_session((height, width), path, radius)

        frame_index = self._frame_index
        try:
            frame_tensor = self._preprocess_frame(image)
            with self._torch.inference_mode():
                if self._device == "cuda":
                    with self._torch.autocast("cuda", dtype=self._dtype):
                        outputs = self._model(
                            self._session, frame=frame_tensor, frame_idx=frame_index
                        )
                else:
                    outputs = self._model(
                        self._session, frame=frame_tensor, frame_idx=frame_index
                    )
            # Sam2VideoModel returns one 4-D tensor [objects, channels, 256,
            # 256], whereas the image processor's helper expects a nested
            # image batch.  Upscale the logits directly to avoid accidentally
            # dropping the channel dimension in that helper.
            processed = self._torch.nn.functional.interpolate(
                outputs.pred_masks.detach().float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ) > 0.0
            mask_array = processed.cpu().numpy()
            while mask_array.ndim > 2:
                mask_array = mask_array[0]
            candidate = np.where(mask_array, 255, 0).astype(np.uint8)
            logits = outputs.object_score_logits.detach().float().cpu().numpy().reshape(-1)
            presence = float(1.0 / (1.0 + np.exp(-float(logits[0]))))
        except Exception as exc:
            self._last_gray = gray.copy()
            result = self._propagate_or_lose(
                flow_prediction,
                path,
                (height, width),
                f"SAM2 video inference failed: {type(exc).__name__}: {exc}",
            )
            if result.mask.shape != (height, width):
                return _empty((height, width), result.reason, result.backend)
            return result
        finally:
            self._frame_index = frame_index + 1

        reference = path if flow_prediction is None else np.column_stack(
            np.nonzero(flow_prediction)[::-1]
        )[::max(1, int(np.count_nonzero(flow_prediction) / 200))]
        candidate = _select_component(candidate, reference)
        corridor = self._corridor(path, radius, (height, width))
        # SAM establishes identity; the material corridor removes transparent
        # lumen and rail pixels while retaining coloured rings on that identity.
        candidate = cv2.bitwise_and(candidate, corridor)
        if is_conditioning_frame:
            # The manual rectangle is the authoritative spatial support on the
            # conditioning frame.  Reflections immediately above/below it may
            # supply useful context to SAM, but cannot become target pixels.
            left, top, right, bottom = self._roi_pixels(
                (height, width), self._target_lock_roi
            )
            margin = max(4, int(round(2.0 * radius)))
            lock_support = np.zeros((height, width), dtype=np.uint8)
            lock_support[
                max(0, top - margin):min(height, bottom + margin),
                max(0, left - margin):min(width, right + margin),
            ] = 255
            candidate = cv2.bitwise_and(candidate, lock_support)
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        candidate = _select_component(candidate, path)

        coverage = float(np.mean(_sample_mask(candidate, path)))
        temporal_iou = 0.50 if flow_prediction is None else _mask_iou(candidate, flow_prediction)
        area = float(np.count_nonzero(candidate))
        previous_area = float(np.count_nonzero(flow_prediction)) if flow_prediction is not None else area
        area_ratio = area / max(previous_area, 1.0)
        slender_area = max(1.0, float(len(path)) * 2.0 * radius)
        slender_quality = math.exp(-abs(math.log(max(area / slender_area, 1e-5))) / 1.5)
        confidence = (
            0.32 * presence
            + 0.30 * coverage
            + 0.25 * temporal_iou
            + 0.13 * slender_quality
        )
        diagnostic = (
            f"presence={presence:.3f} coverage={coverage:.3f} temporal={temporal_iou:.3f} "
            f"area_ratio={area_ratio:.3f} slender={slender_quality:.3f}"
        )
        required_coverage = (
            min(self.minimum_prompt_coverage, 0.62)
            if flow_prediction is None
            # Once object 1 has a memory, identity is established primarily
            # by temporal IoU.  Ring highlights create deliberate gaps in the
            # coarse black-line prompt, so demanding 72% prompt coverage here
            # incorrectly declares a stationary, high-IoU target lost.
            else min(self.minimum_prompt_coverage, 0.50)
        )
        invalid = (
            presence < 0.50
            or coverage < required_coverage
            or confidence < self.minimum_confidence
            or (
                flow_prediction is not None
                and (
                    temporal_iou < self.minimum_temporal_iou
                    or area_ratio < self.minimum_area_ratio
                    or area_ratio > self.maximum_area_ratio
                )
            )
        )
        self._last_gray = gray.copy()
        if invalid:
            result = self._propagate_or_lose(
                flow_prediction, path, (height, width), diagnostic
            )
            if result.mask.shape != (height, width):
                return _empty((height, width), result.reason, result.backend)
            return result

        self._last_mask = candidate
        self._lost_frames = 0
        self.last_backend = f"sam2.1-video-{self._device}"
        self.last_confidence = float(np.clip(confidence, 0.0, 1.0))
        self.last_reason = f"locked object {self.OBJECT_ID} accepted ({diagnostic})"
        return NeuralBodyMaskResult(
            True,
            candidate,
            self.last_confidence,
            float(temporal_iou),
            self.last_backend,
            self.last_reason,
        )
