"""End-to-end seven-marker reconstruction and diagnostic overlay."""

from __future__ import annotations

import time
from dataclasses import replace

import cv2
import numpy as np

from Visual_information.d435_tdcr_capture.models import KEYPOINT_COUNT, CameraFrame, empty_keypoint_observation
from Visual_information.d435_tdcr_capture.tracking import ContinuumSegmentation, MarkerTracker

from .body_first_detector import BodyFirstBandDetector
from .cotracker3_tracker import TrackedPointRefiner
from .material_chain_tracker import MaterialColorChainTracker
from .measurement import MultimodalMeasurementFuser
from .models import SevenMarkerResult
from .smoother import ShapeConstrainedSmoother
from .yolo_pose_detector import TdcrYoloPoseDetector


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    result = np.full_like(points, np.nan, dtype=float)
    valid = np.isfinite(points).all(axis=1)
    if np.any(valid):
        matrix = np.asarray(transform, dtype=float)
        result[valid] = points[valid] @ matrix[:3, :3].T + matrix[:3, 3]
    return result


class SevenMarkerFusionPipeline:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.tracker = MarkerTracker(config)
        self.body_detector = BodyFirstBandDetector(config.get("tracking", {}))
        self.yolo_pose = TdcrYoloPoseDetector(config.get("tracking", {}))
        self.point_refiner = TrackedPointRefiner(config.get("tracking", {}))
        self.material_tracker = MaterialColorChainTracker(config.get("tracking", {}))
        self.measurements = MultimodalMeasurementFuser(config)
        self.smoother = ShapeConstrainedSmoother(config)
        self._external_pixels: np.ndarray | None = None
        self._external_confidence: np.ndarray | None = None
        self._external_visible: np.ndarray | None = None
        self._external_backend = ""
        self._keypoint_order_reference: np.ndarray | None = None
        self._initial_projected_gap = float("nan")
        self._last_material_result = None

    def set_roi(self, roi: tuple[float, float, float, float]) -> None:
        self.tracker.set_color_roi(roi)

    @property
    def target_locked(self) -> bool:
        return self.body_detector.target_locked or self.yolo_pose._target_roi is not None

    def set_target_lock(self, roi: tuple[float, float, float, float]) -> None:
        self.body_detector.set_target_lock(roi)
        self.yolo_pose.set_target_lock(roi)

    def initialise_keypoints(self, image_bgr: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        """Condition colour appearance and SAM identity from one K0--K6 frame."""

        points = np.asarray(pixels, dtype=float).reshape(KEYPOINT_COUNT, 2)
        if not np.isfinite(points).all():
            raise ValueError("K0-K6 initialization must contain seven finite pixels")
        points = self.point_refiner.initialise(image_bgr, points)
        self._keypoint_order_reference = points.copy()
        self._initial_projected_gap = float(np.median(
            np.linalg.norm(np.diff(points, axis=0), axis=1)
        ))
        height, width = np.asarray(image_bgr).shape[:2]
        margin_x = max(8.0, 0.55 * float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1))))
        margin_y = max(10.0, margin_x * 1.8)
        roi = (
            float(np.clip((np.min(points[:, 0]) - margin_x) / width, 0.0, 1.0)),
            float(np.clip((np.min(points[:, 1]) - margin_y) / height, 0.0, 1.0)),
            float(np.clip((np.max(points[:, 0]) + margin_x) / width, 0.0, 1.0)),
            float(np.clip((np.max(points[:, 1]) + margin_y) / height, 0.0, 1.0)),
        )
        self.set_target_lock(roi)
        self.body_detector.condition_keypoints(image_bgr, points)
        self.material_tracker.initialise(image_bgr, points)
        return points.copy()

    def set_external_keypoints(
        self,
        pixels: np.ndarray,
        confidence: np.ndarray,
        visible: np.ndarray,
        backend: str = "cotracker3-offline-cuda",
    ) -> None:
        self._external_pixels = np.asarray(pixels, dtype=float).reshape(KEYPOINT_COUNT, 2).copy()
        self._external_confidence = np.asarray(confidence, dtype=float).reshape(KEYPOINT_COUNT).copy()
        self._external_visible = np.asarray(visible, dtype=bool).reshape(KEYPOINT_COUNT).copy()
        self._external_backend = str(backend)

    def clear_external_keypoints(self) -> None:
        self._external_pixels = None
        self._external_confidence = None
        self._external_visible = None
        self._external_backend = ""

    def clear_target_lock(self) -> None:
        self.body_detector.clear_target_lock()
        self.yolo_pose.clear_target_lock()

    def reset(self) -> None:
        self.tracker.reset_filters()
        self.body_detector.reset()
        self.yolo_pose.reset()
        self.point_refiner.reset()
        self.material_tracker.reset()
        self.clear_external_keypoints()
        self._keypoint_order_reference = None
        self._initial_projected_gap = float("nan")
        self._last_material_result = None
        self.smoother.reset()

    def _align_body_result_to_manual_order(self, body):
        """Keep BodyFirst identities consistent with the user's K0--K6 order."""

        if body is None or not body.accepted:
            return body
        pixels = np.asarray(body.pixels, dtype=float).reshape(KEYPOINT_COUNT, 2)
        reference = self._keypoint_order_reference
        if reference is not None and np.isfinite(reference).all() and np.isfinite(pixels).all():
            # Compare shape after removing whole-chain translation.  This
            # remains valid while the robot inserts hundreds of pixels.
            reference_shape = reference - np.mean(reference, axis=0)
            direct_shape = pixels - np.mean(pixels, axis=0)
            reverse_shape = pixels[::-1] - np.mean(pixels, axis=0)
            direct_error = float(np.mean(np.linalg.norm(direct_shape - reference_shape, axis=1)))
            reverse_error = float(np.mean(np.linalg.norm(reverse_shape - reference_shape, axis=1)))
            if reverse_error < direct_error:
                body = replace(
                    body,
                    pixels=np.asarray(body.pixels)[::-1].copy(),
                    confidence=np.asarray(body.confidence)[::-1].copy(),
                    band_masks=tuple(reversed(body.band_masks)),
                )
                pixels = np.asarray(body.pixels, dtype=float)
            previous_gaps = np.linalg.norm(np.diff(reference, axis=0), axis=1)
            current_gaps = np.linalg.norm(np.diff(pixels, axis=0), axis=1)
            gap_ratio = current_gaps / np.maximum(previous_gaps, 1.0)
            current_median_gap = float(np.median(current_gaps))
            within_chain_ratio = current_gaps / max(current_median_gap, 1.0)
            displacement = pixels - reference
            median_motion = np.median(displacement, axis=0)
            differential = np.linalg.norm(displacement - median_motion, axis=1)
            typical_gap = float(np.median(previous_gaps))
            topology_broken = bool(
                np.any(gap_ratio < 0.55)
                or np.any(gap_ratio > 1.65)
                or np.any(within_chain_ratio < 0.78)
                or np.any(within_chain_ratio > 1.40)
                or np.max(differential) > max(8.0, 0.72 * typical_gap)
                or (
                    np.isfinite(self._initial_projected_gap)
                    and current_median_gap < 0.58 * self._initial_projected_gap
                )
            )
            if topology_broken:
                # Six physical intervals are all 7 mm.  Perspective may change
                # their common projected pitch gradually, but one interval
                # cannot collapse independently.  Use the robust current pitch
                # while limiting its frame-to-frame change.
                repair_gap = float(np.clip(
                    current_median_gap,
                    max(
                        0.82 * typical_gap,
                        0.58 * self._initial_projected_gap
                        if np.isfinite(self._initial_projected_gap) else 0.0,
                    ),
                    1.18 * typical_gap,
                ))
                repaired = TrackedPointRefiner._regularise_material_chain(
                    pixels, reference, repair_gap
                )
                body = replace(
                    body,
                    pixels=repaired,
                    confidence=np.minimum(np.asarray(body.confidence), 0.35),
                    temporally_predicted=True,
                    reason=f"{body.reason}; topology-safe temporal repair",
                )
        if not body.temporally_predicted:
            self._keypoint_order_reference = np.asarray(body.pixels, dtype=float).copy()
        return body

    def process(self, frame: CameraFrame) -> SevenMarkerResult:
        started = time.perf_counter()
        device = frame.metadata.get("device", {}) if isinstance(frame.metadata, dict) else {}
        synthetic = isinstance(device, dict) and str(device.get("serial", "")) == "SIM"
        # The ordering is intentional: establish the black continuum body
        # before any colour-marker observation is allowed to reach 3-D
        # measurement.  MarkerTracker is retained below for its calibrated
        # depth/IR measurement implementation and for synthetic self-tests.
        external = not synthetic and self._external_pixels is not None
        material = (
            self.material_tracker.update(frame.color_bgr)
            if not synthetic and not external and self.material_tracker.initialised
            else None
        )
        self._last_material_result = material
        material_accepted = material is not None and material.accepted
        yolo = (
            None
            if synthetic or external or material_accepted
            else self.yolo_pose.detect(frame.color_bgr)
        )
        body = None
        external_mask = None
        if external:
            # Propagate the last accepted material chain first.  The raw
            # CoTracker proposal is deliberately weak because transparent
            # lung-wall highlights can remain stationary with high learned
            # visibility while the actual continuum translates underneath.
            points = self.point_refiner.predict(
                self._external_pixels, self._external_confidence
            )
            # SAM receives a dense, moving material path rather than seven
            # isolated points or the stale conditioning-frame trajectory.
            dense_parts = []
            for index in range(KEYPOINT_COUNT - 1):
                alpha = np.linspace(0.0, 1.0, 16, endpoint=False)[:, None]
                dense_parts.append((1.0 - alpha) * points[index] + alpha * points[index + 1])
            dense_path = np.vstack((*dense_parts, points[-1:])).astype(np.float32)
            gap = float(np.nanmedian(np.linalg.norm(np.diff(points, axis=0), axis=1)))
            neural_result = self.body_detector.neural_segmenter.segment(
                frame.color_bgr, dense_path, max(2.5, 0.34 * gap)
            )
            if neural_result.accepted:
                external_mask = neural_result.mask
            refined, refined_confidence, refined_valid = self.point_refiner.refine(
                frame.color_bgr,
                self._external_pixels,
                self._external_confidence,
                external_mask,
            )
            refined_confidence = np.where(refined_valid, refined_confidence, 0.0)
        if material_accepted:
            # The seven distinct colour identities and their material topology
            # are the per-frame metric observation.  SAM2 remains a coarse
            # recovery backend, never a hard veto on visible colour rings.
            body = material
        elif not synthetic and not external and (yolo is None or not yolo.accepted):
            # Before a custom best.pt exists, retain the old method only as an
            # explicitly reported fallback.  Strict mode makes the pipeline
            # fail closed, which is recommended for quantitative experiments.
            if not self.yolo_pose.strict:
                body = self.body_detector.detect(
                    frame.color_bgr, self.tracker.color_roi_normalized
                )
                body = self._align_body_result_to_manual_order(body)
        # External CoTracker3 points are already the authoritative 2-D
        # observation.  Running the legacy colour detector here used to spend
        # hundreds of milliseconds and mutate tracking state only to have its
        # output overwritten below.  Build an explicit empty measurement shell
        # and use only MarkerTracker's calibrated D435 measurement primitive.
        observation = (
            empty_keypoint_observation(source="awaiting_cotracker3_measurement")
            if external
            else self.tracker.process(frame)
        )
        visual_pixels = None
        visual_confidence = None
        visual_masks = None
        visual_predicted = False
        visual_valid = None
        if external:
            visual_pixels = refined
            visual_confidence = refined_confidence
            visual_valid = refined_valid
            visual_masks = [np.empty((0, 0), dtype=np.uint8) for _ in range(KEYPOINT_COUNT)]
        elif yolo is not None and yolo.accepted:
            visual_pixels = yolo.pixels
            visual_confidence = yolo.confidence
            visual_masks = [np.empty((0, 0), dtype=np.uint8) for _ in range(KEYPOINT_COUNT)]
        elif body is not None and body.accepted:
            visual_pixels = body.pixels
            visual_confidence = body.confidence
            visual_masks = body.band_masks
            visual_predicted = body.temporally_predicted
        if visual_pixels is not None:
            raw = observation.raw_camera_m.copy()
            covariance = observation.covariance_m2.copy()
            valid = observation.valid.copy()
            confidence = observation.confidence.copy()
            left_pixels = observation.left_pixels.copy()
            right_pixels = observation.right_pixels.copy()
            sources = list(observation.source)
            for index in range(KEYPOINT_COUNT):
                if visual_valid is not None and not visual_valid[index]:
                    raw[index] = np.nan
                    covariance[index] = np.nan
                    valid[index] = False
                    confidence[index] = 0.0
                    left_pixels[index] = np.nan
                    right_pixels[index] = np.nan
                    sources[index] = "cotracker3_rejected"
                elif visual_predicted:
                    raw[index] = np.nan
                    covariance[index] = np.nan
                    valid[index] = False
                    confidence[index] = 0.0
                    left_pixels[index] = np.nan
                    right_pixels[index] = np.nan
                    sources[index] = "body_marker_temporal_prediction"
                else:
                    measured = self.tracker._measure_one(
                        frame,
                        visual_pixels[index],
                        float(visual_confidence[index]),
                        visual_masks[index],
                    )
                    point, point_covariance, left, right, point_confidence, source = measured
                    is_valid = np.isfinite(point).all() and point_confidence > 0.02
                    raw[index] = point
                    covariance[index] = point_covariance
                    valid[index] = is_valid
                    confidence[index] = point_confidence
                    left_pixels[index] = left
                    right_pixels[index] = right
                    sources[index] = source if is_valid else "body_band_2d"
            observation = replace(
                observation,
                raw_camera_m=raw,
                covariance_m2=covariance,
                valid=valid,
                confidence=confidence,
                color_pixels=visual_pixels,
                left_pixels=left_pixels,
                right_pixels=right_pixels,
                source=tuple(sources),
                color_confidence=visual_confidence,
            )
            self.tracker.last_color_pixels[:] = visual_pixels
            segmentation_valid = body is not None and body.accepted
            if external_mask is not None:
                segmentation_valid = True
            self.tracker.last_continuum_segmentation = ContinuumSegmentation(
                valid=segmentation_valid,
                body_mask=(external_mask if external_mask is not None else (body.body_mask if body is not None and body.accepted else np.zeros(frame.color_bgr.shape[:2], np.uint8))),
                search_mask=(external_mask if external_mask is not None else (body.search_mask if body is not None and body.accepted else np.zeros(frame.color_bgr.shape[:2], np.uint8))),
                centerline_xy=(visual_pixels if external else (body.centerline_xy if body is not None and body.accepted else visual_pixels[::-1])),
                ordered_band_pixels=visual_pixels,
                ordered_band_confidence=visual_confidence,
                ordered_band_masks=list(visual_masks),
                ordered_band_colours=np.full((KEYPOINT_COUNT, 3), np.nan),
            )
        elif not synthetic:
            # The old colour-only and equal-spacing fallbacks could always
            # manufacture seven points on the drive rails.  In strict mode a
            # missing body/band observation stays missing.  Dynamics may still
            # predict 3-D state, but no predicted point is drawn on the RGB
            # image as if it had been visually detected.
            observation = replace(
                observation,
                raw_camera_m=np.full_like(observation.raw_camera_m, np.nan),
                covariance_m2=np.full_like(observation.covariance_m2, np.nan),
                valid=np.zeros_like(observation.valid, dtype=bool),
                confidence=np.zeros_like(observation.confidence),
                color_pixels=np.full_like(observation.color_pixels, np.nan),
                left_pixels=np.full_like(observation.left_pixels, np.nan),
                right_pixels=np.full_like(observation.right_pixels, np.nan),
                source=tuple("body_not_confirmed" for _ in range(KEYPOINT_COUNT)),
                color_confidence=np.zeros_like(observation.color_confidence),
            )
        bundle = self.measurements.fuse(frame, observation)
        smoothed, covariance, predicted, confidence = self.smoother.update(
            bundle.fused_points_m,
            bundle.covariance_m2,
            bundle.valid,
            bundle.confidence,
            frame.capture_host_ns,
        )
        base = _transform_points(self.tracker.transform_base_from_camera, smoothed)
        return SevenMarkerResult(
            sequence=frame.sequence,
            capture_host_ns=frame.capture_host_ns,
            raw_camera_m=observation.raw_camera_m,
            fused_camera_m=bundle.fused_points_m,
            smoothed_camera_m=smoothed,
            base_m=base,
            covariance_m2=covariance,
            measured_valid=bundle.valid,
            predicted=predicted,
            confidence=confidence,
            color_pixels=observation.color_pixels,
            left_pixels=observation.left_pixels,
            right_pixels=observation.right_pixels,
            sources=bundle.sources,
            rgb_depth_camera_m=bundle.depth_points_m,
            ir_stereo_camera_m=bundle.stereo_points_m,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )

    def draw_overlay(self, frame: CameraFrame, result: SevenMarkerResult) -> np.ndarray:
        image = np.asarray(frame.color_bgr).copy()
        neural = self.body_detector.neural_segmenter
        yolo_observation = self.yolo_pose.last_observation
        external = bool(self._external_backend)
        material = self._last_material_result
        material_accepted = material is not None and material.accepted
        if external:
            points = result.color_pixels
            if not np.isfinite(points).all():
                points = self.point_refiner.predict(
                    self._external_pixels, self._external_confidence
                )
            x0, y0 = np.nanmin(points, axis=0)
            x1, y1 = np.nanmax(points, axis=0)
            cv2.rectangle(image, (int(x0 - 12), int(y0 - 16)), (int(x1 + 12), int(y1 + 16)), (52, 199, 89), 2, cv2.LINE_AA)
            cv2.putText(
                image,
                f"MATERIAL CHAIN  {self.point_refiner.tracking_source.upper()}",
                (int(x0), max(20, int(y0) - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (52, 199, 89),
                2,
                cv2.LINE_AA,
            )
        elif material_accepted:
            points = np.asarray(material.pixels, dtype=float)
            x0, y0 = np.min(points, axis=0)
            x1, y1 = np.max(points, axis=0)
            cv2.rectangle(
                image,
                (int(x0 - 12), int(y0 - 18)),
                (int(x1 + 12), int(y1 + 18)),
                (52, 199, 89),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "SEVEN-COLOUR MATERIAL CHAIN",
                (int(x0), max(20, int(y0) - 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (52, 199, 89),
                2,
                cv2.LINE_AA,
            )
        elif yolo_observation.accepted:
            left, top, right, bottom = np.rint(yolo_observation.box_xyxy).astype(int)
            cv2.rectangle(image, (left, top), (right, bottom), (52, 199, 89), 2, cv2.LINE_AA)
            cv2.putText(
                image, f"YOLO POSE {yolo_observation.object_confidence:.2f}",
                (left + 5, max(20, top - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (52, 199, 89), 2, cv2.LINE_AA,
            )
        elif neural.target_locked:
            tracking_roi = self.body_detector.tracking_roi_normalized(
                image.shape[:2], self.tracker.color_roi_normalized
            )
            x0, y0, x1, y1 = tracking_roi
            left, top = int(round(x0 * image.shape[1])), int(round(y0 * image.shape[0]))
            right, bottom = int(round(x1 * image.shape[1])), int(round(y1 * image.shape[0]))
            lock_colour = (60, 210, 110) if "lost" not in neural.last_backend else (40, 55, 245)
            cv2.rectangle(image, (left, top), (right, bottom), lock_colour, 2, cv2.LINE_AA)
            cv2.putText(
                image,
                "LOCKED TARGET" if "lost" not in neural.last_backend else "TARGET LOST - RELOCK",
                (left + 5, max(20, top - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                lock_colour,
                2,
                cv2.LINE_AA,
            )
        segmentation = self.tracker.last_continuum_segmentation
        if segmentation is not None and segmentation.valid:
            scale_x = image.shape[1] / segmentation.body_mask.shape[1]
            scale_y = image.shape[0] / segmentation.body_mask.shape[0]
            contour_mask = cv2.resize(
                segmentation.body_mask, image.shape[1::-1], interpolation=cv2.INTER_NEAREST
            )
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (80, 235, 135), 2, cv2.LINE_AA)
            if len(segmentation.centerline_xy) > 1:
                path = segmentation.centerline_xy.copy()
                path[:, 0] *= scale_x
                path[:, 1] *= scale_y
                cv2.polylines(image, [np.rint(path).astype(np.int32)], False, (80, 180, 255), 1, cv2.LINE_AA)
        colours = [tuple(map(int, marker["display_bgr"])) for marker in self.config["markers"]]
        finite_pixels = np.isfinite(result.color_pixels).all(axis=1)
        ordered = result.color_pixels[finite_pixels]
        if len(ordered) > 1:
            cv2.polylines(image, [np.rint(ordered).astype(np.int32)], False, (245, 245, 245), 2, cv2.LINE_AA)
        for index in range(KEYPOINT_COUNT):
            pixel = result.color_pixels[index]
            if not np.isfinite(pixel).all():
                continue
            u, v = np.rint(pixel).astype(int)
            colour = colours[index]
            cv2.circle(image, (u, v), 8, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.circle(image, (u, v), 5, colour, -1, cv2.LINE_AA)
            point = result.base_m[index] * 1000.0
            state = "P" if result.predicted[index] else ("M" if result.measured_valid[index] else "X")
            text = (
                f"K{index} {state} [{point[0]:+.1f},{point[1]:+.1f},{point[2]:+.1f}] mm"
                if np.isfinite(point).all() else f"K{index} invalid"
            )
            cv2.putText(image, text, (u + 10, max(18, v - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        band_count = int(np.count_nonzero(finite_pixels))
        valid_count = int(np.count_nonzero(result.measured_valid))
        if external:
            backend = self._external_backend.upper()
            body_confidence = neural.last_confidence
            band_state = "TRACKED"
        elif material_accepted:
            backend = "LK+LAB 7-COLOUR TOPOLOGY"
            body_confidence = float(material.segmentation_confidence)
            band_state = "PRED" if material.temporally_predicted else "DIRECT"
        elif yolo_observation.accepted:
            backend = "YOLO-POSE-CUDA"
            body_confidence = yolo_observation.object_confidence
            band_state = "LEARNED"
        else:
            backend = "BODY-LATTICE " + neural.last_backend.upper()
            body_confidence = neural.last_confidence
            band_state = "PRED" if self.body_detector.last_band_prediction else "VISION"
        diagnostic_height = 58 if external and body_confidence <= 0.01 else 34
        cv2.rectangle(
            image,
            (0, 0),
            (min(image.shape[1], 1180), diagnostic_height),
            (14, 16, 20),
            -1,
        )
        lock_state = "LOCKED" if neural.target_locked else "UNLOCKED"
        cv2.putText(
            image,
            f"{lock_state}  {backend} BODY {body_confidence:.2f}  bands {band_count}/7 {band_state}  3D measured {valid_count}/7  {result.processing_ms:.1f} ms",
            (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (245, 245, 247), 1, cv2.LINE_AA,
        )
        if external and body_confidence <= 0.01:
            reason = str(neural.last_reason).replace("\n", " ")
            cv2.putText(
                image,
                f"SAM2 NOT ACTIVE: {reason[:150]}",
                (10, 47),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (70, 105, 255),
                1,
                cv2.LINE_AA,
            )
        return image
