"""Reusable capture pipeline shared by the standalone and integrated UIs."""

from __future__ import annotations

import concurrent.futures
import time

import numpy as np

from .axes import NullAxisSource
from .models import AlignedSample, KEYPOINT_COUNT, empty_keypoint_observation
from .multiview import D435_INTERNAL_MODE, DUAL_VIEW_MODE, FUSION_MODES, SideViewFusion
from .recording import SessionRecorder
from .sync import AxisSampler
from .tracking import MarkerTracker


class CaptureEngine:
    def __init__(
        self,
        config: dict,
        camera_source,
        axis_source=None,
        mujoco_bridge=None,
        side_camera_source=None,
    ) -> None:
        self.config = config
        self.camera_source = camera_source
        self.axis_source = axis_source or NullAxisSource(config["axes"])
        self.axis_sampler = AxisSampler(
            self.axis_source,
            sample_rate_hz=float(config["axes"]["sample_rate_hz"]),
        )
        self.bridge = mujoco_bridge
        self.tracker = MarkerTracker(config)
        self.keypoint_tracking_enabled = bool(config.get("tracking", {}).get("enabled", True))
        if hasattr(self.camera_source, "set_infrared_copy_enabled"):
            self.camera_source.set_infrared_copy_enabled(self.keypoint_tracking_enabled)
        self.side_camera_source = side_camera_source
        self.fusion_mode = str(config.get("fusion", {}).get("mode", D435_INTERNAL_MODE))
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
        self.side_fusion = SideViewFusion(config)
        source_transform = getattr(camera_source, "transform_base_from_camera", None)
        if source_transform is not None:
            self.tracker.set_transform_base_from_camera(source_transform, ready=True)
            self.side_fusion.set_d435_transform(source_transform)
        self._use_side_source_calibration()
        self.recorder = SessionRecorder(config)
        self.running = False
        self.latest_frame = None
        self.latest_sample: AlignedSample | None = None
        self.latest_overlay = None
        self.latest_side_frame = None
        self.latest_side_overlay = None
        self.last_processing_ms = 0.0
        self._side_detection_executor: concurrent.futures.ThreadPoolExecutor | None = None

    def _use_side_source_calibration(self) -> None:
        if self.side_camera_source is None:
            return
        transform = getattr(self.side_camera_source, "transform_base_from_camera", None)
        if transform is not None:
            intrinsics = getattr(self.side_camera_source, "intrinsics", None)
            self.side_fusion.set_side_calibration(transform, intrinsics, ready=True)

    def set_d435_calibration(self, transform: np.ndarray, ready: bool = True) -> None:
        self.tracker.set_transform_base_from_camera(transform, ready=ready)
        self.side_fusion.set_d435_transform(transform)

    def set_side_calibration(
        self,
        transform: np.ndarray,
        intrinsics: dict | None = None,
        ready: bool = True,
    ) -> None:
        self.side_fusion.set_side_calibration(transform, intrinsics, ready=ready)
        if intrinsics is not None and self.side_camera_source is not None and hasattr(self.side_camera_source, "set_intrinsics"):
            self.side_camera_source.set_intrinsics(intrinsics, ready=True)

    def set_side_camera_source(self, source) -> None:
        was_running = self.running and self.fusion_mode == DUAL_VIEW_MODE
        if was_running and self.side_camera_source is not None:
            self.side_camera_source.stop()
        self.side_camera_source = source
        self.latest_side_frame = None
        self.latest_side_overlay = None
        self._use_side_source_calibration()
        if was_running and source is not None:
            source.start()

    def set_axis_source(self, source) -> None:
        was_running = self.running
        self.axis_sampler.stop()
        self.axis_source = source or NullAxisSource(self.config["axes"])
        self.axis_sampler = AxisSampler(
            self.axis_source,
            sample_rate_hz=float(self.config["axes"]["sample_rate_hz"]),
        )
        if was_running:
            self.axis_sampler.start()

    def set_mujoco_bridge(self, bridge) -> None:
        self.bridge = bridge

    def set_keypoint_tracking_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self.keypoint_tracking_enabled:
            return
        self.keypoint_tracking_enabled = enabled
        if hasattr(self.camera_source, "set_infrared_copy_enabled"):
            self.camera_source.set_infrared_copy_enabled(enabled)
        self.config.setdefault("tracking", {})["enabled"] = enabled
        self.tracker.reset_filters()
        self.side_fusion.reset()

    def set_tracking_rois(
        self,
        d435_color_roi: tuple[float, float, float, float],
        side_color_roi: tuple[float, float, float, float],
    ) -> None:
        self.tracker.set_color_roi(d435_color_roi)
        self.side_fusion.set_side_roi(side_color_roi)

    def set_recording_root(self, root: str) -> None:
        if self.recorder.active:
            raise RuntimeError("记录过程中不能修改session输出目录")
        self.config["recording"]["root"] = str(root)
        self.recorder = SessionRecorder(self.config)

    def set_fusion_mode(self, mode: str) -> None:
        mode = str(mode)
        if mode not in FUSION_MODES:
            raise ValueError(f"Unsupported fusion mode: {mode}")
        if mode == self.fusion_mode:
            return
        if self.running and mode == DUAL_VIEW_MODE and self.side_camera_source is None:
            raise RuntimeError("双视图模式尚未配置侧面RGB相机")
        if self.running and self.fusion_mode == DUAL_VIEW_MODE and self.side_camera_source is not None:
            self.side_camera_source.stop()
        self.fusion_mode = mode
        self.config["fusion"]["mode"] = mode
        self.tracker.reset_filters()
        self.side_fusion.reset()
        self.latest_side_frame = None
        self.latest_side_overlay = None
        if self.running and mode == DUAL_VIEW_MODE:
            self.side_camera_source.start()

    def start(self) -> None:
        if self.running:
            return
        self.camera_source.start()
        if self.fusion_mode == DUAL_VIEW_MODE:
            if self.side_camera_source is None:
                self.camera_source.stop()
                raise RuntimeError("双视图模式需要侧面RGB相机")
            self.side_camera_source.start()
        self.axis_sampler.start()
        self.running = True

    def stop(self) -> None:
        if self.recorder.active:
            self.stop_recording()
        self.axis_sampler.stop()
        if self.side_camera_source is not None:
            self.side_camera_source.stop()
        self.camera_source.stop()
        if self._side_detection_executor is not None:
            self._side_detection_executor.shutdown(wait=True, cancel_futures=True)
            self._side_detection_executor = None
        self.running = False

    def start_recording(self) -> str:
        calibration = {
            "d435": self.config["calibration"],
            "side_camera": {
                "intrinsics_ready": self.config["side_camera"].get("intrinsics_ready", False),
                "intrinsics": self.config["side_camera"]["intrinsics"],
                "calibration_ready": self.config["side_camera"].get("calibration_ready", False),
                "transform_base_from_camera": self.config["side_camera"]["transform_base_from_camera"],
            },
            "fusion": self.config["fusion"],
        }
        camera_info = {
            "d435": getattr(self.camera_source, "device_info", {}),
            "side_rgb": getattr(self.side_camera_source, "device_info", {}) if self.side_camera_source is not None else None,
            "fusion_mode": self.fusion_mode,
        }
        session = self.recorder.start(camera_info, calibration)
        bag_ok = bool(self.camera_source.start_bag_recording(session / "realsense.bag"))
        self.recorder.mark_bag_recording(bag_ok)
        return str(session)

    def stop_recording(self) -> str | None:
        self.camera_source.stop_bag_recording()
        session = self.recorder.stop()
        return None if session is None else str(session)

    def process_once(self) -> AlignedSample | None:
        if not self.running:
            return None
        frame = self.camera_source.poll()
        if frame is None:
            return None
        processing_started = time.perf_counter()
        self.latest_frame = frame
        side_frame = None
        if self.fusion_mode == DUAL_VIEW_MODE and self.side_camera_source is not None:
            poll_nearest = getattr(self.side_camera_source, "poll_nearest", None)
            side_frame = poll_nearest(frame.capture_host_ns) if callable(poll_nearest) else self.side_camera_source.poll()
            self.latest_side_frame = side_frame
        else:
            self.latest_side_frame = None
        if self.keypoint_tracking_enabled:
            side_detection_future = None
            if self.fusion_mode == DUAL_VIEW_MODE and side_frame is not None:
                if self._side_detection_executor is None:
                    self._side_detection_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix="tdcr-side-detection",
                    )
                side_detection_future = self._side_detection_executor.submit(
                    self.side_fusion.detect_side, side_frame
                )
            internal_observation = self.tracker.process(frame)
            if self.fusion_mode == DUAL_VIEW_MODE:
                side_detection = (
                    side_detection_future.result() if side_detection_future is not None else None
                )
                observation = self.side_fusion.fuse(
                    frame,
                    internal_observation,
                    side_frame,
                    side_detection=side_detection,
                )
                self.latest_side_overlay = self.side_fusion.latest_side_overlay
            else:
                observation = internal_observation
                self.latest_side_overlay = None
        else:
            observation = empty_keypoint_observation(self.fusion_mode)
            self.latest_side_overlay = None if side_frame is None else side_frame.image_bgr.copy()
        axes = self.axis_sampler.interpolate(frame.capture_host_ns)
        simulation = np.full((KEYPOINT_COUNT, 3), np.nan)
        errors = np.full(KEYPOINT_COUNT, np.nan)
        rmse = float("nan")
        valid_count = int(np.count_nonzero(observation.valid))
        if self.bridge is not None:
            self.bridge.apply_controls(axes.control_m, frame.capture_host_ns)
            if self.keypoint_tracking_enabled:
                simulation, errors, rmse, valid_count = self.bridge.compare(
                    observation.base_m, observation.valid
                )
        sample = AlignedSample(
            frame=frame,
            keypoints=observation,
            axes=axes,
            simulation_points_base_m=simulation,
            errors_mm=errors,
            rmse_mm=rmse,
            valid_count=valid_count,
            side_frame=side_frame,
            fusion_mode=self.fusion_mode,
            d435_tracking_roi=np.asarray(self.tracker.color_roi_normalized, dtype=float),
            side_tracking_roi=np.asarray(self.side_fusion.side_roi_normalized, dtype=float),
        )
        if self.keypoint_tracking_enabled:
            overlay = self.tracker.draw_overlay(frame, observation)
        else:
            import cv2
            overlay = frame.color_bgr.copy()
            cv2.putText(
                overlay, "KEYPOINT CAPTURE OFF", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 170, 185), 2, cv2.LINE_AA,
            )
        if np.isfinite(rmse):
            import cv2
            cv2.putText(
                overlay, f"MuJoCo RMSE {rmse:.2f} mm", (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA,
            )
        self.latest_sample = sample
        self.latest_overlay = overlay
        if self.recorder.active:
            self.recorder.append(sample, overlay, self.latest_side_overlay)
        self.last_processing_ms = (time.perf_counter() - processing_started) * 1000.0
        return sample
