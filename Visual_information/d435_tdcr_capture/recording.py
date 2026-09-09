"""Crash-tolerant session recording and deterministic NumPy export."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import queue
import threading
from typing import Any

import cv2
import numpy as np

from .config import PROJECT_ROOT
from .models import AlignedSample, KEYPOINT_COUNT


def _viewer_depth_visual(
    depth_m: np.ndarray,
    minimum: float,
    maximum: float,
    depth_unit_m: float = 1.0,
) -> np.ndarray:
    source = np.asarray(depth_m)
    depth = source.astype(np.float32)
    if source.dtype == np.uint16:
        depth *= float(depth_unit_m)
    valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    clipped = np.nan_to_num(depth, nan=minimum, posinf=maximum, neginf=minimum).copy()
    np.clip(clipped, minimum, maximum, out=clipped)
    scale = 255.0 / max(maximum - minimum, 1e-6)
    normalized = cv2.convertScaleAbs(clipped, alpha=scale, beta=-minimum * scale)
    visual = cv2.applyColorMap(cv2.subtract(255, normalized), cv2.COLORMAP_TURBO)
    visual[~valid] = (12, 13, 16)
    return visual


def _viewer_video_process(
    name: str,
    path: str,
    fps: float,
    codecs: tuple[str, ...],
    minimum_depth_m: float,
    maximum_depth_m: float,
    depth_unit_m: float,
    input_queue,
    result_queue,
    shared_name: str = "",
    shared_shape: tuple[int, ...] = (),
    shared_dtype: str = "",
    slot_bytes: int = 0,
    free_queue=None,
    ready_event=None,
) -> None:
    """Encode one stream outside the UI/camera process."""
    writer = None
    selected_codec = ""
    encoded = 0
    error = ""
    shared_block = None
    try:
        if shared_name:
            shared_block = shared_memory.SharedMemory(name=shared_name)
            dtype = np.dtype(shared_dtype)
        if ready_event is not None:
            ready_event.set()
        while True:
            packet = input_queue.get()
            if packet is None:
                break
            slot = None
            try:
                if shared_block is not None:
                    slot = int(packet)
                    image = np.ndarray(
                        shared_shape,
                        dtype=dtype,
                        buffer=shared_block.buf,
                        offset=slot * slot_bytes,
                    )
                else:
                    image = packet["image"]
                if name == "depth":
                    image = _viewer_depth_visual(
                        image, minimum_depth_m, maximum_depth_m, depth_unit_m
                    )
                image = np.ascontiguousarray(image, dtype=np.uint8)
                if writer is None:
                    height, width = image.shape[:2]
                    failures: list[str] = []
                    for codec in dict.fromkeys((*codecs, "mp4v")):
                        if len(codec) != 4:
                            failures.append(f"{codec}: invalid FourCC")
                            continue
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        backends = (
                            (cv2.CAP_FFMPEG, cv2.CAP_ANY)
                            if hasattr(cv2, "CAP_FFMPEG")
                            else (cv2.CAP_ANY,)
                        )
                        for backend in backends:
                            candidate = cv2.VideoWriter(
                                path,
                                int(backend),
                                fourcc,
                                float(fps),
                                (int(width), int(height)),
                            )
                            if candidate.isOpened():
                                writer = candidate
                                selected_codec = codec
                                break
                            candidate.release()
                        if writer is not None:
                            break
                        failures.append(f"{codec}: unavailable")
                    if writer is None:
                        raise RuntimeError("; ".join(failures))
                writer.write(image)
                encoded += 1
            finally:
                if slot is not None and free_queue is not None:
                    free_queue.put(slot)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if writer is not None:
            writer.release()
        if shared_block is not None:
            shared_block.close()
        result_queue.put((name, encoded, selected_codec, error))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SessionRecorder:
    def __init__(self, config: dict) -> None:
        self.config = config
        root = Path(config["recording"]["root"])
        self.root = (PROJECT_ROOT / root).resolve() if not root.is_absolute() else root.resolve()
        self.session_dir: Path | None = None
        self.csv_file = None
        self.csv_writer = None
        self.video_writer = None
        self.side_video_writer = None
        self.raw_rgb_writer = None
        self.synchronized_side_writer = None
        self.synchronized_pair_writer = None
        self.video_timestamps_file = None
        self.video_timestamps_writer = None
        self.video_frame_count = 0
        self.video_queue: queue.Queue | None = None
        self.video_thread: threading.Thread | None = None
        self.video_error: str | None = None
        self.video_dropped_packets = 0
        self.frame_count = 0
        self.lock = threading.Lock()
        self.buffers: dict[str, list[Any]] = {}

    @property
    def active(self) -> bool:
        return self.csv_file is not None

    @staticmethod
    def _headers() -> list[str]:
        headers = [
            "sequence", "capture_host_ns", "host_arrival_ns", "device_timestamp_ms",
            "color_timestamp_ms", "depth_timestamp_ms", "infrared_timestamp_ms",
            "stream_skew_ms", "axis_status", "rmse_mm", "valid_count", "estimated_delay_ms",
            "fusion_mode", "side_sequence", "side_capture_host_ns", "side_time_offset_ms",
            "d435_roi_x0", "d435_roi_y0", "d435_roi_x1", "d435_roi_y1",
            "side_roi_x0", "side_roi_y0", "side_roi_x1", "side_roi_y1",
        ]
        for axis in range(7):
            headers += [
                f"axis_{axis}_demand_native", f"axis_{axis}_measured_native",
                f"axis_{axis}_demand_valid", f"axis_{axis}_measured_valid", f"control_{axis}_m",
            ]
        for index in range(KEYPOINT_COUNT):
            headers += [
                f"kp{index}_valid", f"kp{index}_predicted", f"kp{index}_confidence", f"kp{index}_source",
                f"kp{index}_color_confidence",
                f"kp{index}_color_u_px", f"kp{index}_color_v_px",
                f"kp{index}_left_u_px", f"kp{index}_left_v_px",
                f"kp{index}_right_u_px", f"kp{index}_right_v_px", f"kp{index}_disparity_px",
                f"kp{index}_sigma_x_mm", f"kp{index}_sigma_y_mm", f"kp{index}_sigma_z_mm",
                f"kp{index}_side_u_px", f"kp{index}_side_v_px", f"kp{index}_side_reprojection_error_px",
                f"kp{index}_raw_cam_x_m", f"kp{index}_raw_cam_y_m", f"kp{index}_raw_cam_z_m",
                f"kp{index}_filtered_cam_x_m", f"kp{index}_filtered_cam_y_m", f"kp{index}_filtered_cam_z_m",
                f"kp{index}_base_x_m", f"kp{index}_base_y_m", f"kp{index}_base_z_m",
                f"kp{index}_sim_x_m", f"kp{index}_sim_y_m", f"kp{index}_sim_z_m", f"kp{index}_error_mm",
            ]
        return headers

    def start(self, camera_info: dict[str, Any], calibration: dict[str, Any]) -> Path:
        with self.lock:
            if self.active:
                raise RuntimeError("A recording session is already active")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.session_dir = self.root / f"session_{stamp}"
            self.session_dir.mkdir(parents=True, exist_ok=False)
            self.csv_file = (self.session_dir / "samples.csv").open("w", newline="", encoding="utf-8-sig")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self._headers())
            self.csv_writer.writeheader()
            self.frame_count = 0
            self.video_frame_count = 0
            self.video_error = None
            self.video_dropped_packets = 0
            self.buffers = {name: [] for name in (
                "capture_host_ns", "device_timestamp_ms", "axis_demand_native", "axis_measured_native",
                "controls_m", "keypoints_raw_camera_m", "keypoints_filtered_camera_m", "keypoints_base_m",
                "keypoint_valid", "keypoint_confidence", "simulation_points_base_m", "errors_mm", "rmse_mm",
                "side_capture_host_ns", "side_pixels", "side_reprojection_error_px", "fusion_mode",
                "keypoint_predicted", "keypoint_covariance_m2", "keypoint_color_pixels",
                "keypoint_left_pixels", "keypoint_right_pixels", "keypoint_source",
                "keypoint_color_confidence",
                "d435_tracking_roi", "side_tracking_roi",
            )}
            xml_path = Path(self.config["mujoco"]["xml"])
            xml_path = (PROJECT_ROOT / xml_path).resolve() if not xml_path.is_absolute() else xml_path.resolve()
            manifest = {
                "schema_version": 3,
                "created_local": datetime.now().isoformat(),
                "camera": camera_info,
                "config": self.config,
                "mujoco_xml": str(xml_path),
                "mujoco_xml_sha256": _sha256(xml_path) if xml_path.exists() else None,
                "raw_bag_requested": True,
                "raw_bag_recorded": False,
                "synchronized_video_enabled": bool(
                    self.config["recording"].get("synchronized_video_enabled", True)
                ),
                "synchronized_video_clock": "d435_capture_host_ns",
            }
            self._write_json("session.json", manifest)
            self._write_json("calibration.json", calibration)
            if bool(self.config["recording"].get("synchronized_video_enabled", True)):
                self.video_timestamps_file = (
                    self.session_dir / "video_timestamps.csv"
                ).open("w", newline="", encoding="utf-8-sig")
                self.video_timestamps_writer = csv.DictWriter(
                    self.video_timestamps_file,
                    fieldnames=[
                        "video_frame_index", "d435_sequence", "d435_capture_host_ns",
                        "d435_color_timestamp_ms", "side_valid", "side_sequence",
                        "side_capture_host_ns", "side_offset_ms",
                    ],
                )
                self.video_timestamps_writer.writeheader()
            self.video_queue = queue.Queue(maxsize=120)
            self.video_thread = threading.Thread(
                target=self._video_loop, name="tdcr-session-video", daemon=True
            )
            self.video_thread.start()
            return self.session_dir

    def mark_bag_recording(self, success: bool) -> None:
        if self.session_dir is None:
            return
        path = self.session_dir / "session.json"
        try:
            with path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            manifest["raw_bag_recorded"] = bool(success)
            with path.open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _write_json(self, name: str, value: Any) -> None:
        with (self.session_dir / name).open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)

    @staticmethod
    def _open_video(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), tuple(map(int, size))
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"无法创建视频文件: {path}")
        return writer

    @staticmethod
    def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
        canvas = np.full((height, width, 3), 14, dtype=np.uint8)
        source_height, source_width = image.shape[:2]
        scale = min(width / max(source_width, 1), height / max(source_height, 1))
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        x0 = (width - resized_width) // 2
        y0 = (height - resized_height) // 2
        canvas[y0:y0 + resized_height, x0:x0 + resized_width] = resized
        return canvas

    def _append_synchronized_video(self, sample: AlignedSample) -> None:
        if not bool(self.config["recording"].get("synchronized_video_enabled", True)):
            return
        fps = float(self.config["recording"]["overlay_fps"])
        primary = np.asarray(sample.frame.color_bgr)
        if self.raw_rgb_writer is None:
            primary_height, primary_width = primary.shape[:2]
            self.raw_rgb_writer = self._open_video(
                self.session_dir / "d435_rgb.mp4", fps, (primary_width, primary_height)
            )
        self.raw_rgb_writer.write(primary)

        panel_width = int(self.config["recording"].get("synchronized_panel_width", 640))
        panel_height = int(self.config["recording"].get("synchronized_panel_height", 480))
        primary_panel = self._letterbox(primary, panel_width, panel_height)
        side_valid = sample.side_frame is not None
        if side_valid:
            side_panel = self._letterbox(sample.side_frame.image_bgr, panel_width, panel_height)
        else:
            side_panel = np.full((panel_height, panel_width, 3), 14, dtype=np.uint8)
            cv2.putText(
                side_panel, "SIDE RGB UNAVAILABLE", (24, panel_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 155, 165), 2, cv2.LINE_AA,
            )
        if self.synchronized_side_writer is None:
            self.synchronized_side_writer = self._open_video(
                self.session_dir / "side_rgb_synchronized.mp4",
                fps,
                (panel_width, panel_height),
            )
        self.synchronized_side_writer.write(side_panel)

        pair = np.hstack((primary_panel, side_panel))
        offset_ms = (
            (sample.side_frame.capture_host_ns - sample.frame.capture_host_ns) * 1e-6
            if side_valid else float("nan")
        )
        cv2.rectangle(pair, (0, 0), (pair.shape[1], 30), (12, 13, 16), -1)
        cv2.putText(pair, "D435 RGB", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 247), 1, cv2.LINE_AA)
        side_label = f"SIDE RGB   dt={offset_ms:+.2f} ms" if side_valid else "SIDE RGB   unavailable"
        cv2.putText(pair, side_label, (panel_width + 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 247), 1, cv2.LINE_AA)
        if self.synchronized_pair_writer is None:
            self.synchronized_pair_writer = self._open_video(
                self.session_dir / "synchronized_rgb.mp4",
                fps,
                (panel_width * 2, panel_height),
            )
        self.synchronized_pair_writer.write(pair)
        self.video_timestamps_writer.writerow({
            "video_frame_index": self.video_frame_count,
            "d435_sequence": sample.frame.sequence,
            "d435_capture_host_ns": sample.frame.capture_host_ns,
            "d435_color_timestamp_ms": sample.frame.color_timestamp_ms,
            "side_valid": int(side_valid),
            "side_sequence": sample.side_frame.sequence if side_valid else "",
            "side_capture_host_ns": sample.side_frame.capture_host_ns if side_valid else "",
            "side_offset_ms": offset_ms if side_valid else "",
        })
        self.video_frame_count += 1

    def _video_loop(self) -> None:
        while self.video_queue is not None:
            packet = self.video_queue.get()
            if packet is None:
                self.video_queue.task_done()
                return
            sample, overlay_bgr, side_overlay_bgr = packet
            try:
                if self.video_error is None:
                    fps = float(self.config["recording"]["overlay_fps"])
                    if overlay_bgr is not None:
                        if self.video_writer is None:
                            height, width = overlay_bgr.shape[:2]
                            self.video_writer = self._open_video(
                                self.session_dir / "overlay.mp4", fps, (width, height)
                            )
                        self.video_writer.write(overlay_bgr)
                    if side_overlay_bgr is not None:
                        if self.side_video_writer is None:
                            height, width = side_overlay_bgr.shape[:2]
                            self.side_video_writer = self._open_video(
                                self.session_dir / "side_overlay.mp4", fps, (width, height)
                            )
                        self.side_video_writer.write(side_overlay_bgr)
                    self._append_synchronized_video(sample)
            except Exception as exc:
                self.video_error = str(exc)
            finally:
                self.video_queue.task_done()

    def append(
        self,
        sample: AlignedSample,
        overlay_bgr: np.ndarray | None = None,
        side_overlay_bgr: np.ndarray | None = None,
    ) -> None:
        with self.lock:
            if not self.active:
                return
            row: dict[str, Any] = {
                "sequence": sample.frame.sequence,
                "capture_host_ns": sample.frame.capture_host_ns,
                "host_arrival_ns": sample.frame.host_arrival_ns,
                "device_timestamp_ms": sample.frame.device_timestamp_ms,
                "color_timestamp_ms": sample.frame.color_timestamp_ms,
                "depth_timestamp_ms": sample.frame.depth_timestamp_ms,
                "infrared_timestamp_ms": sample.frame.infrared_timestamp_ms,
                "stream_skew_ms": max(sample.frame.color_timestamp_ms, sample.frame.depth_timestamp_ms, sample.frame.infrared_timestamp_ms) - min(sample.frame.color_timestamp_ms, sample.frame.depth_timestamp_ms, sample.frame.infrared_timestamp_ms),
                "axis_status": sample.axes.status,
                "rmse_mm": sample.rmse_mm,
                "valid_count": sample.valid_count,
                "estimated_delay_ms": sample.estimated_delay_ms,
                "fusion_mode": sample.fusion_mode,
                "side_sequence": sample.side_frame.sequence if sample.side_frame is not None else "",
                "side_capture_host_ns": sample.side_frame.capture_host_ns if sample.side_frame is not None else "",
                "side_time_offset_ms": sample.keypoints.side_time_offset_ms,
                "d435_roi_x0": sample.d435_tracking_roi[0],
                "d435_roi_y0": sample.d435_tracking_roi[1],
                "d435_roi_x1": sample.d435_tracking_roi[2],
                "d435_roi_y1": sample.d435_tracking_roi[3],
                "side_roi_x0": sample.side_tracking_roi[0],
                "side_roi_y0": sample.side_tracking_roi[1],
                "side_roi_x1": sample.side_tracking_roi[2],
                "side_roi_y1": sample.side_tracking_roi[3],
            }
            for axis in range(7):
                row.update({
                    f"axis_{axis}_demand_native": sample.axes.demand_native[axis],
                    f"axis_{axis}_measured_native": sample.axes.measured_native[axis],
                    f"axis_{axis}_demand_valid": int(sample.axes.demand_valid[axis]),
                    f"axis_{axis}_measured_valid": int(sample.axes.measured_valid[axis]),
                    f"control_{axis}_m": sample.axes.control_m[axis],
                })
            for index in range(KEYPOINT_COUNT):
                kp = sample.keypoints
                covariance = kp.covariance_m2[index]
                sigma_mm = (
                    np.sqrt(np.maximum(np.diag(covariance), 0.0)) * 1000.0
                    if np.isfinite(covariance).all() else np.full(3, np.nan)
                )
                disparity = kp.left_pixels[index, 0] - kp.right_pixels[index, 0]
                row.update({
                    f"kp{index}_valid": int(kp.valid[index]),
                    f"kp{index}_predicted": int(kp.predicted[index]),
                    f"kp{index}_confidence": kp.confidence[index],
                    f"kp{index}_source": kp.source[index],
                    f"kp{index}_color_confidence": kp.color_confidence[index],
                    f"kp{index}_color_u_px": kp.color_pixels[index, 0],
                    f"kp{index}_color_v_px": kp.color_pixels[index, 1],
                    f"kp{index}_left_u_px": kp.left_pixels[index, 0],
                    f"kp{index}_left_v_px": kp.left_pixels[index, 1],
                    f"kp{index}_right_u_px": kp.right_pixels[index, 0],
                    f"kp{index}_right_v_px": kp.right_pixels[index, 1],
                    f"kp{index}_disparity_px": disparity,
                    f"kp{index}_sigma_x_mm": sigma_mm[0],
                    f"kp{index}_sigma_y_mm": sigma_mm[1],
                    f"kp{index}_sigma_z_mm": sigma_mm[2],
                    f"kp{index}_side_u_px": kp.side_pixels[index, 0],
                    f"kp{index}_side_v_px": kp.side_pixels[index, 1],
                    f"kp{index}_side_reprojection_error_px": kp.side_reprojection_error_px[index],
                    f"kp{index}_raw_cam_x_m": kp.raw_camera_m[index, 0],
                    f"kp{index}_raw_cam_y_m": kp.raw_camera_m[index, 1],
                    f"kp{index}_raw_cam_z_m": kp.raw_camera_m[index, 2],
                    f"kp{index}_filtered_cam_x_m": kp.filtered_camera_m[index, 0],
                    f"kp{index}_filtered_cam_y_m": kp.filtered_camera_m[index, 1],
                    f"kp{index}_filtered_cam_z_m": kp.filtered_camera_m[index, 2],
                    f"kp{index}_base_x_m": kp.base_m[index, 0],
                    f"kp{index}_base_y_m": kp.base_m[index, 1],
                    f"kp{index}_base_z_m": kp.base_m[index, 2],
                    f"kp{index}_sim_x_m": sample.simulation_points_base_m[index, 0],
                    f"kp{index}_sim_y_m": sample.simulation_points_base_m[index, 1],
                    f"kp{index}_sim_z_m": sample.simulation_points_base_m[index, 2],
                    f"kp{index}_error_mm": sample.errors_mm[index],
                })
            self.csv_writer.writerow(row)
            self.frame_count += 1
            if self.frame_count % int(self.config["recording"]["flush_interval_frames"]) == 0:
                self.csv_file.flush()
            if self.video_queue is not None:
                packet = (
                    sample,
                    None if overlay_bgr is None else overlay_bgr.copy(),
                    None if side_overlay_bgr is None else side_overlay_bgr.copy(),
                )
                try:
                    self.video_queue.put_nowait(packet)
                except queue.Full:
                    self.video_dropped_packets += 1
            for name, value in (
                ("capture_host_ns", sample.frame.capture_host_ns),
                ("device_timestamp_ms", sample.frame.device_timestamp_ms),
                ("axis_demand_native", sample.axes.demand_native),
                ("axis_measured_native", sample.axes.measured_native),
                ("controls_m", sample.axes.control_m),
                ("keypoints_raw_camera_m", sample.keypoints.raw_camera_m),
                ("keypoints_filtered_camera_m", sample.keypoints.filtered_camera_m),
                ("keypoints_base_m", sample.keypoints.base_m),
                ("keypoint_valid", sample.keypoints.valid),
                ("keypoint_confidence", sample.keypoints.confidence),
                ("keypoint_predicted", sample.keypoints.predicted),
                ("keypoint_covariance_m2", sample.keypoints.covariance_m2),
                ("keypoint_color_pixels", sample.keypoints.color_pixels),
                ("keypoint_left_pixels", sample.keypoints.left_pixels),
                ("keypoint_right_pixels", sample.keypoints.right_pixels),
                ("keypoint_source", np.asarray(sample.keypoints.source, dtype="U64")),
                ("keypoint_color_confidence", sample.keypoints.color_confidence),
                ("d435_tracking_roi", sample.d435_tracking_roi),
                ("side_tracking_roi", sample.side_tracking_roi),
                ("simulation_points_base_m", sample.simulation_points_base_m),
                ("errors_mm", sample.errors_mm),
                ("rmse_mm", sample.rmse_mm),
                ("side_capture_host_ns", sample.side_frame.capture_host_ns if sample.side_frame is not None else -1),
                ("side_pixels", sample.keypoints.side_pixels),
                ("side_reprojection_error_px", sample.keypoints.side_reprojection_error_px),
                ("fusion_mode", sample.fusion_mode),
            ):
                self.buffers[name].append(np.asarray(value).copy())

    def stop(self) -> Path | None:
        with self.lock:
            if not self.active:
                return self.session_dir
            if self.csv_file is not None:
                self.csv_file.flush()
                self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            if self.video_queue is not None:
                self.video_queue.put(None)
            if self.video_thread is not None:
                self.video_thread.join(timeout=15.0)
                if self.video_thread.is_alive() and self.video_error is None:
                    self.video_error = "视频写入线程在15秒内未完成"
                self.video_thread = None
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            if self.side_video_writer is not None:
                self.side_video_writer.release()
                self.side_video_writer = None
            for name in (
                "raw_rgb_writer", "synchronized_side_writer", "synchronized_pair_writer"
            ):
                writer = getattr(self, name)
                if writer is not None:
                    writer.release()
                    setattr(self, name, None)
            if self.video_timestamps_file is not None:
                self.video_timestamps_file.flush()
                self.video_timestamps_file.close()
                self.video_timestamps_file = None
                self.video_timestamps_writer = None
            arrays = {name: np.asarray(values) for name, values in self.buffers.items()}
            np.savez_compressed(self.session_dir / "keypoints.npz", **arrays)
            finite_rmse = arrays["rmse_mm"][np.isfinite(arrays["rmse_mm"])] if self.frame_count else np.asarray([])
            summary = {
                "frames": self.frame_count,
                "valid_fraction": float(np.mean(arrays["keypoint_valid"])) if self.frame_count else 0.0,
                "median_rmse_mm": float(np.median(finite_rmse)) if len(finite_rmse) else None,
                "p95_rmse_mm": float(np.percentile(finite_rmse, 95)) if len(finite_rmse) else None,
                "synchronized_video_frames": self.video_frame_count,
                "video_dropped_packets": self.video_dropped_packets,
                "video_error": self.video_error,
            }
            self._write_json("summary.json", summary)
            self.video_queue = None
            return self.session_dir


class _LegacyViewerStreamRecorder:
    """Low-overhead RGB/depth/3D recorder for the pure acquisition workflow.

    Camera frames and the seven-axis sample interpolated at exposure time are
    queued together.  Each selected viewport is encoded into its own MP4 file;
    recording never runs on the Qt UI thread.
    """

    STREAM_FILES = {
        "rgb": "rgb.mp4",
        "depth": "depth.mp4",
        "pointcloud": "pointcloud_3d.mp4",
    }

    def __init__(self, config: dict) -> None:
        self.config = config
        self.session_dir: Path | None = None
        self.selected: set[str] = set()
        self.fps: dict[str, float] = {}
        self.queue: queue.Queue | None = None
        self.thread: threading.Thread | None = None
        self.capture_thread: threading.Thread | None = None
        self.capture_stop = threading.Event()
        self.camera_source = None
        self.axis_sampler = None
        self.writers: dict[str, cv2.VideoWriter] = {}
        self.axes_file = None
        self.axes_writer = None
        self.frame_counts = {name: 0 for name in self.STREAM_FILES}
        self.output_frame_counts = {name: 0 for name in self.STREAM_FILES}
        self.duplicated_frame_counts = {name: 0 for name in self.STREAM_FILES}
        self.last_output_indices = {name: -1 for name in self.STREAM_FILES}
        self.last_images: dict[str, np.ndarray] = {}
        self.writer_codecs: dict[str, str] = {}
        self.epoch_capture_ns: int | None = None
        self.last_camera_capture_ns: int | None = None
        self.dropped_packets = 0
        self.error: str | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.queue is not None

    @staticmethod
    def _root_path(config: dict, root: str | Path | None) -> Path:
        target = Path(root if root is not None else config["recording"]["root"])
        return (PROJECT_ROOT / target).resolve() if not target.is_absolute() else target.resolve()

    def start(
        self,
        selected: set[str],
        *,
        root: str | Path | None = None,
        camera_info: dict[str, Any] | None = None,
        camera_source=None,
        axis_sampler=None,
    ) -> Path:
        with self._lock:
            if self.active:
                raise RuntimeError("纯采集录像已经在运行")
            selected = set(selected) & set(self.STREAM_FILES)
            if not selected:
                raise ValueError("请至少选择一个需要保存的窗口")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.session_dir = self._root_path(self.config, root) / f"capture_{stamp}"
            self.session_dir.mkdir(parents=True, exist_ok=False)
            self.selected = selected
            camera_config = self.config["camera"]
            output_fps = float(
                self.config["recording"].get(
                    "viewer_output_fps", camera_config["color"][2]
                )
            )
            self.fps = {
                "rgb": output_fps,
                "depth": output_fps,
                "pointcloud": output_fps,
            }
            self.frame_counts = {name: 0 for name in self.STREAM_FILES}
            self.output_frame_counts = {name: 0 for name in self.STREAM_FILES}
            self.duplicated_frame_counts = {name: 0 for name in self.STREAM_FILES}
            self.last_output_indices = {name: -1 for name in self.STREAM_FILES}
            self.last_images = {}
            self.writer_codecs = {}
            self.epoch_capture_ns = None
            self.last_camera_capture_ns = None
            self.dropped_packets = 0
            self.error = None
            self.camera_source = camera_source
            self.axis_sampler = axis_sampler
            self.axes_file = (self.session_dir / "camera_axes.csv").open(
                "w", newline="", encoding="utf-8-sig"
            )
            fields = [
                "sequence", "video_frame_index", "capture_host_ns", "device_timestamp_ms",
                "color_timestamp_ms", "depth_timestamp_ms", "axis_status",
            ]
            for axis in range(7):
                fields.extend([
                    f"axis_{axis}_demand_native", f"axis_{axis}_measured_native",
                    f"axis_{axis}_demand_valid", f"axis_{axis}_measured_valid",
                    f"control_{axis}_m",
                ])
            self.axes_writer = csv.DictWriter(self.axes_file, fieldnames=fields)
            self.axes_writer.writeheader()
            manifest = {
                "schema_version": 2,
                "workflow": "capture",
                "created_local": datetime.now().isoformat(),
                "selected_windows": sorted(selected),
                "video_files": {
                    name: self.STREAM_FILES[name] for name in sorted(selected)
                },
                "nominal_fps": self.fps,
                "source_fps": {
                    "rgb": float(camera_config["color"][2]),
                    "depth": float(camera_config["depth"][2]),
                    "pointcloud": float(
                        self.config.get("ui", {}).get("point_cloud_fps", 15.0)
                    ),
                },
                "timebase": "camera_capture_host_ns",
                "codec_requested": self.config["recording"].get("viewer_codec", "avc1"),
                "camera": camera_info or {},
                "camera_config": camera_config,
                "axis_clock": "interpolated_at_camera_capture_host_ns",
            }
            with (self.session_dir / "capture.json").open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, ensure_ascii=False)
            self.queue = queue.Queue(maxsize=int(self.config["recording"].get("viewer_queue_frames", 90)))
            self.thread = threading.Thread(
                target=self._loop, name="realsense-viewer-recorder", daemon=True
            )
            self.thread.start()
            self.capture_stop.clear()
            if camera_source is not None and callable(getattr(camera_source, "peek_latest", None)):
                self.capture_thread = threading.Thread(
                    target=self._capture_loop,
                    name="realsense-viewer-capture-feed",
                    daemon=True,
                )
                self.capture_thread.start()
            return self.session_dir

    @staticmethod
    def _depth_visual(depth_m: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
        clipped = np.nan_to_num(
            depth, nan=minimum, posinf=maximum, neginf=minimum
        ).copy()
        np.clip(clipped, minimum, maximum, out=clipped)
        scale = 255.0 / max(maximum - minimum, 1e-6)
        normalized = cv2.convertScaleAbs(clipped, alpha=scale, beta=-minimum * scale)
        visual = cv2.applyColorMap(cv2.subtract(255, normalized), cv2.COLORMAP_TURBO)
        visual[~valid] = (12, 13, 16)
        return visual

    def _writer(self, name: str, image: np.ndarray) -> cv2.VideoWriter:
        writer = self.writers.get(name)
        if writer is not None:
            return writer
        height, width = image.shape[:2]
        path = self.session_dir / self.STREAM_FILES[name]
        requested = str(self.config["recording"].get("viewer_codec", "avc1"))
        fallback = str(self.config["recording"].get("viewer_codec_fallback", "mp4v"))
        errors = []
        for codec in dict.fromkeys((requested, fallback, "mp4v")):
            if len(codec) != 4:
                errors.append(f"{codec}: FourCC必须是4个字符")
                continue
            if os.name == "nt" and codec.lower() in ("avc1", "h264"):
                candidate = cv2.VideoWriter(
                    str(path),
                    cv2.CAP_MSMF,
                    cv2.VideoWriter_fourcc(*codec),
                    float(self.fps[name]),
                    (int(width), int(height)),
                )
            else:
                candidate = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*codec),
                    float(self.fps[name]),
                    (int(width), int(height)),
                )
            if candidate.isOpened():
                self.writers[name] = candidate
                self.writer_codecs[name] = codec
                return candidate
            candidate.release()
            errors.append(f"{codec}: unavailable")
        raise RuntimeError(f"无法创建{name}视频：{' | '.join(errors)}")

    def _target_frame_index(self, capture_host_ns: int, name: str) -> int:
        if self.epoch_capture_ns is None:
            self.epoch_capture_ns = int(capture_host_ns)
        elapsed_s = max(0.0, (int(capture_host_ns) - self.epoch_capture_ns) * 1e-9)
        return max(0, int(round(elapsed_s * self.fps[name])))

    def _write_timed_frame(
        self, name: str, image: np.ndarray, capture_host_ns: int
    ) -> None:
        image = np.ascontiguousarray(image, dtype=np.uint8)
        target_index = self._target_frame_index(capture_host_ns, name)
        last_index = self.last_output_indices[name]
        if target_index <= last_index:
            # A higher-rate source may map twice to the same constant-rate MP4
            # slot. Keep the newest image for future gap filling.
            self.last_images[name] = image.copy()
            return
        writer = self._writer(name, image)
        fill_image = self.last_images.get(name, image)
        while last_index + 1 < target_index:
            writer.write(fill_image)
            last_index += 1
            self.output_frame_counts[name] += 1
            self.duplicated_frame_counts[name] += 1
        writer.write(image)
        last_index += 1
        self.output_frame_counts[name] += 1
        self.last_output_indices[name] = last_index
        self.last_images[name] = image.copy()

    def _pad_streams_to_common_end(self) -> None:
        if self.epoch_capture_ns is None or self.last_camera_capture_ns is None:
            return
        for name in self.selected:
            image = self.last_images.get(name)
            writer = self.writers.get(name)
            if image is None or writer is None:
                continue
            target_index = self._target_frame_index(self.last_camera_capture_ns, name)
            last_index = self.last_output_indices[name]
            while last_index < target_index:
                writer.write(image)
                last_index += 1
                self.output_frame_counts[name] += 1
                self.duplicated_frame_counts[name] += 1
            self.last_output_indices[name] = last_index

    def _write_axes(self, packet: dict[str, Any]) -> None:
        axes = packet["axes"]
        row = {
            "sequence": packet["sequence"],
            "video_frame_index": self._target_frame_index(
                packet["capture_host_ns"], "rgb"
            ),
            "capture_host_ns": packet["capture_host_ns"],
            "device_timestamp_ms": packet["device_timestamp_ms"],
            "color_timestamp_ms": packet["color_timestamp_ms"],
            "depth_timestamp_ms": packet["depth_timestamp_ms"],
            "axis_status": axes.status,
        }
        for axis in range(7):
            row.update({
                f"axis_{axis}_demand_native": axes.demand_native[axis],
                f"axis_{axis}_measured_native": axes.measured_native[axis],
                f"axis_{axis}_demand_valid": int(axes.demand_valid[axis]),
                f"axis_{axis}_measured_valid": int(axes.measured_valid[axis]),
                f"control_{axis}_m": axes.control_m[axis],
            })
        self.axes_writer.writerow(row)

    def _loop(self) -> None:
        minimum = float(self.config["camera"].get("depth_min_m", 0.18))
        maximum = float(self.config["camera"].get("depth_max_m", 0.60))
        while self.queue is not None:
            packet = self.queue.get()
            if packet is None:
                self.queue.task_done()
                return
            try:
                kind = packet["kind"]
                if kind == "camera":
                    if self.epoch_capture_ns is None:
                        self.epoch_capture_ns = int(packet["capture_host_ns"])
                    self.last_camera_capture_ns = int(packet["capture_host_ns"])
                    self._write_axes(packet)
                    if packet.get("rgb") is not None:
                        self.frame_counts["rgb"] += 1
                        self._write_timed_frame(
                            "rgb", packet["rgb"], packet["capture_host_ns"]
                        )
                    if packet.get("depth") is not None:
                        visual = self._depth_visual(packet["depth"], minimum, maximum)
                        self.frame_counts["depth"] += 1
                        self._write_timed_frame(
                            "depth", visual, packet["capture_host_ns"]
                        )
                elif kind == "pointcloud" and "pointcloud" in self.selected:
                    visual = packet["image"]
                    self.frame_counts["pointcloud"] += 1
                    self._write_timed_frame(
                        "pointcloud", visual, packet["capture_host_ns"]
                    )
            except Exception as exc:
                if self.error is None:
                    self.error = str(exc)
            finally:
                self.queue.task_done()

    def _enqueue(self, packet: dict[str, Any]) -> None:
        current = self.queue
        if current is None:
            return
        try:
            current.put_nowait(packet)
        except queue.Full:
            self.dropped_packets += 1

    def _capture_loop(self) -> None:
        last_sequence = -1
        while not self.capture_stop.is_set():
            frame = self.camera_source.peek_latest()
            if frame is None or int(frame.sequence) == last_sequence:
                self.capture_stop.wait(0.001)
                continue
            last_sequence = int(frame.sequence)
            if self.axis_sampler is not None:
                axes = self.axis_sampler.interpolate(frame.capture_host_ns)
            else:
                self.capture_stop.wait(0.001)
                continue
            self.append_camera(frame, axes)

    def append_camera(self, frame, axes) -> None:
        if not self.active:
            return
        self._enqueue({
            "kind": "camera",
            "sequence": int(frame.sequence),
            "capture_host_ns": int(frame.capture_host_ns),
            "device_timestamp_ms": float(frame.device_timestamp_ms),
            "color_timestamp_ms": float(frame.color_timestamp_ms),
            "depth_timestamp_ms": float(frame.depth_timestamp_ms),
            "axes": axes,
            "rgb": frame.color_bgr.copy() if "rgb" in self.selected else None,
            "depth": frame.depth_m.copy() if "depth" in self.selected else None,
        })

    def append_pointcloud(
        self, image_bgr: np.ndarray, capture_host_ns: int
    ) -> None:
        if self.active and "pointcloud" in self.selected:
            self._enqueue({
                "kind": "pointcloud",
                "capture_host_ns": int(capture_host_ns),
                "image": np.asarray(image_bgr).copy(),
            })

    def stop(self) -> Path | None:
        with self._lock:
            if not self.active:
                return self.session_dir
            self.capture_stop.set()
            if self.capture_thread is not None:
                self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
            current = self.queue
            current.put(None)
            if self.thread is not None:
                self.thread.join(timeout=20.0)
                if self.thread.is_alive() and self.error is None:
                    self.error = "视频写入线程未能在20秒内结束"
            self.thread = None
            self._pad_streams_to_common_end()
            for writer in self.writers.values():
                writer.release()
            self.writers.clear()
            if self.axes_file is not None:
                self.axes_file.flush()
                self.axes_file.close()
            self.axes_file = None
            self.axes_writer = None
            summary = {
                "source_frames": self.frame_counts,
                "output_frames": self.output_frame_counts,
                "duplicated_frames": self.duplicated_frame_counts,
                "codec": self.writer_codecs,
                "duration_s": {
                    name: (
                        self.output_frame_counts[name] / self.fps[name]
                        if self.output_frame_counts[name]
                        else 0.0
                    )
                    for name in self.STREAM_FILES
                },
                "dropped_packets": self.dropped_packets,
                "error": self.error,
            }
            with (self.session_dir / "summary.json").open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, ensure_ascii=False)
            self.queue = None
            self.camera_source = None
            self.axis_sampler = None
            return self.session_dir


class AuxiliaryRgbRecorder:
    """Record one independently threaded RGB source into an existing session."""

    VIDEO_FILE = "endoscope_rgb.mp4"
    TIMESTAMP_FILE = "endoscope_timestamps.csv"
    SUMMARY_FILE = "endoscope_summary.json"

    def __init__(self, config: dict) -> None:
        self.config = config
        self.session_dir: Path | None = None
        self.source = None
        self._accepting = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = None
        self._ready_queue = None
        self._free_queue = None
        self._result_queue = None
        self._shared_block: shared_memory.SharedMemory | None = None
        self._shape: tuple[int, int, int] = (0, 0, 3)
        self._slot_bytes = 0
        self.frame_count = 0
        self.dropped_count = 0
        self.first_capture_ns: int | None = None
        self.last_capture_ns: int | None = None
        self.codec = ""
        self.error: str | None = None
        self.source_device_info: dict[str, Any] = {}

    @property
    def active(self) -> bool:
        return self._accepting

    def start(self, session_dir: str | Path, source) -> Path:
        if self.active:
            raise RuntimeError("内窥镜录像已经在运行")
        if source is None or not bool(getattr(source, "running", False)):
            raise RuntimeError("内窥镜摄像头尚未启动")
        self.session_dir = Path(session_dir).resolve()
        self.source = source
        self.source_device_info = dict(getattr(source, "device_info", {}) or {})
        settings = self.config["endoscope_camera"]
        width = int(self.source_device_info.get("width") or settings.get("width", 1280))
        height = int(self.source_device_info.get("height") or settings.get("height", 720))
        fps = float(self.source_device_info.get("fps") or settings.get("fps", 30.0))
        if width <= 0 or height <= 0 or fps <= 0:
            raise RuntimeError("内窥镜未返回有效的分辨率或帧率")
        slots = max(4, int(self.config.get("recording", {}).get("viewer_shared_slots", 6)))
        self._shape = (height, width, 3)
        self._slot_bytes = int(np.prod(self._shape))
        context = mp.get_context("spawn")
        self._shared_block = shared_memory.SharedMemory(
            create=True, size=self._slot_bytes * slots
        )
        self._ready_queue = context.Queue(maxsize=slots)
        self._free_queue = context.Queue(maxsize=slots)
        self._result_queue = context.Queue(maxsize=2)
        for slot in range(slots):
            self._free_queue.put(slot)
        ready_event = context.Event()
        self._process = context.Process(
            target=_viewer_video_process,
            args=(
                "endoscope",
                str(self.session_dir / self.VIDEO_FILE),
                fps,
                (
                    str(self.config["recording"].get("viewer_codec", "mp4v")),
                    str(self.config["recording"].get("viewer_codec_fallback", "avc1")),
                ),
                0.0,
                1.0,
                1.0,
                self._ready_queue,
                self._result_queue,
                self._shared_block.name,
                self._shape,
                np.dtype(np.uint8).str,
                self._slot_bytes,
                self._free_queue,
                ready_event,
            ),
            name="endoscope-video-encoder",
            daemon=True,
        )
        self.frame_count = 0
        self.dropped_count = 0
        self.first_capture_ns = None
        self.last_capture_ns = None
        self.codec = ""
        self.error = None
        self._stop.clear()
        self._accepting = True
        self._process.start()
        if not ready_event.wait(timeout=5.0):
            self.error = "内窥镜视频编码进程启动超时"
            self._accepting = False
            self._stop.set()
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            raise RuntimeError(self.error)
        self._thread = threading.Thread(
            target=self._feed_loop, name="endoscope-record-feed", daemon=True
        )
        self._thread.start()
        return self.session_dir / self.VIDEO_FILE

    def _feed_loop(self) -> None:
        timestamp_path = self.session_dir / self.TIMESTAMP_FILE
        last_sequence = -1
        try:
            with timestamp_path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "video_frame_index",
                        "source_sequence",
                        "capture_host_ns",
                        "host_arrival_ns",
                        "elapsed_s",
                    ),
                )
                writer.writeheader()
                while not self._stop.is_set():
                    frame = self.source.poll() if self.source is not None else None
                    if frame is None or int(frame.sequence) == last_sequence:
                        self._stop.wait(0.001)
                        continue
                    last_sequence = int(frame.sequence)
                    try:
                        slot = self._free_queue.get_nowait()
                    except queue.Empty:
                        self.dropped_count += 1
                        continue
                    try:
                        target = np.ndarray(
                            self._shape,
                            dtype=np.uint8,
                            buffer=self._shared_block.buf,
                            offset=int(slot) * self._slot_bytes,
                        )
                        image = np.asarray(frame.image_bgr, dtype=np.uint8)
                        if image.shape != self._shape:
                            image = cv2.resize(
                                image,
                                (self._shape[1], self._shape[0]),
                                interpolation=cv2.INTER_AREA,
                            )
                        np.copyto(target, image, casting="unsafe")
                        self._ready_queue.put_nowait(int(slot))
                    except Exception:
                        self._free_queue.put(int(slot))
                        raise
                    capture_ns = int(frame.capture_host_ns)
                    if self.first_capture_ns is None:
                        self.first_capture_ns = capture_ns
                    self.last_capture_ns = capture_ns
                    writer.writerow({
                        "video_frame_index": self.frame_count,
                        "source_sequence": int(frame.sequence),
                        "capture_host_ns": capture_ns,
                        "host_arrival_ns": int(frame.host_arrival_ns),
                        "elapsed_s": (capture_ns - self.first_capture_ns) * 1e-9,
                    })
                    self.frame_count += 1
                    if self.frame_count % 60 == 0:
                        stream.flush()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> Path | None:
        if not self.active:
            return self.session_dir
        self._accepting = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        if self._ready_queue is not None:
            self._ready_queue.put(None)
        encoded = 0
        if self._process is not None:
            self._process.join(timeout=20.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
                self.error = self.error or "内窥镜视频编码进程未能正常结束"
            try:
                _name, encoded, codec, process_error = self._result_queue.get(timeout=1.0)
                self.codec = str(codec)
                if process_error:
                    self.error = self.error or str(process_error)
            except queue.Empty:
                self.error = self.error or "内窥镜视频编码结果未返回"
            self._process.close()
        video_path = self.session_dir / self.VIDEO_FILE
        opened, decoded_frames, decoded_fps = ViewerStreamRecorder._decoded_video_info(video_path)
        if not opened or decoded_frames <= 0:
            self.error = self.error or "内窥镜MP4完整性校验失败"
        elif int(encoded) != int(self.frame_count):
            self.error = self.error or (
                f"内窥镜帧数不一致：提交{self.frame_count}帧，编码{encoded}帧"
            )
        elif int(decoded_frames) != int(encoded):
            self.error = self.error or (
                f"内窥镜MP4帧数不一致：编码{encoded}帧，解码{decoded_frames}帧"
            )
        duration_s = 0.0
        if (
            self.first_capture_ns is not None and self.last_capture_ns is not None
            and self.last_capture_ns > self.first_capture_ns
        ):
            duration_s = (self.last_capture_ns - self.first_capture_ns) * 1e-9
        summary = {
            "video_file": self.VIDEO_FILE,
            "timestamp_file": self.TIMESTAMP_FILE,
            "device": self.source_device_info,
            "source_frames": self.frame_count,
            "encoded_frames": int(encoded),
            "dropped_frames": self.dropped_count,
            "source_rate_hz": (
                (self.frame_count - 1) / duration_s
                if self.frame_count >= 2 and duration_s > 0 else 0.0
            ),
            "codec": self.codec,
            "decoded_validation": {
                "opened": opened,
                "frames": decoded_frames,
                "fps": decoded_fps,
            },
            "error": self.error,
        }
        (self.session_dir / self.SUMMARY_FILE).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for output_queue in (self._ready_queue, self._free_queue, self._result_queue):
            if output_queue is not None:
                output_queue.close()
                output_queue.join_thread()
        if self._shared_block is not None:
            self._shared_block.close()
            try:
                self._shared_block.unlink()
            except FileNotFoundError:
                pass
        self._ready_queue = self._free_queue = self._result_queue = None
        self._shared_block = None
        self._process = None
        self.source = None
        return self.session_dir


class ViewerStreamRecorder:
    """Reliable multi-stream recorder with independent video and motor paths.

    Every camera frame is encoded at most once. RGB, depth and point-cloud
    streams own separate queues, threads and VideoWriter instances so depth
    colourisation cannot stall RGB. Motor samples are recorded directly from
    ``AxisSampler`` at its native rate (100 Hz by default), while
    ``camera_axes.csv`` keeps the exposure-time interpolation needed for sync.
    """

    STREAM_FILES = {
        "rgb": "rgb.mp4",
        "depth": "depth.mp4",
        "pointcloud": "pointcloud_3d.mp4",
    }

    def __init__(self, config: dict) -> None:
        self.config = config
        self.session_dir: Path | None = None
        self.selected: set[str] = set()
        self.fps: dict[str, float] = {}
        self.video_queues: dict[str, Any] = {}
        self.video_threads: dict[str, Any] = {}
        self.video_result_queue = None
        self.video_ready_events: dict[str, Any] = {}
        self.video_free_queues: dict[str, Any] = {}
        self.video_shared_blocks: dict[str, shared_memory.SharedMemory] = {}
        self.video_shared_specs: dict[str, tuple[tuple[int, ...], np.dtype, int]] = {}
        self.data_queue: queue.Queue | None = None
        self.data_thread: threading.Thread | None = None
        self.capture_thread: threading.Thread | None = None
        self.capture_stop = threading.Event()
        self.camera_source = None
        self.axis_sampler = None
        self._axis_listener = None
        self._accepting = False
        self._lock = threading.Lock()
        self.source_counts = {name: 0 for name in self.STREAM_FILES}
        self.encoded_counts = {name: 0 for name in self.STREAM_FILES}
        self.dropped_counts = {name: 0 for name in self.STREAM_FILES}
        self.writer_codecs: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self.camera_axis_rows = 0
        self.camera_first_ns: int | None = None
        self.camera_last_ns: int | None = None
        self.motor_sample_count = 0
        self.motor_first_ns: int | None = None
        self.motor_last_ns: int | None = None
        self.metadata_dropped = 0

    @property
    def active(self) -> bool:
        return self._accepting

    @property
    def error(self) -> str | None:
        return " | ".join(f"{key}: {value}" for key, value in self.errors.items()) or None

    @property
    def dropped_packets(self) -> int:
        return int(sum(self.dropped_counts.values()) + self.metadata_dropped)

    @property
    def output_frame_counts(self) -> dict[str, int]:
        """Backward-compatible name used by the capture result UI."""

        return self.encoded_counts

    @staticmethod
    def _root_path(config: dict, root: str | Path | None) -> Path:
        target = Path(root if root is not None else config["recording"]["root"])
        return (PROJECT_ROOT / target).resolve() if not target.is_absolute() else target.resolve()

    @staticmethod
    def _depth_visual(depth_m: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
        return _viewer_depth_visual(depth_m, minimum, maximum)

    def _open_video_writer(self, name: str, image: np.ndarray) -> cv2.VideoWriter:
        height, width = image.shape[:2]
        path = self.session_dir / self.STREAM_FILES[name]
        requested = str(self.config["recording"].get("viewer_codec", "mp4v"))
        fallback = str(self.config["recording"].get("viewer_codec_fallback", "avc1"))
        failures: list[str] = []
        for codec in dict.fromkeys((requested, fallback, "mp4v")):
            if len(codec) != 4:
                failures.append(f"{codec}: invalid FourCC")
                continue
            fourcc = cv2.VideoWriter_fourcc(*codec)
            backends = (
                (cv2.CAP_FFMPEG, cv2.CAP_ANY)
                if hasattr(cv2, "CAP_FFMPEG")
                else (cv2.CAP_ANY,)
            )
            for backend in backends:
                writer = cv2.VideoWriter(
                    str(path),
                    int(backend),
                    fourcc,
                    float(self.fps[name]),
                    (int(width), int(height)),
                )
                if writer.isOpened():
                    self.writer_codecs[name] = codec
                    return writer
                writer.release()
            failures.append(f"{codec}: unavailable")
        raise RuntimeError("; ".join(failures))

    def start(
        self,
        selected: set[str],
        *,
        root: str | Path | None = None,
        camera_info: dict[str, Any] | None = None,
        camera_source=None,
        axis_sampler=None,
    ) -> Path:
        with self._lock:
            if self.active:
                raise RuntimeError("纯采集录像已经在运行")
            selected = set(selected) & set(self.STREAM_FILES)
            if not selected:
                raise ValueError("请至少选择一个需要保存的窗口")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.session_dir = self._root_path(self.config, root) / f"capture_{stamp}"
            self.session_dir.mkdir(parents=True, exist_ok=False)
            self.selected = selected
            camera = self.config["camera"]
            self.fps = {
                "rgb": float(camera["color"][2]),
                "depth": float(camera["depth"][2]),
                "pointcloud": float(self.config.get("ui", {}).get("point_cloud_fps", 15.0)),
            }
            self.source_counts = {name: 0 for name in self.STREAM_FILES}
            self.encoded_counts = {name: 0 for name in self.STREAM_FILES}
            self.dropped_counts = {name: 0 for name in self.STREAM_FILES}
            self.writer_codecs = {}
            self.errors = {}
            self.camera_axis_rows = 0
            self.camera_first_ns = None
            self.camera_last_ns = None
            self.motor_sample_count = 0
            self.motor_first_ns = None
            self.motor_last_ns = None
            self.metadata_dropped = 0
            self.camera_source = camera_source
            self.axis_sampler = axis_sampler
            queue_size = max(10, int(self.config["recording"].get("viewer_queue_frames", 90)))
            context = mp.get_context("spawn")
            self.video_result_queue = context.Queue()
            self.video_ready_events = {}
            self.video_queues = {}
            self.video_free_queues = {}
            self.video_shared_blocks = {}
            self.video_shared_specs = {}
            shared_slots = max(
                4, int(self.config["recording"].get("viewer_shared_slots", 12))
            )
            for name in selected:
                if name == "rgb":
                    width, height = map(int, camera["color"][:2])
                    shape, dtype = (height, width, 3), np.dtype(np.uint8)
                elif name == "depth":
                    width, height = map(int, camera["depth"][:2])
                    shape, dtype = (height, width), np.dtype(np.uint16)
                else:
                    self.video_queues[name] = context.Queue(maxsize=queue_size)
                    continue
                slot_bytes = int(np.prod(shape)) * dtype.itemsize
                block = shared_memory.SharedMemory(
                    create=True, size=slot_bytes * shared_slots
                )
                ready = context.Queue(maxsize=shared_slots)
                free = context.Queue(maxsize=shared_slots)
                for slot in range(shared_slots):
                    free.put(slot)
                self.video_queues[name] = ready
                self.video_free_queues[name] = free
                self.video_shared_blocks[name] = block
                self.video_shared_specs[name] = (shape, dtype, slot_bytes)
            self.data_queue = queue.Queue(
                maxsize=max(1000, int(self.config["recording"].get("metadata_queue_samples", 20000)))
            )
            manifest = {
                "schema_version": 3,
                "workflow": "capture",
                "created_local": datetime.now().isoformat(),
                "selected_windows": sorted(selected),
                "video_files": {name: self.STREAM_FILES[name] for name in sorted(selected)},
                "nominal_fps": self.fps,
                "codec_requested": self.config["recording"].get("viewer_codec", "mp4v"),
                "camera": camera_info or {},
                "camera_config": camera,
                "camera_axis_file": "camera_axes.csv",
                "raw_motor_file": "motor_axes_100hz.csv",
                "raw_motor_target_hz": float(self.config["axes"].get("sample_rate_hz", 100.0)),
            }
            with (self.session_dir / "capture.json").open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, ensure_ascii=False)
            self._accepting = True
            self.data_thread = threading.Thread(
                target=self._data_loop, name="capture-data-writer", daemon=True
            )
            self.data_thread.start()
            for name in sorted(selected):
                # Use a module-level process target so Windows spawn never
                # pickles the recorder, its locks, or active camera handles.
                ready_event = context.Event()
                process = context.Process(
                    target=_viewer_video_process,
                    args=(
                        name,
                        str(self.session_dir / self.STREAM_FILES[name]),
                        float(self.fps[name]),
                        (
                            str(self.config["recording"].get("viewer_codec", "mp4v")),
                            str(self.config["recording"].get("viewer_codec_fallback", "avc1")),
                        ),
                        float(camera.get("depth_min_m", 0.18)),
                        float(camera.get("depth_max_m", 0.60)),
                        float(camera.get("depth_unit_m", 0.001)),
                        self.video_queues[name],
                        self.video_result_queue,
                        self.video_shared_blocks[name].name
                        if name in self.video_shared_blocks else "",
                        self.video_shared_specs[name][0]
                        if name in self.video_shared_specs else (),
                        self.video_shared_specs[name][1].str
                        if name in self.video_shared_specs else "",
                        self.video_shared_specs[name][2]
                        if name in self.video_shared_specs else 0,
                        self.video_free_queues.get(name),
                        ready_event,
                    ),
                    name=f"capture-{name}-encoder",
                    daemon=True,
                )
                self.video_threads[name] = process
                self.video_ready_events[name] = ready_event
                process.start()
            for name, ready_event in self.video_ready_events.items():
                if not ready_event.wait(timeout=5.0):
                    self.errors.setdefault(name, "video encoder process did not become ready")
            if axis_sampler is not None and hasattr(axis_sampler, "add_listener"):
                self._axis_listener = self._on_axis_sample
                axis_sampler.add_listener(self._axis_listener)
            self.capture_stop.clear()
            if camera_source is not None and callable(getattr(camera_source, "peek_latest", None)):
                self.capture_thread = threading.Thread(
                    target=self._capture_loop,
                    name="realsense-capture-feed",
                    daemon=True,
                )
                self.capture_thread.start()
            return self.session_dir

    @staticmethod
    def _camera_fields() -> list[str]:
        fields = [
            "sequence", "capture_host_ns", "device_timestamp_ms",
            "color_timestamp_ms", "depth_timestamp_ms", "axis_status",
            "rgb_enqueued", "depth_enqueued",
        ]
        for axis in range(7):
            fields.extend([
                f"axis_{axis}_demand_native", f"axis_{axis}_measured_native",
                f"axis_{axis}_demand_valid", f"axis_{axis}_measured_valid",
                f"control_{axis}_m",
            ])
        return fields

    @staticmethod
    def _motor_fields() -> list[str]:
        fields = ["sample_index", "host_ns", "elapsed_s", "status"]
        for axis in range(7):
            fields.extend([
                f"axis_{axis}_demand_native", f"axis_{axis}_measured_native",
                f"axis_{axis}_demand_valid", f"axis_{axis}_measured_valid",
                f"control_{axis}_m",
            ])
        return fields

    @staticmethod
    def _axis_values(row: dict[str, Any], sample) -> None:
        for axis in range(7):
            row.update({
                f"axis_{axis}_demand_native": sample.demand_native[axis],
                f"axis_{axis}_measured_native": sample.measured_native[axis],
                f"axis_{axis}_demand_valid": int(sample.demand_valid[axis]),
                f"axis_{axis}_measured_valid": int(sample.measured_valid[axis]),
                f"control_{axis}_m": sample.control_m[axis],
            })

    def _data_loop(self) -> None:
        camera_path = self.session_dir / "camera_axes.csv"
        motor_path = self.session_dir / "motor_axes_100hz.csv"
        try:
            with camera_path.open("w", newline="", encoding="utf-8-sig") as camera_file, motor_path.open(
                "w", newline="", encoding="utf-8-sig"
            ) as motor_file:
                camera_writer = csv.DictWriter(camera_file, fieldnames=self._camera_fields())
                motor_writer = csv.DictWriter(motor_file, fieldnames=self._motor_fields())
                camera_writer.writeheader()
                motor_writer.writeheader()
                while True:
                    packet = self.data_queue.get()
                    try:
                        if packet is None:
                            return
                        if packet["kind"] == "camera":
                            frame, axes = packet["frame"], packet["axes"]
                            row = {
                                "sequence": frame.sequence,
                                "capture_host_ns": frame.capture_host_ns,
                                "device_timestamp_ms": frame.device_timestamp_ms,
                                "color_timestamp_ms": frame.color_timestamp_ms,
                                "depth_timestamp_ms": frame.depth_timestamp_ms,
                                "axis_status": axes.status,
                                "rgb_enqueued": int(packet["rgb_enqueued"]),
                                "depth_enqueued": int(packet["depth_enqueued"]),
                            }
                            self._axis_values(row, axes)
                            camera_writer.writerow(row)
                            self.camera_axis_rows += 1
                        else:
                            sample = packet["sample"]
                            if self.motor_first_ns is None:
                                self.motor_first_ns = int(sample.host_ns)
                            self.motor_last_ns = int(sample.host_ns)
                            row = {
                                "sample_index": self.motor_sample_count,
                                "host_ns": sample.host_ns,
                                "elapsed_s": (sample.host_ns - self.motor_first_ns) * 1e-9,
                                "status": sample.status,
                            }
                            self._axis_values(row, sample)
                            motor_writer.writerow(row)
                            self.motor_sample_count += 1
                        if (self.camera_axis_rows + self.motor_sample_count) % 100 == 0:
                            camera_file.flush()
                            motor_file.flush()
                    finally:
                        self.data_queue.task_done()
        except Exception as exc:
            self.errors.setdefault("data", str(exc))

    def _video_loop(self, name: str) -> None:
        writer = None
        minimum = float(self.config["camera"].get("depth_min_m", 0.18))
        maximum = float(self.config["camera"].get("depth_max_m", 0.60))
        stream_queue = self.video_queues[name]
        try:
            while True:
                packet = stream_queue.get()
                try:
                    if packet is None:
                        return
                    if name in self.errors:
                        continue
                    image = packet["image"]
                    if name == "depth":
                        image = self._depth_visual(image, minimum, maximum)
                    image = np.ascontiguousarray(image, dtype=np.uint8)
                    if writer is None:
                        writer = self._open_video_writer(name, image)
                    writer.write(image)
                    self.encoded_counts[name] += 1
                except Exception as exc:
                    self.errors.setdefault(name, str(exc))
                finally:
                    stream_queue.task_done()
        finally:
            if writer is not None:
                writer.release()

    def _enqueue_video(self, name: str, image: np.ndarray, capture_host_ns: int) -> bool:
        stream_queue = self.video_queues.get(name)
        if not self.active or stream_queue is None:
            return False
        self.source_counts[name] += 1
        if name in self.video_shared_blocks:
            free_queue = self.video_free_queues[name]
            try:
                slot = free_queue.get_nowait()
            except queue.Empty:
                self.dropped_counts[name] += 1
                return False
            try:
                shape, dtype, slot_bytes = self.video_shared_specs[name]
                target = np.ndarray(
                    shape,
                    dtype=dtype,
                    buffer=self.video_shared_blocks[name].buf,
                    offset=int(slot) * slot_bytes,
                )
                source = np.asarray(image, dtype=dtype)
                if source.shape != shape:
                    interpolation = cv2.INTER_NEAREST if name == "depth" else cv2.INTER_LINEAR
                    source = cv2.resize(
                        source, (int(shape[1]), int(shape[0])), interpolation=interpolation
                    )
                np.copyto(target, source, casting="unsafe")
                stream_queue.put_nowait(int(slot))
                return True
            except Exception:
                free_queue.put(int(slot))
                self.dropped_counts[name] += 1
                return False
        try:
            stream_queue.put_nowait({
                "capture_host_ns": int(capture_host_ns),
                "image": image,
            })
            return True
        except queue.Full:
            self.dropped_counts[name] += 1
            return False

    def _enqueue_data(self, packet: dict[str, Any]) -> None:
        if not self.active or self.data_queue is None:
            return
        try:
            self.data_queue.put_nowait(packet)
        except queue.Full:
            self.metadata_dropped += 1

    def _on_axis_sample(self, sample) -> None:
        self._enqueue_data({"kind": "motor", "sample": sample})

    def _capture_loop(self) -> None:
        initial = self.camera_source.peek_latest()
        last_sequence = int(initial.sequence) if initial is not None else -1
        while not self.capture_stop.is_set():
            frames_after = getattr(self.camera_source, "frames_after", None)
            if callable(frames_after):
                frames = frames_after(last_sequence)
            else:
                frame = self.camera_source.peek_latest()
                frames = [] if frame is None or int(frame.sequence) == last_sequence else [frame]
            if not frames:
                self.capture_stop.wait(0.001)
                continue
            for frame in frames:
                last_sequence = int(frame.sequence)
                axes = (
                    self.axis_sampler.interpolate(frame.capture_host_ns)
                    if self.axis_sampler is not None
                    else None
                )
                if axes is not None:
                    self.append_camera(frame, axes)

    def append_camera(self, frame, axes) -> None:
        if not self.active:
            return
        if self.camera_first_ns is None:
            self.camera_first_ns = int(frame.capture_host_ns)
        self.camera_last_ns = int(frame.capture_host_ns)
        rgb_ok = (
            self._enqueue_video("rgb", frame.color_bgr, frame.capture_host_ns)
            if "rgb" in self.selected else False
        )
        depth_ok = (
            self._enqueue_video(
                "depth",
                frame.raw_depth_z16
                if frame.raw_depth_z16 is not None
                else np.rint(frame.depth_m / max(float(frame.depth_unit_m), 1e-9)).astype(np.uint16),
                frame.capture_host_ns,
            )
            if "depth" in self.selected else False
        )
        self._enqueue_data({
            "kind": "camera",
            "frame": frame,
            "axes": axes,
            "rgb_enqueued": rgb_ok,
            "depth_enqueued": depth_ok,
        })

    def append_pointcloud(self, image_bgr: np.ndarray, capture_host_ns: int) -> None:
        if "pointcloud" in self.selected:
            self._enqueue_video("pointcloud", np.asarray(image_bgr), capture_host_ns)

    @staticmethod
    def _decoded_video_info(path: Path) -> tuple[bool, int, float]:
        capture = cv2.VideoCapture(str(path))
        opened = bool(capture.isOpened())
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
        capture.release()
        return opened, frames, fps

    def stop(self) -> Path | None:
        with self._lock:
            if not self.active:
                return self.session_dir
            self.capture_stop.set()
            if self.capture_thread is not None:
                self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
            if self.axis_sampler is not None and self._axis_listener is not None:
                self.axis_sampler.remove_listener(self._axis_listener)
            self._axis_listener = None
            self._accepting = False

            # Each encoder owns and releases its writer. Waiting for sentinels
            # guarantees the MP4 moov/index atom is written before returning.
            for name, stream_queue in self.video_queues.items():
                stream_queue.put(None)
            for process in self.video_threads.values():
                process.join()
            received_results = 0
            while received_results < len(self.selected):
                try:
                    name, encoded, codec, error = self.video_result_queue.get(timeout=1.0)
                except queue.Empty:
                    break
                self.encoded_counts[name] = int(encoded)
                if codec:
                    self.writer_codecs[name] = str(codec)
                if error:
                    self.errors.setdefault(name, str(error))
                received_results += 1
            if received_results < len(self.selected):
                self.errors.setdefault(
                    "encoder",
                    f"only {received_results}/{len(self.selected)} encoder results returned",
                )
            self.video_threads.clear()
            self.video_ready_events.clear()
            for stream_queue in self.video_queues.values():
                stream_queue.close()
                stream_queue.join_thread()
            for free_queue in self.video_free_queues.values():
                free_queue.close()
                free_queue.join_thread()
            self.video_free_queues.clear()
            for block in self.video_shared_blocks.values():
                block.close()
                try:
                    block.unlink()
                except FileNotFoundError:
                    pass
            self.video_shared_blocks.clear()
            self.video_shared_specs.clear()
            if self.video_result_queue is not None:
                self.video_result_queue.close()
                self.video_result_queue.join_thread()
                self.video_result_queue = None

            if self.data_queue is not None:
                self.data_queue.put(None)
            if self.data_thread is not None:
                self.data_thread.join()
            self.data_thread = None

            decoded = {}
            for name in sorted(self.selected):
                path = self.session_dir / self.STREAM_FILES[name]
                opened, frames, fps = self._decoded_video_info(path)
                decoded[name] = {"opened": opened, "frames": frames, "fps": fps}
                if not opened or frames <= 0:
                    self.errors.setdefault(name, "MP4完整性校验失败")
            motor_rate = 0.0
            if (
                self.motor_sample_count >= 2
                and self.motor_first_ns is not None
                and self.motor_last_ns is not None
                and self.motor_last_ns > self.motor_first_ns
            ):
                motor_rate = (
                    (self.motor_sample_count - 1) * 1e9
                    / (self.motor_last_ns - self.motor_first_ns)
                )
            camera_rate = 0.0
            capture_duration_s = 0.0
            if (
                self.camera_axis_rows >= 2
                and self.camera_first_ns is not None
                and self.camera_last_ns is not None
                and self.camera_last_ns > self.camera_first_ns
            ):
                capture_duration_s = (
                    self.camera_last_ns - self.camera_first_ns
                ) * 1e-9
                camera_rate = (self.camera_axis_rows - 1) / capture_duration_s
            summary = {
                "source_frames": self.source_counts,
                "encoded_frames": self.encoded_counts,
                "dropped_frames": self.dropped_counts,
                "codec": self.writer_codecs,
                "decoded_validation": decoded,
                "camera_axis_rows": self.camera_axis_rows,
                "camera_frame_rate_hz": camera_rate,
                "capture_duration_s": capture_duration_s,
                "motor_samples": self.motor_sample_count,
                "motor_sample_rate_hz": motor_rate,
                "metadata_dropped": self.metadata_dropped,
                "errors": self.errors,
            }
            with (self.session_dir / "summary.json").open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, ensure_ascii=False)
            self.video_queues.clear()
            self.data_queue = None
            self.camera_source = None
            self.axis_sampler = None
            return self.session_dir
