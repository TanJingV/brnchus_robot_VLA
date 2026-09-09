"""Readers for current and legacy D435 capture sessions."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from Visual_information.d435_tdcr_capture.models import CameraFrame


class SessionKind(str, Enum):
    BAG = "bag"
    KEYPOINTS = "keypoints"
    LEGACY_VIDEO = "legacy_video"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SessionDescriptor:
    path: Path
    kind: SessionKind
    frame_count: int
    fps: float
    duration_s: float
    has_rgb: bool
    has_depth: bool
    has_pointcloud: bool
    has_motor: bool
    has_em: bool
    has_endoscope: bool
    quality_title: str
    quality_detail: str


@dataclass
class LegacyFramePacket:
    index: int
    camera_frame: CameraFrame
    rgb_bgr: np.ndarray
    depth_visual_bgr: np.ndarray
    pointcloud_bgr: np.ndarray | None
    endoscope_bgr: np.ndarray | None
    axis_row: dict[str, Any]
    em_row: dict[str, Any]
    depth_decode_confidence: float


def _video_info(path: Path) -> tuple[bool, int, float]:
    capture = cv2.VideoCapture(str(path))
    opened = bool(capture.isOpened())
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    capture.release()
    return opened, count, fps


def describe_session(path: str | Path) -> SessionDescriptor:
    session = Path(path).expanduser().resolve()
    bag = session / "realsense.bag"
    archive = session / "keypoints.npz"
    rgb = session / "rgb.mp4"
    if not rgb.exists():
        rgb = session / "d435_rgb.mp4"
    depth = session / "depth.mp4"
    has_rgb, frame_count, fps = _video_info(rgb) if rgb.exists() else (False, 0, 0.0)
    has_depth, depth_count, depth_fps = _video_info(depth) if depth.exists() else (False, 0, 0.0)
    if bag.exists():
        kind = SessionKind.BAG
        title = "A级 · 原始多模态"
        detail = "包含 RealSense BAG，可重新计算 RGB、Z16、左右 IR 和三维坐标。"
    elif archive.exists():
        kind = SessionKind.KEYPOINTS
        title = "B级 · 已保存三维点"
        detail = "可重新执行时序和形状优化；不能重新做图像分割。"
    elif has_rgb and has_depth:
        kind = SessionKind.LEGACY_VIDEO
        title = "C级 · 旧版近似深度"
        detail = "Depth 为伪彩 MP4；可逆映射近似深度，但缺少原始 Z16、IR 和精确 RGB-Depth 外参。"
        frame_count = min(frame_count, depth_count) if depth_count else frame_count
        fps = fps or depth_fps
    else:
        kind = SessionKind.UNSUPPORTED
        title = "不可处理"
        detail = "未找到 BAG、keypoints.npz 或成对 RGB/Depth 视频。"
    if frame_count <= 0 and archive.exists():
        try:
            with np.load(archive, allow_pickle=False) as data:
                name = "capture_host_ns" if "capture_host_ns" in data.files else data.files[0]
                frame_count = int(len(data[name]))
        except Exception:
            frame_count = 0
    return SessionDescriptor(
        path=session,
        kind=kind,
        frame_count=frame_count,
        fps=fps or 30.0,
        duration_s=frame_count / max(fps or 30.0, 1e-6),
        has_rgb=has_rgb,
        has_depth=has_depth,
        has_pointcloud=(session / "pointcloud_3d.mp4").exists(),
        has_motor=(session / "motor_axes_100hz.csv").exists() or (session / "camera_axes.csv").exists(),
        has_em=(session / "em_data_100hz.csv").exists(),
        has_endoscope=(session / "endoscope_rgb.mp4").exists(),
        quality_title=title,
        quality_detail=detail,
    )


def candidate_session_roots(project_root: Path) -> list[Path]:
    candidates = [project_root / "d435_sessions"]
    git_file = project_root / ".git"
    if git_file.is_file():
        try:
            content = git_file.read_text(encoding="utf-8").strip().replace("\\", "/")
            if content.lower().startswith("gitdir:") and "/.git/" in content:
                original = Path(content.split(":", 1)[1].strip().split("/.git/", 1)[0])
                candidates.append(original / "d435_sessions")
        except Exception:
            pass
    candidates.extend([
        Path.home() / "d435_sessions",
        Path("D:/d435_sessions"),
        Path("E:/d435_sessions"),
    ])
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def discover_sessions(project_root: Path) -> list[SessionDescriptor]:
    sessions: list[SessionDescriptor] = []
    for root in candidate_session_roots(project_root):
        if not root.exists():
            continue
        for directory in root.iterdir():
            if not directory.is_dir():
                continue
            descriptor = describe_session(directory)
            if descriptor.kind != SessionKind.UNSUPPORTED:
                sessions.append(descriptor)
    sessions.sort(key=lambda item: item.path.stat().st_mtime, reverse=True)
    return sessions


class TurboDepthDecoder:
    """Approximately invert the TURBO visualization used by old recordings."""

    def __init__(self) -> None:
        indices = np.arange(256, dtype=np.uint8).reshape(256, 1)
        self.palette = cv2.applyColorMap(indices, cv2.COLORMAP_TURBO).reshape(256, 3).astype(np.int16)
        levels = np.arange(0, 256, 8, dtype=np.int16) + 4
        blue, green, red = np.meshgrid(levels, levels, levels, indexing="ij")
        colors = np.column_stack((blue.reshape(-1), green.reshape(-1), red.reshape(-1)))
        palette_difference = (
            colors[:, None, :].astype(np.int32) - self.palette[None, :, :].astype(np.int32)
        )
        distances = np.sum(
            palette_difference * palette_difference, axis=2, dtype=np.int64
        )
        self.quantized_index = np.argmin(distances, axis=1).astype(np.uint8)

    def decode(
        self,
        visual_bgr: np.ndarray,
        minimum_m: float,
        maximum_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(visual_bgr, dtype=np.uint8)
        quantized = (image.astype(np.int32) >> 3).clip(0, 31)
        lookup_index = (quantized[..., 0] * 32 + quantized[..., 1]) * 32 + quantized[..., 2]
        color_index = self.quantized_index[lookup_index]
        normalized_depth = 255.0 - color_index.astype(np.float32)
        depth_m = float(minimum_m) + normalized_depth * (
            float(maximum_m) - float(minimum_m)
        ) / 255.0
        reconstructed = self.palette[color_index]
        difference = image.astype(np.int32) - reconstructed.astype(np.int32)
        error = np.sqrt(np.sum(difference * difference, axis=2, dtype=np.int64)).astype(np.float32)
        invalid_color = np.asarray([12, 13, 16], dtype=np.int32)
        invalid_difference = image.astype(np.int32) - invalid_color
        invalid_distance = np.sqrt(
            np.sum(invalid_difference * invalid_difference, axis=2, dtype=np.int64)
        )
        valid = (invalid_distance > 22.0) & (error < 58.0)
        depth_m[~valid] = 0.0
        confidence = np.clip(1.0 - error / 58.0, 0.0, 1.0)
        confidence[~valid] = 0.0
        return depth_m, confidence


class LegacySessionReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.descriptor = describe_session(self.path)
        if self.descriptor.kind != SessionKind.LEGACY_VIDEO:
            raise ValueError(f"Not a legacy video session: {self.path}")
        capture_path = self.path / "capture.json"
        self.manifest = json.loads(capture_path.read_text(encoding="utf-8")) if capture_path.exists() else {}
        self.camera_config = self.manifest.get("camera_config", {})
        self.decoder = TurboDepthDecoder()
        self.captures: dict[str, cv2.VideoCapture] = {}
        self.last_indices: dict[str, int] = {}
        self.axis_rows = self._read_numeric_rows(self.path / "camera_axes.csv")
        self.em_rows = self._read_numeric_rows(self.path / "em_data_100hz.csv")
        self.em_host_ns = np.asarray([
            int(row.get("em_host_ns", 0) or 0) for row in self.em_rows
        ], dtype=np.int64)

    @staticmethod
    def _read_numeric_rows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                converted: dict[str, Any] = {}
                for key, value in row.items():
                    if value is None or value == "":
                        converted[key] = math.nan
                        continue
                    try:
                        converted[key] = float(value)
                    except (TypeError, ValueError):
                        converted[key] = value
                rows.append(converted)
        return rows

    def _capture(self, name: str) -> cv2.VideoCapture | None:
        if name in self.captures:
            return self.captures[name]
        filenames = {
            "rgb": "rgb.mp4",
            "depth": "depth.mp4",
            "pointcloud": "pointcloud_3d.mp4",
            "endoscope": "endoscope_rgb.mp4",
        }
        path = self.path / filenames[name]
        if not path.exists():
            return None
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            return None
        self.captures[name] = capture
        self.last_indices[name] = -1
        return capture

    def _read_video(self, name: str, index: int, source_fps: float | None = None) -> np.ndarray | None:
        capture = self._capture(name)
        if capture is None:
            return None
        target = index
        if source_fps is not None:
            target = int(round(index / self.descriptor.fps * source_fps))
        if self.last_indices[name] + 1 != target:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = capture.read()
        if not ok:
            return None
        self.last_indices[name] = target
        return frame

    def _nearest_em(self, host_ns: int) -> dict[str, Any]:
        if not len(self.em_host_ns) or host_ns <= 0:
            return {}
        location = int(np.searchsorted(self.em_host_ns, host_ns))
        candidates = [max(0, location - 1), min(len(self.em_host_ns) - 1, location)]
        index = min(candidates, key=lambda item: abs(int(self.em_host_ns[item]) - host_ns))
        return self.em_rows[index]

    @staticmethod
    def _estimated_intrinsics(width: int, height: int, parameters: dict[str, float]) -> dict[str, Any]:
        default_fx = width / (2.0 * math.tan(math.radians(69.4) / 2.0))
        default_fy = height / (2.0 * math.tan(math.radians(42.5) / 2.0))
        return {
            "width": width,
            "height": height,
            "fx": float(parameters.get("fx", default_fx)),
            "fy": float(parameters.get("fy", default_fy)),
            "ppx": float(parameters.get("cx", width / 2.0)),
            "ppy": float(parameters.get("cy", height / 2.0)),
            "coeffs": [0.0] * 5,
            "model": "legacy_estimate",
        }

    @staticmethod
    def _register_depth(
        depth_native_m: np.ndarray,
        output_size: tuple[int, int],
        parameters: dict[str, float],
    ) -> np.ndarray:
        width, height = output_size
        resized = cv2.resize(depth_native_m, (width, height), interpolation=cv2.INTER_NEAREST)
        scale_x = float(parameters.get("registration_scale_x", 1.0))
        scale_y = float(parameters.get("registration_scale_y", 1.0))
        shift_x = float(parameters.get("registration_dx_px", 0.0))
        shift_y = float(parameters.get("registration_dy_px", 0.0))
        matrix = np.asarray([[scale_x, 0.0, shift_x], [0.0, scale_y, shift_y]], dtype=np.float32)
        return cv2.warpAffine(
            resized,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def read(self, index: int, parameters: dict[str, float] | None = None) -> LegacyFramePacket:
        parameters = parameters or {}
        rgb = self._read_video("rgb", index)
        depth_visual = self._read_video("depth", index)
        if rgb is None or depth_visual is None:
            raise EOFError(f"Unable to read legacy frame {index}")
        pointcloud_fps = float(self.manifest.get("nominal_fps", {}).get("pointcloud", 15.0))
        pointcloud = self._read_video("pointcloud", index, pointcloud_fps)
        endoscope = self._read_video("endoscope", index, 30.0)
        minimum = float(parameters.get("depth_min_m", self.camera_config.get("depth_min_m", 0.18)))
        maximum = float(parameters.get("depth_max_m", self.camera_config.get("depth_max_m", 0.60)))
        native_depth, decode_confidence = self.decoder.decode(depth_visual, minimum, maximum)
        height, width = rgb.shape[:2]
        depth_aligned = self._register_depth(native_depth, (width, height), parameters)
        intrinsics = self._estimated_intrinsics(width, height, parameters)
        axis_row = self.axis_rows[index] if index < len(self.axis_rows) else {}
        host_ns = int(axis_row.get("capture_host_ns", index / self.descriptor.fps * 1e9) or 0)
        em_row = self._nearest_em(host_ns)
        timestamp_ms = float(axis_row.get("depth_timestamp_ms", index / self.descriptor.fps * 1000.0))
        camera_frame = CameraFrame(
            sequence=int(axis_row.get("sequence", index) or index),
            host_arrival_ns=host_ns,
            capture_host_ns=host_ns,
            device_timestamp_ms=timestamp_ms,
            color_timestamp_ms=float(axis_row.get("color_timestamp_ms", timestamp_ms)),
            depth_timestamp_ms=timestamp_ms,
            infrared_timestamp_ms=float("nan"),
            color_bgr=rgb,
            depth_m=depth_aligned,
            infrared_left=np.empty((0, 0), dtype=np.uint8),
            infrared_right=np.empty((0, 0), dtype=np.uint8),
            color_intrinsics=intrinsics,
            left_intrinsics=intrinsics,
            right_intrinsics=intrinsics,
            transform_left_from_color=np.eye(4),
            transform_right_from_left=np.eye(4),
            metadata={
                "legacy_pseudocolor_depth": True,
                "depth_aligned_to_color": True,
                "depth_registration_approximate": True,
                "source_session": str(self.path),
            },
            depth_unit_m=float(self.camera_config.get("depth_unit_m", 0.001)),
        )
        valid_decode = decode_confidence[decode_confidence > 0]
        return LegacyFramePacket(
            index=index,
            camera_frame=camera_frame,
            rgb_bgr=rgb,
            depth_visual_bgr=depth_visual,
            pointcloud_bgr=pointcloud,
            endoscope_bgr=endoscope,
            axis_row=axis_row,
            em_row=em_row,
            depth_decode_confidence=float(np.mean(valid_decode)) if len(valid_decode) else 0.0,
        )

    def close(self) -> None:
        for capture in self.captures.values():
            capture.release()
        self.captures.clear()
        self.last_indices.clear()
