"""RealSense D435 capture with optional import, replay and synthetic fallback."""

from __future__ import annotations

import concurrent.futures
from collections import deque
from dataclasses import replace
import time
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import CameraFrame, RgbCameraFrame
from .sync import DeviceClockMapper

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


def realsense_available() -> bool:
    return rs is not None


class CudaDepthToColorAligner:
    """Calibrated Z16 depth-to-colour alignment on a CUDA device."""

    def __init__(self, depth_intrinsics, color_intrinsics, extrinsics, depth_scale: float) -> None:
        from numba import cuda

        if not cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if any(abs(float(value)) > 1e-10 for value in depth_intrinsics.coeffs):
            raise RuntimeError("non-zero depth distortion requires SDK alignment")
        if any(abs(float(value)) > 1e-10 for value in color_intrinsics.coeffs):
            raise RuntimeError("non-zero color distortion requires SDK alignment")
        self.cuda = cuda
        self.depth_height = int(depth_intrinsics.height)
        self.depth_width = int(depth_intrinsics.width)
        self.output_height = int(color_intrinsics.height)
        self.output_width = int(color_intrinsics.width)
        self.depth_scale = float(depth_scale)
        self.parameters = np.asarray([
            float(depth_intrinsics.fx), float(depth_intrinsics.fy),
            float(depth_intrinsics.ppx), float(depth_intrinsics.ppy),
            float(color_intrinsics.fx), float(color_intrinsics.fy),
            float(color_intrinsics.ppx), float(color_intrinsics.ppy),
            *[float(value) for value in extrinsics.rotation],
            *[float(value) for value in extrinsics.translation],
            self.depth_scale,
        ], dtype=np.float32)

        @cuda.jit
        def reset_kernel(output):
            y, x = cuda.grid(2)
            if y < output.shape[0] and x < output.shape[1]:
                output[y, x] = 65535

        @cuda.jit
        def align_kernel(source, output, p):
            y, x = cuda.grid(2)
            if y >= source.shape[0] or x >= source.shape[1]:
                return
            raw_depth = int(source[y, x])
            if raw_depth <= 0:
                return
            depth = raw_depth * p[20]
            # Project the upper-left and lower-right corners of one depth pixel.
            rx0 = (x - 0.5 - p[2]) / p[0]
            ry0 = (y - 0.5 - p[3]) / p[1]
            rx1 = (x + 0.5 - p[2]) / p[0]
            ry1 = (y + 0.5 - p[3]) / p[1]
            px0 = rx0 * depth
            py0 = ry0 * depth
            px1 = rx1 * depth
            py1 = ry1 * depth
            # librealsense stores its rotation matrix in column-major order.
            tx0 = p[8] * px0 + p[11] * py0 + p[14] * depth + p[17]
            ty0 = p[9] * px0 + p[12] * py0 + p[15] * depth + p[18]
            tz0 = p[10] * px0 + p[13] * py0 + p[16] * depth + p[19]
            tx1 = p[8] * px1 + p[11] * py1 + p[14] * depth + p[17]
            ty1 = p[9] * px1 + p[12] * py1 + p[15] * depth + p[18]
            tz1 = p[10] * px1 + p[13] * py1 + p[16] * depth + p[19]
            if tz0 <= 0.0 or tz1 <= 0.0:
                return
            u0 = tx0 * p[4] / tz0 + p[6]
            v0 = ty0 * p[5] / tz0 + p[7]
            u1 = tx1 * p[4] / tz1 + p[6]
            v1 = ty1 * p[5] / tz1 + p[7]
            ix0 = int(min(u0, u1) + 0.5)
            ix1 = int(max(u0, u1) + 0.5)
            iy0 = int(min(v0, v1) + 0.5)
            iy1 = int(max(v0, v1) + 0.5)
            if ix1 - ix0 > 3:
                ix1 = ix0 + 3
            if iy1 - iy0 > 3:
                iy1 = iy0 + 3
            for output_y in range(iy0, iy1 + 1):
                if output_y < 0 or output_y >= output.shape[0]:
                    continue
                for output_x in range(ix0, ix1 + 1):
                    if output_x >= 0 and output_x < output.shape[1]:
                        cuda.atomic.min(output, (output_y, output_x), raw_depth)

        self.reset_kernel = reset_kernel
        self.align_kernel = align_kernel
        self.device_source = cuda.device_array(
            (self.depth_height, self.depth_width), dtype=np.uint16
        )
        self.device_output = cuda.device_array(
            (self.output_height, self.output_width), dtype=np.int32
        )
        self.device_parameters = cuda.to_device(self.parameters)
        self.host_output = np.empty(
            (self.output_height, self.output_width), dtype=np.int32
        )
        self.depth_blocks = (
            (self.depth_height + 15) // 16,
            (self.depth_width + 15) // 16,
        )
        self.output_blocks = (
            (self.output_height + 15) // 16,
            (self.output_width + 15) // 16,
        )
        self.threads = (16, 16)
        # Compile kernels before acquisition begins so the first camera frame
        # does not absorb JIT latency.
        self.align_meters(np.zeros((self.depth_height, self.depth_width), dtype=np.uint16))

    def align_meters(self, depth_z16: np.ndarray) -> np.ndarray:
        raw = np.ascontiguousarray(depth_z16, dtype=np.uint16)
        if raw.shape != (self.depth_height, self.depth_width):
            raise ValueError(f"unexpected depth shape {raw.shape}")
        self.device_source.copy_to_device(raw)
        self.reset_kernel[self.output_blocks, self.threads](self.device_output)
        self.align_kernel[self.depth_blocks, self.threads](
            self.device_source, self.device_output, self.device_parameters
        )
        self.device_output.copy_to_host(self.host_output)
        output = self.host_output.astype(np.float32)
        output[self.host_output == 65535] = 0.0
        output *= self.depth_scale
        return output


def _intrinsics_dict(intrinsics) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "coeffs": [float(value) for value in intrinsics.coeffs],
        "model": str(intrinsics.model),
    }


def _transform_matrix(extrinsics) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(extrinsics.rotation, dtype=float).reshape(3, 3)
    result[:3, 3] = np.asarray(extrinsics.translation, dtype=float)
    return result


def _metadata(frame) -> dict[str, Any]:
    if frame is None or rs is None:
        return {}
    entries = {
        "frame_counter": rs.frame_metadata_value.frame_counter,
        "frame_timestamp": rs.frame_metadata_value.frame_timestamp,
        "sensor_timestamp": rs.frame_metadata_value.sensor_timestamp,
        "actual_exposure": rs.frame_metadata_value.actual_exposure,
        "gain_level": rs.frame_metadata_value.gain_level,
    }
    output: dict[str, Any] = {}
    for name, field in entries.items():
        try:
            if frame.supports_frame_metadata(field):
                output[name] = int(frame.get_frame_metadata(field))
        except Exception:
            continue
    return output


class RealSenseSource:
    """Non-blocking D435 source. All returned arrays own their memory."""

    def __init__(self, config: dict, bag_path: str | Path | None = None, repeat: bool = False) -> None:
        self.config = config
        self.bag_path = Path(bag_path).resolve() if bag_path else None
        self.repeat = bool(repeat)
        self.pipeline = None
        self.profile = None
        self.align = None
        self.sequence = 0
        self.clock_mapper = DeviceClockMapper()
        self.recorder = None
        self.device_info: dict[str, Any] = {}
        self._settings_lock = threading.RLock()
        self._settings_dirty = True
        self._depth_sensor = None
        self._color_sensor = None
        self._depth_filters: dict[str, Any] = {}
        self._filter_signature: tuple[Any, ...] | None = None
        self._cached_color_intrinsics: dict[str, Any] | None = None
        self._cached_depth_intrinsics: dict[str, Any] | None = None
        self._cached_left_intrinsics: dict[str, Any] | None = None
        self._cached_right_intrinsics: dict[str, Any] | None = None
        self._cached_left_from_color: np.ndarray | None = None
        self._cached_right_from_left: np.ndarray | None = None
        self._cached_depth_scale = 0.001
        self._cuda_aligner: CudaDepthToColorAligner | None = None
        self._copy_infrared_frames = bool(config.get("copy_infrared_frames", False))
        self._ir_streams_enabled = bool(
            config.get("enable_infrared_streams", False) or self._copy_infrared_frames
        )
        self._align_depth_to_color = bool(config.get("align_depth_to_color", False))
        self._align_thread_local = threading.local()
        self._colorizer_thread_local = threading.local()
        self._empty_infrared = np.empty((0, 0), dtype=np.uint8)
        self._frame_lock = threading.Lock()
        self._capture_stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._latest_frame: CameraFrame | None = None
        self._delivered_sequence = -1
        self._capture_error: Exception | None = None
        self._completed_frames = deque(maxlen=256)

    @staticmethod
    def devices() -> list[dict[str, str]]:
        if rs is None:
            return []
        result = []
        for device in rs.context().query_devices():
            info = {}
            for field, name in (
                (rs.camera_info.name, "name"),
                (rs.camera_info.serial_number, "serial"),
                (rs.camera_info.firmware_version, "firmware"),
                (rs.camera_info.usb_type_descriptor, "usb"),
            ):
                try:
                    info[name] = device.get_info(field)
                except Exception:
                    info[name] = ""
            result.append(info)
        return result

    def set_infrared_copy_enabled(self, enabled: bool) -> None:
        """Copy IR images only while a vision task consumes them."""
        enabled = bool(enabled)
        restart = self.pipeline is not None and enabled != self._ir_streams_enabled
        self._copy_infrared_frames = enabled
        self._ir_streams_enabled = enabled
        if restart:
            self.stop()
            self.start()

    def start(self) -> None:
        if rs is None:
            raise RuntimeError("未安装 pyrealsense2。请运行 requirements-d435.txt 中的安装命令。")
        if self.pipeline is not None:
            return
        if self.bag_path is None:
            try:
                devices = self.devices()
                requested_serial = str(self.config.get("serial", "")).strip()
                selected = next(
                    (device for device in devices if not requested_serial or device.get("serial") == requested_serial),
                    None,
                )
                if selected is not None:
                    self.device_info = dict(selected)
                    usb = str(selected.get("usb", "")).strip()
                    if usb.startswith("2"):
                        raise RuntimeError(
                            f"检测到 {selected.get('name', 'D435')} 通过 USB {usb} 连接。"
                            "当前采集需要 RGB、Depth 和左右 IR 四路数据，USB 2.x 不提供右 IR 所需的完整流配置。"
                            "请使用支持 SuperSpeed 数据的 USB 3.x 线缆，并直接连接电脑蓝色/SS 标识的 USB 3.x 端口；"
                            "重新插入后应显示 USB 3.1 或 3.2。"
                        )
            except RuntimeError:
                raise
            except Exception:
                # Some Windows privacy/sandbox configurations deny enumeration but
                # still allow pipeline startup, so keep the normal fallback path.
                pass
        attempts = [
            (
                tuple(self.config["depth"]),
                tuple(self.config["infrared"]),
                tuple(self.config["color"]),
            )
        ]
        for width, height, fps in self.config.get("fallback_profiles", []):
            attempts.append(((width, height, fps), (width, height, fps), (width, height, fps)))
        errors: list[str] = []
        for depth_profile, ir_profile, color_profile in attempts:
            pipeline = rs.pipeline()
            cfg = rs.config()
            try:
                if self.bag_path is not None:
                    cfg.enable_device_from_file(str(self.bag_path), self.repeat)
                else:
                    serial = str(self.config.get("serial", "")).strip()
                    if serial:
                        cfg.enable_device(serial)
                    dw, dh, df = map(int, depth_profile)
                    iw, ih, inf = map(int, ir_profile)
                    cw, ch, cf = map(int, color_profile)
                    cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, df)
                    if self._ir_streams_enabled:
                        cfg.enable_stream(rs.stream.infrared, 1, iw, ih, rs.format.y8, inf)
                        cfg.enable_stream(rs.stream.infrared, 2, iw, ih, rs.format.y8, inf)
                    cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cf)
                profile = pipeline.start(cfg)
                self.pipeline = pipeline
                self.profile = profile
                self.align = rs.align(rs.stream.color)
                self._read_device_info()
                self._initialize_camera_processing()
                self._start_sdk_capture_thread()
                return
            except Exception as exc:
                errors.append(str(exc))
                try:
                    pipeline.stop()
                except Exception:
                    pass
        raise RuntimeError("无法启动D435数据流：" + " | ".join(errors))

    def _read_device_info(self) -> None:
        device = self.profile.get_device()
        for field, name in (
            (rs.camera_info.name, "name"),
            (rs.camera_info.serial_number, "serial"),
            (rs.camera_info.firmware_version, "firmware"),
            (rs.camera_info.usb_type_descriptor, "usb"),
        ):
            try:
                self.device_info[name] = device.get_info(field)
            except Exception:
                self.device_info[name] = ""

    def _initialize_camera_processing(self) -> None:
        """Create SDK filters once and apply precision-related sensor options."""
        if self.profile is None or rs is None:
            return
        device = self.profile.get_device()
        try:
            self._depth_sensor = device.first_depth_sensor()
            self._cached_depth_scale = float(self._depth_sensor.get_depth_scale())
            self.config["depth_unit_m"] = self._cached_depth_scale
        except Exception:
            self._depth_sensor = None
        try:
            self._color_sensor = device.first_color_sensor()
        except Exception:
            self._color_sensor = None
        self._depth_filters = {
            "depth_to_disparity": rs.disparity_transform(True),
            "spatial": rs.spatial_filter(),
            "temporal": rs.temporal_filter(),
            "disparity_to_depth": rs.disparity_transform(False),
            "hole_filling": rs.hole_filling_filter(),
        }
        try:
            depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            self._cached_color_intrinsics = _intrinsics_dict(color_profile.get_intrinsics())
            backend = str(self.config.get("alignment_backend", "cuda")).lower()
            self._cached_depth_intrinsics = _intrinsics_dict(
                depth_profile.get_intrinsics()
            )
            if self._align_depth_to_color and backend in ("cuda", "auto"):
                try:
                    self._cuda_aligner = CudaDepthToColorAligner(
                        depth_profile.get_intrinsics(),
                        color_profile.get_intrinsics(),
                        depth_profile.get_extrinsics_to(color_profile),
                        self._cached_depth_scale,
                    )
                    self.device_info["alignment_backend"] = "cuda"
                except Exception as exc:
                    self._cuda_aligner = None
                    self.device_info["alignment_backend"] = f"sdk_cpu ({exc})"
            else:
                self._cuda_aligner = None
                self.device_info["alignment_backend"] = "sdk_cpu"
        except Exception:
            self._cached_color_intrinsics = None
            self._cached_depth_intrinsics = None
            self._cuda_aligner = None
        try:
            left_profile = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            right_profile = self.profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            self._cached_left_intrinsics = _intrinsics_dict(left_profile.get_intrinsics())
            self._cached_right_intrinsics = _intrinsics_dict(right_profile.get_intrinsics())
            self._cached_left_from_color = _transform_matrix(
                color_profile.get_extrinsics_to(left_profile)
            )
            self._cached_right_from_left = _transform_matrix(
                left_profile.get_extrinsics_to(right_profile)
            )
        except Exception:
            self._cached_left_intrinsics = self._cached_color_intrinsics
            self._cached_right_intrinsics = self._cached_color_intrinsics
            self._cached_left_from_color = np.eye(4)
            self._cached_right_from_left = np.eye(4)
        self._filter_signature = None
        self._settings_dirty = True
        self._refresh_runtime_settings(force=True)

    @staticmethod
    def _clamped_option_value(sensor, option, value: float) -> float:
        option_range = sensor.get_option_range(option)
        return float(np.clip(float(value), float(option_range.min), float(option_range.max)))

    def _set_sensor_option(self, sensor, option, value: float) -> None:
        if sensor is None:
            return
        try:
            try:
                if not sensor.supports(option):
                    return
            except TypeError:
                # Processing blocks expose camera-info ``supports`` but still
                # implement the options interface (get_option_range/set_option).
                pass
            sensor.set_option(option, self._clamped_option_value(sensor, option, value))
        except Exception:
            # Bag playback and some firmware profiles expose read-only options.
            pass

    def update_runtime_settings(self, updates: dict[str, Any]) -> None:
        """Queue camera/filter settings for application by the capture thread."""
        with self._settings_lock:
            self.config.update(updates)
            self._settings_dirty = True

    def _refresh_runtime_settings(self, *, force: bool = False) -> None:
        if rs is None:
            return
        with self._settings_lock:
            if not force and not self._settings_dirty:
                return
            settings = dict(self.config)
            self._settings_dirty = False

        preset_values = {
            "custom": 0.0,
            "default": 1.0,
            "hand": 2.0,
            "high_accuracy": 3.0,
            "high_density": 4.0,
            "medium_density": 5.0,
        }
        preset = preset_values.get(str(settings.get("visual_preset", "high_accuracy")), 3.0)
        self._set_sensor_option(self._depth_sensor, rs.option.visual_preset, preset)
        self._set_sensor_option(
            self._depth_sensor,
            rs.option.emitter_enabled,
            1.0 if bool(settings.get("emitter_enabled", True)) else 0.0,
        )
        self._set_sensor_option(
            self._depth_sensor,
            rs.option.laser_power,
            float(settings.get("laser_power", 150.0)),
        )
        depth_auto = bool(settings.get("depth_auto_exposure", True))
        self._set_sensor_option(
            self._depth_sensor,
            rs.option.enable_auto_exposure,
            1.0 if depth_auto else 0.0,
        )
        if not depth_auto:
            self._set_sensor_option(
                self._depth_sensor,
                rs.option.exposure,
                float(settings.get("depth_exposure_us", 8500.0)),
            )
            self._set_sensor_option(
                self._depth_sensor,
                rs.option.gain,
                float(settings.get("depth_gain", 16.0)),
            )

        color_auto = bool(settings.get("color_auto_exposure", True))
        self._set_sensor_option(
            self._color_sensor,
            rs.option.enable_auto_exposure,
            1.0 if color_auto else 0.0,
        )
        if not color_auto:
            self._set_sensor_option(
                self._color_sensor,
                rs.option.exposure,
                float(settings.get("color_exposure_us", 156.0)),
            )
            self._set_sensor_option(
                self._color_sensor,
                rs.option.gain,
                float(settings.get("color_gain", 64.0)),
            )
        self._set_sensor_option(
            self._color_sensor,
            rs.option.enable_auto_white_balance,
            1.0 if bool(settings.get("color_auto_white_balance", True)) else 0.0,
        )

        signature = (
            bool(settings.get("depth_postprocess_enabled", True)),
            bool(settings.get("depth_spatial_enabled", True)),
            int(settings.get("depth_spatial_magnitude", 2)),
            float(settings.get("depth_spatial_alpha", 0.5)),
            float(settings.get("depth_spatial_delta", 8.0)),
            int(settings.get("depth_spatial_holes_fill", 1)),
            bool(settings.get("depth_temporal_enabled", True)),
            float(settings.get("depth_temporal_alpha", 0.55)),
            float(settings.get("depth_temporal_delta", 20.0)),
            int(settings.get("depth_temporal_persistency", 1)),
            bool(settings.get("depth_hole_filling_enabled", False)),
            int(settings.get("depth_hole_filling_mode", 1)),
        )
        if force or signature != self._filter_signature:
            spatial = self._depth_filters.get("spatial")
            temporal = self._depth_filters.get("temporal")
            hole_filling = self._depth_filters.get("hole_filling")
            if spatial is not None:
                self._set_sensor_option(spatial, rs.option.filter_magnitude, signature[2])
                self._set_sensor_option(spatial, rs.option.filter_smooth_alpha, signature[3])
                self._set_sensor_option(spatial, rs.option.filter_smooth_delta, signature[4])
                self._set_sensor_option(spatial, rs.option.holes_fill, signature[5])
            if temporal is not None:
                self._set_sensor_option(temporal, rs.option.filter_smooth_alpha, signature[7])
                self._set_sensor_option(temporal, rs.option.filter_smooth_delta, signature[8])
                self._set_sensor_option(temporal, rs.option.holes_fill, signature[9])
            if hole_filling is not None:
                self._set_sensor_option(hole_filling, rs.option.holes_fill, signature[11])
            self._filter_signature = signature

    def _process_depth_frame(self, depth_frame):
        signature = self._filter_signature
        if not signature or not signature[0]:
            return depth_frame
        result = depth_frame
        if signature[1]:
            result = self._depth_filters["depth_to_disparity"].process(result)
            result = self._depth_filters["spatial"].process(result)
            if signature[6]:
                result = self._depth_filters["temporal"].process(result)
            result = self._depth_filters["disparity_to_depth"].process(result)
        elif signature[6]:
            result = self._depth_filters["temporal"].process(result)
        if signature[10]:
            result = self._depth_filters["hole_filling"].process(result)
        return result.as_depth_frame()

    def _colorize_native_depth(self, depth_frame) -> np.ndarray | None:
        """Use librealsense's own colorizer, matching RealSense Viewer."""
        if (
            not bool(self.config.get("native_depth_colorizer_enabled", True))
            or not bool(
                self.config.get("runtime_native_depth_colorizer_enabled", True)
            )
        ):
            return None
        try:
            colorizer = getattr(self._colorizer_thread_local, "colorizer", None)
            if colorizer is None:
                colorizer = rs.colorizer()
                self._colorizer_thread_local.colorizer = colorizer
            options = (
                (rs.option.histogram_equalization_enabled,
                 1.0 if bool(self.config.get("depth_colorizer_histogram_equalization", True)) else 0.0),
                (rs.option.min_distance, float(self.config.get("depth_min_m", 0.18))),
                (rs.option.max_distance, float(self.config.get("depth_max_m", 0.60))),
                (rs.option.color_scheme, float(self.config.get("depth_colorizer_scheme", 0))),
            )
            for option, value in options:
                self._set_sensor_option(colorizer, option, value)
            colored_frame = colorizer.colorize(depth_frame)
            image = np.asanyarray(colored_frame.get_data()).copy()
            try:
                if colored_frame.profile.format() == rs.format.rgb8:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception:
                pass
            return image
        except Exception:
            # Bag files and older SDK builds can omit one or more colorizer
            # options. The UI will fall back to its local visualizer.
            return None

    def start_bag_recording(self, path: str | Path) -> bool:
        if self.profile is None or rs is None:
            return False
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.recorder = rs.recorder(str(target), self.profile.get_device())
            return True
        except Exception:
            self.recorder = None
            return False

    def stop_bag_recording(self) -> None:
        if self.recorder is not None:
            try:
                self.recorder.pause()
            except Exception:
                pass
        self.recorder = None

    def _start_sdk_capture_thread(self) -> None:
        """Continuously drain the SDK pipeline and retain only its newest frame."""
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        with self._frame_lock:
            self._latest_frame = None
            self._delivered_sequence = -1
            self._capture_error = None
            self._completed_frames.clear()
        self._capture_stop.clear()
        self._capture_thread = threading.Thread(
            target=self._sdk_capture_loop,
            name="realsense-sdk-capture",
            daemon=True,
        )
        self._capture_thread.start()

    def _sdk_capture_loop(self) -> None:
        worker_count = max(1, int(self.config.get("alignment_workers", 3)))
        if self._cuda_aligner is not None:
            worker_count = 1
        if bool(self.config.get("depth_postprocess_enabled", False)):
            worker_count = 1
        if self._align_depth_to_color and worker_count > 1:
            self._sdk_parallel_capture_loop(worker_count)
            return
        while not self._capture_stop.is_set():
            try:
                frame = self._poll_sdk_once()
                if frame is None:
                    self._capture_stop.wait(0.001)
                    continue
                self._publish_frame(frame)
            except Exception as exc:
                with self._frame_lock:
                    self._capture_error = exc
                self._capture_stop.wait(0.02)

    def _publish_frame(self, frame: CameraFrame | None) -> None:
        if frame is None:
            return
        with self._frame_lock:
            if self._latest_frame is None or frame.sequence > self._latest_frame.sequence:
                self._latest_frame = frame
                self._capture_error = None
                # The recorder consumes raw Z16 and aligned metric depth, not
                # the preview-only native float/color buffers. Avoid retaining
                # hundreds of extra full-resolution images in its backlog.
                self._completed_frames.append(
                    replace(
                        frame,
                        native_depth_m=None,
                        native_depth_color_bgr=None,
                    )
                )

    def _sdk_parallel_capture_loop(self, worker_count: int) -> None:
        """Drain hardware at full rate while several SDK aligners work in parallel."""
        maximum_inflight = max(worker_count + 1, worker_count * 2)
        futures: dict[concurrent.futures.Future, int] = {}
        ready_frames: dict[int, CameraFrame] = {}
        failed_sequences: set[int] = set()
        next_publish = self.sequence + 1
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="realsense-align"
        ) as executor:
            while not self._capture_stop.is_set():
                completed = [future for future in futures if future.done()]
                for future in completed:
                    sequence = futures.pop(future)
                    try:
                        frame = future.result()
                        if frame is not None:
                            ready_frames[sequence] = frame
                    except Exception as exc:
                        with self._frame_lock:
                            self._capture_error = exc
                        failed_sequences.add(sequence)
                while next_publish in ready_frames or next_publish in failed_sequences:
                    if next_publish in ready_frames:
                        self._publish_frame(ready_frames.pop(next_publish))
                    else:
                        failed_sequences.remove(next_publish)
                    next_publish += 1
                try:
                    frames = self.pipeline.poll_for_frames() if self.pipeline is not None else None
                    if frames and len(futures) < maximum_inflight:
                        prepared = self._prepare_frameset(frames)
                        if prepared is not None:
                            future = executor.submit(self._process_frameset, *prepared)
                            futures[future] = int(prepared[3])
                    else:
                        self._capture_stop.wait(0.0005)
                except Exception as exc:
                    with self._frame_lock:
                        self._capture_error = exc
                    self._capture_stop.wait(0.01)
            for future in concurrent.futures.as_completed(futures):
                try:
                    frame = future.result()
                    if frame is not None:
                        ready_frames[frame.sequence] = frame
                except Exception:
                    pass
            for sequence in sorted(ready_frames):
                self._publish_frame(ready_frames[sequence])

    def stop(self) -> None:
        self.stop_bag_recording()
        self._capture_stop.set()
        capture_thread = self._capture_thread
        if capture_thread is not None:
            capture_thread.join(3.0)
        self._capture_thread = None
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None
                self.profile = None
                self.align = None
                self._depth_sensor = None
                self._color_sensor = None
                self._depth_filters = {}
                self._filter_signature = None
                self._cached_color_intrinsics = None
                self._cached_depth_intrinsics = None
                self._cached_left_intrinsics = None
                self._cached_right_intrinsics = None
                self._cached_left_from_color = None
                self._cached_right_from_left = None
                self._cuda_aligner = None
        with self._frame_lock:
            self._latest_frame = None
            self._delivered_sequence = -1
            self._capture_error = None

    def poll(self) -> CameraFrame | None:
        """Return the newest SDK frame once, never a queued stale frame."""
        with self._frame_lock:
            error = self._capture_error
            self._capture_error = None
            frame = self._latest_frame
            if frame is None or frame.sequence == self._delivered_sequence:
                if error is not None:
                    raise RuntimeError(f"RealSense SDK采集异常: {error}") from error
                return None
            self._delivered_sequence = frame.sequence
            return frame

    def peek_latest(self) -> CameraFrame | None:
        """Expose the newest immutable SDK frame to the UI without consuming it."""
        with self._frame_lock:
            return self._latest_frame

    def frames_after(self, sequence: int) -> list[CameraFrame]:
        """Return every completed frame newer than ``sequence`` in order."""
        with self._frame_lock:
            frames = [frame for frame in self._completed_frames if frame.sequence > sequence]
        frames.sort(key=lambda frame: frame.sequence)
        return frames

    def _poll_sdk_once(self) -> CameraFrame | None:
        if self.pipeline is None:
            return None
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return None
        prepared = self._prepare_frameset(frames)
        return self._process_frameset(*prepared) if prepared is not None else None

    def _prepare_frameset(self, frames):
        arrival_ns = time.perf_counter_ns()
        depth_raw = frames.get_depth_frame()
        color_raw = frames.get_color_frame()
        left_raw = frames.get_infrared_frame(1) if self._ir_streams_enabled else None
        right_raw = frames.get_infrared_frame(2) if self._ir_streams_enabled else None
        if not depth_raw or not color_raw or (
            self._ir_streams_enabled and (not left_raw or not right_raw)
        ):
            return None
        self._refresh_runtime_settings()
        depth_ts = float(depth_raw.get_timestamp())
        capture_ns = self.clock_mapper.update(depth_ts, arrival_ns)
        self.sequence += 1
        return frames, arrival_ns, capture_ns, self.sequence, depth_ts

    def _process_frameset(
        self,
        frames,
        arrival_ns: int,
        capture_ns: int,
        sequence: int,
        depth_ts: float,
    ) -> CameraFrame | None:
        depth_raw = frames.get_depth_frame()
        color_raw = frames.get_color_frame()
        left_raw = frames.get_infrared_frame(1) if self._ir_streams_enabled else None
        right_raw = frames.get_infrared_frame(2) if self._ir_streams_enabled else None
        if not depth_raw or not color_raw or (
            self._ir_streams_enabled and (not left_raw or not right_raw)
        ):
            return None
        runtime_align = self._align_depth_to_color and bool(
            self.config.get("runtime_align_depth_to_color", True)
        )
        cuda_aligned = runtime_align and self._cuda_aligner is not None
        raw_depth_z16 = np.asanyarray(depth_raw.get_data()).copy()
        # On CUDA systems filter once in native stereo coordinates and align
        # that result on the GPU. This is both more faithful to Viewer and far
        # cheaper than filtering a 1920x1080 RGB-aligned depth image.
        native_depth = (
            self._process_depth_frame(depth_raw)
            if cuda_aligned or not runtime_align
            else depth_raw
        )
        native_depth_z16 = np.asanyarray(native_depth.get_data()).copy()
        native_depth_m = (
            native_depth_z16.astype(np.float32) * self._cached_depth_scale
        )
        native_depth_color_bgr = self._colorize_native_depth(native_depth)
        if cuda_aligned:
            depth = None
            color = color_raw
            depth_m = self._cuda_aligner.align_meters(
                native_depth_z16
            )
        elif runtime_align:
            aligner = getattr(self._align_thread_local, "aligner", None)
            if aligner is None:
                aligner = rs.align(rs.stream.color)
                self._align_thread_local.aligner = aligner
            aligned = aligner.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
        else:
            depth = native_depth
            color = color_raw
        if (not cuda_aligned and not depth) or not color:
            return None
        if not cuda_aligned and runtime_align:
            depth = self._process_depth_frame(depth)
            if not depth:
                return None

        if self._cached_color_intrinsics is None:
            color_profile = color.profile.as_video_stream_profile()
            color_intrinsics = _intrinsics_dict(color_profile.get_intrinsics())
            left_intrinsics = color_intrinsics
            right_intrinsics = color_intrinsics
            transform_left_from_color = np.eye(4)
            transform_right_from_left = np.eye(4)
        else:
            color_intrinsics = self._cached_color_intrinsics
            left_intrinsics = self._cached_left_intrinsics
            right_intrinsics = self._cached_right_intrinsics
            transform_left_from_color = self._cached_left_from_color
            transform_right_from_left = self._cached_right_from_left

        if not runtime_align:
            # Reuse the native metric array instead of allocating a second
            # 1280x720 float image in the fast acquisition path.
            depth_m = native_depth_m
        elif not cuda_aligned:
            depth_m = (
                np.asanyarray(depth.get_data()).astype(np.float32)
                * self._cached_depth_scale
            )
        meta = _metadata(depth_raw)
        meta.update({
            "color_frame_counter": int(color_raw.get_frame_number()),
            "depth_frame_counter": int(depth_raw.get_frame_number()),
            "left_frame_counter": int(left_raw.get_frame_number()) if left_raw else -1,
            "right_frame_counter": int(right_raw.get_frame_number()) if right_raw else -1,
            "device": dict(self.device_info),
            "depth_aligned_to_color": bool(runtime_align),
            "native_depth_colorized": native_depth_color_bgr is not None,
        })
        if self._copy_infrared_frames and left_raw and right_raw:
            infrared_left = np.asanyarray(left_raw.get_data()).copy()
            infrared_right = np.asanyarray(right_raw.get_data()).copy()
        else:
            infrared_left = self._empty_infrared
            infrared_right = self._empty_infrared
        return CameraFrame(
            sequence=sequence,
            host_arrival_ns=arrival_ns,
            capture_host_ns=capture_ns,
            device_timestamp_ms=depth_ts,
            color_timestamp_ms=float(color_raw.get_timestamp()),
            depth_timestamp_ms=depth_ts,
            infrared_timestamp_ms=float(left_raw.get_timestamp()) if left_raw else float("nan"),
            color_bgr=np.asanyarray(color.get_data()).copy(),
            depth_m=depth_m,
            infrared_left=infrared_left,
            infrared_right=infrared_right,
            color_intrinsics=color_intrinsics,
            left_intrinsics=left_intrinsics,
            right_intrinsics=right_intrinsics,
            transform_left_from_color=transform_left_from_color,
            transform_right_from_left=transform_right_from_left,
            metadata=meta,
            raw_depth_z16=raw_depth_z16,
            depth_unit_m=self._cached_depth_scale,
            native_depth_m=native_depth_m,
            native_depth_color_bgr=native_depth_color_bgr,
            depth_intrinsics=self._cached_depth_intrinsics or {},
        )


class SyntheticCameraSource:
    """Deterministic source used by tests and UI development without a D435."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.started = False
        self.sequence = 0
        self.start_ns = time.perf_counter_ns()
        self.last_frame_ns = 0
        self.device_info = {"name": "Synthetic D435", "serial": "SIM", "firmware": "n/a", "usb": "virtual"}
        self.transform_base_from_camera = np.eye(4)
        self.transform_base_from_camera[2, 3] = -0.30

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def start_bag_recording(self, _path) -> bool:
        return False

    def stop_bag_recording(self) -> None:
        return None

    def poll(self) -> CameraFrame | None:
        if not self.started:
            return None
        now = time.perf_counter_ns()
        if now - self.last_frame_ns < 30_000_000:
            return None
        self.last_frame_ns = now
        self.sequence += 1
        width, height = 640, 480
        fx = fy = 600.0
        ppx, ppy = width / 2.0, height / 2.0
        baseline = 0.05
        t = (now - self.start_ns) * 1e-9
        color = np.full((height, width, 3), 28, dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32)
        left = np.full((height, width), 170, dtype=np.uint8)
        right = left.copy()
        for index, marker in enumerate(self.config["markers"]):
            s = index / 6.0
            z = 0.30 + 0.01 * np.sin(t * 0.4 + s)
            x = 0.042 * s
            y = 0.018 * np.sin(np.pi * s) * np.sin(t * 0.7)
            u = int(round(fx * x / z + ppx))
            v = int(round(fy * y / z + ppy))
            disparity = fx * baseline / z
            ur = int(round(u - disparity))
            cv2.circle(color, (u, v), 6, tuple(int(c) for c in marker["display_bgr"]), -1, cv2.LINE_AA)
            cv2.circle(depth, (u, v), 6, float(z), -1)
            cv2.circle(left, (u, v), 5, 25, -1, cv2.LINE_AA)
            cv2.circle(right, (ur, v), 5, 25, -1, cv2.LINE_AA)
        intr = {"width": width, "height": height, "fx": fx, "fy": fy, "ppx": ppx, "ppy": ppy, "coeffs": [0.0] * 5, "model": "none"}
        right_from_left = np.eye(4)
        right_from_left[0, 3] = -baseline
        device_ms = (now - self.start_ns) / 1e6
        return CameraFrame(
            sequence=self.sequence,
            host_arrival_ns=now,
            capture_host_ns=now,
            device_timestamp_ms=device_ms,
            color_timestamp_ms=device_ms,
            depth_timestamp_ms=device_ms,
            infrared_timestamp_ms=device_ms,
            color_bgr=color,
            depth_m=depth,
            infrared_left=left,
            infrared_right=right,
            color_intrinsics=intr,
            left_intrinsics=intr,
            right_intrinsics=intr,
            transform_left_from_color=np.eye(4),
            transform_right_from_left=right_from_left,
            metadata={
                "frame_counter": self.sequence,
                "depth_frame_counter": self.sequence,
                "depth_aligned_to_color": True,
                "device": self.device_info,
            },
            raw_depth_z16=np.rint(depth / 0.001).astype(np.uint16),
            depth_unit_m=0.001,
            native_depth_m=depth.copy(),
            depth_intrinsics=intr,
        )


class OpenCvRgbSource:
    """Low-latency auxiliary RGB source with a background grab thread."""

    def __init__(self, config: dict, index: int | None = None) -> None:
        self.config = config
        self.index = int(config.get("index", 0) if index is None else index)
        self.source_name = str(config.get("source_name", "side")).strip() or "side"
        self.display_name = str(
            config.get(
                "display_name",
                "内窥镜摄像头" if self.source_name == "endoscope" else "侧面RGB相机",
            )
        )
        self.capture = None
        self.thread: threading.Thread | None = None
        self.running = False
        self.lock = threading.Lock()
        self.latest: RgbCameraFrame | None = None
        self.sequence = 0
        self.device_info = {
            "name": self.display_name,
            "index": self.index,
            "backend": str(config.get("backend", "dshow")),
        }

    def _backend(self) -> int:
        name = str(self.config.get("backend", "dshow")).lower()
        if name == "dshow" and hasattr(cv2, "CAP_DSHOW"):
            return int(cv2.CAP_DSHOW)
        if name in ("msmf", "mediafoundation") and hasattr(cv2, "CAP_MSMF"):
            return int(cv2.CAP_MSMF)
        return int(cv2.CAP_ANY)

    def start(self) -> None:
        if self.running:
            return
        backend = self._backend()
        capture = cv2.VideoCapture(self.index, backend) if backend else cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开{self.display_name} index={self.index}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config["width"]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config["height"]))
        capture.set(cv2.CAP_PROP_FPS, float(self.config["fps"]))
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, float(self.config.get("buffer_size", 1)))
        if not bool(self.config.get("autofocus", False)) and hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
            capture.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
        if not bool(self.config.get("auto_exposure", False)) and hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        exposure = self.config.get("exposure")
        if exposure is not None:
            capture.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        self.capture = capture
        actual_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        self.device_info.update({
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "fourcc": "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00"),
        })
        self.running = True
        self.thread = threading.Thread(
            target=self._loop, name=f"{self.source_name}-rgb-capture", daemon=True
        )
        self.thread.start()

    def _scaled_intrinsics(self, width: int, height: int) -> dict[str, Any]:
        source = dict(self.config["intrinsics"])
        source_width = max(float(source.get("width", width)), 1.0)
        source_height = max(float(source.get("height", height)), 1.0)
        sx, sy = width / source_width, height / source_height
        return {
            "width": int(width),
            "height": int(height),
            "fx": float(source["fx"]) * sx,
            "fy": float(source["fy"]) * sy,
            "ppx": float(source["ppx"]) * sx,
            "ppy": float(source["ppy"]) * sy,
            "coeffs": [float(value) for value in source.get("coeffs", [0.0] * 5)],
            "model": str(source.get("model", "opencv")),
        }

    def _loop(self) -> None:
        while self.running and self.capture is not None:
            ok, image = self.capture.read()
            arrival_ns = time.perf_counter_ns()
            if not ok or image is None:
                time.sleep(0.005)
                continue
            self.sequence += 1
            height, width = image.shape[:2]
            frame = RgbCameraFrame(
                sequence=self.sequence,
                host_arrival_ns=arrival_ns,
                capture_host_ns=arrival_ns,
                image_bgr=image.copy(),
                intrinsics=self._scaled_intrinsics(width, height),
                metadata={
                    "camera_index": self.index,
                    "width": width,
                    "height": height,
                    "fps_requested": float(self.config["fps"]),
                },
            )
            with self.lock:
                self.latest = frame

    def poll_nearest(self, target_host_ns: int) -> RgbCameraFrame | None:
        with self.lock:
            frame = self.latest
        if frame is None:
            return None
        maximum_age_ns = float(self.config.get("max_frame_age_ms", 60.0)) * 1e6
        if abs(frame.capture_host_ns - int(target_host_ns)) > maximum_age_ns:
            return None
        return frame

    def poll(self) -> RgbCameraFrame | None:
        with self.lock:
            return self.latest

    def set_intrinsics(self, intrinsics: dict[str, Any], ready: bool = True) -> None:
        self.config["intrinsics"] = dict(intrinsics)
        self.config["intrinsics_ready"] = bool(ready)

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        with self.lock:
            self.latest = None


class SyntheticSideRgbSource:
    """Orthogonal RGB view consistent with :class:`SyntheticCameraSource`."""

    def __init__(self, config: dict, primary_source: SyntheticCameraSource | None = None) -> None:
        self.config = config
        self.primary_source = primary_source
        self.started = False
        self.sequence = 0
        self.last_frame_ns = 0
        self.latest: RgbCameraFrame | None = None
        self.start_ns = primary_source.start_ns if primary_source is not None else time.perf_counter_ns()
        self.device_info = {"name": "Synthetic side RGB", "index": "SIM-SIDE", "backend": "virtual"}
        self.transform_base_from_camera = np.eye(4)
        self.transform_base_from_camera[:3, :3] = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ])
        self.transform_base_from_camera[:3, 3] = np.asarray([0.0, -0.30, 0.0])
        self.intrinsics = {
            "width": 640, "height": 480, "fx": 600.0, "fy": 600.0,
            "ppx": 320.0, "ppy": 240.0, "coeffs": [0.0] * 5, "model": "none",
        }

    def start(self) -> None:
        self.started = True

    @property
    def running(self) -> bool:
        return self.started

    def stop(self) -> None:
        self.started = False
        self.latest = None

    def poll_nearest(self, target_host_ns: int) -> RgbCameraFrame | None:
        frame = self.poll()
        if frame is None:
            return None
        return frame if abs(frame.capture_host_ns - int(target_host_ns)) <= 60_000_000 else None

    def poll(self) -> RgbCameraFrame | None:
        if not self.started:
            return None
        now = time.perf_counter_ns()
        if now - self.last_frame_ns < 30_000_000:
            return self.latest
        self.last_frame_ns = now
        self.sequence += 1
        image = np.full((480, 640, 3), 28, dtype=np.uint8)
        t = (now - self.start_ns) * 1e-9
        camera_from_base = np.linalg.inv(self.transform_base_from_camera)
        for index, marker in enumerate(self.config["markers"]):
            s = index / 6.0
            point_base = np.asarray([
                0.042 * s,
                0.018 * np.sin(np.pi * s) * np.sin(t * 0.7),
                0.01 * np.sin(t * 0.4 + s),
                1.0,
            ])
            point = (camera_from_base @ point_base)[:3]
            if point[2] <= 0:
                continue
            u = int(round(self.intrinsics["fx"] * point[0] / point[2] + self.intrinsics["ppx"]))
            v = int(round(self.intrinsics["fy"] * point[1] / point[2] + self.intrinsics["ppy"]))
            cv2.circle(image, (u, v), 6, tuple(int(c) for c in marker["display_bgr"]), -1, cv2.LINE_AA)
        self.latest = RgbCameraFrame(
            sequence=self.sequence,
            host_arrival_ns=now,
            capture_host_ns=now,
            image_bgr=image,
            intrinsics=self.intrinsics,
            metadata={"frame_counter": self.sequence, "device": self.device_info},
        )
        return self.latest
