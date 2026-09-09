"""Standalone PyQt5 application for D435 TDCR data collection."""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import deque
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from PyQt5 import QtMultimedia
except Exception:
    QtMultimedia = None

from .axes import NullAxisSource, SyntheticAxisSource, TrioAxisSource
from .calibration import (
    CharucoBaseCalibrator,
    CharucoIntrinsicCalibrator,
    load_calibration,
    load_intrinsic_calibration,
    save_calibration,
)
from .camera import OpenCvRgbSource, RealSenseSource, SyntheticCameraSource, SyntheticSideRgbSource
from .config import load_config, save_config
from .em import EmMotorCsvRecorder, NdiEmSource, list_ndi_serial_ports
from .engine import CaptureEngine
from .mujoco_bridge import MujocoAlignmentBridge
from .multiview import D435_INTERNAL_MODE, DUAL_VIEW_MODE
from .recording import AuxiliaryRgbRecorder, ViewerStreamRecorder
from .tracking import NumbaCudaColourPreprocessor, numba_cuda_available

try:
    import pyqtgraph.opengl as gl
except Exception:
    gl = None


_REALSENSE_CAMERA_TOKENS = ("realsense", "real sense", "d435", "depth camera 435")


def configure_compute_backend(config: dict) -> str:
    """Select a CPU, OpenCV GPU or PyTorch CUDA preprocessing backend."""

    performance = config.setdefault("performance", {})
    cv2.setUseOptimized(True)
    requested_threads = int(performance.get("opencv_threads", 0))
    if requested_threads > 0:
        cv2.setNumThreads(requested_threads)
    requested = str(performance.get("preprocess_backend", "auto")).lower()
    cuda_devices = int(cv2.cuda.getCudaEnabledDeviceCount()) if hasattr(cv2, "cuda") else 0
    opencl_available = bool(cv2.ocl.haveOpenCL())
    numba_cuda = numba_cuda_available()
    probe = np.zeros((720, 1280, 3), dtype=np.uint8)

    def cpu_probe() -> None:
        resized = cv2.resize(probe, (848, 477), interpolation=cv2.INTER_AREA)
        cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

    def opencl_probe() -> None:
        uploaded = cv2.UMat(probe)
        resized = cv2.resize(uploaded, (848, 477), interpolation=cv2.INTER_AREA)
        cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).get()
        cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).get()

    def opencv_cuda_probe() -> None:
        uploaded = cv2.cuda_GpuMat()
        uploaded.upload(probe)
        resized = cv2.cuda.resize(uploaded, (848, 477), interpolation=cv2.INTER_AREA)
        cv2.cuda.cvtColor(resized, cv2.COLOR_BGR2HSV).download()
        cv2.cuda.cvtColor(resized, cv2.COLOR_BGR2LAB).download()

    numba_preprocessor = NumbaCudaColourPreprocessor() if numba_cuda else None

    def numba_cuda_probe() -> None:
        numba_preprocessor.process(probe, (848, 477))

    def benchmark(function) -> float:
        function()
        function()
        started = time.perf_counter()
        for _ in range(3):
            function()
        return (time.perf_counter() - started) / 3.0

    timings: dict[str, float] = {}
    for name, available, function in (
        ("cpu", True, cpu_probe),
        ("cuda", cuda_devices > 0, opencv_cuda_probe),
        ("numba_cuda", numba_cuda, numba_cuda_probe),
        ("opencl", opencl_available, opencl_probe),
    ):
        if available:
            try:
                timings[name] = benchmark(function)
            except Exception:
                pass
    resolved = "cpu"
    if requested == "auto":
        gpu_candidates = {key: value for key, value in timings.items() if key != "cpu"}
        if gpu_candidates:
            fastest_gpu = min(gpu_candidates, key=gpu_candidates.get)
            # A slightly slower GPU conversion is still useful because it frees CPU
            # cores for IR stereo, side-view detection and Qt painting.
            dual_view = str(config.get("fusion", {}).get("mode", "")) == DUAL_VIEW_MODE
            gpu_tolerance = 1.40 if dual_view else 1.15
            if timings.get(fastest_gpu, float("inf")) <= timings.get("cpu", float("inf")) * gpu_tolerance:
                resolved = fastest_gpu
    elif requested == "cuda":
        resolved = "cuda" if "cuda" in timings else ("numba_cuda" if "numba_cuda" in timings else "cpu")
    elif requested in timings:
        resolved = requested
    performance["resolved_preprocess_backend"] = resolved
    performance["cuda_device_count"] = cuda_devices
    performance["numba_cuda_available"] = numba_cuda
    performance["opencl_available"] = opencl_available
    performance["backend_benchmark_ms"] = {
        key: round(value * 1000.0, 3) for key, value in timings.items()
    }
    cuda_status = "Numba CUDA可用" if numba_cuda else "Numba CUDA不可用"
    return f"{resolved.upper()} · OpenCV {cv2.getNumThreads()}线程 · {cuda_status}"


@dataclass(frozen=True)
class CapturePacket:
    sample: object
    d435_overlay: np.ndarray | None
    side_overlay: np.ndarray | None
    side_frame: object | None
    processing_ms: float
    processing_fps: float


class LatestCaptureWorker:
    """Continuously process camera frames without ever queueing stale UI frames."""

    def __init__(self, engine: CaptureEngine) -> None:
        self.engine = engine
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._version = 0
        self._packet: CapturePacket | None = None
        self._error: Exception | None = None
        self._smoothed_fps = 0.0
        self._last_frame_ns = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="tdcr-capture-processing",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout_s)
        if thread.is_alive():
            raise RuntimeError("采集处理线程未能及时停止")
        self._thread = None

    def latest_after(self, version: int) -> tuple[int, CapturePacket | None, Exception | None]:
        with self._lock:
            error = self._error
            self._error = None
            if self._version == version:
                return version, None, error
            return self._version, self._packet, error

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                sample = self.engine.process_once()
                if sample is None:
                    self._stop_event.wait(0.001)
                    continue
                now_ns = time.perf_counter_ns()
                if self._last_frame_ns:
                    instantaneous = 1e9 / max(now_ns - self._last_frame_ns, 1)
                    self._smoothed_fps = (
                        instantaneous
                        if self._smoothed_fps <= 0.0
                        else 0.12 * instantaneous + 0.88 * self._smoothed_fps
                    )
                self._last_frame_ns = now_ns
                packet = CapturePacket(
                    sample=sample,
                    d435_overlay=self.engine.latest_overlay,
                    side_overlay=self.engine.latest_side_overlay,
                    side_frame=self.engine.latest_side_frame,
                    processing_ms=float(self.engine.last_processing_ms),
                    processing_fps=float(self._smoothed_fps),
                )
                with self._lock:
                    self._version += 1
                    self._packet = packet
            except Exception as exc:
                with self._lock:
                    self._error = exc
                self._stop_event.wait(0.02)


def discover_side_rgb_cameras(camera_infos=None) -> list[dict[str, object]]:
    """Return Qt camera devices mapped to their matching OpenCV indexes.

    Qt and OpenCV's DirectShow backend use the same Windows enumeration order.  We
    retain that original index when filtering the D435, so choosing an item opens
    the intended ordinary RGB camera instead of the RealSense color endpoint.
    """

    if camera_infos is None:
        if QtMultimedia is None:
            return []
        camera_infos = QtMultimedia.QCameraInfo.availableCameras()
    devices: list[dict[str, object]] = []
    for index, info in enumerate(camera_infos):
        try:
            description = str(info.description()).strip()
        except Exception:
            description = ""
        try:
            raw_device_id = info.deviceName()
            if isinstance(raw_device_id, bytes):
                device_id = raw_device_id.decode("utf-8", errors="replace")
            else:
                device_id = str(raw_device_id)
        except Exception:
            device_id = ""
        display_name = description or device_id or f"RGB Camera {index}"
        searchable = f"{display_name} {device_id}".lower()
        if any(token in searchable for token in _REALSENSE_CAMERA_TOKENS):
            continue
        devices.append({
            "index": int(index),
            "name": display_name,
            "device_id": device_id,
        })
    return devices


class SegmentedModeControl(QtWidgets.QFrame):
    """Two-option segmented control with a QComboBox-compatible surface."""

    currentIndexChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("segmentedControl")
        self._items: list[tuple[str, str, QtWidgets.QPushButton]] = []
        self._current_index = -1
        self._compact = False
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)
        self._layout = layout

    def addItem(self, text: str, data: str) -> None:
        button = QtWidgets.QPushButton(text)
        button.setCheckable(True)
        if self._compact:
            button.setFixedHeight(20)
        else:
            button.setMinimumHeight(38)
        index = len(self._items)
        button.clicked.connect(lambda _checked, value=index: self.setCurrentIndex(value))
        self._group.addButton(button, index)
        self._layout.addWidget(button, 1)
        self._items.append((text, data, button))
        if self._current_index < 0:
            self.setCurrentIndex(0)

    def setCompact(self, compact: bool = True) -> None:
        """Use a header-sized segmented control without changing its API."""
        self._compact = bool(compact)
        self.setProperty("compact", self._compact)
        margins = (2, 2, 2, 2) if self._compact else (3, 3, 3, 3)
        self._layout.setContentsMargins(*margins)
        self._layout.setSpacing(1 if self._compact else 2)
        if self._compact:
            self.setFixedHeight(24)
        else:
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
        for _text, _data, button in self._items:
            if self._compact:
                button.setFixedHeight(20)
            else:
                button.setMinimumHeight(38)
                button.setMaximumHeight(16777215)
        self.style().unpolish(self)
        self.style().polish(self)

    def count(self) -> int:
        return len(self._items)

    def findData(self, data: str) -> int:
        for index, (_text, value, _button) in enumerate(self._items):
            if value == data:
                return index
        return -1

    def currentData(self):
        return self._items[self._current_index][1] if 0 <= self._current_index < len(self._items) else None

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._items):
            return
        changed = index != self._current_index
        self._current_index = index
        self._items[index][2].setChecked(True)
        if changed and not self.signalsBlocked():
            self.currentIndexChanged.emit(index)


class StatusPill(QtWidgets.QLabel):
    def __init__(self, text: str, tone: str = "neutral", parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("statusPill")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumHeight(28)
        self.setContentsMargins(10, 0, 10, 0)
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        self.setProperty("tone", tone)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_status(self, text: str, tone: str = "neutral") -> None:
        self.setText(text)
        self.set_tone(tone)


class CollapsiblePanel(QtWidgets.QFrame):
    """Compact disclosure panel styled after the RealSense Viewer sidebar."""

    toggled = QtCore.pyqtSignal(bool)

    def __init__(self, title: str, *, expanded: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("cameraTuningPanel")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header_button = QtWidgets.QToolButton()
        self.header_button.setObjectName("cameraPanelHeader")
        self.header_button.setText(title)
        self.header_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.header_button.setArrowType(QtCore.Qt.RightArrow)
        self.header_button.setCheckable(True)
        self.header_button.setChecked(False)
        self.header_button.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.header_button.toggled.connect(self.set_expanded)
        layout.addWidget(self.header_button)
        self.body = QtWidgets.QWidget()
        self.body.setObjectName("cameraPanelBody")
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(10, 8, 10, 10)
        self.body_layout.setSpacing(7)
        layout.addWidget(self.body)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        self.header_button.blockSignals(True)
        self.header_button.setChecked(expanded)
        self.header_button.blockSignals(False)
        self.header_button.setArrowType(
            QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow
        )
        self.body.setVisible(expanded)
        self.toggled.emit(expanded)
        self.updateGeometry()


class WheelToScrollAreaFilter(QtCore.QObject):
    """Prevent accidental wheel edits and use the wheel only to scroll the panel."""

    def __init__(self, scroll_area: QtWidgets.QScrollArea, parent=None) -> None:
        super().__init__(parent)
        self.scroll_area = scroll_area

    def eventFilter(self, watched, event) -> bool:
        if event.type() != QtCore.QEvent.Wheel:
            return super().eventFilter(watched, event)
        scrollbar = self.scroll_area.verticalScrollBar()
        pixel_delta = event.pixelDelta().y()
        if pixel_delta:
            scroll_delta = pixel_delta
        else:
            wheel_steps = event.angleDelta().y() / 120.0
            scroll_delta = wheel_steps * max(scrollbar.singleStep() * 3, 24)
        scrollbar.setValue(scrollbar.value() - int(round(scroll_delta)))
        event.accept()
        return True


class ZoomableImageLabel(QtWidgets.QLabel):
    """Image viewport with wheel zoom, mouse pan and double-click reset."""

    zoomChanged = QtCore.pyqtSignal(float)
    viewChanged = QtCore.pyqtSignal(float, float, float)
    imageClicked = QtCore.pyqtSignal(int, int)

    def __init__(self, placeholder: str, parent=None) -> None:
        super().__init__(placeholder, parent)
        self._source_pixmap: QtGui.QPixmap | None = None
        self._logical_image_size: tuple[int, int] | None = None
        self._fit_source_aspect = False
        self._source_aspect_ratio = 16.0 / 9.0
        self._zoom = 1.0
        self._center = QtCore.QPointF(0.5, 0.5)
        self._drag_position: QtCore.QPoint | None = None
        self._press_position: QtCore.QPoint | None = None
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self.setToolTip("鼠标滚轮缩放 · 按住左键拖动 · 双击恢复1×")

    def set_fit_source_aspect(self, enabled: bool, fallback_ratio: float = 16.0 / 9.0) -> None:
        """Keep the viewport close to its image aspect ratio without cropping."""
        self._fit_source_aspect = bool(enabled)
        if fallback_ratio > 0:
            self._source_aspect_ratio = float(fallback_ratio)
        if self._fit_source_aspect:
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Preferred,
            )
        self.updateGeometry()

    def hasHeightForWidth(self) -> bool:
        return self._fit_source_aspect

    def heightForWidth(self, width: int) -> int:
        if not self._fit_source_aspect:
            return super().heightForWidth(width)
        return max(1, int(round(width / max(self._source_aspect_ratio, 1e-6))))

    def sizeHint(self) -> QtCore.QSize:
        if self._fit_source_aspect:
            width = 320
            return QtCore.QSize(width, self.heightForWidth(width))
        return super().sizeHint()

    def minimumSizeHint(self) -> QtCore.QSize:
        if self._fit_source_aspect:
            width = 180
            return QtCore.QSize(width, self.heightForWidth(width))
        return super().minimumSizeHint()

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def view_state(self) -> tuple[float, float, float]:
        return self._zoom, self._center.x(), self._center.y()

    @property
    def normalized_view_rect(self) -> tuple[float, float, float, float]:
        half = 0.5 / self._zoom
        return (
            self._center.x() - half,
            self._center.y() - half,
            self._center.x() + half,
            self._center.y() + half,
        )

    def set_bgr_image(
        self,
        image_bgr: np.ndarray,
        *,
        logical_size: tuple[int, int] | None = None,
    ) -> None:
        image_bgr = np.ascontiguousarray(image_bgr, dtype=np.uint8)
        height, width, channels = image_bgr.shape
        if hasattr(QtGui.QImage, "Format_BGR888"):
            image = QtGui.QImage(
                image_bgr.data,
                width,
                height,
                channels * width,
                QtGui.QImage.Format_BGR888,
            )
        else:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(
                rgb.data, width, height, channels * width, QtGui.QImage.Format_RGB888
            )
        self._source_pixmap = QtGui.QPixmap.fromImage(image)
        self._logical_image_size = logical_size or (width, height)
        aspect_ratio = width / max(height, 1)
        if self._fit_source_aspect and abs(aspect_ratio - self._source_aspect_ratio) > 1e-3:
            self._source_aspect_ratio = aspect_ratio
            self.updateGeometry()
        self.update()

    def set_view(self, zoom: float, center_x: float, center_y: float, *, emit: bool = False) -> None:
        previous_zoom = self._zoom
        self._zoom = float(np.clip(zoom, 1.0, 8.0))
        self._center = QtCore.QPointF(float(center_x), float(center_y))
        self._clamp_center()
        if abs(previous_zoom - self._zoom) > 1e-6:
            self.zoomChanged.emit(self._zoom)
        if emit:
            self.viewChanged.emit(self._zoom, self._center.x(), self._center.y())
        self.update()

    def set_zoom(self, zoom: float) -> None:
        value = float(np.clip(zoom, 1.0, 8.0))
        if abs(value - self._zoom) < 1e-6:
            return
        self._zoom = value
        self._clamp_center()
        self.zoomChanged.emit(value)
        self.viewChanged.emit(self._zoom, self._center.x(), self._center.y())
        self.update()

    def reset_zoom(self) -> None:
        self._center = QtCore.QPointF(0.5, 0.5)
        if abs(self._zoom - 1.0) > 1e-6:
            self._zoom = 1.0
            self.zoomChanged.emit(1.0)
        self.viewChanged.emit(1.0, 0.5, 0.5)
        self.update()

    def _clamp_center(self) -> None:
        half = 0.5 / self._zoom
        self._center.setX(float(np.clip(self._center.x(), half, 1.0 - half)))
        self._center.setY(float(np.clip(self._center.y(), half, 1.0 - half)))

    def wheelEvent(self, event) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        self.set_zoom(self._zoom * factor)
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._source_pixmap is not None:
            self._drag_position = event.pos()
            self._press_position = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_position is not None and self._zoom > 1.0:
            delta = event.pos() - self._drag_position
            self._drag_position = event.pos()
            self._center.setX(self._center.x() - delta.x() / max(self.width() * self._zoom, 1.0))
            self._center.setY(self._center.y() - delta.y() / max(self.height() * self._zoom, 1.0))
            self._clamp_center()
            self.viewChanged.emit(self._zoom, self._center.x(), self._center.y())
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            if self._press_position is not None and (
                event.pos() - self._press_position
            ).manhattanLength() <= 4:
                pixel = self._widget_to_source(event.pos())
                if pixel is not None:
                    self.imageClicked.emit(pixel[0], pixel[1])
            self._drag_position = None
            self._press_position = None
            self.setCursor(QtCore.Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event) -> None:
        if self._source_pixmap is None:
            super().paintEvent(event)
            return
        painter = QtGui.QPainter(self)
        # The SDK preview is already resized to the viewport scale.  Bilinear
        # resampling in every paint event is surprisingly expensive on the
        # Windows raster backend and makes an otherwise 30 Hz stream feel
        # delayed.  Keep it only while zoomed, where interpolation is useful.
        painter.setRenderHint(
            QtGui.QPainter.SmoothPixmapTransform, self._zoom > 1.001
        )
        painter.fillRect(self.rect(), QtGui.QColor(16, 17, 20))
        source_width = self._source_pixmap.width() / self._zoom
        source_height = self._source_pixmap.height() / self._zoom
        source_x = self._center.x() * self._source_pixmap.width() - source_width * 0.5
        source_y = self._center.y() * self._source_pixmap.height() - source_height * 0.5
        source_rect = QtCore.QRectF(source_x, source_y, source_width, source_height)
        scaled = QtCore.QSizeF(source_width, source_height)
        scaled.scale(QtCore.QSizeF(self.width(), self.height()), QtCore.Qt.KeepAspectRatio)
        target = QtCore.QRectF(
            (self.width() - scaled.width()) * 0.5,
            (self.height() - scaled.height()) * 0.5,
            scaled.width(),
            scaled.height(),
        )
        painter.drawPixmap(target, self._source_pixmap, source_rect)
        if self._zoom > 1.001:
            badge = f"{self._zoom:.1f}×"
            painter.setPen(QtGui.QColor(245, 245, 247))
            painter.setBrush(QtGui.QColor(0, 0, 0, 145))
            badge_rect = QtCore.QRectF(9, self.height() - 32, 54, 23)
            painter.drawRoundedRect(badge_rect, 8, 8)
            painter.drawText(badge_rect, QtCore.Qt.AlignCenter, badge)

    def _view_rectangles(self) -> tuple[QtCore.QRectF, QtCore.QRectF] | None:
        if self._source_pixmap is None or self.width() <= 0 or self.height() <= 0:
            return None
        source_width = self._source_pixmap.width() / self._zoom
        source_height = self._source_pixmap.height() / self._zoom
        source_rect = QtCore.QRectF(
            self._center.x() * self._source_pixmap.width() - source_width * 0.5,
            self._center.y() * self._source_pixmap.height() - source_height * 0.5,
            source_width,
            source_height,
        )
        scaled = QtCore.QSizeF(source_width, source_height)
        scaled.scale(QtCore.QSizeF(self.width(), self.height()), QtCore.Qt.KeepAspectRatio)
        target_rect = QtCore.QRectF(
            (self.width() - scaled.width()) * 0.5,
            (self.height() - scaled.height()) * 0.5,
            scaled.width(),
            scaled.height(),
        )
        return source_rect, target_rect

    def _widget_to_source(self, position: QtCore.QPoint) -> tuple[int, int] | None:
        rectangles = self._view_rectangles()
        if rectangles is None:
            return None
        source_rect, target_rect = rectangles
        if not target_rect.contains(QtCore.QPointF(position)):
            return None
        relative_x = (position.x() - target_rect.left()) / max(target_rect.width(), 1e-6)
        relative_y = (position.y() - target_rect.top()) / max(target_rect.height(), 1e-6)
        u = int(np.clip(round(source_rect.left() + relative_x * source_rect.width()), 0, self._source_pixmap.width() - 1))
        v = int(np.clip(round(source_rect.top() + relative_y * source_rect.height()), 0, self._source_pixmap.height() - 1))
        if self._logical_image_size is not None:
            logical_width, logical_height = self._logical_image_size
            u = int(np.clip(round(u * logical_width / max(self._source_pixmap.width(), 1)), 0, logical_width - 1))
            v = int(np.clip(round(v * logical_height / max(self._source_pixmap.height(), 1)), 0, logical_height - 1))
        return u, v


class ShapeView(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.available = gl is not None
        if not self.available:
            self.fallback = QtWidgets.QLabel("3D视图不可用")
            self.fallback.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(self.fallback)
            return
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=0.10, elevation=25, azimuth=-60)
        self.real_line = gl.GLLinePlotItem(color=(0.1, 1.0, 0.3, 1.0), width=4, antialias=True)
        self.sim_line = gl.GLLinePlotItem(color=(0.1, 0.5, 1.0, 1.0), width=3, antialias=True)
        self.real_points = gl.GLScatterPlotItem(color=(0.1, 1.0, 0.3, 1.0), size=10)
        self.sim_points = gl.GLScatterPlotItem(color=(0.1, 0.5, 1.0, 1.0), size=8)
        grid = gl.GLGridItem()
        grid.scale(0.005, 0.005, 0.005)
        self.view.addItem(grid)
        for item in (self.real_line, self.sim_line, self.real_points, self.sim_points):
            self.view.addItem(item)
        layout.addWidget(self.view)

    def update_shapes(self, real: np.ndarray, simulation: np.ndarray) -> None:
        if not self.available:
            return
        real = np.asarray(real, dtype=float)
        simulation = np.asarray(simulation, dtype=float)
        real_valid = real[np.isfinite(real).all(axis=1)]
        sim_valid = simulation[np.isfinite(simulation).all(axis=1)]
        self.real_line.setData(pos=real_valid if len(real_valid) else np.zeros((0, 3)))
        self.real_points.setData(pos=real_valid if len(real_valid) else np.zeros((0, 3)))
        self.sim_line.setData(pos=sim_valid if len(sim_valid) else np.zeros((0, 3)))
        self.sim_points.setData(pos=sim_valid if len(sim_valid) else np.zeros((0, 3)))


def apply_apple_product_style(widget: QtWidgets.QWidget) -> None:
    widget.setStyleSheet(
        """
        QWidget {
            color: #1D1D1F;
            font-family: "Microsoft YaHei UI";
            font-size: 13px;
        }
        QWidget#appSurface { background: #F5F5F7; }
        QFrame#topBar {
            background: rgba(255, 255, 255, 242);
            border: 1px solid #E5E5EA;
            border-radius: 18px;
        }
        QLabel#productTitle { font-size: 24px; font-weight: 700; color: #1D1D1F; }
        QLabel#productSubtitle { font-size: 12px; color: #6E6E73; }
        QLabel#viewTitle { font-size: 14px; font-weight: 600; color: #1D1D1F; }
        QLabel#viewSubtitle { font-size: 11px; color: #86868B; }
        QLabel#statusPill {
            border-radius: 14px;
            font-size: 11px;
            font-weight: 600;
            padding: 0 4px;
        }
        QLabel#statusPill[tone="neutral"] { background: #ECECF0; color: #515154; }
        QLabel#statusPill[tone="blue"] { background: #E7F2FF; color: #0066CC; }
        QLabel#statusPill[tone="green"] { background: #E8F8ED; color: #188038; }
        QLabel#statusPill[tone="orange"] { background: #FFF3E0; color: #B35A00; }
        QLabel#statusPill[tone="red"] { background: #FFE9E7; color: #C5221F; }
        QFrame#cameraCard, QFrame#shapeCard {
            background: #FFFFFF;
            border: 1px solid #E5E5EA;
            border-radius: 18px;
        }
        QFrame#cameraViewport {
            background: #101114;
            border: none;
            border-radius: 13px;
        }
        QFrame#streamTile {
            background: #101114;
            border: 1px solid #282A30;
            border-radius: 12px;
        }
        QFrame#streamTile[compact="true"] {
            border-color: #22242A;
            border-radius: 9px;
        }
        QFrame#cameraTuningPanel {
            background: #F7F8FA;
            border: 1px solid #D9DCE2;
            border-radius: 10px;
        }
        QToolButton#cameraPanelHeader {
            min-height: 34px;
            background: #ECEEF2;
            border: none;
            border-radius: 9px;
            padding: 0 10px;
            color: #303136;
            font-size: 12px;
            font-weight: 700;
            text-align: left;
        }
        QToolButton#cameraPanelHeader:hover { background: #E4E7EC; }
        QWidget#cameraPanelBody { background: #F7F8FA; border: none; }
        QLabel#cameraSectionLabel {
            color: #767980;
            font-size: 10px;
            font-weight: 700;
            padding-top: 3px;
        }
        QLabel#cameraParameterLabel { color: #3A3A3C; font-size: 11px; }
        QLabel#cameraParameterStatus {
            color: #5F6368;
            background: #FFFFFF;
            border: 1px solid #E1E3E8;
            border-radius: 7px;
            padding: 6px 8px;
            font-size: 10px;
        }
        QLabel#streamTitle {
            color: #F5F5F7;
            font-size: 11px;
            font-weight: 600;
        }
        QLabel#streamMeta {
            color: #8E8E93;
            font-size: 10px;
        }
        QFrame#inspectorSurface {
            background: #F5F5F7;
            border: none;
        }
        QScrollArea { background: transparent; border: none; }
        QScrollArea > QWidget > QWidget { background: transparent; }
        QGroupBox {
            background: #FFFFFF;
            border: 1px solid #E5E5EA;
            border-radius: 15px;
            margin-top: 15px;
            padding: 15px 12px 12px 12px;
            font-weight: 600;
            color: #1D1D1F;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 13px;
            padding: 0 6px;
            color: #515154;
            background: #F5F5F7;
        }
        QFrame#segmentedControl {
            background: #ECECF0;
            border: 1px solid #DEDEE3;
            border-radius: 11px;
        }
        QFrame#segmentedControl QPushButton {
            min-height: 34px;
            border: none;
            border-radius: 8px;
            background: transparent;
            color: #515154;
            font-size: 12px;
            font-weight: 600;
            padding: 0 10px;
        }
        QFrame#segmentedControl QPushButton:checked {
            background: #FFFFFF;
            color: #0071E3;
            border: 1px solid #D8D8DC;
        }
        QFrame#segmentedControl[compact="true"] {
            border-radius: 7px;
        }
        QFrame#segmentedControl[compact="true"] QPushButton {
            min-height: 20px;
            max-height: 20px;
            border-radius: 5px;
            padding: 0 5px;
            font-size: 10px;
        }
        QPushButton {
            min-height: 34px;
            border: 1px solid #D2D2D7;
            border-radius: 9px;
            background: #FFFFFF;
            color: #1D1D1F;
            padding: 0 13px;
            font-weight: 600;
        }
        QPushButton:hover { background: #F2F2F4; border-color: #B8B8BD; }
        QPushButton:pressed { background: #E8E8ED; }
        QPushButton:disabled { color: #AEAEB2; background: #F2F2F4; border-color: #E5E5EA; }
        QPushButton[role="primary"] { background: #0071E3; border-color: #0071E3; color: white; }
        QPushButton[role="primary"]:hover { background: #0077ED; border-color: #0077ED; }
        QPushButton[role="danger"] { background: #FF3B30; border-color: #FF3B30; color: white; }
        QPushButton[role="soft"] { background: #F2F2F7; border-color: transparent; color: #0071E3; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            min-height: 34px;
            background: #F2F2F7;
            border: 1px solid transparent;
            border-radius: 9px;
            padding: 0 10px;
            selection-background-color: #0071E3;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            background: #FFFFFF;
            border: 1px solid #0071E3;
        }
        QComboBox::drop-down { border: none; width: 26px; }
        QSlider::groove:horizontal {
            height: 4px;
            background: #D8DAE0;
            border-radius: 2px;
        }
        QSlider::sub-page:horizontal { background: #0071E3; border-radius: 2px; }
        QSlider::handle:horizontal {
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
            background: #FFFFFF;
            border: 1px solid #AEB1B8;
        }
        QCheckBox { spacing: 8px; font-weight: 600; }
        QCheckBox::indicator { width: 18px; height: 18px; }
        QCheckBox::indicator:unchecked { border: 1px solid #C7C7CC; border-radius: 5px; background: white; }
        QCheckBox::indicator:checked { border: 1px solid #0071E3; border-radius: 5px; background: #0071E3; }
        QPlainTextEdit {
            background: #FFFFFF;
            border: 1px solid #E5E5EA;
            border-radius: 14px;
            padding: 10px;
            color: #3A3A3C;
            font-family: "Cascadia Mono", "Consolas";
            font-size: 11px;
        }
        QSplitter::handle { background: transparent; width: 8px; }
        QScrollBar:vertical { background: transparent; width: 8px; margin: 4px 1px; }
        QScrollBar::handle:vertical { background: #C7C7CC; min-height: 36px; border-radius: 4px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QStatusBar { background: #F5F5F7; color: #6E6E73; border-top: 1px solid #E5E5EA; }
        """
    )


class D435CaptureWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        config: dict,
        camera_source=None,
        axis_source=None,
        mujoco_bridge=None,
        side_camera_source=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("TDCR Capture Studio")
        self.resize(1580, 940)
        self.setMinimumSize(1240, 760)
        self.config = config
        self.workflow_mode = str(config.get("ui", {}).get("workflow_mode", "capture"))
        if self.workflow_mode not in ("capture", "experiment"):
            self.workflow_mode = "capture"
        self.compute_backend_info = configure_compute_backend(config)
        self.camera_source = camera_source or RealSenseSource(config["camera"])
        self.side_camera_source = side_camera_source
        self._embedded_mujoco_bridge = mujoco_bridge
        self._embedded_mujoco_xml = str(config["mujoco"]["xml"]) if mujoco_bridge is not None else None
        self.identify_process = None
        self.last_session_path = None
        initial_mujoco_bridge = mujoco_bridge if bool(config["mujoco"].get("enabled", False)) else None
        self.engine = CaptureEngine(
            config, self.camera_source, axis_source, initial_mujoco_bridge, side_camera_source
        )
        if self.workflow_mode == "capture":
            self.engine.set_keypoint_tracking_enabled(False)
            self.engine.set_mujoco_bridge(None)
        self.viewer_recorder = ViewerStreamRecorder(config)
        self.endoscope_source = None
        self.endoscope_recorder = AuxiliaryRgbRecorder(config)
        self.em_source = NdiEmSource(config.setdefault("em", {}))
        self.em_recorder = EmMotorCsvRecorder(config["em"])
        self.capture_worker = LatestCaptureWorker(self.engine)
        self._capture_packet_version = 0
        self._cloud_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tdcr-cloud-render"
        )
        self._cloud_future: concurrent.futures.Future | None = None
        self._cloud_future_capture_ns: int | None = None
        self._depth_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tdcr-depth-render"
        )
        self._depth_future: concurrent.futures.Future | None = None
        self._rgb_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tdcr-rgb-render"
        )
        self._rgb_future: concurrent.futures.Future | None = None
        self._cloud_revision = 0
        self._displayed_cloud_revision = -1
        self.calibrator = CharucoBaseCalibrator(config["calibration"])
        self.side_calibrator = CharucoBaseCalibrator(config["calibration"])
        self.side_intrinsic_calibrator = CharucoIntrinsicCalibrator(config["calibration"])
        self.calibration_preview = None
        self.calibration_preview_target = None
        self._owns_axis_source = isinstance(axis_source, TrioAxisSource)
        self._side_device_signature = None
        self._endoscope_device_signature = None
        self._last_endoscope_sequence = -1
        self._depth_preview_m = None
        self._depth_preview_age = None
        self._cloud_depth_preview_m = None
        self._cloud_depth_preview_age = None
        self._depth_preview_filter_lock = threading.Lock()
        self._cloud_depth_filter_lock = threading.Lock()
        self._latest_sample_for_inspection = None
        self._ir_xyz_cache: dict[tuple[int, str], np.ndarray] = {}
        self._point_cloud_pick_left_m = None
        self._point_cloud_pick_base_m = None
        self._last_preview_ns = 0
        self._sdk_frame_samples = deque(maxlen=120)
        self._camera_processing_demand_signature = None
        self._last_depth_preview_ns = 0
        self._last_direct_sdk_sequence = -1
        self._last_telemetry_ns = 0
        self._last_axis_ui_ns = 0
        self._last_em_ui_ns = 0
        self._latest_axis_ui_sample = None
        self._last_point_cloud_ns = 0
        self._cached_point_cloud_visual = None
        self._setup_ui()
        self._update_camera_processing_demand(force=True)
        self._apply_workflow_mode_ui()
        apply_apple_product_style(self)
        self._refresh_side_camera_devices(force=True)
        self._refresh_endoscope_camera_devices(force=True)
        self._refresh_ndi_ports()
        self._update_header_status()
        side_selection_available = (
            isinstance(self.camera_source, SyntheticCameraSource)
            or isinstance(self.side_camera_combo.currentData(), int)
        )
        if (
            self.engine.fusion_mode == DUAL_VIEW_MODE
            and self.engine.side_camera_source is None
            and side_selection_available
        ):
            self._reconnect_side_camera()
        if self.endoscope_enabled_check.isChecked():
            self._reconnect_endoscope_camera(show_errors=False)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(8)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        self.camera_discovery_timer = QtCore.QTimer(self)
        self.camera_discovery_timer.setInterval(3000)
        self.camera_discovery_timer.timeout.connect(self._refresh_all_rgb_camera_devices)
        self.camera_discovery_timer.start()

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget()
        central.setObjectName("appSurface")
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 12)
        root.setSpacing(14)

        top_bar = QtWidgets.QFrame()
        top_bar.setObjectName("topBar")
        top_bar.setMinimumHeight(76)
        top_layout = QtWidgets.QHBoxLayout(top_bar)
        top_layout.setContentsMargins(20, 12, 18, 12)
        title_column = QtWidgets.QVBoxLayout()
        title_column.setSpacing(1)
        product_title = QtWidgets.QLabel("TDCR Capture Studio")
        product_title.setObjectName("productTitle")
        product_subtitle = QtWidgets.QLabel("双机位三维感知 · MuJoCo动态对齐 · 参数辨识")
        product_subtitle.setObjectName("productSubtitle")
        title_column.addWidget(product_title)
        title_column.addWidget(product_subtitle)
        top_layout.addLayout(title_column)
        top_layout.addSpacing(24)
        self.workflow_mode_control = SegmentedModeControl()
        self.workflow_mode_control.setFixedWidth(260)
        self.workflow_mode_control.addItem("纯采集", "capture")
        self.workflow_mode_control.addItem("实验模式", "experiment")
        workflow_index = self.workflow_mode_control.findData(self.workflow_mode)
        self.workflow_mode_control.setCurrentIndex(max(workflow_index, 0))
        self.workflow_mode_control.currentIndexChanged.connect(self._change_workflow_mode)
        top_layout.addWidget(self.workflow_mode_control)
        top_layout.addStretch(1)
        self.header_mode_pill = StatusPill("D435内部", "blue")
        self.header_tracking_pill = StatusPill("关键点开启", "green")
        self.header_device_pill = StatusPill("相机未启动", "neutral")
        self.header_record_pill = StatusPill("未记录", "neutral")
        top_layout.addWidget(self.header_mode_pill)
        top_layout.addWidget(self.header_tracking_pill)
        top_layout.addWidget(self.header_device_pill)
        top_layout.addWidget(self.header_record_pill)
        root.addWidget(top_bar)

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(14)
        self.image_label = ZoomableImageLabel("点击“启动相机”连接D435")
        self.depth_image_label = ZoomableImageLabel("等待Depth数据")
        self.stereo_cloud_image_label = ZoomableImageLabel("等待左右IR双目稠密点云")
        self.side_image_label = ZoomableImageLabel("侧面RGB相机")
        self.endoscope_image_label = ZoomableImageLabel("等待内窥镜摄像头")
        self._zoom_labels = (
            self.image_label,
            self.depth_image_label,
            self.stereo_cloud_image_label,
            self.side_image_label,
            self.endoscope_image_label,
        )
        self._d435_zoom_labels = (
            self.image_label,
            self.depth_image_label,
            self.stereo_cloud_image_label,
        )
        for label in self._zoom_labels:
            label.viewChanged.connect(
                lambda zoom, center_x, center_y, source=label: self._synchronize_image_views(
                    source, zoom, center_x, center_y
                )
            )
        self.image_label.imageClicked.connect(lambda u, v: self._inspect_image_pixel("color", u, v))
        self.depth_image_label.imageClicked.connect(lambda u, v: self._inspect_image_pixel("depth", u, v))
        self.stereo_cloud_image_label.imageClicked.connect(
            lambda u, v: self._inspect_image_pixel("point_cloud", u, v)
        )
        self.side_image_label.imageClicked.connect(lambda u, v: self._inspect_image_pixel("side", u, v))
        for image_label in (
            self.image_label,
            self.depth_image_label,
            self.stereo_cloud_image_label,
            self.side_image_label,
            self.endoscope_image_label,
        ):
            image_label.setMinimumSize(180, 130)
            image_label.setAlignment(QtCore.Qt.AlignCenter)
            image_label.setStyleSheet("background:transparent;color:#AEAEB2;border:none;")
        for image_label in (self.image_label, self.depth_image_label):
            image_label.setMinimumSize(180, 101)
            image_label.set_fit_source_aspect(True)

        def card_header(
            layout,
            title: str,
            subtitle: str,
            pill: StatusPill,
            control: QtWidgets.QWidget | None = None,
        ) -> None:
            header = QtWidgets.QHBoxLayout()
            labels = QtWidgets.QVBoxLayout()
            labels.setSpacing(0)
            title_label = QtWidgets.QLabel(title)
            title_label.setObjectName("viewTitle")
            subtitle_label = QtWidgets.QLabel(subtitle)
            subtitle_label.setObjectName("viewSubtitle")
            labels.addWidget(title_label)
            labels.addWidget(subtitle_label)
            header.addLayout(labels)
            header.addStretch(1)
            if control is not None:
                header.addWidget(control)
            header.addWidget(pill)
            layout.addLayout(header)

        def stream_tile(
            title: str,
            meta: str,
            image_label: QtWidgets.QLabel,
            control: QtWidgets.QWidget | None = None,
            *,
            compact: bool = False,
        ):
            tile = QtWidgets.QFrame()
            tile.setObjectName("streamTile")
            tile.setProperty("compact", compact)
            if compact:
                tile.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding,
                    QtWidgets.QSizePolicy.Preferred,
                )
            tile_layout = QtWidgets.QVBoxLayout(tile)
            if compact:
                tile_layout.setContentsMargins(4, 3, 4, 4)
                tile_layout.setSpacing(2)
            else:
                tile_layout.setContentsMargins(8, 6, 8, 8)
                tile_layout.setSpacing(5)
            stream_header = QtWidgets.QHBoxLayout()
            title_label = QtWidgets.QLabel(title)
            title_label.setObjectName("streamTitle")
            meta_label = QtWidgets.QLabel(meta)
            meta_label.setObjectName("streamMeta")
            stream_header.addWidget(title_label)
            stream_header.addStretch(1)
            stream_header.addWidget(meta_label)
            if control is not None:
                stream_header.addWidget(control)
            tile_layout.addLayout(stream_header)
            tile_layout.addWidget(image_label, 0 if compact else 1)
            return tile

        def camera_card(
            title: str,
            subtitle: str,
            image_label: QtWidgets.QLabel,
            pill: StatusPill,
            header_control: QtWidgets.QWidget | None = None,
        ):
            card = QtWidgets.QFrame()
            card.setObjectName("cameraCard")
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 12)
            card_layout.setSpacing(9)
            card_header(card_layout, title, subtitle, pill, header_control)
            viewport = QtWidgets.QFrame()
            viewport.setObjectName("cameraViewport")
            viewport_layout = QtWidgets.QVBoxLayout(viewport)
            viewport_layout.setContentsMargins(0, 0, 0, 0)
            viewport_layout.addWidget(image_label)
            card_layout.addWidget(viewport, 1)
            return card

        self.d435_view_pill = StatusPill("主视图", "blue")
        self.side_view_pill = StatusPill("辅助视图", "neutral")
        self.d435_camera_card = QtWidgets.QFrame()
        self.d435_camera_card.setObjectName("cameraCard")
        self.d435_camera_card.setMinimumWidth(620)
        d435_layout = QtWidgets.QVBoxLayout(self.d435_camera_card)
        d435_layout.setContentsMargins(12, 10, 12, 12)
        d435_layout.setSpacing(9)
        self.d435_display_mode = SegmentedModeControl()
        self.d435_display_mode.setFixedWidth(150)
        self.d435_display_mode.addItem("2D", "2d")
        self.d435_display_mode.addItem("3D", "3d")
        display_index = self.d435_display_mode.findData(
            str(self.config.get("ui", {}).get("view_mode", "2d"))
        )
        self.d435_display_mode.setCurrentIndex(max(display_index, 0))
        self.d435_display_mode.currentIndexChanged.connect(self._change_d435_view_mode)
        card_header(
            d435_layout,
            "RealSense D435 Viewer",
            "RGB · DEPTH · 2D/3D",
            self.d435_view_pill,
            self.d435_display_mode,
        )
        self.d435_view_stack = QtWidgets.QStackedWidget()
        d435_2d_page = QtWidgets.QWidget()
        d435_grid = QtWidgets.QGridLayout()
        d435_grid.setContentsMargins(0, 0, 0, 0)
        d435_grid.setHorizontalSpacing(6)
        d435_grid.setVerticalSpacing(6)
        self.rgb_zoom_reset = QtWidgets.QPushButton("1×")
        self.rgb_zoom_reset.setFixedSize(42, 26)
        self.rgb_zoom_reset.setToolTip("恢复主RGB视图缩放")
        self.rgb_zoom_reset.clicked.connect(self.image_label.reset_zoom)
        self.image_label.zoomChanged.connect(lambda value: self.rgb_zoom_reset.setText(f"{value:.1f}×" if value > 1.01 else "1×"))
        self.depth_space_mode = SegmentedModeControl()
        self.depth_space_mode.setCompact(True)
        self.depth_space_mode.setFixedWidth(92)
        self.depth_space_mode.addItem("原生", "native")
        self.depth_space_mode.addItem("对齐", "aligned")
        self.depth_space_mode.setToolTip(
            "原生：Stereo Module深度；对齐：Depth→RGB深度"
        )
        depth_space_index = self.depth_space_mode.findData(
            str(self.config.get("ui", {}).get("depth_view_space", "native"))
        )
        self.depth_space_mode.setCurrentIndex(max(depth_space_index, 0))
        self.depth_space_mode.currentIndexChanged.connect(
            self._change_depth_view_space
        )
        d435_grid.addWidget(
            stream_tile(
                "RGB",
                "COLOR",
                self.image_label,
                self.rgb_zoom_reset,
                compact=True,
            ),
            0,
            0,
        )
        d435_grid.addWidget(
            stream_tile(
                "Depth",
                "DEPTH",
                self.depth_image_label,
                self.depth_space_mode,
                compact=True,
            ),
            0,
            1,
        )
        d435_grid.setColumnStretch(0, 1)
        d435_grid.setColumnStretch(1, 1)
        d435_grid.setRowStretch(0, 1)
        d435_2d_page.setLayout(d435_grid)
        d435_3d_page = QtWidgets.QWidget()
        d435_3d_layout = QtWidgets.QVBoxLayout(d435_3d_page)
        d435_3d_layout.setContentsMargins(0, 0, 0, 0)
        d435_3d_layout.addWidget(
            stream_tile(
                "彩色三维点云",
                "DEPTH + RGB 3D",
                self.stereo_cloud_image_label,
            )
        )
        self.d435_view_stack.addWidget(d435_2d_page)
        self.d435_view_stack.addWidget(d435_3d_page)
        self.d435_view_stack.setCurrentIndex(max(display_index, 0))
        d435_layout.addWidget(self.d435_view_stack, 1)
        self.side_zoom_reset = QtWidgets.QPushButton("1×")
        self.side_zoom_reset.setFixedSize(42, 26)
        self.side_zoom_reset.setToolTip("恢复侧面RGB视图缩放")
        self.side_zoom_reset.clicked.connect(self.side_image_label.reset_zoom)
        self.side_image_label.zoomChanged.connect(lambda value: self.side_zoom_reset.setText(f"{value:.1f}×" if value > 1.01 else "1×"))
        self.side_camera_card = camera_card(
            "侧面 RGB",
            "二维观测 · 重投影验证 · 滚轮缩放",
            self.side_image_label,
            self.side_view_pill,
            self.side_zoom_reset,
        )
        self.endoscope_view_pill = StatusPill("未连接", "neutral")
        self.endoscope_zoom_reset = QtWidgets.QPushButton("1×")
        self.endoscope_zoom_reset.setFixedSize(42, 26)
        self.endoscope_zoom_reset.setToolTip("恢复内窥镜RGB视图缩放")
        self.endoscope_zoom_reset.clicked.connect(self.endoscope_image_label.reset_zoom)
        self.endoscope_image_label.zoomChanged.connect(
            lambda value: self.endoscope_zoom_reset.setText(
                f"{value:.1f}×" if value > 1.01 else "1×"
            )
        )
        self.endoscope_camera_card = camera_card(
            "内窥镜 RGB",
            "独立采集线程 · 独立MP4记录 · 滚轮缩放",
            self.endoscope_image_label,
            self.endoscope_view_pill,
            self.endoscope_zoom_reset,
        )
        self.shape_card = QtWidgets.QFrame()
        self.shape_card.setObjectName("shapeCard")
        shape_layout = QtWidgets.QVBoxLayout(self.shape_card)
        shape_layout.setContentsMargins(8, 8, 8, 8)
        shape_layout.setSpacing(7)

        self.camera_parameter_controls: dict[str, QtWidgets.QWidget] = {}
        self.camera_tuning_panel = CollapsiblePanel(
            "D435 CAMERA CONTROLS",
            expanded=bool(self.config.get("ui", {}).get("camera_tuning_expanded", False)),
        )
        self.camera_tuning_panel.toggled.connect(
            lambda expanded: self.config.setdefault("ui", {}).__setitem__(
                "camera_tuning_expanded", bool(expanded)
            )
        )
        tuning_scroll = QtWidgets.QScrollArea()
        tuning_scroll.setWidgetResizable(True)
        tuning_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        tuning_scroll.setMaximumHeight(330)
        self.camera_parameter_wheel_filter = WheelToScrollAreaFilter(tuning_scroll, self)
        tuning_content = QtWidgets.QWidget()
        tuning_layout = QtWidgets.QVBoxLayout(tuning_content)
        tuning_layout.setContentsMargins(0, 0, 4, 0)
        tuning_layout.setSpacing(7)
        def tuning_section(text: str) -> None:
            label = QtWidgets.QLabel(text.upper())
            label.setObjectName("cameraSectionLabel")
            tuning_layout.addWidget(label)

        def tuning_check(text: str, key: str) -> QtWidgets.QCheckBox:
            control = QtWidgets.QCheckBox(text)
            control.setChecked(bool(self.config["camera"].get(key, False)))
            control.toggled.connect(
                lambda checked, name=key: self._set_camera_parameter(name, bool(checked))
            )
            tuning_layout.addWidget(control)
            self.camera_parameter_controls[key] = control
            return control

        def tuning_slider(
            text: str,
            key: str,
            minimum: float,
            maximum: float,
            step: float,
            *,
            decimals: int = 0,
            suffix: str = "",
        ) -> QtWidgets.QAbstractSpinBox:
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QGridLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setHorizontalSpacing(7)
            row_layout.setVerticalSpacing(2)
            label = QtWidgets.QLabel(text)
            label.setObjectName("cameraParameterLabel")
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            scale = 10 ** decimals
            slider.setRange(int(round(minimum * scale)), int(round(maximum * scale)))
            slider.setSingleStep(max(1, int(round(step * scale))))
            if decimals:
                spin: QtWidgets.QAbstractSpinBox = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(decimals)
                spin.setRange(minimum, maximum)
                spin.setSingleStep(step)
            else:
                spin = QtWidgets.QSpinBox()
                spin.setRange(int(round(minimum)), int(round(maximum)))
                spin.setSingleStep(max(1, int(round(step))))
            spin.setSuffix(suffix)
            spin.setFixedWidth(92)
            slider.installEventFilter(self.camera_parameter_wheel_filter)
            spin.installEventFilter(self.camera_parameter_wheel_filter)
            value = float(self.config["camera"].get(key, minimum))
            slider.setValue(int(round(value * scale)))
            spin.setValue(value if decimals else int(round(value)))
            slider.valueChanged.connect(lambda raw, target=spin, divisor=scale: target.setValue(raw / divisor))
            spin.valueChanged.connect(lambda current, target=slider, multiplier=scale: target.setValue(int(round(float(current) * multiplier))))
            spin.valueChanged.connect(lambda current, name=key: self._set_camera_parameter(name, current))
            row_layout.addWidget(label, 0, 0, 1, 2)
            row_layout.addWidget(slider, 1, 0)
            row_layout.addWidget(spin, 1, 1)
            tuning_layout.addWidget(row)
            self.camera_parameter_controls[key] = spin
            return spin

        tuning_section("Stereo Module")
        preset_row = QtWidgets.QHBoxLayout()
        preset_label = QtWidgets.QLabel("Visual Preset")
        preset_label.setObjectName("cameraParameterLabel")
        self.visual_preset_combo = QtWidgets.QComboBox()
        self.visual_preset_combo.addItem("High Accuracy · 推荐", "high_accuracy")
        self.visual_preset_combo.addItem("High Density", "high_density")
        self.visual_preset_combo.addItem("Medium Density", "medium_density")
        self.visual_preset_combo.addItem("Default", "default")
        preset_index = self.visual_preset_combo.findData(
            str(self.config["camera"].get("visual_preset", "high_accuracy"))
        )
        self.visual_preset_combo.setCurrentIndex(max(0, preset_index))
        self.visual_preset_combo.installEventFilter(self.camera_parameter_wheel_filter)
        self.visual_preset_combo.currentIndexChanged.connect(
            lambda _index: self._set_camera_parameter(
                "visual_preset", str(self.visual_preset_combo.currentData())
            )
        )
        preset_row.addWidget(preset_label)
        preset_row.addWidget(self.visual_preset_combo, 1)
        tuning_layout.addLayout(preset_row)
        emitter_check = tuning_check("启用红外散斑投射器", "emitter_enabled")
        laser_spin = tuning_slider("Laser Power", "laser_power", 0, 360, 5)
        depth_auto_check = tuning_check("Depth Auto Exposure", "depth_auto_exposure")
        depth_exposure_spin = tuning_slider(
            "Depth Exposure", "depth_exposure_us", 1, 33000, 100, suffix=" μs"
        )
        depth_gain_spin = tuning_slider("Depth Gain", "depth_gain", 16, 248, 1)
        emitter_check.toggled.connect(laser_spin.setEnabled)
        depth_auto_check.toggled.connect(lambda checked: depth_exposure_spin.setEnabled(not checked))
        depth_auto_check.toggled.connect(lambda checked: depth_gain_spin.setEnabled(not checked))
        laser_spin.setEnabled(emitter_check.isChecked())
        depth_exposure_spin.setEnabled(not depth_auto_check.isChecked())
        depth_gain_spin.setEnabled(not depth_auto_check.isChecked())

        tuning_section("Depth Post-Processing")
        tuning_check("启用SDK视差域后处理", "depth_postprocess_enabled")
        tuning_check("原生Depth使用SDK Colorizer", "native_depth_colorizer_enabled")
        tuning_check("SDK Colorizer直方图均衡", "depth_colorizer_histogram_equalization")
        colorizer_row = QtWidgets.QHBoxLayout()
        colorizer_label = QtWidgets.QLabel("SDK Color Scheme")
        colorizer_label.setObjectName("cameraParameterLabel")
        self.depth_colorizer_scheme_combo = QtWidgets.QComboBox()
        for label, value in (
            ("Jet（Viewer默认）", 0),
            ("Classic", 1),
            ("White → Black", 2),
            ("Black → White", 3),
            ("Bio", 4),
            ("Cold", 5),
            ("Warm", 6),
            ("Quantized", 7),
            ("Pattern", 8),
            ("Hue", 9),
        ):
            self.depth_colorizer_scheme_combo.addItem(label, value)
        scheme_index = self.depth_colorizer_scheme_combo.findData(
            int(self.config["camera"].get("depth_colorizer_scheme", 0))
        )
        self.depth_colorizer_scheme_combo.setCurrentIndex(max(0, scheme_index))
        self.depth_colorizer_scheme_combo.installEventFilter(
            self.camera_parameter_wheel_filter
        )
        self.depth_colorizer_scheme_combo.currentIndexChanged.connect(
            lambda _index: self._set_camera_parameter(
                "depth_colorizer_scheme",
                int(self.depth_colorizer_scheme_combo.currentData()),
            )
        )
        colorizer_row.addWidget(colorizer_label)
        colorizer_row.addWidget(self.depth_colorizer_scheme_combo, 1)
        tuning_layout.addLayout(colorizer_row)
        tuning_check("Spatial Edge-Preserving Filter", "depth_spatial_enabled")
        tuning_slider("Spatial Alpha", "depth_spatial_alpha", 0.25, 1.0, 0.05, decimals=2)
        tuning_slider("Spatial Delta", "depth_spatial_delta", 1, 50, 1)
        tuning_slider("Spatial Hole Radius", "depth_spatial_holes_fill", 0, 5, 1)
        tuning_check("Temporal Threshold Filter", "depth_temporal_enabled")
        tuning_slider("Temporal Alpha", "depth_temporal_alpha", 0.1, 1.0, 0.05, decimals=2)
        tuning_slider("Temporal Delta", "depth_temporal_delta", 1, 100, 1)
        tuning_slider("Temporal Persistency", "depth_temporal_persistency", 0, 8, 1)
        tuning_check("Hole Filling（可能生成估计深度）", "depth_hole_filling_enabled")

        tuning_section("Depth Range & Point Cloud")
        tuning_slider("Minimum Distance", "depth_min_m", 0.10, 1.00, 0.01, decimals=2, suffix=" m")
        tuning_slider("Maximum Distance", "depth_max_m", 0.20, 6.00, 0.05, decimals=2, suffix=" m")
        tuning_slider("Point Sampling", "point_cloud_stride_px", 1, 4, 1, suffix=" px")
        tuning_slider("Maximum Points", "point_cloud_max_points", 50000, 400000, 10000)
        tuning_check("预览稳定填洞（不修改原始数据）", "depth_display_hole_fill_enabled")
        tuning_slider("瞬时丢深度保持", "depth_display_persist_frames", 0, 12, 1, suffix=" 帧")
        tuning_check("3D点云稳定滤波", "point_cloud_filter_enabled")
        tuning_slider("3D表面铺点半径", "point_cloud_splat_radius_px", 0, 2, 1, suffix=" px")

        tuning_section("RGB Sensor")
        color_auto_check = tuning_check("RGB Auto Exposure", "color_auto_exposure")
        color_exposure_spin = tuning_slider(
            "RGB Exposure", "color_exposure_us", 1, 10000, 10, suffix=" μs"
        )
        color_gain_spin = tuning_slider("RGB Gain", "color_gain", 0, 128, 1)
        tuning_check("RGB Auto White Balance", "color_auto_white_balance")
        color_auto_check.toggled.connect(lambda checked: color_exposure_spin.setEnabled(not checked))
        color_auto_check.toggled.connect(lambda checked: color_gain_spin.setEnabled(not checked))
        color_exposure_spin.setEnabled(not color_auto_check.isChecked())
        color_gain_spin.setEnabled(not color_auto_check.isChecked())

        preset_buttons = QtWidgets.QHBoxLayout()
        high_accuracy_button = QtWidgets.QPushButton("近距离高精度")
        high_accuracy_button.setProperty("role", "soft")
        high_accuracy_button.clicked.connect(self._apply_high_accuracy_camera_preset)
        high_density_button = QtWidgets.QPushButton("Viewer稠密显示")
        high_density_button.setProperty("role", "soft")
        high_density_button.clicked.connect(self._apply_high_density_camera_preset)
        preset_buttons.addWidget(high_accuracy_button)
        preset_buttons.addWidget(high_density_button)
        tuning_layout.addLayout(preset_buttons)
        self.camera_parameter_status = QtWidgets.QLabel(
            "参数在下一帧生效；关闭孔洞填充可避免虚假表面。"
        )
        self.camera_parameter_status.setObjectName("cameraParameterStatus")
        self.camera_parameter_status.setWordWrap(True)
        tuning_layout.addWidget(self.camera_parameter_status)
        tuning_layout.addStretch(1)
        tuning_scroll.setWidget(tuning_content)
        self.camera_tuning_panel.body_layout.addWidget(tuning_scroll)
        shape_layout.addWidget(self.camera_tuning_panel)
        self.shape_view = ShapeView()
        self.shape_view.setMinimumSize(300, 150)
        shape_layout.addWidget(self.shape_view, 1)

        self.secondary_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.secondary_splitter.addWidget(self.endoscope_camera_card)
        self.secondary_splitter.addWidget(self.side_camera_card)
        self.secondary_splitter.addWidget(self.shape_card)
        self.secondary_splitter.setStretchFactor(0, 3)
        self.secondary_splitter.setStretchFactor(1, 3)
        self.secondary_splitter.setStretchFactor(2, 2)
        workspace_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        workspace_splitter.addWidget(self.d435_camera_card)
        workspace_splitter.addWidget(self.secondary_splitter)
        workspace_splitter.setStretchFactor(0, 3)
        workspace_splitter.setStretchFactor(1, 2)
        workspace_splitter.setSizes([720, 430])
        content.addWidget(workspace_splitter, 3)

        controls_widget = QtWidgets.QWidget()
        controls = QtWidgets.QVBoxLayout(controls_widget)
        controls.setContentsMargins(4, 4, 4, 4)
        mode_group = QtWidgets.QGroupBox("三维测量模式")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self.mode_combo = SegmentedModeControl()
        self.mode_combo.addItem("双视图融合", DUAL_VIEW_MODE)
        self.mode_combo.addItem("D435内部", D435_INTERNAL_MODE)
        current_index = self.mode_combo.findData(self.engine.fusion_mode)
        self.mode_combo.setCurrentIndex(max(current_index, 0))
        self.mode_combo.currentIndexChanged.connect(self._change_fusion_mode)
        mode_layout.addWidget(self.mode_combo)
        side_row = QtWidgets.QHBoxLayout()
        self.side_camera_combo = QtWidgets.QComboBox()
        self.side_camera_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.side_camera_combo.setMinimumContentsLength(24)
        self.side_camera_combo.setToolTip("自动检测到的普通RGB摄像头；RealSense D435已排除")
        self.side_camera_combo.activated.connect(self._on_side_camera_selected)
        side_row.addWidget(self.side_camera_combo, 1)
        refresh_side = QtWidgets.QPushButton("刷新设备")
        refresh_side.setProperty("role", "soft")
        refresh_side.setToolTip("立即重新扫描可用RGB摄像头")
        refresh_side.clicked.connect(lambda: self._refresh_side_camera_devices(force=True))
        side_row.addWidget(refresh_side)
        mode_layout.addLayout(side_row)
        self.side_status = QtWidgets.QLabel("正在检查可用侧相机……")
        self.side_status.setWordWrap(True)
        mode_layout.addWidget(self.side_status)
        controls.addWidget(mode_group)

        tracking_group = QtWidgets.QGroupBox("关键点感知")
        tracking_layout = QtWidgets.QVBoxLayout(tracking_group)
        self.keypoint_tracking_check = QtWidgets.QCheckBox("启用7点关键点捕捉")
        self.keypoint_tracking_check.setChecked(self.engine.keypoint_tracking_enabled)
        self.keypoint_tracking_check.setToolTip("关闭后仅预览和记录相机原始流，不执行颜色分割、IR匹配、三角化或卡尔曼更新")
        self.keypoint_tracking_check.toggled.connect(self._toggle_keypoint_tracking)
        tracking_layout.addWidget(self.keypoint_tracking_check)
        self.keypoint_tracking_status = QtWidgets.QLabel(
            "RGB定位 + 左右IR极线亚像素匹配 + Depth一致性校验"
        )
        self.keypoint_tracking_status.setWordWrap(True)
        tracking_layout.addWidget(self.keypoint_tracking_status)
        calibrate_colours = QtWidgets.QPushButton("用当前帧校准7条颜色带")
        calibrate_colours.setProperty("role", "soft")
        calibrate_colours.setToolTip("确保7条颜色带都清晰可见后点击；更新HSV＋Lab颜色原型，不改变K0–K6编号")
        calibrate_colours.clicked.connect(self._calibrate_marker_colours)
        tracking_layout.addWidget(calibrate_colours)
        self.synchronized_zoom_check = QtWidgets.QCheckBox("D435三视图联动缩放")
        self.synchronized_zoom_check.setChecked(
            bool(self.config.get("ui", {}).get("synchronized_zoom", True))
        )
        self.synchronized_zoom_check.setToolTip("联动D435 RGB、Depth和左右IR稠密点云；侧相机RGB始终独立控制")
        self.synchronized_zoom_check.toggled.connect(self._toggle_synchronized_zoom)
        tracking_layout.addWidget(self.synchronized_zoom_check)
        controls.addWidget(tracking_group)

        inspection_group = QtWidgets.QGroupBox("三维点详细信息")
        inspection_layout = QtWidgets.QVBoxLayout(inspection_group)
        inspection_hint = QtWidgets.QLabel("点击任意图像查询三维点；点击表格行查看K0–K6完整信息")
        inspection_hint.setWordWrap(True)
        inspection_layout.addWidget(inspection_hint)
        self.keypoint_table = QtWidgets.QTableWidget(7, 8)
        self.keypoint_table.setHorizontalHeaderLabels(
            ["点", "状态", "X/mm", "Y/mm", "Z/mm", "σZ/mm", "置信", "误差/mm"]
        )
        self.keypoint_table.verticalHeader().setVisible(False)
        self.keypoint_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.keypoint_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.keypoint_table.setAlternatingRowColors(True)
        self.keypoint_table.setMaximumHeight(205)
        table_header = self.keypoint_table.horizontalHeader()
        table_header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        for column, width in enumerate((38, 48, 62, 62, 62, 66, 54, 70)):
            self.keypoint_table.setColumnWidth(column, width)
        self.keypoint_table.cellClicked.connect(lambda row, _column: self._inspect_keypoint(row))
        inspection_layout.addWidget(self.keypoint_table)
        self.point_inspector = QtWidgets.QPlainTextEdit()
        self.point_inspector.setReadOnly(True)
        self.point_inspector.setMaximumHeight(185)
        self.point_inspector.setPlainText("等待有效图像或关键点……")
        inspection_layout.addWidget(self.point_inspector)
        export_cloud = QtWidgets.QPushButton("导出当前稠密点云（PLY＋NPZ）")
        export_cloud.setProperty("role", "soft")
        export_cloud.clicked.connect(lambda: self._export_current_point_cloud())
        inspection_layout.addWidget(export_cloud)
        controls.addWidget(inspection_group)

        source_group = QtWidgets.QGroupBox("D435数据源")
        source_layout = QtWidgets.QVBoxLayout(source_group)
        self.device_status = QtWidgets.QLabel("相机未启动")
        self.device_status.setWordWrap(True)
        source_layout.addWidget(self.device_status)
        self.start_button = QtWidgets.QPushButton("启动相机")
        self.start_button.setProperty("role", "primary")
        self.start_button.setCheckable(True)
        self.start_button.toggled.connect(self._toggle_camera)
        source_layout.addWidget(self.start_button)
        live_button = QtWidgets.QPushButton("切换到实时D435")
        live_button.clicked.connect(self._use_live_d435)
        source_layout.addWidget(live_button)
        replay_button = QtWidgets.QPushButton("打开 RealSense Bag 回放")
        replay_button.clicked.connect(self._open_bag)
        source_layout.addWidget(replay_button)
        synthetic_button = QtWidgets.QPushButton("切换到模拟数据源")
        synthetic_button.clicked.connect(self._use_synthetic)
        source_layout.addWidget(synthetic_button)
        controls.addWidget(source_group)

        endoscope_group = QtWidgets.QGroupBox("内窥镜摄像头")
        endoscope_layout = QtWidgets.QVBoxLayout(endoscope_group)
        self.endoscope_status = QtWidgets.QLabel("正在检查内窥镜摄像头……")
        self.endoscope_status.setWordWrap(True)
        endoscope_layout.addWidget(self.endoscope_status)
        endoscope_device_row = QtWidgets.QHBoxLayout()
        self.endoscope_camera_combo = QtWidgets.QComboBox()
        self.endoscope_camera_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.endoscope_camera_combo.setMinimumContentsLength(22)
        self.endoscope_camera_combo.setToolTip(
            "选择内窥镜UVC摄像头；双视图模式下不能与侧相机选择同一设备"
        )
        self.endoscope_camera_combo.installEventFilter(self.camera_parameter_wheel_filter)
        self.endoscope_camera_combo.activated.connect(self._on_endoscope_camera_selected)
        endoscope_device_row.addWidget(self.endoscope_camera_combo, 1)
        refresh_endoscope = QtWidgets.QPushButton("刷新设备")
        refresh_endoscope.setProperty("role", "soft")
        refresh_endoscope.clicked.connect(
            lambda: self._refresh_endoscope_camera_devices(force=True)
        )
        endoscope_device_row.addWidget(refresh_endoscope)
        endoscope_layout.addLayout(endoscope_device_row)
        endoscope_actions = QtWidgets.QHBoxLayout()
        self.endoscope_enabled_check = QtWidgets.QCheckBox("启用预览")
        self.endoscope_enabled_check.setChecked(
            bool(self.config.get("endoscope_camera", {}).get("enabled", True))
        )
        self.endoscope_enabled_check.toggled.connect(self._toggle_endoscope_enabled)
        endoscope_actions.addWidget(self.endoscope_enabled_check)
        self.endoscope_connect_button = QtWidgets.QPushButton("连接内窥镜")
        self.endoscope_connect_button.clicked.connect(self._connect_endoscope_from_ui)
        endoscope_actions.addWidget(self.endoscope_connect_button, 1)
        endoscope_layout.addLayout(endoscope_actions)
        self.endoscope_record_check = QtWidgets.QCheckBox("记录内窥镜视频和时间戳")
        self.endoscope_record_check.setChecked(
            bool(self.config.get("recording", {}).get("endoscope_enabled", True))
        )
        self.endoscope_record_check.toggled.connect(
            lambda checked: self.config.setdefault("recording", {}).__setitem__(
                "endoscope_enabled", bool(checked)
            )
        )
        endoscope_layout.addWidget(self.endoscope_record_check)
        controls.addWidget(endoscope_group)

        axis_group = QtWidgets.QGroupBox("七轴数据源")
        axis_layout = QtWidgets.QVBoxLayout(axis_group)
        self.axis_status = QtWidgets.QLabel(self._axis_source_description())
        self.axis_status.setWordWrap(True)
        axis_layout.addWidget(self.axis_status)
        axis_header = QtWidgets.QGridLayout()
        axis_header.setHorizontalSpacing(8)
        for column, text in enumerate(("Axis", "DPOS", "MPOS", "使用")):
            label = QtWidgets.QLabel(text)
            label.setObjectName("cameraSectionLabel")
            axis_header.addWidget(label, 0, column)
        self.axis_value_labels: list[tuple[QtWidgets.QLabel, QtWidgets.QLabel, QtWidgets.QLabel]] = []
        for axis in range(7):
            axis_label = QtWidgets.QLabel(str(axis))
            demand_label = QtWidgets.QLabel("--")
            measured_label = QtWidgets.QLabel("--")
            source_label = QtWidgets.QLabel("--")
            for label in (demand_label, measured_label, source_label):
                label.setStyleSheet('font-family:"Cascadia Mono","Consolas";font-size:11px;')
            axis_header.addWidget(axis_label, axis + 1, 0)
            axis_header.addWidget(demand_label, axis + 1, 1)
            axis_header.addWidget(measured_label, axis + 1, 2)
            axis_header.addWidget(source_label, axis + 1, 3)
            self.axis_value_labels.append((demand_label, measured_label, source_label))
        axis_layout.addLayout(axis_header)
        axis_row = QtWidgets.QHBoxLayout()
        self.trio_address_edit = QtWidgets.QLineEdit(str(self.config["axes"].get("controller", "192.168.0.250")))
        self.trio_address_edit.setPlaceholderText("控制器IP，例如 192.168.0.250")
        axis_row.addWidget(self.trio_address_edit)
        trio_connect_button = QtWidgets.QPushButton("连接Trio")
        trio_connect_button.clicked.connect(self._connect_trio_from_ui)
        axis_row.addWidget(trio_connect_button)
        axis_layout.addLayout(axis_row)
        no_axis_button = QtWidgets.QPushButton("使用零输入/断开Trio")
        no_axis_button.clicked.connect(self._use_null_axes)
        axis_layout.addWidget(no_axis_button)
        controls.addWidget(axis_group)

        em_group = QtWidgets.QGroupBox("NDI Aurora 电磁跟踪")
        em_layout = QtWidgets.QVBoxLayout(em_group)
        self.em_status = QtWidgets.QLabel("NDI未连接")
        self.em_status.setWordWrap(True)
        em_layout.addWidget(self.em_status)
        em_port_row = QtWidgets.QHBoxLayout()
        self.em_port_combo = QtWidgets.QComboBox()
        self.em_port_combo.setToolTip("实时检测串口，并优先显示NDI Aurora SCU")
        self.em_port_combo.installEventFilter(self.camera_parameter_wheel_filter)
        em_port_row.addWidget(self.em_port_combo, 1)
        em_refresh_button = QtWidgets.QPushButton("刷新串口")
        em_refresh_button.setProperty("role", "soft")
        em_refresh_button.clicked.connect(self._refresh_ndi_ports)
        em_port_row.addWidget(em_refresh_button)
        em_layout.addLayout(em_port_row)
        em_rom_row = QtWidgets.QHBoxLayout()
        configured_roms = self.config.get("em", {}).get("rom_files", ["window/8700339.rom"])
        configured_rom = str(configured_roms[0]) if configured_roms else "window/8700339.rom"
        configured_rom_path = Path(configured_rom).expanduser()
        if not configured_rom_path.is_absolute():
            configured_rom_path = (Path(__file__).resolve().parent.parent / configured_rom_path).resolve()
        self.em_rom_edit = QtWidgets.QLineEdit(str(configured_rom_path))
        self.em_rom_edit.setToolTip("当前NDI传感器对应的.rom工具定义文件")
        em_rom_row.addWidget(self.em_rom_edit, 1)
        em_rom_button = QtWidgets.QPushButton("选择ROM")
        em_rom_button.clicked.connect(self._browse_ndi_rom)
        em_rom_row.addWidget(em_rom_button)
        em_layout.addLayout(em_rom_row)
        em_options_row = QtWidgets.QHBoxLayout()
        em_options_row.addWidget(QtWidgets.QLabel("主Handle"))
        self.em_handle_spin = QtWidgets.QSpinBox()
        self.em_handle_spin.setRange(-1, 255)
        self.em_handle_spin.setSpecialValueText("自动")
        self.em_handle_spin.setValue(int(self.config.get("em", {}).get("primary_handle", 10)))
        self.em_handle_spin.installEventFilter(self.camera_parameter_wheel_filter)
        self.em_handle_spin.valueChanged.connect(
            lambda value: self.config.setdefault("em", {}).__setitem__("primary_handle", int(value))
        )
        em_options_row.addWidget(self.em_handle_spin)
        self.em_connect_button = QtWidgets.QPushButton("连接NDI")
        self.em_connect_button.setProperty("role", "primary")
        self.em_connect_button.clicked.connect(self._toggle_ndi_connection)
        em_options_row.addWidget(self.em_connect_button, 1)
        em_layout.addLayout(em_options_row)
        self.em_record_check = QtWidgets.QCheckBox("随电机采样同步保存 EM CSV")
        self.em_record_check.setChecked(
            bool(self.config.get("em", {}).get("record_with_motor", True))
        )
        self.em_record_check.setToolTip(
            "每个电机采样时刻保存最近一帧EM；new_em_frame与em_age_ms保留NDI真实时序"
        )
        self.em_record_check.toggled.connect(
            lambda checked: self.config.setdefault("em", {}).__setitem__(
                "record_with_motor", bool(checked)
            )
        )
        em_layout.addWidget(self.em_record_check)
        controls.addWidget(em_group)

        mujoco_group = QtWidgets.QGroupBox("MuJoCo实时对齐")
        mujoco_layout = QtWidgets.QVBoxLayout(mujoco_group)
        self.mujoco_enable_check = QtWidgets.QCheckBox("启用MuJoCo实时对齐")
        self.mujoco_enable_check.setChecked(self.engine.bridge is not None)
        self.mujoco_enable_check.toggled.connect(self._toggle_mujoco_sync)
        mujoco_layout.addWidget(self.mujoco_enable_check)
        self.mujoco_xml_edit = QtWidgets.QLineEdit(str(self.config["mujoco"]["xml"]))
        mujoco_layout.addWidget(self.mujoco_xml_edit)
        mujoco_buttons = QtWidgets.QHBoxLayout()
        browse_xml = QtWidgets.QPushButton("选择XML")
        browse_xml.clicked.connect(self._browse_mujoco_xml)
        mujoco_buttons.addWidget(browse_xml)
        apply_mujoco = QtWidgets.QPushButton("应用")
        apply_mujoco.clicked.connect(self._apply_mujoco_settings)
        mujoco_buttons.addWidget(apply_mujoco)
        mujoco_layout.addLayout(mujoco_buttons)
        self.mujoco_status = QtWidgets.QLabel("MuJoCo同步已关闭，不计算仿真误差")
        self.mujoco_status.setWordWrap(True)
        mujoco_layout.addWidget(self.mujoco_status)
        controls.addWidget(mujoco_group)

        calibration_group = QtWidgets.QGroupBox("ChArUco 相机—基座标定")
        calibration_layout = QtWidgets.QVBoxLayout(calibration_group)
        self.calibration_target = QtWidgets.QComboBox()
        self.calibration_target.addItem("D435—基座外参", "d435_extrinsic")
        self.calibration_target.addItem("侧相机内参（移动标定板）", "side_intrinsic")
        self.calibration_target.addItem("侧相机—基座外参（固定标定板）", "side_extrinsic")
        calibration_layout.addWidget(self.calibration_target)
        self.calibration_status = QtWidgets.QLabel("尚未采集标定帧")
        self.calibration_status.setWordWrap(True)
        calibration_layout.addWidget(self.calibration_status)
        capture_calibration = QtWidgets.QPushButton("采集当前标定帧")
        capture_calibration.clicked.connect(self._capture_calibration)
        calibration_layout.addWidget(capture_calibration)
        solve_calibration = QtWidgets.QPushButton("求解并保存标定")
        solve_calibration.setProperty("role", "soft")
        solve_calibration.clicked.connect(self._solve_calibration)
        calibration_layout.addWidget(solve_calibration)
        load_calibration_button = QtWidgets.QPushButton("加载标定文件")
        load_calibration_button.clicked.connect(self._load_calibration)
        calibration_layout.addWidget(load_calibration_button)
        controls.addWidget(calibration_group)

        record_group = QtWidgets.QGroupBox("同步数据记录")
        record_layout = QtWidgets.QVBoxLayout(record_group)
        output_row = QtWidgets.QHBoxLayout()
        self.output_root_edit = QtWidgets.QLineEdit(str(self.config["recording"]["root"]))
        output_row.addWidget(self.output_root_edit)
        output_browse = QtWidgets.QPushButton("输出目录")
        output_browse.clicked.connect(self._browse_output_root)
        output_row.addWidget(output_browse)
        record_layout.addLayout(output_row)
        self.synchronized_video_check = QtWidgets.QCheckBox("同步保存D435 RGB＋侧相机RGB视频")
        self.synchronized_video_check.setChecked(
            bool(self.config["recording"].get("synchronized_video_enabled", True))
        )
        self.synchronized_video_check.setToolTip("保存两路原始RGB、时间配对表和双画面同步预览；以D435帧时钟为主")
        self.synchronized_video_check.toggled.connect(
            lambda checked: self.config["recording"].__setitem__("synchronized_video_enabled", bool(checked))
        )
        record_layout.addWidget(self.synchronized_video_check)
        self.viewer_stream_label = QtWidgets.QLabel("选择需要分别保存的窗口")
        self.viewer_stream_label.setObjectName("cameraSectionLabel")
        record_layout.addWidget(self.viewer_stream_label)
        selected_streams = set(
            self.config["recording"].get("viewer_streams", ["rgb", "depth"])
        )
        viewer_stream_row = QtWidgets.QHBoxLayout()
        self.record_rgb_check = QtWidgets.QCheckBox("RGB")
        self.record_depth_check = QtWidgets.QCheckBox("Depth")
        self.record_pointcloud_check = QtWidgets.QCheckBox("3D点云")
        for name, control in (
            ("rgb", self.record_rgb_check),
            ("depth", self.record_depth_check),
            ("pointcloud", self.record_pointcloud_check),
        ):
            control.setChecked(name in selected_streams)
            control.toggled.connect(
                lambda _checked, stream=name: self._update_viewer_stream_selection()
            )
            viewer_stream_row.addWidget(control)
        record_layout.addLayout(viewer_stream_row)
        self.record_button = QtWidgets.QPushButton("开始记录")
        self.record_button.setProperty("role", "primary")
        self.record_button.setCheckable(True)
        self.record_button.toggled.connect(self._toggle_recording)
        record_layout.addWidget(self.record_button)
        self.record_status = QtWidgets.QLabel("未记录")
        self.record_status.setWordWrap(True)
        record_layout.addWidget(self.record_status)
        controls.addWidget(record_group)

        config_group = QtWidgets.QGroupBox("配置文件")
        config_layout = QtWidgets.QHBoxLayout(config_group)
        load_config_button = QtWidgets.QPushButton("加载配置")
        load_config_button.clicked.connect(self._load_config_from_ui)
        config_layout.addWidget(load_config_button)
        save_config_button = QtWidgets.QPushButton("保存当前配置")
        save_config_button.clicked.connect(self._save_config_from_ui)
        config_layout.addWidget(save_config_button)
        controls.addWidget(config_group)

        identify_group = QtWidgets.QGroupBox("离线参数辨识")
        identify_layout = QtWidgets.QVBoxLayout(identify_group)
        self.identify_session_edit = QtWidgets.QLineEdit()
        self.identify_session_edit.setPlaceholderText("选择已记录的session目录")
        identify_layout.addWidget(self.identify_session_edit)
        identify_options = QtWidgets.QHBoxLayout()
        identify_options.addWidget(QtWidgets.QLabel("最大帧数"))
        self.identify_frames_spin = QtWidgets.QSpinBox()
        self.identify_frames_spin.setRange(20, 100000)
        self.identify_frames_spin.setValue(180)
        identify_options.addWidget(self.identify_frames_spin)
        identify_options.addWidget(QtWidgets.QLabel("迭代"))
        self.identify_maxiter_spin = QtWidgets.QSpinBox()
        self.identify_maxiter_spin.setRange(1, 500)
        self.identify_maxiter_spin.setValue(10)
        identify_options.addWidget(self.identify_maxiter_spin)
        identify_layout.addLayout(identify_options)
        identify_row = QtWidgets.QHBoxLayout()
        choose_session = QtWidgets.QPushButton("选择Session")
        choose_session.clicked.connect(self._choose_identification_session)
        identify_row.addWidget(choose_session)
        self.identify_button = QtWidgets.QPushButton("开始辨识")
        self.identify_button.setProperty("role", "primary")
        self.identify_button.clicked.connect(self._start_identification)
        identify_row.addWidget(self.identify_button)
        identify_layout.addLayout(identify_row)
        self.identify_status = QtWidgets.QLabel("未运行")
        self.identify_status.setWordWrap(True)
        identify_layout.addWidget(self.identify_status)
        controls.addWidget(identify_group)

        self.metrics = QtWidgets.QPlainTextEdit()
        self.metrics.setReadOnly(True)
        self.metrics.setMaximumBlockCount(200)
        controls.addWidget(self.metrics, 1)
        self._experiment_only_widgets = [
            mode_group,
            tracking_group,
            inspection_group,
            mujoco_group,
            calibration_group,
            identify_group,
            self.metrics,
        ]
        self._capture_stream_widgets = [
            self.viewer_stream_label,
            self.record_rgb_check,
            self.record_depth_check,
            self.record_pointcloud_check,
        ]
        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setMinimumWidth(390)
        controls_scroll.setMaximumWidth(470)
        content.addWidget(controls_scroll, 1)
        root.addLayout(content, 1)
        self.side_camera_card.setVisible(self.engine.fusion_mode == DUAL_VIEW_MODE)
        self._add_card_shadow(top_bar, blur=24, y_offset=4, alpha=18)
        self._add_card_shadow(self.d435_camera_card, blur=28, y_offset=6, alpha=20)
        self._add_card_shadow(self.endoscope_camera_card, blur=28, y_offset=6, alpha=20)
        self._add_card_shadow(self.side_camera_card, blur=28, y_offset=6, alpha=20)
        self._add_card_shadow(self.shape_card, blur=24, y_offset=5, alpha=16)

    @staticmethod
    def _add_card_shadow(widget: QtWidgets.QWidget, blur: int, y_offset: int, alpha: int) -> None:
        shadow = QtWidgets.QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(float(blur))
        shadow.setOffset(0.0, float(y_offset))
        shadow.setColor(QtGui.QColor(0, 0, 0, int(alpha)))
        widget.setGraphicsEffect(shadow)

    @staticmethod
    def _refresh_style(widget: QtWidgets.QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _processing_required(self) -> bool:
        """The pure acquisition path must never start the vision worker."""
        return self.workflow_mode == "experiment"

    def _update_camera_processing_demand(self, *, force: bool = False) -> None:
        """Enable expensive depth products only while a visible feature needs them."""
        if not hasattr(self, "d435_display_mode"):
            return
        view_3d = str(self.d435_display_mode.currentData()) == "3d"
        native_2d = (
            not view_3d
            and str(self.config.get("ui", {}).get("depth_view_space", "native"))
            == "native"
        )
        record_3d = (
            hasattr(self, "viewer_recorder")
            and self.viewer_recorder.active
            and hasattr(self, "record_pointcloud_check")
            and self.record_pointcloud_check.isChecked()
        )
        aligned_required = (
            self.workflow_mode == "experiment"
            or view_3d
            or record_3d
            or not native_2d
        )
        signature = (bool(aligned_required), bool(native_2d))
        if not force and signature == self._camera_processing_demand_signature:
            return
        self._camera_processing_demand_signature = signature
        updates = {
            "runtime_align_depth_to_color": bool(aligned_required),
            "runtime_native_depth_colorizer_enabled": bool(native_2d),
        }
        self.config.setdefault("camera", {}).update(updates)
        if hasattr(self.camera_source, "update_runtime_settings"):
            self.camera_source.update_runtime_settings(updates)

    def _selected_viewer_streams(self) -> set[str]:
        selected = set()
        if self.record_rgb_check.isChecked():
            selected.add("rgb")
        if self.record_depth_check.isChecked():
            selected.add("depth")
        if self.record_pointcloud_check.isChecked():
            selected.add("pointcloud")
        return selected

    def _update_viewer_stream_selection(self) -> None:
        self.config["recording"]["viewer_streams"] = sorted(
            self._selected_viewer_streams()
        )

    def _change_d435_view_mode(self, _row: int) -> None:
        mode = str(self.d435_display_mode.currentData())
        self.config.setdefault("ui", {})["view_mode"] = mode
        self.d435_view_stack.setCurrentIndex(1 if mode == "3d" else 0)
        self._update_camera_processing_demand()

    def _change_depth_view_space(self, _row: int) -> None:
        space = str(self.depth_space_mode.currentData())
        self.config.setdefault("ui", {})["depth_view_space"] = space
        self._reset_depth_display_filters()
        self._last_depth_preview_ns = 0
        self._update_camera_processing_demand()
        if hasattr(self, "camera_parameter_status"):
            description = (
                "原生Stereo Module深度（不经过RGB重投影）"
                if space == "native"
                else "Depth→RGB对齐深度（用于关键点融合）"
            )
            self.camera_parameter_status.setText(description)

    def _apply_workflow_mode_ui(self) -> None:
        capture = self.workflow_mode == "capture"
        for widget in self._experiment_only_widgets:
            widget.setVisible(not capture)
        for widget in self._capture_stream_widgets:
            widget.setVisible(capture)
        self.synchronized_video_check.setVisible(not capture)
        self.side_camera_card.setVisible(
            not capture and self.engine.fusion_mode == DUAL_VIEW_MODE
        )
        self.shape_card.setVisible(True)
        self.shape_view.setVisible(not capture)
        self.camera_tuning_panel.setVisible(True)
        self.header_tracking_pill.setVisible(not capture)
        if capture:
            self.header_mode_pill.set_status(
                f"纯采集 {int(self.config['camera']['color'][2])} FPS", "blue"
            )
            self.record_status.setText("纯采集：相机帧与七轴数据按曝光时刻同步")
        else:
            self.header_mode_pill.set_status("实验模式", "blue")

    def _change_workflow_mode(self, _row: int) -> None:
        requested = str(self.workflow_mode_control.currentData())
        if requested == self.workflow_mode:
            return
        if (
            self.viewer_recorder.active
            or self.engine.recorder.active
            or self.endoscope_recorder.active
            or self.em_recorder.active
        ):
            current_index = self.workflow_mode_control.findData(self.workflow_mode)
            self.workflow_mode_control.blockSignals(True)
            self.workflow_mode_control.setCurrentIndex(current_index)
            self.workflow_mode_control.blockSignals(False)
            QtWidgets.QMessageBox.information(
                self, "无法切换模式", "请先停止当前记录，再切换工作模式。"
            )
            return
        self._stop_capture_processing()
        self.workflow_mode = requested
        self.config.setdefault("ui", {})["workflow_mode"] = requested
        self._update_camera_processing_demand(force=True)
        if requested == "capture":
            if self.engine.fusion_mode != D435_INTERNAL_MODE:
                self.engine.set_fusion_mode(D435_INTERNAL_MODE)
            self.engine.set_keypoint_tracking_enabled(False)
            self.engine.set_mujoco_bridge(None)
            self.keypoint_tracking_check.blockSignals(True)
            self.keypoint_tracking_check.setChecked(False)
            self.keypoint_tracking_check.blockSignals(False)
            self.mujoco_enable_check.blockSignals(True)
            self.mujoco_enable_check.setChecked(False)
            self.mujoco_enable_check.blockSignals(False)
        self.camera_discovery_timer.start()
        self._apply_workflow_mode_ui()
        self._update_header_status()
        self._start_capture_processing()

    def _update_header_status(self) -> None:
        if not hasattr(self, "header_mode_pill"):
            return
        dual = self.engine.fusion_mode == DUAL_VIEW_MODE
        if self.workflow_mode == "capture":
            self.header_mode_pill.set_status(
                f"纯采集 {int(self.config['camera']['color'][2])} FPS", "blue"
            )
        else:
            self.header_mode_pill.set_status("双视图融合" if dual else "实验模式", "blue")
        tracking_on = self.engine.keypoint_tracking_enabled
        self.header_tracking_pill.set_status("关键点开启" if tracking_on else "关键点关闭", "green" if tracking_on else "neutral")
        self.header_device_pill.set_status("采集中" if self.engine.running else "相机未启动", "green" if self.engine.running else "neutral")
        recording = (
            self.engine.recorder.active
            or self.viewer_recorder.active
            or self.endoscope_recorder.active
            or self.em_recorder.active
        )
        self.header_record_pill.set_status("正在记录" if recording else "未记录", "red" if recording else "neutral")
        self.d435_view_pill.set_status("在线" if self.engine.running else "待机", "green" if self.engine.running else "neutral")
        side_ready = dual and self.engine.latest_side_frame is not None
        self.side_view_pill.set_status("在线" if side_ready else ("待连接" if dual else "未启用"), "green" if side_ready else "neutral")
        self.start_button.setProperty("role", "danger" if self.engine.running else "primary")
        self.record_button.setProperty("role", "danger" if recording else "primary")
        self._refresh_style(self.start_button)
        self._refresh_style(self.record_button)

    def _axis_source_description(self) -> str:
        source = self.engine.axis_source
        if isinstance(source, TrioAxisSource):
            return f"Trio: {source.endpoint}"
        if isinstance(source, SyntheticAxisSource):
            return "模拟七轴"
        if isinstance(source, NullAxisSource):
            return "零输入（未连接控制器）"
        return f"复用外部七轴源: {type(source).__name__}"

    def _show_axis_sample(self, sample) -> None:
        self._latest_axis_ui_sample = sample
        for axis, (demand_label, measured_label, source_label) in enumerate(
            self.axis_value_labels
        ):
            demand_valid = bool(sample.demand_valid[axis])
            measured_valid = bool(sample.measured_valid[axis])
            demand_label.setText(
                f"{sample.demand_native[axis]:.5f}" if demand_valid else "--"
            )
            measured_label.setText(
                f"{sample.measured_native[axis]:.5f}" if measured_valid else "--"
            )
            if measured_valid:
                source_label.setText("MPOS")
                source_label.setStyleSheet(
                    'font-family:"Cascadia Mono","Consolas";font-size:11px;color:#248A3D;'
                )
            elif demand_valid:
                source_label.setText("DPOS")
                source_label.setStyleSheet(
                    'font-family:"Cascadia Mono","Consolas";font-size:11px;color:#007AFF;'
                )
            else:
                source_label.setText("--")
                source_label.setStyleSheet(
                    'font-family:"Cascadia Mono","Consolas";font-size:11px;color:#8E8E93;'
                )
        demand_count = int(np.count_nonzero(sample.demand_valid))
        measured_count = int(np.count_nonzero(sample.measured_valid))
        source = self.engine.axis_source
        if isinstance(source, TrioAxisSource):
            self.axis_status.setText(
                f"Trio {source.endpoint} · DPOS {demand_count}/7 · "
                f"MPOS {measured_count}/7 · {sample.status}"
            )
        else:
            self.axis_status.setText(
                f"{self._axis_source_description()} · DPOS {demand_count}/7 · "
                f"MPOS {measured_count}/7 · {sample.status}"
            )

    def _update_axis_live_values(self, now_ns: int) -> None:
        if now_ns - self._last_axis_ui_ns < 100_000_000:
            return
        self._last_axis_ui_ns = now_ns
        sampler = self.engine.axis_sampler
        sampler_running = sampler.thread is not None and sampler.thread.is_alive()
        try:
            if sampler_running:
                sample = sampler.interpolate(now_ns)
            else:
                sample = self.engine.axis_source.sample()
            self._show_axis_sample(sample)
        except Exception as exc:
            self.axis_status.setText(f"七轴读取失败: {exc}")

    def _connect_trio_from_ui(self) -> None:
        endpoint = (
            self.trio_address_edit.text().strip()
            or str(self.config["axes"].get("controller", "192.168.0.250"))
        )
        try:
            source = TrioAxisSource(self.config["axes"], endpoint)
            source.connect()
            first_sample = source.sample()
            if not np.any(first_sample.demand_valid):
                source.close()
                raise RuntimeError(f"连接已建立，但Axis 0–6均无法读取：{first_sample.status}")
            previous = self.engine.axis_source
            self.engine.set_axis_source(source)
            if self._owns_axis_source and hasattr(previous, "close"):
                previous.close()
            self._owns_axis_source = True
            self.config["axes"]["controller"] = endpoint
            self._show_axis_sample(first_sample)
            self.statusBar().showMessage(
                f"Trio已连接：{endpoint}，Axis 0–6读取正常", 5000
            )
        except Exception as exc:
            self.axis_status.setText(f"Trio连接失败: {exc}")
            QtWidgets.QMessageBox.warning(self, "Trio连接失败", str(exc))

    def _use_null_axes(self) -> None:
        previous = self.engine.axis_source
        null_source = NullAxisSource(self.config["axes"])
        self.engine.set_axis_source(null_source)
        if self._owns_axis_source and hasattr(previous, "close"):
            try:
                previous.close()
            except Exception:
                pass
        self._owns_axis_source = False
        self._show_axis_sample(null_source.sample())

    def _refresh_ndi_ports(self) -> None:
        current = ""
        if hasattr(self, "em_port_combo"):
            current = str(self.em_port_combo.currentData() or "")
        configured = str(self.config.get("em", {}).get("serial_port", ""))
        preferred = current or configured
        ports = list_ndi_serial_ports()
        self.em_port_combo.blockSignals(True)
        self.em_port_combo.clear()
        if not ports:
            self.em_port_combo.addItem("未检测到串口设备", "")
        else:
            for port in ports:
                label = f"{port['device']}  ·  {port['description']}"
                self.em_port_combo.addItem(label, port["device"])
            index = self.em_port_combo.findData(preferred)
            self.em_port_combo.setCurrentIndex(index if index >= 0 else 0)
        self.em_port_combo.blockSignals(False)

    def _browse_ndi_rom(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择NDI工具ROM", self.em_rom_edit.text(), "NDI ROM (*.rom);;所有文件 (*)"
        )
        if path:
            self.em_rom_edit.setText(path)

    def _toggle_ndi_connection(self) -> None:
        if self.em_recorder.active:
            QtWidgets.QMessageBox.information(self, "NDI", "请先停止数据记录，再断开或更换NDI设备。")
            return
        if self.em_source.active:
            self.em_source.close()
            self.em_connect_button.setText("连接NDI")
            self.em_port_combo.setEnabled(True)
            self.em_rom_edit.setEnabled(True)
            self.em_status.setText("NDI已断开")
            return
        port = str(self.em_port_combo.currentData() or "").strip()
        rom = self.em_rom_edit.text().strip()
        try:
            self.em_source.connect(port, [rom], int(self.config.get("em", {}).get("baud_rate", 9600)))
            self.config.setdefault("em", {}).update(
                serial_port=port,
                rom_files=[str(Path(rom).expanduser().resolve())],
                primary_handle=int(self.em_handle_spin.value()),
            )
            self.em_connect_button.setText("断开NDI")
            self.em_port_combo.setEnabled(False)
            self.em_rom_edit.setEnabled(False)
            self.em_status.setText(f"正在连接 {port}，初始化Aurora和工具ROM……")
        except Exception as exc:
            self.em_status.setText(f"NDI连接失败：{exc}")
            QtWidgets.QMessageBox.warning(self, "NDI连接失败", str(exc))

    def _update_em_status(self, now_ns: int) -> None:
        if now_ns - self._last_em_ui_ns < 250_000_000:
            return
        self._last_em_ui_ns = now_ns
        state = self.em_source.state
        frame = self.em_source.latest()
        if state == "tracking":
            valid = sum(int(tool.valid) for tool in frame.tools) if frame is not None else 0
            total = len(frame.tools) if frame is not None else 0
            handles = ", ".join(str(tool.port_handle) for tool in frame.tools) if frame is not None else "--"
            self.em_status.setText(
                f"NDI在线 · {self.em_source.device_rate_hz():.1f} Hz · "
                f"有效工具 {valid}/{total} · Handle {handles}"
            )
            self.em_connect_button.setText("断开NDI")
        elif state == "connecting":
            self.em_status.setText(f"正在连接 {self.em_source.serial_port}，请稍候……")
            self.em_connect_button.setText("断开NDI")
        elif state == "error":
            self.em_status.setText(f"NDI异常：{self.em_source.error}")
            self.em_connect_button.setText("重新连接NDI")
            self.em_port_combo.setEnabled(True)
            self.em_rom_edit.setEnabled(True)
        else:
            self.em_connect_button.setText("连接NDI")
            self.em_port_combo.setEnabled(True)
            self.em_rom_edit.setEnabled(True)

    def _browse_mujoco_xml(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择MuJoCo XML", self.mujoco_xml_edit.text(), "MuJoCo XML (*.xml)")
        if path:
            self.mujoco_xml_edit.setText(path)

    def _apply_mujoco_settings(self) -> None:
        if self.engine.recorder.active:
            QtWidgets.QMessageBox.information(self, "MuJoCo", "请先停止记录，再更改MuJoCo设置。")
            self.mujoco_enable_check.blockSignals(True)
            self.mujoco_enable_check.setChecked(self.engine.bridge is not None)
            self.mujoco_enable_check.blockSignals(False)
            return
        try:
            if not self.mujoco_enable_check.isChecked():
                self.engine.set_mujoco_bridge(None)
                self.config["mujoco"]["enabled"] = False
                self.mujoco_status.setText("MuJoCo同步已关闭，不计算仿真误差")
                return
            selected = self.mujoco_xml_edit.text().strip()
            self.config["mujoco"]["xml"] = selected
            if self._embedded_mujoco_bridge is not None and selected == self._embedded_mujoco_xml:
                bridge = self._embedded_mujoco_bridge
                status = "已复用主界面MuJoCo实例"
            else:
                bridge = MujocoAlignmentBridge(self.config["mujoco"])
                status = f"已加载MuJoCo: {selected}"
            self.engine.set_mujoco_bridge(bridge)
            self.config["mujoco"]["enabled"] = True
            self.mujoco_status.setText(status)
        except Exception as exc:
            self.mujoco_enable_check.blockSignals(True)
            self.mujoco_enable_check.setChecked(False)
            self.mujoco_enable_check.blockSignals(False)
            self.engine.set_mujoco_bridge(None)
            self.config["mujoco"]["enabled"] = False
            self.mujoco_status.setText(f"MuJoCo加载失败: {exc}")
            QtWidgets.QMessageBox.warning(self, "MuJoCo加载失败", str(exc))

    def _toggle_mujoco_sync(self, _checked: bool) -> None:
        self._apply_mujoco_settings()

    def _toggle_keypoint_tracking(self, checked: bool) -> None:
        resume_processing = self.capture_worker.running
        self._stop_capture_processing()
        self.engine.set_keypoint_tracking_enabled(checked)
        if resume_processing:
            self._start_capture_processing()
        if checked:
            self.keypoint_tracking_status.setText("已开启：RGB定位 + 左右IR极线亚像素匹配 + Depth一致性校验")
            self.d435_view_pill.set_status("正在捕捉", "green" if self.engine.running else "blue")
        else:
            self.keypoint_tracking_status.setText("已关闭：仅显示/记录原始相机流，不计算关键点与三维坐标")
            self.shape_view.update_shapes(
                np.full((7, 3), np.nan), np.full((7, 3), np.nan)
            )
        self._update_tracking_view_rois()
        self._update_header_status()

    def _calibrate_marker_colours(self) -> None:
        sample = self._latest_sample_for_inspection
        if sample is None:
            QtWidgets.QMessageBox.information(self, "颜色带校准", "请先启动相机并等待关键点画面。")
            return
        resume_processing = self.capture_worker.running
        self._stop_capture_processing()
        try:
            calibrated = self.engine.tracker.calibrate_color_prototypes(sample.frame, sample.keypoints)
            side_calibrated = None
            if sample.side_frame is not None:
                side_calibrated = self.engine.side_fusion.calibrate_color_prototypes(
                    sample.side_frame, sample.keypoints.side_pixels
                )
        finally:
            if resume_processing:
                self._start_capture_processing()
        complete = calibrated == 7 and (side_calibrated is None or side_calibrated == 7)
        side_text = "" if side_calibrated is None else f"，侧相机 {side_calibrated}/7"
        if complete:
            self.keypoint_tracking_status.setText(
                f"颜色带校准完成：D435 {calibrated}/7{side_text}。HSV＋Lab原型已更新，可保存配置。"
            )
            self.statusBar().showMessage(f"颜色带校准成功：D435 7/7{side_text}", 6000)
        else:
            self.keypoint_tracking_status.setText(
                f"颜色带校准：D435 {calibrated}/7{side_text}。请调整曝光、ROI或姿态后重试。"
            )
            QtWidgets.QMessageBox.information(
                self,
                "颜色带校准不完整",
                f"D435校准 {calibrated}/7{side_text}。未识别点没有被错误更新。",
            )

    def _synchronize_image_views(
        self,
        source: ZoomableImageLabel,
        zoom: float,
        center_x: float,
        center_y: float,
    ) -> None:
        if source in self._d435_zoom_labels:
            if hasattr(self, "synchronized_zoom_check") and self.synchronized_zoom_check.isChecked():
                for label in self._d435_zoom_labels:
                    if label is not source:
                        label.set_view(zoom, center_x, center_y, emit=False)
        self._update_tracking_view_rois()

    def _update_tracking_view_rois(self) -> None:
        if not hasattr(self, "engine") or not hasattr(self, "image_label"):
            return
        self.engine.set_tracking_rois(
            self.image_label.normalized_view_rect,
            self.side_image_label.normalized_view_rect,
        )
        if hasattr(self, "keypoint_tracking_status") and self.engine.keypoint_tracking_enabled:
            roi = self.image_label.normalized_view_rect
            self.keypoint_tracking_status.setText(
                "已开启：RGB＋IR亚像素匹配＋Depth校验\n"
                f"当前D435处理ROI: [{roi[0]:.3f}, {roi[1]:.3f}]–[{roi[2]:.3f}, {roi[3]:.3f}]"
            )

    def _toggle_synchronized_zoom(self, checked: bool) -> None:
        self.config.setdefault("ui", {})["synchronized_zoom"] = bool(checked)
        if checked:
            zoom, center_x, center_y = self.image_label.view_state
            self._synchronize_image_views(self.image_label, zoom, center_x, center_y)
        else:
            self._update_tracking_view_rois()

    @staticmethod
    def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
        transform = np.asarray(transform, dtype=float)
        return transform[:3, :3] @ np.asarray(point, dtype=float) + transform[:3, 3]

    @staticmethod
    def _deproject_pixel(pixel: tuple[float, float], depth_m: float, intrinsics) -> np.ndarray:
        u, v = pixel
        return np.asarray([
            (u - float(intrinsics["ppx"])) * depth_m / float(intrinsics["fx"]),
            (v - float(intrinsics["ppy"])) * depth_m / float(intrinsics["fy"]),
            depth_m,
        ])

    def _depth_uncertainty_mm(self, frame, point_left: np.ndarray) -> tuple[float, float, float]:
        baseline = float(np.linalg.norm(frame.transform_right_from_left[:3, 3]))
        fx = float(frame.left_intrinsics["fx"])
        z = max(float(point_left[2]), 1e-6)
        disparity_std_px = 0.35
        sigma_z = z * z * disparity_std_px / max(fx * baseline, 1e-9)
        sigma_xy = max(0.00015, z * 0.35 / max(fx, 1e-9))
        return sigma_xy * 1000.0, sigma_xy * 1000.0, sigma_z * 1000.0

    def _color_depth_point(self, frame, u: int, v: int) -> tuple[np.ndarray, np.ndarray, float] | None:
        if not bool((frame.metadata or {}).get("depth_aligned_to_color", True)):
            return None
        depth = np.asarray(frame.depth_m)
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            return None
        radius = 2
        values = depth[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1]
        minimum = float(self.config["camera"].get("depth_min_m", 0.18))
        maximum = float(self.config["camera"].get("depth_max_m", 0.60))
        values = values[np.isfinite(values) & (values >= minimum) & (values <= maximum)]
        if not len(values):
            return None
        depth_m = float(np.median(values))
        point_color = self._deproject_pixel((u, v), depth_m, frame.color_intrinsics)
        point_left = self._transform_point(frame.transform_left_from_color, point_color)
        point_base = self._transform_point(self.engine.tracker.transform_base_from_camera, point_left)
        return point_left, point_base, depth_m

    def _native_depth_point(self, frame, u: int, v: int) -> tuple[np.ndarray, np.ndarray, float] | None:
        native = getattr(frame, "native_depth_m", None)
        depth = np.asarray(native if native is not None else frame.depth_m)
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            return None
        radius = 2
        values = depth[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1]
        minimum = float(self.config["camera"].get("depth_min_m", 0.18))
        maximum = float(self.config["camera"].get("depth_max_m", 0.60))
        values = values[np.isfinite(values) & (values >= minimum) & (values <= maximum)]
        intrinsics = getattr(frame, "depth_intrinsics", {}) or frame.left_intrinsics
        if not len(values) or not intrinsics:
            return None
        depth_m = float(np.median(values))
        point_left = self._deproject_pixel((u, v), depth_m, intrinsics)
        point_base = self._transform_point(
            self.engine.tracker.transform_base_from_camera, point_left
        )
        return point_left, point_base, depth_m

    def _ir_xyz_map(self, frame, stream: str) -> np.ndarray:
        key = (int(frame.sequence), stream)
        cached = self._ir_xyz_cache.get(key)
        if cached is not None:
            return cached
        depth = np.asarray(frame.depth_m, dtype=np.float32)
        minimum = float(self.config["camera"].get("depth_min_m", 0.18))
        maximum = float(self.config["camera"].get("depth_max_m", 0.60))
        if self.workflow_mode == "capture":
            return (
                ViewerStreamRecorder._depth_visual(depth, minimum, maximum),
                (depth_width, depth_height),
            )
        valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
        vv, uu = np.nonzero(valid)
        z = depth[vv, uu].astype(np.float64)
        color_intr = frame.color_intrinsics
        points_color = np.column_stack((
            (uu - float(color_intr["ppx"])) * z / float(color_intr["fx"]),
            (vv - float(color_intr["ppy"])) * z / float(color_intr["fy"]),
            z,
        ))
        transform_left = np.asarray(frame.transform_left_from_color, dtype=float)
        points_left = points_color @ transform_left[:3, :3].T + transform_left[:3, 3]
        if stream == "ir_right":
            transform_right = np.asarray(frame.transform_right_from_left, dtype=float)
            projected_points = points_left @ transform_right[:3, :3].T + transform_right[:3, 3]
            intrinsics = frame.right_intrinsics
            image_shape = frame.infrared_right.shape
        else:
            projected_points = points_left
            intrinsics = frame.left_intrinsics
            image_shape = frame.infrared_left.shape
        positive = projected_points[:, 2] > 1e-6
        projected_points = projected_points[positive]
        points_left = points_left[positive]
        pu = np.rint(float(intrinsics["fx"]) * projected_points[:, 0] / projected_points[:, 2] + float(intrinsics["ppx"])).astype(int)
        pv = np.rint(float(intrinsics["fy"]) * projected_points[:, 1] / projected_points[:, 2] + float(intrinsics["ppy"])).astype(int)
        inside = (pu >= 0) & (pu < image_shape[1]) & (pv >= 0) & (pv < image_shape[0])
        pu, pv = pu[inside], pv[inside]
        projected_z = projected_points[inside, 2]
        points_left = points_left[inside]
        linear = pv * image_shape[1] + pu
        order = np.lexsort((projected_z, linear))
        sorted_linear = linear[order]
        first = np.unique(sorted_linear, return_index=True)[1]
        selected = order[first]
        xyz = np.full((image_shape[0], image_shape[1], 3), np.nan, dtype=np.float32)
        xyz[pv[selected], pu[selected]] = points_left[selected].astype(np.float32)
        self._ir_xyz_cache = {key: xyz}
        return xyz

    @staticmethod
    def _nearest_xyz(xyz_map: np.ndarray, u: int, v: int, radius: int = 5) -> np.ndarray | None:
        y0, y1 = max(0, v - radius), min(xyz_map.shape[0], v + radius + 1)
        x0, x1 = max(0, u - radius), min(xyz_map.shape[1], u + radius + 1)
        patch = xyz_map[y0:y1, x0:x1]
        valid = np.isfinite(patch).all(axis=2)
        if not np.any(valid):
            return None
        yy, xx = np.nonzero(valid)
        index = int(np.argmin((xx + x0 - u) ** 2 + (yy + y0 - v) ** 2))
        return patch[yy[index], xx[index]].astype(float)

    def _inspect_image_pixel(self, stream: str, u: int, v: int) -> None:
        sample = self._latest_sample_for_inspection
        if sample is None:
            self.point_inspector.setPlainText("尚无可检查的相机帧。")
            return
        frame = sample.frame
        if stream == "side":
            pixels = sample.keypoints.side_pixels
            distances = np.linalg.norm(pixels - np.asarray([u, v]), axis=1)
            distances[~np.isfinite(distances)] = np.inf
            index = int(np.argmin(distances))
            if distances[index] <= 15.0 and np.isfinite(sample.keypoints.base_m[index]).all():
                self._inspect_keypoint(index, prefix=f"侧RGB点击 ({u}, {v}) 匹配到K{index}\n")
            else:
                self.point_inspector.setPlainText(
                    f"侧RGB像素: ({u}, {v})\n无可用三维对应。\n"
                    "普通单目RGB不能单独恢复深度；需点击已匹配关键点，或增加稠密跨相机特征匹配。"
                )
            return
        if stream == "point_cloud":
            point_left = self._nearest_xyz(self._point_cloud_pick_left_m, u, v) if self._point_cloud_pick_left_m is not None else None
            point_base = self._nearest_xyz(self._point_cloud_pick_base_m, u, v) if self._point_cloud_pick_base_m is not None else None
            source_label = "左右IR稠密点云拾取"
        elif stream in ("color", "depth"):
            native_depth_view = (
                stream == "depth"
                and str(self.config.get("ui", {}).get("depth_view_space", "native"))
                == "native"
            )
            result = (
                self._native_depth_point(frame, u, v)
                if native_depth_view
                else self._color_depth_point(frame, u, v)
            )
            if result is None:
                self.point_inspector.setPlainText(f"{stream}像素: ({u}, {v})\n该位置没有有效Depth。")
                return
            point_left, point_base, _depth_m = result
            source_label = (
                "原生Stereo Depth"
                if native_depth_view
                else "RGB对齐Depth"
            )
        else:
            point_left = self._nearest_xyz(self._ir_xyz_map(frame, stream), u, v)
            point_base = None if point_left is None else self._transform_point(
                self.engine.tracker.transform_base_from_camera, point_left
            )
            source_label = "左IR反投影" if stream == "ir_left" else "右IR反投影"
        if point_left is None or point_base is None or not np.isfinite(point_left).all():
            self.point_inspector.setPlainText(f"{source_label}: ({u}, {v})\n附近没有可靠三维点。")
            return
        sigma = self._depth_uncertainty_mm(frame, point_left)
        self.point_inspector.setPlainText(
            f"{source_label}  pixel=({u}, {v})\n"
            f"左IR相机系 XYZ = {point_left[0]*1000:.3f}, {point_left[1]*1000:.3f}, {point_left[2]*1000:.3f} mm\n"
            f"机器人基座系 XYZ = {point_base[0]*1000:.3f}, {point_base[1]*1000:.3f}, {point_base[2]*1000:.3f} mm\n"
            f"估计标准差 σXYZ = {sigma[0]:.3f}, {sigma[1]:.3f}, {sigma[2]:.3f} mm\n"
            f"Frame={frame.sequence}  capture_host_ns={frame.capture_host_ns}"
        )

    def _inspect_keypoint(self, index: int, prefix: str = "") -> None:
        sample = self._latest_sample_for_inspection
        if sample is None or not 0 <= index < 7:
            return
        kp = sample.keypoints
        covariance = kp.covariance_m2[index]
        sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0)) * 1000.0 if np.isfinite(covariance).all() else np.full(3, np.nan)
        disparity = kp.left_pixels[index, 0] - kp.right_pixels[index, 0]
        self.point_inspector.setPlainText(
            prefix
            + f"K{index}  valid={bool(kp.valid[index])} predicted={bool(kp.predicted[index])}\n"
            f"source={kp.source[index]}  color_confidence={kp.color_confidence[index]:.4f}  3D_confidence={kp.confidence[index]:.4f}\n"
            f"RGB uv={kp.color_pixels[index]}\n"
            f"Left IR uv={kp.left_pixels[index]}\nRight IR uv={kp.right_pixels[index]}  disparity={disparity:.3f}px\n"
            f"Raw camera XYZ={kp.raw_camera_m[index] * 1000.0} mm\n"
            f"Filtered camera XYZ={kp.filtered_camera_m[index] * 1000.0} mm\n"
            f"Base XYZ={kp.base_m[index] * 1000.0} mm\n"
            f"σXYZ={sigma} mm\n"
            f"Side uv={kp.side_pixels[index]}  reproj={kp.side_reprojection_error_px[index]:.3f}px\n"
            f"MuJoCo XYZ={sample.simulation_points_base_m[index] * 1000.0} mm  error={sample.errors_mm[index]:.3f}mm"
        )

    def _update_keypoint_table(self, sample) -> None:
        kp = sample.keypoints
        for index in range(7):
            covariance = kp.covariance_m2[index]
            sigma_z = np.sqrt(max(float(covariance[2, 2]), 0.0)) * 1000.0 if np.isfinite(covariance[2, 2]) else float("nan")
            base = kp.base_m[index] * 1000.0
            values = (
                f"K{index}",
                "有效" if kp.valid[index] else ("预测" if kp.predicted[index] else "无效"),
                f"{base[0]:.2f}", f"{base[1]:.2f}", f"{base[2]:.2f}",
                f"{sigma_z:.3f}", f"{kp.confidence[index]:.2f}", f"{sample.errors_mm[index]:.2f}",
            )
            for column, value in enumerate(values):
                item = self.keypoint_table.item(index, column)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    self.keypoint_table.setItem(index, column, item)
                item.setText(value)

    def _dense_points_from_frame(
        self, frame, stride: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stride = max(1, int(stride))
        depth = np.asarray(frame.depth_m, dtype=np.float32)
        sampled = depth[::stride, ::stride]
        yy, xx = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        minimum = float(self.config["camera"].get("depth_min_m", 0.18))
        maximum = float(self.config["camera"].get("depth_max_m", 0.60))
        valid = np.isfinite(sampled) & (sampled >= minimum) & (sampled <= maximum)
        u, v, z = xx[valid].astype(np.float64), yy[valid].astype(np.float64), sampled[valid].astype(np.float64)
        intrinsics = frame.color_intrinsics
        points_color = np.column_stack((
            (u - float(intrinsics["ppx"])) * z / float(intrinsics["fx"]),
            (v - float(intrinsics["ppy"])) * z / float(intrinsics["fy"]),
            z,
        ))
        transform_left = np.asarray(frame.transform_left_from_color, dtype=float)
        points_left = points_color @ transform_left[:3, :3].T + transform_left[:3, 3]
        transform_base = np.asarray(self.engine.tracker.transform_base_from_camera, dtype=float)
        points_base = points_left @ transform_base[:3, :3].T + transform_base[:3, 3]
        if frame.color_bgr.shape[:2] == depth.shape:
            colours_bgr = frame.color_bgr[v.astype(int), u.astype(int)].copy()
        else:
            colours_bgr = np.full((len(points_base), 3), 220, dtype=np.uint8)
        pixels = np.column_stack((u, v)).astype(np.float32)
        return points_left.astype(np.float32), points_base.astype(np.float32), colours_bgr, pixels

    def _export_current_point_cloud(self, target_path: str | Path | None = None) -> Path | None:
        sample = self._latest_sample_for_inspection
        if sample is None:
            if target_path is not None:
                raise RuntimeError("当前没有可导出的D435帧")
            QtWidgets.QMessageBox.information(self, "导出点云", "当前没有可导出的D435帧。")
            return None
        interactive = target_path is None
        if interactive:
            default_path = Path(self.output_root_edit.text().strip() or Path.cwd()) / f"pointcloud_frame_{sample.frame.sequence}.ply"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出当前稠密点云", str(default_path), "PLY point cloud (*.ply)")
            if not path:
                return None
            target_path = path
        try:
            target = Path(target_path).with_suffix(".ply").resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            stride = int(self.config["camera"].get("point_cloud_export_stride_px", 1))
            points_left, points_base, colours_bgr, pixels = self._dense_points_from_frame(sample.frame, stride)
            vertices = np.empty(len(points_base), dtype=[
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ])
            vertices["x"], vertices["y"], vertices["z"] = points_base.T
            vertices["red"] = colours_bgr[:, 2]
            vertices["green"] = colours_bgr[:, 1]
            vertices["blue"] = colours_bgr[:, 0]
            header = (
                "ply\nformat binary_little_endian 1.0\n"
                f"comment frame {sample.frame.sequence} base_coordinate_meters\n"
                f"element vertex {len(vertices)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            ).encode("ascii")
            with target.open("wb") as stream:
                stream.write(header)
                stream.write(vertices.tobytes(order="C"))
            np.savez_compressed(
                target.with_suffix(".npz"),
                frame_sequence=sample.frame.sequence,
                capture_host_ns=sample.frame.capture_host_ns,
                points_left_camera_m=points_left,
                points_base_m=points_base,
                colours_bgr=colours_bgr,
                source_color_pixels=pixels,
            )
            self.statusBar().showMessage(f"已导出 {len(vertices):,} 点: {target}", 8000)
            return target
        except Exception as exc:
            if not interactive:
                raise
            QtWidgets.QMessageBox.warning(self, "点云导出失败", str(exc))
            return None

    def _browse_output_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择session输出目录", self.output_root_edit.text())
        if path:
            self.output_root_edit.setText(path)
            try:
                self.engine.set_recording_root(path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "输出目录", str(exc))

    def _save_config_from_ui(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存当前配置", str(Path.cwd() / "d435_capture_config.json"), "JSON (*.json)")
        if not path:
            return
        try:
            self.config["recording"]["root"] = self.output_root_edit.text().strip()
            selected_endoscope = self.endoscope_camera_combo.currentData()
            if isinstance(selected_endoscope, int):
                self.config.setdefault("endoscope_camera", {})["index"] = selected_endoscope
            self.config.setdefault("endoscope_camera", {})["enabled"] = bool(
                self.endoscope_enabled_check.isChecked()
            )
            self.config.setdefault("recording", {})["endoscope_enabled"] = bool(
                self.endoscope_record_check.isChecked()
            )
            self.config.setdefault("em", {}).update(
                serial_port=str(self.em_port_combo.currentData() or ""),
                rom_files=[self.em_rom_edit.text().strip()],
                primary_handle=int(self.em_handle_spin.value()),
                record_with_motor=bool(self.em_record_check.isChecked()),
            )
            save_config(self.config, path)
            self.statusBar().showMessage(f"配置已保存: {path}", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "保存配置失败", str(exc))

    def _load_config_from_ui(self) -> None:
        if (
            self.engine.running
            or self.engine.recorder.active
            or self.viewer_recorder.active
            or self.endoscope_recorder.active
            or self.em_recorder.active
            or self.em_source.active
        ):
            QtWidgets.QMessageBox.information(self, "加载配置", "请先停止相机、记录和NDI连接，再加载配置。")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "加载D435采集配置", "", "JSON (*.json)")
        if not path:
            return
        try:
            new_config = load_config(path)
            old_camera = self.camera_source
            if isinstance(old_camera, SyntheticCameraSource):
                camera_source = SyntheticCameraSource(new_config)
            else:
                camera_source = RealSenseSource(
                    new_config["camera"],
                    getattr(old_camera, "bag_path", None),
                    repeat=bool(getattr(old_camera, "repeat", False)),
                )
            axis_source = self.engine.axis_source
            if hasattr(axis_source, "config"):
                axis_source.config = new_config["axes"]
            side_source = None
            if new_config["fusion"]["mode"] == DUAL_VIEW_MODE:
                side_source = (
                    SyntheticSideRgbSource(new_config, camera_source)
                    if isinstance(camera_source, SyntheticCameraSource)
                    else OpenCvRgbSource(new_config["side_camera"])
                )
            bridge = None
            if bool(new_config["mujoco"].get("enabled", False)):
                if (
                    self._embedded_mujoco_bridge is not None
                    and str(new_config["mujoco"]["xml"]) == self._embedded_mujoco_xml
                ):
                    bridge = self._embedded_mujoco_bridge
                else:
                    bridge = MujocoAlignmentBridge(new_config["mujoco"])
            self.config = new_config
            self._stop_endoscope_camera()
            self.em_source.close()
            self.em_source = NdiEmSource(new_config.setdefault("em", {}))
            self.em_recorder = EmMotorCsvRecorder(new_config["em"])
            self.endoscope_recorder = AuxiliaryRgbRecorder(new_config)
            self.camera_source = camera_source
            self.side_camera_source = side_source
            self.engine = CaptureEngine(new_config, camera_source, axis_source, bridge, side_source)
            self.calibrator = CharucoBaseCalibrator(new_config["calibration"])
            self.side_calibrator = CharucoBaseCalibrator(new_config["calibration"])
            self.side_intrinsic_calibrator = CharucoIntrinsicCalibrator(new_config["calibration"])
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(self.mode_combo.findData(self.engine.fusion_mode))
            self.mode_combo.blockSignals(False)
            self.trio_address_edit.setText(str(new_config["axes"].get("controller", "192.168.0.250")))
            configured_roms = new_config.get("em", {}).get("rom_files", ["window/8700339.rom"])
            configured_rom = str(configured_roms[0]) if configured_roms else "window/8700339.rom"
            configured_rom_path = Path(configured_rom).expanduser()
            if not configured_rom_path.is_absolute():
                configured_rom_path = (Path(__file__).resolve().parent.parent / configured_rom_path).resolve()
            self.em_rom_edit.setText(str(configured_rom_path))
            self.em_handle_spin.setValue(int(new_config.get("em", {}).get("primary_handle", 10)))
            self.em_record_check.setChecked(bool(new_config.get("em", {}).get("record_with_motor", True)))
            self.endoscope_enabled_check.blockSignals(True)
            self.endoscope_enabled_check.setChecked(
                bool(new_config.get("endoscope_camera", {}).get("enabled", True))
            )
            self.endoscope_enabled_check.blockSignals(False)
            self.endoscope_record_check.setChecked(
                bool(new_config.get("recording", {}).get("endoscope_enabled", True))
            )
            self.em_port_combo.clear()
            self._refresh_ndi_ports()
            self.mujoco_xml_edit.setText(str(new_config["mujoco"]["xml"]))
            self.mujoco_enable_check.blockSignals(True)
            self.mujoco_enable_check.setChecked(bridge is not None)
            self.mujoco_enable_check.blockSignals(False)
            self.keypoint_tracking_check.blockSignals(True)
            self.keypoint_tracking_check.setChecked(self.engine.keypoint_tracking_enabled)
            self.keypoint_tracking_check.blockSignals(False)
            self.keypoint_tracking_status.setText(
                "已开启：RGB定位 + 左右IR极线亚像素匹配 + Depth一致性校验"
                if self.engine.keypoint_tracking_enabled
                else "已关闭：仅显示/记录原始相机流，不计算关键点与三维坐标"
            )
            self.mujoco_status.setText(
                "MuJoCo同步已启用" if bridge is not None else "MuJoCo同步已关闭，不计算仿真误差"
            )
            self.output_root_edit.setText(str(new_config["recording"]["root"]))
            self.synchronized_video_check.blockSignals(True)
            self.synchronized_video_check.setChecked(
                bool(new_config["recording"].get("synchronized_video_enabled", True))
            )
            self.synchronized_video_check.blockSignals(False)
            self.synchronized_zoom_check.blockSignals(True)
            self.synchronized_zoom_check.setChecked(
                bool(new_config.get("ui", {}).get("synchronized_zoom", True))
            )
            self.synchronized_zoom_check.blockSignals(False)
            self.side_camera_card.setVisible(self.engine.fusion_mode == DUAL_VIEW_MODE)
            self.calibration_status.setText(f"配置已加载: {path}")
            self._side_device_signature = None
            self._endoscope_device_signature = None
            self._refresh_side_camera_devices(force=True)
            self._refresh_endoscope_camera_devices(force=True)
            self._reconnect_endoscope_camera(show_errors=False)
            self._update_header_status()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "加载配置失败", str(exc))

    def _choose_identification_session(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择待辨识session", self.identify_session_edit.text())
        if path:
            self.identify_session_edit.setText(path)

    def _start_identification(self) -> None:
        session = self.identify_session_edit.text().strip()
        if not session or not Path(session).is_dir():
            QtWidgets.QMessageBox.information(self, "参数辨识", "请先选择有效的session目录。")
            return
        if self.identify_process is not None and self.identify_process.state() != QtCore.QProcess.NotRunning:
            QtWidgets.QMessageBox.information(self, "参数辨识", "参数辨识正在运行。")
            return
        process = QtCore.QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments([
            "-m", "Visual_information.d435_tdcr_capture.identify",
            "--session", session,
            "--maximum-frames", str(self.identify_frames_spin.value()),
            "--maxiter", str(self.identify_maxiter_spin.value()),
        ])
        process.setWorkingDirectory(str(Path(__file__).resolve().parent.parent))
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_identification_output)
        process.finished.connect(self._identification_finished)
        self.identify_process = process
        self.identify_button.setEnabled(False)
        self.identify_status.setText("正在运行参数辨识……")
        process.start()

    def _read_identification_output(self) -> None:
        if self.identify_process is None:
            return
        text = bytes(self.identify_process.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
        if text:
            self.identify_status.setText(text[-500:])

    def _identification_finished(self, exit_code: int, _status) -> None:
        self.identify_button.setEnabled(True)
        session = Path(self.identify_session_edit.text().strip())
        if exit_code == 0:
            self.identify_status.setText(f"辨识完成: {session / 'identified_profile.json'}")
        else:
            current = self.identify_status.text()
            self.identify_status.setText(f"辨识失败，退出码 {exit_code}\n{current}")

    def _refresh_all_rgb_camera_devices(self) -> None:
        self._refresh_side_camera_devices()
        self._refresh_endoscope_camera_devices()

    def _refresh_endoscope_camera_devices(self, force: bool = False) -> None:
        if not hasattr(self, "endoscope_camera_combo"):
            return
        if isinstance(self.camera_source, SyntheticCameraSource):
            signature = (("synthetic", "模拟内窥镜"),)
            if force or signature != self._endoscope_device_signature:
                self.endoscope_camera_combo.blockSignals(True)
                self.endoscope_camera_combo.clear()
                self.endoscope_camera_combo.addItem("模拟内窥镜", "synthetic")
                self.endoscope_camera_combo.setEnabled(False)
                self.endoscope_camera_combo.blockSignals(False)
                self._endoscope_device_signature = signature
                self.endoscope_status.setText("模拟内窥镜已就绪")
            return
        devices = discover_side_rgb_cameras()
        signature = tuple(
            (int(device["index"]), str(device["name"]), str(device["device_id"]))
            for device in devices
        )
        if not force and signature == self._endoscope_device_signature:
            return
        configured_index = int(self.config.get("endoscope_camera", {}).get("index", 0))
        preferred_name = str(
            self.config.get("endoscope_camera", {}).get("preferred_name", "")
        ).strip().casefold()
        current = self.endoscope_camera_combo.currentData()
        if isinstance(current, int):
            configured_index = current
        self.endoscope_camera_combo.blockSignals(True)
        self.endoscope_camera_combo.clear()
        selected_row = -1
        for row, device in enumerate(devices):
            index = int(device["index"])
            self.endoscope_camera_combo.addItem(
                f"{device['name']}  ·  Index {index}", index
            )
            self.endoscope_camera_combo.setItemData(
                row, str(device["device_id"]), QtCore.Qt.ToolTipRole
            )
            if index == configured_index:
                selected_row = row
        if selected_row < 0 and preferred_name:
            for row, device in enumerate(devices):
                if preferred_name in str(device["name"]).casefold():
                    selected_row = row
                    break
        if devices:
            self.endoscope_camera_combo.setCurrentIndex(
                selected_row if selected_row >= 0 else 0
            )
            self.endoscope_camera_combo.setEnabled(not self.endoscope_source or not self.endoscope_source.running)
            selected_index = int(self.endoscope_camera_combo.currentData())
            self.config.setdefault("endoscope_camera", {})["index"] = selected_index
            selected_name = self.endoscope_camera_combo.currentText().split("  ·  Index", 1)[0]
            self.config.setdefault("endoscope_camera", {})["preferred_name"] = selected_name
            if self.endoscope_source is None or not self.endoscope_source.running:
                self.endoscope_status.setText(
                    f"已检测：{selected_name}（Index {selected_index}）"
                )
        else:
            self.endoscope_camera_combo.addItem("未检测到可用内窥镜摄像头", None)
            self.endoscope_camera_combo.setEnabled(False)
            self.endoscope_status.setText("未检测到可用内窥镜摄像头")
        self.endoscope_camera_combo.blockSignals(False)
        self._endoscope_device_signature = signature

    def _selected_endoscope_camera_index(self) -> int:
        selected = self.endoscope_camera_combo.currentData()
        if not isinstance(selected, int):
            raise RuntimeError("未检测到可用内窥镜摄像头")
        return selected

    def _make_endoscope_source(self):
        if isinstance(self.camera_source, SyntheticCameraSource):
            return SyntheticSideRgbSource(self.config, self.camera_source)
        index = self._selected_endoscope_camera_index()
        if (
            self.engine.fusion_mode == DUAL_VIEW_MODE
            and isinstance(self.engine.side_camera_source, OpenCvRgbSource)
            and int(self.engine.side_camera_source.index) == index
        ):
            raise RuntimeError("内窥镜与侧面RGB不能选择同一个摄像头设备")
        settings = self.config.setdefault("endoscope_camera", {})
        settings["index"] = index
        settings["preferred_name"] = self.endoscope_camera_combo.currentText().split(
            "  ·  Index", 1
        )[0]
        settings.setdefault("source_name", "endoscope")
        return OpenCvRgbSource(settings, index)

    def _reconnect_endoscope_camera(self, *, show_errors: bool = True) -> None:
        if self.viewer_recorder.active or self.engine.recorder.active:
            if show_errors:
                QtWidgets.QMessageBox.information(
                    self, "内窥镜", "请先停止当前记录，再切换内窥镜摄像头。"
                )
            return
        old_source = self.endoscope_source
        if old_source is not None:
            old_source.stop()
        self.endoscope_source = None
        self._last_endoscope_sequence = -1
        try:
            source = self._make_endoscope_source()
            self.endoscope_source = source
            if self.endoscope_enabled_check.isChecked():
                source.start()
                self.endoscope_camera_combo.setEnabled(False)
                self.endoscope_connect_button.setText("断开内窥镜")
                self.endoscope_status.setText(
                    f"内窥镜在线：Index {getattr(source, 'index', '--')} · "
                    f"{source.device_info.get('width', '--')}×{source.device_info.get('height', '--')} "
                    f"@ {source.device_info.get('fps', '--')} FPS"
                )
                self.endoscope_view_pill.set_status("在线", "green")
            else:
                self.endoscope_status.setText("内窥镜已选择，点击“连接内窥镜”开始预览")
        except Exception as exc:
            self.endoscope_source = None
            self.endoscope_view_pill.set_status("异常", "red")
            self.endoscope_status.setText(f"内窥镜连接失败：{exc}")
            if show_errors:
                QtWidgets.QMessageBox.warning(self, "内窥镜连接失败", str(exc))

    def _start_endoscope_camera(self) -> None:
        if not self.endoscope_enabled_check.isChecked():
            return
        if self.endoscope_source is None:
            self._reconnect_endoscope_camera()
            return
        if not self.endoscope_source.running:
            self.endoscope_source.start()
        self.endoscope_camera_combo.setEnabled(False)
        self.endoscope_connect_button.setText("断开内窥镜")
        self.endoscope_view_pill.set_status("在线", "green")
        self.endoscope_status.setText(
            f"内窥镜在线：Index {getattr(self.endoscope_source, 'index', '--')} · "
            f"{self.endoscope_source.device_info.get('width', '--')}×"
            f"{self.endoscope_source.device_info.get('height', '--')} @ "
            f"{self.endoscope_source.device_info.get('fps', '--')} FPS"
        )

    def _stop_endoscope_camera(self) -> None:
        if self.endoscope_source is not None:
            self.endoscope_source.stop()
        self.endoscope_camera_combo.setEnabled(True)
        self.endoscope_connect_button.setText("连接内窥镜")
        self.endoscope_view_pill.set_status("未连接", "neutral")
        self.endoscope_status.setText("内窥镜已停止")
        self._last_endoscope_sequence = -1

    def _connect_endoscope_from_ui(self) -> None:
        if self.endoscope_source is not None and self.endoscope_source.running:
            if self.endoscope_recorder.active:
                QtWidgets.QMessageBox.information(
                    self, "内窥镜", "请先停止记录，再断开内窥镜。"
                )
                return
            self._stop_endoscope_camera()
            return
        self.endoscope_enabled_check.blockSignals(True)
        self.endoscope_enabled_check.setChecked(True)
        self.endoscope_enabled_check.blockSignals(False)
        self.config.setdefault("endoscope_camera", {})["enabled"] = True
        self._reconnect_endoscope_camera()

    def _on_endoscope_camera_selected(self, _row: int) -> None:
        self._reconnect_endoscope_camera()

    def _toggle_endoscope_enabled(self, checked: bool) -> None:
        self.config.setdefault("endoscope_camera", {})["enabled"] = bool(checked)
        if checked:
            self._reconnect_endoscope_camera()
        else:
            if self.endoscope_recorder.active:
                self.endoscope_enabled_check.blockSignals(True)
                self.endoscope_enabled_check.setChecked(True)
                self.endoscope_enabled_check.blockSignals(False)
                QtWidgets.QMessageBox.information(
                    self, "内窥镜", "请先停止记录，再关闭内窥镜预览。"
                )
                return
            self._stop_endoscope_camera()

    def _poll_endoscope_camera(self) -> None:
        source = self.endoscope_source
        if source is None or not source.running:
            return
        frame = source.poll()
        if frame is None or int(frame.sequence) == self._last_endoscope_sequence:
            return
        self._last_endoscope_sequence = int(frame.sequence)
        self._show_image(frame.image_bgr, self.endoscope_image_label)
        if self.endoscope_view_pill.text() != "在线":
            self.endoscope_view_pill.set_status("在线", "green")

    def _refresh_side_camera_devices(self, force: bool = False) -> None:
        if not hasattr(self, "side_camera_combo"):
            return
        if isinstance(self.camera_source, SyntheticCameraSource):
            signature = (("synthetic", "模拟侧相机"),)
            if force or signature != self._side_device_signature:
                self.side_camera_combo.blockSignals(True)
                self.side_camera_combo.clear()
                self.side_camera_combo.addItem("模拟侧相机", "synthetic")
                self.side_camera_combo.setEnabled(False)
                self.side_camera_combo.blockSignals(False)
                self._side_device_signature = signature
                self.side_status.setText("模拟侧相机已就绪")
            return

        devices = discover_side_rgb_cameras()
        signature = tuple(
            (int(device["index"]), str(device["name"]), str(device["device_id"]))
            for device in devices
        )
        if not force and signature == self._side_device_signature:
            return
        configured_index = int(self.config["side_camera"].get("index", 0))
        current_data = self.side_camera_combo.currentData()
        if isinstance(current_data, int):
            configured_index = current_data
        self.side_camera_combo.blockSignals(True)
        self.side_camera_combo.clear()
        selected_row = -1
        for row, device in enumerate(devices):
            index = int(device["index"])
            self.side_camera_combo.addItem(f"{device['name']}  ·  Index {index}", index)
            self.side_camera_combo.setItemData(row, str(device["device_id"]), QtCore.Qt.ToolTipRole)
            if index == configured_index:
                selected_row = row
        if devices:
            if selected_row < 0:
                selected_row = 0
            self.side_camera_combo.setCurrentIndex(selected_row)
            self.side_camera_combo.setEnabled(True)
            selected_index = int(self.side_camera_combo.currentData())
            self.config["side_camera"]["index"] = selected_index
            selected_name = str(devices[selected_row]["name"])
            if self.engine.running and self.engine.latest_side_frame is not None:
                self.side_status.setText(f"侧相机在线：{selected_name}（Index {selected_index}）")
            else:
                self.side_status.setText(f"已检测：{selected_name}（Index {selected_index}）")
        else:
            self.side_camera_combo.addItem("未检测到可用普通RGB摄像头", None)
            self.side_camera_combo.setEnabled(False)
            self.side_status.setText("未检测到可用普通RGB摄像头；RealSense D435已自动排除")
        self.side_camera_combo.blockSignals(False)
        self._side_device_signature = signature

    def _selected_side_camera_index(self) -> int:
        if not hasattr(self, "side_camera_combo"):
            return int(self.config["side_camera"].get("index", 0))
        selected = self.side_camera_combo.currentData()
        if not isinstance(selected, int):
            raise RuntimeError("未检测到可用的普通RGB侧相机，请连接设备后点击“刷新设备”")
        return selected

    def _start_capture_processing(self) -> None:
        source_has_direct_preview = callable(getattr(self.camera_source, "peek_latest", None))
        if self.engine.running and (
            self._processing_required() or not source_has_direct_preview
        ):
            self.capture_worker.start()

    def _stop_capture_processing(self) -> None:
        if self.capture_worker.running:
            self.capture_worker.stop()

    def _on_side_camera_selected(self, _row: int) -> None:
        if isinstance(self.camera_source, SyntheticCameraSource):
            return
        if self.engine.recorder.active:
            QtWidgets.QMessageBox.information(self, "切换侧相机", "请先停止当前记录，再切换侧相机。")
            self._refresh_side_camera_devices(force=True)
            return
        self._reconnect_side_camera()

    def _make_side_source(self):
        if isinstance(self.camera_source, SyntheticCameraSource):
            return SyntheticSideRgbSource(self.config, self.camera_source)
        index = self._selected_side_camera_index()
        self.config["side_camera"]["index"] = index
        return OpenCvRgbSource(self.config["side_camera"], index)

    def _reconnect_side_camera(self) -> None:
        resume_processing = self.capture_worker.running
        self._stop_capture_processing()
        try:
            source = self._make_side_source()
            self.side_camera_source = source
            self.engine.set_side_camera_source(source)
            if isinstance(source, SyntheticSideRgbSource):
                self.side_status.setText("模拟侧相机已配置")
            else:
                selected_name = self.side_camera_combo.currentText().split("  ·  Index", 1)[0]
                state = "在线" if self.engine.running else "已选择"
                self.side_status.setText(f"侧相机{state}：{selected_name}（Index {source.index}）")
            self._update_header_status()
        except Exception as exc:
            self.side_status.setText(f"侧相机配置失败: {exc}")
            QtWidgets.QMessageBox.warning(self, "侧相机失败", str(exc))
        finally:
            if resume_processing:
                self._start_capture_processing()

    def _change_fusion_mode(self, _index: int) -> None:
        mode = str(self.mode_combo.currentData())
        previous = self.engine.fusion_mode
        if mode == previous:
            self.side_camera_card.setVisible(mode == DUAL_VIEW_MODE)
            self._update_header_status()
            return
        if self.engine.recorder.active:
            QtWidgets.QMessageBox.information(self, "模式切换", "请先停止当前记录，再切换三维测量模式。")
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(self.mode_combo.findData(previous))
            self.mode_combo.blockSignals(False)
            return
        try:
            resume_processing = self.capture_worker.running
            self._stop_capture_processing()
            if mode == DUAL_VIEW_MODE and self.engine.side_camera_source is None:
                source = self._make_side_source()
                self.side_camera_source = source
                self.engine.set_side_camera_source(source)
            self.engine.set_fusion_mode(mode)
            self.compute_backend_info = configure_compute_backend(self.config)
            self.side_camera_card.setVisible(mode == DUAL_VIEW_MODE)
            if mode == DUAL_VIEW_MODE:
                ready = self.engine.side_fusion.calibration_ready
                self.side_status.setText("双视图已启用 | " + ("侧相机标定有效" if ready else "需完成侧相机内参与外参标定"))
            else:
                self.side_status.setText("侧相机未参与测量")
            self._update_header_status()
        except Exception as exc:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(self.mode_combo.findData(previous))
            self.mode_combo.blockSignals(False)
            self.side_camera_card.setVisible(previous == DUAL_VIEW_MODE)
            self.side_status.setText(f"模式切换失败: {exc}")
            self._update_header_status()
            QtWidgets.QMessageBox.warning(self, "模式切换失败", str(exc))
        finally:
            if 'resume_processing' in locals() and resume_processing:
                self._start_capture_processing()

    def _set_camera_parameter(self, key: str, value) -> None:
        camera_config = self.config.setdefault("camera", {})
        if isinstance(value, np.generic):
            value = value.item()
        camera_config[key] = value
        if key == "depth_min_m" and float(value) >= float(camera_config.get("depth_max_m", 3.0)):
            camera_config["depth_max_m"] = min(6.0, float(value) + 0.05)
        elif key == "depth_max_m" and float(value) <= float(camera_config.get("depth_min_m", 0.1)):
            camera_config["depth_min_m"] = max(0.1, float(value) - 0.05)
        if hasattr(self.camera_source, "update_runtime_settings"):
            self.camera_source.update_runtime_settings({key: value})
        if key.startswith("depth_") or key.startswith("point_cloud_"):
            self._reset_depth_display_filters()
            self._cached_point_cloud_visual = None
            self._displayed_cloud_revision = -1
        if hasattr(self, "camera_parameter_status"):
            state = "实时应用" if self.engine.running else "已保存，启动相机后应用"
            self.camera_parameter_status.setText(f"{key} = {value}  ·  {state}")

    def _apply_high_accuracy_camera_preset(self) -> None:
        values = {
            "visual_preset": "high_accuracy",
            "emitter_enabled": True,
            "laser_power": 150.0,
            "depth_auto_exposure": True,
            "depth_postprocess_enabled": True,
            "depth_spatial_enabled": True,
            "depth_spatial_magnitude": 2,
            "depth_spatial_alpha": 0.50,
            "depth_spatial_delta": 8.0,
            "depth_spatial_holes_fill": 1,
            "depth_temporal_enabled": True,
            "depth_temporal_alpha": 0.55,
            "depth_temporal_delta": 20.0,
            "depth_temporal_persistency": 1,
            "depth_hole_filling_enabled": False,
            "depth_min_m": 0.18,
            "depth_max_m": 0.60,
            "point_cloud_stride_px": 1,
            "point_cloud_max_points": 80000,
            "color_auto_exposure": True,
            "color_auto_white_balance": True,
        }
        self.config.setdefault("camera", {}).update(values)
        if hasattr(self.camera_source, "update_runtime_settings"):
            self.camera_source.update_runtime_settings(values)
        self.visual_preset_combo.blockSignals(True)
        self.visual_preset_combo.setCurrentIndex(
            max(0, self.visual_preset_combo.findData("high_accuracy"))
        )
        self.visual_preset_combo.blockSignals(False)
        for key, value in values.items():
            control = self.camera_parameter_controls.get(key)
            if control is None:
                continue
            control.blockSignals(True)
            if isinstance(control, QtWidgets.QCheckBox):
                control.setChecked(bool(value))
            elif hasattr(control, "setValue"):
                control.setValue(value)
            control.blockSignals(False)
        self.camera_parameter_controls["laser_power"].setEnabled(True)
        self.camera_parameter_controls["depth_exposure_us"].setEnabled(False)
        self.camera_parameter_controls["depth_gain"].setEnabled(False)
        self.camera_parameter_controls["color_exposure_us"].setEnabled(False)
        self.camera_parameter_controls["color_gain"].setEnabled(False)
        self._reset_depth_display_filters()
        self._cached_point_cloud_visual = None
        self._displayed_cloud_revision = -1
        self.camera_parameter_status.setText(
            "已恢复近距离高精度参数；SDK将在下一帧应用。"
        )

    def _apply_high_density_camera_preset(self) -> None:
        values = {
            "visual_preset": "high_density",
            "emitter_enabled": True,
            "laser_power": 360.0,
            "depth_auto_exposure": True,
            "depth_postprocess_enabled": True,
            "native_depth_colorizer_enabled": True,
            "depth_colorizer_histogram_equalization": True,
            "depth_colorizer_scheme": 0,
            "depth_spatial_enabled": True,
            "depth_spatial_magnitude": 2,
            "depth_spatial_alpha": 0.50,
            "depth_spatial_delta": 8.0,
            "depth_spatial_holes_fill": 2,
            "depth_temporal_enabled": True,
            "depth_temporal_alpha": 0.45,
            "depth_temporal_delta": 20.0,
            "depth_temporal_persistency": 3,
            "depth_hole_filling_enabled": False,
            "depth_display_persist_frames": 5,
            "point_cloud_filter_width_px": 848,
            "point_cloud_render_width_px": 848,
            "point_cloud_max_points": 120000,
        }
        self.config.setdefault("camera", {}).update(values)
        self.config.setdefault("ui", {})["depth_view_space"] = "native"
        if hasattr(self.camera_source, "update_runtime_settings"):
            self.camera_source.update_runtime_settings(values)
        self.visual_preset_combo.blockSignals(True)
        self.visual_preset_combo.setCurrentIndex(
            max(0, self.visual_preset_combo.findData("high_density"))
        )
        self.visual_preset_combo.blockSignals(False)
        self.depth_space_mode.blockSignals(True)
        self.depth_space_mode.setCurrentIndex(
            max(0, self.depth_space_mode.findData("native"))
        )
        self.depth_space_mode.blockSignals(False)
        self.depth_colorizer_scheme_combo.blockSignals(True)
        self.depth_colorizer_scheme_combo.setCurrentIndex(
            max(0, self.depth_colorizer_scheme_combo.findData(0))
        )
        self.depth_colorizer_scheme_combo.blockSignals(False)
        for key, value in values.items():
            control = self.camera_parameter_controls.get(key)
            if control is None:
                continue
            control.blockSignals(True)
            if isinstance(control, QtWidgets.QCheckBox):
                control.setChecked(bool(value))
            elif hasattr(control, "setValue"):
                control.setValue(value)
            control.blockSignals(False)
        self._reset_depth_display_filters()
        self._last_depth_preview_ns = 0
        self.camera_parameter_status.setText(
            "已启用Viewer稠密预设、原生Depth和SDK Colorizer。"
        )

    def _update_camera_parameter_telemetry(self, frame) -> None:
        if not hasattr(self, "camera_parameter_status"):
            return
        metadata = frame.metadata or {}
        device = metadata.get("device", {})
        usb = str(device.get("usb", "--"))
        exposure = metadata.get("actual_exposure", "--")
        gain = metadata.get("gain_level", "--")
        native_view = (
            str(self.config.get("ui", {}).get("depth_view_space", "native"))
            == "native"
        )
        native_depth = getattr(frame, "native_depth_m", None)
        depth = np.asarray(
            native_depth if native_view and native_depth is not None else frame.depth_m
        )
        minimum = float(self.config["camera"].get("depth_min_m", 0.18))
        maximum = float(self.config["camera"].get("depth_max_m", 0.60))
        valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
        valid_percent = 100.0 * float(np.count_nonzero(valid)) / max(valid.size, 1)
        self.camera_parameter_status.setText(
            f"USB {usb}  ·  Exposure {exposure}  ·  Gain {gain}  ·  "
            f"{'原生' if native_view else 'RGB对齐'}有效深度 {valid_percent:.1f}%"
        )

    def _set_source(self, source) -> bool:
        if (
            self.engine.recorder.active
            or self.viewer_recorder.active
            or self.endoscope_recorder.active
        ):
            QtWidgets.QMessageBox.information(self, "切换数据源", "请先停止当前记录，再切换D435数据源。")
            return False
        was_running = self.engine.running
        if was_running:
            self._stop_capture_processing()
            self.engine.stop()
        self.camera_source = source
        self.engine.camera_source = source
        self._reset_depth_display_filters()
        self._sdk_frame_samples.clear()
        self._last_depth_preview_ns = 0
        self._last_direct_sdk_sequence = -1
        source_transform = getattr(source, "transform_base_from_camera", None)
        if source_transform is not None:
            self.engine.set_d435_calibration(source_transform, ready=True)
        self._side_device_signature = None
        self._endoscope_device_signature = None
        self._refresh_side_camera_devices(force=True)
        self._refresh_endoscope_camera_devices(force=True)
        if self.engine.fusion_mode == DUAL_VIEW_MODE:
            self.side_camera_source = self._make_side_source()
            self.engine.set_side_camera_source(self.side_camera_source)
        if was_running:
            self.engine.start()
            self._start_capture_processing()
        self._update_header_status()
        return True

    def _toggle_camera(self, checked: bool) -> None:
        try:
            if checked:
                self.engine.start()
                self._start_capture_processing()
                self.start_button.setText("停止相机")
                info = {
                    "D435": getattr(self.camera_source, "device_info", {}),
                    "side": getattr(self.engine.side_camera_source, "device_info", None),
                    "mode": self.engine.fusion_mode,
                    "compute": self.compute_backend_info,
                }
                self.device_status.setText("已连接: " + json.dumps(info, ensure_ascii=False))
                self._update_header_status()
            else:
                if (
                    self.viewer_recorder.active
                    or self.engine.recorder.active
                    or self.endoscope_recorder.active
                    or self.em_recorder.active
                ):
                    self.record_button.blockSignals(True)
                    self.record_button.setChecked(True)
                    self.record_button.blockSignals(False)
                    self.start_button.blockSignals(True)
                    self.start_button.setChecked(True)
                    self.start_button.blockSignals(False)
                    QtWidgets.QMessageBox.information(
                        self,
                        "正在记录",
                        "D435、内窥镜和NDI是独立设备。请先点击“停止采集/停止记录”完整结束本次session，再单独停止D435。",
                    )
                    return
                self._stop_capture_processing()
                self.engine.stop()
                self.start_button.setText("启动相机")
                self.device_status.setText("相机已停止")
                self._update_header_status()
        except Exception as exc:
            self.start_button.blockSignals(True)
            self.start_button.setChecked(False)
            self.start_button.blockSignals(False)
            self.device_status.setText(f"启动失败: {exc}")
            self._update_header_status()
            QtWidgets.QMessageBox.warning(self, "D435启动失败", str(exc))

    def _open_bag(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择RealSense Bag", "", "RealSense Bag (*.bag)")
        if path:
            if self._set_source(RealSenseSource(self.config["camera"], path, repeat=True)):
                self.device_status.setText(f"已选择回放: {path}")

    def _use_live_d435(self) -> None:
        if self._set_source(RealSenseSource(self.config["camera"])):
            self.device_status.setText("已切换到实时D435，点击“启动相机”连接设备")

    def _use_synthetic(self) -> None:
        if self._set_source(SyntheticCameraSource(self.config)):
            self.device_status.setText("已切换到模拟D435数据源")

    def _capture_calibration(self) -> None:
        target = str(self.calibration_target.currentData())
        if target == "d435_extrinsic":
            if self.engine.latest_frame is None:
                QtWidgets.QMessageBox.information(self, "标定", "请先启动D435并显示标定板。")
                return
            accepted, message, overlay = self.calibrator.add_frame(self.engine.latest_frame)
        elif target == "side_intrinsic":
            if self.engine.latest_side_frame is None:
                QtWidgets.QMessageBox.information(self, "标定", "请启用模式1并启动侧相机。")
                return
            accepted, message, overlay = self.side_intrinsic_calibrator.add_frame(self.engine.latest_side_frame)
        else:
            if self.engine.latest_side_frame is None:
                QtWidgets.QMessageBox.information(self, "标定", "请启用模式1并启动侧相机。")
                return
            if not bool(self.config["side_camera"].get("intrinsics_ready", False)):
                QtWidgets.QMessageBox.information(self, "标定", "请先完成或加载侧相机内参标定。")
                return
            accepted, message, overlay = self.side_calibrator.add_rgb_frame(self.engine.latest_side_frame)
        self.calibration_preview = overlay
        self.calibration_preview_target = target
        self.calibration_status.setText(message)
        if not accepted:
            self.statusBar().showMessage(message, 3000)

    def _solve_calibration(self) -> None:
        try:
            target = str(self.calibration_target.currentData())
            if target == "d435_extrinsic":
                result = self.calibrator.solve()
                default_name = "d435_calibration.json"
            elif target == "side_intrinsic":
                result = self.side_intrinsic_calibrator.solve()
                default_name = "side_camera_intrinsics.json"
            else:
                result = self.side_calibrator.solve()
                result["intrinsics"] = dict(self.config["side_camera"]["intrinsics"])
                result["intrinsics_ready"] = True
                default_name = "side_camera_calibration.json"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "保存标定", str(Path.cwd() / default_name), "JSON (*.json)"
            )
            if not path:
                return
            save_calibration(result, path)
            if target == "d435_extrinsic":
                self.config["calibration"].update(result)
                self.engine.set_d435_calibration(np.asarray(result["transform_base_from_camera"]), ready=True)
                message = (
                    f"D435外参完成: {result['accepted_frames']}帧 | "
                    f"重投影 {result['mean_reprojection_error_px']:.3f}px | "
                    f"平移P95 {result['translation_repeatability_p95_mm']:.3f}mm"
                )
            elif target == "side_intrinsic":
                self.config["side_camera"].update(result)
                if self.engine.side_camera_source is not None and hasattr(self.engine.side_camera_source, "set_intrinsics"):
                    self.engine.side_camera_source.set_intrinsics(result["intrinsics"], ready=True)
                message = (
                    f"侧相机内参完成: {result['captured_frames']}帧 | "
                    f"重投影 {result['mean_reprojection_error_px']:.3f}px"
                )
            else:
                self.config["side_camera"].update({
                    "transform_base_from_camera": result["transform_base_from_camera"],
                    "calibration_ready": True,
                })
                self.engine.set_side_calibration(
                    np.asarray(result["transform_base_from_camera"]),
                    self.config["side_camera"]["intrinsics"],
                    ready=True,
                )
                message = (
                    f"侧相机外参完成: {result['accepted_frames']}帧 | "
                    f"重投影 {result['mean_reprojection_error_px']:.3f}px | "
                    f"平移P95 {result['translation_repeatability_p95_mm']:.3f}mm"
                )
            self.calibration_status.setText(message)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "标定失败", str(exc))

    def _load_calibration(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "加载标定", "", "JSON (*.json)")
        if not path:
            return
        try:
            target = str(self.calibration_target.currentData())
            if target == "side_intrinsic":
                result = load_intrinsic_calibration(path)
                self.config["side_camera"].update(result)
                if self.engine.side_camera_source is not None and hasattr(self.engine.side_camera_source, "set_intrinsics"):
                    self.engine.side_camera_source.set_intrinsics(result["intrinsics"], ready=True)
                self.calibration_status.setText(f"已加载侧相机内参: {path}")
            elif target == "side_extrinsic":
                result = load_calibration(path)
                intrinsics = result.get("intrinsics", self.config["side_camera"]["intrinsics"])
                intrinsics_ready = bool(
                    result.get("intrinsics_ready", self.config["side_camera"].get("intrinsics_ready", False))
                )
                if not intrinsics_ready:
                    raise ValueError("侧相机外参文件不包含有效内参；请先加载侧相机内参标定。")
                self.config["side_camera"].update({
                    "transform_base_from_camera": result["transform_base_from_camera"],
                    "calibration_ready": True,
                    "intrinsics": intrinsics,
                    "intrinsics_ready": intrinsics_ready,
                })
                self.engine.set_side_calibration(np.asarray(result["transform_base_from_camera"]), intrinsics, ready=True)
                self.calibration_status.setText(f"已加载侧相机外参: {path}")
            else:
                result = load_calibration(path)
                self.config["calibration"].update(result)
                self.engine.set_d435_calibration(np.asarray(result["transform_base_from_camera"]), ready=True)
                self.calibration_status.setText(f"已加载D435标定: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "加载标定失败", str(exc))

    def _start_em_recording_for_session(self, session: str | Path) -> str:
        if not self.em_record_check.isChecked():
            self.em_recorder.error = None
            return "EM记录：已关闭"
        if not self.em_source.active:
            self.em_recorder.error = "已勾选EM记录，但NDI未连接"
            return "EM记录：NDI未连接，本次不保存EM"
        try:
            self.config.setdefault("em", {})["primary_handle"] = int(self.em_handle_spin.value())
            self.em_recorder = EmMotorCsvRecorder(self.config["em"])
            path = self.em_recorder.start(session, self.engine.axis_sampler, self.em_source)
            self.em_record_check.setEnabled(False)
            return f"EM同步：{path.name}（逐电机采样）"
        except Exception as exc:
            self.em_recorder.error = f"{type(exc).__name__}: {exc}"
            return f"EM记录启动失败：{exc}"

    def _stop_em_recording(self) -> str:
        if not self.em_recorder.active:
            self.em_record_check.setEnabled(True)
            return ""
        try:
            self.em_recorder.stop()
        except Exception as exc:
            self.em_recorder.error = f"{type(exc).__name__}: {exc}"
        self.em_record_check.setEnabled(True)
        if self.em_recorder.error:
            return f"EM写入异常：{self.em_recorder.error}"
        return (
            f"EM {self.em_recorder.row_count}行 · "
            f"NDI新帧 {self.em_recorder.unique_frame_count}"
        )

    def _start_endoscope_recording_for_session(self, session: str | Path) -> str:
        if not self.endoscope_record_check.isChecked():
            self.endoscope_recorder.error = None
            return "内窥镜记录：已关闭"
        if self.endoscope_source is None or not self.endoscope_source.running:
            self.endoscope_recorder.error = "已勾选内窥镜记录，但摄像头未连接"
            return "内窥镜记录：摄像头未连接，本次不保存"
        try:
            self.endoscope_recorder = AuxiliaryRgbRecorder(self.config)
            path = self.endoscope_recorder.start(session, self.endoscope_source)
            self.endoscope_record_check.setEnabled(False)
            return f"内窥镜视频：{path.name}"
        except Exception as exc:
            self.endoscope_recorder.error = f"{type(exc).__name__}: {exc}"
            return f"内窥镜记录启动失败：{exc}"

    def _stop_endoscope_recording(self) -> str:
        if not self.endoscope_recorder.active:
            self.endoscope_record_check.setEnabled(True)
            return ""
        try:
            self.endoscope_recorder.stop()
        except Exception as exc:
            self.endoscope_recorder.error = f"{type(exc).__name__}: {exc}"
        self.endoscope_record_check.setEnabled(True)
        if self.endoscope_recorder.error:
            return f"内窥镜写入异常：{self.endoscope_recorder.error}"
        return (
            f"内窥镜 {self.endoscope_recorder.frame_count}帧 · "
            f"丢帧 {self.endoscope_recorder.dropped_count}"
        )

    def _validate_requested_recording_sources(self) -> None:
        if self.endoscope_record_check.isChecked() and (
            self.endoscope_source is None or not self.endoscope_source.running
        ):
            raise RuntimeError(
                "已勾选内窥镜记录，但内窥镜摄像头尚未连接。请先连接，或取消该记录项。"
            )
        if self.em_record_check.isChecked() and not self.em_source.active:
            raise RuntimeError(
                "已勾选EM记录，但NDI尚未连接。请先连接NDI，或取消该记录项。"
            )

    def _abort_recording_start(self) -> None:
        self._stop_endoscope_recording()
        self._stop_em_recording()
        try:
            if self.viewer_recorder.active:
                self.viewer_recorder.stop()
        except Exception:
            pass
        try:
            if self.engine.recorder.active:
                self.engine.stop_recording()
        except Exception:
            pass

    def _show_recording_success(self, session: str | Path, detail: str) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "记录成功",
            f"本次采集已完整保存并通过文件校验。\n\n保存目录：\n{session}\n\n{detail}",
        )

    def _toggle_recording(self, checked: bool) -> None:
        if self.workflow_mode == "capture":
            try:
                if checked:
                    self._validate_requested_recording_sources()
                    selected = self._selected_viewer_streams()
                    if not selected:
                        raise ValueError("请至少勾选 RGB、Depth 或 3D点云中的一个窗口")
                    if not self.engine.running:
                        self.engine.start()
                        self.start_button.blockSignals(True)
                        self.start_button.setChecked(True)
                        self.start_button.setText("停止相机")
                        self.start_button.blockSignals(False)
                    output_root = self.output_root_edit.text().strip() or None
                    session = self.viewer_recorder.start(
                        selected,
                        root=output_root,
                        camera_info=getattr(self.camera_source, "device_info", {}),
                        camera_source=self.camera_source,
                        axis_sampler=self.engine.axis_sampler,
                    )
                    em_status = self._start_em_recording_for_session(session)
                    endoscope_status = self._start_endoscope_recording_for_session(session)
                    if self.em_recorder.error or self.endoscope_recorder.error:
                        raise RuntimeError(
                            self.em_recorder.error or self.endoscope_recorder.error
                        )
                    self.last_session_path = str(session)
                    self.record_button.setText("停止采集")
                    for control in (
                        self.record_rgb_check,
                        self.record_depth_check,
                        self.record_pointcloud_check,
                    ):
                        control.setEnabled(False)
                    files = ", ".join(
                        ViewerStreamRecorder.STREAM_FILES[name]
                        for name in sorted(selected)
                    )
                    self.record_status.setText(
                        f"正在采集：{session}\n独立视频：{files}\n"
                        "相机对齐电机：camera_axes.csv\n"
                        "原始高频电机：motor_axes_100hz.csv\n"
                        f"{em_status}\n{endoscope_status}"
                    )
                else:
                    endoscope_status = self._stop_endoscope_recording()
                    em_status = self._stop_em_recording()
                    session = self.viewer_recorder.stop()
                    self.record_button.setText("开始记录")
                    for control in (
                        self.record_rgb_check,
                        self.record_depth_check,
                        self.record_pointcloud_check,
                    ):
                        control.setEnabled(True)
                    output_counts = self.viewer_recorder.encoded_counts
                    codecs = self.viewer_recorder.writer_codecs
                    dropped = self.viewer_recorder.dropped_packets
                    error = self.viewer_recorder.error
                    detail = " · ".join(
                        f"{name} {output_counts[name]}帧/{codecs.get(name, '--')}"
                        for name in sorted(self.viewer_recorder.selected)
                    )
                    result = f"已保存：{session}\n{detail} · 丢包 {dropped}"
                    if em_status:
                        result += f"\n{em_status}"
                    if endoscope_status:
                        result += f"\n{endoscope_status}"
                    if error:
                        result += f"\n写入异常：{error}"
                    self.record_status.setText(result)
                    if not error and not self.em_recorder.error and not self.endoscope_recorder.error:
                        self._show_recording_success(session, f"{detail}\n{em_status}\n{endoscope_status}")
                self._update_header_status()
            except Exception as exc:
                self._abort_recording_start()
                self.record_button.blockSignals(True)
                self.record_button.setChecked(False)
                self.record_button.blockSignals(False)
                for control in (
                    self.record_rgb_check,
                    self.record_depth_check,
                    self.record_pointcloud_check,
                ):
                    control.setEnabled(True)
                self._update_header_status()
                QtWidgets.QMessageBox.warning(self, "采集失败", str(exc))
            return
        self._stop_capture_processing()
        try:
            if checked:
                self._validate_requested_recording_sources()
                output_root = self.output_root_edit.text().strip()
                if output_root:
                    self.engine.set_recording_root(output_root)
                    if not self.engine.running:
                        self.engine.start()
                        self.start_button.blockSignals(True)
                    self.start_button.setChecked(True)
                    self.start_button.setText("停止相机")
                    self.start_button.blockSignals(False)
                session = self.engine.start_recording()
                em_status = self._start_em_recording_for_session(session)
                endoscope_status = self._start_endoscope_recording_for_session(session)
                if self.em_recorder.error or self.endoscope_recorder.error:
                    raise RuntimeError(
                        self.em_recorder.error or self.endoscope_recorder.error
                    )
                self.synchronized_video_check.setEnabled(False)
                self.last_session_path = session
                self.identify_session_edit.setText(session)
                self.record_button.setText("停止记录")
                video_status = "双RGB同步视频已开启" if self.synchronized_video_check.isChecked() else "仅记录数据"
                self.record_status.setText(
                    f"正在记录: {session}\n{video_status}\n{em_status}\n{endoscope_status}"
                )
                self._update_header_status()
                self._start_capture_processing()
            else:
                endoscope_status = self._stop_endoscope_recording()
                em_status = self._stop_em_recording()
                session = self.engine.stop_recording()
                self.synchronized_video_check.setEnabled(True)
                self.record_button.setText("开始记录")
                suffix = "".join(
                    f"\n{text}" for text in (em_status, endoscope_status) if text
                )
                self.record_status.setText(f"已保存: {session}{suffix}")
                recorder_error = getattr(self.engine.recorder, "error", None)
                if not recorder_error and not self.em_recorder.error and not self.endoscope_recorder.error:
                    self._show_recording_success(
                        session, f"实验数据已保存{suffix}"
                    )
                self._update_header_status()
                self._start_capture_processing()
        except Exception as exc:
            self._abort_recording_start()
            self.record_button.blockSignals(True)
            self.record_button.setChecked(False)
            self.record_button.blockSignals(False)
            self.synchronized_video_check.setEnabled(True)
            self._update_header_status()
            QtWidgets.QMessageBox.warning(self, "记录失败", str(exc))
            self._start_capture_processing()

    def _show_image(
        self,
        image_bgr: np.ndarray,
        target_label=None,
        *,
        logical_size: tuple[int, int] | None = None,
    ) -> None:
        target_label = target_label or self.image_label
        if isinstance(target_label, ZoomableImageLabel):
            target_label.set_bgr_image(image_bgr, logical_size=logical_size)
            return
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QtGui.QImage(rgb.data, width, height, channels * width, QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
            target_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        target_label.setPixmap(pixmap)

    def _reset_depth_display_filters(self) -> None:
        """Reset preview-only history without touching camera/recorded depth."""

        self._depth_preview_m = None
        self._depth_preview_age = None
        self._cloud_depth_preview_m = None
        self._cloud_depth_preview_age = None

    def _filtered_depth_preview(
        self,
        depth_m: np.ndarray,
        *,
        state: str = "depth",
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Stabilise display depth while preserving discontinuities.

        Small holes are filled only when enough neighbours agree on a local
        surface.  A previously valid pixel may survive a few missing frames,
        but only while the current neighbourhood remains geometrically
        compatible.  This removes flickering stereo dropouts without smearing
        foreground silhouettes into the background.
        """

        lock = (
            self._cloud_depth_filter_lock
            if state == "cloud" else self._depth_preview_filter_lock
        )
        previous_name = "_cloud_depth_preview_m" if state == "cloud" else "_depth_preview_m"
        age_name = "_cloud_depth_preview_age" if state == "cloud" else "_depth_preview_age"
        with lock:
            depth = np.asarray(depth_m, dtype=np.float32)
            minimum = float(self.config["camera"].get("depth_min_m", 0.18))
            maximum = float(self.config["camera"].get("depth_max_m", 0.60))
            raw_valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
            kernel_size = int(self.config["camera"].get("depth_display_median_ksize", 3))
            kernel_size = max(1, kernel_size | 1)
            valid_u8 = raw_valid.astype(np.uint8)
            neighbor_count = cv2.boxFilter(
                valid_u8, cv2.CV_16U, (kernel_size, kernel_size), normalize=False
            )
            minimum_neighbors = int(self.config["camera"].get("depth_display_min_neighbors", 5))
            depth_zeroed = np.where(raw_valid, depth, 0.0).astype(np.float32)
            spatial_candidate = (
                cv2.medianBlur(depth_zeroed, kernel_size)
                if kernel_size > 1 else depth_zeroed
            )
            edge_threshold = max(
                0.0,
                float(self.config["camera"].get("depth_display_edge_threshold_m", 0.004)),
            )
            candidate_valid = (spatial_candidate >= minimum) & (spatial_candidate <= maximum)
            preserve_edge = np.abs(spatial_candidate - depth) <= edge_threshold
            smooth = raw_valid & candidate_valid & preserve_edge
            spatial = depth.copy()
            spatial[smooth] = spatial_candidate[smooth]
            valid = raw_valid & (neighbor_count >= minimum_neighbors)
            filtered = np.full_like(depth, np.nan)
            filtered[valid] = spatial[valid]

            # Normalised local mean ignores invalid zero-depth pixels.  Only
            # fill holes inside a coherent surface, never across a depth edge.
            hole_kernel_size = int(
                self.config["camera"].get("depth_display_hole_kernel_size", 5)
            )
            hole_kernel_size = max(3, hole_kernel_size | 1)
            hole_kernel = np.ones((hole_kernel_size, hole_kernel_size), np.uint8)
            hole_neighbor_count = cv2.boxFilter(
                valid_u8,
                cv2.CV_16U,
                (hole_kernel_size, hole_kernel_size),
                normalize=False,
            )
            local_sum = cv2.boxFilter(
                depth_zeroed,
                cv2.CV_32F,
                (hole_kernel_size, hole_kernel_size),
                normalize=False,
            )
            local_mean = local_sum / np.maximum(
                hole_neighbor_count.astype(np.float32), 1.0
            )
            local_min = cv2.erode(
                np.where(raw_valid, depth, maximum + 1.0).astype(np.float32),
                hole_kernel,
            )
            local_max = cv2.dilate(depth_zeroed, hole_kernel)
            hole_neighbors = int(
                self.config["camera"].get("depth_display_hole_min_neighbors", minimum_neighbors)
            )
            hole_edge = max(
                0.0,
                float(
                    self.config["camera"].get("depth_display_hole_edge_threshold_m", 0.012)
                ),
            )
            holes = (
                bool(self.config["camera"].get("depth_display_hole_fill_enabled", True))
                & ~raw_valid
                & (hole_neighbor_count >= hole_neighbors)
                & ((local_max - local_min) <= hole_edge)
                & (local_mean >= minimum)
                & (local_mean <= maximum)
            )
            filtered[holes] = local_mean[holes]
            valid |= holes

            alpha = float(
                np.clip(
                    self.config["camera"].get("depth_display_temporal_alpha", 0.72),
                    0.0,
                    1.0,
                )
            )
            temporal_delta = max(
                0.0,
                float(self.config["camera"].get("depth_display_temporal_delta_m", 0.006)),
            )
            previous = getattr(self, previous_name)
            previous_age = getattr(self, age_name)
            if previous is not None and previous.shape == filtered.shape:
                previous_valid = np.isfinite(previous)
                both = valid & previous_valid & (np.abs(filtered - previous) <= temporal_delta)
                filtered[both] = alpha * filtered[both] + (1.0 - alpha) * previous[both]

                if previous_age is None or previous_age.shape != filtered.shape:
                    previous_age = np.zeros(filtered.shape, dtype=np.uint8)
                hold_frames = max(
                    0, int(self.config["camera"].get("depth_display_persist_frames", 3))
                )
                persist_delta = max(
                    temporal_delta,
                    float(self.config["camera"].get("depth_display_persist_delta_m", 0.015)),
                )
                persist = (
                    ~valid
                    & previous_valid
                    & (previous_age < hold_frames)
                    & (hole_neighbor_count >= hole_neighbors)
                    & (np.abs(previous - local_mean) <= persist_delta)
                )
                filtered[persist] = previous[persist]
                valid |= persist
                age = np.zeros(filtered.shape, dtype=np.uint8)
                missing_now = valid & ~raw_valid
                age[missing_now] = np.minimum(
                    previous_age[missing_now].astype(np.uint16) + 1, 255
                ).astype(np.uint8)
            else:
                age = np.zeros(filtered.shape, dtype=np.uint8)
                age[valid & ~raw_valid] = 1
            setattr(self, previous_name, filtered.copy())
            setattr(self, age_name, age)
            return filtered, valid, minimum, maximum

    def _depth_information_map(
        self,
        depth_m: np.ndarray,
        prepared_depth: tuple[np.ndarray, np.ndarray, float, float] | None = None,
    ) -> np.ndarray:
        depth = np.asarray(depth_m, dtype=np.float32)
        filtered, valid, minimum, maximum = (
            prepared_depth if prepared_depth is not None else self._filtered_depth_preview(depth)
        )
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            normalized[valid] = np.clip(
                (filtered[valid] - minimum) * 255.0 / max(maximum - minimum, 1e-6),
                0.0,
                255.0,
            ).astype(np.uint8)
        colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        colored[~valid] = (12, 13, 16)
        valid_percent = 100.0 * float(np.count_nonzero(valid)) / max(valid.size, 1)
        if np.any(valid):
            median = float(np.median(filtered[valid]))
            status = f"FILTERED  {minimum:.2f}-{maximum:.2f} m   median {median:.3f} m   valid {valid_percent:.1f}%"
        else:
            status = f"{minimum:.2f}-{maximum:.2f} m   no valid depth"
        cv2.rectangle(colored, (0, 0), (colored.shape[1], 27), (12, 13, 16), -1)
        cv2.putText(
            colored,
            status,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (235, 235, 240),
            1,
            cv2.LINE_AA,
        )
        bar_width = max(80, min(190, colored.shape[1] // 3))
        x1 = colored.shape[1] - 10
        x0 = x1 - bar_width
        if x0 > 8 and colored.shape[0] > 50:
            gradient = np.linspace(255, 0, bar_width, dtype=np.uint8)[None, :]
            gradient = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
            colored[-15:-7, x0:x1] = gradient
            cv2.putText(colored, f"{minimum:.2f}", (x0, colored.shape[0] - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (230, 230, 235), 1, cv2.LINE_AA)
            cv2.putText(colored, f"{maximum:.2f}m", (x1 - 42, colored.shape[0] - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (230, 230, 235), 1, cv2.LINE_AA)
        return colored

    def _point_cloud_information_map(
        self,
        frame,
        prepared_depth: tuple[np.ndarray, np.ndarray, float, float] | None = None,
        sdk_filtered: bool = False,
    ) -> np.ndarray:
        """Render the D435 hardware IR-stereo depth as an RGB-coloured 3D cloud."""

        original_depth = np.asarray(frame.depth_m, dtype=np.float32)
        original_height, original_width = original_depth.shape
        if prepared_depth is not None:
            filtered, valid, minimum, maximum = prepared_depth
        elif sdk_filtered:
            if bool(self.config["camera"].get("point_cloud_filter_enabled", True)):
                filter_width = min(
                    original_width,
                    max(
                        320,
                        int(self.config["camera"].get("point_cloud_filter_width_px", 640)),
                    ),
                )
                if filter_width != original_width:
                    depth = cv2.resize(
                        original_depth,
                        (
                            filter_width,
                            max(1, int(round(original_height * filter_width / original_width))),
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    depth = original_depth
                filtered, valid, minimum, maximum = self._filtered_depth_preview(
                    depth, state="cloud"
                )
            else:
                minimum = float(self.config["camera"].get("depth_min_m", 0.18))
                maximum = float(self.config["camera"].get("depth_max_m", 0.60))
                valid = (
                    np.isfinite(original_depth)
                    & (original_depth >= minimum)
                    & (original_depth <= maximum)
                )
                filtered = np.where(valid, original_depth, np.nan)
        else:
            filtered, valid, minimum, maximum = self._filtered_depth_preview(
                frame.depth_m, state="cloud"
            )
        source_height, source_width = filtered.shape
        render_width = min(
            source_width,
            max(320, int(self.config["camera"].get("point_cloud_render_width_px", 720))),
        )
        render_height = max(
            180, int(round(source_height * render_width / max(source_width, 1)))
        )
        canvas = np.full(
            (render_height, render_width, 3), (12, 13, 16), dtype=np.uint8
        )
        stride = max(1, int(self.config["camera"].get("point_cloud_stride_px", 1)))
        yy, xx = np.mgrid[0:source_height:stride, 0:source_width:stride]
        z = filtered[::stride, ::stride]
        mask = np.isfinite(z)
        valid_percent = 100.0 * float(np.count_nonzero(mask)) / max(mask.size, 1)
        if not np.any(mask):
            self._point_cloud_pick_left_m = None
            self._point_cloud_pick_base_m = None
            cv2.putText(canvas, "IR STEREO POINT CLOUD - NO VALID DEPTH", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 215, 225), 1, cv2.LINE_AA)
            return canvas
        u = xx[mask].astype(np.float32)
        v = yy[mask].astype(np.float32)
        z = z[mask].astype(np.float32)
        intrinsics = frame.color_intrinsics
        scale_x = source_width / max(original_width, 1)
        scale_y = source_height / max(original_height, 1)
        fx = float(intrinsics["fx"]) * scale_x
        fy = float(intrinsics["fy"]) * scale_y
        ppx = float(intrinsics["ppx"]) * scale_x
        ppy = float(intrinsics["ppy"]) * scale_y
        x = (u - ppx) * z / fx
        y = (v - ppy) * z / fy
        points = np.column_stack((x, y, z)).astype(np.float32)
        maximum_points = max(1000, int(self.config["camera"].get("point_cloud_max_points", 120000)))
        if len(points) > maximum_points:
            indices = np.linspace(0, len(points) - 1, maximum_points, dtype=int)
            points, u, v, z = points[indices], u[indices], v[indices], z[indices]

        transform_left = np.asarray(frame.transform_left_from_color, dtype=np.float32)
        points_left = points @ transform_left[:3, :3].T + transform_left[:3, 3]
        transform_base = np.asarray(self.engine.tracker.transform_base_from_camera, dtype=np.float32)
        points_base = points_left @ transform_base[:3, :3].T + transform_base[:3, 3]

        colour_image = frame.color_bgr
        if colour_image.size:
            colour_height, colour_width = colour_image.shape[:2]
            colour_u = np.clip(
                np.round(u * colour_width / max(source_width, 1)).astype(int),
                0,
                colour_width - 1,
            )
            colour_v = np.clip(
                np.round(v * colour_height / max(source_height, 1)).astype(int),
                0,
                colour_height - 1,
            )
            colours = colour_image[
                colour_v,
                colour_u,
            ]
        else:
            depth_colour = np.clip((z - minimum) * 255.0 / max(maximum - minimum, 1e-6), 0, 255).astype(np.uint8)
            colours = cv2.applyColorMap(255 - depth_colour[:, None], cv2.COLORMAP_TURBO)[:, 0]

        center = np.median(points, axis=0)
        centered = points - center
        yaw, pitch = np.deg2rad(-28.0), np.deg2rad(18.0)
        rotation_y = np.asarray([
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ], dtype=np.float32)
        rotation_x = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ], dtype=np.float32)
        view_points = centered @ (rotation_x @ rotation_y).T
        horizontal = np.percentile(np.abs(view_points[:, 0]), 98.0)
        vertical = np.percentile(np.abs(view_points[:, 1]), 98.0)
        scale = 0.43 * min(
            render_width / max(horizontal, 1e-4),
            render_height / max(vertical, 1e-4),
        )
        screen_x = np.round(render_width * 0.5 + view_points[:, 0] * scale).astype(int)
        screen_y = np.round(render_height * 0.54 - view_points[:, 1] * scale).astype(int)
        on_screen = (
            (screen_x >= 1)
            & (screen_x < render_width - 1)
            & (screen_y >= 31)
            & (screen_y < render_height - 1)
        )
        screen_x, screen_y = screen_x[on_screen], screen_y[on_screen]
        colours = colours[on_screen]
        screen_points_left = points_left[on_screen]
        screen_points_base = points_base[on_screen]
        view_depth = view_points[on_screen, 2]
        order = np.argsort(view_depth)[::-1]
        screen_x, screen_y, colours = screen_x[order], screen_y[order], colours[order]
        screen_points_left = screen_points_left[order]
        screen_points_base = screen_points_base[order]
        splat_radius = max(
            0,
            min(2, int(self.config["camera"].get("point_cloud_splat_radius_px", 1))),
        )
        for offset_y in range(-splat_radius, splat_radius + 1):
            for offset_x in range(-splat_radius, splat_radius + 1):
                px = np.clip(screen_x + offset_x, 0, render_width - 1)
                py = np.clip(screen_y + offset_y, 30, render_height - 1)
                canvas[py, px] = colours
        pick_left_m = np.full((render_height, render_width, 3), np.nan, dtype=np.float32)
        pick_base_m = np.full((render_height, render_width, 3), np.nan, dtype=np.float32)
        for offset_y in range(-splat_radius, splat_radius + 1):
            for offset_x in range(-splat_radius, splat_radius + 1):
                px = np.clip(screen_x + offset_x, 0, render_width - 1)
                py = np.clip(screen_y + offset_y, 30, render_height - 1)
                pick_left_m[py, px] = screen_points_left
                pick_base_m[py, px] = screen_points_base
        self._point_cloud_pick_left_m = pick_left_m
        self._point_cloud_pick_base_m = pick_base_m

        cv2.rectangle(canvas, (0, 0), (render_width, 29), (12, 13, 16), -1)
        status = (
            f"LEFT + RIGHT IR DENSE CLOUD   {len(screen_x):,} pts   "
            f"valid {valid_percent:.1f}%   {minimum:.2f}-{maximum:.2f} m"
        )
        cv2.putText(canvas, status, (9, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 240), 1, cv2.LINE_AA)
        origin = (38, render_height - 35)
        cv2.arrowedLine(canvas, origin, (origin[0] + 30, origin[1]), (80, 100, 255), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(canvas, origin, (origin[0], origin[1] - 30), (90, 230, 100), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(canvas, "X", (origin[0] + 33, origin[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 100, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Y", (origin[0] - 5, origin[1] - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 230, 100), 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _infrared_information_map(infrared: np.ndarray) -> np.ndarray:
        image = np.asarray(infrared)
        finite = image[np.isfinite(image)]
        if finite.size:
            low, high = np.percentile(finite, (1.0, 99.0))
            if high > low:
                display = np.clip((image.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
            else:
                display = image.astype(np.uint8, copy=False)
        else:
            display = np.zeros(image.shape, dtype=np.uint8)
        return cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    def _infrared_tracking_map(
        self,
        infrared: np.ndarray,
        pixels: np.ndarray,
        observation,
        *,
        right: bool,
    ) -> np.ndarray:
        display = self._infrared_information_map(infrared)
        if not self.engine.keypoint_tracking_enabled:
            cv2.putText(display, "RAW IR - TRACKING OFF", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 195, 205), 1, cv2.LINE_AA)
            return display
        for index, pixel in enumerate(pixels):
            if not np.isfinite(pixel).all():
                continue
            center = tuple(int(round(value)) for value in pixel)
            colour = tuple(int(value) for value in self.config["markers"][index]["display_bgr"])
            cv2.line(display, (max(0, center[0] - 18), center[1]), (min(display.shape[1] - 1, center[0] + 18), center[1]), colour, 1, cv2.LINE_AA)
            cv2.circle(display, center, 6, colour, 2, cv2.LINE_AA)
            disparity = observation.left_pixels[index, 0] - observation.right_pixels[index, 0]
            suffix = f" d={disparity:.2f}px" if right and np.isfinite(disparity) else ""
            cv2.putText(display, f"K{index}{suffix}", (center[0] + 7, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
        return display

    def _direct_sdk_overlay(self, frame, preview_width: int) -> np.ndarray:
        """Resize the newest SDK RGB frame first, then draw the tracking overlay."""
        source = np.asarray(frame.color_bgr)
        source_height, source_width = source.shape[:2]
        preview_width = min(source_width, max(1, int(preview_width)))
        preview_height = int(round(source_height * preview_width / max(source_width, 1)))
        if preview_width == source_width:
            image = source if self.workflow_mode == "capture" else source.copy()
        else:
            image = cv2.resize(
                source, (preview_width, preview_height), interpolation=cv2.INTER_AREA
            )
        scale_x = preview_width / max(source_width, 1)
        scale_y = preview_height / max(source_height, 1)
        sample = self._latest_sample_for_inspection
        if self.engine.keypoint_tracking_enabled and sample is not None:
            for index, pixel in enumerate(sample.keypoints.color_pixels):
                if not np.isfinite(pixel).all():
                    continue
                center = (
                    int(round(float(pixel[0]) * scale_x)),
                    int(round(float(pixel[1]) * scale_y)),
                )
                colour = tuple(int(value) for value in self.config["markers"][index]["display_bgr"])
                cv2.circle(image, center, 6, colour, 2, cv2.LINE_AA)
                cv2.putText(
                    image,
                    f"K{index}",
                    (center[0] + 7, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    colour,
                    1,
                    cv2.LINE_AA,
                )
        if self.workflow_mode != "capture":
            cv2.putText(
                image,
                f"SDK DIRECT  frame {frame.sequence}",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 247),
                2,
                cv2.LINE_AA,
            )
        return image

    def _show_direct_sdk_preview(self, frame, *, update_depth: bool = True) -> bool:
        source_height, source_width = frame.color_bgr.shape[:2]
        preview_width = min(
            source_width,
            max(480, int(self.config.get("ui", {}).get("sdk_preview_width_px", 720))),
        )
        if self.calibration_preview is not None and self.calibration_preview_target == "d435_extrinsic":
            calibration = self.calibration_preview
            if calibration.shape[1] != preview_width:
                calibration = cv2.resize(
                    calibration,
                    (
                        preview_width,
                        int(round(calibration.shape[0] * preview_width / calibration.shape[1])),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            self._show_image(
                calibration,
                self.image_label,
                logical_size=(source_width, source_height),
            )
        elif self._rgb_future is None:
            self._rgb_future = self._rgb_executor.submit(
                lambda current=frame, width=preview_width, logical=(source_width, source_height): (
                    self._direct_sdk_overlay(current, width), logical
                )
            )
        if not update_depth or self._depth_future is not None:
            return False
        self._depth_future = self._depth_executor.submit(
            self._make_direct_sdk_depth_visual, frame
        )
        return True

    def _make_direct_sdk_depth_visual(
        self, frame
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Create the SDK-filtered depth preview away from the Qt UI thread."""
        native_view = (
            str(self.config.get("ui", {}).get("depth_view_space", "native"))
            == "native"
        )
        native_depth = getattr(frame, "native_depth_m", None)
        source_depth = np.asarray(
            native_depth if native_view and native_depth is not None else frame.depth_m,
            dtype=np.float32,
        )
        depth_height, depth_width = source_depth.shape
        sdk_color = getattr(frame, "native_depth_color_bgr", None)
        if native_view and sdk_color is not None:
            preview_width = min(
                depth_width,
                max(
                    480,
                    int(
                        self.config.get("ui", {}).get(
                            "depth_preview_width_px", 848
                        )
                    ),
                ),
            )
            visual = np.asarray(sdk_color)
            if visual.shape[1] != preview_width:
                visual = cv2.resize(
                    visual,
                    (
                        preview_width,
                        max(1, int(round(depth_height * preview_width / depth_width))),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                visual = visual.copy()
            minimum = float(self.config["camera"].get("depth_min_m", 0.18))
            maximum = float(self.config["camera"].get("depth_max_m", 0.60))
            valid = (
                np.isfinite(source_depth)
                & (source_depth >= minimum)
                & (source_depth <= maximum)
            )
            valid_percent = 100.0 * float(np.count_nonzero(valid)) / max(valid.size, 1)
            cv2.rectangle(visual, (0, 0), (visual.shape[1], 25), (12, 13, 16), -1)
            cv2.putText(
                visual,
                f"VALID DEPTH {valid_percent:.1f}%   {minimum:.2f}-{maximum:.2f} m   BLACK = INVALID",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (235, 235, 240) if valid_percent >= 50.0 else (80, 180, 255),
                1,
                cv2.LINE_AA,
            )
            # Only the display texture is reduced. logical_size preserves
            # native-pixel click mapping and all recorded/processed depth stays
            # at 1280x720.
            return visual, (depth_width, depth_height)
        preview_width = min(
            depth_width,
            max(
                480,
                int(
                    self.config.get("ui", {}).get(
                        "depth_preview_width_px",
                        self.config.get("ui", {}).get("sdk_preview_width_px", 720),
                    )
                ),
            ),
        )
        if depth_width != preview_width:
            depth = cv2.resize(
                source_depth,
                (
                    preview_width,
                    int(round(depth_height * preview_width / max(depth_width, 1))),
                ),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            depth = source_depth
        if bool(self.config["camera"].get("depth_postprocess_enabled", False)):
            minimum = float(self.config["camera"].get("depth_min_m", 0.18))
            maximum = float(self.config["camera"].get("depth_max_m", 0.60))
            valid = (
                np.isfinite(depth)
                & (depth >= minimum)
                & (depth <= maximum)
            )
            # The SDK has already performed disparity-domain spatial/temporal
            # filtering. Do not repeat the costly OpenCV neighbourhood passes.
            prepared = (np.where(valid, depth, np.nan), valid, minimum, maximum)
        else:
            prepared = self._filtered_depth_preview(depth, state="depth")
        return (
            self._depth_information_map(depth, prepared),
            (depth_width, depth_height),
        )

    def _update_direct_pointcloud(self, frame, now_ns: int) -> None:
        view_3d = str(self.d435_display_mode.currentData()) == "3d"
        record_3d = (
            self.viewer_recorder.active
            and "pointcloud" in self.viewer_recorder.selected
        )
        cloud_needed = view_3d or record_3d
        if cloud_needed and not bool(
            (frame.metadata or {}).get("depth_aligned_to_color", True)
        ):
            return
        if self._cloud_future is not None and self._cloud_future.done():
            try:
                self._cached_point_cloud_visual = self._cloud_future.result()
                self._cloud_revision += 1
                if record_3d:
                    self.viewer_recorder.append_pointcloud(
                        self._cached_point_cloud_visual,
                        self._cloud_future_capture_ns
                        if self._cloud_future_capture_ns is not None
                        else frame.capture_host_ns,
                    )
            except Exception as cloud_error:
                self.device_status.setText(f"点云渲染异常: {cloud_error}")
            self._cloud_future = None
            self._cloud_future_capture_ns = None
        if (
            view_3d
            and self._cached_point_cloud_visual is not None
            and self._displayed_cloud_revision != self._cloud_revision
        ):
            self._show_image(
                self._cached_point_cloud_visual, self.stereo_cloud_image_label
            )
            self._displayed_cloud_revision = self._cloud_revision
        if not cloud_needed or self._cloud_future is not None:
            return
        cloud_fps = max(
            float(self.config.get("ui", {}).get("point_cloud_fps", 15.0)), 1.0
        )
        if now_ns - self._last_point_cloud_ns < int(1e9 / cloud_fps):
            return
        self._cloud_future = self._cloud_executor.submit(
            self._point_cloud_information_map,
            frame,
            None,
            True,
        )
        self._cloud_future_capture_ns = int(frame.capture_host_ns)
        self._last_point_cloud_ns = now_ns

    def _poll(self) -> None:
        try:
            self._update_camera_processing_demand()
            source_has_direct_preview = callable(getattr(self.camera_source, "peek_latest", None))
            if (
                self.engine.running
                and not self.capture_worker.running
                and (self._processing_required() or not source_has_direct_preview)
            ):
                self._start_capture_processing()
            elif (
                self.capture_worker.running
                and not self._processing_required()
                and source_has_direct_preview
            ):
                self._stop_capture_processing()
            now_ns = time.perf_counter_ns()
            self._update_axis_live_values(now_ns)
            self._update_em_status(now_ns)
            self._poll_endoscope_camera()
            ui_config = self.config.get("ui", {})
            preview_period_ns = int(
                1e9 / max(float(ui_config.get("preview_fps", 30.0)), 1.0)
            )
            peek_latest = getattr(self.camera_source, "peek_latest", None)
            direct_frame = peek_latest() if callable(peek_latest) else None
            direct_sdk_active = direct_frame is not None
            if self._rgb_future is not None and self._rgb_future.done():
                try:
                    rgb_visual, rgb_logical_size = self._rgb_future.result()
                    self._show_image(
                        rgb_visual,
                        self.image_label,
                        logical_size=rgb_logical_size,
                    )
                except Exception as rgb_error:
                    self.device_status.setText(f"RGB图渲染异常: {rgb_error}")
                self._rgb_future = None
            if self._depth_future is not None and self._depth_future.done():
                try:
                    depth_visual, depth_logical_size = self._depth_future.result()
                    self._show_image(
                        depth_visual,
                        self.depth_image_label,
                        logical_size=depth_logical_size,
                    )
                except Exception as depth_error:
                    self.device_status.setText(f"深度图渲染异常: {depth_error}")
                self._depth_future = None
            if (
                direct_sdk_active
                and direct_frame.sequence != self._last_direct_sdk_sequence
            ):
                metadata = direct_frame.metadata or {}
                frame_counter = int(
                    metadata.get("depth_frame_counter", direct_frame.sequence)
                )
                self._sdk_frame_samples.append(
                    (frame_counter, float(direct_frame.depth_timestamp_ms))
                )
                if len(self._sdk_frame_samples) >= 2:
                    first_counter, first_timestamp = self._sdk_frame_samples[0]
                    last_counter, last_timestamp = self._sdk_frame_samples[-1]
                    counter_delta = last_counter - first_counter
                    timestamp_delta_ms = last_timestamp - first_timestamp
                    if counter_delta > 0 and timestamp_delta_ms > 0.0:
                        sdk_fps = 1000.0 * counter_delta / timestamp_delta_ms
                        tone = "green" if sdk_fps >= 29.0 else "orange"
                        self.d435_view_pill.set_status(
                            f"Depth {sdk_fps:.1f} FPS", tone
                        )
                depth_period_ns = int(
                    1e9 / max(float(ui_config.get("depth_preview_fps", 15.0)), 1.0)
                )
                update_depth = (
                    self.workflow_mode == "capture"
                    or now_ns - self._last_depth_preview_ns >= depth_period_ns
                )
                show_2d = str(self.d435_display_mode.currentData()) == "2d"
                depth_scheduled = False
                if show_2d:
                    depth_scheduled = self._show_direct_sdk_preview(
                        direct_frame, update_depth=update_depth
                    )
                if depth_scheduled:
                    self._last_depth_preview_ns = now_ns
                self._last_direct_sdk_sequence = int(direct_frame.sequence)
                self._last_preview_ns = now_ns
            if direct_sdk_active:
                self._update_direct_pointcloud(direct_frame, now_ns)
            version, packet, worker_error = self.capture_worker.latest_after(
                self._capture_packet_version
            )
            if worker_error is not None:
                self.device_status.setText(f"后台采集异常: {worker_error}")
            if packet is None:
                return
            self._capture_packet_version = version
            sample = packet.sample
            self._latest_sample_for_inspection = sample
            self._ir_xyz_cache = {
                key: value for key, value in self._ir_xyz_cache.items() if key[0] == sample.frame.sequence
            }
            telemetry_period_ns = int(1e9 / max(float(ui_config.get("telemetry_fps", 10.0)), 1.0))
            preview_due = (
                direct_sdk_active
                or now_ns - self._last_preview_ns >= preview_period_ns
                or self.calibration_preview is not None
            )
            telemetry_due = now_ns - self._last_telemetry_ns >= telemetry_period_ns
            if telemetry_due:
                self._last_telemetry_ns = now_ns
                self._update_keypoint_table(sample)
                self._update_camera_parameter_telemetry(sample.frame)
            if not preview_due:
                return
            if not direct_sdk_active:
                self._last_preview_ns = now_ns
            d435_image = packet.d435_overlay
            side_image = packet.side_overlay
            if self.calibration_preview is not None:
                if self.calibration_preview_target == "d435_extrinsic":
                    d435_image = self.calibration_preview
                else:
                    side_image = self.calibration_preview
            self.calibration_preview = None
            self.calibration_preview_target = None
            if not direct_sdk_active:
                self._show_image(d435_image, self.image_label)
            if not direct_sdk_active:
                prepared_depth = self._filtered_depth_preview(sample.frame.depth_m)
                depth_visual = self._depth_information_map(sample.frame.depth_m, prepared_depth)
                self._show_image(depth_visual, self.depth_image_label)
            else:
                # RealSenseSource already applies the SDK spatial/temporal
                # filters.  Do not repeat a full-frame median pass in the UI.
                prepared_depth = None
            if (
                not direct_sdk_active
                and self._cloud_future is not None
                and self._cloud_future.done()
            ):
                try:
                    self._cached_point_cloud_visual = self._cloud_future.result()
                    self._cloud_revision += 1
                except Exception as cloud_error:
                    self.device_status.setText(f"点云渲染异常: {cloud_error}")
                self._cloud_future = None
            cloud_fps = max(float(ui_config.get("point_cloud_fps", 6.0)), 1.0)
            cloud_period_ns = int(1e9 / cloud_fps)
            if (
                not direct_sdk_active
                and
                self._cloud_future is None
                and now_ns - self._last_point_cloud_ns >= cloud_period_ns
            ):
                self._cloud_future = self._cloud_executor.submit(
                    self._point_cloud_information_map,
                    sample.frame,
                    prepared_depth,
                    direct_sdk_active,
                )
                self._last_point_cloud_ns = now_ns
            if (
                not direct_sdk_active
                and
                self._cached_point_cloud_visual is not None
                and self._displayed_cloud_revision != self._cloud_revision
            ):
                self._show_image(self._cached_point_cloud_visual, self.stereo_cloud_image_label)
                self._displayed_cloud_revision = self._cloud_revision
            if self.engine.fusion_mode == DUAL_VIEW_MODE:
                if side_image is not None:
                    self._show_image(side_image, self.side_image_label)
                elif packet.side_frame is not None:
                    self._show_image(packet.side_frame.image_bgr, self.side_image_label)
                if packet.side_frame is not None and self.side_view_pill.text() != "在线":
                    self.side_view_pill.set_status("在线", "green")
                    if not isinstance(self.camera_source, SyntheticCameraSource):
                        selected_name = self.side_camera_combo.currentText().split("  ·  Index", 1)[0]
                        self.side_status.setText(
                            f"侧相机在线：{selected_name}（Index {self.config['side_camera']['index']}）"
                        )
            self.shape_view.update_shapes(sample.keypoints.base_m, sample.simulation_points_base_m)
            if not telemetry_due:
                return
            measured = int(np.count_nonzero(sample.axes.measured_valid))
            stream_skew = max(sample.frame.color_timestamp_ms, sample.frame.depth_timestamp_ms, sample.frame.infrared_timestamp_ms) - min(sample.frame.color_timestamp_ms, sample.frame.depth_timestamp_ms, sample.frame.infrared_timestamp_ms)
            finite_side_errors = sample.keypoints.side_reprojection_error_px[np.isfinite(sample.keypoints.side_reprojection_error_px)]
            side_reprojection = float(np.median(finite_side_errors)) if len(finite_side_errors) else float("nan")
            mujoco_metric = f"{sample.rmse_mm:.3f} mm" if self.engine.bridge is not None else "未启用"
            self.metrics.setPlainText(
                f"Frame: {sample.frame.sequence}\n"
                f"测量模式: {sample.fusion_mode}\n"
                f"关键点捕捉: {'开启' if self.engine.keypoint_tracking_enabled else '关闭'}\n"
                f"D435处理ROI: {np.array2string(sample.d435_tracking_roi, precision=3)}\n"
                f"侧相机处理ROI: {np.array2string(sample.side_tracking_roi, precision=3)}\n"
                f"有效关键点: {int(np.count_nonzero(sample.keypoints.valid))}/7\n"
                f"预测关键点: {int(np.count_nonzero(sample.keypoints.predicted))}/7\n"
                f"MPOS有效轴: {measured}/7\n"
                f"流时间偏差: {stream_skew:.3f} ms\n"
                f"侧相机时间差: {sample.keypoints.side_time_offset_ms:.3f} ms\n"
                f"侧视重投影中位数: {side_reprojection:.3f} px\n"
                f"MuJoCo RMSE: {mujoco_metric}\n"
                f"处理耗时: {packet.processing_ms:.1f} ms · {packet.processing_fps:.1f} FPS\n"
                f"计算后端: {self.compute_backend_info}\n"
                f"轴状态: {sample.axes.status}\n"
                f"源: {', '.join(sample.keypoints.source)}"
            )
        except Exception as exc:
            self.device_status.setText(f"采集异常: {exc}")

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.camera_discovery_timer.stop()
        if self.identify_process is not None and self.identify_process.state() != QtCore.QProcess.NotRunning:
            self.identify_process.terminate()
            if not self.identify_process.waitForFinished(1500):
                self.identify_process.kill()
        try:
            self._stop_capture_processing()
        except Exception:
            pass
        try:
            self.endoscope_recorder.stop()
        except Exception:
            pass
        try:
            self.em_recorder.stop()
        except Exception:
            pass
        try:
            self.viewer_recorder.stop()
        except Exception:
            pass
        try:
            self._cloud_executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        try:
            self._depth_executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        try:
            self._rgb_executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        try:
            self._stop_endoscope_camera()
        except Exception:
            pass
        try:
            self.engine.stop()
        except Exception:
            pass
        if self._owns_axis_source and hasattr(self.engine.axis_source, "close"):
            try:
                self.engine.axis_source.close()
            except Exception:
                pass
        try:
            self.em_source.close()
        except Exception:
            pass
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bag", type=Path, help="Replay a RealSense bag instead of a live D435")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic camera and axis data")
    parser.add_argument("--trio", type=str, help="Connect a read-only Trio source, e.g. 192.168.0.250")
    parser.add_argument("--no-mujoco", action="store_true")
    parser.add_argument(
        "--mode", choices=(D435_INTERNAL_MODE, DUAL_VIEW_MODE),
        help="3D measurement mode: d435_internal or dual_view",
    )
    parser.add_argument("--side-camera-index", type=int, help="OpenCV camera index for dual_view mode")
    parser.add_argument("--side-calibration", type=Path, help="Combined side-camera intrinsics/extrinsics JSON")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.mode:
        config["fusion"]["mode"] = args.mode
    if args.side_camera_index is not None:
        config["side_camera"]["index"] = int(args.side_camera_index)
    if args.side_calibration:
        result = load_calibration(args.side_calibration)
        intrinsics = result.get("intrinsics", config["side_camera"]["intrinsics"])
        intrinsics_ready = bool(result.get("intrinsics_ready", "intrinsics" in result))
        config["side_camera"].update({
            "transform_base_from_camera": result["transform_base_from_camera"],
            "calibration_ready": intrinsics_ready,
            "intrinsics": intrinsics,
            "intrinsics_ready": intrinsics_ready,
        })
    if args.synthetic:
        camera_source = SyntheticCameraSource(config)
        axis_source = SyntheticAxisSource(config["axes"])
        side_camera_source = SyntheticSideRgbSource(config, camera_source) if config["fusion"]["mode"] == DUAL_VIEW_MODE else None
    else:
        camera_source = RealSenseSource(config["camera"], args.bag, repeat=bool(args.bag))
        axis_source = NullAxisSource(config["axes"])
        side_camera_source = OpenCvRgbSource(config["side_camera"]) if config["fusion"]["mode"] == DUAL_VIEW_MODE else None
    if args.trio:
        axis_source = TrioAxisSource(config["axes"], args.trio)
        axis_source.connect()
    bridge = None
    if args.no_mujoco:
        config["mujoco"]["enabled"] = False
    elif bool(config["mujoco"].get("enabled", False)):
        try:
            bridge = MujocoAlignmentBridge(config["mujoco"])
        except Exception as exc:
            config["mujoco"]["enabled"] = False
            print(f"[D435] MuJoCo bridge disabled: {exc}")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = D435CaptureWindow(
        config, camera_source, axis_source, bridge,
        side_camera_source=side_camera_source,
    )
    window.show()
    application.exec_()


if __name__ == "__main__":
    main()
