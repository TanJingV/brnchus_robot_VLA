"""Reusable, business-agnostic workstation widgets."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..curve_model import fit_continuum_shape, tube_mesh

try:
    import pyqtgraph.opengl as gl
except Exception:  # pragma: no cover - 2-D UI remains usable
    gl = None


MARKER_HEX = ("#FF453A", "#FF9F0A", "#FFD60A", "#30D158", "#64D2FF", "#0A84FF", "#BF5AF2")


def card(title: str, subtitle: str = "", hero: bool = False) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
    frame = QtWidgets.QFrame()
    frame.setObjectName("heroCard" if hero else "card")
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(16, 15, 16, 16)
    layout.setSpacing(10)
    header = QtWidgets.QHBoxLayout()
    label = QtWidgets.QLabel(title)
    label.setObjectName("cardTitle")
    header.addWidget(label)
    if subtitle:
        detail = QtWidgets.QLabel(subtitle)
        detail.setObjectName("muted")
        header.addStretch(1)
        header.addWidget(detail)
    layout.addLayout(header)
    return frame, layout


class NoWheelSpinBox(QtWidgets.QSpinBox):
    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        event.ignore()


class VideoCanvas(QtWidgets.QWidget):
    """Aspect-fit video display with zoom, pan and seven-point interaction."""

    keypointsCompleted = QtCore.pyqtSignal(object)
    pointCountChanged = QtCore.pyqtSignal(int)

    def __init__(self, empty_text: str = "尚无画面") -> None:
        super().__init__()
        self.setMinimumSize(420, 260)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.setStyleSheet("background:#101114;border-radius:12px;")
        self.empty_text = empty_text
        self._source: QtGui.QImage | None = None
        self._points_normalized: list[tuple[float, float]] = []
        self._point_mode = False
        self._zoom = 1.0
        self._pan = QtCore.QPointF(0.0, 0.0)
        self._drag_origin: QtCore.QPointF | None = None
        self._drag_pan_origin = QtCore.QPointF(0.0, 0.0)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self.setToolTip("滚轮缩放 · 左键拖动画面 · 中键随时拖动 · 双击复位")

    @property
    def source_size(self) -> tuple[int, int]:
        if self._source is None:
            return 0, 0
        return self._source.width(), self._source.height()

    def set_bgr(self, image_bgr: np.ndarray | None) -> None:
        if image_bgr is None or not np.asarray(image_bgr).size:
            self._source = None
            self.reset_view()
            self.update()
            return
        rgb = cv2.cvtColor(np.ascontiguousarray(image_bgr), cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        previous_size = self.source_size
        self._source = QtGui.QImage(
            rgb.data, width, height, int(rgb.strides[0]), QtGui.QImage.Format_RGB888
        ).copy()
        if previous_size != (0, 0) and previous_size != (width, height):
            self.reset_view()
        self.update()

    @property
    def zoom_factor(self) -> float:
        return float(self._zoom)

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QtCore.QPointF(0.0, 0.0)
        self._drag_origin = None
        self.update()

    def _fit_scale(self) -> float:
        if self._source is None:
            return 1.0
        return min(
            self.width() / max(self._source.width(), 1),
            self.height() / max(self._source.height(), 1),
        )

    def _clamp_pan(self) -> None:
        if self._source is None:
            self._pan = QtCore.QPointF(0.0, 0.0)
            return
        scale = self._fit_scale() * self._zoom
        image_width = self._source.width() * scale
        image_height = self._source.height() * scale
        limit_x = max(0.0, 0.5 * (image_width - self.width()))
        limit_y = max(0.0, 0.5 * (image_height - self.height()))
        self._pan = QtCore.QPointF(
            float(np.clip(self._pan.x(), -limit_x, limit_x)),
            float(np.clip(self._pan.y(), -limit_y, limit_y)),
        )

    def begin_seven_points(self) -> None:
        if self._source is None:
            return
        self._points_normalized = []
        self._point_mode = True
        self.setCursor(QtCore.Qt.CrossCursor)
        self.pointCountChanged.emit(0)
        self.update()

    def cancel_seven_points(self) -> None:
        self._points_normalized = []
        self._point_mode = False
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self.pointCountChanged.emit(0)
        self.update()

    def set_keypoints(self, pixels: np.ndarray | None) -> None:
        if pixels is None or self._source is None:
            self._points_normalized = []
        else:
            points = np.asarray(pixels, dtype=float).reshape(-1, 2)
            self._points_normalized = [
                (float(point[0] / self._source.width()), float(point[1] / self._source.height()))
                for point in points
            ]
        self.update()

    def native_keypoints(self) -> np.ndarray | None:
        if self._source is None or len(self._points_normalized) != 7:
            return None
        return np.asarray([
            [x * self._source.width(), y * self._source.height()]
            for x, y in self._points_normalized
        ], dtype=float)

    def _image_rect(self) -> QtCore.QRectF:
        if self._source is None:
            return QtCore.QRectF()
        scale = self._fit_scale() * self._zoom
        width = self._source.width() * scale
        height = self._source.height() * scale
        return QtCore.QRectF(
            (self.width() - width) / 2.0 + self._pan.x(),
            (self.height() - height) / 2.0 + self._pan.y(),
            width,
            height,
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#101114"))
        if self._source is None:
            painter.setPen(QtGui.QColor("#77777E"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, self.empty_text)
            return
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        rect = self._image_rect()
        painter.drawImage(rect, self._source)
        if len(self._points_normalized) > 1:
            path = QtGui.QPainterPath()
            first = self._points_normalized[0]
            path.moveTo(rect.left() + first[0] * rect.width(), rect.top() + first[1] * rect.height())
            for point in self._points_normalized[1:]:
                path.lineTo(rect.left() + point[0] * rect.width(), rect.top() + point[1] * rect.height())
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 190), 2))
            painter.drawPath(path)
        for index, point in enumerate(self._points_normalized):
            x = rect.left() + point[0] * rect.width()
            y = rect.top() + point[1] * rect.height()
            painter.setPen(QtGui.QPen(QtGui.QColor("#111216"), 4))
            painter.setBrush(QtGui.QColor(MARKER_HEX[index]))
            painter.drawEllipse(QtCore.QPointF(x, y), 7, 7)
            painter.setPen(QtGui.QColor("#FFFFFF"))
            painter.drawText(QtCore.QPointF(x + 10, y - 8), f"K{index}")
        if self._zoom > 1.001:
            badge = QtCore.QRectF(self.width() - 92, 12, 76, 28)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(18, 20, 24, 210))
            painter.drawRoundedRect(badge, 8, 8)
            painter.setPen(QtGui.QColor("#F5F5F7"))
            painter.drawText(badge, QtCore.Qt.AlignCenter, f"{self._zoom:.1f}×")

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._source is None or event.angleDelta().y() == 0:
            event.ignore()
            return
        position = event.posF()
        old_rect = self._image_rect()
        if old_rect.width() <= 0 or old_rect.height() <= 0:
            return
        source_x = (position.x() - old_rect.left()) / old_rect.width()
        source_y = (position.y() - old_rect.top()) / old_rect.height()
        steps = event.angleDelta().y() / 120.0
        new_zoom = float(np.clip(self._zoom * (1.18 ** steps), 1.0, 8.0))
        if abs(new_zoom - self._zoom) < 1e-6:
            event.accept()
            return
        self._zoom = new_zoom
        scale = self._fit_scale() * self._zoom
        new_width = self._source.width() * scale
        new_height = self._source.height() * scale
        new_left = position.x() - source_x * new_width
        new_top = position.y() - source_y * new_height
        self._pan = QtCore.QPointF(
            new_left + 0.5 * new_width - 0.5 * self.width(),
            new_top + 0.5 * new_height - 0.5 * self.height(),
        )
        self._clamp_pan()
        self.update()
        event.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.button() == QtCore.Qt.MiddleButton
            or (event.button() == QtCore.Qt.LeftButton and not self._point_mode)
        ):
            self._drag_origin = event.localPos()
            self._drag_pan_origin = QtCore.QPointF(self._pan)
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        if not self._point_mode:
            return
        if event.button() == QtCore.Qt.RightButton:
            if self._points_normalized:
                self._points_normalized.pop()
                self.pointCountChanged.emit(len(self._points_normalized))
                self.update()
            return
        rect = self._image_rect()
        if event.button() != QtCore.Qt.LeftButton or not rect.contains(event.pos()):
            return
        x = float(np.clip((event.x() - rect.left()) / rect.width(), 0.0, 1.0))
        y = float(np.clip((event.y() - rect.top()) / rect.height(), 0.0, 1.0))
        if len(self._points_normalized) < 7:
            self._points_normalized.append((x, y))
        count = len(self._points_normalized)
        self.pointCountChanged.emit(count)
        if count == 7:
            self._point_mode = False
            self.setCursor(QtCore.Qt.OpenHandCursor)
            self.keypointsCompleted.emit(self.native_keypoints())
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_origin is None:
            return
        delta = event.localPos() - self._drag_origin
        self._pan = self._drag_pan_origin + delta
        self._clamp_pan()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_origin is None:
            return
        self._drag_origin = None
        self.setCursor(
            QtCore.Qt.CrossCursor if self._point_mode else QtCore.Qt.OpenHandCursor
        )
        event.accept()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and not self._point_mode:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self._clamp_pan()
        super().resizeEvent(event)


class Shape3DView(QtWidgets.QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("card")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("连续体三维形状")
        title.setObjectName("cardTitle")
        self.metrics = QtWidgets.QLabel("等待重建结果")
        self.metrics.setObjectName("muted")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.metrics)
        layout.addLayout(header)
        if gl is None:
            self.view = None
            fallback = QtWidgets.QLabel("当前环境缺少 PyOpenGL")
            fallback.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(fallback, 1)
            return
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((16, 17, 20))
        self.view.opts.update(distance=0.085, elevation=24, azimuth=-48)
        grid = gl.GLGridItem()
        grid.setSize(0.10, 0.10)
        grid.setSpacing(0.01, 0.01)
        grid.setColor((100, 103, 112, 80))
        self.view.addItem(grid)
        self.centerline = gl.GLLinePlotItem(pos=np.zeros((0, 3)), color=(0.25, 0.78, 1, 1), width=3, antialias=True)
        self.tube = gl.GLMeshItem(vertexes=np.empty((0, 3)), faces=np.empty((0, 3), np.int32), smooth=True, shader="shaded")
        self.measured = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), color=(0.2, 0.95, 0.5, 1), size=10)
        self.predicted = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), color=(1, 0.6, 0.12, 1), size=9)
        for item in (self.tube, self.centerline, self.measured, self.predicted):
            self.view.addItem(item)
        layout.addWidget(self.view, 1)

    def set_result(self, result) -> None:
        if self.view is None or result is None:
            return
        points = np.asarray(result.smoothed_camera_m, dtype=float).reshape(7, 3)
        valid = np.isfinite(points).all(axis=1)
        if np.count_nonzero(valid) < 2:
            self.metrics.setText("有效三维点不足")
            return
        origin = points[0] if valid[0] else np.nanmean(points[valid], axis=0)
        local = points - origin
        shape = fit_continuum_shape(local, samples=85)
        vertices, faces, colors = tube_mesh(shape.centerline_m)
        if len(vertices):
            self.tube.setMeshData(meshdata=gl.MeshData(vertexes=vertices, faces=faces, vertexColors=colors), smooth=True, shader="shaded")
        self.centerline.setData(pos=shape.centerline_m)
        measured = valid & np.asarray(result.measured_valid, bool)
        self.measured.setData(pos=local[measured])
        self.predicted.setData(pos=local[valid & ~measured])
        self.metrics.setText(
            f"实测 {np.count_nonzero(measured)}/7 · 弧长 {shape.total_length_m * 1000:.1f} mm"
        )


class KeypointTable(QtWidgets.QTableWidget):
    def __init__(self) -> None:
        super().__init__(7, 6)
        self.setHorizontalHeaderLabels(("点", "状态", "X mm", "Y mm", "Z mm", "置信度"))
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for row in range(7):
            self.setItem(row, 0, QtWidgets.QTableWidgetItem(f"K{row}"))

    def set_result(self, result) -> None:
        if result is None:
            return
        for index in range(7):
            point = np.asarray(result.smoothed_camera_m[index]) * 1000.0
            state = "预测" if result.predicted[index] else ("实测" if result.measured_valid[index] else "无效")
            values = (
                f"K{index}", state,
                f"{point[0]:+.2f}" if np.isfinite(point[0]) else "—",
                f"{point[1]:+.2f}" if np.isfinite(point[1]) else "—",
                f"{point[2]:+.2f}" if np.isfinite(point[2]) else "—",
                f"{result.confidence[index]:.2f}",
            )
            for column, value in enumerate(values):
                self.setItem(index, column, QtWidgets.QTableWidgetItem(value))
