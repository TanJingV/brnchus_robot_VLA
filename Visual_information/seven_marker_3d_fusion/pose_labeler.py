"""Small product-facing annotation tool for the custom seven-keypoint model."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


VISUAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VISUAL_ROOT.parent
DATASET_ROOT = VISUAL_ROOT / "seven_marker_yolo_pose"
COLOURS = [
    (235, 30, 30), (255, 140, 0), (235, 210, 0), (40, 210, 40),
    (20, 190, 225), (25, 90, 235), (210, 30, 210),
]


class ClickImage(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(900, 500)
        self.setStyleSheet("background:#111318;border-radius:12px;")
        self._image_size = (1, 1)
        self._display_rect = QtCore.QRectF()

    def set_frame(self, bgr: np.ndarray) -> None:
        height, width = bgr.shape[:2]
        self._image_size = (width, height)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(rgb.data, width, height, 3 * width, QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(image).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        x = (self.width() - pixmap.width()) / 2.0
        y = (self.height() - pixmap.height()) / 2.0
        self._display_rect = QtCore.QRectF(x, y, pixmap.width(), pixmap.height())
        self.setPixmap(pixmap)

    def mousePressEvent(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton or not self._display_rect.contains(event.pos()):
            return
        width, height = self._image_size
        x = (event.x() - self._display_rect.left()) / self._display_rect.width() * width
        y = (event.y() - self._display_rect.top()) / self._display_rect.height() * height
        self.clicked.emit(float(x), float(y))


class PoseLabelWindow(QtWidgets.QMainWindow):
    """Click K0..K6 in semantic order and store native YOLO Pose labels."""

    def __init__(self, video: str = "") -> None:
        super().__init__()
        self.setWindowTitle("TDCR YOLO Pose — K0–K6 标注")
        self.resize(1450, 900)
        self.capture: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.frame = np.empty((0, 0, 3), np.uint8)
        self.frame_index = 0
        self.frame_count = 0
        self.points: list[tuple[float, float]] = []
        self._build_ui()
        if video:
            self.open_video(Path(video))

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        bar = QtWidgets.QHBoxLayout()
        open_button = QtWidgets.QPushButton("打开RGB视频")
        open_button.clicked.connect(self.choose_video)
        self.path_label = QtWidgets.QLabel("未打开视频")
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        bar.addWidget(open_button); bar.addWidget(self.path_label, 1)
        root.addLayout(bar)
        self.image = ClickImage(); self.image.clicked.connect(self.add_point)
        root.addWidget(self.image, 1)
        controls = QtWidgets.QHBoxLayout()
        self.previous_button = QtWidgets.QPushButton("上一帧")
        self.next_button = QtWidgets.QPushButton("下一帧")
        self.skip_button = QtWidgets.QPushButton("跳过未标注")
        self.undo_button = QtWidgets.QPushButton("撤销点")
        self.clear_button = QtWidgets.QPushButton("清空本帧")
        self.save_button = QtWidgets.QPushButton("保存标注并前进")
        self.save_button.setStyleSheet("background:#007AFF;color:white;padding:8px 16px;border-radius:8px;")
        self.previous_button.clicked.connect(lambda: self.step(-self.stride.value()))
        self.next_button.clicked.connect(lambda: self.step(self.stride.value()))
        self.skip_button.clicked.connect(lambda: self.step(self.stride.value()))
        self.undo_button.clicked.connect(self.undo)
        self.clear_button.clicked.connect(self.clear_points)
        self.save_button.clicked.connect(self.save_and_next)
        self.stride = QtWidgets.QSpinBox(); self.stride.setRange(1, 300); self.stride.setValue(30)
        for widget in (self.previous_button, self.next_button, self.skip_button, self.undo_button, self.clear_button):
            controls.addWidget(widget)
        controls.addWidget(QtWidgets.QLabel("帧间隔")); controls.addWidget(self.stride)
        controls.addStretch(1); controls.addWidget(self.save_button)
        root.addLayout(controls)
        self.timeline = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.timeline.valueChanged.connect(self.seek)
        root.addWidget(self.timeline)
        self.status = QtWidgets.QLabel(
            "操作：从连接器一侧开始，依次点击 K0、K1、…、K6。必须点在每条色带的中心。"
        )
        self.status.setStyleSheet("font-size:15px;padding:8px;")
        root.addWidget(self.status)

    def choose_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择D435 RGB视频", str(PROJECT_ROOT / "d435_sessions"), "Video (*.mp4 *.avi *.mkv)"
        )
        if path:
            self.open_video(Path(path))

    def open_video(self, path: Path) -> None:
        if self.capture is not None:
            self.capture.release()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            QtWidgets.QMessageBox.critical(self, "视频错误", f"无法打开：{path}")
            return
        self.capture = capture; self.video_path = path.resolve()
        self.frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.timeline.blockSignals(True); self.timeline.setRange(0, self.frame_count - 1); self.timeline.setValue(0); self.timeline.blockSignals(False)
        self.path_label.setText(str(self.video_path)); self.seek(0)

    def seek(self, index: int) -> None:
        if self.capture is None:
            return
        self.frame_index = int(np.clip(index, 0, self.frame_count - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        ok, frame = self.capture.read()
        if not ok:
            return
        self.frame = frame; self.points = []
        self.timeline.blockSignals(True); self.timeline.setValue(self.frame_index); self.timeline.blockSignals(False)
        self.redraw()

    def step(self, delta: int) -> None:
        self.seek(self.frame_index + int(delta))

    def add_point(self, x: float, y: float) -> None:
        if len(self.points) >= 7:
            return
        self.points.append((x, y)); self.redraw()

    def undo(self) -> None:
        if self.points:
            self.points.pop(); self.redraw()

    def clear_points(self) -> None:
        self.points = []; self.redraw()

    def redraw(self) -> None:
        if not self.frame.size:
            return
        canvas = self.frame.copy()
        if len(self.points) > 1:
            cv2.polylines(canvas, [np.rint(self.points).astype(np.int32)], False, (245, 245, 245), 2, cv2.LINE_AA)
        for index, (x, y) in enumerate(self.points):
            colour = COLOURS[index][::-1]
            cv2.circle(canvas, (round(x), round(y)), 8, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.circle(canvas, (round(x), round(y)), 5, colour, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"K{index}", (round(x) + 9, round(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, .55, colour, 2, cv2.LINE_AA)
        self.image.set_frame(canvas)
        next_point = f"K{len(self.points)}" if len(self.points) < 7 else "完成，可保存"
        self.status.setText(f"帧 {self.frame_index + 1}/{self.frame_count}　已标 {len(self.points)}/7　下一点：{next_point}")

    def save_and_next(self) -> None:
        if len(self.points) != 7 or self.video_path is None:
            QtWidgets.QMessageBox.warning(self, "标注不完整", "必须按K0到K6准确点击全部7个点。")
            return
        height, width = self.frame.shape[:2]
        points = np.asarray(self.points, float)
        span = np.ptp(points, axis=0)
        padding = max(12.0, 0.18 * float(np.linalg.norm(span)))
        x0, y0 = np.maximum(np.min(points, axis=0) - padding, 0.0)
        x1, y1 = np.minimum(np.max(points, axis=0) + padding, [width - 1.0, height - 1.0])
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        bw, bh = (x1 - x0) / width, (y1 - y0) / height
        keypoints = []
        for x, y in points:
            keypoints.extend((x / width, y / height, 2.0))
        stem = f"{self.video_path.parent.name}_{self.video_path.stem}_{self.frame_index:06d}"
        # Split whole sessions, never adjacent frames.  Otherwise validation
        # would contain near-duplicates of training images and report a false
        # high accuracy.  Across the normally recorded nine sessions the
        # stable parent-name hash assigns roughly one fifth to validation.
        stable_session_hash = sum(self.video_path.parent.name.encode("utf-8"))
        split = "val" if stable_session_hash % 5 == 0 else "train"
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_dir / f"{stem}.jpg"), self.frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
        values = [0, cx, cy, bw, bh, *keypoints]
        (label_dir / f"{stem}.txt").write_text(
            " ".join(str(int(v)) if index == 0 else f"{float(v):.8f}" for index, v in enumerate(values)) + "\n",
            encoding="utf-8",
        )
        self.step(self.stride.value())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event); self.redraw()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--video", default="")
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = PoseLabelWindow(args.video); window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
