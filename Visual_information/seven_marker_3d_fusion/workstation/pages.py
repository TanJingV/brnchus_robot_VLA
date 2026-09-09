"""Focused pages for the product workstation.

Each page owns presentation only.  Session IO, state transitions and worker
control stay in ``window.py``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets

from Visual_information.d435_tdcr_capture.config import VISUAL_ROOT

from ..session import SessionDescriptor, SessionKind
from .widgets import (
    KeypointTable,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    Shape3DView,
    VideoCanvas,
    card,
)


class DataPage(QtWidgets.QWidget):
    refreshRequested = QtCore.pyqtSignal()
    browseRequested = QtCore.pyqtSignal()
    sessionSelected = QtCore.pyqtSignal(object)
    continueRequested = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)
        hero, layout = card("选择采集数据", "步骤 1 / 4", hero=True)
        description = QtWidgets.QLabel(
            "选择同时包含 RGB 与 Depth 的会话。识别任务只读取原始采集数据，"
            "不会覆盖采集文件。"
        )
        description.setObjectName("muted")
        description.setWordWrap(True)
        layout.addWidget(description)
        row = QtWidgets.QHBoxLayout()
        self.session_combo = QtWidgets.QComboBox()
        self.session_combo.setMinimumWidth(420)
        self.session_combo.currentIndexChanged.connect(self._selection_changed)
        refresh = QtWidgets.QPushButton("重新扫描")
        refresh.clicked.connect(self.refreshRequested)
        browse = QtWidgets.QPushButton("选择文件夹…")
        browse.clicked.connect(self.browseRequested)
        row.addWidget(self.session_combo, 1)
        row.addWidget(refresh)
        row.addWidget(browse)
        layout.addLayout(row)
        root.addWidget(hero)

        middle = QtWidgets.QHBoxLayout()
        summary, summary_layout = card("会话质量")
        self.quality_badge = QtWidgets.QLabel("未选择")
        self.quality_badge.setObjectName("warningBadge")
        self.quality_detail = QtWidgets.QLabel("请选择一个采集会话。")
        self.quality_detail.setObjectName("muted")
        self.quality_detail.setWordWrap(True)
        self.path_label = QtWidgets.QLabel("—")
        self.path_label.setObjectName("micro")
        self.path_label.setWordWrap(True)
        self.path_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        summary_layout.addWidget(self.quality_badge, 0, QtCore.Qt.AlignLeft)
        summary_layout.addWidget(self.quality_detail)
        summary_layout.addWidget(self.path_label)
        middle.addWidget(summary, 1)

        streams, stream_layout = card("数据通道")
        self.stream_labels: dict[str, QtWidgets.QLabel] = {}
        grid = QtWidgets.QGridLayout()
        for index, (key, title) in enumerate((
            ("rgb", "RGB"), ("depth", "Depth"), ("pointcloud", "点云"),
            ("motor", "电机"), ("em", "EM"), ("endoscope", "内窥镜"),
        )):
            label = QtWidgets.QLabel(f"{title}  —")
            label.setObjectName("muted")
            self.stream_labels[key] = label
            grid.addWidget(label, index // 2, index % 2)
        stream_layout.addLayout(grid)
        self.duration_label = QtWidgets.QLabel("时长 —")
        self.duration_label.setObjectName("micro")
        stream_layout.addWidget(self.duration_label)
        middle.addWidget(streams, 1)
        root.addLayout(middle)
        root.addStretch(1)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.continue_button = QtWidgets.QPushButton("继续：初始化七点  →")
        self.continue_button.setObjectName("primary")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.continueRequested)
        footer.addWidget(self.continue_button)
        root.addLayout(footer)

    def _selection_changed(self, index: int) -> None:
        descriptor = self.session_combo.itemData(index)
        if isinstance(descriptor, SessionDescriptor):
            self.sessionSelected.emit(descriptor)

    def set_sessions(self, sessions: list[SessionDescriptor], current: Path | None = None) -> None:
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        for descriptor in sessions:
            self.session_combo.addItem(descriptor.path.name, descriptor)
        self.session_combo.blockSignals(False)
        if sessions:
            index = next((i for i, item in enumerate(sessions) if item.path == current), 0)
            self.session_combo.setCurrentIndex(index)
            self.sessionSelected.emit(sessions[index])

    def set_descriptor(self, descriptor: SessionDescriptor | None) -> None:
        if descriptor is None:
            return
        usable = descriptor.kind == SessionKind.LEGACY_VIDEO and descriptor.has_rgb and descriptor.has_depth
        self.quality_badge.setText("可用于七点重建" if usable else descriptor.quality_title)
        self.quality_badge.setObjectName("successBadge" if usable else "warningBadge")
        self.quality_badge.style().unpolish(self.quality_badge)
        self.quality_badge.style().polish(self.quality_badge)
        self.quality_detail.setText(
            "RGB 用于 CoTracker3 七点轨迹，Depth 用于三维坐标恢复。"
            if usable else descriptor.quality_detail
        )
        path_text = str(descriptor.path)
        self.path_label.setText(descriptor.path.name)
        self.path_label.setToolTip(path_text)
        values = {
            "rgb": descriptor.has_rgb, "depth": descriptor.has_depth,
            "pointcloud": descriptor.has_pointcloud, "motor": descriptor.has_motor,
            "em": descriptor.has_em, "endoscope": descriptor.has_endoscope,
        }
        titles = {"rgb":"RGB", "depth":"Depth", "pointcloud":"点云", "motor":"电机", "em":"EM", "endoscope":"内窥镜"}
        for key, present in values.items():
            self.stream_labels[key].setText(f"{titles[key]}  {'✓' if present else '—'}")
            self.stream_labels[key].setStyleSheet(f"color:{'#17834D' if present else '#A0A0A5'};")
        self.duration_label.setText(
            f"{descriptor.frame_count:,} 帧 · {descriptor.fps:.1f} fps · {descriptor.duration_s:.1f} 秒"
        )
        self.continue_button.setEnabled(usable)


class InitializationPage(QtWidgets.QWidget):
    frameRequested = QtCore.pyqtSignal(int)
    playToggled = QtCore.pyqtSignal(bool)
    startClicking = QtCore.pyqtSignal()
    clearRequested = QtCore.pyqtSignal()
    pointsCompleted = QtCore.pyqtSignal(object)
    continueRequested = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 22)
        root.setSpacing(16)
        left, left_layout = card("首帧七点初始化", "步骤 2 / 4", hero=True)
        self.canvas = VideoCanvas("加载会话后显示 RGB")
        self.canvas.keypointsCompleted.connect(self.pointsCompleted)
        self.canvas.pointCountChanged.connect(self._point_count_changed)
        left_layout.addWidget(self.canvas, 1)
        transport = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("播放")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self.playToggled)
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.frameRequested)
        self.frame_label = QtWidgets.QLabel("0 / 0")
        self.frame_label.setMinimumWidth(100)
        transport.addWidget(self.play_button)
        transport.addWidget(self.frame_slider, 1)
        transport.addWidget(self.frame_label)
        left_layout.addLayout(transport)
        root.addWidget(left, 1)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(330)
        side = QtWidgets.QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(12)
        guide, guide_layout = card("操作顺序")
        step = QtWidgets.QLabel(
            "1. 停在七条色带清晰的一帧\n\n"
            "2. 点击“开始标记”\n\n"
            "3. 从近端向末端粗略点击 K0 → K6\n\n"
            "完成后程序会自动吸附到黑色中心线和七个等弧长位置；"
            "右键可以撤销上一个点"
        )
        step.setObjectName("muted")
        step.setWordWrap(True)
        guide_layout.addWidget(step)
        side.addWidget(guide)
        state_card, state_layout = card("初始化状态")
        self.status_badge = QtWidgets.QLabel("等待初始化")
        self.status_badge.setObjectName("warningBadge")
        self.point_status = QtWidgets.QLabel("已标记 0 / 7")
        self.point_status.setObjectName("muted")
        state_layout.addWidget(self.status_badge, 0, QtCore.Qt.AlignLeft)
        state_layout.addWidget(self.point_status)
        row = QtWidgets.QHBoxLayout()
        start = QtWidgets.QPushButton("开始标记")
        start.setObjectName("primary")
        start.clicked.connect(self.startClicking)
        clear = QtWidgets.QPushButton("重新标记")
        clear.clicked.connect(self.clearRequested)
        row.addWidget(start, 1)
        row.addWidget(clear)
        state_layout.addLayout(row)
        side.addWidget(state_card)
        side.addStretch(1)
        self.continue_button = QtWidgets.QPushButton("继续：开始重建  →")
        self.continue_button.setObjectName("primary")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.continueRequested)
        side.addWidget(self.continue_button)
        root.addWidget(panel)

    def _point_count_changed(self, count: int) -> None:
        self.point_status.setText(f"已标记 {count} / 7" + (f" · 下一点 K{count}" if count < 7 else ""))

    def configure_timeline(self, count: int) -> None:
        self.frame_slider.setRange(0, max(0, count - 1))
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"1 / {max(1, count)}")

    def set_frame(self, index: int, total: int, image: np.ndarray | None) -> None:
        self.canvas.set_bgr(image)
        self.frame_label.setText(f"{index + 1} / {max(1, total)}")

    def set_initialized(self, pixels: np.ndarray, frame: int) -> None:
        self.canvas.set_keypoints(pixels)
        self.status_badge.setText(f"自动中心线初始化完成 · 第 {frame + 1} 帧")
        self.status_badge.setObjectName("successBadge")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.point_status.setText("K0–K6 已自动吸附并锁定身份顺序")
        self.continue_button.setEnabled(True)

    def reset(self) -> None:
        self.canvas.cancel_seven_points()
        self.status_badge.setText("等待初始化")
        self.status_badge.setObjectName("warningBadge")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.point_status.setText("已标记 0 / 7")
        self.continue_button.setEnabled(False)


class ReconstructionPage(QtWidgets.QWidget):
    startRequested = QtCore.pyqtSignal()
    stopRequested = QtCore.pyqtSignal()
    startRecordingRequested = QtCore.pyqtSignal()
    stopRecordingRequested = QtCore.pyqtSignal()
    resultRequested = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)
        hero, hero_layout = card("七点三维重建", "步骤 3 / 4", hero=True)
        chain = QtWidgets.QLabel(
            "CoTracker3 七点轨迹  →  SAM2 主体审核  →  D435 Depth/IR 融合  →  两段连续体曲线约束"
        )
        chain.setObjectName("muted")
        chain.setWordWrap(True)
        hero_layout.addWidget(chain)
        self.readiness = QtWidgets.QLabel("等待数据与初始化")
        self.readiness.setObjectName("warningBadge")
        hero_layout.addWidget(self.readiness, 0, QtCore.Qt.AlignLeft)
        root.addWidget(hero)

        body = QtWidgets.QHBoxLayout()
        preview_card, preview_layout = card("处理预览")
        self.preview = VideoCanvas("处理开始后显示轨迹叠加")
        preview_layout.addWidget(self.preview, 1)
        body.addWidget(preview_card, 3)
        progress_card, progress_layout = card("任务状态")
        self.phase = QtWidgets.QLabel("尚未开始")
        self.phase.setObjectName("sectionTitle")
        self.detail = QtWidgets.QLabel("系统将先生成完整七点轨迹，再逐帧计算三维坐标。")
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.counter = QtWidgets.QLabel("0 / 0 帧")
        self.counter.setObjectName("micro")
        progress_layout.addWidget(self.phase)
        progress_layout.addWidget(self.detail)
        progress_layout.addWidget(self.progress)
        progress_layout.addWidget(self.counter)
        progress_layout.addStretch(1)
        controls = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("开始追踪")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.startRequested)
        self.stop_button = QtWidgets.QPushButton("停止追踪")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stopRequested)
        controls.addWidget(self.start_button, 1)
        controls.addWidget(self.stop_button, 1)
        progress_layout.addLayout(controls)
        recording_controls = QtWidgets.QHBoxLayout()
        self.start_recording_button = QtWidgets.QPushButton("开始记录")
        self.start_recording_button.setObjectName("secondaryBlue")
        self.start_recording_button.setEnabled(False)
        self.start_recording_button.clicked.connect(self.startRecordingRequested)
        self.stop_recording_button = QtWidgets.QPushButton("停止记录并保存")
        self.stop_recording_button.setObjectName("danger")
        self.stop_recording_button.setEnabled(False)
        self.stop_recording_button.clicked.connect(self.stopRecordingRequested)
        recording_controls.addWidget(self.start_recording_button, 1)
        recording_controls.addWidget(self.stop_recording_button, 1)
        progress_layout.addLayout(recording_controls)
        self.recording_status = QtWidgets.QLabel("记录：关闭（仅追踪预览）")
        self.recording_status.setObjectName("micro")
        progress_layout.addWidget(self.recording_status)
        self.result_button = QtWidgets.QPushButton("查看重建结果  →")
        self.result_button.setObjectName("secondaryBlue")
        self.result_button.setEnabled(False)
        self.result_button.clicked.connect(self.resultRequested)
        progress_layout.addWidget(self.result_button)
        body.addWidget(progress_card, 2)
        root.addLayout(body, 1)

    def set_ready(self, ready: bool) -> None:
        self.start_button.setEnabled(ready)
        self.readiness.setText("可以开始" if ready else "请先完成数据选择与七点初始化")
        self.readiness.setObjectName("successBadge" if ready else "warningBadge")
        self.readiness.style().unpolish(self.readiness)
        self.readiness.style().polish(self.readiness)

    def set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.start_recording_button.setEnabled(running)
        if not running:
            self.set_recording(False)

    def set_recording(self, recording: bool) -> None:
        self.start_recording_button.setEnabled(
            self.stop_button.isEnabled() and not recording
        )
        self.stop_recording_button.setEnabled(recording)
        self.recording_status.setText(
            "记录：进行中（叠加视频、七点2D/3D与质量数据）"
            if recording else "记录：关闭（追踪继续运行）"
        )
        self.recording_status.setObjectName(
            "successBadge" if recording else "micro"
        )
        self.recording_status.style().unpolish(self.recording_status)
        self.recording_status.style().polish(self.recording_status)

    def update_progress(self, done: int, total: int, visuals: dict) -> None:
        self.progress.setValue(int(1000 * done / max(total, 1)))
        self.counter.setText(f"{done:,} / {total:,} 帧")
        if visuals.get("phase") == "cotracker3":
            self.phase.setText("正在生成七点轨迹")
            self.detail.setText("CoTracker3 CUDA 正在联合跟踪 K0–K6。此阶段不会执行深度和其他先验。")
        else:
            self.phase.setText("正在重建三维坐标")
            self.detail.setText("轨迹已经固定；当前进行主体审核、深度融合与曲线约束。")
            self.preview.set_bgr(visuals.get("overlay"))

    def complete(self, target: Path) -> None:
        self.progress.setValue(1000)
        self.phase.setText("重建完成")
        self.detail.setText(f"结果已保存到：\n{target}")
        self.result_button.setEnabled(True)


class ResultsPage(QtWidgets.QWidget):
    openFolderRequested = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 22)
        root.setSpacing(14)
        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("重建结果")
        title.setObjectName("sectionTitle")
        self.summary = QtWidgets.QLabel("处理过程中会实时更新三维形状和七点坐标。")
        self.summary.setObjectName("muted")
        title_box.addWidget(title)
        title_box.addWidget(self.summary)
        header.addLayout(title_box)
        header.addStretch(1)
        self.open_button = QtWidgets.QPushButton("打开输出文件夹")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.openFolderRequested)
        header.addWidget(self.open_button)
        root.addLayout(header)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.shape_view = Shape3DView()
        right = QtWidgets.QTabWidget()
        self.table = KeypointTable()
        right.addTab(self.table, "七点坐标")
        self.overlay = VideoCanvas("暂无叠加预览")
        right.addTab(self.overlay, "识别叠加")
        self.depth = VideoCanvas("暂无深度预览")
        right.addTab(self.depth, "深度")
        split.addWidget(self.shape_view)
        split.addWidget(right)
        split.setSizes((900, 620))
        root.addWidget(split, 1)

    def update_result(self, result, visuals: dict) -> None:
        if result is None:
            return
        self.shape_view.set_result(result)
        self.table.set_result(result)
        self.overlay.set_bgr(visuals.get("overlay"))
        self.depth.set_bgr(visuals.get("depth"))
        count = int(np.count_nonzero(result.measured_valid))
        self.summary.setText(
            f"当前帧实测 {count}/7 · 单帧融合 {result.processing_ms:.1f} ms"
        )

    def set_output(self, target: Path) -> None:
        self.open_button.setEnabled(True)
        self.summary.setText(f"重建完成 · {target}")


class SettingsPage(QtWidgets.QWidget):
    browseOutputRequested = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(content)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(14)

        notice, notice_layout = card("主识别链路保持固定", hero=True)
        text = QtWidgets.QLabel(
            "CoTracker3、SAM2、Depth/IR 融合和两段曲线约束始终按固定顺序执行。"
            "以下设置均为可选增强，不会改变 K0–K6 的身份。"
        )
        text.setObjectName("muted")
        text.setWordWrap(True)
        notice_layout.addWidget(text)
        root.addWidget(notice)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._processing_tab(), "处理策略")
        tabs.addTab(self._calibration_tab(), "相机与配准")
        tabs.addTab(self._output_tab(), "输出")
        root.addWidget(tabs)
        root.addStretch(1)
        scroll.setWidget(content)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def _processing_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.motor_prior = QtWidgets.QCheckBox("启用电机/PCC形状先验（遮挡时补全）")
        self.motor_prior.setChecked(False)
        self.local_depth = QtWidgets.QCheckBox("启用局部深度曲线修正")
        self.local_depth.setChecked(True)
        self.em_constraint = QtWidgets.QCheckBox("启用EM双探子端距约束")
        self.em_constraint.setChecked(False)
        self.offline_refine = QtWidgets.QCheckBox("完成后执行零相位时序与三维曲线精修")
        self.offline_refine.setChecked(True)
        for item in (self.motor_prior, self.local_depth, self.em_constraint, self.offline_refine):
            layout.addWidget(item)
        form = QtWidgets.QFormLayout()
        self.local_depth_radius = NoWheelSpinBox()
        self.local_depth_radius.setRange(6, 48)
        self.local_depth_radius.setValue(22)
        self.local_depth_gate = NoWheelDoubleSpinBox()
        self.local_depth_gate.setRange(0.003, 0.060)
        self.local_depth_gate.setDecimals(3)
        self.local_depth_gate.setValue(0.018)
        self.temporal_strength = NoWheelDoubleSpinBox()
        self.temporal_strength.setRange(0, 30)
        self.temporal_strength.setValue(5)
        self.curve_samples = NoWheelSpinBox()
        self.curve_samples.setRange(43, 169)
        self.curve_samples.setValue(85)
        form.addRow("局部深度半径 px", self.local_depth_radius)
        form.addRow("深度门限 m", self.local_depth_gate)
        form.addRow("离线时序强度", self.temporal_strength)
        form.addRow("曲线采样点", self.curve_samples)
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _calibration_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.dx = NoWheelDoubleSpinBox(); self.dx.setRange(-300, 300); self.dx.setDecimals(1)
        self.dy = NoWheelDoubleSpinBox(); self.dy.setRange(-300, 300); self.dy.setDecimals(1)
        self.scale_x = NoWheelDoubleSpinBox(); self.scale_x.setRange(0.8, 1.2); self.scale_x.setDecimals(4); self.scale_x.setValue(1)
        self.scale_y = NoWheelDoubleSpinBox(); self.scale_y.setRange(0.8, 1.2); self.scale_y.setDecimals(4); self.scale_y.setValue(1)
        self.em_first = NoWheelSpinBox(); self.em_first.setRange(0, 99); self.em_first.setValue(10)
        self.em_second = NoWheelSpinBox(); self.em_second.setRange(0, 99); self.em_second.setValue(11)
        form.addRow("Depth→RGB 水平偏移 px", self.dx)
        form.addRow("Depth→RGB 垂直偏移 px", self.dy)
        form.addRow("Depth→RGB 水平尺度", self.scale_x)
        form.addRow("Depth→RGB 垂直尺度", self.scale_y)
        form.addRow("EM近端 Port", self.em_first)
        form.addRow("EM远端 Port", self.em_second)
        return page

    def _output_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        row = QtWidgets.QHBoxLayout()
        self.output_edit = QtWidgets.QLineEdit(str((VISUAL_ROOT / "outputs").resolve()))
        browse = QtWidgets.QPushButton("选择…")
        browse.clicked.connect(self.browseOutputRequested)
        row.addWidget(self.output_edit, 1)
        row.addWidget(browse)
        layout.addWidget(QtWidgets.QLabel("结果根目录"))
        layout.addLayout(row)
        self.save_overlay = QtWidgets.QCheckBox("保存七点叠加视频")
        self.save_overlay.setChecked(True)
        layout.addWidget(self.save_overlay)
        hint = QtWidgets.QLabel("每次任务创建独立的带时间戳目录；不会覆盖已有结果。")
        hint.setObjectName("muted")
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def parameters(self) -> dict:
        return {
            "motor_prior_enabled": self.motor_prior.isChecked(),
            "local_depth_curve_enabled": self.local_depth.isChecked(),
            "em_chord_constraint_enabled": self.em_constraint.isChecked(),
            "offline_refine_enabled": self.offline_refine.isChecked(),
            "local_depth_search_radius_px": self.local_depth_radius.value(),
            "local_depth_gate_m": self.local_depth_gate.value(),
            "offline_temporal_strength": self.temporal_strength.value(),
            "curve_samples": self.curve_samples.value(),
            "registration_dx_px": self.dx.value(),
            "registration_dy_px": self.dy.value(),
            "registration_scale_x": self.scale_x.value(),
            "registration_scale_y": self.scale_y.value(),
            "em_first_port": self.em_first.value(),
            "em_second_port": self.em_second.value(),
            "preview_stride": 3,
        }
