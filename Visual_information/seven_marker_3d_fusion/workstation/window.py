"""Top-level product shell and workflow controller."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from Visual_information.d435_tdcr_capture.config import PROJECT_ROOT

from ..config import load_fusion_config
from ..cotracker3_tracker import TrackedPointRefiner
from ..session import LegacySessionReader, SessionDescriptor, SessionKind, describe_session, discover_sessions
from .pages import DataPage, InitializationPage, ReconstructionPage, ResultsPage, SettingsPage
from .state import WorkstationState
from .theme import PAGE_TITLES, PRODUCT_QSS
from .worker import ReconstructionWorker


class WorkstationWindow(QtWidgets.QMainWindow):
    """A linear four-step workflow with isolated advanced settings."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TDCR Vision Studio")
        self.resize(1640, 980)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(PRODUCT_QSS)
        self.state = WorkstationState()
        self.config = load_fusion_config()
        self.reader: LegacySessionReader | None = None
        self.current_rgb_bgr: np.ndarray | None = None
        self.worker: ReconstructionWorker | None = None
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)
        self._build_shell()
        self._wire_pages()
        self._scan_sessions()

    def _build_shell(self) -> None:
        root = QtWidgets.QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        horizontal = QtWidgets.QHBoxLayout(root)
        horizontal.setContentsMargins(0, 0, 0, 0)
        horizontal.setSpacing(0)
        horizontal.addWidget(self._build_sidebar())
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self._build_topbar())
        self.pages = QtWidgets.QStackedWidget()
        self.data_page = DataPage()
        self.init_page = InitializationPage()
        self.reconstruction_page = ReconstructionPage()
        self.results_page = ResultsPage()
        self.settings_page = SettingsPage()
        for page in (
            self.data_page, self.init_page, self.reconstruction_page,
            self.results_page, self.settings_page,
        ):
            self.pages.addWidget(page)
        content_layout.addWidget(self.pages, 1)
        horizontal.addWidget(content, 1)

    def _build_sidebar(self) -> QtWidgets.QWidget:
        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(228)
        layout = QtWidgets.QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 20, 16, 18)
        layout.setSpacing(8)
        brand = QtWidgets.QHBoxLayout()
        mark = QtWidgets.QLabel("T")
        mark.setObjectName("brandMark")
        mark.setAlignment(QtCore.Qt.AlignCenter)
        names = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("TDCR Vision")
        title.setObjectName("brandTitle")
        caption = QtWidgets.QLabel("3D Reconstruction Studio")
        caption.setObjectName("brandCaption")
        names.addWidget(title)
        names.addWidget(caption)
        brand.addWidget(mark)
        brand.addLayout(names)
        layout.addLayout(brand)
        layout.addSpacing(24)
        self.nav_group = QtWidgets.QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: list[QtWidgets.QPushButton] = []
        icons = ("01  数据", "02  初始化", "03  重建", "04  结果", "⚙   设置")
        for index, text in enumerate(icons):
            button = QtWidgets.QPushButton(text)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked, page=index: self._show_page(page))
            self.nav_group.addButton(button, index)
            self.nav_buttons.append(button)
            layout.addWidget(button)
        self.nav_buttons[0].setChecked(True)
        layout.addStretch(1)
        self.pipeline_status = QtWidgets.QLabel("主链路\nCoTracker3 → SAM2 → 3D")
        self.pipeline_status.setObjectName("sidebarStatus")
        self.pipeline_status.setWordWrap(True)
        layout.addWidget(self.pipeline_status)
        return sidebar

    def _build_topbar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame()
        bar.setObjectName("topbar")
        bar.setFixedHeight(76)
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(24, 10, 24, 10)
        titles = QtWidgets.QVBoxLayout()
        self.page_title = QtWidgets.QLabel(PAGE_TITLES[0][0])
        self.page_title.setObjectName("pageTitle")
        self.page_subtitle = QtWidgets.QLabel(PAGE_TITLES[0][1])
        self.page_subtitle.setObjectName("pageSubtitle")
        titles.addWidget(self.page_title)
        titles.addWidget(self.page_subtitle)
        layout.addLayout(titles)
        layout.addStretch(1)
        self.session_chip = QtWidgets.QLabel("未选择会话")
        self.session_chip.setObjectName("warningBadge")
        layout.addWidget(self.session_chip)
        return bar

    def _wire_pages(self) -> None:
        self.data_page.refreshRequested.connect(self._scan_sessions)
        self.data_page.browseRequested.connect(self._browse_session)
        self.data_page.sessionSelected.connect(self._open_session)
        self.data_page.continueRequested.connect(lambda: self._show_page(1))
        self.init_page.frameRequested.connect(self._load_frame)
        self.init_page.playToggled.connect(self._toggle_play)
        self.init_page.startClicking.connect(self.init_page.canvas.begin_seven_points)
        self.init_page.clearRequested.connect(self._clear_initialization)
        self.init_page.pointsCompleted.connect(self._accept_initialization)
        self.init_page.continueRequested.connect(lambda: self._show_page(2))
        self.reconstruction_page.startRequested.connect(self._start_reconstruction)
        self.reconstruction_page.stopRequested.connect(self._stop_reconstruction)
        self.reconstruction_page.startRecordingRequested.connect(self._start_recording)
        self.reconstruction_page.stopRecordingRequested.connect(self._stop_recording)
        self.reconstruction_page.resultRequested.connect(lambda: self._show_page(3))
        self.results_page.openFolderRequested.connect(self._open_output_folder)
        self.settings_page.browseOutputRequested.connect(self._browse_output)

    def _show_page(self, index: int) -> None:
        index = int(np.clip(index, 0, self.pages.count() - 1))
        self.pages.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)
        self.page_title.setText(PAGE_TITLES[index][0])
        self.page_subtitle.setText(PAGE_TITLES[index][1])

    def _scan_sessions(self) -> None:
        current = self.state.descriptor.path if self.state.descriptor else None
        self.data_page.set_sessions(discover_sessions(PROJECT_ROOT), current)

    def _browse_session(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择采集 session")
        if folder:
            self._open_session(describe_session(folder))

    def _close_reader(self) -> None:
        self.play_timer.stop()
        if self.reader is not None:
            self.reader.close()
        self.reader = None

    def _open_session(self, descriptor: SessionDescriptor) -> None:
        if self.worker is not None:
            return
        self._close_reader()
        self.state.descriptor = descriptor
        self.state.current_frame = 0
        self.state.reset_initialization()
        self.data_page.set_descriptor(descriptor)
        self.init_page.reset()
        self.init_page.configure_timeline(descriptor.frame_count)
        self.reconstruction_page.set_ready(False)
        self.session_chip.setText(descriptor.path.name)
        self.session_chip.setObjectName("successBadge" if self.state.source_ready else "warningBadge")
        self.session_chip.style().unpolish(self.session_chip)
        self.session_chip.style().polish(self.session_chip)
        if descriptor.kind == SessionKind.LEGACY_VIDEO:
            try:
                self.reader = LegacySessionReader(descriptor.path)
                self._load_frame(0)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "无法读取会话", str(exc))

    def _load_frame(self, index: int) -> None:
        if self.reader is None or self.state.descriptor is None:
            return
        index = int(np.clip(index, 0, self.state.descriptor.frame_count - 1))
        try:
            packet = self.reader.read(index, {})
            self.current_rgb_bgr = np.asarray(packet.rgb_bgr, np.uint8).copy()
            self.state.current_frame = index
            self.init_page.set_frame(index, self.state.descriptor.frame_count, packet.rgb_bgr)
        except Exception as exc:
            self.statusBar().showMessage(f"读取第 {index + 1} 帧失败：{exc}", 5000)

    def _toggle_play(self, playing: bool) -> None:
        if playing:
            fps = self.state.descriptor.fps if self.state.descriptor else 30.0
            self.play_timer.start(max(1, int(round(1000.0 / fps))))
        else:
            self.play_timer.stop()

    def _play_tick(self) -> None:
        if self.state.descriptor is None:
            return
        next_frame = self.init_page.frame_slider.value() + 1
        if next_frame >= self.state.descriptor.frame_count:
            self.init_page.play_button.setChecked(False)
            return
        self.init_page.frame_slider.setValue(next_frame)

    def _clear_initialization(self) -> None:
        self.state.reset_initialization()
        self.init_page.reset()
        self.reconstruction_page.set_ready(False)

    def _accept_initialization(self, pixels) -> None:
        width, height = self.init_page.canvas.source_size
        if width <= 0 or height <= 0:
            return
        seed = np.asarray(pixels, dtype=float)
        corrected = seed
        if self.current_rgb_bgr is not None:
            refiner = TrackedPointRefiner(self.config.get("tracking", {}))
            corrected = refiner.initialise(self.current_rgb_bgr, seed)
        self.state.set_initialization(
            self.state.current_frame,
            corrected,
            (height, width),
        )
        self.init_page.set_initialized(self.state.initial_keypoints_px, self.state.initial_frame)
        self.reconstruction_page.set_ready(True)

    def _parameters(self) -> dict:
        values = self.settings_page.parameters()
        values.update({
            "initial_keypoints_px": self.state.initial_keypoints_px.tolist(),
            "keypoint_init_frame": int(self.state.initial_frame),
            "target_lock_roi": list(self.state.target_roi) if self.state.target_roi else None,
            "roi": list(self.state.target_roi) if self.state.target_roi else [0, 0, 1, 1],
        })
        return values

    def _start_reconstruction(self) -> None:
        if not self.state.source_ready or not self.state.initialization_ready or self.worker is not None:
            QtWidgets.QMessageBox.warning(self, "尚未准备完成", "请先选择有效会话并完成 K0–K6 初始化。")
            return
        self._close_reader()
        descriptor = self.state.descriptor
        self.state.output_root = Path(self.settings_page.output_edit.text()).expanduser().resolve()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = self.state.output_root / f"{descriptor.path.name}_reconstruction_{stamp}"
        config = load_fusion_config()
        # The architecture exposes one unambiguous primary mode.
        # The transparent guide is stationary while the material lattice moves;
        # CoTracker locks to that guide.  BodyFirst's arc-unwrapped seven-band
        # matcher is the production tracker, with the clicks used for identity.
        config["tracking"]["cotracker3_enabled"] = False
        config["tracking"]["tdcr_yolo_pose_strict"] = False
        self.worker = ReconstructionWorker(
            descriptor.path,
            target,
            config,
            self._parameters(),
            self.state.initial_frame,
            descriptor.frame_count,
            self.settings_page.save_overlay.isChecked(),
        )
        self.worker.progress.connect(self._processing_progress)
        self.worker.completed.connect(self._processing_completed)
        self.worker.failed.connect(self._processing_failed)
        self.state.processing = True
        self.reconstruction_page.set_running(True)
        self.reconstruction_page.result_button.setEnabled(False)
        self.worker.start()

    def _start_recording(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        self.worker.start_recording()
        self.reconstruction_page.set_recording(True)
        self.statusBar().showMessage("记录已开始；停止记录不会停止追踪。", 4000)

    def _stop_recording(self) -> None:
        if self.worker is None:
            return
        self.worker.stop_recording()
        self.reconstruction_page.set_recording(False)
        self.statusBar().showMessage("记录已停止并正在安全写入；追踪继续运行。", 5000)

    def _processing_progress(self, done: int, total: int, result, visuals: dict) -> None:
        self.reconstruction_page.update_progress(done, total, visuals)
        if result is not None:
            self.results_page.update_result(result, visuals)

    def _stop_reconstruction(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.reconstruction_page.stop_button.setEnabled(False)
            self.reconstruction_page.start_recording_button.setEnabled(False)
            self.reconstruction_page.phase.setText("正在安全停止追踪…")

    def _processing_completed(self, target: str) -> None:
        path = Path(target)
        recorded_frames = 0
        summary_path = path / "tracking_summary.json"
        if summary_path.is_file():
            try:
                recorded_frames = int(json.loads(
                    summary_path.read_text(encoding="utf-8")
                ).get("recorded_frames", 0))
            except Exception:
                recorded_frames = 0
        self.state.latest_output = path
        self.state.processing = False
        self.reconstruction_page.set_running(False)
        self.reconstruction_page.complete(path)
        self.results_page.set_output(path)
        self._finish_worker()
        QtWidgets.QMessageBox.information(
            self,
            "追踪完成",
            (
                f"追踪与记录完成，共保存 {recorded_frames:,} 帧：\n{path}"
                if recorded_frames
                else "追踪已结束。本次没有开启记录，因此未写入七点数据和叠加视频。"
            ),
        )

    def _processing_failed(self, detail: str) -> None:
        self.state.processing = False
        self.reconstruction_page.set_running(False)
        self.reconstruction_page.phase.setText("任务未完成")
        self.reconstruction_page.detail.setText(detail.splitlines()[-1] if detail else "未知错误")
        self._finish_worker()
        if "用户停止" not in detail:
            QtWidgets.QMessageBox.critical(self, "重建失败", detail)

    def _finish_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None
        if self.state.descriptor is not None and self.state.descriptor.kind == SessionKind.LEGACY_VIDEO:
            self.reader = LegacySessionReader(self.state.descriptor.path)

    def _browse_output(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择结果根目录")
        if folder:
            self.settings_page.output_edit.setText(folder)

    def _open_output_folder(self) -> None:
        if self.state.latest_output is not None and self.state.latest_output.exists():
            os.startfile(str(self.state.latest_output))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            answer = QtWidgets.QMessageBox.question(
                self, "正在执行重建", "停止任务并退出？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.worker.request_stop()
            self.worker.wait(15000)
        self._close_reader()
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("TDCR Vision Studio")
    window = WorkstationWindow()
    window.show()
    return app.exec_()
