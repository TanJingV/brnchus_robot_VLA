"""Background processing boundary for the Qt application."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PyQt5 import QtCore

from ..processor import process_session


class ReconstructionWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, object, dict)
    completed = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        session: Path,
        output: Path,
        config: dict,
        parameters: dict,
        start_frame: int,
        end_frame: int,
        save_overlay: bool,
    ) -> None:
        super().__init__()
        self.session = Path(session)
        self.output = Path(output)
        self.config = config
        self.parameters = parameters
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.save_overlay = bool(save_overlay)
        self.stop_event = threading.Event()
        self.recording_event = threading.Event()

    def request_stop(self) -> None:
        self.stop_event.set()

    def start_recording(self) -> None:
        self.recording_event.set()

    def stop_recording(self) -> None:
        self.recording_event.clear()

    @property
    def is_recording(self) -> bool:
        return self.recording_event.is_set()

    def run(self) -> None:
        try:
            target = process_session(
                self.session,
                self.output,
                self.config,
                self.parameters,
                lambda done, total, result, visuals: self.progress.emit(
                    done, total, result, visuals
                ),
                self.stop_event,
                self.start_frame,
                self.end_frame if self.end_frame > 0 else None,
                self.save_overlay,
                self.recording_event.is_set,
            )
            self.completed.emit(str(target))
        except InterruptedError:
            self.failed.emit("任务已由用户停止。")
        except Exception:
            self.failed.emit(traceback.format_exc())
