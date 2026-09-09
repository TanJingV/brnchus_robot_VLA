from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PyQt5 import QtCore, QtGui, QtWidgets

from Visual_information.seven_marker_3d_fusion.workstation.state import WorkstationState
from Visual_information.seven_marker_3d_fusion.workstation.widgets import VideoCanvas
from Visual_information.seven_marker_3d_fusion.workstation.worker import ReconstructionWorker
from Visual_information.seven_marker_3d_fusion.workstation.window import WorkstationWindow


class WorkstationArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_five_pages_and_linear_navigation_exist(self):
        window = WorkstationWindow()
        self.assertEqual(window.pages.count(), 5)
        self.assertEqual(len(window.nav_buttons), 5)
        self.assertEqual(window.windowTitle(), "TDCR Vision Studio")
        window.close()

    def test_initialization_is_the_reconstruction_gate(self):
        state = WorkstationState()
        self.assertFalse(state.initialization_ready)
        points = np.column_stack((np.linspace(600, 300, 7), np.full(7, 220.0)))
        state.set_initialization(12, points, (480, 848))
        self.assertTrue(state.initialization_ready)
        self.assertEqual(state.initial_frame, 12)
        self.assertIsNotNone(state.target_roi)

    def test_interfering_priors_are_disabled_by_default(self):
        window = WorkstationWindow()
        self.assertFalse(window.settings_page.motor_prior.isChecked())
        self.assertFalse(window.settings_page.em_constraint.isChecked())
        self.assertTrue(window.settings_page.local_depth.isChecked())
        window.close()

    def test_tracking_and_recording_controls_are_independent(self):
        window = WorkstationWindow()
        page = window.reconstruction_page
        page.set_running(True)
        self.assertTrue(page.stop_button.isEnabled())
        self.assertTrue(page.start_recording_button.isEnabled())
        self.assertFalse(page.stop_recording_button.isEnabled())
        page.set_recording(True)
        self.assertTrue(page.stop_button.isEnabled())
        self.assertFalse(page.start_recording_button.isEnabled())
        self.assertTrue(page.stop_recording_button.isEnabled())
        page.set_recording(False)
        self.assertTrue(page.stop_button.isEnabled())
        self.assertTrue(page.start_recording_button.isEnabled())
        window.close()

    def test_worker_recording_gate_does_not_stop_tracking(self):
        worker = ReconstructionWorker(
            Path("session"), Path("output"), {}, {}, 0, 1, False
        )
        self.assertFalse(worker.is_recording)
        worker.start_recording()
        self.assertTrue(worker.is_recording)
        self.assertFalse(worker.stop_event.is_set())
        worker.stop_recording()
        self.assertFalse(worker.is_recording)
        self.assertFalse(worker.stop_event.is_set())

    def test_video_canvas_wheel_zoom_and_reset(self):
        canvas = VideoCanvas()
        canvas.resize(640, 360)
        canvas.set_bgr(np.zeros((360, 640, 3), dtype=np.uint8))
        event = QtGui.QWheelEvent(
            QtCore.QPointF(320, 180),
            QtCore.QPointF(320, 180),
            QtCore.QPoint(0, 0),
            QtCore.QPoint(0, 120),
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
            QtCore.Qt.ScrollUpdate,
            False,
        )
        canvas.wheelEvent(event)
        self.assertGreater(canvas.zoom_factor, 1.0)
        canvas.reset_view()
        self.assertEqual(canvas.zoom_factor, 1.0)


if __name__ == "__main__":
    unittest.main()
