import sys
import os
import time
import json
import csv
import copy
import importlib.util

# MuJoCo, PyTorch, OpenCV, VTK, and MKL can load different OpenMP runtimes on
# Windows. Without this guard the process may abort when simulation is enabled.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
for _path in (BASE_DIR, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scipy.spatial import cKDTree
from queue import Queue, Empty
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMessageBox,QTabWidget, QLabel, QPushButton,QSplitter,QWidget,QVBoxLayout,QComboBox,QHBoxLayout,QLineEdit,QSizePolicy
from pyvistaqt import QtInteractor
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5 import QtGui
import torch
from Visual_information.legacy_segmentation.unet import Unet
from scipy.spatial.transform import Rotation as R
import pyvista as pv
import serial.tools.list_ports
from queue import Queue
from datetime import datetime
import numpy as np
from NDITrackerWrapper import NDITrackerWrapper
from compute_rigid_transform import compute_rigid_transform
from PyQt5.QtMultimedia import QCamera, QCameraInfo
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
import pyiges
from pathplan import pathplan
import vtk
import math
import Trio_UnifiedApi as TUA
import cv2
from torchvision import transforms
from PIL import Image
import pyqtgraph as pg
from collections import deque
from autonomous_navigation_v2 import AdaptiveRecedingHorizonNavigator
from two_segment_tdcr_opencr.contact_stabilization import ActiveTipContactRegularizer
from two_segment_tdcr_opencr.passive_joint_control import (
    PassiveActiveFollowerController,
    PassiveJointParameterController,
)
from two_segment_tdcr_opencr.tendon_compass_control import TendonCompassController
from dual_camera_recording import (
    DualCameraRecordingMixin,
    inspect_video_file,
    open_compatible_avi_writer,
)
from simulation_automation_mixin import (
    SimulationAutomationMixin,
    bind_runtime_globals as bind_automation_runtime_globals,
)
from simulation_runtime_mixin import (
    SimulationRuntimeMixin,
    bind_runtime_globals as bind_simulation_runtime_globals,
)

try:
    import mujoco
except Exception:
    mujoco = None

try:
    import mujoco.viewer as mujoco_viewer
except Exception:
    mujoco_viewer = None

DEFAULT_MUJOCO_XML = os.path.join(PROJECT_ROOT, "meshes", "cable_robot_bronch_final_seg2.xml")
REAL_ENVIRONMENT_SCRIPT = os.path.join(
    BASE_DIR, "界面1220（可以使用版本+数据记录）双探子版.py"
)
MUJOCO_TIP_CAMERA = "tip_camera"
MUJOCO_WIDE_FOVY = 70.0
MUJOCO_STEPS_PER_FRAME = 20
MUJOCO_COMPASS_RADIUS = 130
MUJOCO_MAX_INSERTION_RATE = 0.08
MUJOCO_MANUAL_INSERTION_LEAD_M = 0.015
MUJOCO_AUTO_INSERTION_STEP_LIMIT_M = 0.00035
MUJOCO_AUTO_LEAD_MIN_M = 0.004
MUJOCO_AUTO_LEAD_MAX_M = 0.012
MUJOCO_AUTO_STALL_RAMP_FRAMES = 40
MUJOCO_LUNG_DISPLAY_GROUP = 4


def mujoco_quat_diff_angle(q1, q2):
    dot = float(np.dot(np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)))
    return 2.0 * np.arccos(np.clip(abs(dot), -1.0, 1.0))
def EventHandler(et, ival, sval):
    if et == TUA.EventType.Error or et == TUA.EventType.Warning:
        print("MC Error: (%x) %s" % (ival, sval))
    elif et == TUA.EventType.Message:
        print("MC Message:", sval)

def create_mc_connection(ip: str):
    key = ip.upper()
    if key == "PCMCAT":
        return TUA.TrioConnectionPCMCAT(EventHandler)
    elif key == "FLEX7":
        return TUA.TrioConnectionFlex7(EventHandler)
    else:
        return TUA.TrioConnectionTCP(EventHandler, ip)


class WindowsXInputGamepad:
    """Small dependency-free XInput reader for common Windows gamepads."""

    def __init__(self, index=0):
        self.index = index
        self._xinput = None
        self._state_type = None
        self._ctypes = None
        if os.name != "nt":
            return
        try:
            import ctypes
            self._ctypes = ctypes

            class Gamepad(ctypes.Structure):
                _fields_ = [
                    ("buttons", ctypes.c_ushort),
                    ("left_trigger", ctypes.c_ubyte),
                    ("right_trigger", ctypes.c_ubyte),
                    ("thumb_lx", ctypes.c_short),
                    ("thumb_ly", ctypes.c_short),
                    ("thumb_rx", ctypes.c_short),
                    ("thumb_ry", ctypes.c_short),
                ]

            class State(ctypes.Structure):
                _fields_ = [("packet_number", ctypes.c_uint), ("gamepad", Gamepad)]

            for dll_name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
                try:
                    self._xinput = ctypes.WinDLL(dll_name)
                    self._state_type = State
                    break
                except OSError:
                    continue
        except Exception:
            self._xinput = None

    @staticmethod
    def _axis(value, deadzone=0.12):
        normalized = float(value) / (32767.0 if value >= 0 else 32768.0)
        if abs(normalized) <= deadzone:
            return 0.0
        return math.copysign((abs(normalized) - deadzone) / (1.0 - deadzone), normalized)

    def read(self):
        if self._xinput is None or self._state_type is None:
            return None
        state = self._state_type()
        if self._xinput.XInputGetState(self.index, self._ctypes.byref(state)) != 0:
            return None
        pad = state.gamepad
        return {
            "left": np.array([self._axis(pad.thumb_lx), self._axis(pad.thumb_ly)]),
            "right": np.array([self._axis(pad.thumb_rx), self._axis(pad.thumb_ry)]),
            "left_trigger": float(pad.left_trigger) / 255.0,
            "right_trigger": float(pad.right_trigger) / 255.0,
        }


class RobotInitialPoseWindow(QtWidgets.QWidget):
    """Dedicated editor for the complete robot assembly's initial pose."""

    def __init__(self, simulator):
        super().__init__(None, Qt.Tool)
        self.simulator = simulator
        self.setWindowTitle("MuJoCo 机器人整体初始位姿")
        self.setMinimumWidth(420)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setStyleSheet("""
            QWidget {
                background: #f5f5f7;
                color: #1d1d1f;
                font-family: "Microsoft YaHei UI";
                font-size: 13px;
            }
            QLabel#poseTitle { font-size: 19px; font-weight: 650; }
            QLabel#poseHint { color: #6e6e73; }
            QGroupBox {
                background: white;
                border: 1px solid #d9d9df;
                border-radius: 12px;
                margin-top: 10px;
                padding: 14px 10px 10px 10px;
                font-weight: 600;
            }
            QDoubleSpinBox {
                min-height: 30px;
                padding: 2px 7px;
                background: #fbfbfd;
                border: 1px solid #c7c7cc;
                border-radius: 7px;
            }
            QPushButton {
                min-height: 34px;
                border-radius: 9px;
                padding: 3px 14px;
                background: #e9e9ee;
                font-weight: 600;
            }
            QPushButton#applyPose { color: white; background: #0071e3; }
            QLabel#poseStatus {
                background: white;
                border: 1px solid #d9d9df;
                border-radius: 9px;
                padding: 9px;
                color: #4b4b50;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(11)
        title = QLabel("机器人整体初始位姿")
        title.setObjectName("poseTitle")
        root.addWidget(title)
        hint = QLabel("相对原始模型调整整套机器人；应用后自动返回 Home 状态。")
        hint.setObjectName("poseHint")
        root.addWidget(hint)

        self.position_boxes = self._add_pose_group(
            root, "世界坐标平移", ("X", "Y", "Z"),
            -500.0, 500.0, 1.0, " mm",
        )
        self.rotation_boxes = self._add_pose_group(
            root, "世界坐标旋转", ("Roll X", "Pitch Y", "Yaw Z"),
            -180.0, 180.0, 1.0, " deg",
        )

        buttons = QHBoxLayout()
        reset_button = QPushButton("归零")
        reset_button.clicked.connect(self.reset_pose)
        apply_button = QPushButton("应用并重置机器人")
        apply_button.setObjectName("applyPose")
        apply_button.clicked.connect(self.apply_pose)
        buttons.addWidget(reset_button)
        buttons.addWidget(apply_button, 1)
        root.addLayout(buttons)

        self.status_label = QLabel()
        self.status_label.setObjectName("poseStatus")
        root.addWidget(self.status_label)
        self.sync_from_simulator()

    @staticmethod
    def _add_pose_group(
        root, title, labels, minimum, maximum, step, suffix
    ):
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QGridLayout(group)
        boxes = []
        for column, label_text in enumerate(labels):
            layout.addWidget(QLabel(label_text), 0, column)
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(minimum, maximum)
            box.setDecimals(2)
            box.setSingleStep(step)
            box.setSuffix(suffix)
            layout.addWidget(box, 1, column)
            boxes.append(box)
        root.addWidget(group)
        return boxes

    def sync_from_simulator(self):
        state = self.simulator.get_robot_initial_pose()
        for box, value in zip(self.position_boxes, state["translation_mm"]):
            box.setValue(float(value))
        for box, value in zip(self.rotation_boxes, state["rotation_rpy_deg"]):
            box.setValue(float(value))
        self._set_status(state)

    def _set_status(self, state):
        world = np.asarray(state["world_position_mm"], dtype=float)
        self.status_label.setText(
            "根节点世界坐标: "
            f"X {world[0]:.2f} / Y {world[1]:.2f} / Z {world[2]:.2f} mm"
        )

    def apply_pose(self):
        translation = [box.value() for box in self.position_boxes]
        rotation = [box.value() for box in self.rotation_boxes]
        try:
            state = self.simulator.set_robot_initial_pose(
                translation, rotation
            )
        except Exception as exc:
            QMessageBox.critical(self, "整体位姿设置失败", str(exc))
            return
        self._set_status(state)

    def reset_pose(self):
        for box in (*self.position_boxes, *self.rotation_boxes):
            box.setValue(0.0)
        self.apply_pose()


class MujocoPhysicsControlPanel:
    """Two-compass control panel driving the six physical tendon lengths."""

    def __init__(self, model, data, simulator):
        self.model = model
        self.data = data
        self.simulator = simulator
        self.win_name = "MuJoCo Bronchoscope Control"
        self.c1_center = (200, 500)
        self.c2_center = (600, 500)
        self.dragging = None
        self.slider_val = 0.0
        self.slider_command = 0.0
        self.previous_auto_insertion_delta = 0.0
        self.last_auto_slider_position_m = None
        self.auto_insertion_stall_frames = 0
        self.last_control_wall_time = time.perf_counter()
        self.calibration_capture = False
        self.input_mode = "ui"
        self.gamepad = WindowsXInputGamepad()
        self.gamepad_connected = False
        self.window_open = False
        self.latest_frame_bgr = None
        self.passive_debug_locked = False
        self.passive_debug_button_rect = (28, 38, 252, 108)
        self.lung_model_enabled = bool(
            getattr(simulator, "lung_model_enabled", True)
        )
        self.lung_toggle_button_rect = (570, 42, 770, 102)
        self.robot_pose_button_rect = (590, 118, 770, 168)
        self.auto_action = None
        self.distal_feedback_integral_gain = 0.03
        self.distal_feedback_limit = 0.30
        self.distal_feedback_correction = np.zeros(2, dtype=float)
        # Manual compass/gamepad screen-right is raw tendon -X. Keep the
        # visual knob intuitive and invert only the horizontal motor command.
        self.manual_compass_motor_map = np.array(
            [[-1.0, 0.0], [0.0, 1.0]], dtype=float
        )
        self.distal_proximal_feedforward_map = np.array(
            [[0.94, -0.026], [0.0, 0.92]], dtype=float
        )
        # Identified Wire 4-6 command to distal [rotation_y, rotation_z].
        self.distal_motor_rotation_map = np.array(
            [[0.0, -2.4741], [2.4889, 0.0516]], dtype=float
        )
        self.distal_rotation_motor_map = np.linalg.pinv(
            self.distal_motor_rotation_map
        )

        self.body_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_plate")
        self.body_mid_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seg1_B_last")
        if self.body_mid_id == -1:
            self.body_mid_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seg2_body")
        self.body_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_6")
        self.slider_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_slid_M")
        self.slider_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slid_M")
        self.slider_qpos_adr = (
            int(model.jnt_qposadr[self.slider_joint]) if self.slider_joint != -1 else -1
        )
        self.tendon_controller = simulator.tendon_controller
        self.interface_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "interface_center"
        )
        self.tip_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "tip_center"
        )
        self.segments = [
            {"name": "Proximal / Wire 1-3", "wires": (1, 2, 3),
             "wire_angles": (0.0, 120.0, 240.0),
             "ctrl_vec": np.zeros(2), "motor_vec": np.zeros(2),
             "bend_cmd_deg": 0.0, "direction_deg": 0.0,
             "motor_bend_deg": 0.0, "motor_direction_deg": 0.0,
             "pull_mm": np.zeros(3), "real_angle": 0.0},
            {"name": "Distal / Wire 4-6", "wires": (4, 5, 6),
             "wire_angles": (60.0, 180.0, 300.0),
             "ctrl_vec": np.zeros(2), "motor_vec": np.zeros(2),
             "bend_cmd_deg": 0.0, "direction_deg": 0.0,
             "motor_bend_deg": 0.0, "motor_direction_deg": 0.0,
             "pull_mm": np.zeros(3), "real_angle": 0.0},
        ]

        self._open_ui_window()

    def _open_ui_window(self):
        if self.window_open:
            return
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)
        self.window_open = True

    def _close_ui_window(self):
        if not self.window_open:
            return
        try:
            cv2.destroyWindow(self.win_name)
        except Exception:
            pass
        self.window_open = False

    def set_input_mode(self, mode):
        if mode not in ("ui", "gamepad"):
            raise ValueError(f"不支持的仿真控制模式: {mode}")
        self.input_mode = mode
        self.dragging = None
        if mode == "ui":
            self._open_ui_window()
        else:
            self._close_ui_window()

    def connect_gamepad(self):
        self.gamepad_connected = self.gamepad.read() is not None
        return self.gamepad_connected

    def _update_gamepad_input(self, elapsed):
        state = self.gamepad.read()
        self.gamepad_connected = state is not None
        if state is None:
            self.segments[0]["ctrl_vec"][:] = 0.0
            self.segments[1]["ctrl_vec"][:] = 0.0
            if self.slider_qpos_adr >= 0:
                self.slider_command = float(self.data.qpos[self.slider_qpos_adr])
                self.slider_val = self.slider_command / 0.577
            return
        self.segments[0]["ctrl_vec"] = np.asarray(state["left"], dtype=float)
        self.segments[1]["ctrl_vec"] = np.asarray(state["right"], dtype=float)
        insertion_delta = state["right_trigger"] - state["left_trigger"]
        actual = float(self.data.qpos[self.slider_qpos_adr]) if self.slider_qpos_adr >= 0 else self.slider_command
        if abs(insertion_delta) < 0.05:
            self.slider_command = actual
        else:
            direction = math.copysign(1.0, insertion_delta)
            base = self.slider_command
            if (base - actual) * direction < 0.0:
                base = actual
            self.slider_command = float(np.clip(
                base + insertion_delta * MUJOCO_MAX_INSERTION_RATE * elapsed,
                max(0.0, actual - MUJOCO_MANUAL_INSERTION_LEAD_M), min(0.577, actual + MUJOCO_MANUAL_INSERTION_LEAD_M),
            ))
        self.slider_val = float(np.clip(self.slider_command / 0.577, 0.0, 1.0))

    def set_calibration_capture(self, enabled=True):
        self.calibration_capture = enabled

    def set_passive_debug_lock(self, enabled):
        self.passive_debug_locked = bool(enabled)
        state = self.simulator.passive_joint_controller.set_debug_lock(
            self.data, self.passive_debug_locked
        )
        self.simulator.passive_follower_controller.reset()
        return state

    def _passive_debug_button_hit(self, x, y):
        x1, y1, x2, y2 = self.passive_debug_button_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def set_lung_model_enabled(self, enabled):
        state = self.simulator.set_lung_model_enabled(enabled)
        self.lung_model_enabled = bool(state["enabled"])
        return state

    def _lung_toggle_button_hit(self, x, y):
        x1, y1, x2, y2 = self.lung_toggle_button_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _robot_pose_button_hit(self, x, y):
        x1, y1, x2, y2 = self.robot_pose_button_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._passive_debug_button_hit(x, y):
                self.dragging = None
                self.set_passive_debug_lock(not self.passive_debug_locked)
                return
            if self._lung_toggle_button_hit(x, y):
                self.dragging = None
                self.set_lung_model_enabled(not self.lung_model_enabled)
                return
            if self._robot_pose_button_hit(x, y):
                self.dragging = None
                self.simulator.show_robot_pose_control()
                return
            d1 = math.hypot(x - self.c1_center[0], y - self.c1_center[1])
            d2 = math.hypot(x - self.c2_center[0], y - self.c2_center[1])
            if d1 < MUJOCO_COMPASS_RADIUS + 20:
                self.dragging = 0
            elif d2 < MUJOCO_COMPASS_RADIUS + 20:
                self.dragging = 1
            elif y > 700:
                self.dragging = 2
                self._set_manual_slider(x)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging == 0:
                dx = x - self.c1_center[0]
                dy = -(y - self.c1_center[1])
                self.segments[0]["ctrl_vec"] = np.array([dx, dy]) / MUJOCO_COMPASS_RADIUS
            elif self.dragging == 1:
                dx = x - self.c2_center[0]
                dy = -(y - self.c2_center[1])
                self.segments[1]["ctrl_vec"] = np.array([dx, dy]) / MUJOCO_COMPASS_RADIUS
            elif self.dragging == 2:
                self._set_manual_slider(x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    def _set_manual_slider(self, x):
        requested = float(np.clip((x - 50) / 700.0, 0.0, 1.0))
        change = requested - self.slider_val
        if self.slider_qpos_adr >= 0 and abs(change) > 1e-9:
            actual = float(self.data.qpos[self.slider_qpos_adr])
            # Reverse from the measured position instead of chasing old demand.
            if change * (self.slider_command - actual) < 0.0:
                self.slider_command = actual
                requested = float(np.clip(actual / 0.577 + change, 0.0, 1.0))
        self.slider_val = requested

    def update_control(self):
        now = time.perf_counter()
        elapsed = float(np.clip(now - self.last_control_wall_time, 0.0, 0.1))
        self.last_control_wall_time = now
        if self.input_mode == "gamepad" and self.auto_action is None:
            self._update_gamepad_input(elapsed)
        if self.auto_action is not None:
            proximal = np.asarray(
                self.auto_action.get("steering_proximal", self.auto_action.get("steering", [0.0, 0.0])),
                dtype=float,
            )
            distal = np.asarray(
                self.auto_action.get("steering_distal", self.auto_action.get("steering", [0.0, 0.0])),
                dtype=float,
            )
            self.segments[0]["ctrl_vec"] = proximal
            self.segments[1]["ctrl_vec"] = distal
            insertion_delta = float(self.auto_action.get("insertion_delta", 0.0))
            if self.slider_qpos_adr >= 0:
                # Automatic insertion is an incremental closed loop around the
                # physical slider position. A bounded command lead supplies
                # useful insertion force, but a direction change starts from
                # the measured position so an unstick reversal acts at once.
                actual_slider_m = float(self.data.qpos[self.slider_qpos_adr])
                insertion_step_m = float(np.clip(
                    insertion_delta * 0.577,
                    -MUJOCO_AUTO_INSERTION_STEP_LIMIT_M,
                    MUJOCO_AUTO_INSERTION_STEP_LIMIT_M,
                ))
                actual_motion_m = (
                    0.0
                    if self.last_auto_slider_position_m is None
                    else abs(actual_slider_m - self.last_auto_slider_position_m)
                )
                requested_forward = insertion_delta > 1e-9
                if requested_forward and actual_motion_m < 0.00001:
                    self.auto_insertion_stall_frames = min(
                        MUJOCO_AUTO_STALL_RAMP_FRAMES,
                        self.auto_insertion_stall_frames + 1,
                    )
                elif actual_motion_m > 0.00003 or not requested_forward:
                    self.auto_insertion_stall_frames = max(
                        0, self.auto_insertion_stall_frames - 3
                    )
                self.last_auto_slider_position_m = actual_slider_m
                stall_ratio = float(np.clip(
                    self.auto_insertion_stall_frames
                    / MUJOCO_AUTO_STALL_RAMP_FRAMES,
                    0.0,
                    1.0,
                ))
                command_lead_m = (
                    MUJOCO_AUTO_LEAD_MIN_M
                    + stall_ratio
                    * (MUJOCO_AUTO_LEAD_MAX_M - MUJOCO_AUTO_LEAD_MIN_M)
                )
                direction_changed = bool(
                    insertion_delta * self.previous_auto_insertion_delta < 0.0
                )
                if direction_changed or abs(insertion_delta) <= 1e-9:
                    command_base_m = actual_slider_m
                else:
                    command_base_m = self.slider_command
                target_slider_m = float(np.clip(
                    command_base_m + insertion_step_m,
                    max(0.0, actual_slider_m - command_lead_m),
                    min(0.577, actual_slider_m + command_lead_m),
                ))
                self.slider_val = target_slider_m / 0.577
                self.slider_command = target_slider_m
                self.previous_auto_insertion_delta = insertion_delta
            else:
                self.slider_val = float(np.clip(
                    self.slider_val + insertion_delta, 0.0, 1.0
                ))
        else:
            self.previous_auto_insertion_delta = 0.0
            self.last_auto_slider_position_m = None
            self.auto_insertion_stall_frames = 0

        if self.slider_act != -1:
            target = self.slider_val * 0.577
            max_step = MUJOCO_MAX_INSERTION_RATE * elapsed
            if self.auto_action is None and self.input_mode != "gamepad":
                delta = np.clip(
                    target - self.slider_command,
                    -max_step,
                    max_step,
                )
                self.slider_command += float(delta)
                if self.slider_qpos_adr >= 0:
                    actual = float(self.data.qpos[self.slider_qpos_adr])
                    self.slider_command = float(np.clip(
                        self.slider_command, max(0.0, actual - MUJOCO_MANUAL_INSERTION_LEAD_M),
                        min(0.577, actual + MUJOCO_MANUAL_INSERTION_LEAD_M),
                    ))
            self.data.ctrl[self.slider_act] = self.slider_command

        desired_vectors = [
            self._clamp_compass_vector(segment["ctrl_vec"])
            for segment in self.segments
        ]
        if self.auto_action is None:
            desired_vectors = [self._manual_bend_vector(v) for v in desired_vectors]
            motor_vectors = self._manual_motor_vectors(*desired_vectors)
        else:
            self.distal_feedback_correction[:] = 0.0
            motor_vectors = desired_vectors

        for segment_index, (segment, desired, motor) in enumerate(zip(
            self.segments, desired_vectors, motor_vectors
        )):
            # Keep raw input for the knob; shaping it in-place compounds gain.
            segment["ctrl_vec"] = self._clamp_compass_vector(segment["ctrl_vec"])
            segment["motor_vec"] = motor
            desired_magnitude, desired_direction = self._vector_metrics(desired)
            motor_magnitude, motor_direction = self._vector_metrics(motor)
            segment["bend_cmd_deg"] = (
                self.tendon_controller.max_bend_deg * desired_magnitude
            )
            segment["direction_deg"] = desired_direction
            segment["motor_bend_deg"] = (
                self.tendon_controller.max_bend_deg * motor_magnitude
            )
            segment["motor_direction_deg"] = motor_direction
            self.tendon_controller.set_segment_vector(segment_index, motor)
            state = self.tendon_controller.segment_state(segment_index)
            segment["pull_mm"] = state["pull_mm"]

    @staticmethod
    def _manual_bend_vector(vector):
        vector = MujocoPhysicsControlPanel._clamp_compass_vector(vector)
        return vector * (0.5 * float(np.linalg.norm(vector)))

    @staticmethod
    def _clamp_compass_vector(vector):
        vector = np.asarray(vector, dtype=float).reshape(2)
        magnitude = float(np.linalg.norm(vector))
        return vector / magnitude if magnitude > 1.0 else vector

    @staticmethod
    def _vector_metrics(vector):
        vector = np.asarray(vector, dtype=float)
        magnitude = float(np.linalg.norm(vector))
        direction = (
            math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 360.0
            if magnitude > 1e-9 else 0.0
        )
        return magnitude, direction

    def _measured_distal_kinematic_vector(self):
        if self.interface_site_id < 0 or self.tip_site_id < 0:
            return np.zeros(2, dtype=float)
        interface_rotation = self.data.site_xmat[
            self.interface_site_id
        ].reshape(3, 3)
        tip_rotation = self.data.site_xmat[self.tip_site_id].reshape(3, 3)
        relative_rotation = interface_rotation.T @ tip_rotation
        rotation_vector = R.from_matrix(relative_rotation).as_rotvec()
        return self.distal_rotation_motor_map @ rotation_vector[1:3]

    def _manual_motor_vectors(self, proximal_desired, distal_desired):
        proximal_motor_desired = (
            self.manual_compass_motor_map @ proximal_desired
        )
        distal_motor_desired = self.manual_compass_motor_map @ distal_desired
        measured_distal = self._measured_distal_kinematic_vector()
        self.distal_feedback_correction += self.distal_feedback_integral_gain * (
            distal_motor_desired - measured_distal
        )
        correction_norm = float(np.linalg.norm(self.distal_feedback_correction))
        if correction_norm > self.distal_feedback_limit:
            self.distal_feedback_correction *= (
                self.distal_feedback_limit / correction_norm
            )
        proximal_feedforward = (
            self.distal_proximal_feedforward_map @ proximal_motor_desired
        )
        distal_motor = self.tendon_controller.clamp_segment_vector(
            1,
            proximal_feedforward
            + distal_motor_desired
            + self.distal_feedback_correction,
        )
        return proximal_motor_desired, distal_motor

    def apply_physics(self):
        if self.passive_debug_locked:
            self.simulator.passive_joint_controller.enforce_debug_lock(self.data)
            return
        self.simulator.passive_follower_controller.apply()

    def measure_real_angles(self):
        if self.body_base_id == -1 or self.body_mid_id == -1 or self.body_tip_id == -1:
            return
        q_base = self.data.xquat[self.body_base_id]
        q_mid = self.data.xquat[self.body_mid_id]
        q_tip = self.data.xquat[self.body_tip_id]
        self.segments[0]["real_angle"] = np.degrees(mujoco_quat_diff_angle(q_base, q_mid))
        self.segments[1]["real_angle"] = np.degrees(mujoco_quat_diff_angle(q_mid, q_tip))

    def draw_ui(self, cam_img=None):
        show_window = self.input_mode == "ui"
        if not show_window and not self.simulator.viewer_recording_enabled:
            return
        if show_window:
            self._open_ui_window()
        h, w = 840, 800
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (18, 18, 20)

        x1, y1, x2, y2 = self.passive_debug_button_rect
        button_color = (
            (43, 152, 91) if self.passive_debug_locked else (55, 58, 64)
        )
        border_color = (
            (91, 222, 145) if self.passive_debug_locked else (105, 108, 116)
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), button_color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2)
        cv2.putText(
            img, "PASSIVE DEBUG LOCK", (x1 + 13, y1 + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.51, (245, 247, 249), 1,
        )
        status = "ON  |  JOINTS RIGID" if self.passive_debug_locked else "OFF | FOLLOW MODE"
        cv2.putText(
            img, status, (x1 + 13, y1 + 53),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46,
            (225, 255, 235) if self.passive_debug_locked else (205, 208, 214), 1,
        )

        x1, y1, x2, y2 = self.lung_toggle_button_rect
        lung_color = (43, 152, 91) if self.lung_model_enabled else (55, 58, 64)
        lung_border = (91, 222, 145) if self.lung_model_enabled else (105, 108, 116)
        cv2.rectangle(img, (x1, y1), (x2, y2), lung_color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), lung_border, 2)
        cv2.putText(
            img, "LUNG MODEL", (x1 + 12, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 247, 249), 1,
        )
        lung_status = "ON  |  VISIBLE" if self.lung_model_enabled else "OFF | HIDDEN"
        cv2.putText(
            img, lung_status, (x1 + 12, y1 + 47),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44,
            (225, 255, 235) if self.lung_model_enabled else (205, 208, 214), 1,
        )

        x1, y1, x2, y2 = self.robot_pose_button_rect
        cv2.rectangle(img, (x1, y1), (x2, y2), (112, 75, 30), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (218, 151, 73), 2)
        cv2.putText(
            img, "ROBOT INITIAL POSE", (x1 + 10, y1 + 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (248, 248, 250), 1,
        )
        cv2.putText(
            img, "OPEN POSITION EDITOR", (x1 + 10, y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.37, (224, 235, 248), 1,
        )

        if cam_img is not None:
            disp_w, disp_h = 240, 240
            small_cam = cv2.resize(cam_img, (disp_w, disp_h))
            sx, sy = self.simulator.tip_camera_rect[:2]
            img[sy:sy + disp_h, sx:sx + disp_w] = small_cam
            cv2.rectangle(img, (sx, sy), (sx + disp_w, sy + disp_h), (120, 120, 128), 2)
            cv2.putText(img, "Tip Camera / Virtual Real Camera", (sx, sy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 235), 1)

        for i, center in enumerate([self.c1_center, self.c2_center]):
            cv2.circle(img, center, MUJOCO_COMPASS_RADIUS, (42, 42, 48), -1)
            cv2.circle(img, center, MUJOCO_COMPASS_RADIUS, (120, 120, 130), 2)
            cv2.putText(img, self.segments[i]["name"], (center[0] - 75, center[1] - MUJOCO_COMPASS_RADIUS - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 248), 2)
            wire_colour = (220, 40, 230) if i == 0 else (30, 130, 245)
            for wire, wire_angle in zip(
                self.segments[i]["wires"], self.segments[i]["wire_angles"]
            ):
                angle_rad = math.radians(wire_angle)
                marker = (
                    int(center[0] + math.cos(angle_rad) * (MUJOCO_COMPASS_RADIUS - 12)),
                    int(center[1] - math.sin(angle_rad) * (MUJOCO_COMPASS_RADIUS - 12)),
                )
                cv2.circle(img, marker, 7, wire_colour, -1)
                cv2.putText(img, f"W{wire}", (marker[0] - 11, marker[1] - 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (245, 245, 248), 1)
            cv2.putText(
                img,
                f"Cmd: {self.segments[i]['bend_cmd_deg']:.1f} deg @ {self.segments[i]['direction_deg']:.0f} deg",
                        (center[0] - 92, center[1] + MUJOCO_COMPASS_RADIUS + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 225), 1)
            cv2.putText(
                img,
                f"Motor: {self.segments[i]['motor_bend_deg']:.1f} deg @ {self.segments[i]['motor_direction_deg']:.0f} deg",
                        (center[0] - 92, center[1] + MUJOCO_COMPASS_RADIUS + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, wire_colour, 1)
            pulls = self.segments[i]["pull_mm"]
            cv2.putText(
                img,
                f"Pull mm: {pulls[0]:+.2f}/{pulls[1]:+.2f}/{pulls[2]:+.2f}",
                        (center[0] - 92, center[1] + MUJOCO_COMPASS_RADIUS + 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, wire_colour, 1)
            cv2.putText(img, f"Real: {self.segments[i]['real_angle']:.1f} deg",
                        (center[0] - 92, center[1] + MUJOCO_COMPASS_RADIUS + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 128), 2)
            ctrl_vec = self.segments[i]["ctrl_vec"]
            px = int(center[0] + ctrl_vec[0] * MUJOCO_COMPASS_RADIUS)
            py = int(center[1] - ctrl_vec[1] * MUJOCO_COMPASS_RADIUS)
            cv2.line(img, center, (px, py), (0, 255, 255), 2)
            cv2.circle(img, (px, py), 12, (0, 120, 255), -1)

        bar_y = 790
        cv2.line(img, (50, bar_y), (750, bar_y), (90, 90, 96), 6)
        sx = int(50 + self.slider_val * 700)
        cv2.circle(img, (sx, bar_y), 15, (0, 210, 110), -1)
        cv2.putText(img, f"Insertion Depth: {self.slider_val * 100:.0f}%",
                    (315, bar_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 235), 1)
        self.latest_frame_bgr = img.copy()
        if show_window:
            cv2.imshow(self.win_name, img)

    def close(self):
        self._close_ui_window()


class MujocoDualScopeSimulator:
    """Replace the real NDI/MC data stream with MuJoCo-generated poses."""

    PORT_SCOPE_1 = 10
    PORT_REF = 11
    PORT_SCOPE_2 = 12

    def __init__(self, xml_path, scale_to_mm=1000.0, open_viewer=False):
        if mujoco is None:
            raise RuntimeError("MuJoCo is not installed. Please install the mujoco Python package first.")

        self.xml_path = os.path.abspath(xml_path)
        self.scale_to_mm = scale_to_mm
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.t0 = time.time()
        self.step_count = 0
        self.viewer = None
        self.physics_panel = None
        self.robot_pose_window = None
        self.viewer_record_renderer = None
        self.viewer_record_size = None
        self.viewer_recording_enabled = False
        self.latest_viewer_frame_bgr = None
        self._viewer_recording_error = None
        # Match the square PyVista virtual viewport so both cameras have the
        # same horizontal and vertical field of view.
        self.renderer = mujoco.Renderer(self.model, height=480, width=480)
        self.tip_scene_option = mujoco.MjvOption()
        self.tip_scene_option.geomgroup[5] = 0
        self.tip_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, MUJOCO_TIP_CAMERA)
        if self.tip_camera_id != -1:
            self.model.cam_fovy[self.tip_camera_id] = MUJOCO_WIDE_FOVY
        self.tip_camera_rect = (280, 35, 240, 240)
        self.latest_tip_frame_bgr = None
        self.clicked_em_pos = np.zeros(3, dtype=float)
        self.pick_callback = None
        self.navigation_path_world_m = np.empty((0, 3), dtype=float)
        self.tdcr_real_points_base_m = np.empty((0, 3), dtype=float)
        self.tdcr_real_points_valid = np.zeros(0, dtype=bool)
        self.model_pick_requested = False
        self.viewer_pick_enabled = False
        self._viewer_pick_prev_down = False
        self._viewer_hwnd = None
        self._pick_scene = mujoco.MjvScene(self.model, 1000)
        self.contact_regularizer = ActiveTipContactRegularizer(self.model, self.data)
        self.passive_joint_controller = PassiveJointParameterController(self.model)
        self.passive_follower_controller = PassiveActiveFollowerController(
            self.model, self.data
        )
        self.tendon_controller = TendonCompassController(self.model, self.data)

        try:
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            self.model.opt.timestep = min(float(self.model.opt.timestep), 0.0002)
            self.model.opt.impratio = max(float(self.model.opt.impratio), 100.0)
        except Exception:
            pass

        self.body_ref_id = self._body_id(["base_plate", "base_link"])
        self.body_scope1_id = self._body_id(["end_6", "slider"])
        self.body_scope2_id = self._body_id(["seg1_B_last", "seg2_body"])
        self.keypoint_tip_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "tip_center",
        )
        self.keypoint_middle_body_id = self._body_id(
            ["seg1_B_last", "seg2_body"]
        )
        self.keypoint_connection_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "base_frame",
        )
        self.robot_root_body_id = self._body_id(["base_link"])
        self.robot_root_original_pos = self.model.body_pos[
            self.robot_root_body_id
        ].copy()
        self.robot_root_original_quat = self.model.body_quat[
            self.robot_root_body_id
        ].copy()
        self.robot_translation_mm = np.zeros(3, dtype=float)
        self.robot_rotation_rpy_deg = np.zeros(3, dtype=float)
        self.home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        self.lung_geom_id = self._find_lung_geom_id()
        self.lung_flex_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_FLEX,
            "bronchial_wall_nonconvex",
        )
        self.lung_model_enabled = True
        self.lung_flex_collision_masks = None
        if self.lung_geom_id >= 0:
            self.model.geom_group[self.lung_geom_id] = MUJOCO_LUNG_DISPLAY_GROUP
        if self.lung_flex_id >= 0:
            self.lung_flex_collision_masks = (
                int(self.model.flex_contype[self.lung_flex_id]),
                int(self.model.flex_conaffinity[self.lung_flex_id]),
            )
        self._apply_lung_render_state()
        self.slider_act_id = self._actuator_id(["act_slid_M"])
        self.cable_act_ids = {
            f"wire_{wire}": self._actuator_id([f"act_t{wire}"])
            for wire in range(1, 7)
        }

        mujoco.mj_forward(self.model, self.data)
        self.clicked_em_pos = self._camera_position_mm()
        if open_viewer:
            self.open_viewer()

    def open_viewer(self):
        if self.viewer is not None:
            return
        if mujoco_viewer is None:
            raise RuntimeError("mujoco.viewer is unavailable in this environment.")
        self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
        self._set_viewer_top_down_camera()
        self._apply_lung_render_state()
        self.physics_panel = MujocoPhysicsControlPanel(self.model, self.data, self)

    def _set_viewer_top_down_camera(self):
        if self.viewer is None:
            return
        try:
            mujoco.mj_forward(self.model, self.data)
            positions = np.asarray(self.data.geom_xpos, dtype=float)
            valid = np.isfinite(positions).all(axis=1)
            positions = positions[valid]
            center = np.mean(positions, axis=0) if len(positions) else np.zeros(3)
            extent = np.ptp(positions, axis=0) if len(positions) else np.ones(3)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.lookat[:] = center
            self.viewer.cam.azimuth = 270.0
            self.viewer.cam.elevation = -89.0
            self.viewer.cam.distance = max(0.8, float(np.max(extent)) * 1.35)
        except Exception as e:
            print(f"MuJoCo 俯视相机设置失败: {e}")

    def set_control_mode(self, mode):
        if self.physics_panel is None:
            return False
        self.physics_panel.set_input_mode(mode)
        return True

    def connect_gamepad(self):
        if self.physics_panel is None:
            return False
        return self.physics_panel.connect_gamepad()

    def set_auto_navigation_action(self, action):
        if self.physics_panel is not None:
            self.physics_panel.auto_action = action
            if action is None:
                for segment in self.physics_panel.segments:
                    segment["ctrl_vec"][:] = 0.0
                self.physics_panel.tendon_controller.reset()

    def gamepad_is_connected(self):
        return bool(self.physics_panel is not None and self.physics_panel.gamepad_connected)

    def set_passive_joint_scales(self, stiffness_scale, damping_scale):
        """Update passive compliance without touching active-tip parameters."""
        return self.passive_joint_controller.set_scales(
            stiffness_scale, damping_scale
        )

    def get_passive_joint_parameters(self):
        return self.passive_joint_controller.current_parameters()

    def get_robot_initial_pose(self):
        return {
            "translation_mm": self.robot_translation_mm.copy(),
            "rotation_rpy_deg": self.robot_rotation_rpy_deg.copy(),
            "world_position_mm": (
                self.model.body_pos[self.robot_root_body_id] * 1000.0
            ).copy(),
        }

    def set_robot_initial_pose(self, translation_mm, rotation_rpy_deg):
        """Rigidly reposition the complete robot assembly and return Home."""
        translation = np.asarray(translation_mm, dtype=float).reshape(3)
        rotation = np.asarray(rotation_rpy_deg, dtype=float).reshape(3)
        if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
            raise ValueError("机器人整体位姿必须是有限数值。")

        baseline_quat_xyzw = self.robot_root_original_quat[[1, 2, 3, 0]]
        baseline_rotation = R.from_quat(baseline_quat_xyzw).as_matrix()
        offset_rotation = R.from_euler(
            "xyz", rotation, degrees=True
        ).as_matrix()
        world_rotation = offset_rotation @ baseline_rotation
        world_quat_xyzw = R.from_matrix(world_rotation).as_quat()

        self.model.body_pos[self.robot_root_body_id] = (
            self.robot_root_original_pos + translation / 1000.0
        )
        self.model.body_quat[self.robot_root_body_id] = world_quat_xyzw[
            [3, 0, 1, 2]
        ]
        self.robot_translation_mm = translation.copy()
        self.robot_rotation_rpy_deg = rotation.copy()

        mujoco.mj_setConst(self.model, self.data)
        if self.home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(
                self.model, self.data, self.home_key_id
            )
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.contact_regularizer.reset()
        self.passive_follower_controller.reset()
        self.tendon_controller.reset()

        if self.physics_panel is not None:
            panel = self.physics_panel
            panel.auto_action = None
            panel.slider_val = 0.0
            panel.slider_command = 0.0
            panel.previous_auto_insertion_delta = 0.0
            panel.last_auto_slider_position_m = None
            panel.auto_insertion_stall_frames = 0
            panel.distal_feedback_correction[:] = 0.0
            for segment in panel.segments:
                segment["ctrl_vec"][:] = 0.0
                segment["motor_vec"][:] = 0.0
            if panel.passive_debug_locked:
                self.passive_joint_controller.enforce_debug_lock(self.data)

        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self._set_viewer_top_down_camera()
        return self.get_robot_initial_pose()

    def reset_robot_initial_pose(self):
        return self.set_robot_initial_pose(np.zeros(3), np.zeros(3))

    def show_robot_pose_control(self):
        if QtWidgets.QApplication.instance() is None:
            raise RuntimeError("机器人整体位姿界面需要 Qt 应用环境。")
        if self.robot_pose_window is None:
            self.robot_pose_window = RobotInitialPoseWindow(self)
        else:
            self.robot_pose_window.sync_from_simulator()
        self.robot_pose_window.show()
        self.robot_pose_window.raise_()
        self.robot_pose_window.activateWindow()
        return True

    def _apply_lung_render_state(self):
        visible = int(self.lung_model_enabled)
        self.tip_scene_option.geomgroup[MUJOCO_LUNG_DISPLAY_GROUP] = visible
        # The green non-convex flex is collision-only. Keep it hidden even
        # while its contact masks are enabled; only the red visual mesh is shown.
        self.tip_scene_option.flexgroup[MUJOCO_LUNG_DISPLAY_GROUP] = 0
        if self.viewer is not None:
            self.viewer.opt.geomgroup[MUJOCO_LUNG_DISPLAY_GROUP] = visible
            self.viewer.opt.flexgroup[MUJOCO_LUNG_DISPLAY_GROUP] = 0

    def set_lung_model_enabled(self, enabled):
        """Show/hide the lung and enable/disable its physical collision."""
        self.lung_model_enabled = bool(enabled)
        if self.lung_flex_id >= 0 and self.lung_flex_collision_masks is not None:
            contype, conaffinity = self.lung_flex_collision_masks
            self.model.flex_contype[self.lung_flex_id] = (
                contype if self.lung_model_enabled else 0
            )
            self.model.flex_conaffinity[self.lung_flex_id] = (
                conaffinity if self.lung_model_enabled else 0
            )
        if not self.lung_model_enabled:
            self.contact_regularizer.reset()
        self._apply_lung_render_state()
        mujoco.mj_forward(self.model, self.data)
        return {
            "enabled": self.lung_model_enabled,
            "collision_enabled": bool(
                self.lung_model_enabled and self.lung_flex_id >= 0
            ),
        }

    def _body_id(self, names):
        for name in names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                return bid
        raise RuntimeError(f"MuJoCo 模型缺少 body: {names}")

    def _actuator_id(self, names):
        for name in names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid != -1:
                return aid
        return -1

    def _find_lung_geom_id(self):
        mesh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, "visual_mesh")
        if mesh_id == -1:
            return -1
        matches = np.flatnonzero(np.asarray(self.model.geom_dataid) == mesh_id)
        return int(matches[0]) if len(matches) else -1

    def _lung_original_model_to_world(self):
        """Recover the original STL-to-world transform before MuJoCo mesh recentering."""
        if self.lung_geom_id == -1:
            raise RuntimeError("MuJoCo 模型中未找到 visual_mesh 肺模型。")
        mesh_id = int(self.model.geom_dataid[self.lung_geom_id])
        mesh_rotation = np.zeros(9, dtype=float)
        mujoco.mju_quat2Mat(mesh_rotation, self.model.mesh_quat[mesh_id])
        mesh_rotation = mesh_rotation.reshape(3, 3)
        mesh_position = np.asarray(self.model.mesh_pos[mesh_id], dtype=float)

        compiled_rotation = self.data.geom_xmat[self.lung_geom_id].reshape(3, 3).copy()
        compiled_position = self.data.geom_xpos[self.lung_geom_id].copy()
        original_rotation = compiled_rotation @ mesh_rotation.T
        original_position = compiled_position - original_rotation @ mesh_position
        return original_rotation, original_position

    def get_lung_auto_registration(self):
        """Return the exact MuJoCo-world-mm to lung-model-mm rigid transform."""
        mujoco.mj_forward(self.model, self.data)
        rotation_model_to_world, position_world_m = self._lung_original_model_to_world()
        position_world_mm = position_world_m * self.scale_to_mm
        rotation_world_to_model = rotation_model_to_world.T
        translation_world_to_model = -rotation_world_to_model @ position_world_mm
        return rotation_world_to_model, translation_world_to_model

    def model_points_to_world_m(self, points_model_mm):
        points = np.asarray(points_model_mm, dtype=float).reshape(-1, 3)
        rotation, position = self._lung_original_model_to_world()
        return (rotation @ (points / self.scale_to_mm).T).T + position

    def set_navigation_path_model_mm(self, points_model_mm):
        points = np.asarray(points_model_mm, dtype=float).reshape(-1, 3)
        if len(points) > 300:
            indices = np.linspace(0, len(points) - 1, 300).astype(int)
            points = points[indices]
        self.navigation_path_model_mm = points.copy()
        self.navigation_path_world_m = self.model_points_to_world_m(points)
        self.navigation_target_world_m = None
        self._refresh_viewer_overlays()

    def set_navigation_waypoint_state(self, remaining_model_mm, target_model_mm=None):
        points = np.asarray(remaining_model_mm, dtype=float).reshape(-1, 3)
        self.navigation_path_model_mm = points.copy()
        self.navigation_path_world_m = self.model_points_to_world_m(points) if len(points) else np.empty((0, 3))
        self.navigation_target_world_m = (
            self.model_points_to_world_m(np.asarray(target_model_mm, dtype=float).reshape(1, 3))[0]
            if target_model_mm is not None else None
        )
        self._refresh_viewer_overlays()

    def _refresh_navigation_path_world(self):
        if hasattr(self, "navigation_path_model_mm") and len(self.navigation_path_model_mm):
            self.navigation_path_world_m = self.model_points_to_world_m(self.navigation_path_model_mm)
            self._refresh_viewer_overlays()

    def set_tdcr_measurement_overlay(self, points_base_m, valid):
        """Display measured D435 points without changing MuJoCo physics."""
        self.tdcr_real_points_base_m = np.asarray(points_base_m, dtype=float).reshape(-1, 3).copy()
        self.tdcr_real_points_valid = np.asarray(valid, dtype=bool).reshape(-1).copy()
        self._refresh_viewer_overlays()

    def _body_matrix(self, body_id):
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        mat[:3, 3] = self.data.xpos[body_id] * self.scale_to_mm
        return mat

    def _site_matrix(self, site_id):
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = self.data.site_xmat[site_id].reshape(3, 3)
        mat[:3, 3] = self.data.site_xpos[site_id] * self.scale_to_mm
        return mat

    @staticmethod
    def _matrix_to_pose(matrix):
        matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
        return {
            "position_mm": matrix[:3, 3].copy(),
            "quaternion_xyzw": R.from_matrix(matrix[:3, :3]).as_quat(),
        }

    def get_keypoint_poses(self):
        poses = {}
        if self.keypoint_tip_site_id >= 0:
            poses["tip"] = self._matrix_to_pose(
                self._site_matrix(self.keypoint_tip_site_id)
            )
        if self.keypoint_middle_body_id >= 0:
            poses["middle_platform"] = self._matrix_to_pose(
                self._body_matrix(self.keypoint_middle_body_id)
            )
        if self.keypoint_connection_site_id >= 0:
            poses["master_slave_connection"] = self._matrix_to_pose(
                self._site_matrix(self.keypoint_connection_site_id)
            )
        return poses

    def _camera_matrix(self):
        mat = np.eye(4, dtype=float)
        if self.tip_camera_id != -1:
            camera_rotation = self.data.cam_xmat[self.tip_camera_id].reshape(3, 3)
            # MuJoCo cameras look along local -Z. Expose a right-handed probe
            # frame whose +Z axis follows the bronchoscope viewing direction.
            mat[:3, :3] = camera_rotation @ np.diag([-1.0, 1.0, -1.0])
            mat[:3, 3] = self.data.cam_xpos[self.tip_camera_id] * self.scale_to_mm
            return mat
        return self._body_matrix(self.body_scope1_id)

    def _seg1_probe_matrix(self):
        body_mat = self._body_matrix(self.body_scope2_id)
        body_rotation = body_mat[:3, :3]
        # Cable bodies extend along local +X. Reorder axes so probe +Z follows
        # the cable longitudinal direction while preserving a right-handed frame.
        body_mat[:3, :3] = body_rotation[:, [1, 2, 0]]
        return body_mat

    def _camera_position_mm(self):
        return self._camera_matrix()[:3, 3].copy()

    def render_tip_camera(self):
        if self.tip_camera_id == -1:
            return None
        self.renderer.update_scene(
            self.data,
            camera=MUJOCO_TIP_CAMERA,
            scene_option=self.tip_scene_option,
        )
        rgb = self.renderer.render()
        self.latest_tip_frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self.latest_tip_frame_bgr

    def set_screen_recording_enabled(self, enabled):
        self.viewer_recording_enabled = bool(enabled)
        self.latest_viewer_frame_bgr = None
        self._viewer_recording_error = None
        self.viewer_record_size = None
        if not self.viewer_recording_enabled and self.viewer_record_renderer is not None:
            try:
                self.viewer_record_renderer.close()
            except Exception:
                pass
            self.viewer_record_renderer = None

    def render_viewer_camera(self):
        if (
            not self.viewer_recording_enabled
            or self.viewer is None
            or not self.viewer.is_running()
        ):
            return None
        try:
            with self.viewer.lock():
                camera = copy.copy(self.viewer.cam)
                scene_option = copy.copy(self.viewer.opt)
                viewport = self.viewer.viewport
                viewport_width = int(getattr(viewport, "width", 0))
                viewport_height = int(getattr(viewport, "height", 0))
            if self.viewer_record_renderer is None:
                if viewport_width < 2 or viewport_height < 2:
                    return None
                record_width = viewport_width - (viewport_width % 2)
                record_height = viewport_height - (viewport_height % 2)
                self.model.vis.global_.offwidth = max(
                    int(self.model.vis.global_.offwidth),
                    record_width,
                )
                self.model.vis.global_.offheight = max(
                    int(self.model.vis.global_.offheight),
                    record_height,
                )
                self.viewer_record_renderer = mujoco.Renderer(
                    self.model,
                    height=record_height,
                    width=record_width,
                )
                self.viewer_record_size = (record_width, record_height)
            self.viewer_record_renderer.update_scene(
                self.data,
                camera=camera,
                scene_option=scene_option,
            )
            rgb = self.viewer_record_renderer.render()
            self.latest_viewer_frame_bgr = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            )
            self._viewer_recording_error = None
            return self.latest_viewer_frame_bgr
        except Exception as exc:
            error_text = str(exc)
            if error_text != self._viewer_recording_error:
                print(f"MuJoCo viewer recording frame failed: {error_text}")
                self._viewer_recording_error = error_text
            return None

    def set_clicked_em_position_mm(self, point_mm, notify=True):
        self.clicked_em_pos = np.asarray(point_mm, dtype=float).reshape(3)
        self._refresh_viewer_overlays()
        if notify and callable(self.pick_callback):
            self.pick_callback(self.clicked_em_pos.copy())
        return True

    def enable_viewer_point_picking(self, enabled=True):
        self.viewer_pick_enabled = bool(enabled)
        if self.physics_panel is not None:
            self.physics_panel.calibration_capture = bool(enabled)
        if self.viewer is not None:
            text = "EM calibration: click lung model in MuJoCo viewer" if enabled else ""
            try:
                self.viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ""))
            except Exception:
                pass

    def _find_mujoco_viewer_hwnd(self):
        if os.name != "nt":
            return None
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            current_pid = kernel32.GetCurrentProcessId()
            candidates = []
            fallback_candidates = []

            @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            def enum_proc(hwnd, lparam):
                if not user32.IsWindowVisible(hwnd):
                    return True
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value != current_pid:
                    return True
                length = user32.GetWindowTextLengthW(hwnd)
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, title, length + 1)
                title_text = title.value
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                area = max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)
                excluded = (
                    "Bronchoscope Control" in title_text
                    or "Control Center" in title_text
                    or "Pick" in title_text
                )
                if not excluded:
                    fallback_candidates.append((area, hwnd))
                if "MuJoCo" in title_text and not excluded:
                    candidates.append((area, hwnd))
                return True

            user32.EnumWindows(enum_proc, 0)
            if candidates:
                return max(candidates, key=lambda x: x[0])[1]
            if fallback_candidates:
                return max(fallback_candidates, key=lambda x: x[0])[1]
        except Exception:
            return None
        return None

    def _poll_mujoco_viewer_pick(self):
        if not self.viewer_pick_enabled or self.viewer is None or not self.viewer.is_running():
            self._viewer_pick_prev_down = False
            return
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            if self._viewer_hwnd is None or not user32.IsWindow(self._viewer_hwnd):
                self._viewer_hwnd = self._find_mujoco_viewer_hwnd()
            if self._viewer_hwnd is None:
                return

            is_down = bool(user32.GetAsyncKeyState(0x01) & 0x8000)
            if not is_down:
                self._viewer_pick_prev_down = False
                return
            if self._viewer_pick_prev_down:
                return
            self._viewer_pick_prev_down = True

            pt = wintypes.POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            rect = wintypes.RECT()
            user32.GetWindowRect(self._viewer_hwnd, ctypes.byref(rect))
            width = max(1, rect.right - rect.left)
            height = max(1, rect.bottom - rect.top)
            if not (rect.left <= pt.x <= rect.right and rect.top <= pt.y <= rect.bottom):
                return

            relx = (pt.x - rect.left) / width
            rely = 1.0 - ((pt.y - rect.top) / height)
            self._select_mujoco_viewer_point(relx, rely)
        except Exception:
            return

    def _select_mujoco_viewer_point(self, relx, rely):
        if self.viewer is None or self.viewer.viewport is None:
            return False
        viewport = self.viewer.viewport
        aspect = viewport.width / max(viewport.height, 1)
        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32)
        flexid = np.array([-1], dtype=np.int32)
        skinid = np.array([-1], dtype=np.int32)

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.viewer.opt,
            self.viewer.perturb,
            self.viewer.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._pick_scene,
        )
        bodyid = mujoco.mjv_select(
            self.model,
            self.data,
            self.viewer.opt,
            aspect,
            float(relx),
            float(rely),
            self._pick_scene,
            selpnt,
            geomid,
            flexid,
            skinid,
        )
        if bodyid == -1:
            return False
        self.set_clicked_em_position_mm(selpnt * self.scale_to_mm)
        return True

    def _refresh_viewer_overlays(self):
        if self.viewer is None or self.viewer.user_scn is None:
            return
        scn = self.viewer.user_scn
        scn.ngeom = 0
        mat = np.eye(3, dtype=np.float64).reshape(-1)

        path_rgba = np.array([0.0, 0.48, 1.0, 1.0], dtype=np.float32)
        path_size = np.array([0.0018, 0.0018, 0.0018], dtype=np.float64)
        for point in self.navigation_path_world_m:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                path_size,
                np.asarray(point, dtype=np.float64),
                mat,
                path_rgba,
            )
            scn.ngeom += 1

        target = getattr(self, "navigation_target_world_m", None)
        if target is not None and scn.ngeom < scn.maxgeom:
            target_size = np.array([0.005, 0.005, 0.005], dtype=np.float64)
            target_rgba = np.array([1.0, 0.78, 0.0, 1.0], dtype=np.float32)
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                target_size,
                np.asarray(target, dtype=np.float64),
                mat,
                target_rgba,
            )
            scn.ngeom += 1

        real_local = getattr(self, "tdcr_real_points_base_m", np.empty((0, 3)))
        real_valid = getattr(self, "tdcr_real_points_valid", np.zeros(0, dtype=bool))
        if len(real_local) and self.body_base_id >= 0:
            # base_frame is fixed to active_tdcr_base; use its site pose so
            # measured local coordinates follow insertion/passive motion only
            # for visualization. No qpos/qvel values are modified.
            base_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "base_frame"
            )
            site_ids = [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, f"measurement_kp_{i}"
                )
                for i in range(min(7, len(real_local)))
            ]
            if base_site_id >= 0 and all(site_id >= 0 for site_id in site_ids):
                base_rotation = self.data.site_xmat[base_site_id].reshape(3, 3)
                base_position = self.data.site_xpos[base_site_id]
                real_world = (base_rotation @ real_local.T).T + base_position
                real_rgba = np.array([1.0, 0.82, 0.0, 1.0], dtype=np.float32)
                error_rgba = np.array([1.0, 0.25, 0.1, 0.9], dtype=np.float32)
                point_size = np.array([0.0010, 0.0010, 0.0010], dtype=np.float64)
                for index, (point, is_valid) in enumerate(zip(real_world, real_valid)):
                    if not is_valid or not np.isfinite(point).all() or scn.ngeom >= scn.maxgeom:
                        continue
                    mujoco.mjv_initGeom(
                        scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        point_size, np.asarray(point, dtype=np.float64), mat, real_rgba,
                    )
                    scn.ngeom += 1
                    if index >= len(site_ids) or scn.ngeom >= scn.maxgeom:
                        continue
                    simulation_point = np.asarray(
                        self.data.site_xpos[site_ids[index]], dtype=np.float64
                    )
                    mujoco.mjv_initGeom(
                        scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE,
                        np.ones(3, dtype=np.float64), np.zeros(3, dtype=np.float64),
                        mat, error_rgba,
                    )
                    mujoco.mjv_connector(
                        scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 2,
                        np.asarray(point, dtype=np.float64), simulation_point,
                    )
                    scn.ngeom += 1

    def step(self, steps=8):
        if self.physics_panel is not None:
            self.physics_panel.update_control()
            for _ in range(MUJOCO_STEPS_PER_FRAME):
                self.data.qfrc_applied[:] = 0
                self.contact_regularizer.before_step()
                self.physics_panel.apply_physics()
                self.passive_joint_controller.enforce_base_guide(self.data)
                mujoco.mj_step(self.model, self.data)
                self.passive_joint_controller.enforce_base_guide(self.data)
                self.step_count += 1
            mujoco.mj_forward(self.model, self.data)
            self.physics_panel.measure_real_angles()
            self._refresh_navigation_path_world()
            cam_img = self.render_tip_camera()
            if self.viewer is not None and self.viewer.is_running():
                self._poll_mujoco_viewer_pick()
                self.viewer.sync()
            self.render_viewer_camera()
            self.physics_panel.draw_ui(cam_img)
            if self.physics_panel.input_mode == "ui":
                cv2.waitKey(1)
            return

        t = time.time() - self.t0

        if self.slider_act_id != -1:
            self.data.ctrl[self.slider_act_id] = 0.25 + 0.08 * math.sin(t * 0.45)

        # Gentle deterministic motion keeps the virtual sensors alive without real hardware.
        self.tendon_controller.set_segment_vector(
            0,
            [0.25 * math.sin(t * 0.65), 0.25 * math.cos(t * 0.52)],
        )
        self.tendon_controller.set_segment_vector(
            1,
            [0.18 * math.sin(t * 0.83 + 0.7), 0.18 * math.cos(t * 0.71 + 0.4)],
        )

        for _ in range(max(1, int(steps))):
            self.contact_regularizer.before_step()
            mujoco.mj_step(self.model, self.data)
            self.step_count += 1
        self._refresh_navigation_path_world()
        if self.viewer is not None and self.viewer.is_running():
            self._poll_mujoco_viewer_pick()
            self.viewer.sync()
        self.render_viewer_camera()
        self.render_tip_camera()

    def get_tools_dict(self):
        ref_mat = np.eye(4, dtype=float)
        ref_mat[:3, 3] = self.clicked_em_pos
        return {
            self.PORT_SCOPE_1: self._camera_matrix(),
            self.PORT_REF: ref_mat,
            self.PORT_SCOPE_2: self._seg1_probe_matrix(),
        }

    def get_axis_values(self):
        values = [0.0] * 7
        tendon_values = self.tendon_controller.commanded_pull_mm()
        for i, value in enumerate(tendon_values[:6]):
            values[i] = float(value)
        if self.slider_act_id != -1:
            values[6] = float(self.data.ctrl[self.slider_act_id] * self.scale_to_mm)
        elif self.model.nq > 0:
            values[6] = float(self.data.qpos[0] * self.scale_to_mm)
        return values

    def close(self):
        if self.robot_pose_window is not None:
            try:
                self.robot_pose_window.close()
            except Exception:
                pass
            self.robot_pose_window = None
        if self.physics_panel is not None:
            try:
                self.physics_panel.close()
            except Exception:
                pass
            self.physics_panel = None
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
        if self.viewer_record_renderer is not None:
            try:
                self.viewer_record_renderer.close()
            except Exception:
                pass
            self.viewer_record_renderer = None
            self.viewer_record_size = None
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

class DebugImageWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("深度图与 Mask 调试")
        self.resize(800, 400)  # 加宽窗口，便于同时显示深度图和 Mask。

        # 左侧显示深度图，右侧显示 Mask。
        self.layout = QHBoxLayout(self)
        self.label_depth = QLabel("Depth")
        self.label_mask = QLabel("Mask")

        self.label_depth.setAlignment(Qt.AlignCenter)
        self.label_mask.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.label_depth)
        self.layout.addWidget(self.label_mask)

    def update_image(self, mask_arr, depth_arr=None):
        # 说明已清理。
        if mask_arr is not None:
            h, w = mask_arr.shape
            img_mask = QImage(mask_arr.data, w, h, w, QImage.Format_Grayscale8)
            self.label_mask.setPixmap(QPixmap.fromImage(img_mask).scaled(
                self.label_mask.size(), Qt.KeepAspectRatio))

        # 说明已清理。
        if depth_arr is not None:
            # 深度图为 float，需要归一化到 0~255 后显示。
            d_min = np.min(depth_arr)
            d_max = np.max(depth_arr)
            # print(f"深度图范围: Min={d_min:.2f} mm, Max={d_max:.2f} mm")
            if d_max - d_min < 1e-3: d_max = d_min + 1.0

            # 归一化。
            depth_norm = ((depth_arr - d_min) / (d_max - d_min) * 255).astype(np.uint8)

            # 说明已清理。
            # JET 色图用于突出深浅变化。
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            h, w, ch = depth_color.shape
            img_depth = QImage(depth_color.data, w, h, ch * w, QImage.Format_BGR888)
            self.label_depth.setPixmap(QPixmap.fromImage(img_depth).scaled(
                self.label_depth.size(), Qt.KeepAspectRatio))


class NavigationTargetWindow(QtWidgets.QWidget):
    """Independent tip-camera waypoint observation window."""

    def __init__(self):
        super().__init__(None, Qt.Window)
        self.setWindowTitle("自动导航目标观察")
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(720, 780)
        self.setMinimumSize(520, 600)
        self.setStyleSheet("""
            QWidget { background: #f5f5f7; color: #1d1d1f; }
            QLabel#targetTitle { font-size: 20px; font-weight: 650; padding: 8px; }
            QLabel#targetImage { background: #101114; border-radius: 14px; }
            QLabel#pathOverview { background: #101114; border-radius: 12px; }
            QLabel#targetStatus {
                background: white; border: 1px solid #d8d8dc; border-radius: 12px;
                padding: 12px; font-size: 14px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        title = QLabel("当前路径点视觉引导")
        title.setObjectName("targetTitle")
        title.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel("等待摄像头画面")
        self.image_label.setObjectName("targetImage")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(480, 480)
        self.overview_label = QLabel("全部路径点概览")
        self.overview_label.setObjectName("pathOverview")
        self.overview_label.setAlignment(Qt.AlignCenter)
        self.overview_label.setMinimumHeight(150)
        self.overview_label.setMaximumHeight(190)
        self.status_label = QLabel("当前路径点: N/A")
        self.status_label.setObjectName("targetStatus")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.overview_label)
        layout.addWidget(self.status_label)

    def closeEvent(self, event):
        self.hide()
        event.ignore()

    def update_guidance(self, frame_bgr, action):
        if frame_bgr is None or action is None:
            return
        try:
            frame = np.asarray(frame_bgr).copy()
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            cv2.drawMarker(frame, center, (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
            fov_y = math.radians(MUJOCO_WIDE_FOVY)
            focal = height / (2.0 * math.tan(fov_y / 2.0))

            # Draw every planned waypoint that currently projects into the tip
            # camera. The active target is orange; all other points are blue.
            all_local = np.asarray(action.get("all_path_local_mm", []), dtype=float)
            target_index = int(action.get("target_index", 0))
            projected_points = []
            if all_local.ndim == 2 and all_local.shape[1:] == (3,):
                for point_index, point_local in enumerate(all_local):
                    if point_local[2] <= 1e-6:
                        projected_points.append(None)
                        continue
                    point_u = int(round(center[0] + focal * point_local[0] / point_local[2]))
                    point_v = int(round(center[1] - focal * point_local[1] / point_local[2]))
                    if 0 <= point_u < width and 0 <= point_v < height:
                        projected_points.append((point_u, point_v))
                    else:
                        projected_points.append(None)

                for point_index in range(len(projected_points) - 1):
                    first = projected_points[point_index]
                    second = projected_points[point_index + 1]
                    if first is not None and second is not None:
                        color = (210, 130, 40) if point_index < target_index else (255, 170, 40)
                        cv2.line(frame, first, second, color, 1, cv2.LINE_AA)

                for point_index, projected in enumerate(projected_points):
                    if projected is None:
                        continue
                    if point_index == target_index:
                        cv2.circle(frame, projected, 10, (0, 170, 255), 3, cv2.LINE_AA)
                        cv2.circle(frame, projected, 3, (0, 170, 255), -1, cv2.LINE_AA)
                    else:
                        color = (150, 150, 150) if point_index < target_index else (255, 170, 40)
                        radius = 2 if point_index < target_index else 4
                        cv2.circle(frame, projected, radius, color, -1, cv2.LINE_AA)

                # Complete path overview: PCA fits the entire 3D path into a
                # compact 2D map, so points outside the camera FOV remain visible.
                overview_height, overview_width = 170, 680
                overview = np.full((overview_height, overview_width, 3), (20, 21, 24), dtype=np.uint8)
                centered_path = all_local - np.mean(all_local, axis=0, keepdims=True)
                try:
                    _, _, basis = np.linalg.svd(centered_path, full_matrices=False)
                    path_2d = centered_path @ basis[:2].T
                except Exception:
                    path_2d = centered_path[:, :2]
                low = np.min(path_2d, axis=0)
                high = np.max(path_2d, axis=0)
                span = np.maximum(high - low, 1e-6)
                margin = np.array([30.0, 24.0])
                scale = float(np.min((np.array([overview_width, overview_height]) - margin * 2.0) / span))
                overview_points = (path_2d - low) * scale + margin
                overview_points[:, 1] = overview_height - overview_points[:, 1]
                overview_points = np.rint(overview_points).astype(int)
                for point_index in range(len(overview_points) - 1):
                    cv2.line(overview, tuple(overview_points[point_index]),
                             tuple(overview_points[point_index + 1]), (110, 105, 95), 2, cv2.LINE_AA)
                for point_index, overview_point in enumerate(overview_points):
                    if point_index == target_index:
                        cv2.circle(overview, tuple(overview_point), 8, (0, 170, 255), -1, cv2.LINE_AA)
                    else:
                        color = (100, 100, 100) if point_index < target_index else (255, 170, 40)
                        cv2.circle(overview, tuple(overview_point), 3, color, -1, cv2.LINE_AA)
                overview_rgb = cv2.cvtColor(overview, cv2.COLOR_BGR2RGB)
                overview_image = QImage(
                    overview_rgb.data, overview_width, overview_height,
                    overview_rgb.strides[0], QImage.Format_RGB888,
                ).copy()
                self.overview_label.setPixmap(QPixmap.fromImage(overview_image).scaled(
                    self.overview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))

            local = np.asarray(action.get("waypoint_local_mm", [0.0, 0.0, -1.0]), dtype=float)
            visible = local.shape == (3,) and local[2] > 1e-6
            target_in_front = visible
            if target_in_front:
                u = int(round(center[0] + focal * local[0] / local[2]))
                v = int(round(center[1] - focal * local[1] / local[2]))
                visible = 0 <= u < width and 0 <= v < height
                if visible:
                    cv2.line(frame, center, (u, v), (0, 190, 255), 2)
                    cv2.circle(frame, (u, v), 12, (0, 190, 255), 3)
                    cv2.circle(frame, (u, v), 3, (0, 190, 255), -1)
                else:
                    direction = np.asarray([u - center[0], v - center[1]], dtype=float)
                    direction /= max(float(np.linalg.norm(direction)), 1e-9)
                    edge_radius = 0.42 * min(width, height)
                    edge = (
                        int(round(center[0] + direction[0] * edge_radius)),
                        int(round(center[1] + direction[1] * edge_radius)),
                    )
                    cv2.arrowedLine(frame, center, edge, (0, 165, 255), 3, tipLength=0.22)
            else:
                # A rear target is shown as a red turn-around cue instead of
                # disappearing from the observation window.
                cv2.circle(frame, center, 44, (0, 0, 255), 3)
                cv2.putText(frame, "TARGET BEHIND", (center[0] - 92, center[1] - 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
            self.image_label.setPixmap(QPixmap.fromImage(image).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.status_label.setText(
                f"路径点 {action.get('target_index', 0) + 1} | "
                f"距离 {action.get('distance_to_target_mm', 0.0):.2f} mm | "
                f"视图偏差 {action.get('visual_error_norm', 0.0):.3f} | "
                f"速度 {action.get('estimated_speed_mm_per_frame', 0.0):.2f} mm/帧 | "
                f"方位误差 {action.get('relative_bearing_error_deg', 0.0):.1f}°\n"
                f"横向误差 {action.get('lateral_distance_mm', 0.0):.2f} mm | "
                f"{action.get('servo_phase', '跟踪')} | "
                f"{'画面内' if visible else ('画面外' if target_in_front else '目标在后方')}"
            )
        except Exception as e:
            self.status_label.setText(f"导航观察窗口更新失败: {e}")


class CentroidOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None, color=QColor(255, 0, 0, 200)):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self._centroids = []  # 瀛樺偍 [(u1, v1), (u2, v2)...]
        self.color = color  # 保存颜色。

    def set_centroids(self, points_list):
        """Set normalized centroid points."""
        self._centroids = points_list if points_list else []
        self.update()

    def paintEvent(self, event):
        if not self._centroids:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.color)

        w = self.width()
        h = self.height()
        r = max(4, min(w, h) * 0.015)  # 鍗婂緞

        # 寰幆缁樺埗鎵€鏈夌偣
        for (u, v) in self._centroids:
            x = u * w
            y = v * h
            painter.drawEllipse(QtCore.QPointF(x, y), r, r)

            # 鍙€夛細鐢讳釜缂栧彿
            # painter.setPen(Qt.white)
            # painter.drawText(QtCore.QPointF(x+5, y), "Target")
            # painter.setPen(Qt.NoPen)

        painter.end()


class NavigationOverlay(QtWidgets.QWidget):
    """Navigation overlay for target guidance."""




    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

        self.target_uv = None  # 鐩爣鍧愭爣 (0~1)
        self.status = "LOST"  # LOCKED, GHOST, LOST

    def update_status(self, target_uv, status):
        self.target_uv = target_uv
        self.status = status
        self.update()

    def paintEvent(self, event):
        # 说明已清理。
        if self.status == "LOST" or self.target_uv is None:
            # 说明已清理。
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        tx, ty = self.target_uv[0] * w, self.target_uv[1] * h

        if self.status == "LOCKED":
            # 说明已清理。
            pen = QPen(QColor(0, 255, 0))  # 缁胯壊
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            s = 40  # 妗嗗ぇ灏?
            # 鐢绘
            painter.drawRect(int(tx - s / 2), int(ty - s / 2), s, s)
            # 鐢诲崄瀛?
            painter.drawLine(int(tx), int(ty - s / 2), int(tx), int(ty + s / 2))
            painter.drawLine(int(tx - s / 2), int(ty), int(tx + s / 2), int(ty))

            # 鏂囧瓧
            painter.drawText(int(tx + s / 2) + 5, int(ty), "LOCKED")

        elif self.status == "GHOST":
            # 说明已清理。
            pen = QPen(QColor(0, 122, 255))
            pen.setWidth(3)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            s = 60
            painter.drawRect(int(tx - s / 2), int(ty - s / 2), s, s)

            # 鏂囧瓧
            painter.setPen(QColor(0, 122, 255))
            painter.drawText(int(tx + s / 2) + 5, int(ty), "GHOST")

        painter.end()

class SegmentationWorker(QtCore.QObject):
    # 输出完整二值 mask 和候选质心列表，用于 UI 显示与调试。
    resultReady = QtCore.pyqtSignal(object, list)
    errorOccurred = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, weights_path, frame_queue, img_size=400, parent=None):
        super().__init__(parent)
        self.weights_path = weights_path
        self.frame_queue = frame_queue
        self.img_size = img_size
        self._running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.model = None

    @QtCore.pyqtSlot()
    def process_loop(self):
        # 延迟加载分割模型，避免启动时阻塞主界面。
        if self.model is None:
            try:
                print(f"正在加载分割模型: {self.weights_path} ...")
                self.model = Unet(3, 1)
                state = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.to(self.device)
                self.model.eval()
                print("模型加载成功，开始推理循环。")
            except Exception as e:
                print(f"[严重错误] 模型加载失败: {e}")
                import traceback
                traceback.print_exc()
                self.errorOccurred.emit(str(e))
                self.finished.emit()
                return

        self._running = True

        while self._running:
            start_time = time.time()

            try:
                # 1. 获取图像帧。
                frame_bgr = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                # 2. 执行分割推理与质心分析。
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb)
                inp = self.transform(img_pil).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(inp)
                    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

                mask = (prob > 0.05).astype(np.uint8) * 255

                # 说明已清理。
                num_labels, labels, stats, centroids_data = cv2.connectedComponentsWithStats(mask, connectivity=8)
                found_centroids = []
                h_img, w_img = mask.shape
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area < 20: continue
                    cx, cy = centroids_data[i]
                    found_centroids.append((cx / w_img, cy / h_img))

                # 3. 鍙戦€佺粨鏋?
                self.resultReady.emit(mask, found_centroids)

            except Exception as e:
                print(f"Error: {e}")

            # 说明已清理。
            # 说明已清理。
            elapsed = time.time() - start_time
            sleep_time = 0.05 - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.finished.emit()

    def stop(self):
        self._running = False


class SquareInteractorContainer(QtWidgets.QWidget):
    """Container that keeps QtInteractor square."""



    def __init__(self, parent=None):
        super().__init__(parent)
        self.interactor = QtInteractor(self)
        self.interactor.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                      QtWidgets.QSizePolicy.Fixed)

        # 说明已清理。
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

    def resizeEvent(self, event):
        w = self.width()
        h = self.height()
        side = min(w, h)
        x = (w - side) // 2
        y = (h - side) // 2

        self.interactor.setGeometry(x, y, side, side)
        super().resizeEvent(event)


class SquareLabelContainer(QtWidgets.QWidget):
    """Container that keeps an inner QLabel square."""




    def __init__(self, parent=None):
        super().__init__(parent)
        # 璁╁鍣ㄦ湰韬敖鍙兘濉弧甯冨眬
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

        # 鍐呴儴鐪熸鐨勬樉绀烘帶浠?
        self.label = QLabel("摄像头未启动", self)
        self.label.setAlignment(Qt.AlignCenter)
        # 说明已清理。
        self.label.setStyleSheet("background-color: black;")

    def resizeEvent(self, event):
        """Keep the inner label square on resize."""
        w = self.width()
        h = self.height()
        side = min(w, h)

        # 璁＄畻灞呬腑鍧愭爣
        x = (w - side) // 2
        y = (h - side) // 2

        # 寮哄埗璁剧疆 Label 鐨勫ぇ灏忓拰浣嶇疆
        self.label.setGeometry(x, y, side, side)

        super().resizeEvent(event)


class MujocoLungPickWindow(QtWidgets.QWidget):
    def __init__(self, mesh, point_callback, parent=None):
        super().__init__(parent)
        self.mesh = mesh
        self.point_callback = point_callback
        self.current_point = None
        self.point_actor = None
        self.setWindowTitle("MuJoCo 肺部电磁点采集")
        self.resize(920, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QLabel("在 MuJoCo 肺部模型上点击，更新当前虚拟电磁定位点。")
        header.setObjectName("sectionTitle")
        layout.addWidget(header)

        self.coord_label = QLabel("当前点: 未采集")
        self.coord_label.setObjectName("statusPill")
        layout.addWidget(self.coord_label)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter, 1)
        self.plotter.add_mesh(
            self.mesh,
            color="#D98989",
            opacity=0.72,
            smooth_shading=True,
            specular=0.25,
            specular_power=18,
        )
        self.plotter.add_axes()
        self.plotter.reset_camera()
        self._enable_picking()

    def _enable_picking(self):
        self.plotter.enable_point_picking(
            callback=self._on_point_picked,
            show_message=True,
            use_picker=True,
            show_point=False,
        )

    def _on_point_picked(self, point, *args):
        if point is None:
            return
        point = np.asarray(point, dtype=float)
        self.current_point = point
        self.coord_label.setText("当前点: X={:.3f}, Y={:.3f}, Z={:.3f}".format(point[0], point[1], point[2]))

        if self.point_actor is not None:
            try:
                self.plotter.remove_actor(self.point_actor)
            except Exception:
                pass
        sphere = pv.Sphere(radius=2.4, center=point)
        self.point_actor = self.plotter.add_mesh(
            sphere,
            color="#00C7BE",
            render_points_as_spheres=True,
        )
        self.plotter.render()
        self.point_callback(point)

    def closeEvent(self, event):
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)


def apply_product_style(app):
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget {
            font-family: "SF Pro Display", "Segoe UI", "Microsoft YaHei UI";
            font-size: 13px;
            color: #1D1D1F;
        }
        QMainWindow, QDialog {
            background: #F5F5F7;
        }
        QLabel#heroTitle {
            font-size: 30px;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #111113;
        }
        QLabel#heroSubtitle {
            font-size: 14px;
            line-height: 1.4;
            color: #6E6E73;
        }
        QLabel#sectionTitle {
            font-size: 15px;
            font-weight: 700;
            color: #1D1D1F;
        }
        QLabel#statusPill {
            padding: 7px 12px;
            border-radius: 14px;
            background: #E8F2FF;
            color: #0066CC;
            font-weight: 600;
        }
        QTabWidget::pane {
            border: 1px solid #E5E5EA;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.92);
            top: -1px;
        }
        QTabBar::tab {
            min-width: 86px;
            padding: 9px 18px;
            margin: 4px 3px;
            border-radius: 12px;
            background: transparent;
            color: #6E6E73;
            font-weight: 600;
        }
        QTabBar::tab:selected {
            background: #FFFFFF;
            color: #111113;
            border: 1px solid #E5E5EA;
        }
        QPushButton {
            min-height: 30px;
            padding: 7px 14px;
            border-radius: 10px;
            border: 1px solid #D8D8DE;
            background: #FFFFFF;
            color: #1D1D1F;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #F9F9FB;
            border-color: #BFC0C8;
        }
        QPushButton:pressed {
            background: #ECECF0;
        }
        QPushButton:checked, QPushButton#primaryButton {
            background: #0071E3;
            border-color: #0071E3;
            color: white;
        }
        QPushButton#dangerButton {
            background: #FFF1F0;
            border-color: #FFCCC7;
            color: #C41D1D;
        }
        QPushButton#modeCard {
            text-align: left;
            min-height: 120px;
            padding: 18px;
            border-radius: 22px;
            border: 1px solid #E5E5EA;
            background: #FFFFFF;
            font-size: 15px;
        }
        QPushButton#modeCard:checked {
            border: 2px solid #0071E3;
            background: #F0F7FF;
            color: #003A75;
        }
        QComboBox, QLineEdit {
            min-height: 30px;
            padding: 5px 10px;
            border-radius: 9px;
            border: 1px solid #D8D8DE;
            background: #FFFFFF;
            selection-background-color: #0071E3;
        }
        QCheckBox {
            spacing: 7px;
            color: #3A3A3C;
        }
        QSplitter::handle {
            background: #E5E5EA;
        }
        QFrame#glassCard {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #E5E5EA;
            border-radius: 24px;
        }
        QFrame#toolbarCard {
            background: #FFFFFF;
            border: 1px solid #E5E5EA;
            border-radius: 18px;
        }
    """)


class StartupEnvironmentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_environment = "real"
        self.selected_mujoco_xml = DEFAULT_MUJOCO_XML
        self.setObjectName("startupDialog")
        self.setWindowTitle("支气管镜 VLA 控制中心")
        self.setModal(True)
        self.resize(760, 500)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(18)

        card = QtWidgets.QFrame()
        card.setObjectName("glassCard")
        shadow = QtWidgets.QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 14)
        shadow.setColor(QColor(0, 0, 0, 38))
        card.setGraphicsEffect(shadow)
        root.addWidget(card)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(34, 32, 34, 30)
        layout.setSpacing(18)

        title = QLabel("支气管镜控制中心")
        title.setObjectName("heroTitle")
        subtitle = QLabel("请选择运行环境。真实环境使用 NDI、MC 控制器和实体相机；MuJoCo 仿真环境会替换传感器数据并打开仿真窗口。")
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        cards = QHBoxLayout()
        cards.setSpacing(16)
        self.real_btn = QPushButton("真实环境\n\n连接 NDI 跟踪器、MC 控制器和实体相机。")
        self.real_btn.setObjectName("modeCard")
        self.real_btn.setCheckable(True)
        self.real_btn.setChecked(True)
        self.sim_btn = QPushButton("MuJoCo 仿真\n\n使用 MuJoCo 模型作为虚拟传感器，并打开仿真窗口。")
        self.sim_btn.setObjectName("modeCard")
        self.sim_btn.setCheckable(True)
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.real_btn)
        group.addButton(self.sim_btn)
        self.real_btn.clicked.connect(lambda: self._set_mode("real"))
        self.sim_btn.clicked.connect(lambda: self._set_mode("simulation"))
        cards.addWidget(self.real_btn)
        cards.addWidget(self.sim_btn)
        layout.addLayout(cards)

        model_row = QHBoxLayout()
        model_label = QLabel("仿真模型")
        model_label.setObjectName("sectionTitle")
        self.model_combo = QComboBox()
        self._populate_models()
        model_row.addWidget(model_label)
        model_row.addWidget(self.model_combo, 1)
        layout.addLayout(model_row)

        actions = QHBoxLayout()
        actions.addStretch()
        self.cancel_btn = QPushButton("退出")
        self.cancel_btn.clicked.connect(self.reject)
        self.start_btn = QPushButton("进入工作台")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.clicked.connect(self.accept)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.start_btn)
        layout.addLayout(actions)
        self._set_mode("real")

    def _populate_models(self):
        self.model_combo.clear()
        meshes_dir = os.path.join(PROJECT_ROOT, "meshes")
        if os.path.isdir(meshes_dir):
            for name in sorted(os.listdir(meshes_dir)):
                if name.lower().endswith(".xml"):
                    path = os.path.join(meshes_dir, name)
                    self.model_combo.addItem(name, path)
        if self.model_combo.count() == 0 and os.path.exists(DEFAULT_MUJOCO_XML):
            self.model_combo.addItem(os.path.basename(DEFAULT_MUJOCO_XML), DEFAULT_MUJOCO_XML)
        default_index = self.model_combo.findData(DEFAULT_MUJOCO_XML)
        if default_index >= 0:
            self.model_combo.setCurrentIndex(default_index)

    def _set_mode(self, mode):
        self.selected_environment = mode
        self.real_btn.setChecked(mode == "real")
        self.sim_btn.setChecked(mode == "simulation")
        self.model_combo.setEnabled(mode == "simulation")

    def accept(self):
        self.selected_mujoco_xml = self.model_combo.currentData() or DEFAULT_MUJOCO_XML
        super().accept()


_REAL_ENVIRONMENT_WINDOW_CLASS = None


def load_real_environment_window_class():
    """Load the physical-hardware workbench while leaving simulation untouched."""
    global _REAL_ENVIRONMENT_WINDOW_CLASS
    if _REAL_ENVIRONMENT_WINDOW_CLASS is not None:
        return _REAL_ENVIRONMENT_WINDOW_CLASS
    if not os.path.isfile(REAL_ENVIRONMENT_SCRIPT):
        raise FileNotFoundError(f"真实环境界面文件不存在: {REAL_ENVIRONMENT_SCRIPT}")
    module_name = "brnchus_robot_real_dual_probe_ui"
    specification = importlib.util.spec_from_file_location(
        module_name, REAL_ENVIRONMENT_SCRIPT
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"无法加载真实环境界面: {REAL_ENVIRONMENT_SCRIPT}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    window_class = getattr(module, "MainWindow", None)
    if window_class is None:
        raise AttributeError("真实环境界面缺少 MainWindow 类")
    _REAL_ENVIRONMENT_WINDOW_CLASS = window_class
    return window_class


class MainWindow(
    DualCameraRecordingMixin,
    SimulationAutomationMixin,
    SimulationRuntimeMixin,
    QtWidgets.QMainWindow,
):
    def __init__(self, data_queue: Queue, initial_environment="real", initial_mujoco_xml=None):
        super().__init__()
        self._is_closing = False
        self.setWindowTitle("支气管镜控制中心")
        self.setMinimumSize(1320, 860)
        self.data_queue = data_queue
        self.initial_environment = initial_environment
        self.is_simulation_mode = False
        self.mujoco_simulator = None
        self.mujoco_xml_path = initial_mujoco_xml or DEFAULT_MUJOCO_XML
        self.intrinsic_camera_matrix = np.array([[229.59784258, 0, 200.52304737],
                                                 [0, 227.66142868, 166.61526572],
                                                 [0, 0, 1]])
        self.distortion_coeffs = np.array([-0.31049457, 0.40897277, -0.00838951, 0.00158787, -0.38224321])
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(50)
        self.position = []
        self.direction = []
        self.R = []
        self.t = []
        self.scope_pos = None
        self.scope_dir = None
        self.ref_pos = None
        self.virtual_position = []
        self.calibration_ready = False
        self.calibration_rmse = None
        self.calibration_max_error = None
        self.calibration_residuals = []
        self.calibration_threshold_mm = 3.0
        self.auto_simulation_mapping_ready = False
        self.sim_em_points = []
        self.sim_em_pick_index = 0
        self.sim_em_pick_active = False
        self.sim_candidate_em_point = None
        self.all_points = []
        self.smoothpath = []
        self.endpoint = [50.776901, 144.23911, 39.526905]
        self.ready_control = False
        self.axis_history_len = 500
        # 说明已清理。
        self.axis_histories = [deque(maxlen=self.axis_history_len) for _ in range(6)]
        # 说明已清理。
        for h in self.axis_histories:
            for _ in range(self.axis_history_len):
                h.append(0.0)
        # ======================================

        self.cap = None
        self.cap = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)

        # 说明已清理。
        self.seg_thread = None
        self.seg_worker = None
        self.seg_running = False
        self.latest_real_frame_bgr = None
        self.latest_segmentation_mask = None
        self.latest_tools_dict = None
        self.auto_nav_enabled = False
        self.auto_nav_state = "stopped"
        self.auto_nav_action = None
        self.auto_nav_tracking_path = None
        self.auto_nav_global_profile = None
        self.auto_nav_target_index = 0
        self.auto_nav_nearest_index = 0
        self.auto_nav_path_distance = 0.0
        self.auto_nav_passed_count = 0
        self.auto_nav_progress = 0.0
        self.auto_nav_action_cache = deque(maxlen=2000)
        self.auto_nav_filtered_steering = np.zeros(2, dtype=float)
        self.auto_nav_proximal_command = np.zeros(2, dtype=float)
        self.auto_nav_distal_command = np.zeros(2, dtype=float)
        self.auto_nav_previous_heading_error = np.zeros(2, dtype=float)
        self.auto_nav_pid_integral = np.zeros(2, dtype=float)
        self.auto_nav_pid_derivative = np.zeros(2, dtype=float)
        self.auto_nav_pid_output_filtered = np.zeros(2, dtype=float)
        self.auto_nav_pid_last_time = None
        self.auto_nav_steering_reversal_frames = 0
        self.auto_nav_filtered_distal_assist = 0.0
        self.auto_nav_elastic_release_active = False
        self.auto_nav_elastic_release_frames = 0
        self.auto_nav_curve_command = np.zeros(2, dtype=float)
        self.auto_nav_curve_direction = np.zeros(2, dtype=float)
        self.auto_nav_curve_release_frames = 0
        self.auto_nav_curve_release_cooldown = 0
        self.auto_nav_control_basis = None
        self.auto_nav_last_curve_heading_error = None
        self.auto_nav_curve_error_rise_frames = 0
        self.auto_nav_filtered_position = None
        self.auto_nav_filtered_forward = None
        self.auto_nav_waypoint_reached_frames = 0
        self.auto_nav_waypoint_controller_index = -1
        self.auto_nav_elastic_bend_direction = np.zeros(2, dtype=float)
        self.auto_nav_elastic_release_countdown = 0
        self.auto_nav_elastic_settle_frames = 0
        self.auto_nav_straight_mode = True
        self.auto_nav_filtered_path_curve = 0.0
        self.auto_nav_straight_correction_active = False
        self.auto_nav_last_turn_direction = np.zeros(2, dtype=float)
        self.auto_nav_committed_turn_direction = np.zeros(2, dtype=float)
        self.auto_nav_turn_severity = 0.0
        self.auto_nav_turn_active = False
        self.auto_nav_control_frames = 0
        self.auto_nav_servo_target_index = -1
        self.auto_nav_target_transition_frames = 0
        self.auto_nav_filtered_bearing = np.zeros(2, dtype=float)
        self.auto_nav_last_bearing_angle_deg = None
        self.auto_nav_bearing_improvement = 0.0
        self.auto_nav_action_hold_countdown = 0
        self.auto_nav_held_proximal_target = np.zeros(2, dtype=float)
        self.auto_nav_held_distal_target = np.zeros(2, dtype=float)
        self.auto_nav_curve_memory = 0.0
        self.auto_nav_guidance_direction = None
        self.auto_nav_straight_error_frames = 0
        self.auto_nav_straight_clear_frames = 0
        self.auto_nav_motion_mode = "forward"
        self.auto_nav_behind_frames = 0
        self.auto_nav_front_frames = 0
        self.auto_nav_bend_gain_scale = 1.0
        self.auto_nav_forward_reengage_frames = 0
        self.auto_nav_filtered_speed = 0.0
        self.auto_nav_last_insertion_step = 0.0
        self.auto_nav_last_position = None
        self.auto_nav_stall_frames = 0
        self.auto_nav_v2 = AdaptiveRecedingHorizonNavigator()
        self.navigation_target_window = None
        self.vla_recording = False
        self.vla_episode_dir = None
        self.vla_sample_index = 0
        self.vla_jsonl_file = None
        self.vla_csv_file = None
        self.vla_csv_writer = None
        self.setup_ui()
        self.virt_opening_actor = None
        self.virt_opening_poly = None
        # 说明已清理。
        self.is_mc_connected = False
        self.has_reached_target = False
        # =======================================================
        # 说明已清理。
        # 轨迹与视频记录状态。
        self.is_recording_trajectory = False
        self.trajectory_data = []
        self.recording_start_time = 0.0
        self.save_dir = os.path.join(PROJECT_ROOT, "实际轨迹")
        self.real_video_writer = None
        self.virt_video_writer = None
        self.real_video_path = None
        self.virt_video_path = None
        self.real_video_fps = 20.0
        self.virt_video_fps = 4.0
        self.initialize_dual_camera_recording()
        self.current_axis_values = [0.0] * 7
        # Keep controller demand and encoder measurement distinct. DPOS is a
        # demand position; MPOS is the measured position when the drive/API
        # exposes it. current_axis_values prefers MPOS and falls back to DPOS.
        self.current_axis_demand_values = [float("nan")] * 7
        self.current_axis_measured_values = [float("nan")] * 7
        self.current_axis_measurement_valid = [False] * 7
        self.d435_capture_window = None

        # 预先创建保存目录。
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
                print(f"已创建轨迹保存目录: {self.save_dir}")
            except Exception as e:
                print(f"无法创建保存目录: {e}")
        if self.initial_environment == "simulation":
            QtCore.QTimer.singleShot(250, self._start_initial_simulation)


    def setup_ui(self):
        # 左侧显示 3D 模型，右侧放置标定、控制与仿真控件。
        splitter =QSplitter(QtCore.Qt.Horizontal)

        # 左侧 PyVista 交互窗口。
        self.plotter = QtInteractor(self)
        self.add_example_mesh()
        splitter.addWidget(self.plotter)



        # 右侧控制面板。
        tab_widget = QTabWidget()
        # Tab 1: 标定相关控件。
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setAlignment(QtCore.Qt.AlignTop)
        tab_widget.addTab(tab1, "标定")
        # COM 口。
        self.com_label = QtWidgets.QLabel("选择 COM 口")
        self.comComboBox = QComboBox()
        self.refresh_ports()
        tab1_layout.addWidget(self.com_label)
        tab1_layout.addWidget(self.comComboBox)

        #connection
        self.ndi_connect_button = QPushButton("连接 NDI")
        self.ndi_connect_button.clicked.connect(self.toggle_connection)
        tab1_layout.addWidget(self.ndi_connect_button)

        # 选择主界面 3D 肺部模型上的 3 个虚拟标记点。
        group_box = QtWidgets.QGroupBox("模型点选择")
        group_layout = QtWidgets.QHBoxLayout(group_box)
        self.select_points_button = QPushButton("选择模型点")
        self.select_points_button.setMinimumSize(60, 100)
        self.select_points_button.clicked.connect(self.select_points)
        group_layout.addWidget(self.select_points_button)

        # 3x3 坐标网格，显示 3 个点的 X/Y/Z。
        grid_widget = QWidget()
        grid_layout = QVBoxLayout(grid_widget)

        # 使用 QGridLayout 排列 3 行 3 列。
        coord_grid = QtWidgets.QGridLayout()
        self.coord_labels = []

        # 创建 3 个点的 X/Y/Z 标签。
        for i in range(3):
            row_labels = []
            for j, coord in enumerate(["X", "Y", "Z"]):
                label = QLabel(f"P{i + 1}_{coord}: N/A")
                coord_grid.addWidget(label, i, j)
                row_labels.append(label)
            self.coord_labels.append(row_labels)

        # 将坐标网格放入容器。
        grid_container = QWidget()
        grid_container.setLayout(coord_grid)
        group_layout.addWidget(grid_container)

        # 加入标定页。
        tab1_layout.addWidget(group_box)

        group_box2 = QtWidgets.QGroupBox("电磁定位点")
        group_layout2 = QtWidgets.QHBoxLayout(group_box2)

        # 记录 3 个电磁定位点。
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        self.button1 = QPushButton("记录标记点 1")
        self.button2 = QPushButton("记录标记点 2")
        self.button3 = QPushButton("记录标记点 3")
        self.button1.clicked.connect(self.record_points_1)
        self.button2.clicked.connect(self.record_points_2)
        self.button3.clicked.connect(self.record_points_3)
        button_layout.addWidget(self.button1)
        button_layout.addWidget(self.button2)
        button_layout.addWidget(self.button3)
        group_layout2.addWidget(button_container)

        # 3x3 坐标网格，显示电磁点坐标。
        coord_grid2 = QtWidgets.QGridLayout()
        self.coord_labels2 = []
        for i in range(3):
            row_labels = []
            for j, label_text in enumerate(["X", "Y", "Z"]):
                label = QLabel(f"P{i + 1}_{label_text}: 0")
                coord_grid2.addWidget(label, i, j)
                row_labels.append(label)
            self.coord_labels2.append(row_labels)
        grid_container2 = QWidget()
        grid_container2.setLayout(coord_grid2)
        group_layout2.addWidget(grid_container2)

        # 加入标定页。
        tab1_layout.addWidget(group_box2)
        # 配准并计算刚体变换矩阵。
        self.comput_button = QPushButton("配准")
        self.comput_button.clicked.connect(self.Kabsch_computer)
        tab1_layout.addWidget(self.comput_button)
        coord_display_layout = QHBoxLayout()
        self.actual_point_label = QLabel("实际点")
        self.actual_point_label.setAlignment(QtCore.Qt.AlignCenter)
        coord_display_layout.addWidget(self.actual_point_label)
        self.x_label = QLabel("X: 0.00")
        self.y_label = QLabel("Y: 0.00")
        self.z_label = QLabel("Z: 0.00")

        # 居中显示坐标。
        self.x_label.setAlignment(QtCore.Qt.AlignCenter)
        self.y_label.setAlignment(QtCore.Qt.AlignCenter)
        self.z_label.setAlignment(QtCore.Qt.AlignCenter)

        coord_display_layout.addWidget(self.x_label)
        coord_display_layout.addWidget(self.y_label)
        coord_display_layout.addWidget(self.z_label)

        # 显示真实点坐标。
        tab1_layout.addLayout(coord_display_layout)
        coord_display_layout2 = QHBoxLayout()

        self.actual_point_label2 = QLabel("虚拟点")
        self.actual_point_label2.setAlignment(QtCore.Qt.AlignCenter)
        coord_display_layout2.addWidget(self.actual_point_label2)
        self.xx_label = QLabel("X: 0.00")
        self.yy_label = QLabel("Y: 0.00")
        self.zz_label = QLabel("Z: 0.00")

        # 居中显示坐标。
        self.xx_label.setAlignment(QtCore.Qt.AlignCenter)
        self.yy_label.setAlignment(QtCore.Qt.AlignCenter)
        self.zz_label.setAlignment(QtCore.Qt.AlignCenter)

        coord_display_layout2.addWidget(self.xx_label)
        coord_display_layout2.addWidget(self.yy_label)
        coord_display_layout2.addWidget(self.zz_label)

        # 显示虚拟点坐标。
        tab1_layout.addLayout(coord_display_layout2)
        # 终点选择。
        self.endpoint_button = QPushButton("选择导航终点")
        self.endpoint_button.clicked.connect(self.select_endpoint_mode)
        tab1_layout.addWidget(self.endpoint_button)
        # 路径规划。
        self.pathplan_button = QPushButton("路径规划")
        self.pathplan_button.clicked.connect(self.route_plan)
        tab1_layout.addWidget(self.pathplan_button)

        # Tab 2: 控制。
        tab2 = QtWidgets.QWidget()
        tab_widget.addTab(tab2, "控制")
        tab2_layout = QtWidgets.QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(10, 10, 10, 10)

        # 相机选择与分割调试工具栏。
        cam_select_layout = QHBoxLayout()
        cam_select_layout.addWidget(QLabel("选择相机:"))

        self.camera_combo = QComboBox()
        self.cameras = QCameraInfo.availableCameras()
        for cam in self.cameras:
            self.camera_combo.addItem(cam.description())
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        cam_select_layout.addWidget(self.camera_combo)
        self.seg_button = QPushButton("开始分割")
        self.seg_button.setCheckable(True)
        self.seg_button.clicked.connect(self.toggle_segmentation)
        cam_select_layout.addWidget(self.seg_button)
        from PyQt5.QtWidgets import QCheckBox
        self.debug_cb = QCheckBox("显示 Mask 调试")
        self.debug_cb.setChecked(False)
        self.debug_cb.toggled.connect(self.on_debug_toggled)
        cam_select_layout.addWidget(self.debug_cb)

        self.show_all_centroids_cb = QCheckBox("显示所有红点")
        self.show_all_centroids_cb.setChecked(False)
        cam_select_layout.addWidget(self.show_all_centroids_cb)

        self.virtual_debug_cb = QCheckBox("显示虚拟 Mask")
        self.virtual_debug_cb.setChecked(False)
        self.virtual_debug_cb.toggled.connect(self.on_virtual_debug_toggled)
        cam_select_layout.addWidget(self.virtual_debug_cb)

        self.record_data_cb = QtWidgets.QCheckBox("记录数据")
        self.record_data_cb.setChecked(False)
        self.record_data_cb.toggled.connect(self.toggle_recording)
        cam_select_layout.addWidget(self.record_data_cb)

        cam_select_layout.addStretch()
        tab2_layout.addLayout(cam_select_layout)

        env_card = QtWidgets.QFrame()
        env_card.setObjectName("toolbarCard")
        env_card_layout = QHBoxLayout(env_card)
        env_card_layout.setContentsMargins(16, 14, 16, 14)
        env_card_layout.setSpacing(14)

        env_title_box = QVBoxLayout()
        env_title = QLabel("运行环境")
        env_title.setObjectName("sectionTitle")
        env_subtitle = QLabel("选择真实环境或 MuJoCo 仿真数据源。")
        env_subtitle.setObjectName("heroSubtitle")
        env_subtitle.setWordWrap(True)
        env_title_box.addWidget(env_title)
        env_title_box.addWidget(env_subtitle)
        env_card_layout.addLayout(env_title_box, 2)

        sim_select_layout = QHBoxLayout()
        sim_select_layout.addWidget(QLabel("仿真模型:"))
        self.sim_model_combo = QComboBox()
        self._populate_mujoco_models()
        sim_select_layout.addWidget(self.sim_model_combo, 1)

        self.sim_toggle_btn = QPushButton("启用 MuJoCo 仿真")
        self.sim_toggle_btn.setCheckable(True)
        self.sim_toggle_btn.toggled.connect(self.toggle_mujoco_simulation)
        sim_select_layout.addWidget(self.sim_toggle_btn)

        self.sim_status_label = QLabel("当前数据源: 真实环境")
        self.sim_status_label.setObjectName("statusPill")
        sim_select_layout.addWidget(self.sim_status_label)
        env_card_layout.addLayout(sim_select_layout, 3)
        tab2_layout.addWidget(env_card)

        d435_card = QtWidgets.QGroupBox("D435 连续体三维关键点")
        d435_layout = QHBoxLayout(d435_card)
        self.d435_open_button = QPushButton("打开 D435 采集与MuJoCo对齐")
        self.d435_open_button.clicked.connect(self.open_d435_capture_window)
        d435_layout.addWidget(self.d435_open_button)
        self.d435_status_label = QLabel("7点: 0/7/14/21/28/35/42 mm | 相机未启动")
        self.d435_status_label.setWordWrap(True)
        d435_layout.addWidget(self.d435_status_label, 1)
        tab2_layout.addWidget(d435_card)

        self.debug_window = None
        self.virt_debug_window = None
        self.enable_virtual_seg = False
        # 说明已清理。
        top_layout = QtWidgets.QHBoxLayout()
        virtual_view_container = QtWidgets.QWidget()
        virtual_view_layout = QtWidgets.QVBoxLayout(virtual_view_container)
        virtual_view_layout.setContentsMargins(0, 0, 0, 0)
        virtual_view_layout.setSpacing(2)

        # 说明已清理。
        self.virtual_square = SquareInteractorContainer(virtual_view_container)

        # 说明已清理。
        self.virtual_view = self.virtual_square.interactor

        # 说明已清理。
        self.virtual_view.add_mesh(
            self.mesh,
            color="#c06c6c",
            specular=0.3,
            specular_power=20,
            diffuse=0.3,
            ambient=0.6,
            smooth_shading=True,
            opacity=1.0
        )
        self.virt_green_points = vtk.vtkPoints()
        self.virt_green_cells = vtk.vtkCellArray()
        # 说明已清理。
        self.virt_green_poly = vtk.vtkPolyData()
        self.virt_green_poly.SetPoints(self.virt_green_points)
        self.virt_green_poly.SetVerts(self.virt_green_cells)
        # 说明已清理。
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(self.virt_green_poly)

        # 说明已清理。
        self.virt_green_actor = vtk.vtkActor2D()
        self.virt_green_actor.SetMapper(mapper)

        # 说明已清理。
        self.virt_green_actor.GetProperty().SetColor(0, 1, 0)  # R=0, G=1, B=0
        self.virt_green_actor.GetProperty().SetPointSize(10)

        # 说明已清理。
        # 说明已清理。
        self.virtual_view.renderer.AddActor2D(self.virt_green_actor)
        # 1. 鏁版嵁缁撴瀯
        self.virt_box_points = vtk.vtkPoints()
        self.virt_box_cells = vtk.vtkCellArray()
        self.virt_box_poly = vtk.vtkPolyData()
        self.virt_box_poly.SetPoints(self.virt_box_points)
        self.virt_box_poly.SetLines(self.virt_box_cells)

        # 说明已清理。
        box_mapper = vtk.vtkPolyDataMapper2D()
        box_mapper.SetInputData(self.virt_box_poly)

        # 3. Actor
        self.virt_box_actor = vtk.vtkActor2D()
        self.virt_box_actor.SetMapper(box_mapper)

        # 说明已清理。
        self.virt_box_actor.GetProperty().SetColor(1, 0, 0)
        self.virt_box_actor.GetProperty().SetLineWidth(3)

        # 加入渲染器。
        self.virtual_view.renderer.AddActor2D(self.virt_box_actor)
        self.debug_guide_points = vtk.vtkPoints()
        self.debug_guide_cells = vtk.vtkCellArray()
        self.debug_guide_poly = vtk.vtkPolyData()
        self.debug_guide_poly.SetPoints(self.debug_guide_points)
        self.debug_guide_poly.SetVerts(self.debug_guide_cells)

        mapper_blue = vtk.vtkPolyDataMapper2D()
        mapper_blue.SetInputData(self.debug_guide_poly)

        self.debug_guide_actor = vtk.vtkActor2D()
        self.debug_guide_actor.SetMapper(mapper_blue)
        self.debug_guide_actor.GetProperty().SetColor(0, 0, 1)  # 钃濊壊
        self.debug_guide_actor.GetProperty().SetPointSize(15)

        self.virtual_view.renderer.AddActor2D(self.debug_guide_actor)

        self.sync_virtual_camera_with_intrinsics()

        virtual_label = QtWidgets.QLabel("虚拟视角")
        virtual_label.setAlignment(Qt.AlignCenter)

        # 说明已清理。
        virtual_view_layout.addWidget(self.virtual_square)
        virtual_view_layout.addWidget(virtual_label)

        # 说明已清理。
        actual_view_container = QtWidgets.QWidget()
        actual_view_layout = QtWidgets.QVBoxLayout(actual_view_container)
        actual_view_layout.setContentsMargins(0, 0, 0, 0)
        actual_view_layout.setSpacing(0)

        # 说明已清理。
        self.actual_square_container = SquareLabelContainer()

        # 说明已清理。
        # 说明已清理。
        self.actual_view = self.actual_square_container.label

        # 鎶婂鍣ㄥ姞鍏ュ竷灞€
        actual_view_layout.addWidget(self.actual_square_container)
        # 说明已清理。
        self.centroid_overlay = CentroidOverlay(self.actual_view)
        self.centroid_overlay.raise_()
        self.nav_overlay = NavigationOverlay(self.actual_view)
        self.nav_overlay.raise_()
        actual_label = QtWidgets.QLabel("实际视角")
        actual_label.setAlignment(Qt.AlignCenter)
        actual_view_layout.addWidget(actual_label)

        top_layout.addWidget(virtual_view_container)
        top_layout.addWidget(actual_view_container)
        top_layout.setStretchFactor(virtual_view_container, 1)
        top_layout.setStretchFactor(actual_view_container, 1)

        tab2_layout.addLayout(top_layout, 3)

        control_card = QtWidgets.QFrame()
        control_card.setObjectName("toolbarCard")
        control_layout = QHBoxLayout(control_card)
        control_layout.setContentsMargins(16, 10, 16, 10)
        control_layout.setSpacing(12)
        control_layout.addWidget(QLabel("仿真控制模式"))

        self.sim_control_combo = QComboBox()
        self.sim_control_combo.addItem("双罗盘 UI 控制", "ui")
        self.sim_control_combo.addItem("手柄控制", "gamepad")
        self.sim_control_combo.currentIndexChanged.connect(self.on_sim_control_mode_changed)
        self.sim_control_combo.setEnabled(False)
        control_layout.addWidget(self.sim_control_combo)

        self.gamepad_connect_btn = QPushButton("检测并连接手柄")
        self.gamepad_connect_btn.clicked.connect(self.connect_sim_gamepad)
        self.gamepad_connect_btn.setEnabled(False)
        self.gamepad_connect_btn.setVisible(False)
        control_layout.addWidget(self.gamepad_connect_btn)

        self.sim_control_status = QLabel("启动 MuJoCo 后可选择控制方式")
        self.sim_control_status.setObjectName("statusPill")
        control_layout.addWidget(self.sim_control_status, 1)
        tab2_layout.addWidget(control_card)

        self.passive_joint_group = QtWidgets.QGroupBox("被动段柔顺性（仅 MuJoCo）")
        passive_layout = QHBoxLayout(self.passive_joint_group)
        passive_layout.setContentsMargins(14, 8, 14, 8)
        passive_layout.setSpacing(10)

        passive_layout.addWidget(QLabel("刚度倍率（小 = 易弯）"))
        self.passive_stiffness_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.passive_stiffness_slider.setRange(10, 500)
        self.passive_stiffness_slider.setValue(100)
        self.passive_stiffness_slider.setSingleStep(5)
        self.passive_stiffness_slider.setPageStep(25)
        self.passive_stiffness_slider.setMinimumWidth(180)
        self.passive_stiffness_slider.setToolTip(
            "缩放 29 个 cable_stiffJ_* 球关节的等效弯曲刚度。"
        )
        passive_layout.addWidget(self.passive_stiffness_slider)

        self.passive_stiffness_value = QLabel("1.00×  (500 N·m/rad)")
        self.passive_stiffness_value.setMinimumWidth(165)
        passive_layout.addWidget(self.passive_stiffness_value)

        passive_layout.addWidget(QLabel("阻尼倍率"))
        self.passive_damping_spin = QtWidgets.QDoubleSpinBox()
        self.passive_damping_spin.setRange(0.10, 5.00)
        self.passive_damping_spin.setDecimals(2)
        self.passive_damping_spin.setSingleStep(0.10)
        self.passive_damping_spin.setValue(1.00)
        self.passive_damping_spin.setSuffix(" ×")
        self.passive_damping_spin.setToolTip(
            "调小响应更灵活但更易振荡；调大可抑制被动段摆动。"
        )
        passive_layout.addWidget(self.passive_damping_spin)

        self.passive_joint_reset_btn = QPushButton("恢复默认")
        passive_layout.addWidget(self.passive_joint_reset_btn)
        self.passive_joint_status = QLabel("启动 MuJoCo 后生效")
        self.passive_joint_status.setObjectName("statusPill")
        passive_layout.addWidget(self.passive_joint_status, 1)
        self.passive_joint_group.setEnabled(False)

        self.passive_stiffness_slider.valueChanged.connect(
            self.on_passive_joint_parameters_changed
        )
        self.passive_damping_spin.valueChanged.connect(
            self.on_passive_joint_parameters_changed
        )
        self.passive_joint_reset_btn.clicked.connect(
            self.reset_passive_joint_parameters
        )
        tab2_layout.addWidget(self.passive_joint_group)

        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("轴位置"))

        self.axis_pos_labels = []
        for i in range(7):
            lbl = QLabel(f"Axis {i}: N/A")
            self.axis_pos_labels.append(lbl)
            axis_layout.addWidget(lbl)

        tab2_layout.addLayout(axis_layout)
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('w')
        self.graph_widget.showGrid(x=True, y=True, alpha=0.3)
        self.graph_widget.setTitle("Axis position realtime curves", color="k", size="12pt")
        self.graph_widget.setLabel('left', 'Position', color='k')
        self.graph_widget.setLabel('bottom', 'Time', color='k')
        self.graph_widget.setMinimumHeight(240)
        self.graph_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 说明已清理。
        # 说明已清理。
        self.legend = self.graph_widget.addLegend(offset=(10, 10), colCount=6)

        # 说明已清理。
        self.legend.layout.setVerticalSpacing(0)

        # 说明已清理。
        self.legend.layout.setContentsMargins(5, 5, 5, 5)

        # 说明已清理。
        self.curves = []
        # 说明已清理。
        colors = [
            (255, 0, 0),
            (0, 200, 0),
            (0, 0, 255),
            (255, 165, 0),
            (128, 0, 128),
            (0, 200, 200)
        ]

        for i in range(6):
            # 说明已清理。
            pen = pg.mkPen(color=colors[i], width=2)
            curve = self.graph_widget.plot(name=f"Axis {i}", pen=pen)
            name_str = f'<span style="font-size: 9pt">Axis {i}</span>'
            self.curves.append(curve)

        # 3. 灏嗙粯鍥炬帶浠舵坊鍔犲埌甯冨眬搴曢儴
        # 浣跨敤 setStretch 璁╁浘琛ㄥ崰鎹墿浣欑殑鎵€鏈夌┖鐧藉尯鍩?
        tab2_layout.addWidget(self.graph_widget, 2)
        tab2_layout.setStretchFactor(self.graph_widget, 2)

        self._setup_automation_tab(tab_widget)

        # =======================================================
        # 说明已清理。
        splitter.addWidget(tab_widget)
        self.setCentralWidget(splitter)
        if self.initial_environment == "simulation":
            self.actual_view.setText("MuJoCo 仿真模式\n真实相机数据流已停用")
        elif len(self.cameras) > 0:
            self.on_camera_changed(0)
    def on_mc_connect(self):
        ip = self.mc_ip_edit.text().strip()
        try:
            # 说明已清理。
            self.mc_connection = create_mc_connection(ip)
            self.mc_connection.OpenConnection()
            self.is_mc_connected = True
            QMessageBox.information(self, "MC 连接", f"已成功连接到 {ip}")
            # 鎸夐挳鐘舵€佹洿鏂?
            self.mc_connect_btn.setEnabled(False)
            self.mc_disconnect_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "MC 连接失败", str(e))
            self.is_mc_connected = False
            if hasattr(self, "mc_connection"):
                try: self.mc_connection.CloseConnection()
                except: pass

    def on_mc_disconnect(self):
        try:
            if hasattr(self, "mc_connection"):
                self.mc_connection.CloseConnection()
                QMessageBox.information(self, "MC 断开", "已断开 MC 连接")
                self.is_mc_connected = False
        except Exception as e:
            QMessageBox.warning(self, "MC 断开异常", str(e))
        finally:
            # 说明已清理。
            self.is_mc_connected = False
            self.mc_connect_btn.setEnabled(True)
            self.mc_disconnect_btn.setEnabled(False)

    def on_wdog_enable(self):
        """浣胯兘 WDOG"""
        try:
            self.mc_connection.SetSystemParameter_WDOG(True)
            QMessageBox.information(self, "WDOG", "已使能 WDOG")
            self.wdog_enable_btn.setEnabled(False)
            self.wdog_disable_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "WDOG 错误", str(e))

    def on_wdog_disable(self):
        """鍏抽棴 WDOG"""
        try:
            self.mc_connection.SetSystemParameter_WDOG(False)
            QMessageBox.information(self, "WDOG", "已关闭 WDOG")
        except Exception as e:
            QMessageBox.critical(self, "WDOG 错误", str(e))
        finally:
            self.wdog_enable_btn.setEnabled(True)
            self.wdog_disable_btn.setEnabled(False)

    def toggle_recording(self, checked: bool):
        """
        说明已清理。
        """
        if checked:
            # 说明已清理。
            self.is_recording_trajectory = True
            self.trajectory_data = []
            self.recording_start_time = datetime.now().timestamp()
            self.current_axis_values = [0.0] * 7
            self.current_axis_demand_values = [float("nan")] * 7
            self.current_axis_measured_values = [float("nan")] * 7
            self.current_axis_measurement_valid = [False] * 7

            # 初始化视频记录。
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            recording_dir = self.prepare_recording_session(timestamp_str)
            self.real_video_path = os.path.join(
                recording_dir, f"RealMask_{timestamp_str}.avi"
            )
            self.virt_video_path = os.path.join(
                recording_dir, f"VirtualMask_{timestamp_str}.avi"
            )
            # Mask writers are opened lazily on the first frame. This avoids
            # leaving an invalid, header-only AVI when segmentation is disabled.
            self.real_video_writer = None
            self.virt_video_writer = None
            print(">>> [Record] 视频记录已准备")
            try:
                self.start_dual_camera_recording(timestamp_str)
            except Exception as e:
                print(f"初始化双路摄像头记录失败: {e}")
                self.dual_camera_recorder = None
        else:
            # 停止记录并保存。
            self.is_recording_trajectory = False
            print(">>> [Record] 停止记录，正在保存...")
            total_time = time.time() - self.recording_start_time
            total_frames = len(self.trajectory_data)

            if total_time > 0:
                actual_fps = total_frames / total_time
                print(f"========================================")
                print(f"Real recording duration: {total_time:.2f} s")
                print(f"Real recording frames: {total_frames}")
                print(f"建议将 real_fps 调整为: {actual_fps:.2f}")
                print(f"========================================")
            self.stop_dual_camera_recording()

            if self.real_video_writer is not None:
                self.real_video_writer.release()
                self.real_video_writer = None
                self._report_finalized_mask_video(self.real_video_path)
            if self.virt_video_writer is not None:
                self.virt_video_writer.release()
                self.virt_video_writer = None
                self._report_finalized_mask_video(self.virt_video_path)
            self.save_trajectory_to_file()

    @staticmethod
    def _report_finalized_mask_video(path):
        info = inspect_video_file(path)
        if info["readable"]:
            print(
                f">>> [Record] 已验证视频: {os.path.basename(path)}, "
                f"{info['frames']} 帧, {info['size'][0]}x{info['size'][1]}"
            )
        else:
            print(f">>> [Record] WARNING: 视频无法解码: {path}")

    def save_trajectory_to_file(self):
        """将记录的双探子轨迹数据保存为 CSV 文件。"""
        if not self.trajectory_data:
            return
        try:
            timestamp_str = (
                getattr(self, "recording_timestamp_str", None)
                or datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            filename = f"{timestamp_str}_DualScope.csv"
            output_dir = getattr(self, "recording_session_dir", None) or self.save_dir
            file_path = os.path.join(output_dir, filename)

            import csv
            with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 构建表头。
                
                # Proj 为投影后的坐标，Raw 为原始配准坐标，Q 为姿态四元数。
                headers = ["Time(s)"]
                headers += ["S1_Proj_X", "S1_Proj_Y", "S1_Proj_Z", 
                            "S1_Raw_X",  "S1_Raw_Y",  "S1_Raw_Z", 
                            "S1_QX", "S1_QY", "S1_QZ", "S1_QW"]
                
                # 说明已清理。
                headers += [f"Axis_{i}" for i in range(7)]
                headers += [f"Axis_{i}_DPOS" for i in range(7)]
                headers += [f"Axis_{i}_MPOS" for i in range(7)]
                headers += [f"Axis_{i}_MPOS_Valid" for i in range(7)]
                
                # 3. Scope 2 (鏂伴暅瀛? - 缁撴瀯瀹屽叏鐩稿悓
                headers += ["S2_Proj_X", "S2_Proj_Y", "S2_Proj_Z", 
                            "S2_Raw_X",  "S2_Raw_Y",  "S2_Raw_Z", 
                            "S2_QX", "S2_QY", "S2_QZ", "S2_QW"]
                
                writer.writerow(headers)
                writer.writerows(self.trajectory_data)

            QMessageBox.information(
                self,
                "记录完成",
                f"本次全部记录已保存至:\n{output_dir}",
            )
        except Exception as e:
            print(f"保存失败: {e}")

    def update_axis_positions(self):
        """Read seven independent axes, preserving DPOS versus MPOS."""
        if getattr(self, "is_simulation_mode", False) and self.mujoco_simulator is not None:
            try:
                values = self.mujoco_simulator.get_axis_values()
                for i, pos in enumerate(values):
                    self.current_axis_values[i] = float(pos)
                    self.current_axis_demand_values[i] = float(pos)
                    self.current_axis_measured_values[i] = float(pos)
                    self.current_axis_measurement_valid[i] = True
                    if i < len(self.axis_pos_labels):
                        self.axis_pos_labels[i].setText(f"Axis {i}: {pos:.2f} [SIM]")
                    if i < 6:
                        self.axis_histories[i].append(pos)
                        self.curves[i].setData(list(self.axis_histories[i]))
            except Exception:
                pass
            return
        if not getattr(self, "is_mc_connected", False):
            return
        if not hasattr(self, "mc_connection"):
            return
        for i in range(7):
            demand = float("nan")
            measured = float("nan")
            try:
                demand = float(self.mc_connection.GetAxisParameter_DPOS(i))
            except Exception:
                pass
            try:
                measured = float(self.mc_connection.GetAxisParameter_MPOS(i))
            except Exception:
                pass
            self.current_axis_demand_values[i] = demand
            measured_valid = bool(np.isfinite(measured))
            self.current_axis_measured_values[i] = measured
            self.current_axis_measurement_valid[i] = measured_valid
            selected = measured if measured_valid else demand
            if np.isfinite(selected):
                self.current_axis_values[i] = selected
                source = "M" if measured_valid else "D"
                if i < len(self.axis_pos_labels):
                    self.axis_pos_labels[i].setText(
                        f"Axis {i}: {selected:.2f} [{source}]"
                    )
                if i < 6:
                    self.axis_histories[i].append(selected)
                    self.curves[i].setData(list(self.axis_histories[i]))
    def on_virtual_debug_toggled(self, checked):
        """Toggle the virtual mask debug window."""
        if checked:
            if self.virt_debug_window is None:
                self.virt_debug_window = DebugImageWindow()
                self.virt_debug_window.setWindowTitle("虚拟 Mask 调试")
            self.virt_debug_window.show()
        else:
            if self.virt_debug_window:
                self.virt_debug_window.hide()
    def on_debug_toggled(self, checked):
        if checked:
            if self.debug_window is None:
                self.debug_window = DebugImageWindow()
            self.debug_window.show()
        else:
            if self.debug_window:
                self.debug_window.hide()

    # 说明已清理。
    @QtCore.pyqtSlot(object, list)
    def on_segmentation_result(self, mask, centroids):
        if not self.seg_running:
            return
        self.latest_segmentation_mask = None if mask is None else np.asarray(mask).copy()

        # 说明已清理。
        virtual_target_uv = self.get_path_lookahead_uv()

        # 说明已清理。
        matched_idx = -1
        status = "LOST"
        target_display_uv = None

        if virtual_target_uv is not None:
            # perform_robust_matching 杩斿洖: best_idx, status, target_point_uv
            matched_idx, status, target_display_uv = self.perform_robust_matching(centroids, virtual_target_uv)
        else:
            status = "LOST"

        # 说明已清理。
        points_to_draw = []

        if self.show_all_centroids_cb.isChecked():
            # 说明已清理。
            points_to_draw = centroids
        else:
            # 说明已清理。
            if status == "LOCKED" and matched_idx != -1:
                # 说明已清理。
                points_to_draw = [centroids[matched_idx]]
            else:
                # 说明已清理。
                points_to_draw = []
        if getattr(self, "is_recording_trajectory", False) and mask is not None:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            if mask_bgr.shape[:2] != (400, 400):
                mask_bgr = cv2.resize(mask_bgr, (400, 400), interpolation=cv2.INTER_NEAREST)
            if self.real_video_writer is None and self.real_video_path:
                self.real_video_writer, codec = open_compatible_avi_writer(
                    self.real_video_path,
                    self.real_video_fps,
                    (400, 400),
                )
                print(f">>> [Record] RealMask codec: {codec}")
            self.real_video_writer.write(mask_bgr)
        # 说明已清理。
        self.centroid_overlay.set_centroids(points_to_draw)

        # 说明已清理。
        # 说明已清理。
        if hasattr(self, "nav_overlay"):
            self.nav_overlay.update_status(target_display_uv, status)

        # 说明已清理。
        if self.debug_cb.isChecked() and self.debug_window and self.debug_window.isVisible():
            self.debug_window.update_image(mask)


    def on_camera_changed(self, index: int):
        """Switch physical OpenCV camera."""
        if getattr(self, "is_simulation_mode", False):
            self.video_timer.stop()
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
                self.cap = None
            self.actual_view.setText("MuJoCo tip_camera 数据流")
            return

        self.video_timer.stop()
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

        # 2. 灏濊瘯鎵撳紑鏂扮浉鏈?
        # 说明已清理。
        # 说明已清理。
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if self.cap.isOpened():
            # Keep the device's native frame size; only the Qt preview scales it.
            self.video_timer.start(30)
            print(f"相机 {index} 已通过 OpenCV 打开")
        else:
            self.actual_view.setText(f"无法打开相机 {index}")
            print(f"相机 {index} 打开失败")

    def add_example_mesh(self):
        bronch_mesh_path = self._resolve_bronch_mesh_path()
        self.mesh = pv.read(bronch_mesh_path)
        self.plotter.add_mesh(self.mesh, color="lightgray", opacity=0.5)
        self.plotter.add_axes()
        self.plotter.reset_camera()


    def toggle_connection(self):
        """Handle connect/disconnect button click"""
        if self.ndi_connect_button.text() == "Connect":
            selected_port = self.comComboBox.currentData()
            if selected_port:
                try:
                    self.tracker_wrapper = NDITrackerWrapper()
                    self.tracker_wrapper.COM_PORT = selected_port
                    self.data_queue = self.tracker_wrapper.visualization_queue

                    if self.tracker_wrapper.initialize_tracker():
                        self.ndi_connect_button.setText("断开 NDI")
                        self.comComboBox.setEnabled(False)
                        self.tracker_wrapper.start_tracking_thread()
                    else:
                        QMessageBox.critical(self, "NDI 连接失败", "跟踪器初始化失败")
                except Exception as e:
                    QMessageBox.critical(self, "NDI 连接失败", f"连接失败: {str(e)}")
        else:
            # 断开 NDI。
            if self.tracker_wrapper:
                self.tracker_wrapper.stop_tracking()
                self.tracker_wrapper = None
            self.ndi_connect_button.setText("连接 NDI")
            self.comComboBox.setEnabled(True)

    def refresh_ports(self):
        """刷新可用 COM 口。"""
        self.comComboBox.clear()
        ports = list(serial.tools.list_ports.comports())
        selected_index = 0
        for i, port in enumerate(ports):
            self.comComboBox.addItem(f"{port.device} - {port.description}", port.device)
            if port.device == "COM3":
                selected_index = i
        self.comComboBox.setCurrentIndex(selected_index)







    def select_points(self):
        """Enable picking three reference points in the main 3D view."""
        if getattr(self, "is_simulation_mode", False):
            self._reset_simulation_calibration(clear_model_points=True)
        self.picked_points = []
        if hasattr(self, "picked_point_actors"):
            for actor in self.picked_point_actors:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        self.picked_point_actors = []
        self.point_picker_observer = self.plotter.enable_point_picking(
            callback=self.point_picked_callback,
            show_message=True,
            use_picker=True
        )
        print("模型点选择已启用，请在左侧 3D 肺模型中依次点击 M1、M2、M3。")

    def point_picked_callback(self, point, *args):
        if point is None:
            return
        point = np.asarray(point, dtype=float)
        if len(self.picked_points) >= 3:
            return
        self.picked_points.append(point)
        print(f"选取模型点 M{len(self.picked_points)}: {point}")

        sphere = pv.Sphere(radius=2, center=point)
        sphere_actor = self.plotter.add_mesh(sphere, color="#00AEEF")
        self.picked_point_actors.append(sphere_actor)

        if len(self.picked_points) == 3:
            # 说明已清理。
            self.plotter.disable_picking()
            # 说明已清理。
            self.update_labels_with_points()
            if getattr(self, "is_simulation_mode", False) and self.mujoco_simulator is not None:
                QMessageBox.information(
                    self,
                    "MuJoCo 电磁点采集",
                    "模型点 M1/M2/M3 已记录。\n请在 MuJoCo viewer 的肺部模型上依次点击对应的 E1/E2/E3。"
                )
                self._start_simulated_em_calibration_sequence()

    def update_labels_with_points(self):
        """Show picked reference points in labels."""
        for i, point in enumerate(self.picked_points):
            self.coord_labels[i][0].setText(f"P{i+1}_X: {point[0]:.3f}")
            self.coord_labels[i][1].setText(f"P{i+1}_Y: {point[1]:.3f}")
            self.coord_labels[i][2].setText(f"P{i+1}_Z: {point[2]:.3f}")

    def record_points_1(self):
        self._record_ref_point(0)

    def record_points_2(self):
        self._record_ref_point(1)

    def record_points_3(self):
        self._record_ref_point(2)




    def select_endpoint_mode(self):
        """
        说明已清理。
        说明已清理。
        """
        # 说明已清理。
        if hasattr(self, "endpoint_actor") and self.endpoint_actor:
            self.plotter.remove_actor(self.endpoint_actor)
            self.endpoint_actor = None

        # 说明已清理。
        # 说明已清理。
        self.plotter.add_text("请在模型中右键点击选择导航终点...", position='upper_left', font_size=12,
                              color='white', name='msg')

        # 说明已清理。
        # 说明已清理。
        self.plotter.enable_point_picking(
            callback=self.on_endpoint_picked,
            show_message=True,  # 閫夊畬鍚庢樉绀哄潗鏍囦俊鎭?
            font_size=12,
            use_picker=True,
            show_point=False
        )
        print(">>> 进入终点选择模式，请在模型上右键点击。")

    def on_endpoint_picked(self, point, *args):
        """
        说明已清理。
        """
        if point is None: return

        print(f"选中终点坐标: {point}")
        self.endpoint = list(point)  # 鏇存柊缁堢偣鍧愭爣

        # 说明已清理。
        self.plotter.add_text("导航终点已设定", position='upper_left', font_size=12, color='green', name='msg')
        self.has_reached_target = False
        # 说明已清理。
        sphere = pv.Sphere(radius=3, center=point)
        self.endpoint_actor = self.plotter.add_mesh(sphere, color="blue", label="Goal")

        # 说明已清理。
        self.plotter.disable_picking()

        # 4. 鏍囪绯荤粺宸插氨缁?
        self.ready_control = True

        # 说明已清理。
        self.plotter.render()

    def update_visualization(self):
        """
        鏇存柊鍙鍖栵細鏀寔鍙岄暅瀛?(Scope1 & Scope2)
        说明已清理。
        说明已清理。
        """
        if self._is_closing:
            return
        try:
            self.update_axis_positions()

            # 说明已清理。
            self.scope_pos = None
            self.scope_dir = None 
            
            scope2_pos = None
            scope2_dir = None
            
            # 说明已清理。
            rec_scope1 = [0]*7 
            rec_quat1 = [0, 0, 0, 1]
            rec_scope2 = [0]*7 
            rec_quat2 = [0, 0, 0, 1]

            # ====== 1. 鑾峰彇鏁版嵁 ======
            tools_dict = None
            if getattr(self, "is_simulation_mode", False) and self.mujoco_simulator is not None:
                self.mujoco_simulator.step()
                if self.auto_simulation_mapping_ready:
                    self.R, self.t = self.mujoco_simulator.get_lung_auto_registration()
                self.update_simulated_camera_frame()
                if getattr(self.mujoco_simulator, "model_pick_requested", False):
                    self.mujoco_simulator.model_pick_requested = False
                    self.begin_mujoco_em_point_picking()
                tools_dict = self.mujoco_simulator.get_tools_dict()
            elif hasattr(self, 'data_queue') and not self.data_queue.empty():
                tools_dict = self.data_queue.get()
                while not self.data_queue.empty():
                    tools_dict = self.data_queue.get()
            self.latest_tools_dict = tools_dict

            if tools_dict:

                # --- 绔彛瀹氫箟 ---
                # 说明已清理。
                PORT_SCOPE_1 = 10  # 鍘熼暅瀛?
                PORT_REF     = 11  # 閰嶅噯鎺㈤拡
                PORT_SCOPE_2 = 12  # 鏂伴暅瀛?(Scope 2)

                # --- A. 閰嶅噯鎺㈤拡 (Port 11) ---
                if PORT_REF in tools_dict:
                    mat_ref = tools_dict[PORT_REF]
                    self.ref_pos = mat_ref[:3, 3]
                    
                    # 说明已清理。
                    if len(self.R) == 0:
                        self.x_label.setText("Ref X: {:.2f}".format(self.ref_pos[0]))
                        self.y_label.setText("Ref Y: {:.2f}".format(self.ref_pos[1]))
                        self.z_label.setText("Ref Z: {:.2f}".format(self.ref_pos[2]))
                    
                    if hasattr(self, 'actual_point_actor') and self.actual_point_actor:
                        self.actual_point_actor.SetPosition(self.ref_pos)

                # 说明已清理。
                # 说明已清理。
                # 说明已清理。
                # 说明已清理。
                fix_mat = np.array([
                    [-1, 0, 0, 0],
                    [ 0,-1, 0, 0],
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1]
                ])
                if getattr(self, "is_simulation_mode", False):
                    fix_mat = np.eye(4, dtype=float)

                # --- B. 闀滃瓙 1 (Port 10) ---
                if PORT_SCOPE_1 in tools_dict:
                    # 说明已清理。
                    mat_scope = tools_dict[PORT_SCOPE_1] @ fix_mat
                    self.scope_pos = mat_scope[:3, 3]
                    self.scope_dir = mat_scope[:3, :3]
                    
                    if len(self.R) > 0:
                        self.x_label.setText("S1 X: {:.2f}".format(self.scope_pos[0]))
                        self.y_label.setText("S1 Y: {:.2f}".format(self.scope_pos[1]))
                        self.z_label.setText("S1 Z: {:.2f}".format(self.scope_pos[2]))

                # --- C. 闀滃瓙 2 (Port 12) ---
                if PORT_SCOPE_2 in tools_dict:
                    # 说明已清理。
                    mat_scope2 = tools_dict[PORT_SCOPE_2] @ fix_mat
                    scope2_pos = mat_scope2[:3, 3]
                    scope2_dir = mat_scope2[:3, :3]

            registration_ready = len(self.R) > 0 and len(self.t) > 0
            if getattr(self, "is_simulation_mode", False):
                registration_ready = registration_ready and self.calibration_ready

            if registration_ready:
                
                # 说明已清理。
                if self.scope_pos is not None:
                    raw_virtual = np.dot(self.R, self.scope_pos) + self.t
                    mapped_scope_dir = np.dot(self.R, self.scope_dir) if self.scope_dir is not None else None
                    self.mapped_tip_camera_dir = mapped_scope_dir.copy() if mapped_scope_dir is not None else None
                    self.virtual_position_raw = raw_virtual.copy()
                    
                    # 说明已清理。
                    if hasattr(self, "centerline_points") and not self.auto_simulation_mapping_ready:
                        self.virtual_position = self.project_onto_centerline(raw_virtual)
                    else:
                        self.virtual_position = raw_virtual
                    
                    # 鏇存柊 UI
                    self.xx_label.setText("X: {:.2f}".format(self.virtual_position[0]))
                    self.yy_label.setText("Y: {:.2f}".format(self.virtual_position[1]))
                    self.zz_label.setText("Z: {:.2f}".format(self.virtual_position[2]))

                    # 说明已清理。
                    # 说明已清理。
                    self._update_scope_actor(mapped_scope_dir, self.virtual_position, "scope1")

                    # 记录映射到肺模型坐标系后的姿态。
                    rec_quat1 = R.from_matrix(mapped_scope_dir).as_quat() if mapped_scope_dir is not None else [0, 0, 0, 1]
                    rec_scope1 = [self.virtual_position[0], self.virtual_position[1], self.virtual_position[2],
                                  raw_virtual[0], raw_virtual[1], raw_virtual[2]]

                    # 说明已清理。
                    if hasattr(self, "endpoint") and self.endpoint is not None:
                        dist = np.linalg.norm(self.virtual_position - self.endpoint)
                        if dist < 2.0 and not self.has_reached_target:
                            self.has_reached_target = True
                            QMessageBox.information(self, "导航提示", "Scope 1 已到达目标点。")

                # 说明已清理。
                if scope2_pos is not None:
                    raw_virtual_2 = np.dot(self.R, scope2_pos) + self.t
                    mapped_scope2_dir = np.dot(self.R, scope2_dir) if scope2_dir is not None else None
                    
                    if hasattr(self, "centerline_points") and not self.auto_simulation_mapping_ready:
                        virtual_position_2 = self.project_onto_centerline(raw_virtual_2)
                    else:
                        virtual_position_2 = raw_virtual_2
                    
                    # 说明已清理。
                    self._update_scope_actor(mapped_scope2_dir, virtual_position_2, "scope2")

                    # 记录映射到肺模型坐标系后的姿态。
                    rec_quat2 = R.from_matrix(mapped_scope2_dir).as_quat() if mapped_scope2_dir is not None else [0, 0, 0, 1]
                    rec_scope2 = [virtual_position_2[0], virtual_position_2[1], virtual_position_2[2],
                                  raw_virtual_2[0], raw_virtual_2[1], raw_virtual_2[2]]

                # ====== 3. 鏁版嵁璁板綍 ======
                if getattr(self, "is_recording_trajectory", False):
                    current_t = time.time() - self.recording_start_time
                    axes_pos = list(self.current_axis_values) if hasattr(self, 'current_axis_values') else [0]*7
                    axes_demand = list(getattr(self, "current_axis_demand_values", [float("nan")] * 7))
                    axes_measured = list(getattr(self, "current_axis_measured_values", [float("nan")] * 7))
                    axes_measured_valid = [
                        int(value) for value in getattr(self, "current_axis_measurement_valid", [False] * 7)
                    ]
                    
                    # 缁勫悎鏁版嵁: [Time] + Scope1 + Axis + Scope2
                    s1_data = rec_scope1[:3] + rec_scope1[3:6] + list(rec_quat1)
                    s2_data = rec_scope2[:3] + rec_scope2[3:6] + list(rec_quat2)

                    row_data = (
                        [current_t] + s1_data + axes_pos + axes_demand
                        + axes_measured + axes_measured_valid + s2_data
                    )
                    self.trajectory_data.append(row_data)

            # 说明已清理。
            if hasattr(self, 'plotter') and not self.plotter._closed:
                self.plotter.render()
                
            # 说明已清理。
            # 说明已清理。
            # 说明已清理。
            self._update_virtual_camera_logic()
            self.record_dual_camera_frame_pair()
            self._automation_frame_update()

        except Exception as e:
            # 说明已清理。
            print(f"可视化更新异常: {e}")

    def _update_virtual_camera_logic(self):
        """Update virtual camera state."""
        should_run_virtual = False
        if self.ready_control: should_run_virtual = True
        if self.virtual_debug_cb.isChecked(): should_run_virtual = True

        if should_run_virtual and hasattr(self, 'virtual_view'):
            if not hasattr(self, 'cyl_transform'): 
                 self.cyl_transform = vtk.vtkTransform()
            
            # 璋冪敤鍘熸湰鐨?update_virtual_camera 鍑芥暟鏉ヨ缃浉鏈哄弬鏁?
            self.update_virtual_camera()

    def _update_scope_actor(self, rotation_matrix, position, actor_name):
        """Update scope cylinder actor."""
        if rotation_matrix is None or position is None:
            return
        rotation_matrix = np.asarray(rotation_matrix, dtype=float)
        position = np.asarray(position, dtype=float)
        if rotation_matrix.shape != (3, 3):
            return
        dir_z = rotation_matrix[:, 2]
        if np.linalg.norm(dir_z) < 1e-6: dir_z = np.array([0, 0, 1])
        else: dir_z = dir_z / np.linalg.norm(dir_z)

        # 说明已清理。
        base_z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(base_z, dir_z)
        if np.linalg.norm(axis) < 1e-6: axis = np.array([1.0, 0.0, 0.0])
        else: axis = axis / np.linalg.norm(axis)
        angle = math.degrees(math.acos(np.clip(np.dot(base_z, dir_z), -1.0, 1.0)))

        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.RotateWXYZ(angle, axis.tolist())
        transform.Translate(position.tolist())

        # 说明已清理。
        actor_attr = f"{actor_name}_actor" # e.g., scope1_actor, scope2_actor
        
        # 说明已清理。
        if not hasattr(self, actor_attr) or getattr(self, actor_attr) is None:
            # 说明已清理。
            color = "black" if actor_name == "scope1" else "cyan"
            cyl = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=1, height=5)
            new_actor = self.plotter.add_mesh(cyl, color=color, name=actor_name)
            setattr(self, actor_attr, new_actor)
        
        actor = getattr(self, actor_attr)
        if actor:
            actor.SetUserTransform(transform)

    def _update_virtual_camera_logic(self):
        """Update virtual camera state."""
        should_run_virtual = False
        if self.ready_control: should_run_virtual = True
        if self.virtual_debug_cb.isChecked(): should_run_virtual = True

        if should_run_virtual and hasattr(self, 'virtual_view'):
            # 说明已清理。
            # 说明已清理。
            if not hasattr(self, 'cyl_transform'):
                 self.cyl_transform = vtk.vtkTransform()
            
            # 说明已清理。
            self.update_virtual_camera()


    def get_path_lookahead_uv(self):
        """
        说明已清理。
        说明已清理。
        """
        action = getattr(self, "auto_nav_action", None)
        if self.auto_nav_state == "running" and action is not None and action.get("target_model_mm") is not None:
            target_pt = np.asarray(action["target_model_mm"], dtype=float)
        else:
            if not hasattr(self, "smoothpath") or len(self.smoothpath) == 0:
                return None
            path_arr = np.array(self.smoothpath)
            dists = np.linalg.norm(path_arr - self.virtual_position, axis=1)
            curr_idx = np.argmin(dists)
            look_ahead_steps = 12
            target_idx = min(curr_idx + look_ahead_steps, len(path_arr) - 1)
            target_pt = path_arr[target_idx]

        # 说明已清理。
        if not hasattr(self, "cam_R_cw") or not hasattr(self, "cam_C"):
            return None

        P_w_minus_C = target_pt - self.cam_C
        P_c = self.cam_R_cw @ P_w_minus_C
        x, y, z = P_c

        # 说明已清理。
        if z < 1e-3:
            return None

        # 说明已清理。

        # 说明已清理。
        cam = self.virtual_view.camera
        fov_deg = cam.GetViewAngle()

        # 说明已清理。
        h_sensor = 400.0
        w_sensor = 400.0

        # C. 鏍规嵁 FOV 鍙嶆帹鐒﹁窛
        # f = (h / 2) / tan(fov / 2)
        f_pixel_new = (h_sensor / 2.0) / math.tan(math.radians(fov_deg / 2.0))

        # 说明已清理。
        fx = f_pixel_new
        fy = f_pixel_new

        # 说明已清理。
        K = self.intrinsic_camera_matrix
        cx = w_sensor - K[0, 2]
        cy = h_sensor - K[1, 2]

        # E. 鎶曞奖鍏紡
        u = fx * x / z + cx
        v = fy * y / z + cy

        # 说明已清理。
        u_norm = u / w_sensor
        v_norm = v / h_sensor

        return (u_norm, v_norm)
    def get_virtual_visual_center(self):
        """
        说明已清理。
        说明已清理。
        """
        if not hasattr(self, 'virtual_view') or self.virtual_view is None:
            return [], None

        # 说明已清理。
        ren_win = self.virtual_view.GetRenderWindow()

        # 说明已清理。
        # 说明已清理。
        # 说明已清理。
        ren_win.SetSwapBuffers(0)

        actors_to_hide = []

        # 说明已清理。
        if hasattr(self, "path_actor_virtual") and self.path_actor_virtual:
            if self.path_actor_virtual.GetVisibility():
                actors_to_hide.append(self.path_actor_virtual)

        if hasattr(self, "virt_opening_actor") and self.virt_opening_actor:
            if self.virt_opening_actor.GetVisibility():
                actors_to_hide.append(self.virt_opening_actor)

        if hasattr(self, "virt_green_actor") and self.virt_green_actor:
            if self.virt_green_actor.GetVisibility():
                actors_to_hide.append(self.virt_green_actor)

        if hasattr(self, "virtual_point_actor") and self.virtual_point_actor:
            if self.virtual_point_actor.GetVisibility():
                actors_to_hide.append(self.virtual_point_actor)

        if hasattr(self, "picked_point_actors") and self.picked_point_actors:
            for actor in self.picked_point_actors:
                if actor and actor.GetVisibility():
                    actors_to_hide.append(actor)
        if hasattr(self, "virt_box_actor") and self.virt_box_actor:
            actors_to_hide.append(self.virt_box_actor)
        if hasattr(self, "debug_guide_actor") and self.debug_guide_actor:
            actors_to_hide.append(self.debug_guide_actor)
        # 2. 鎵ц闅愯棌
        for actor in actors_to_hide:
            actor.SetVisibility(False)

        # 说明已清理。
        if actors_to_hide:
            self.virtual_view.render()

        try:
            # 说明已清理。
            depth = self.virtual_view.get_image_depth(fill_value=60.0)
        except AttributeError:
            # 说明已清理。
            for actor in actors_to_hide:
                actor.SetVisibility(True)
            ren_win.SetSwapBuffers(1)  # <--- 鎭㈠
            return [], -1, None, None

        # 说明已清理。
        for actor in actors_to_hide:
            actor.SetVisibility(True)

        # 说明已清理。
        # 说明已清理。
        ren_win.SetSwapBuffers(1)

        # 说明已清理。
        if actors_to_hide:
            self.virtual_view.render()
            self.virtual_view.render()

        # 说明已清理。

        if depth is None or depth.size == 0:
            return [], -1, None, None
        # 2. 鐢熸垚 Mask
        dist = np.abs(depth)
        valid_mask = (dist > 1e-3) & (~np.isnan(dist))
        if np.any(valid_mask):
            max_valid = np.max(dist[valid_mask])
        else:
            max_valid = 10.0

        # 说明已清理。
        infinity_val = max_valid * 2.0
        # 说明已清理。
        if infinity_val < 100.0: infinity_val = 100.0

        # 3. 鏇挎崲鍧忓€?
        # 说明已清理。
        dist = np.nan_to_num(dist, nan=infinity_val, posinf=infinity_val, neginf=infinity_val)
        # 说明已清理。
        dist[dist < 1e-3] = infinity_val
        # 说明已清理。
        # 说明已清理。
        current_max = np.percentile(dist, 98)
        # 说明已清理。
        effective_clip = min(60.0, current_max * 1.2)
        dist = np.clip(dist, 0, effective_clip)

        max_d = np.percentile(dist, 98)

        if max_d < 1.0:  # 闂ㄦ涔熻皟浣庣偣
            return [], -1, None, dist

        # 说明已清理。
        # 说明已清理。
        ratio = 0.8
        thresh_val = max_d * ratio
        mask = (dist > thresh_val).astype(np.uint8) * 255

        # 说明已清理。
        # 说明已清理。
        # 说明已清理。
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        centers_list = []
        h, w = mask.shape

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # 说明已清理。
            # 说明已清理。
            if area < 5: continue

            cx, cy = centroids[i]
            u_norm = cx / w
            v_norm = cy / h
            centers_list.append((u_norm, v_norm))
        if not centers_list:
            return [], -1, mask, dist
            # 说明已清理。
        guide_uv = self.get_path_lookahead_uv()
        best_idx = -1
        if guide_uv is not None:
            g_u, g_v = guide_uv
            min_dist = float('inf')

            # 2. 閬嶅巻鎵炬渶杩?
            for i, (c_u, c_v) in enumerate(centers_list):
                # 说明已清理。
                match_dist = (c_u - g_u) ** 2 + (c_v - g_v) ** 2

                if match_dist < min_dist:
                    min_dist = match_dist
                    best_idx = i
        else:
            best_idx = -1

        return centers_list, best_idx, mask, dist

    def update_virtual_camera(self):
        """
        说明已清理。
        说明已清理。
        """
        if self._is_closing: return
        if not hasattr(self, 'virtual_view') or self.virtual_view is None:
            return
        if hasattr(self.virtual_view, '_closed') and self.virtual_view._closed:
            return

        if not hasattr(self, 'virtual_position') or len(self.virtual_position) != 3:
            return

        if (
            getattr(self, "is_simulation_mode", False)
            and getattr(self, "mapped_tip_camera_dir", None) is not None
        ):
            camera_frame = np.asarray(self.mapped_tip_camera_dir, dtype=float)
            dir_view = camera_frame[:, 2]
            view_up = camera_frame[:, 1]
        else:
            if not hasattr(self, 'cyl_transform'):
                return
            mat = self.cyl_transform.GetMatrix()
            dir_view = np.array([mat.GetElement(i, 1) for i in range(3)], dtype=float)
            view_up = np.array([0.0, 0.0, 1.0], dtype=float)

        dir_norm = np.linalg.norm(dir_view)
        dir_view = dir_view / dir_norm if dir_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
        up_norm = np.linalg.norm(view_up)
        view_up = view_up / up_norm if up_norm > 1e-6 else np.array([0.0, 0.0, 1.0])

        cam = self.virtual_view.camera
        cam.SetPosition(self.virtual_position.tolist())
        cam.SetViewUp(view_up.tolist())
        focal = np.asarray(self.virtual_position, dtype=float) + dir_view * 10.0
        cam.SetFocalPoint(focal.tolist())
        if getattr(self, "is_simulation_mode", False):
            cam.SetViewAngle(float(self.mujoco_simulator.model.cam_fovy[self.mujoco_simulator.tip_camera_id]))
            cam.SetWindowCenter(0.0, 0.0)
            cam.SetClippingRange(0.1, 40000.0)

        # 渲染更新。
        self.virtual_view.render()

        # 说明已清理。
        # 说明已清理。
        z_cam = np.array(dir_view, dtype=float)
        z_norm = np.linalg.norm(z_cam)
        if z_norm > 1e-6:
            z_cam /= z_norm
        else:
            z_cam = np.array([0.0, 1.0, 0.0])

        cam_pos = np.array(self.virtual_position, dtype=float)
        up_world = np.array([0.0, 0.0, 1.0])

        # 鏋勫缓鐩告満鍧愭爣绯?(Right-Down-Forward)
        x_cam = np.cross(z_cam, up_world) # Right
        if np.linalg.norm(x_cam) < 1e-6:
            x_cam = np.array([1.0, 0.0, 0.0])
        else:
            x_cam /= np.linalg.norm(x_cam)

        y_cam = np.cross(z_cam, x_cam) # Down

        self.cam_R_cw = np.vstack([x_cam, y_cam, z_cam])
        self.cam_C = cam_pos
        self.cam_z_dir = z_cam

        # 说明已清理。
        if not hasattr(self, "opening_update_counter"):
            self.opening_update_counter = 0
        self.opening_update_counter += 1

        if self.opening_update_counter >= 5:
            self.opening_update_counter = 0

            is_debug = self.virtual_debug_cb.isChecked()
            need_calc = getattr(self, "enable_virtual_seg", False) or is_debug

            if need_calc:
                if hasattr(self, "get_virtual_visual_center"):
                    v_centers_list, best_idx, v_mask, v_dist = self.get_virtual_visual_center()
                    
                    if getattr(self, "is_recording_trajectory", False):
                        try:
                            if v_mask is None: mask_bgr = np.zeros((400, 400, 3), dtype=np.uint8)
                            else: mask_bgr = cv2.cvtColor(v_mask, cv2.COLOR_GRAY2BGR)

                            if v_dist is None: depth_color = np.zeros((400, 400, 3), dtype=np.uint8)
                            else:
                                safe_dist = np.nan_to_num(v_dist, nan=60.0, posinf=60.0, neginf=60.0)
                                safe_dist = np.clip(safe_dist, 0, 60.0)
                                depth_norm = (safe_dist / 60.0 * 255).astype(np.uint8)
                                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

                            if mask_bgr.shape[:2] != (400, 400): mask_bgr = cv2.resize(mask_bgr, (400, 400))
                            if depth_color.shape[:2] != (400, 400): depth_color = cv2.resize(depth_color, (400, 400))
                            combined_frame = np.hstack([mask_bgr, depth_color])
                            if combined_frame.shape[:2] != (400, 800): combined_frame = cv2.resize(combined_frame, (800, 400))
                            if self.virt_video_writer is None and self.virt_video_path:
                                self.virt_video_writer, codec = open_compatible_avi_writer(
                                    self.virt_video_path,
                                    self.virt_video_fps,
                                    (800, 400),
                                )
                                print(f">>> [Record] VirtualMask codec: {codec}")
                            self.virt_video_writer.write(combined_frame)
                        except Exception as e:
                            print(f"写入虚拟视频失败: {e}")

                    guide_uv = self.get_path_lookahead_uv()
                    self.debug_guide_points.Reset()
                    self.debug_guide_cells.Reset()
                    if guide_uv is not None:
                        gu, gv = guide_uv
                        w = self.virtual_view.width()
                        h = self.virtual_view.height()
                        gx_px = gu * w
                        gy_vtk = h - (gv * h)
                        pid = self.debug_guide_points.InsertNextPoint(gx_px, gy_vtk, 0)
                        self.debug_guide_cells.InsertNextCell(1)
                        self.debug_guide_cells.InsertCellPoint(pid)
                    self.debug_guide_points.Modified()
                    self.debug_guide_cells.Modified()
                    self.debug_guide_poly.Modified()

                    self.virt_green_points.Reset()
                    self.virt_green_cells.Reset()
                    self.virt_box_points.Reset()
                    self.virt_box_cells.Reset()

                    if v_centers_list:
                        w = self.virtual_view.width()
                        h = self.virtual_view.height()
                        for i, (u_norm, v_norm) in enumerate(v_centers_list):
                            x_px = u_norm * w
                            y_vtk = h - (v_norm * h)
                            pid = self.virt_green_points.InsertNextPoint(x_px, y_vtk, 0)
                            self.virt_green_cells.InsertNextCell(1)
                            self.virt_green_cells.InsertCellPoint(pid)
                            if i == best_idx:
                                r = 20
                                p0, p1 = (x_px - r, y_vtk - r), (x_px + r, y_vtk - r)
                                p2, p3 = (x_px + r, y_vtk + r), (x_px - r, y_vtk + r)
                                base_id = self.virt_box_points.GetNumberOfPoints()
                                for p in [p0, p1, p2, p3]: self.virt_box_points.InsertNextPoint(p[0], p[1], 0)
                                for pair in [(0,1), (1,2), (2,3), (3,0)]:
                                    line = vtk.vtkLine()
                                    line.GetPointIds().SetId(0, base_id + pair[0])
                                    line.GetPointIds().SetId(1, base_id + pair[1])
                                    self.virt_box_cells.InsertNextCell(line)
                                self.virtual_center_cache = (u_norm, v_norm)

                    self.virt_green_points.Modified()
                    self.virt_green_cells.Modified()
                    self.virt_green_poly.Modified()
                    self.virt_box_points.Modified()
                    self.virt_box_cells.Modified()
                    self.virt_box_poly.Modified()

                    if is_debug and self.virt_debug_window:
                        self.virt_debug_window.update_image(v_mask, v_dist)

    def extract_points_from_labels(self,label_grid):
        points = []
        for row in label_grid:
            point = []
            for label in row:
                text = label.text()
                try:
                    number_str = text.split(":")[-1].strip()
                    value = float(number_str)
                except Exception as e:
                    value = 0.0
                point.append(value)
            points.append(point)
        return np.array(points)
    def Kabsch_computer(self):
        points_A = self.extract_points_from_labels(self.coord_labels2)
        points_B = self.extract_points_from_labels(self.coord_labels)
        if points_A.shape != (3, 3) or points_B.shape != (3, 3):
            QMessageBox.warning(self, "配准点不足", "需要 3 个模型点和 3 个电磁点才能完成配准。")
            return
        if not (np.isfinite(points_A).all() and np.isfinite(points_B).all()):
            QMessageBox.warning(self, "配准点无效", "标定点坐标包含无效值，请重新采集。")
            return
        if getattr(self, "is_simulation_mode", False) and self.mujoco_simulator is not None:
            if len(self.sim_em_points) < 3:
                QMessageBox.information(self, "MuJoCo 电磁点未采满", "请先在 MuJoCo viewer 中采集 E1/E2/E3。")
                self.sim_em_pick_active = True
                self.sim_em_pick_index = min(len(self.sim_em_points), 2)
                self.begin_mujoco_em_point_picking()
                return
        self.R, self.t = compute_rigid_transform(points_A, points_B)
        transformed = (self.R @ points_A.T).T + self.t
        residual_vecs = points_B - transformed
        residuals = np.linalg.norm(residual_vecs, axis=1)
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        max_error = float(np.max(residuals))
        self.calibration_residuals = residuals.tolist()
        self.calibration_rmse = rmse
        self.calibration_max_error = max_error
        is_sim_calibration = getattr(self, "is_simulation_mode", False) and self.mujoco_simulator is not None
        if is_sim_calibration:
            self.calibration_ready = rmse < self.calibration_threshold_mm
            self.ready_control = self.ready_control or self.calibration_ready
        else:
            self.calibration_ready = False
            self.ready_control = True

        print("Kabsch 输入电磁点 E:", points_A)
        print("Kabsch 输入模型点 M:", points_B)
        print("R:", self.R)
        print("t:", self.t)
        print(f"标定残差: {residuals}, RMSE={rmse:.3f} mm, Max={max_error:.3f} mm")

        message = (
            f"残差: E1={residuals[0]:.3f} mm, E2={residuals[1]:.3f} mm, E3={residuals[2]:.3f} mm\n"
            f"RMSE: {rmse:.3f} mm\n"
            f"最大误差: {max_error:.3f} mm\n"
            f"阈值: {self.calibration_threshold_mm:.3f} mm"
        )
        if is_sim_calibration:
            if self.calibration_ready:
                self.sim_status_label.setText(f"MuJoCo 标定完成: RMSE {rmse:.2f} mm")
                QMessageBox.information(self, "标定完成", message)
            else:
                self.sim_status_label.setText(f"MuJoCo 标定误差过大: RMSE {rmse:.2f} mm")
                QMessageBox.warning(self, "标定误差过大", message + "\n\n请重新采集 M1/M2/M3 与 E1/E2/E3。")
        else:
            QMessageBox.information(self, "配准完成", message)



    def route_plan(self):
        try:
            centerline_path = self._resolve_centerline_iges_path()
            self.all_points = self.read_iges(centerline_path)
            valid_lines = [np.asarray(points, dtype=float) for points in self.all_points if len(points) > 0]
            if not valid_lines:
                raise ValueError(f"中心线文件未包含有效曲线: {centerline_path}")
            self.all_points = valid_lines
            self.centerline_points = np.vstack(self.all_points)

            print(f"正在构建 KDTree，共 {len(self.centerline_points)} 个点...")
            self.kdtree = cKDTree(self.centerline_points)

            nearest_point_goal, d, i = self.find_nearest_point(self.centerline_points, self.endpoint)
            path, self.smoothpath, graph = pathplan(
                self.all_points,
                [9.1551647, -94.828995, 43.316994],
                nearest_point_goal,
            )
            if self.smoothpath is None or len(self.smoothpath) < 2:
                raise ValueError("路径规划未生成有效路径，请重新选择导航终点。")
        except Exception as e:
            QMessageBox.critical(self, "路径规划失败", str(e))
            print(f"路径规划失败: {e}")
            return

        # 使用规划后的路径点替换全局中心线点。
        self.centerline_points = np.array(self.smoothpath)

        # 重建 KDTree，此时树中只包含规划路径上的点。
        print(f"正在切换导航树，锁定路径点数: {len(self.centerline_points)}")
        self.kdtree = cKDTree(self.centerline_points)
        line = pv.lines_from_points(self.smoothpath)
        if hasattr(self, "path_actor"):
            self.plotter.remove_actor(self.path_actor)

            # 说明已清理。
        self.path_actor = self.plotter.add_mesh(line, color="red", line_width=4)

        # 渲染更新界面。
        self.plotter.render()
        if hasattr(self, "path_actor_virtual"):
            try:
                self.virtual_view.remove_actor(self.path_actor_virtual)
            except Exception:
                pass
        # Keep the clinical virtual camera unobstructed. Navigation targets are
        # available in the optional target-observation window instead.
        self.path_actor_virtual = None
        self.virtual_view.render()
        if getattr(self, "is_simulation_mode", False) and self.mujoco_simulator is not None:
            try:
                self.mujoco_simulator.set_navigation_path_model_mm(self.smoothpath)
                self.sim_status_label.setText("MuJoCo 自动同步已启用，规划路径已同步")
            except Exception as e:
                QMessageBox.warning(self, "路径同步失败", str(e))
        self.enable_virtual_seg = True
        self.auto_nav_tracking_path = None
        self.auto_nav_global_profile = None
        self.auto_nav_target_index = 0
        self.auto_nav_nearest_index = 0
        self.auto_nav_path_distance = 0.0
        self.auto_nav_passed_count = 0
        self.auto_nav_progress = 0.0
        self.auto_nav_filtered_steering[:] = 0.0
        self.auto_nav_proximal_command[:] = 0.0
        self.auto_nav_distal_command[:] = 0.0
        self.auto_nav_previous_heading_error[:] = 0.0
        self.auto_nav_pid_integral[:] = 0.0
        self.auto_nav_pid_derivative[:] = 0.0
        self.auto_nav_pid_output_filtered[:] = 0.0
        self.auto_nav_pid_last_time = None
        self.auto_nav_steering_reversal_frames = 0
        self.auto_nav_filtered_distal_assist = 0.0
        self.auto_nav_elastic_release_active = False
        self.auto_nav_elastic_release_frames = 0
        self.auto_nav_curve_command[:] = 0.0
        self.auto_nav_curve_direction[:] = 0.0
        self.auto_nav_curve_release_frames = 0
        self.auto_nav_curve_release_cooldown = 0
        self.auto_nav_control_basis = None
        self.auto_nav_last_curve_heading_error = None
        self.auto_nav_curve_error_rise_frames = 0
        self.auto_nav_filtered_position = None
        self.auto_nav_filtered_forward = None
        self.auto_nav_waypoint_reached_frames = 0
        self.auto_nav_waypoint_controller_index = -1
        self.auto_nav_elastic_bend_direction[:] = 0.0
        self.auto_nav_elastic_release_countdown = 0
        self.auto_nav_elastic_settle_frames = 0
        self.auto_nav_straight_mode = True
        self.auto_nav_filtered_path_curve = 0.0
        self.auto_nav_straight_correction_active = False
        self.auto_nav_last_turn_direction[:] = 0.0
        self.auto_nav_committed_turn_direction[:] = 0.0
        self.auto_nav_turn_severity = 0.0
        self.auto_nav_turn_active = False
        self.auto_nav_control_frames = 0
        self.auto_nav_servo_target_index = -1
        self.auto_nav_target_transition_frames = 0
        self.auto_nav_filtered_bearing[:] = 0.0
        self.auto_nav_last_bearing_angle_deg = None
        self.auto_nav_bearing_improvement = 0.0
        self.auto_nav_action_hold_countdown = 0
        self.auto_nav_held_proximal_target[:] = 0.0
        self.auto_nav_held_distal_target[:] = 0.0
        self.auto_nav_curve_memory = 0.0
        self.auto_nav_guidance_direction = None
        self.auto_nav_straight_error_frames = 0
        self.auto_nav_straight_clear_frames = 0
        self.auto_nav_motion_mode = "forward"
        self.auto_nav_behind_frames = 0
        self.auto_nav_front_frames = 0
        self.auto_nav_bend_gain_scale = 1.0
        self.auto_nav_forward_reengage_frames = 0
        self.auto_nav_filtered_speed = 0.0
        self.auto_nav_last_insertion_step = 0.0
        self.auto_nav_last_position = None
        self.auto_nav_stall_frames = 0
        self.refresh_automation_path_status()
        print("虚拟视角分割准备完成。")
    def read_iges(self,file_path):
        iges = pyiges.read(file_path)
        all_points = []
        for entity in iges:
            parameters = entity.parameters
            num_points = int(parameters[0][2])
            coords_strings = parameters[0][3:]
            coords_floats = [float(value.strip()) for value in coords_strings]

            # 灏嗗潗鏍囪浆鎹负 (X, Y, Z) 鏍煎紡
            points = []
            for i in range(num_points):
                x = coords_floats[i * 3]
                y = coords_floats[i * 3 + 1]
                z = coords_floats[i * 3 + 2]
                points.append([x, y, z])

            # 说明已清理。
            points = np.array(points)
            all_points.append(points)
        return all_points

    def find_nearest_point(self, points, point):
        """
        说明已清理。
        说明已清理。
        """
        point = np.array(point)

        # 1. 浼樺厛浣跨敤 KDTree 鏌ヨ (鏋佸揩)
        if hasattr(self, 'kdtree') and self.kdtree is not None:
            # query 杩斿洖 (璺濈, 绱㈠紩)
            # 说明已清理。
            distance, index = self.kdtree.query(point, k=1)

            nearest_point = self.centerline_points[index]
            return tuple(nearest_point), distance, index

        # 说明已清理。
        # 说明已清理。
        points = np.array(points)
        distances = np.linalg.norm(points - point, axis=1)
        nearest_index = np.argmin(distances)
        nearest_point = points[nearest_index]
        nearest_distance = distances[nearest_index]

        return tuple(nearest_point), nearest_distance, nearest_index

    def get_closest_point_on_segment(self, p, a, b):
        """
        说明已清理。
        """
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)

        ab = b - a
        length_sq = np.sum(ab ** 2)

        if length_sq < 1e-6:
            return a  # a 鍜?b 閲嶅悎

        # 鎶曞奖鍏紡: t = (ap . ab) / (ab . ab)
        ap = p - a
        t = np.dot(ap, ab) / length_sq

        # 说明已清理。
        t = np.clip(t, 0.0, 1.0)

        # 说明已清理。
        closest = a + t * ab
        return closest
    def project_onto_centerline(self, pt):
        if not hasattr(self, "centerline_points") or self.centerline_points is None:
            # 说明已清理。
            return pt

        # 1. 鍏堢敤 KDTree 鎵惧埌鏈€杩戠殑閭ｄ釜"椤剁偣" (绮楀畾浣?
        # 说明已清理。
        _, distance, idx = self.find_nearest_point(self.centerline_points, pt)

        # 说明已清理。
        N = len(self.centerline_points)
        if N < 2:
            return self.centerline_points[idx]

        # 说明已清理。
        # 说明已清理。
        # 说明已清理。

        candidates = []

        # 说明已清理。
        P_i = self.centerline_points[idx]

        # 说明已清理。
        if idx > 0:
            P_prev = self.centerline_points[idx - 1]
            proj_prev = self.get_closest_point_on_segment(pt, P_prev, P_i)
            dist_prev = np.linalg.norm(proj_prev - pt)
            candidates.append((dist_prev, proj_prev))

        # 说明已清理。
        if idx < N - 1:
            P_next = self.centerline_points[idx + 1]
            proj_next = self.get_closest_point_on_segment(pt, P_i, P_next)
            dist_next = np.linalg.norm(proj_next - pt)
            candidates.append((dist_next, proj_next))

        # 说明已清理。
        if not candidates:
            return P_i

        # 鎸夎窛绂绘帓搴忥紝鍙栨渶灏忕殑
        candidates.sort(key=lambda x: x[0])
        best_point = candidates[0][1]

        return np.array(best_point, dtype=float)

    def compute_virtual_branch_openings(self, plane_z=15.0):

        if not hasattr(self, "all_points") or self.all_points is None:
            return []
        if not hasattr(self, "cam_R_cw") or not hasattr(self, "cam_C"):
            return []

        K = self.intrinsic_camera_matrix
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        w = max(self.virtual_view.width(), 1)
        h = max(self.virtual_view.height(), 1)

        openings = []

        for bid, poly in enumerate(self.all_points):
            if poly is None:
                continue
            poly = np.asarray(poly)
            if poly.shape[0] < 2:
                continue

            Pw = poly
            Pw_minus_C = Pw - self.cam_C[None, :]
            Pc_all = (self.cam_R_cw @ Pw_minus_C.T).T  # (N,3)

            open_point_cam = None
            for i in range(len(Pc_all) - 1):
                p1 = Pc_all[i]
                p2 = Pc_all[i + 1]
                z1, z2 = p1[2], p2[2]

                if z1 <= 1e-6 and z2 <= 1e-6:
                    continue
                val1 = z1 - plane_z
                val2 = z2 - plane_z
                if val1 * val2 > 0:
                    continue
                denom = (z2 - z1)
                if abs(denom) < 1e-6:
                    continue
                t = (plane_z - z1) / denom
                if t < -0.1 or t > 1.1:
                    continue

                open_point_cam = p1 + t * (p2 - p1)
                break

            if open_point_cam is None:
                continue

            x, y, z = open_point_cam
            if z <= 1e-6:
                continue

            u = fx * x / z + cx
            v = fy * y / z + cy
            if not (0 <= u < w and 0 <= v < h):
                continue

            Pw_open = self.cam_R_cw.T @ open_point_cam + self.cam_C

            openings.append({
                "branch_id": bid,
                "Pw": Pw_open,
                "Pc": open_point_cam,
                "uv": (float(u), float(v)),
            })

        return openings

    def update_virtual_opening_glyphs(self, plane_z=15.0):

        openings = self.compute_virtual_branch_openings(plane_z=plane_z)

        # 说明已清理。
        if not openings:
            # 说明已清理。
            if self.virt_opening_actor is not None:
                try:
                    self.virt_opening_actor.SetVisibility(False)
                except Exception:
                    pass
            self.virtual_view.render()
            return

        # 说明已清理。
        pts = np.array([o["Pw"] for o in openings], dtype=float)  # (N,3)

        # 说明已清理。
        if self.virt_opening_actor is None or self.virt_opening_poly is None:
            # 说明已清理。
            self.virt_opening_poly = pv.PolyData(pts)

            # 说明已清理。
            self.virt_opening_actor = self.virtual_view.add_mesh(
                self.virt_opening_poly,
                color="green",
                point_size=10,
                render_points_as_spheres=True
            )
        else:
            # 说明已清理。
            try:
                # 说明已清理。
                self.virt_opening_poly.points = pts
                # 说明已清理。
                self.virt_opening_actor.SetVisibility(True)
            except Exception as e:
                print("更新虚拟开口点时出错:", e)

        # 鏈€鍚庢覆鏌撲竴娆?
        self.virtual_view.render()
    def sync_virtual_camera_with_intrinsics(self):
        if getattr(self, "is_simulation_mode", False) and self.mujoco_simulator is not None:
            cam = self.virtual_view.camera
            cam.SetViewAngle(float(self.mujoco_simulator.model.cam_fovy[self.mujoco_simulator.tip_camera_id]))
            cam.SetWindowCenter(0.0, 0.0)
            cam.SetClippingRange(0.1, 40000.0)
            self.virtual_view.render()
            return
        K = self.intrinsic_camera_matrix
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 鐢ㄦ墜鍐屼腑鐨勭湡瀹炲弬鏁?
        pixel_size_mm = 1.008e-3  # 1.008 碌m
        h_pixels = 400
        sensor_height = pixel_size_mm * h_pixels

        f_mm = fy * pixel_size_mm
        fovy = 100

        cam = self.virtual_view.camera
        cam.SetViewAngle(fovy)

        w, h = 400, 400
        wx = (cx - w / 2) / (w / 2)
        wy = (h / 2 - cy) / (h / 2)
        cam.SetWindowCenter(wx, wy)
        cam.SetClippingRange(0.1, 30)
        self.virtual_view.render()

    def update_video_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret: return
        self.latest_real_frame_bgr = frame.copy()

        if not self._is_closing:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.actual_view.size(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
            self.actual_view.setPixmap(scaled_pixmap)

            # Overlay 鐩存帴閾烘弧 Label (鍥犱负 Label 灏辨槸鏈夋晥鐢婚潰)
            self.centroid_overlay.setGeometry(0, 0, self.actual_view.width(), self.actual_view.height())
            self.nav_overlay.setGeometry(0, 0, self.actual_view.width(), self.actual_view.height())
        # 说明已清理。
        if self.seg_running and hasattr(self, 'seg_queue'):
            try:
                if self.seg_queue.full():
                    try:
                        self.seg_queue.get_nowait()
                    except:
                        pass
                self.seg_queue.put(frame, block=False)
            except Exception:
                pass
    def toggle_segmentation(self, checked: bool):

        if checked:
            # 启动分割。
            self.seg_button.setText("停止分割")
            self.start_segmentation()
        else:
            # 停止分割。
            self.seg_button.setText("开始分割")
            self.stop_segmentation()

    def start_segmentation(self):
        if self.seg_running:
            return

        try:
            weights_path = self._resolve_segmentation_weights_path()
        except Exception as e:
            self._reset_segmentation_ui()
            QMessageBox.critical(self, "分割模型加载失败", str(e))
            print(f"分割启动失败: {e}")
            return

        self.seg_queue = Queue(maxsize=1)

        self.seg_thread = QtCore.QThread(self)

        self.seg_worker = SegmentationWorker(
            weights_path=weights_path,
            frame_queue=self.seg_queue,  # <--- 浼犲叆闃熷垪
            img_size=400
        )
        self.seg_worker.moveToThread(self.seg_thread)

        self.seg_thread.started.connect(self.seg_worker.process_loop)
        self.seg_worker.finished.connect(self.seg_thread.quit)
        self.seg_worker.finished.connect(self.seg_worker.deleteLater)
        self.seg_worker.errorOccurred.connect(self._on_segmentation_error)
        self.seg_thread.finished.connect(self._on_segmentation_thread_finished)
        self.seg_thread.finished.connect(self.seg_thread.deleteLater)
        self.seg_worker.resultReady.connect(self.on_segmentation_result)

        self.seg_running = True
        self.seg_thread.start()




    def stop_segmentation(self):
        if not self.seg_running:
            return
        self.seg_running = False
        if self.seg_worker is not None:
            self.seg_worker.stop()
        self.centroid_overlay.set_centroids([])

    def perform_robust_matching(self, real_centroids, virtual_target_uv):
        """
        说明已清理。
        说明已清理。
        """

        # 1. 鍩虹妫€鏌?
        if virtual_target_uv is None:
            return -1, "LOST", None

        g_u, g_v = virtual_target_uv

        if not real_centroids:
            return -1, "GHOST", (g_u, g_v)

        # ==========================================================
        # ====== 绛栫暐 A: 鍗曞鍗曞己琛岄攣瀹?(涔嬪墠鐨勯€昏緫) ======
        # ==========================================================
        if len(real_centroids) == 1:
            return 0, "LOCKED", real_centroids[0]

        # ==========================================================
        # 说明已清理。
        # ==========================================================
        best_idx = -1
        min_dist = float('inf')

        for i, (r_u, r_v) in enumerate(real_centroids):
            # 计算欧氏距离平方。
            dist = (r_u - g_u) ** 2 + (r_v - g_v) ** 2
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        # ==========================================================
        # ====== 绛栫暐 C: 瀹芥澗鍖归厤 (閽堝鎮ㄥ浘鐗囦腑鐨勬儏鍐? ======
        # ==========================================================
        # 说明已清理。
        # 说明已清理。
        # 说明已清理。

        LOOSE_THRESHOLD = 0.3
        STRICT_THRESHOLD = 0.2

        current_threshold = STRICT_THRESHOLD
        if len(real_centroids) <= 3:
            current_threshold = LOOSE_THRESHOLD

        # 说明已清理。
        if min_dist < current_threshold ** 2:
            return best_idx, "LOCKED", real_centroids[best_idx]
        else:
            # 说明已清理。
            return -1, "GHOST", (g_u, g_v)
    def closeEvent(self, event):
        print("[Exit] 正在退出...")

        # 说明已清理。
        if hasattr(self, 'timer'): self.timer.stop()
        if hasattr(self, 'video_timer'): self.video_timer.stop()
        self.is_recording_trajectory = False
        self.stop_dual_camera_recording()
        for writer_name in ("real_video_writer", "virt_video_writer"):
            writer = getattr(self, writer_name, None)
            if writer is not None:
                writer.release()
                setattr(self, writer_name, None)
        if getattr(self, "vla_recording", False):
            self.stop_vla_episode()
        self.stop_auto_navigation()

        # 2. 鍋滄 NDI Tracker
        if hasattr(self, "tracker_wrapper") and self.tracker_wrapper:
            try:
                self.tracker_wrapper.stop_tracking()
            except:
                pass

        if hasattr(self, "mujoco_simulator") and self.mujoco_simulator:
            try:
                self.mujoco_simulator.close()
            except:
                pass

        # 说明已清理。
        if hasattr(self, "seg_worker") and self.seg_worker:
            self.seg_worker.stop()  # 鎶?_running 璁句负 False

        # 说明已清理。
        if hasattr(self, "debug_window") and self.debug_window:
            self.debug_window.close()
        if hasattr(self, "virt_debug_window") and self.virt_debug_window:
            self.virt_debug_window.close()
        if hasattr(self, "mujoco_pick_window") and self.mujoco_pick_window:
            self.mujoco_pick_window.close()
        if getattr(self, "navigation_target_window", None):
            self.navigation_target_window.close()
        if getattr(self, "d435_capture_window", None):
            try:
                self.d435_capture_window.close()
            except Exception:
                pass
        self.virt_green_actor = None
        print("[Exit] Bye!")
        event.accept()
        os._exit(0)


bind_automation_runtime_globals(globals())
bind_simulation_runtime_globals(globals())


def create_environment_window(environment, data_queue, mujoco_xml=None):
    """Create the exact workbench selected by the startup environment dialog."""
    if str(environment).lower() == "real":
        return load_real_environment_window_class()(data_queue)
    return MainWindow(
        data_queue,
        initial_environment="simulation",
        initial_mujoco_xml=mujoco_xml,
    )


def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_product_style(app)
    startup = StartupEnvironmentDialog()
    if startup.exec_() != QtWidgets.QDialog.Accepted:
        return
    window = create_environment_window(
        startup.selected_environment,
        Queue(),
        mujoco_xml=startup.selected_mujoco_xml,
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
