"""Simulation-only methods extracted from the dual-probe main window.

Method bodies are intentionally unchanged; runtime globals are bound by the
host UI module after its classes and constants have been defined.
"""

from __future__ import annotations

from PyQt5 import QtCore


def bind_runtime_globals(namespace):
    for name, value in namespace.items():
        if not name.startswith('__'):
            globals()[name] = value


class SimulationRuntimeMixin:
    def _start_initial_simulation(self):
        if hasattr(self, "sim_model_combo"):
            idx = self.sim_model_combo.findData(self.mujoco_xml_path)
            if idx >= 0:
                self.sim_model_combo.setCurrentIndex(idx)
        if hasattr(self, "sim_toggle_btn") and not self.sim_toggle_btn.isChecked():
            self.sim_toggle_btn.setChecked(True)

    def _populate_mujoco_models(self):
        self.sim_model_combo.clear()
        candidate_paths = []
        meshes_dir = os.path.join(PROJECT_ROOT, "meshes")
        if os.path.isdir(meshes_dir):
            for name in sorted(os.listdir(meshes_dir)):
                if name.lower().endswith(".xml"):
                    candidate_paths.append(os.path.join(meshes_dir, name))
        if DEFAULT_MUJOCO_XML not in candidate_paths and os.path.exists(DEFAULT_MUJOCO_XML):
            candidate_paths.insert(0, DEFAULT_MUJOCO_XML)

        for path in candidate_paths:
            self.sim_model_combo.addItem(os.path.basename(path), path)

        default_index = self.sim_model_combo.findData(DEFAULT_MUJOCO_XML)
        if default_index >= 0:
            self.sim_model_combo.setCurrentIndex(default_index)

    def open_d435_capture_window(self):
        """Open the shared D435 collector with this window's axis/sim sources."""
        if self.d435_capture_window is not None:
            try:
                self.d435_capture_window.show()
                self.d435_capture_window.raise_()
                self.d435_capture_window.activateWindow()
                return
            except RuntimeError:
                self.d435_capture_window = None
        try:
            from Visual_information.d435_tdcr_capture.app import D435CaptureWindow
            from Visual_information.d435_tdcr_capture.axes import CallbackAxisSource
            from Visual_information.d435_tdcr_capture.config import load_config as load_d435_config
            from Visual_information.d435_tdcr_capture.mujoco_bridge import ExistingMujocoBridge

            capture_config = load_d435_config()

            def current_trio_connection():
                if getattr(self, "is_mc_connected", False):
                    return getattr(self, "mc_connection", None)
                return None

            axis_source = CallbackAxisSource(
                capture_config["axes"], current_trio_connection
            )
            bridge = None
            if self.mujoco_simulator is not None:
                bridge = ExistingMujocoBridge(
                    capture_config["mujoco"], self.mujoco_simulator
                )
            self.d435_capture_window = D435CaptureWindow(
                capture_config,
                axis_source=axis_source,
                mujoco_bridge=bridge,
                parent=self,
            )
            self.d435_capture_window.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            self.d435_capture_window.destroyed.connect(self._on_d435_window_closed)
            self.d435_capture_window.show()
            self.d435_status_label.setText(
                "D435窗口已打开 | "
                + ("已复用当前MuJoCo模型" if bridge is not None else "启动MuJoCo后重新打开可实时对齐")
            )
        except Exception as exc:
            self.d435_status_label.setText(f"D435窗口启动失败: {exc}")
            QMessageBox.critical(self, "D435启动失败", str(exc))

    def _on_d435_window_closed(self, *_args):
        self.d435_capture_window = None
        if hasattr(self, "d435_status_label"):
            self.d435_status_label.setText(
                "7点: 0/7/14/21/28/35/42 mm | 相机未启动"
            )

    def toggle_mujoco_simulation(self, checked):
        if checked:
            xml_path = self.sim_model_combo.currentData() or self.mujoco_xml_path or DEFAULT_MUJOCO_XML
            try:
                self.mujoco_simulator = MujocoDualScopeSimulator(xml_path, open_viewer=True)
                self.mujoco_xml_path = xml_path
                self.is_simulation_mode = True
                self._reset_simulation_calibration()
                self.mujoco_simulator.pick_callback = self.on_mujoco_em_point_picked
                self.sim_toggle_btn.setText("关闭 MuJoCo 仿真")
                self.sim_status_label.setText(f"当前数据源: MuJoCo ({os.path.basename(xml_path)})")
                self.ndi_connect_button.setEnabled(False)
                self.comComboBox.setEnabled(False)
                self.camera_combo.setEnabled(False)
                self.actual_view.setText("MuJoCo 仿真模式\ntip_camera 数据流已启用")
                self.sim_control_combo.setEnabled(True)
                self.gamepad_connect_btn.setEnabled(True)
                self.passive_joint_group.setEnabled(True)
                self._apply_passive_joint_parameters()
                self.on_sim_control_mode_changed(self.sim_control_combo.currentIndex())
                self._enable_automatic_simulation_mapping()
            except Exception as e:
                self.is_simulation_mode = False
                self.mujoco_simulator = None
                self.sim_toggle_btn.blockSignals(True)
                self.sim_toggle_btn.setChecked(False)
                self.sim_toggle_btn.blockSignals(False)
                self.sim_status_label.setText("当前数据源: 真实环境")
                self.sim_control_combo.setEnabled(False)
                self.gamepad_connect_btn.setEnabled(False)
                self.passive_joint_group.setEnabled(False)
                self.sim_control_status.setText("MuJoCo 启动失败")
                QMessageBox.critical(self, "MuJoCo 仿真启动失败", str(e))
        else:
            if self.mujoco_simulator is not None:
                self.mujoco_simulator.pick_callback = None
                try:
                    self.mujoco_simulator.close()
                except Exception:
                    pass
            self.mujoco_simulator = None
            self.is_simulation_mode = False
            self.sim_em_pick_active = False
            self.calibration_ready = False
            self.auto_simulation_mapping_ready = False
            self.sim_toggle_btn.setText("启用 MuJoCo 仿真")
            self.sim_status_label.setText("当前数据源: 真实环境")
            self.ndi_connect_button.setEnabled(True)
            self.comComboBox.setEnabled(True)
            self.camera_combo.setEnabled(True)
            self.sim_control_combo.setEnabled(False)
            self.gamepad_connect_btn.setEnabled(False)
            self.passive_joint_group.setEnabled(False)
            self.passive_joint_status.setText("启动 MuJoCo 后生效")
            self.sim_control_status.setText("启动 MuJoCo 后可选择控制方式")

    def on_passive_joint_parameters_changed(self, _value=None):
        stiffness_scale = self.passive_stiffness_slider.value() / 100.0
        effective_stiffness = 500.0 * stiffness_scale
        self.passive_stiffness_value.setText(
            f"{stiffness_scale:.2f}×  ({effective_stiffness:.1f} N·m/rad)"
        )
        self._apply_passive_joint_parameters()

    def _apply_passive_joint_parameters(self):
        if self.mujoco_simulator is None:
            return
        stiffness_scale = self.passive_stiffness_slider.value() / 100.0
        damping_scale = self.passive_damping_spin.value()
        try:
            values = self.mujoco_simulator.set_passive_joint_scales(
                stiffness_scale, damping_scale
            )
            self.passive_joint_status.setText(
                f"{values['joint_count']}关节 | "
                f"K={values['stiffness']:.1f} | D={values['damping']:.2f}"
            )
        except Exception as exc:
            self.passive_joint_status.setText(f"应用失败: {exc}")

    def reset_passive_joint_parameters(self):
        self.passive_stiffness_slider.setValue(100)
        self.passive_damping_spin.setValue(1.00)
        self.on_passive_joint_parameters_changed()

    def on_sim_control_mode_changed(self, index):
        mode = self.sim_control_combo.itemData(index)
        is_gamepad = mode == "gamepad"
        self.gamepad_connect_btn.setVisible(is_gamepad)
        if self.mujoco_simulator is None:
            return
        try:
            self.mujoco_simulator.set_control_mode(mode)
            if is_gamepad:
                connected = self.mujoco_simulator.connect_gamepad()
                self.sim_control_status.setText(
                    "手柄已连接：左/右摇杆控制两段，LT/RT 控制进给"
                    if connected else "未检测到手柄，请连接后点击“检测并连接手柄”"
                )
            else:
                self.sim_control_status.setText("双罗盘 UI 控制窗口已打开")
        except Exception as e:
            self.sim_control_status.setText(f"控制模式切换失败: {e}")

    def connect_sim_gamepad(self):
        if self.mujoco_simulator is None:
            QMessageBox.information(self, "手柄控制", "请先启用 MuJoCo 仿真。")
            return
        connected = self.mujoco_simulator.connect_gamepad()
        if connected:
            self.sim_control_status.setText("手柄已连接：左/右摇杆控制两段，LT/RT 控制进给")
        else:
            self.sim_control_status.setText("未检测到 XInput 手柄")
            QMessageBox.warning(self, "手柄连接", "未检测到 XInput 手柄，请检查 USB 或蓝牙连接。")

    def update_simulated_camera_frame(self):
        frame = None
        if self.mujoco_simulator is not None:
            frame = self.mujoco_simulator.latest_tip_frame_bgr
        if frame is None or self._is_closing:
            return

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
        self.centroid_overlay.setGeometry(0, 0, self.actual_view.width(), self.actual_view.height())
        self.nav_overlay.setGeometry(0, 0, self.actual_view.width(), self.actual_view.height())

        if self.seg_running and hasattr(self, 'seg_queue'):
            try:
                if self.seg_queue.full():
                    try:
                        self.seg_queue.get_nowait()
                    except Exception:
                        pass
                self.seg_queue.put(frame, block=False)
            except Exception:
                pass

    def _resolve_bronch_mesh_path(self):
        preferred = os.path.join(BASE_DIR, "支气管.stl")
        if os.path.exists(preferred):
            return preferred
        for name in os.listdir(BASE_DIR):
            lower_name = name.lower()
            if lower_name.endswith(".stl") and ("支气管" in name or "bronch" in lower_name):
                return os.path.join(BASE_DIR, name)
        raise FileNotFoundError(f"未找到支气管 STL 模型: {preferred}")

    def _reset_simulation_calibration(self, clear_model_points=False):
        self.calibration_ready = False
        self.calibration_rmse = None
        self.calibration_max_error = None
        self.calibration_residuals = []
        self.sim_em_points = []
        self.sim_em_pick_index = 0
        self.sim_em_pick_active = False
        self.sim_candidate_em_point = None
        self.auto_simulation_mapping_ready = False
        self.R = []
        self.t = []
        if clear_model_points:
            for grid in (getattr(self, "coord_labels", []), getattr(self, "coord_labels2", [])):
                for i, row in enumerate(grid):
                    for j, label in enumerate(row):
                        axis = ["X", "Y", "Z"][j]
                        label.setText(f"P{i + 1}_{axis}: 0")
        if self.mujoco_simulator is not None:
            self.mujoco_simulator.enable_viewer_point_picking(False)

    def _enable_automatic_simulation_mapping(self):
        if not getattr(self, "is_simulation_mode", False) or self.mujoco_simulator is None:
            return False
        try:
            self.R, self.t = self.mujoco_simulator.get_lung_auto_registration()
            self.calibration_ready = True
            self.auto_simulation_mapping_ready = True
            self.ready_control = True
            self.sim_status_label.setText("MuJoCo 自动同步已启用，无需手工标定")
            print("MuJoCo 自动模型映射已启用")
            print("R:", self.R)
            print("t:", self.t)
            if len(getattr(self, "smoothpath", [])) > 0:
                self.mujoco_simulator.set_navigation_path_model_mm(self.smoothpath)
            return True
        except Exception as e:
            self.calibration_ready = False
            self.auto_simulation_mapping_ready = False
            self.ready_control = False
            QMessageBox.warning(self, "自动模型同步失败", str(e))
            return False

    def _write_em_point_to_labels(self, row_index, point):
        if not (0 <= row_index < len(self.coord_labels2)):
            return
        point = np.asarray(point, dtype=float)
        point_num = row_index + 1
        self.coord_labels2[row_index][0].setText(f"P{point_num}_X: {point[0]:.3f}")
        self.coord_labels2[row_index][1].setText(f"P{point_num}_Y: {point[1]:.3f}")
        self.coord_labels2[row_index][2].setText(f"P{point_num}_Z: {point[2]:.3f}")

    def _confirm_simulated_em_point(self, row_index):
        if not self.sim_em_pick_active:
            return False
        if row_index != self.sim_em_pick_index:
            QMessageBox.information(
                self,
                "请按顺序记录",
                f"当前需要确认 E{self.sim_em_pick_index + 1}，请点击“记录标记点 {self.sim_em_pick_index + 1}”。",
            )
            return True
        if self.sim_candidate_em_point is None:
            QMessageBox.information(
                self,
                "尚未选择候选点",
                f"请先在 MuJoCo viewer 的肺部模型上点击候选 E{self.sim_em_pick_index + 1}。",
            )
            return True

        point = np.asarray(self.sim_candidate_em_point, dtype=float)
        self.sim_em_points.append(point.copy())
        self._write_em_point_to_labels(row_index, point)
        print(f"确认 MuJoCo 电磁点 E{row_index + 1}: {point}")
        self.sim_em_pick_index += 1
        self.sim_candidate_em_point = None

        if self.sim_em_pick_index < 3:
            self._prompt_next_mujoco_em_point()
        else:
            self.sim_em_pick_active = False
            if self.mujoco_simulator is not None:
                self.mujoco_simulator.enable_viewer_point_picking(False)
            self.plotter.add_text(
                "E1/E2/E3 已确认完成，正在自动配准...",
                position="upper_left",
                font_size=12,
                color="white",
                name="mujoco_pick_msg",
            )
            self.plotter.render()
            self.Kabsch_computer()
        return True

    def _start_simulated_em_calibration_sequence(self):
        if not getattr(self, "is_simulation_mode", False) or self.mujoco_simulator is None:
            return False
        if len(getattr(self, "picked_points", [])) != 3:
            QMessageBox.warning(self, "标定点不足", "请先在左侧 3D 肺模型中选择 M1/M2/M3。")
            return False
        self.sim_em_points = []
        self.sim_em_pick_index = 0
        self.sim_em_pick_active = True
        self.calibration_ready = False
        self.calibration_residuals = []
        self.mujoco_simulator.pick_callback = self.on_mujoco_em_point_picked
        return self.begin_mujoco_em_point_picking()

    def _prompt_next_mujoco_em_point(self):
        if not self.sim_em_pick_active or self.mujoco_simulator is None:
            return False
        if self.sim_em_pick_index >= 3:
            return False
        point_name = f"E{self.sim_em_pick_index + 1}"
        self.sim_status_label.setText(f"MuJoCo 标定采集中: 请点击 {point_name}")
        self.plotter.add_text(
            f"请在 MuJoCo viewer 肺部模型上点击候选 {point_name}，确认后点击右侧“记录标记点 {self.sim_em_pick_index + 1}”",
            position="upper_left",
            font_size=12,
            color="white",
            name="mujoco_pick_msg",
        )
        self.plotter.render()
        self.mujoco_simulator.enable_viewer_point_picking(True)
        return True

    def _record_ref_point(self, row_index):
        if getattr(self, "is_simulation_mode", False) and self.mujoco_simulator is not None:
            if self._confirm_simulated_em_point(row_index):
                return
        if self.ref_pos is None:
            QMessageBox.warning(self, "未检测到定位探头", "未检测到用于标定的电磁探头（Port 2）。")
            return
        if not (0 <= row_index < len(self.coord_labels2)):
            return
        point_num = row_index + 1
        self.coord_labels2[row_index][0].setText(f"P{point_num}_X: {self.ref_pos[0]:.3f}")
        self.coord_labels2[row_index][1].setText(f"P{point_num}_Y: {self.ref_pos[1]:.3f}")
        self.coord_labels2[row_index][2].setText(f"P{point_num}_Z: {self.ref_pos[2]:.3f}")

    def begin_mujoco_em_point_picking(self):
        if not getattr(self, "is_simulation_mode", False) or self.mujoco_simulator is None:
            return False
        self.mujoco_simulator.pick_callback = self.on_mujoco_em_point_picked
        if self.mujoco_simulator.viewer is not None:
            self._prompt_next_mujoco_em_point()
            return True
        if not hasattr(self, "mujoco_pick_window") or self.mujoco_pick_window is None:
            self.mujoco_pick_window = MujocoLungPickWindow(self.mesh, self.on_mujoco_em_point_picked, self)
        self.mujoco_pick_window.show()
        self.mujoco_pick_window.raise_()
        self.mujoco_pick_window.activateWindow()
        return True

    def on_mujoco_em_point_picked(self, point, *args):
        if point is None:
            return
        point = np.asarray(point, dtype=float)
        if self.mujoco_simulator is not None:
            self.mujoco_simulator.set_clicked_em_position_mm(point, notify=False)
        self.ref_pos = point
        self.x_label.setText("Ref X: {:.2f}".format(point[0]))
        self.y_label.setText("Ref Y: {:.2f}".format(point[1]))
        self.z_label.setText("Ref Z: {:.2f}".format(point[2]))

        if self.sim_em_pick_active:
            if self.sim_em_pick_index >= 3:
                return
            self.sim_candidate_em_point = point.copy()
            point_name = f"E{self.sim_em_pick_index + 1}"
            print(f"更新 MuJoCo 候选电磁点 {point_name}: {point}")
            self.sim_status_label.setText(
                f"MuJoCo 候选 {point_name}: X={point[0]:.2f}, Y={point[1]:.2f}, Z={point[2]:.2f}"
            )
            self.plotter.add_text(
                f"候选 {point_name}: X={point[0]:.2f}, Y={point[1]:.2f}, Z={point[2]:.2f}\n确认请点击右侧“记录标记点 {self.sim_em_pick_index + 1}”",
                position="upper_left",
                font_size=12,
                color="white",
                name="mujoco_pick_msg",
            )
            self.plotter.render()
            return

        self.plotter.add_text(
            "已更新 MuJoCo 虚拟电磁点，可点击记录标记点",
            position="upper_left",
            font_size=12,
            color="green",
            name="mujoco_pick_msg",
        )
        self.plotter.render()

    def _resolve_centerline_iges_path(self):
        candidates = [
            os.path.join(BASE_DIR, "老模型中心线.igs"),
            os.path.join(PROJECT_ROOT, "老模型中心线.igs"),
            os.path.join(os.getcwd(), "老模型中心线.igs"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        for directory in (BASE_DIR, PROJECT_ROOT):
            if not os.path.isdir(directory):
                continue
            for name in os.listdir(directory):
                if name.lower().endswith((".igs", ".iges")):
                    return os.path.join(directory, name)
        raise FileNotFoundError("未找到中心线 IGES 文件，请确认 window/老模型中心线.igs 存在。")

    def _resolve_segmentation_weights_path(self):
        weights_name = "1205weights_49.pth"
        candidates = [
            os.path.join(
                PROJECT_ROOT,
                "Visual_information",
                "models",
                "legacy_unet",
                weights_name,
            ),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return os.path.abspath(path)
        raise FileNotFoundError(
            f"未找到分割模型权重 {weights_name}，请确认 Visual_information/models/legacy_unet 目录完整。"
        )

    def _reset_segmentation_ui(self):
        self.seg_running = False
        self.seg_button.blockSignals(True)
        self.seg_button.setChecked(False)
        self.seg_button.setText("开始分割")
        self.seg_button.blockSignals(False)

    @QtCore.pyqtSlot(str)
    def _on_segmentation_error(self, message):
        self._reset_segmentation_ui()
        QMessageBox.critical(self, "分割模型加载失败", message)

    @QtCore.pyqtSlot()
    def _on_segmentation_thread_finished(self):
        self._reset_segmentation_ui()
        self.seg_worker = None
        self.seg_thread = None
