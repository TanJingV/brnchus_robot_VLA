"""Simulation-only methods extracted from the dual-probe main window.

Method bodies are intentionally unchanged; runtime globals are bound by the
host UI module after its classes and constants have been defined.
"""

from __future__ import annotations


def bind_runtime_globals(namespace):
    for name, value in namespace.items():
        if not name.startswith('__'):
            globals()[name] = value


class SimulationAutomationMixin:
    def _setup_automation_tab(self, tab_widget):
        tab = QWidget()
        tab_widget.addTab(tab, "自动化")
        outer = QVBoxLayout(tab)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)

        status_group = QtWidgets.QGroupBox("路径状态")
        status_grid = QtWidgets.QGridLayout(status_group)
        self.auto_path_ready_label = QLabel("路径规划: 未完成")
        self.auto_path_count_label = QLabel("路径点数量: 0")
        self.auto_target_label = QLabel("当前目标点: N/A")
        self.auto_position_label = QLabel("当前镜体位置: N/A")
        self.auto_progress_label = QLabel("跟踪进度: 0.0%")
        self.auto_progress_bar = QtWidgets.QProgressBar()
        self.auto_progress_bar.setRange(0, 1000)
        for index, widget in enumerate((
            self.auto_path_ready_label, self.auto_path_count_label, self.auto_target_label,
            self.auto_position_label, self.auto_progress_label, self.auto_progress_bar,
        )):
            status_grid.addWidget(widget, index // 2, index % 2)
        layout.addWidget(status_group)

        nav_group = QtWidgets.QGroupBox("自动导航")
        nav_layout = QHBoxLayout(nav_group)
        self.auto_nav_enable_cb = QtWidgets.QCheckBox("允许自动导航")
        self.auto_nav_enable_cb.toggled.connect(self._on_auto_nav_enabled)
        nav_layout.addWidget(self.auto_nav_enable_cb)
        self.auto_start_btn = QPushButton("启动")
        self.auto_pause_btn = QPushButton("暂停")
        self.auto_resume_btn = QPushButton("恢复")
        self.auto_stop_btn = QPushButton("停止")
        self.auto_start_btn.clicked.connect(self.start_auto_navigation)
        self.auto_pause_btn.clicked.connect(self.pause_auto_navigation)
        self.auto_resume_btn.clicked.connect(self.resume_auto_navigation)
        self.auto_stop_btn.clicked.connect(self.stop_auto_navigation)
        for button in (self.auto_start_btn, self.auto_pause_btn, self.auto_resume_btn, self.auto_stop_btn):
            nav_layout.addWidget(button)
        self.auto_nav_status_label = QLabel("状态: 已停止")
        self.auto_nav_status_label.setObjectName("statusPill")
        nav_layout.addWidget(self.auto_nav_status_label, 1)
        self.open_navigation_target_btn = QPushButton("打开目标观察")
        self.close_navigation_target_btn = QPushButton("关闭目标观察")
        self.open_navigation_target_btn.clicked.connect(lambda: self._toggle_navigation_target_window(True))
        self.close_navigation_target_btn.clicked.connect(lambda: self._toggle_navigation_target_window(False))
        nav_layout.addWidget(self.open_navigation_target_btn)
        nav_layout.addWidget(self.close_navigation_target_btn)
        layout.addWidget(nav_group)

        params_group = QtWidgets.QGroupBox("自动导航参数")
        params_form = QtWidgets.QFormLayout(params_group)
        self.auto_param_widgets = {}
        param_specs = (
            ("speed_scale", "速度比例", 0.05, 1.0, 0.65, 0.05, ""),
            ("lookahead", "前视距离", 1.0, 50.0, 10.0, 1.0, " mm"),
            ("point_threshold", "到点阈值", 0.2, 20.0, 3.0, 0.2, " mm"),
            ("max_step", "单步最大位移", 0.1, 10.0, 3.0, 0.1, " mm"),
            ("centerline_tolerance", "中心线允许偏差", 0.5, 30.0, 5.0, 0.5, " mm"),
        )
        for key, label, minimum, maximum, value, step, suffix in param_specs:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(2)
            spin.setValue(value)
            spin.setSingleStep(step)
            spin.setSuffix(suffix)
            self.auto_param_widgets[key] = spin
            params_form.addRow(label, spin)
        layout.addWidget(params_group)

        vla_group = QtWidgets.QGroupBox("VLA 训练数据采集")
        vla_layout = QVBoxLayout(vla_group)
        checks_grid = QtWidgets.QGridLayout()
        self.vla_capture_checks = {}
        capture_items = (
            ("real_image", "真实相机图像"), ("mujoco_image", "MuJoCo 虚拟图像"),
            ("mask", "分割 mask"), ("raw_pose", "原始 EM / MuJoCo 位姿"),
            ("model_pose", "肺模型坐标系位姿"), ("axis", "机器人轴位置 / 控制量"),
            ("action", "自动导航 action"), ("path_state", "路径状态"),
            ("calibration", "标定矩阵 R、t 和标定误差"),
        )
        for index, (key, text) in enumerate(capture_items):
            checkbox = QtWidgets.QCheckBox(text)
            checkbox.setChecked(True)
            self.vla_capture_checks[key] = checkbox
            checks_grid.addWidget(checkbox, index // 3, index % 3)
        vla_layout.addLayout(checks_grid)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("保存目录"))
        self.vla_save_path_edit = QLineEdit(os.path.join(PROJECT_ROOT, "vla_dataset"))
        path_layout.addWidget(self.vla_save_path_edit, 1)
        browse_btn = QPushButton("选择目录")
        browse_btn.clicked.connect(self.choose_vla_save_directory)
        path_layout.addWidget(browse_btn)
        vla_layout.addLayout(path_layout)

        controls = QHBoxLayout()
        self.vla_record_cb = QtWidgets.QCheckBox("启用 VLA 采集")
        controls.addWidget(self.vla_record_cb)
        self.vla_start_btn = QPushButton("开始 Episode")
        self.vla_stop_btn = QPushButton("结束 Episode")
        self.vla_start_btn.clicked.connect(self.start_vla_episode)
        self.vla_stop_btn.clicked.connect(self.stop_vla_episode)
        controls.addWidget(self.vla_start_btn)
        controls.addWidget(self.vla_stop_btn)
        self.vla_status_label = QLabel("未开始采集")
        self.vla_status_label.setObjectName("statusPill")
        controls.addWidget(self.vla_status_label, 1)
        vla_layout.addLayout(controls)
        layout.addWidget(vla_group)
        layout.addStretch()
        self.refresh_automation_path_status()

    def _on_auto_nav_enabled(self, enabled):
        self.auto_nav_enabled = bool(enabled)
        if not enabled:
            self.stop_auto_navigation()

    def _auto_param(self, name):
        return float(self.auto_param_widgets[name].value())

    def _toggle_navigation_target_window(self, checked):
        try:
            if checked:
                if self.navigation_target_window is None:
                    self.navigation_target_window = NavigationTargetWindow()
                self.navigation_target_window.show()
                self.navigation_target_window.raise_()
            elif self.navigation_target_window is not None:
                self.navigation_target_window.hide()
        except Exception as e:
            print(f"导航目标观察窗口切换失败: {e}")

    def _update_navigation_target_window(self, action):
        window = getattr(self, "navigation_target_window", None)
        if window is None or not window.isVisible():
            return
        frame = None
        if getattr(self, "is_simulation_mode", False) and self.mujoco_simulator is not None:
            frame = self.mujoco_simulator.latest_tip_frame_bgr
        if frame is None:
            frame = getattr(self, "latest_real_frame_bgr", None)
        window.update_guidance(frame, action)

    def refresh_automation_path_status(self):
        if not hasattr(self, "auto_path_ready_label"):
            return
        path = np.asarray(getattr(self, "smoothpath", []), dtype=float)
        valid = path.ndim == 2 and len(path) >= 2
        self.auto_path_ready_label.setText(f"路径规划: {'已完成' if valid else '未完成'}")
        tracking_path = self._get_auto_navigation_path()
        self.auto_path_count_label.setText(
            f"规划点数量: {len(path) if valid else 0} | 当前导航点数量: {len(tracking_path)}"
        )
        position = getattr(self, "virtual_position", None)
        if position is not None and len(position) == 3:
            self.auto_position_label.setText(
                f"当前镜体位置: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] mm"
            )
        else:
            self.auto_position_label.setText("当前镜体位置: N/A")
        if len(tracking_path) >= 2 and 0 <= self.auto_nav_target_index < len(tracking_path):
            target = tracking_path[self.auto_nav_target_index]
            self.auto_target_label.setText(
                f"当前目标点 {self.auto_nav_target_index + 1}/{len(tracking_path)}: "
                f"[{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}] mm"
            )
        else:
            self.auto_target_label.setText("当前目标点: N/A")
        self.auto_progress_label.setText(f"跟踪进度: {self.auto_nav_progress:.1f}%")
        self.auto_progress_bar.setValue(int(np.clip(self.auto_nav_progress * 10.0, 0, 1000)))

    def _update_auto_navigation_progress(self):
        path = self._get_auto_navigation_path()
        position = np.asarray(getattr(self, "virtual_position", []), dtype=float)
        if path.ndim != 2 or len(path) < 2 or position.shape != (3,):
            return
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total_length = float(np.sum(lengths))
        if total_length <= 1e-9:
            return
        target_index = int(np.clip(self.auto_nav_target_index, 1, len(path) - 1))
        segment_index = target_index - 1
        segment = path[target_index] - path[segment_index]
        segment_length = max(float(lengths[segment_index]), 1e-9)
        alpha = float(np.clip(
            np.dot(position - path[segment_index], segment)
            / (segment_length * segment_length),
            0.0,
            1.0,
        ))
        completed_length = float(np.sum(lengths[:segment_index])) + alpha * segment_length
        progress = 100.0 * completed_length / total_length
        self.auto_nav_progress = max(float(self.auto_nav_progress), float(progress))

    def start_auto_navigation(self):
        path = np.asarray(getattr(self, "smoothpath", []), dtype=float)
        if not self.auto_nav_enabled:
            QMessageBox.information(self, "自动导航", "请先勾选“允许自动导航”。")
            return
        if path.ndim != 2 or len(path) < 2:
            QMessageBox.warning(self, "自动导航", "尚未完成路径规划，无法启动自动导航。")
            return
        current_position = np.asarray(getattr(self, "virtual_position", []), dtype=float)
        if current_position.shape != (3,) or not np.isfinite(current_position).all():
            QMessageBox.warning(self, "自动导航", "尚未获得当前镜体位置，无法启动自动导航。")
            return
        tip_rotation = np.asarray(getattr(self, "mapped_tip_camera_dir", []), dtype=float)
        if tip_rotation.shape != (3, 3) or not np.isfinite(tip_rotation).all():
            QMessageBox.warning(self, "自动导航", "尚未获得完整 tip 位姿，无法启动自动导航。")
            return
        registration_ready = bool(getattr(self, "calibration_ready", False))
        if getattr(self, "is_simulation_mode", False):
            registration_ready = registration_ready or bool(
                getattr(self, "auto_simulation_mapping_ready", False)
            )
        rotation = np.asarray(getattr(self, "R", []), dtype=float)
        translation = np.asarray(getattr(self, "t", []), dtype=float)
        registration_ready = bool(
            registration_ready
            and rotation.shape == (3, 3)
            and translation.size == 3
            and np.isfinite(rotation).all()
            and np.isfinite(translation).all()
        )
        if not registration_ready:
            QMessageBox.warning(self, "自动导航", "尚未完成坐标标定或自动仿真映射，无法启动自动导航。")
            return
        path_distances = np.linalg.norm(path - current_position, axis=1)
        entry_index = int(np.argmin(path_distances))
        entry_distance = float(path_distances[entry_index])
        remaining_path = path[entry_index:]
        if entry_distance > self._auto_param("point_threshold"):
            approach_count = max(2, int(math.ceil(entry_distance / 2.0)) + 1)
            approach = np.linspace(current_position, remaining_path[0], approach_count)
            self.auto_nav_tracking_path = np.vstack([approach[:-1], remaining_path])
            self.auto_nav_status_label.setText(
                f"状态: 正在接入最近中心线点 {entry_index + 1}，距离 {entry_distance:.1f} mm"
            )
        elif entry_distance > 0.1:
            self.auto_nav_tracking_path = np.vstack([current_position, remaining_path])
        else:
            self.auto_nav_tracking_path = remaining_path.copy()
        tracking_path = self._get_auto_navigation_path()
        # Precompute the complete route geometry. The active controller still
        # closes the loop on the tip pose, but uses this profile to distinguish
        # true bends from harmless waypoint-to-waypoint direction changes.
        self.auto_nav_global_profile = self._build_auto_navigation_global_profile(tracking_path)
        self.auto_nav_target_index = 1 if len(tracking_path) > 1 else 0
        self.auto_nav_nearest_index = self.auto_nav_target_index
        self.auto_nav_passed_count = self.auto_nav_target_index
        self.auto_nav_progress = 0.0
        path_lengths = np.linalg.norm(np.diff(tracking_path, axis=0), axis=1)
        self.auto_nav_path_distance = float(np.sum(path_lengths[:self.auto_nav_nearest_index]))
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
        self.auto_nav_filtered_position = current_position.copy()
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
        self.auto_nav_last_position = current_position.copy()
        self.auto_nav_stall_frames = 0
        self.auto_nav_v2.reset(
            tracking_path,
            current_position,
            tip_rotation,
        )
        # Discard stale manual/automatic cable torques before the first
        # navigation frame. Otherwise a straight start can inherit a hard bend.
        self._apply_simulation_auto_action(None)
        self.auto_nav_state = "running"
        self._refresh_auto_navigation_guides()
        if entry_distance <= self._auto_param("point_threshold"):
            status = "状态: 仿真自动导航运行中" if self.is_simulation_mode else "状态: 真实模式，仅生成并缓存 action"
            self.auto_nav_status_label.setText(status)

    def pause_auto_navigation(self):
        if self.auto_nav_state == "running":
            self.auto_nav_state = "paused"
            self.auto_nav_status_label.setText("状态: 已暂停")
            self._apply_simulation_auto_action(None)

    def resume_auto_navigation(self):
        if self.auto_nav_state == "paused":
            self.auto_nav_pid_derivative[:] = 0.0
            self.auto_nav_pid_last_time = None
            self.auto_nav_state = "running"
            self.auto_nav_status_label.setText("状态: 自动导航运行中")

    def stop_auto_navigation(self):
        self.auto_nav_state = "stopped"
        self.auto_nav_action = None
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
        self.auto_nav_stall_frames = 0
        self.auto_nav_v2.reset()
        if hasattr(self, "auto_nav_status_label"):
            self.auto_nav_status_label.setText("状态: 已停止")
        self._apply_simulation_auto_action(None)

    def choose_vla_save_directory(self):
        try:
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self, "选择 VLA 数据保存目录", self.vla_save_path_edit.text().strip() or PROJECT_ROOT
            )
            if directory:
                self.vla_save_path_edit.setText(directory)
        except Exception as e:
            QMessageBox.warning(self, "目录选择失败", str(e))

    def _get_auto_navigation_path(self):
        tracking_path = getattr(self, "auto_nav_tracking_path", None)
        if tracking_path is not None:
            path = np.asarray(tracking_path, dtype=float)
            if path.ndim == 2 and len(path) >= 2:
                return path
        path = np.asarray(getattr(self, "smoothpath", []), dtype=float)
        return path if path.ndim == 2 else np.empty((0, 3), dtype=float)

    def _build_auto_navigation_global_profile(self, path):
        """Precompute smooth global path references for feedforward control."""
        points = np.asarray(path, dtype=float)
        if points.ndim != 2 or len(points) < 2:
            return None

        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        tangents = np.gradient(points, axis=0)
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents /= np.maximum(tangent_norms, 1e-9)

        # Symmetric smoothing removes action jumps caused by dense/noisy path
        # samples while preserving the global branch direction.
        smooth_tangents = np.zeros_like(tangents)
        for index in range(len(points)):
            start = max(0, index - 3)
            end = min(len(points), index + 4)
            weights = 1.0 / (1.0 + np.abs(np.arange(start, end) - index))
            averaged = np.sum(tangents[start:end] * weights[:, None], axis=0)
            smooth_tangents[index] = averaged / max(float(np.linalg.norm(averaged)), 1e-9)

        curve_vectors = np.zeros_like(points)
        curve_angles_deg = np.zeros(len(points), dtype=float)
        preview_angles_deg = np.zeros(len(points), dtype=float)
        reference_speed_scale = np.ones(len(points), dtype=float)
        reference_proximal = np.zeros(len(points), dtype=float)
        reference_distal = np.zeros(len(points), dtype=float)
        try:
            preview_horizon = max(float(self._auto_param("lookahead")), 3.0)
        except Exception:
            preview_horizon = 10.0
        for index in range(len(points)):
            before = max(0, index - 3)
            after = min(len(points) - 1, index + 3)
            delta_tangent = smooth_tangents[after] - smooth_tangents[before]
            curve_vectors[index] = delta_tangent
            curve_angles_deg[index] = math.degrees(math.acos(float(np.clip(
                np.dot(smooth_tangents[before], smooth_tangents[after]), -1.0, 1.0
            ))))
            preview_distance = min(
                float(cumulative[-1]), float(cumulative[index]) + preview_horizon
            )
            preview_after = int(np.clip(
                np.searchsorted(cumulative, preview_distance), index, len(points) - 1
            ))
            preview_angles_deg[index] = math.degrees(math.acos(float(np.clip(
                np.dot(smooth_tangents[index], smooth_tangents[preview_after]), -1.0, 1.0
            ))))
            anticipated = max(curve_angles_deg[index], preview_angles_deg[index])
            reference_speed_scale[index] = float(np.clip(1.0 - anticipated / 115.0, 0.45, 1.0))
            reference_proximal[index] = float(np.clip(anticipated / 115.0, 0.0, 0.62))
            reference_distal[index] = float(np.clip((anticipated - 28.0) / 105.0, 0.0, 0.42))

        return {
            "points": points.copy(),
            "cumulative": cumulative,
            "tangents": smooth_tangents,
            "curve_vectors": curve_vectors,
            "curve_angles_deg": curve_angles_deg,
            "preview_angles_deg": preview_angles_deg,
            "speed_scale": reference_speed_scale,
            "proximal_reference": reference_proximal,
            "distal_reference": reference_distal,
        }

    def _refresh_auto_navigation_guides(self):
        try:
            path = self._get_auto_navigation_path()
            index = int(np.clip(self.auto_nav_target_index, 0, max(0, len(path) - 1)))
            completed = len(path) > 0 and self.auto_nav_passed_count >= len(path)
            remaining = path[index:] if len(path) and not completed else np.empty((0, 3), dtype=float)
            target = path[index] if len(path) and index < len(path) and not completed else None

            for actor_name in ("path_actor", "path_actor_virtual"):
                actor = getattr(self, actor_name, None)
                if actor is not None:
                    try:
                        actor.SetVisibility(False)
                    except Exception:
                        pass

            for view, line_attr, target_attr in (
                (getattr(self, "plotter", None), "auto_remaining_path_actor", "auto_target_actor"),
                (getattr(self, "virtual_view", None), "auto_remaining_path_actor_virtual", "auto_target_actor_virtual"),
            ):
                if view is None:
                    continue
                old_line = getattr(self, line_attr, None)
                if old_line is not None:
                    try:
                        view.remove_actor(old_line)
                    except Exception:
                        pass
                    setattr(self, line_attr, None)
                old_target = getattr(self, target_attr, None)
                if old_target is not None:
                    try:
                        view.remove_actor(old_target)
                    except Exception:
                        pass
                    setattr(self, target_attr, None)
                if len(remaining) >= 2 and view is getattr(self, "plotter", None):
                    actor = view.add_mesh(pv.lines_from_points(remaining), color="#1687ff", line_width=5)
                    setattr(self, line_attr, actor)
                if target is not None and view is getattr(self, "plotter", None):
                    actor = view.add_mesh(pv.Sphere(radius=2.2, center=target), color="#ffb000")
                    setattr(self, target_attr, actor)
                view.render()

            if self.mujoco_simulator is not None:
                self.mujoco_simulator.set_navigation_waypoint_state(remaining, target)
        except Exception as e:
            print(f"自动导航引导刷新失败: {e}")

    def _compute_waypoint_visual_servo_action(self):
        """Stable point-to-point controller using only the active waypoint pose."""
        path = self._get_auto_navigation_path()
        position = getattr(self, "virtual_position", None)
        camera_rotation = getattr(self, "mapped_tip_camera_dir", None)
        if (
            path.ndim != 2 or len(path) < 2 or position is None
            or camera_rotation is None or np.asarray(camera_rotation).shape != (3, 3)
        ):
            return None

        position = np.asarray(position, dtype=float)
        camera_rotation = np.asarray(camera_rotation, dtype=float)
        target_index = int(np.clip(self.auto_nav_target_index, 1, len(path) - 1))
        target = np.asarray(path[target_index], dtype=float)
        if target_index != getattr(self, "auto_nav_servo_target_index", -1):
            self.auto_nav_servo_target_index = target_index
            self.auto_nav_target_transition_frames = 10
        else:
            self.auto_nav_target_transition_frames = max(
                0, int(getattr(self, "auto_nav_target_transition_frames", 0)) - 1
            )
        delta = target - position
        waypoint_distance = float(np.linalg.norm(delta))
        local = camera_rotation.T @ delta
        local_x, local_y, local_z = [float(value) for value in local]

        segment = np.asarray(path[target_index] - path[target_index - 1], dtype=float)
        segment_length = max(float(np.linalg.norm(segment)), 1e-9)
        segment_tangent = segment / segment_length
        reference_tangent = segment_tangent.copy()
        profile = getattr(self, "auto_nav_global_profile", None)
        profile_valid = bool(
            isinstance(profile, dict)
            and len(profile.get("tangents", [])) == len(path)
        )
        if profile_valid:
            reference_index = int(np.clip(target_index, 0, len(path) - 1))
            reference_tangent = np.asarray(profile["tangents"][reference_index], dtype=float)
            global_curve_vector = np.asarray(profile["curve_vectors"][reference_index], dtype=float)
            global_curve_angle_deg = float(profile["curve_angles_deg"][reference_index])
            global_preview_angle_deg = float(profile["preview_angles_deg"][reference_index])
            global_speed_scale = float(profile["speed_scale"][reference_index])
            global_proximal_reference = float(profile["proximal_reference"][reference_index])
            global_distal_reference = float(profile["distal_reference"][reference_index])
        else:
            global_curve_vector = np.zeros(3, dtype=float)
            global_curve_angle_deg = 0.0
            global_preview_angle_deg = 0.0
            global_speed_scale = 1.0
            global_proximal_reference = 0.0
            global_distal_reference = 0.0
        passed_distance = float(np.dot(position - target, segment_tangent))
        segment_progress = float(np.clip(
            np.dot(position - path[target_index - 1], segment) / (segment_length * segment_length),
            0.0,
            1.0,
        ))
        projected_position = path[target_index - 1] + segment_progress * segment
        cross_track_vector = projected_position - position
        cross_track = float(np.linalg.norm(cross_track_vector))
        threshold = self._auto_param("point_threshold")
        previous_position = (
            None if self.auto_nav_last_position is None
            else np.asarray(self.auto_nav_last_position, dtype=float).copy()
        )
        if previous_position is None:
            frame_motion = 0.0
            swept_waypoint_distance = waypoint_distance
        else:
            frame_vector = position - previous_position
            frame_motion = float(np.linalg.norm(frame_vector))
            if frame_motion > 1e-9:
                swept_alpha = float(np.clip(
                    np.dot(target - previous_position, frame_vector) / (frame_motion * frame_motion),
                    0.0,
                    1.0,
                ))
                swept_waypoint_distance = float(np.linalg.norm(
                    previous_position + swept_alpha * frame_vector - target
                ))
            else:
                swept_waypoint_distance = waypoint_distance
        # The continuous tip trajectory must enter every waypoint sphere, even
        # if a fast frame update samples just before and just after the point.
        waypoint_reached = bool(min(waypoint_distance, swept_waypoint_distance) <= threshold)
        self.auto_nav_last_position = position.copy()
        self.auto_nav_filtered_speed += 0.16 * (frame_motion - self.auto_nav_filtered_speed)

        # The waypoint's normalized image-plane displacement is the steering
        # error. No future path curvature or bifurcation prediction is used.
        forward_depth = max(local_z, 1.0)
        visual_error = np.asarray([local_x, local_y], dtype=float) / forward_depth
        visual_error_norm = float(np.linalg.norm(visual_error))
        bearing = np.asarray([
            math.atan2(local_x, max(local_z, 1e-6)),
            math.atan2(local_y, max(local_z, 1e-6)),
        ], dtype=float)

        # Vehicle-style lateral controller: combine path-heading alignment
        # with a speed-aware Stanley cross-track correction. On straight
        # segments the tangent dominates, preventing point-chasing oscillation.
        local_tangent = camera_rotation.T @ reference_tangent
        local_curve_vector = camera_rotation.T @ global_curve_vector
        tangent_bearing = np.asarray([
            math.atan2(float(local_tangent[0]), max(float(local_tangent[2]), 1e-6)),
            math.atan2(float(local_tangent[1]), max(float(local_tangent[2]), 1e-6)),
        ], dtype=float)
        local_cross_track = camera_rotation.T @ cross_track_vector
        cross_track_direction = np.asarray(local_cross_track[:2], dtype=float)
        cross_track_direction /= max(float(np.linalg.norm(cross_track_direction)), 1e-9)
        stanley_angle = math.atan2(
            0.32 * cross_track,
            max(self.auto_nav_filtered_speed, 0.08) + 0.45,
        )
        stanley_correction = cross_track_direction * min(stanley_angle, math.radians(18.0))
        tangent_error_deg = math.degrees(float(np.linalg.norm(tangent_bearing)))
        straight_segment = bool(tangent_error_deg < 12.0)
        if straight_segment:
            bearing = 0.78 * tangent_bearing + 0.22 * bearing + stanley_correction * 0.32
        else:
            bearing = 0.42 * tangent_bearing + 0.58 * bearing + stanley_correction * 0.20

        # Global-path feedforward: each waypoint already knows the bend
        # direction and magnitude required by the complete planned route.
        curve_direction_2d = np.asarray(local_curve_vector[:2], dtype=float)
        curve_direction_norm = float(np.linalg.norm(curve_direction_2d))
        if curve_direction_norm > 1e-6:
            curve_direction_2d /= curve_direction_norm
            bearing += curve_direction_2d * math.radians(global_preview_angle_deg) * 0.18
        # Low-pass the target bearing so switching to the next waypoint cannot
        # produce a step change in cable torque.
        bearing_filter = 0.12 if self.auto_nav_target_transition_frames > 0 else 0.20
        self.auto_nav_filtered_bearing += bearing_filter * (bearing - self.auto_nav_filtered_bearing)
        filtered_bearing = self.auto_nav_filtered_bearing.copy()
        bearing_angle_deg = math.degrees(float(np.linalg.norm(filtered_bearing)))

        # Ignore small image-plane errors to prevent straight-line oscillation.
        path_curve_reference = max(global_curve_angle_deg, global_preview_angle_deg)
        path_straight_reference = path_curve_reference < 8.0
        steering_deadband_deg = 5.0 if path_straight_reference and cross_track < threshold else 3.5
        if bearing_angle_deg < steering_deadband_deg and path_straight_reference:
            desired_proximal = np.zeros(2, dtype=float)
            desired_distal = np.zeros(2, dtype=float)
        else:
            desired_proximal = filtered_bearing * (0.16 if path_straight_reference else 0.25)
            if curve_direction_norm > 1e-6:
                desired_proximal += curve_direction_2d * global_proximal_reference * 0.22
            desired_proximal = desired_proximal * min(
                1.0, (0.10 if path_straight_reference else 0.20) / max(np.linalg.norm(desired_proximal), 1e-9)
            )

            # The distal section joins only when the active waypoint itself is
            # far from the image center; it never reacts to future path points.
            distal_ratio = 0.0 if path_straight_reference else float(np.clip(
                max(bearing_angle_deg - 24.0, path_curve_reference - 24.0) / 36.0, 0.0, 1.0
            ))
            desired_distal = filtered_bearing * 0.20 * distal_ratio
            if curve_direction_norm > 1e-6:
                desired_distal += curve_direction_2d * global_distal_reference * 0.18
            desired_distal = desired_distal * min(
                1.0, 0.15 / max(np.linalg.norm(desired_distal), 1e-9)
            )
        # Global feedforward remains continuous across waypoint boundaries;
        # strict slew limits prevent any residual feedback step.
        proximal_rate = 0.0012 if path_straight_reference else (
            0.0014 if self.auto_nav_target_transition_frames > 0 else 0.0025
        )
        distal_rate = 0.0010 if self.auto_nav_target_transition_frames > 0 else 0.0018
        self.auto_nav_proximal_command += np.clip(
            desired_proximal - self.auto_nav_proximal_command, -proximal_rate, proximal_rate
        )
        self.auto_nav_distal_command += np.clip(
            desired_distal - self.auto_nav_distal_command, -distal_rate, distal_rate
        )

        proximal = self.auto_nav_proximal_command.copy()
        distal = self.auto_nav_distal_command.copy()

        # Three-stage visual servo: align first, then approach, then dock at the
        # waypoint. This prevents a large bend from being requested while the
        # robot is still advancing toward an off-center target.
        max_step = self._auto_param("max_step") * self._auto_param("speed_scale")
        centering = float(np.clip(1.0 - visual_error_norm / 0.75, 0.12, 1.0))
        # Vehicle-style longitudinal planner. It estimates a stopping distance
        # from current motion and starts braking before the active waypoint.
        # If the tip crosses the waypoint plane without entering its tolerance,
        # controlled reverse is allowed until the waypoint is recovered.
        stopping_distance = max(
            threshold * 1.8,
            self.auto_nav_filtered_speed * 7.0 + max_step * 2.5,
        )
        overshot_waypoint = bool(passed_distance > threshold * 0.20 and waypoint_distance > threshold)
        final_waypoint = target_index >= len(path) - 1
        waypoint_requires_braking = bool(final_waypoint or path_curve_reference >= 16.0)
        if overshot_waypoint:
            servo_phase = "越过路径点，受控回撤"
            desired_insertion_step = -max_step * float(np.clip(
                0.22 + passed_distance / max(stopping_distance * 4.0, 1e-9),
                0.22,
                0.48,
            ))
        else:
            distance_clearance = max(0.0, waypoint_distance - threshold * 0.75)
            braking_factor = float(np.clip(
                math.sqrt(distance_clearance / max(stopping_distance, 1e-9)),
                0.16,
                1.0,
            ))
            heading_factor = float(np.clip(1.0 - bearing_angle_deg / 70.0, 0.42, 1.0))
            cross_track_factor = float(np.clip(
                threshold * 1.5 / max(cross_track, threshold * 1.5),
                0.55,
                1.0,
            ))
            if not waypoint_requires_braking:
                braking_factor = 1.0
            desired_insertion_step = (
                max_step * global_speed_scale * braking_factor * heading_factor * cross_track_factor
            )
            if waypoint_requires_braking and waypoint_distance <= stopping_distance:
                servo_phase = "提前减速到点"
            elif global_speed_scale < 0.78:
                servo_phase = "全局路径弯道限速"
            elif bearing_angle_deg > 14.0:
                servo_phase = "转向并持续进给"
            elif bearing_angle_deg > 7.0:
                servo_phase = "居中接近"
            else:
                servo_phase = "稳定巡航"

        # Acceleration and braking rate limits avoid throttle pulses between
        # dense waypoints. Braking/reverse responds faster than acceleration.
        previous_insertion = float(getattr(self, "auto_nav_last_insertion_step", 0.0))
        insertion_rate = max_step * (0.16 if desired_insertion_step >= previous_insertion else 0.32)
        insertion_step_mm = previous_insertion + float(np.clip(
            desired_insertion_step - previous_insertion,
            -insertion_rate,
            insertion_rate,
        ))
        if not overshot_waypoint:
            insertion_step_mm = max(insertion_step_mm, max_step * 0.12)
            insertion_step_mm = min(insertion_step_mm, max(waypoint_distance, 1e-4))
        self.auto_nav_last_insertion_step = insertion_step_mm

        self.auto_nav_nearest_index = target_index
        self.auto_nav_progress = 100.0 * self.auto_nav_passed_count / max(1, len(path))
        all_path_local = (camera_rotation.T @ (path - position).T).T
        return {
            "steering": distal.tolist(),
            "steering_proximal": proximal.tolist(),
            "steering_distal": distal.tolist(),
            "insertion_delta": float(insertion_step_mm / 577.0),
            "insertion_step_mm": float(insertion_step_mm),
            "target_index": target_index,
            "nearest_index": target_index,
            "target_model_mm": target.tolist(),
            "all_path_local_mm": all_path_local.tolist(),
            "distance_to_target_mm": waypoint_distance,
            "waypoint_reached": waypoint_reached,
            "swept_waypoint_distance_mm": swept_waypoint_distance,
            "waypoint_local_mm": local.tolist(),
            "target_local_mm": local.tolist(),
            "guidance_target_model_mm": target.tolist(),
            "guidance_target_distance_mm": waypoint_distance,
            "visual_center_error": visual_error.tolist(),
            "visual_error_norm": visual_error_norm,
            "visual_center_speed": centering,
            "lateral_distance_mm": float(math.hypot(local_x, local_y)),
            "azimuth_deg": math.degrees(float(filtered_bearing[0])),
            "elevation_deg": math.degrees(float(filtered_bearing[1])),
            "heading_error_deg": bearing_angle_deg,
            "servo_phase": servo_phase,
            "target_transition_frames": self.auto_nav_target_transition_frames,
            "stopping_distance_mm": stopping_distance,
            "overshot_waypoint": overshot_waypoint,
            "estimated_speed_mm_per_frame": self.auto_nav_filtered_speed,
            "straight_segment": straight_segment,
            "path_straight_reference": path_straight_reference,
            "global_curve_angle_deg": global_curve_angle_deg,
            "global_preview_angle_deg": global_preview_angle_deg,
            "global_speed_scale": global_speed_scale,
            "global_proximal_reference": global_proximal_reference,
            "global_distal_reference": global_distal_reference,
            "waypoint_requires_braking": waypoint_requires_braking,
            "tangent_error_deg": tangent_error_deg,
            "cross_track_error_mm": cross_track,
            "forward_alignment": float(local_z / max(waypoint_distance, 1e-9)),
            "straight_path_mode": straight_segment,
            "turn_active": not straight_segment,
            "turn_severity": float(np.clip(bearing_angle_deg / 55.0, 0.0, 1.0)),
            "distal_assist": float(np.clip((bearing_angle_deg - 18.0) / 35.0, 0.0, 1.0)),
            "proximal_utilization": float(np.linalg.norm(proximal) / 0.18),
            "cruise_boost": 1.0,
            "recovery_mode": overshot_waypoint,
            "retracting": insertion_step_mm < 0.0,
            "special_case": overshot_waypoint,
            "anti_curl_release": overshot_waypoint,
            "within_centerline_tolerance": cross_track <= self._auto_param("centerline_tolerance"),
        }

    def _compute_tip_target_pose_action(self):
        """Direct closed-loop control from tip pose to the active waypoint."""
        path = self._get_auto_navigation_path()
        position = getattr(self, "virtual_position", None)
        camera_rotation = getattr(self, "mapped_tip_camera_dir", None)
        if (
            path.ndim != 2 or len(path) < 2 or position is None
            or camera_rotation is None or np.asarray(camera_rotation).shape != (3, 3)
        ):
            return None

        position = np.asarray(position, dtype=float)
        camera_rotation = np.asarray(camera_rotation, dtype=float)
        target_index = int(np.clip(self.auto_nav_target_index, 1, len(path) - 1))
        target = np.asarray(path[target_index], dtype=float)
        error_world = target - position
        error_local = camera_rotation.T @ error_world
        local_x, local_y, local_z = [float(value) for value in error_local]
        distance_error = float(np.linalg.norm(error_world))
        lateral_error = float(math.hypot(local_x, local_y))
        threshold = self._auto_param("point_threshold")

        previous_position = (
            None if self.auto_nav_last_position is None
            else np.asarray(self.auto_nav_last_position, dtype=float).copy()
        )
        if previous_position is None:
            frame_motion = 0.0
        else:
            frame_motion = float(np.linalg.norm(position - previous_position))
        self.auto_nav_last_position = position.copy()
        self.auto_nav_filtered_speed += 0.18 * (frame_motion - self.auto_nav_filtered_speed)

        # Analyse the complete planned route before steering. On a geometrically
        # straight section, follow its stable tangent and correct cross-track
        # error instead of chasing each dense waypoint. In a genuine bend,
        # blend the active waypoint with a smooth forward route reference.
        heading_angle = float(math.atan2(lateral_error, local_z))
        previous_index = max(0, target_index - 1)
        segment = target - np.asarray(path[previous_index], dtype=float)
        segment_length = float(np.linalg.norm(segment))
        segment_tangent = segment / max(segment_length, 1e-9)
        target_plane_progress = float(np.dot(position - target, segment_tangent))
        target_plane_cross = float(np.linalg.norm(
            (position - target) - target_plane_progress * segment_tangent
        ))
        if previous_position is None:
            swept_waypoint_distance = distance_error
        else:
            motion = position - previous_position
            motion_length_sq = float(np.dot(motion, motion))
            if motion_length_sq <= 1e-12:
                swept_waypoint_distance = distance_error
            else:
                swept_fraction = float(np.clip(
                    np.dot(target - previous_position, motion) / motion_length_sq, 0.0, 1.0
                ))
                swept_point = previous_position + swept_fraction * motion
                swept_waypoint_distance = float(np.linalg.norm(target - swept_point))
        reference_tangent = segment_tangent.copy()
        path_curve_angle_deg = 0.0
        path_preview_angle_deg = 0.0
        profile = getattr(self, "auto_nav_global_profile", None)
        if (
            profile is not None
            and len(profile.get("points", [])) == len(path)
            and len(profile.get("tangents", [])) == len(path)
        ):
            reference_tangent = np.asarray(profile["tangents"][target_index], dtype=float)
            # The profile preview is already distance-based. Sampling future
            # indices again would make bend activation depend on point density
            # and can trigger a large bend far too early on a straight section.
            path_curve_angle_deg = float(profile["curve_angles_deg"][target_index])
            path_preview_angle_deg = float(profile["preview_angles_deg"][target_index])

        path_curve_metric = max(path_curve_angle_deg, path_preview_angle_deg)
        self.auto_nav_filtered_path_curve += 0.18 * (
            path_curve_metric - self.auto_nav_filtered_path_curve
        )
        # Hysteresis prevents rapidly switching between straight and bend
        # control around a noisy curvature threshold.
        if self.auto_nav_straight_mode:
            if self.auto_nav_filtered_path_curve > 10.0:
                self.auto_nav_straight_mode = False
        else:
            if self.auto_nav_filtered_path_curve < 5.0:
                self.auto_nav_straight_mode = True
        path_straight_mode = bool(self.auto_nav_straight_mode)

        line_origin = np.asarray(path[previous_index], dtype=float)
        projected_distance = float(np.dot(position - line_origin, reference_tangent))
        projected_point = line_origin + projected_distance * reference_tangent
        cross_track_world = projected_point - position
        cross_track_distance = float(np.linalg.norm(cross_track_world))
        lookahead_distance = max(self._auto_param("lookahead"), 5.0)
        centerline_tolerance = self._auto_param("centerline_tolerance")
        camera_forward = np.asarray(camera_rotation[:, 2], dtype=float)
        tangent_alignment = float(np.clip(np.dot(camera_forward, reference_tangent), -1.0, 1.0))
        tangent_misalignment_deg = math.degrees(math.acos(tangent_alignment))
        intermediate_waypoint = target_index < len(path) - 1
        waypoint_pass_tolerance = max(threshold * 1.5, centerline_tolerance)
        crossed_target_plane = target_plane_progress >= -max(0.25 * threshold, 0.3)
        passed_inside_corridor = bool(
            intermediate_waypoint
            and crossed_target_plane
            and target_plane_cross <= waypoint_pass_tolerance
        )
        swept_through_waypoint = bool(
            intermediate_waypoint and swept_waypoint_distance <= threshold
        )
        waypoint_reached = bool(
            distance_error <= threshold
            or passed_inside_corridor
            or swept_through_waypoint
        )

        if path_straight_mode:
            if self.auto_nav_straight_correction_active:
                if (
                    cross_track_distance < centerline_tolerance * 0.35
                    and tangent_misalignment_deg < 7.0
                ):
                    self.auto_nav_straight_correction_active = False
            elif (
                cross_track_distance > centerline_tolerance * 0.70
                or tangent_misalignment_deg > 15.0
            ):
                self.auto_nav_straight_correction_active = True

            correction_gain = float(np.clip(
                cross_track_distance / max(centerline_tolerance, 1e-6),
                0.0,
                1.0,
            ))
            if self.auto_nav_straight_correction_active:
                desired_world = (
                    reference_tangent
                    + cross_track_world * (0.55 * correction_gain / lookahead_distance)
                )
            else:
                # Inside the straight-line corridor, retain the current tip
                # direction and unload bending instead of constantly hunting
                # an ideal tangent that is already sufficiently close.
                desired_world = camera_forward.copy()
        else:
            self.auto_nav_straight_correction_active = False
            future_point = target
            if profile is not None and len(profile.get("cumulative", [])) == len(path):
                cumulative = np.asarray(profile["cumulative"], dtype=float)
                future_distance = min(
                    float(cumulative[-1]),
                    float(cumulative[target_index]) + lookahead_distance,
                )
                future_index = int(np.clip(
                    np.searchsorted(cumulative, future_distance), target_index, len(path) - 1
                ))
                future_point = np.asarray(path[future_index], dtype=float)
            direct_direction = error_world / max(distance_error, 1e-9)
            future_delta = future_point - position
            future_direction = future_delta / max(float(np.linalg.norm(future_delta)), 1e-9)
            desired_world = 0.62 * direct_direction + 0.38 * future_direction

        desired_world /= max(float(np.linalg.norm(desired_world)), 1e-9)
        target_heading_error = heading_error
        heading_error = math.acos(float(np.clip(
            np.dot(forward, desired_world), -1.0, 1.0
        )))
        desired_local = camera_rotation.T @ desired_world
        desired_lateral = np.asarray(desired_local[:2], dtype=float)
        desired_lateral_norm = float(np.linalg.norm(desired_lateral))
        control_heading_angle = float(math.atan2(
            desired_lateral_norm, max(float(desired_local[2]), -1.0)
        ))
        if desired_lateral_norm > 1e-6:
            turn_direction = desired_lateral / desired_lateral_norm
            self.auto_nav_last_turn_direction = turn_direction.copy()
        elif control_heading_angle > math.radians(90.0):
            turn_direction = self.auto_nav_last_turn_direction.copy()
            if np.linalg.norm(turn_direction) < 1e-6:
                turn_direction = np.array([1.0, 0.0], dtype=float)
        else:
            turn_direction = np.zeros(2, dtype=float)
        raw_bearing = turn_direction * control_heading_angle

        now = time.perf_counter()
        previous_pid_time = getattr(self, "auto_nav_pid_last_time", None)
        dt = 0.05 if previous_pid_time is None else float(np.clip(now - previous_pid_time, 0.02, 0.15))
        self.auto_nav_pid_last_time = now

        target_changed = target_index != getattr(self, "auto_nav_servo_target_index", -1)
        if target_changed:
            first_target = getattr(self, "auto_nav_servo_target_index", -1) < 0
            self.auto_nav_servo_target_index = target_index
            self.auto_nav_target_transition_frames = 3
            if first_target:
                self.auto_nav_pid_integral[:] = 0.0
                self.auto_nav_pid_derivative[:] = 0.0
                self.auto_nav_filtered_bearing = raw_bearing.copy()
                self.auto_nav_previous_heading_error = raw_bearing.copy()
            else:
                # Dense route points form one continuous path. Changing the
                # active gate must not restart steering, especially through a
                # bend where repeated PID resets create rhythmic oscillation.
                self.auto_nav_pid_integral *= 0.94
                self.auto_nav_pid_derivative *= 0.62
        else:
            self.auto_nav_target_transition_frames = max(0, self.auto_nav_target_transition_frames - 1)
        # Keep only a light measurement filter. Heavy filtering here creates
        # visible steering delay and makes the continuum robot overshoot.
        bearing_filter = 0.55 if self.auto_nav_target_transition_frames > 0 else 0.72
        self.auto_nav_filtered_bearing += bearing_filter * (raw_bearing - self.auto_nav_filtered_bearing)
        bearing = self.auto_nav_filtered_bearing.copy()
        bearing_angle_deg = math.degrees(float(np.linalg.norm(bearing)))

        # Stateful 2-D PID steering in the tip frame. Derivative filtering
        # suppresses pose noise, while integral limiting and saturation
        # rollback prevent wind-up when cable torque reaches its limit.
        raw_derivative = (bearing - self.auto_nav_previous_heading_error) / max(dt, 1e-6)
        self.auto_nav_previous_heading_error = bearing.copy()
        derivative_alpha = dt / (0.16 + dt)
        self.auto_nav_pid_derivative += derivative_alpha * (
            raw_derivative - self.auto_nav_pid_derivative
        )
        self.auto_nav_pid_integral += bearing * dt
        integral_limit = 0.38
        integral_norm = float(np.linalg.norm(self.auto_nav_pid_integral))
        if integral_norm > integral_limit:
            self.auto_nav_pid_integral *= integral_limit / integral_norm

        if path_straight_mode:
            kp, ki, kd = 0.24, 0.004, 0.20
            proximal_limit = 0.18
        else:
            # Continuum bending has substantial mechanical lag. Conservative
            # proportional/integral gains plus derivative damping avoid the
            # repeated over-correction that appears as side-to-side shaking.
            kp, ki, kd = 0.50, 0.018, 0.18
            proximal_limit = 0.42
        pid_output = (
            kp * bearing
            + ki * self.auto_nav_pid_integral
            + kd * self.auto_nav_pid_derivative
        )
        pid_norm = float(np.linalg.norm(pid_output))
        pid_saturated = pid_norm > proximal_limit
        if pid_saturated:
            pid_output *= proximal_limit / pid_norm
            self.auto_nav_pid_integral *= 0.94

        # Cable force may only bend toward the current geometric error.
        # Derivative damping is allowed to reduce force, but never to create an
        # active opposite-direction pull; elasticity provides the return force.
        if bearing_angle_deg > 1e-6:
            bearing_direction = bearing / max(float(np.linalg.norm(bearing)), 1e-9)
            pid_output = bearing_direction * max(float(np.dot(pid_output, bearing_direction)), 0.0)
        else:
            pid_output[:] = 0.0

        current_bend = self.auto_nav_proximal_command + 0.65 * self.auto_nav_distal_command
        current_bend_norm = float(np.linalg.norm(current_bend))
        requested_norm = float(np.linalg.norm(pid_output))
        opposite_bend_requested = bool(
            requested_norm > 0.018
            and current_bend_norm > 0.025
            and float(np.dot(pid_output, current_bend))
            < -0.12 * requested_norm * current_bend_norm
        )
        if opposite_bend_requested:
            self.auto_nav_elastic_release_active = True
            self.auto_nav_elastic_release_frames = 0

        if self.auto_nav_elastic_release_active:
            self.auto_nav_elastic_release_frames += 1
            self.auto_nav_pid_integral[:] = 0.0
            self.auto_nav_pid_derivative[:] = 0.0
            self.auto_nav_pid_output_filtered *= 0.48
            pid_output[:] = 0.0
            if (
                current_bend_norm < 0.012
                and float(np.linalg.norm(self.auto_nav_pid_output_filtered)) < 0.008
                and self.auto_nav_elastic_release_frames >= 4
            ):
                self.auto_nav_elastic_release_active = False
                self.auto_nav_elastic_release_frames = 0

        output_time_constant = 0.34 if path_straight_mode else 0.24
        output_alpha = dt / (output_time_constant + dt)
        self.auto_nav_pid_output_filtered += output_alpha * (
            pid_output - self.auto_nav_pid_output_filtered
        )
        pid_output = self.auto_nav_pid_output_filtered.copy()

        steering_deadband = 2.6 if (
            path_straight_mode
            and cross_track_distance < self._auto_param("centerline_tolerance") * 0.55
        ) else 1.4
        if self.auto_nav_elastic_release_active or bearing_angle_deg < steering_deadband:
            # A small deadband removes straight-line hunting without delaying
            # meaningful steering corrections.
            self.auto_nav_pid_integral *= 0.82
            proximal_target = np.zeros(2, dtype=float)
            distal_target = np.zeros(2, dtype=float)
            distal_assist = 0.0
            self.auto_nav_filtered_distal_assist *= 0.72
        else:
            proximal_target = pid_output
            requested_distal_assist = 0.0 if path_straight_mode else float(np.clip(
                max(bearing_angle_deg - 28.0, self.auto_nav_filtered_path_curve - 18.0) / 42.0,
                0.0,
                1.0,
            ))
            distal_assist_alpha = dt / (0.42 + dt)
            self.auto_nav_filtered_distal_assist += distal_assist_alpha * (
                requested_distal_assist - self.auto_nav_filtered_distal_assist
            )
            distal_assist = self.auto_nav_filtered_distal_assist
            distal_target = pid_output * 0.68 * distal_assist
            distal_limit = 0.30
            distal_norm = float(np.linalg.norm(distal_target))
            if distal_norm > distal_limit:
                distal_target *= distal_limit / distal_norm

        def slew_vector(current, desired, rate_per_second):
            delta = desired - current
            delta_norm = float(np.linalg.norm(delta))
            max_delta = rate_per_second * dt
            if delta_norm > max_delta:
                delta *= max_delta / delta_norm
            return current + delta

        self.auto_nav_proximal_command = slew_vector(
            self.auto_nav_proximal_command, proximal_target, 0.36 if path_straight_mode else 0.60
        )
        self.auto_nav_distal_command = slew_vector(
            self.auto_nav_distal_command, distal_target, 0.42 if path_straight_mode else 0.45
        )
        if self.auto_nav_elastic_release_active:
            # Remove cable commands immediately. The continuum's own elasticity
            # returns both sections toward centre without any active callback.
            self.auto_nav_proximal_command[:] = 0.0
            self.auto_nav_distal_command[:] = 0.0
        proximal = self.auto_nav_proximal_command.copy()
        distal = self.auto_nav_distal_command.copy()

        # Insertion behaves like a throttle: navigation continuously advances
        # with a minimum cruise command. It slows for large steering errors,
        # and reverses only after the tip has clearly passed the active point.
        max_step = self._auto_param("max_step") * self._auto_param("speed_scale")
        alignment = float(np.clip(math.cos(control_heading_angle), -1.0, 1.0))
        # Intermediate points are route gates, not parking targets. Never
        # reverse to chase an old gate after crossing it; immediately advance
        # to the next point. Precision reverse remains available at the final
        # destination only.
        clearly_overshot = bool(
            not intermediate_waypoint
            and local_z < -max(threshold * 0.75, 1.2)
        )
        if clearly_overshot:
            desired_insertion = -max_step * float(np.clip(-alignment, 0.20, 0.45))
            servo_phase = "越过目标点，PID 受控回撤"
        else:
            curve_severity = float(np.clip(
                max(
                    self.auto_nav_filtered_path_curve / 65.0,
                    bearing_angle_deg / 80.0,
                    cross_track_distance / max(centerline_tolerance * 2.2, 1e-6),
                    0.85 if pid_saturated else 0.0,
                ),
                0.0,
                1.0,
            ))
            if self.auto_nav_elastic_release_active:
                curve_severity = max(curve_severity, 0.82)
            # Keep a healthy cruise throttle on straight sections. Through a
            # large bend, slow progressively so cable curvature can settle
            # before insertion pushes the tip past the centreline.
            curvature_throttle = float(np.clip(1.0 - 0.78 * curve_severity, 0.18, 1.0))
            minimum_throttle = 0.62 if path_straight_mode else float(
                np.clip(0.34 - 0.18 * curve_severity, 0.14, 0.34)
            )
            heading_throttle = float(np.clip(
                (alignment + 0.35) / 1.35, minimum_throttle, 1.0
            ))
            distance_throttle = float(np.clip(
                distance_error / max(threshold * 2.8, 4.0), 0.38, 1.0
            ))
            desired_insertion = max_step * max(
                minimum_throttle,
                heading_throttle * distance_throttle * curvature_throttle,
            )
            servo_phase = "全局直线路径稳定巡航" if path_straight_mode else "全局弯道 PID 跟踪"
            if intermediate_waypoint and local_z < 0.0:
                servo_phase = "已穿过中间路径点，持续向前"
            if not path_straight_mode and curve_severity > 0.68:
                servo_phase = "大弯曲稳定建弯，动态降低进给"
            if self.auto_nav_elastic_release_active:
                servo_phase = "停止弯曲控制，弹性自然回中"

        previous_insertion = float(getattr(self, "auto_nav_last_insertion_step", 0.0))
        insertion_rate = max_step * (0.22 if desired_insertion < previous_insertion else 0.32)
        insertion_step_mm = previous_insertion + float(np.clip(
            desired_insertion - previous_insertion, -insertion_rate, insertion_rate
        ))
        self.auto_nav_last_insertion_step = insertion_step_mm

        forward_depth = max(abs(local_z), 1.0)
        visual_error = np.asarray([local_x, local_y], dtype=float) / forward_depth
        visual_error_norm = float(np.linalg.norm(visual_error))
        all_path_local = (camera_rotation.T @ (path - position).T).T
        self.auto_nav_nearest_index = target_index
        self.auto_nav_progress = 100.0 * self.auto_nav_passed_count / max(1, len(path))

        return {
            "steering": distal.tolist(),
            "steering_proximal": proximal.tolist(),
            "steering_distal": distal.tolist(),
            "insertion_delta": float(insertion_step_mm / 577.0),
            "insertion_step_mm": float(insertion_step_mm),
            "target_index": target_index,
            "nearest_index": target_index,
            "target_model_mm": target.tolist(),
            "all_path_local_mm": all_path_local.tolist(),
            "distance_to_target_mm": distance_error,
            "waypoint_reached": waypoint_reached,
            "waypoint_pass_tolerance_mm": waypoint_pass_tolerance,
            "target_plane_progress_mm": target_plane_progress,
            "target_plane_cross_mm": target_plane_cross,
            "crossed_target_plane": crossed_target_plane,
            "passed_inside_corridor": passed_inside_corridor,
            "swept_waypoint_distance_mm": swept_waypoint_distance,
            "waypoint_local_mm": error_local.tolist(),
            "target_local_mm": error_local.tolist(),
            "guidance_target_model_mm": target.tolist(),
            "guidance_target_distance_mm": distance_error,
            "relative_position_error_mm": error_local.tolist(),
            "relative_bearing_error_deg": bearing_angle_deg,
            "pid_dt_s": dt,
            "pid_p": (kp * bearing).tolist(),
            "pid_i": (ki * self.auto_nav_pid_integral).tolist(),
            "pid_d": (kd * self.auto_nav_pid_derivative).tolist(),
            "pid_saturated": pid_saturated,
            "steering_reversal_frames": self.auto_nav_steering_reversal_frames,
            "elastic_release_active": bool(self.auto_nav_elastic_release_active),
            "elastic_release_frames": self.auto_nav_elastic_release_frames,
            "curve_severity": curve_severity if not clearly_overshot else 1.0,
            "curvature_throttle": curvature_throttle if not clearly_overshot else 0.0,
            "visual_center_error": visual_error.tolist(),
            "visual_error_norm": visual_error_norm,
            "visual_center_speed": float(np.clip(1.0 - visual_error_norm, 0.0, 1.0)),
            "lateral_distance_mm": cross_track_distance,
            "azimuth_deg": math.degrees(float(bearing[0])),
            "elevation_deg": math.degrees(float(bearing[1])),
            "heading_error_deg": bearing_angle_deg,
            "servo_phase": servo_phase,
            "target_transition_frames": self.auto_nav_target_transition_frames,
            "estimated_speed_mm_per_frame": self.auto_nav_filtered_speed,
            "stopping_distance_mm": 0.0,
            "overshot_waypoint": local_z < 0.0,
            "straight_segment": path_straight_mode,
            "tangent_error_deg": tangent_misalignment_deg,
            "global_curve_angle_deg": path_curve_angle_deg,
            "global_preview_angle_deg": path_preview_angle_deg,
            "global_speed_scale": 1.0,
            "global_proximal_reference": 0.0,
            "global_distal_reference": 0.0,
            "waypoint_requires_braking": False,
            "straight_path_mode": path_straight_mode,
            "straight_correction_active": bool(self.auto_nav_straight_correction_active),
            "turn_active": not path_straight_mode,
            "turn_severity": float(np.clip(
                max(bearing_angle_deg, self.auto_nav_filtered_path_curve) / 70.0, 0.0, 1.0
            )),
            "distal_assist": distal_assist,
            "proximal_utilization": float(np.linalg.norm(proximal) / 0.24),
            "cruise_boost": 1.0,
            "recovery_mode": local_z < 0.0,
            "retracting": insertion_step_mm < 0.0,
            "special_case": local_z < 0.0,
            "anti_curl_release": False,
            "cross_track_error_mm": cross_track_distance,
            "forward_alignment": float(local_z / max(distance_error, 1e-9)),
            "within_centerline_tolerance": (
                cross_track_distance <= self._auto_param("centerline_tolerance")
            ),
        }

    def _compute_waypoint_elastic_action(self):
        """Track one waypoint at a time with elastic, non-reversing bend control."""
        path = self._get_auto_navigation_path()
        raw_position = np.asarray(getattr(self, "virtual_position", []), dtype=float)
        camera_rotation = np.asarray(getattr(self, "mapped_tip_camera_dir", []), dtype=float)
        if (
            path.ndim != 2 or len(path) < 2 or raw_position.shape != (3,)
            or camera_rotation.shape != (3, 3) or not np.isfinite(raw_position).all()
        ):
            return None

        # Low-pass the measured tip pose. Orientation filtering uses the forward
        # axis only, preserving the camera frame for image-plane steering.
        if self.auto_nav_filtered_position is None:
            self.auto_nav_filtered_position = raw_position.copy()
        self.auto_nav_filtered_position += 0.38 * (
            raw_position - self.auto_nav_filtered_position
        )
        position = self.auto_nav_filtered_position.copy()
        raw_forward = np.asarray(camera_rotation[:, 2], dtype=float)
        raw_forward /= max(float(np.linalg.norm(raw_forward)), 1e-9)
        if self.auto_nav_filtered_forward is None:
            self.auto_nav_filtered_forward = raw_forward.copy()
        self.auto_nav_filtered_forward += 0.24 * (
            raw_forward - self.auto_nav_filtered_forward
        )
        self.auto_nav_filtered_forward /= max(
            float(np.linalg.norm(self.auto_nav_filtered_forward)), 1e-9
        )
        forward = self.auto_nav_filtered_forward.copy()

        target_index = int(np.clip(self.auto_nav_target_index, 1, len(path) - 1))
        target_changed = target_index != self.auto_nav_waypoint_controller_index
        if target_changed:
            self.auto_nav_waypoint_controller_index = target_index
            self.auto_nav_waypoint_reached_frames = 0

        previous_point = np.asarray(path[target_index - 1], dtype=float)
        target = np.asarray(path[target_index], dtype=float)
        next_index = min(target_index + 1, len(path) - 1)
        intermediate_waypoint = target_index < len(path) - 1
        next_point = np.asarray(path[next_index], dtype=float)
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))

        def sample_path_at_s(distance):
            distance = float(np.clip(distance, 0.0, cumulative[-1]))
            segment = int(np.clip(
                np.searchsorted(cumulative, distance, side="right") - 1,
                0,
                len(path) - 2,
            ))
            length = max(float(segment_lengths[segment]), 1e-9)
            alpha = float(np.clip(
                (distance - cumulative[segment]) / length, 0.0, 1.0
            ))
            point = (1.0 - alpha) * path[segment] + alpha * path[segment + 1]
            return np.asarray(point, dtype=float), segment

        target_s = float(cumulative[target_index])
        tangent_span = float(np.clip(max(self._auto_param("lookahead") * 0.35, 3.0), 3.0, 6.0))
        before_point, _ = sample_path_at_s(target_s - tangent_span)
        after_point, _ = sample_path_at_s(target_s + tangent_span)
        preview_point, preview_index = sample_path_at_s(
            target_s + max(self._auto_param("lookahead"), 6.0)
        )
        incoming = target - before_point
        outgoing = after_point - target
        if np.linalg.norm(incoming) <= 1e-9:
            incoming = target - previous_point
        if np.linalg.norm(outgoing) <= 1e-9:
            outgoing = next_point - target if next_index > target_index else incoming.copy()
        incoming /= max(float(np.linalg.norm(incoming)), 1e-9)
        outgoing /= max(float(np.linalg.norm(outgoing)), 1e-9)
        curve_angle = math.acos(float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0)))
        preview_direction = preview_point - target
        if np.linalg.norm(preview_direction) <= 1e-9:
            preview_direction = outgoing.copy()
        preview_direction /= max(float(np.linalg.norm(preview_direction)), 1e-9)
        preview_angle = math.acos(float(np.clip(
            np.dot(incoming, preview_direction), -1.0, 1.0
        )))
        bend_angle = max(curve_angle, preview_angle)
        preview_arc_length = max(
            0.0,
            min(float(cumulative[-1]), target_s + max(self._auto_param("lookahead"), 6.0))
            - target_s,
        )
        # The same turn angle requires much more bending when it is packed
        # into a short arc. Include curvature (angle / arc length) so tight
        # branches engage the distal section more strongly than broad bends.
        tight_curve_severity = float(np.clip(
            bend_angle / max(preview_arc_length, 1.0) * 12.0,
            0.0,
            1.0,
        ))
        curve_severity = float(np.clip(max(
            bend_angle / math.radians(50.0),
            tight_curve_severity,
        ), 0.0, 1.0))
        # Curvature enters quickly but decays slowly. Closely spaced points in
        # one physical bend must not repeatedly switch the controller between
        # bend and straight states.
        if curve_severity > self.auto_nav_curve_memory:
            curve_memory_alpha = 0.32
        elif curve_severity < 0.05:
            # Arc-length curvature confirms that the physical bend has ended.
            # Release smoothly but promptly; nearby points inside a bend do
            # not reach this branch because their arc curvature stays nonzero.
            curve_memory_alpha = 0.10
        else:
            curve_memory_alpha = 0.045
        self.auto_nav_curve_memory += curve_memory_alpha * (
            curve_severity - self.auto_nav_curve_memory
        )
        if self.auto_nav_curve_memory < 0.06 and curve_severity < 0.04:
            self.auto_nav_curve_memory = 0.0
        effective_curve_severity = float(max(
            curve_severity,
            self.auto_nav_curve_memory,
        ))
        threshold = self._auto_param("point_threshold")
        tolerance = self._auto_param("centerline_tolerance")

        error_world = target - position
        distance_error = float(np.linalg.norm(error_world))
        target_direction = error_world / max(distance_error, 1e-9)
        error_local = camera_rotation.T @ error_world
        lateral_error = float(np.linalg.norm(error_local[:2]))
        forward_error = float(error_local[2])
        heading_error = math.acos(float(np.clip(np.dot(forward, target_direction), -1.0, 1.0)))
        path_lateral_error = float(np.linalg.norm(
            error_world - float(np.dot(error_world, incoming)) * incoming
        ))

        # Control uses an arc-length guidance point, not the current gate.
        # The gate remains responsible for waypoint completion only. This
        # prevents clusters of nearby points from producing bend/straight
        # oscillation as the target index advances.
        guidance_lookahead = float(np.clip(
            max(
                self._auto_param("lookahead") * (0.42 if effective_curve_severity > 0.25 else 0.70),
                threshold * 2.5,
                4.0,
            ),
            4.0,
            14.0,
        ))
        guidance_s = min(
            float(cumulative[-1]),
            float(cumulative[target_index]) + guidance_lookahead,
        )
        guidance_segment = int(np.clip(
            np.searchsorted(cumulative, guidance_s, side="right") - 1,
            target_index,
            len(path) - 2,
        ))
        guidance_segment_length = max(float(segment_lengths[guidance_segment]), 1e-9)
        guidance_alpha = float(np.clip(
            (guidance_s - cumulative[guidance_segment]) / guidance_segment_length,
            0.0,
            1.0,
        ))
        guidance_point = (
            (1.0 - guidance_alpha) * path[guidance_segment]
            + guidance_alpha * path[guidance_segment + 1]
        )
        guidance_delta = np.asarray(guidance_point, dtype=float) - position
        guidance_direction = guidance_delta / max(float(np.linalg.norm(guidance_delta)), 1e-9)
        guidance_tangent = np.asarray(
            path[guidance_segment + 1] - path[guidance_segment], dtype=float
        )
        guidance_tangent /= max(float(np.linalg.norm(guidance_tangent)), 1e-9)
        correction_weight = float(np.clip(
            path_lateral_error / max(tolerance * 1.8, 1e-9), 0.12, 0.72
        ))
        # The arc-length preview supplies smooth route anticipation, but the
        # active waypoint remains a mandatory gate. Near a tight turn or with
        # a large cross-track error, steer increasingly toward that gate
        # instead of cutting the corner toward a point farther down the path.
        waypoint_capture_weight = float(np.clip(
            0.16
            + 0.48 * effective_curve_severity
            + 0.26 * np.clip(
                1.0 - distance_error / max(guidance_lookahead * 1.8, 1e-9),
                0.0,
                1.0,
            )
            + 0.18 * np.clip(
                path_lateral_error / max(tolerance * 2.0, 1e-9),
                0.0,
                1.0,
            ),
            0.16,
            0.88,
        ))
        route_direction = (
            correction_weight * guidance_direction
            + (1.0 - correction_weight) * guidance_tangent
        )
        desired_world = (
            waypoint_capture_weight * target_direction
            + (1.0 - waypoint_capture_weight) * route_direction
        )
        desired_world /= max(float(np.linalg.norm(desired_world)), 1e-9)
        if self.auto_nav_guidance_direction is None:
            self.auto_nav_guidance_direction = desired_world.copy()
        guidance_direction_alpha = 0.08 if effective_curve_severity < 0.20 else 0.13
        blended_guidance = (
            (1.0 - guidance_direction_alpha) * self.auto_nav_guidance_direction
            + guidance_direction_alpha * desired_world
        )
        self.auto_nav_guidance_direction = blended_guidance / max(
            float(np.linalg.norm(blended_guidance)), 1e-9
        )
        desired_world = self.auto_nav_guidance_direction.copy()
        target_heading_error = heading_error
        heading_error = math.acos(float(np.clip(
            np.dot(forward, desired_world), -1.0, 1.0
        )))
        desired_local = camera_rotation.T @ desired_world
        desired_lateral = np.asarray(desired_local[:2], dtype=float)
        desired_lateral_norm = float(np.linalg.norm(desired_lateral))
        requested_direction = (
            desired_lateral / desired_lateral_norm
            if desired_lateral_norm > 1e-6 else np.zeros(2, dtype=float)
        )

        target_plane_progress = float(np.dot(position - target, incoming))
        raw_target_plane_progress = float(np.dot(raw_position - target, incoming))
        target_plane_cross = path_lateral_error
        inside_target = distance_error <= threshold
        crossed_target = bool(
            target_index < len(path) - 1
            and target_plane_progress >= 0.0
            and target_plane_cross <= max(threshold * 1.5, tolerance)
        )
        if inside_target or crossed_target:
            self.auto_nav_waypoint_reached_frames += 1
        else:
            self.auto_nav_waypoint_reached_frames = 0
        waypoint_reached = self.auto_nav_waypoint_reached_frames >= 2

        # Motion-direction state machine. A correctly crossed intermediate
        # gate advances normally; reverse is reserved for a target that stays
        # behind the tip after the tip has physically crossed its path-normal
        # plane while outside the valid centreline pass corridor. A target can
        # legitimately appear behind the camera during a tight bend before
        # that plane is crossed; reversing there causes forward/reverse loops.
        behind_threshold = max(threshold * 0.75, 0.8)
        target_plane_overshot = bool(
            raw_target_plane_progress > max(threshold * 0.55, 0.7)
        )
        target_confirmed_behind = bool(
            forward_error < -behind_threshold
            and distance_error > threshold
            and not crossed_target
            and target_plane_overshot
            and target_plane_cross > max(threshold * 1.25, tolerance * 0.75)
        )
        target_confirmed_front = bool(
            forward_error > max(threshold * 0.55, 0.6)
            or distance_error <= threshold
            or raw_target_plane_progress <= max(threshold * 0.10, 0.15)
        )
        if target_confirmed_behind:
            self.auto_nav_behind_frames += 1
            self.auto_nav_front_frames = 0
        elif target_confirmed_front:
            self.auto_nav_front_frames += 1
            self.auto_nav_behind_frames = max(0, self.auto_nav_behind_frames - 1)
        else:
            self.auto_nav_behind_frames = max(0, self.auto_nav_behind_frames - 1)
            self.auto_nav_front_frames = max(0, self.auto_nav_front_frames - 1)

        if (
            self.auto_nav_motion_mode != "reverse"
            and self.auto_nav_behind_frames >= 5
        ):
            self.auto_nav_motion_mode = "reverse"
            self.auto_nav_front_frames = 0
            self.auto_nav_forward_reengage_frames = 0
        elif (
            self.auto_nav_motion_mode == "reverse"
            and self.auto_nav_front_frames >= 4
        ):
            self.auto_nav_motion_mode = "forward"
            self.auto_nav_behind_frames = 0
            # Let insertion pass continuously through zero after a recovery
            # retraction; forcing the normal positive-feed floor here creates
            # a sharp longitudinal impulse and a new bending overshoot.
            self.auto_nav_forward_reengage_frames = 8
        reversing_to_recover = self.auto_nav_motion_mode == "reverse"
        forward_reengage_active = bool(
            not reversing_to_recover
            and self.auto_nav_forward_reengage_frames > 0
        )
        if forward_reengage_active:
            self.auto_nav_forward_reengage_frames -= 1

        # Damped 2-D PID in the tip camera plane. The error vector is angular,
        # so gains are independent of waypoint spacing. A filtered derivative
        # supplies damping for the elastic continuum sections.
        control_heading = float(math.atan2(
            desired_lateral_norm, max(float(desired_local[2]), -0.20)
        ))
        raw_bearing = requested_direction * control_heading
        now = time.perf_counter()
        previous_pid_time = getattr(self, "auto_nav_pid_last_time", None)
        dt = 0.05 if previous_pid_time is None else float(
            np.clip(now - previous_pid_time, 0.02, 0.12)
        )
        self.auto_nav_pid_last_time = now

        bearing_alpha = dt / (0.14 + dt)
        self.auto_nav_filtered_bearing += bearing_alpha * (
            raw_bearing - self.auto_nav_filtered_bearing
        )
        bearing = self.auto_nav_filtered_bearing.copy()
        bearing_angle = float(np.linalg.norm(bearing))
        bearing_angle_deg = math.degrees(bearing_angle)
        previous_bearing_angle = self.auto_nav_last_bearing_angle_deg
        raw_improvement = (
            0.0 if previous_bearing_angle is None
            else float(previous_bearing_angle - bearing_angle_deg)
        )
        self.auto_nav_last_bearing_angle_deg = bearing_angle_deg
        self.auto_nav_bearing_improvement += 0.20 * (
            raw_improvement - self.auto_nav_bearing_improvement
        )
        raw_derivative = (
            bearing - self.auto_nav_previous_heading_error
        ) / max(dt, 1e-6)
        raw_derivative_norm = float(np.linalg.norm(raw_derivative))
        if raw_derivative_norm > 0.65:
            raw_derivative *= 0.65 / raw_derivative_norm
        self.auto_nav_previous_heading_error = bearing.copy()
        derivative_alpha = dt / (0.24 + dt)
        self.auto_nav_pid_derivative += derivative_alpha * (
            raw_derivative - self.auto_nav_pid_derivative
        )

        # Integral is only useful for a persistent moderate bias. Leak it near
        # centre and during large turns to prevent elastic wind-up.
        if math.radians(3.0) < bearing_angle < math.radians(38.0):
            self.auto_nav_pid_integral += bearing * dt
        else:
            self.auto_nav_pid_integral *= 0.90
        integral_limit = 0.22
        integral_norm = float(np.linalg.norm(self.auto_nav_pid_integral))
        if integral_norm > integral_limit:
            self.auto_nav_pid_integral *= integral_limit / integral_norm

        # A geometrically straight route is only a gentle-control case while
        # the tip is already close and roughly aligned. Once the target has a
        # large bearing or cross-track error, full two-section steering is
        # required even if the centreline segment itself is straight.
        gentle_path_mode = bool(
            curve_severity < 0.22
            and effective_curve_severity < 0.25
            and bearing_angle_deg < 24.0
            and path_lateral_error < tolerance * 2.4
        )
        straight_error_exceeded = bool(
            path_lateral_error > tolerance * 0.78
            or bearing_angle_deg > 10.0
        )
        straight_error_clear = bool(
            path_lateral_error < tolerance * 0.42
            and bearing_angle_deg < 5.0
        )
        if gentle_path_mode:
            if straight_error_exceeded:
                self.auto_nav_straight_error_frames += 1
                self.auto_nav_straight_clear_frames = 0
            elif straight_error_clear:
                self.auto_nav_straight_clear_frames += 1
                self.auto_nav_straight_error_frames = max(
                    0, self.auto_nav_straight_error_frames - 1
                )
            else:
                self.auto_nav_straight_error_frames = max(
                    0, self.auto_nav_straight_error_frames - 1
                )
                self.auto_nav_straight_clear_frames = max(
                    0, self.auto_nav_straight_clear_frames - 1
                )
            if self.auto_nav_straight_correction_active:
                if self.auto_nav_straight_clear_frames >= 12:
                    self.auto_nav_straight_correction_active = False
            elif self.auto_nav_straight_error_frames >= 8:
                self.auto_nav_straight_correction_active = True
        else:
            self.auto_nav_straight_error_frames = 0
            self.auto_nav_straight_clear_frames = 0
            self.auto_nav_straight_correction_active = False

        straight_neutral_mode = bool(
            gentle_path_mode and not self.auto_nav_straight_correction_active
        )
        straight_correction_mode = bool(
            gentle_path_mode and self.auto_nav_straight_correction_active
        )
        straight_pid_mode = gentle_path_mode
        if straight_pid_mode:
            kp, ki, kd = 0.08, 0.0015, 0.10
            proximal_limit = 0.045
            steering_deadband_deg = 4.5
        else:
            kp, ki, kd = 0.20, 0.006, 0.14
            proximal_limit = 0.18
            steering_deadband_deg = 1.8

        pid_output = (
            kp * bearing
            + ki * self.auto_nav_pid_integral
            + kd * self.auto_nav_pid_derivative
        )
        # A small bend feed-forward helps establish a large curve without
        # raising PID gains enough to excite the elastic robot.
        if effective_curve_severity > 0.28 and bearing_angle_deg > steering_deadband_deg:
            pid_output += requested_direction * (0.020 * effective_curve_severity)

        # Cables may pull only toward the current geometric error. The D term
        # may unload that pull, but can never create an active return command.
        if bearing_angle > 1e-9:
            bearing_direction = bearing / bearing_angle
            pid_output = bearing_direction * max(
                float(np.dot(pid_output, bearing_direction)), 0.0
            )
            direction_alpha = 0.10
            if np.linalg.norm(self.auto_nav_elastic_bend_direction) < 0.5:
                self.auto_nav_elastic_bend_direction = bearing_direction.copy()
            else:
                bend_direction = (
                    (1.0 - direction_alpha) * self.auto_nav_elastic_bend_direction
                    + direction_alpha * bearing_direction
                )
                self.auto_nav_elastic_bend_direction = bend_direction / max(
                    float(np.linalg.norm(bend_direction)), 1e-9
                )
        else:
            pid_output[:] = 0.0
        pid_norm = float(np.linalg.norm(pid_output))
        if pid_norm > proximal_limit:
            pid_output *= proximal_limit / pid_norm
            self.auto_nav_pid_integral *= 0.94

        current_bend = (
            self.auto_nav_proximal_command
            + 0.65 * self.auto_nav_distal_command
        )
        current_bend_norm = float(np.linalg.norm(current_bend))
        opposite_direction = bool(
            pid_norm > 0.012
            and current_bend_norm > 0.015
            and float(np.dot(pid_output, current_bend))
            < -0.08 * pid_norm * current_bend_norm
        )
        if opposite_direction:
            self.auto_nav_steering_reversal_frames += 1
        else:
            self.auto_nav_steering_reversal_frames = max(
                0, self.auto_nav_steering_reversal_frames - 1
            )
        # A momentary opposite request is common when advancing through a
        # cluster of nearby points. Release only after a persistent reversal,
        # and never in the middle of a remembered physical bend.
        if (
            self.auto_nav_steering_reversal_frames >= 6
            and effective_curve_severity < 0.16
        ):
            self.auto_nav_elastic_release_active = True
            self.auto_nav_steering_reversal_frames = 0

        distal_required_magnitude = 0.0
        if self.auto_nav_elastic_release_active:
            # Never actively pull back. Set the requested cable force to zero
            # and wait for the two elastic sections to recenter naturally.
            self.auto_nav_pid_integral[:] = 0.0
            self.auto_nav_pid_output_filtered *= 0.72
            proximal_target = np.zeros(2, dtype=float)
            distal_target = np.zeros(2, dtype=float)
            distal_assist = 0.0
            elastic_release = True
            if current_bend_norm < 0.008:
                self.auto_nav_elastic_release_active = False
                self.auto_nav_elastic_settle_frames = 8
                self.auto_nav_pid_derivative[:] = 0.0
                self.auto_nav_pid_output_filtered[:] = 0.0
        elif self.auto_nav_elastic_settle_frames > 0:
            # Cable force is already zero, but the physical sections still
            # need a few frames to finish their passive elastic return.
            self.auto_nav_elastic_settle_frames -= 1
            self.auto_nav_pid_integral[:] = 0.0
            self.auto_nav_pid_derivative[:] = 0.0
            self.auto_nav_pid_output_filtered[:] = 0.0
            proximal_target = np.zeros(2, dtype=float)
            distal_target = np.zeros(2, dtype=float)
            distal_assist = 0.0
            elastic_release = True
        elif bearing_angle_deg <= steering_deadband_deg:
            self.auto_nav_pid_integral *= 0.80
            self.auto_nav_pid_output_filtered *= 0.72
            if effective_curve_severity >= 0.12 and intermediate_waypoint:
                # Alignment inside a bend does not mean the cables should be
                # released. Maintain the quasi-static curvature required to
                # counteract elastic recentering until the path actually exits
                # the bend.
                hold_direction = self.auto_nav_elastic_bend_direction.copy()
                if np.linalg.norm(hold_direction) < 0.5:
                    current_direction = (
                        self.auto_nav_proximal_command
                        + 0.65 * self.auto_nav_distal_command
                    )
                    hold_direction = current_direction / max(
                        float(np.linalg.norm(current_direction)), 1e-9
                    )
                if np.linalg.norm(hold_direction) < 0.5:
                    curvature_local = camera_rotation.T @ (outgoing - incoming)
                    hold_direction = curvature_local[:2] / max(
                        float(np.linalg.norm(curvature_local[:2])), 1e-9
                    )
                proximal_target = hold_direction * float(np.clip(
                    0.035 + 0.105 * effective_curve_severity,
                    0.0,
                    proximal_limit,
                ))
                distal_assist = float(np.clip(
                    (effective_curve_severity - 0.18) / 0.55, 0.0, 0.75
                ))
                distal_required_magnitude = float(np.clip(
                    distal_assist * (0.06 + 0.20 * effective_curve_severity),
                    0.0,
                    0.28,
                ))
                distal_target = hold_direction * distal_required_magnitude
            else:
                proximal_target = np.zeros(2, dtype=float)
                distal_target = np.zeros(2, dtype=float)
                distal_assist = 0.0
            elastic_release = False
        else:
            elastic_release = False
            output_alpha = dt / ((0.20 if straight_pid_mode else 0.14) + dt)
            self.auto_nav_pid_output_filtered += output_alpha * (
                pid_output - self.auto_nav_pid_output_filtered
            )
            # Filtering must not preserve a stale pull after the target moves
            # across the camera centre.
            if bearing_angle > 1e-9:
                self.auto_nav_pid_output_filtered = bearing_direction * max(
                    float(np.dot(self.auto_nav_pid_output_filtered, bearing_direction)),
                    0.0,
                )
            # Quasi-static inverse bending model. This predicts the absolute
            # cable command required for the measured bearing instead of
            # accumulating a fresh correction on every frame.
            predicted_proximal_magnitude = float(np.clip(
                0.00155 * bearing_angle_deg + 0.030 * effective_curve_severity,
                0.0,
                proximal_limit,
            ))
            inverse_proximal_target = bearing_direction * predicted_proximal_magnitude
            proximal_target = (
                0.78 * inverse_proximal_target
                + 0.22 * self.auto_nav_pid_output_filtered
            )
            # Distal bending is computed independently from the limited
            # proximal PID output. It smoothly supplies the curvature that the
            # proximal section cannot produce alone in a sharp bend.
            requested_distal_assist = float(np.clip(max(
                (effective_curve_severity - 0.16) / 0.46,
                (bearing_angle_deg - 42.0) / 45.0,
            ), 0.0, 0.85))
            distal_alpha = dt / (0.55 + dt)
            self.auto_nav_filtered_distal_assist += distal_alpha * (
                requested_distal_assist - self.auto_nav_filtered_distal_assist
            )
            distal_assist = self.auto_nav_filtered_distal_assist
            distal_required_magnitude = float(np.clip(
                distal_assist * (
                    0.08
                    + 0.22 * effective_curve_severity
                    + 0.0015 * min(bearing_angle_deg, 90.0)
                ),
                0.0,
                0.36,
            ))
            distal_direction = (
                bearing_direction
                if bearing_angle > 1e-9 else requested_direction
            )
            distal_target = distal_direction * distal_required_magnitude

        # Coordinate both continuum sections from one task-space bending
        # demand. The proximal section establishes the route curvature; the
        # distal section supplies the remaining tip-bearing correction and
        # becomes more active close to a waypoint or in a tight branch.
        control_direction = self.auto_nav_elastic_bend_direction.copy()
        if np.linalg.norm(control_direction) < 0.5:
            control_direction = requested_direction.copy()
        if np.linalg.norm(control_direction) < 0.5:
            control_direction = np.zeros(2, dtype=float)
        else:
            control_direction /= max(float(np.linalg.norm(control_direction)), 1e-9)

        target_bearing_angle = float(math.atan2(
            lateral_error, max(abs(forward_error), 0.5)
        ))
        total_bend_demand = float(np.clip(
            0.115 * effective_curve_severity
            + 0.105 * np.clip(target_bearing_angle / math.radians(55.0), 0.0, 1.0)
            + 0.105 * np.clip(bearing_angle_deg / 75.0, 0.0, 1.0)
            + 0.055 * np.clip(path_lateral_error / max(tolerance * 2.0, 1e-9), 0.0, 1.0),
            0.0,
            0.42,
        ))
        # Online response adaptation. If a sustained bend does not reduce the
        # bearing error, increase authority slowly. If bearing is falling fast
        # or nearing alignment, reduce authority before the elastic sections
        # overshoot. Adaptation is deliberately slow and bounded.
        if not gentle_path_mode and not reversing_to_recover:
            if self.auto_nav_bearing_improvement < -0.12 and bearing_angle_deg > 10.0:
                self.auto_nav_bend_gain_scale += 0.004
            elif self.auto_nav_bearing_improvement > 0.18 or bearing_angle_deg < 6.0:
                self.auto_nav_bend_gain_scale -= 0.006
            else:
                self.auto_nav_bend_gain_scale += 0.002 * (
                    1.0 - self.auto_nav_bend_gain_scale
                )
            self.auto_nav_bend_gain_scale = float(np.clip(
                self.auto_nav_bend_gain_scale, 0.58, 1.28
            ))
            total_bend_demand *= self.auto_nav_bend_gain_scale
        elif gentle_path_mode:
            self.auto_nav_bend_gain_scale += 0.04 * (
                1.0 - self.auto_nav_bend_gain_scale
            )
        if straight_neutral_mode:
            total_bend_demand = 0.0
        elif straight_correction_mode:
            # A sustained straight-line error receives one small proximal-only
            # correction. The elastic robot naturally recentres when this
            # command is removed; distal bending is unnecessary here.
            total_bend_demand = float(np.clip(
                0.012
                + 0.020 * np.clip(
                    path_lateral_error / max(tolerance * 1.5, 1e-9), 0.0, 1.0
                )
                + 0.012 * np.clip(bearing_angle_deg / 18.0, 0.0, 1.0),
                0.0,
                0.044,
            ))
        waypoint_nearness = float(np.clip(
            1.0 - distance_error / max(guidance_lookahead, threshold * 3.0),
            0.0,
            1.0,
        ))
        distal_share = float(np.clip(
            0.18
            + 0.36 * effective_curve_severity
            + 0.24 * waypoint_nearness
            + 0.28 * np.clip(bearing_angle_deg / 70.0, 0.0, 1.0),
            0.15,
            0.76,
        ))
        if gentle_path_mode:
            distal_share = 0.0
        proximal_share = 1.0 - 0.58 * distal_share
        target_lateral_direction = (
            np.asarray(error_local[:2], dtype=float) / max(lateral_error, 1e-9)
            if lateral_error > 1e-6 else control_direction.copy()
        )
        distal_precision_weight = float(np.clip(
            0.24 + 0.52 * waypoint_nearness, 0.24, 0.72
        ))
        if float(np.dot(target_lateral_direction, control_direction)) < 0.0:
            # Avoid opposing the proximal path-shaping command. A reversal is
            # handled through passive release, not simultaneous cable pulls.
            distal_precision_weight = min(distal_precision_weight, 0.32)
        distal_control_direction = (
            (1.0 - distal_precision_weight) * control_direction
            + distal_precision_weight * target_lateral_direction
        )
        distal_control_direction /= max(
            float(np.linalg.norm(distal_control_direction)), 1e-9
        )
        coordinated_proximal = control_direction * min(
            proximal_limit,
            total_bend_demand * proximal_share,
        )
        coordinated_distal = distal_control_direction * min(
            0.36,
            total_bend_demand * distal_share,
        )
        # The coordinated allocation is dominant. PID and curvature-holding
        # targets are retained only as small residual corrections, preventing
        # both sections from independently applying the full requested bend.
        raw_proximal_target = proximal_target.copy()
        raw_distal_target = distal_target.copy()
        proximal_target = (
            0.78 * coordinated_proximal + 0.22 * raw_proximal_target
        )
        distal_target = (
            0.82 * coordinated_distal + 0.18 * raw_distal_target
        )
        proximal_target_norm = float(np.linalg.norm(proximal_target))
        if proximal_target_norm > proximal_limit:
            proximal_target *= proximal_limit / proximal_target_norm
        distal_coordination_limit = min(
            0.36,
            total_bend_demand * min(distal_share + 0.10, 0.78) + 0.015,
        )
        distal_target_norm = float(np.linalg.norm(distal_target))
        if distal_target_norm > distal_coordination_limit:
            distal_target *= distal_coordination_limit / distal_target_norm
        if straight_neutral_mode:
            proximal_target[:] = 0.0
            distal_target[:] = 0.0
            self.auto_nav_pid_integral[:] = 0.0
            self.auto_nav_pid_output_filtered *= 0.70
        elif straight_correction_mode:
            proximal_target = coordinated_proximal.copy()
            distal_target[:] = 0.0
        distal_required_magnitude = float(np.linalg.norm(distal_target))

        # Continuously evolve the action reference. Intermediate waypoint
        # changes never clear or restart the controller; the reference simply
        # flows from the current cable state toward the next required state.
        action_recalculated = True
        action_held = False
        predicted_bearing_angle_deg = max(
            0.0,
            bearing_angle_deg - max(self.auto_nav_bearing_improvement, 0.0) * 8.0,
        )
        near_alignment = bool(
            bearing_angle_deg <= max(steering_deadband_deg + 1.5, 4.5)
            and path_lateral_error <= tolerance * 0.65
        )
        predicted_alignment = bool(
            predicted_bearing_angle_deg <= max(steering_deadband_deg + 0.8, 3.8)
            and self.auto_nav_bearing_improvement > 0.10
        )
        final_alignment = bool(
            not intermediate_waypoint and (near_alignment or predicted_alignment)
        )
        if reversing_to_recover:
            # During a short recovery retraction, preserve the established
            # shape instead of issuing a new aggressive bend or snapping
            # straight. Recompute forward steering after the target returns.
            desired_proximal_reference = self.auto_nav_proximal_command.copy()
            desired_distal_reference = self.auto_nav_distal_command.copy()
        elif elastic_release or final_alignment:
            desired_proximal_reference = np.zeros(2, dtype=float)
            desired_distal_reference = np.zeros(2, dtype=float)
        else:
            desired_proximal_reference = proximal_target.copy()
            desired_distal_reference = distal_target.copy()

        # Slow the reference evolution while the error is already improving;
        # speed it up only when the measured bearing is clearly getting worse.
        reference_tau = 0.52 if straight_pid_mode else (
            0.28 if effective_curve_severity > 0.55 else 0.38
        )
        if self.auto_nav_bearing_improvement > 0.08:
            reference_tau *= 1.45
        elif self.auto_nav_bearing_improvement < -0.18:
            reference_tau *= 0.72
        reference_alpha = dt / (reference_tau + dt)

        def update_continuous_reference(current, desired, max_delta):
            delta = reference_alpha * (desired - current)
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_delta:
                delta *= max_delta / delta_norm
            return current + delta

        self.auto_nav_held_proximal_target = update_continuous_reference(
            self.auto_nav_held_proximal_target,
            desired_proximal_reference,
            0.0022 if straight_pid_mode else 0.0030,
        )
        self.auto_nav_held_distal_target = update_continuous_reference(
            self.auto_nav_held_distal_target,
            desired_distal_reference,
            0.0025,
        )
        proximal_target = self.auto_nav_held_proximal_target.copy()
        distal_target = self.auto_nav_held_distal_target.copy()
        self.auto_nav_action_hold_countdown = 0

        def smooth_command(current, desired, alpha, max_delta):
            filtered = current + alpha * (desired - current)
            delta = filtered - current
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_delta:
                delta *= max_delta / delta_norm
            return current + delta

        # During release the command only decays toward zero. This is not an
        # active callback; the physical elastic model supplies the return.
        proximal_delta = 0.004 if elastic_release else (0.0025 if straight_pid_mode else 0.0035)
        distal_delta = 0.0030 if elastic_release else 0.0030
        self.auto_nav_proximal_command = smooth_command(
            self.auto_nav_proximal_command, proximal_target,
            0.20 if elastic_release else 0.14, proximal_delta
        )
        self.auto_nav_distal_command = smooth_command(
            self.auto_nav_distal_command, distal_target,
            0.18 if elastic_release else 0.12, distal_delta
        )
        proximal = self.auto_nav_proximal_command.copy()
        distal = self.auto_nav_distal_command.copy()
        combined_bend_load = float(
            np.linalg.norm(proximal) + 0.65 * np.linalg.norm(distal)
        )
        desired_combined_bend = (
            self.auto_nav_held_proximal_target
            + 0.65 * self.auto_nav_held_distal_target
        )
        current_combined_bend = proximal + 0.65 * distal
        bend_tracking_error = float(np.linalg.norm(
            desired_combined_bend - current_combined_bend
        ))
        desired_combined_load = float(np.linalg.norm(desired_combined_bend))
        bend_magnitude_readiness = float(np.clip(
            1.0 - bend_tracking_error / max(desired_combined_load + 0.025, 1e-9),
            0.16,
            1.0,
        ))
        if desired_combined_load > 0.012 and np.linalg.norm(current_combined_bend) > 0.008:
            bend_direction_alignment = float(np.clip(
                np.dot(current_combined_bend, desired_combined_bend)
                / max(
                    float(np.linalg.norm(current_combined_bend)) * desired_combined_load,
                    1e-9,
                ),
                -1.0,
                1.0,
            ))
            bend_direction_readiness = float(np.clip(
                (bend_direction_alignment + 0.20) / 1.20, 0.12, 1.0
            ))
        elif desired_combined_load <= 0.012:
            bend_direction_alignment = 1.0
            bend_direction_readiness = 1.0
        else:
            bend_direction_alignment = 0.0
            bend_direction_readiness = 0.20
        bend_readiness = bend_magnitude_readiness * bend_direction_readiness

        # Bending and insertion share one predicted next-step calculation.
        # Feed is reduced until the requested curvature has actually formed,
        # and is capped near the active waypoint plane to avoid overshooting.
        max_step = self._auto_param("max_step") * self._auto_param("speed_scale")
        alignment_scale = float(np.clip(1.0 - heading_error / math.radians(105.0), 0.22, 1.0))
        lateral_scale = float(np.clip(
            tolerance / max(path_lateral_error, tolerance), 0.30, 1.0
        ))
        curve_scale = float(np.clip(1.0 - 0.76 * effective_curve_severity, 0.22, 1.0))
        release_scale = 0.24 if elastic_release else 1.0
        bend_load_scale = float(np.clip(1.0 - 1.8 * combined_bend_load, 0.20, 1.0))
        minimum_feed_ratio = 0.10 if (
            effective_curve_severity > 0.55 or combined_bend_load > 0.25
        ) else 0.14
        desired_insertion = max_step * max(
            minimum_feed_ratio,
            alignment_scale
            * lateral_scale
            * curve_scale
            * release_scale
            * bend_load_scale
            * bend_readiness,
        )
        if reversing_to_recover:
            reverse_ratio = float(np.clip(
                0.12
                + 0.15 * abs(forward_error) / max(guidance_lookahead, 1.0)
                + 0.08 * path_lateral_error / max(tolerance * 2.0, 1e-9),
                0.12,
                0.34,
            ))
            desired_insertion = -max_step * reverse_ratio
        distance_to_target_plane = max(-target_plane_progress, 0.0)
        near_waypoint_plane = bool(
            not reversing_to_recover
            and
            target_plane_progress < 0.0
            and distance_to_target_plane < max(max_step * 3.0, threshold * 2.0)
        )
        waypoint_step_cap = max_step
        if near_waypoint_plane:
            waypoint_step_cap = max(
                max_step * 0.08,
                min(max_step, distance_to_target_plane * 0.58 + threshold * 0.12),
            )
            desired_insertion = min(desired_insertion, waypoint_step_cap)
        predicted_forward_progress = desired_insertion * max(
            float(np.dot(forward, incoming)), 0.12
        )
        predicted_plane_remaining = max(
            distance_to_target_plane - predicted_forward_progress, 0.0
        )
        previous_insertion = float(getattr(self, "auto_nav_last_insertion_step", 0.0))
        insertion_step = previous_insertion + float(np.clip(
            desired_insertion - previous_insertion,
            -max_step * (0.12 if reversing_to_recover else 0.06),
            max_step * 0.06,
        ))
        if reversing_to_recover:
            insertion_step = min(insertion_step, -max_step * 0.06)
        elif not forward_reengage_active:
            insertion_step = max(insertion_step, max_step * 0.08)
        if near_waypoint_plane and not reversing_to_recover:
            insertion_step = min(insertion_step, waypoint_step_cap)
        predicted_tip_position = position + forward * insertion_step
        predicted_waypoint_distance = float(np.linalg.norm(
            target - predicted_tip_position
        ))
        predicted_distance_improvement = distance_error - predicted_waypoint_distance
        if (
            predicted_distance_improvement <= 0.0
            and not crossed_target
            and not reversing_to_recover
            and not forward_reengage_active
        ):
            insertion_step = max_step * 0.08
            predicted_tip_position = position + forward * insertion_step
            predicted_waypoint_distance = float(np.linalg.norm(
                target - predicted_tip_position
            ))
            predicted_distance_improvement = distance_error - predicted_waypoint_distance
        predicted_forward_progress = insertion_step * max(
            float(np.dot(forward, incoming)), 0.12
        )
        predicted_plane_remaining = max(
            distance_to_target_plane - predicted_forward_progress, 0.0
        )
        self.auto_nav_last_insertion_step = insertion_step

        previous_position = self.auto_nav_last_position
        frame_motion = 0.0 if previous_position is None else float(
            np.linalg.norm(raw_position - np.asarray(previous_position, dtype=float))
        )
        self.auto_nav_last_position = raw_position.copy()
        self.auto_nav_filtered_speed += 0.18 * (
            frame_motion - self.auto_nav_filtered_speed
        )
        self.auto_nav_nearest_index = target_index
        self.auto_nav_progress = 100.0 * self.auto_nav_passed_count / max(1, len(path))
        all_path_local = (camera_rotation.T @ (path - position).T).T
        visual_error = np.asarray(error_local[:2], dtype=float) / max(abs(forward_error), 1.0)
        servo_phase = (
            "目标持续位于后方，保持曲率并低速后退调整"
            if reversing_to_recover else (
                "弹性自然回中，保持低速正向进给"
                if elastic_release else (
                "直线走廊自由回中"
                if straight_neutral_mode else (
                    "直线持续偏差小幅近端修正"
                    if straight_correction_mode else (
                        "大弯道近端主导、远端协同"
                        if effective_curve_severity > 0.45 else "逐路径点稳定跟踪"
                    )
                )
            )
            )
        )
        return {
            "steering": distal.tolist(),
            "steering_proximal": proximal.tolist(),
            "steering_distal": distal.tolist(),
            "insertion_delta": float(insertion_step / 577.0),
            "insertion_step_mm": float(insertion_step),
            "target_index": target_index,
            "nearest_index": target_index,
            "target_model_mm": target.tolist(),
            "all_path_local_mm": all_path_local.tolist(),
            "distance_to_target_mm": distance_error,
            "waypoint_reached": waypoint_reached,
            "waypoint_hold_frames": self.auto_nav_waypoint_reached_frames,
            "waypoint_local_mm": error_local.tolist(),
            "target_local_mm": error_local.tolist(),
            "guidance_target_model_mm": target.tolist(),
            "guidance_target_distance_mm": distance_error,
            "relative_position_error_mm": error_local.tolist(),
            "relative_bearing_error_deg": math.degrees(heading_error),
            "pid_dt_s": dt,
            "pid_p": (kp * bearing).tolist(),
            "pid_i": (ki * self.auto_nav_pid_integral).tolist(),
            "pid_d": (kd * self.auto_nav_pid_derivative).tolist(),
            "pid_output": proximal_target.tolist(),
            "pid_straight_mode": straight_pid_mode,
            "gentle_path_mode": gentle_path_mode,
            "straight_neutral_mode": straight_neutral_mode,
            "straight_correction_mode": straight_correction_mode,
            "straight_error_frames": self.auto_nav_straight_error_frames,
            "straight_clear_frames": self.auto_nav_straight_clear_frames,
            "proximal_limit": proximal_limit,
            "distal_required_magnitude": distal_required_magnitude,
            "total_bend_demand": total_bend_demand,
            "bend_gain_scale": self.auto_nav_bend_gain_scale,
            "proximal_share": proximal_share,
            "distal_share": distal_share,
            "distal_precision_weight": distal_precision_weight,
            "distal_control_direction": distal_control_direction.tolist(),
            "distal_coordination_limit": distal_coordination_limit,
            "bend_tracking_error": bend_tracking_error,
            "bend_magnitude_readiness": bend_magnitude_readiness,
            "bend_direction_alignment": bend_direction_alignment,
            "bend_direction_readiness": bend_direction_readiness,
            "bend_readiness": bend_readiness,
            "bearing_improvement_deg_per_frame": self.auto_nav_bearing_improvement,
            "predicted_bearing_angle_deg": predicted_bearing_angle_deg,
            "action_hold_countdown": self.auto_nav_action_hold_countdown,
            "action_held": action_held,
            "action_recalculated": action_recalculated,
            "continuous_action": True,
            "target_transition": target_changed,
            "intermediate_waypoint": intermediate_waypoint,
            "combined_bend_load": combined_bend_load,
            "bend_load_scale": bend_load_scale,
            "visual_center_error": visual_error.tolist(),
            "visual_error_norm": float(np.linalg.norm(visual_error)),
            "visual_center_speed": alignment_scale,
            "lateral_distance_mm": lateral_error,
            "forward_error_mm": forward_error,
            "azimuth_deg": math.degrees(float(math.atan2(error_local[0], max(forward_error, 1e-6)))),
            "elevation_deg": math.degrees(float(math.atan2(error_local[1], max(forward_error, 1e-6)))),
            "heading_error_deg": math.degrees(heading_error),
            "target_heading_error_deg": math.degrees(target_heading_error),
            "servo_phase": servo_phase,
            "estimated_speed_mm_per_frame": self.auto_nav_filtered_speed,
            "straight_segment": effective_curve_severity < 0.12,
            "straight_path_mode": effective_curve_severity < 0.12,
            "turn_active": effective_curve_severity >= 0.12,
            "turn_severity": effective_curve_severity,
            "curve_severity": effective_curve_severity,
            "raw_curve_severity": curve_severity,
            "curve_memory": self.auto_nav_curve_memory,
            "curvature_throttle": curve_scale,
            "distal_assist": distal_assist,
            "proximal_utilization": float(np.linalg.norm(proximal) / 0.30),
            "cruise_boost": 1.0,
            "recovery_mode": elastic_release or reversing_to_recover,
            "retracting": reversing_to_recover,
            "motion_mode": self.auto_nav_motion_mode,
            "target_confirmed_behind": target_confirmed_behind,
            "behind_confirmation_frames": self.auto_nav_behind_frames,
            "front_confirmation_frames": self.auto_nav_front_frames,
            "forward_reengage_frames": self.auto_nav_forward_reengage_frames,
            "special_case": elastic_release or reversing_to_recover,
            "anti_curl_release": elastic_release,
            "cross_track_error_mm": target_plane_cross,
            "within_centerline_tolerance": target_plane_cross <= tolerance,
            "global_curve_angle_deg": math.degrees(curve_angle),
            "global_preview_angle_deg": math.degrees(preview_angle),
            "preview_arc_length_mm": preview_arc_length,
            "tight_curve_severity": tight_curve_severity,
            "guidance_point_model_mm": np.asarray(guidance_point, dtype=float).tolist(),
            "guidance_lookahead_mm": guidance_lookahead,
            "distance_to_target_plane_mm": distance_to_target_plane,
            "raw_target_plane_progress_mm": raw_target_plane_progress,
            "waypoint_step_cap_mm": waypoint_step_cap,
            "predicted_forward_progress_mm": predicted_forward_progress,
            "predicted_plane_remaining_mm": predicted_plane_remaining,
            "predicted_tip_position_mm": predicted_tip_position.tolist(),
            "predicted_waypoint_distance_mm": predicted_waypoint_distance,
            "predicted_distance_improvement_mm": predicted_distance_improvement,
            "global_speed_scale": curve_scale,
            "tangent_error_deg": math.degrees(heading_error),
            "forward_alignment": float(np.dot(forward, target_direction)),
            "overshot_waypoint": crossed_target,
            "stopping_distance_mm": 0.0,
            "elastic_release_active": elastic_release,
            "elastic_settle_frames": self.auto_nav_elastic_settle_frames,
        }

    def _compute_continuum_path_action(self):
        """Continuous arc-length controller tailored to an elastic continuum robot."""
        path = self._get_auto_navigation_path()
        position = np.asarray(getattr(self, "virtual_position", []), dtype=float)
        camera_rotation = np.asarray(getattr(self, "mapped_tip_camera_dir", []), dtype=float)
        if path.ndim != 2 or len(path) < 2 or position.shape != (3,) or camera_rotation.shape != (3, 3):
            return None
        control_basis = getattr(self, "auto_nav_control_basis", None)
        if control_basis is None or np.asarray(control_basis).shape != (3, 3):
            self.auto_nav_control_basis = camera_rotation.copy()
        control_basis = np.asarray(self.auto_nav_control_basis, dtype=float)

        segments = np.diff(path, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        valid_lengths = np.maximum(lengths, 1e-9)
        tangents = segments / valid_lengths[:, None]
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        previous_position = (
            None if self.auto_nav_last_position is None
            else np.asarray(self.auto_nav_last_position, dtype=float).copy()
        )
        frame_motion = 0.0 if previous_position is None else float(np.linalg.norm(position - previous_position))
        self.auto_nav_last_position = position.copy()
        self.auto_nav_filtered_speed += 0.16 * (frame_motion - self.auto_nav_filtered_speed)

        # Project only onto a forward neighbourhood. Arc-length progress never
        # moves backward, so a curved branch cannot make the target jump behind.
        old_s = float(np.clip(getattr(self, "auto_nav_path_distance", 0.0), 0.0, cumulative[-1]))
        old_segment = int(np.clip(np.searchsorted(cumulative, old_s, side="right") - 1, 0, len(segments) - 1))
        search_start = max(0, old_segment - 1)
        search_end = min(len(segments), old_segment + 7)
        best_distance = float("inf")
        projected_s = old_s
        projected_point = path[old_segment].copy()
        projected_tangent = tangents[old_segment].copy()
        for index in range(search_start, search_end):
            alpha = float(np.clip(
                np.dot(position - path[index], segments[index]) / (valid_lengths[index] ** 2),
                0.0,
                1.0,
            ))
            point = path[index] + alpha * segments[index]
            distance = float(np.linalg.norm(position - point))
            candidate_s = float(cumulative[index] + alpha * lengths[index])
            if distance < best_distance and candidate_s >= old_s - 1.0:
                best_distance = distance
                projected_s = candidate_s
                projected_point = point
                projected_tangent = tangents[index]
        tolerance = self._auto_param("centerline_tolerance")
        progress_gate = max(tolerance * 2.2, 8.0)
        max_progress_step = max(1.2, frame_motion * 2.2, self._auto_param("max_step") * 1.8)
        if best_distance <= progress_gate:
            projected_s = min(projected_s, old_s + max_progress_step)
            self.auto_nav_path_distance = max(old_s, projected_s)
        else:
            projected_s = old_s
            self.auto_nav_path_distance = old_s
            projected_point, projected_tangent, _ = sample_at_s(old_s) if "sample_at_s" in locals() else (
                path[old_segment], tangents[old_segment], old_segment
            )

        def sample_at_s(distance):
            distance = float(np.clip(distance, 0.0, cumulative[-1]))
            index = int(np.clip(np.searchsorted(cumulative, distance, side="right") - 1, 0, len(segments) - 1))
            alpha = float(np.clip(
                (distance - cumulative[index]) / valid_lengths[index], 0.0, 1.0
            ))
            return path[index] + alpha * segments[index], tangents[index], index

        projected_point, projected_tangent, _ = sample_at_s(self.auto_nav_path_distance)
        lookahead = max(self._auto_param("lookahead"), 6.0)
        near_point, near_tangent, near_segment = sample_at_s(self.auto_nav_path_distance + lookahead * 0.45)
        far_point, far_tangent, far_segment = sample_at_s(self.auto_nav_path_distance + lookahead * 1.35)
        curve_angle = math.acos(float(np.clip(np.dot(near_tangent, far_tangent), -1.0, 1.0)))
        curve_severity = float(np.clip(curve_angle / math.radians(65.0), 0.0, 1.0))

        cross_track_world = projected_point - position
        cross_track = float(np.linalg.norm(cross_track_world))
        # Feedforward tangent dominates. Cross-track feedback is deliberately
        # low-frequency and bounded so it cannot cause left-right point chasing.
        correction_weight = float(np.clip(cross_track / max(tolerance * 2.0, 1e-9), 0.0, 0.55))
        desired_world = (
            (1.0 - 0.42 * curve_severity) * near_tangent
            + 0.42 * curve_severity * far_tangent
            + cross_track_world * (correction_weight / max(lookahead, 1e-9))
        )
        desired_world /= max(float(np.linalg.norm(desired_world)), 1e-9)
        desired_local = camera_rotation.T @ desired_world
        camera_forward = np.asarray(camera_rotation[:, 2], dtype=float)
        heading_error = math.acos(float(np.clip(np.dot(camera_forward, desired_world), -1.0, 1.0)))
        # MuJoCo cable torques use a fixed world-referenced control plane.
        # Project onto the navigation-start basis so tip roll cannot rotate the
        # bend command and create a spiral oscillation.
        desired_bend_lateral = np.asarray([
            np.dot(desired_world, control_basis[:, 0]),
            np.dot(desired_world, control_basis[:, 1]),
        ], dtype=float)
        curve_world = far_tangent - near_tangent
        curve_bend_lateral = np.asarray([
            np.dot(curve_world, control_basis[:, 0]),
            np.dot(curve_world, control_basis[:, 1]),
        ], dtype=float)
        if curve_severity >= 0.12 and np.linalg.norm(curve_bend_lateral) > 1e-6:
            # In a bend, the planned route curvature owns the bending plane.
            # Cross-track error changes force magnitude only; it must not spin
            # the cable direction around while the robot is already bending.
            bend_lateral = curve_bend_lateral
        else:
            bend_lateral = desired_bend_lateral
        bend_lateral_norm = float(np.linalg.norm(bend_lateral))
        bend_direction = (
            bend_lateral / bend_lateral_norm
            if bend_lateral_norm > 1e-6 else np.zeros(2, dtype=float)
        )

        # Curvature feedforward supplies most of the bend in large turns.
        # Heading/cross-track feedback only trims its magnitude.
        bend_magnitude = float(np.clip(
            0.14 * curve_severity
            + 0.08 * np.clip(heading_error / math.radians(45.0), 0.0, 1.0)
            + 0.03 * np.clip(cross_track / max(tolerance, 1e-9), 0.0, 1.0),
            0.0,
            0.22,
        ))
        if heading_error < math.radians(12.0) and cross_track < tolerance * 2.0:
            bend_magnitude *= 0.42
        if curve_severity < 0.08 and heading_error < math.radians(5.0) and cross_track < tolerance * 0.65:
            bend_magnitude = 0.0

        previous_curve_heading = getattr(self, "auto_nav_last_curve_heading_error", None)
        if (
            previous_curve_heading is not None
            and heading_error > previous_curve_heading + math.radians(0.8)
            and np.linalg.norm(self.auto_nav_curve_command) > 0.045
            and self.auto_nav_curve_release_cooldown <= 0
        ):
            self.auto_nav_curve_error_rise_frames += 1
        else:
            self.auto_nav_curve_error_rise_frames = max(0, self.auto_nav_curve_error_rise_frames - 1)
        self.auto_nav_last_curve_heading_error = heading_error

        current_direction_norm = float(np.linalg.norm(self.auto_nav_curve_direction))
        self.auto_nav_curve_release_cooldown = max(
            0, int(getattr(self, "auto_nav_curve_release_cooldown", 0)) - 1
        )
        direction_flip = bool(
            bend_magnitude > 0.04
            and curve_severity > 0.24
            and current_direction_norm > 0.5
            and self.auto_nav_curve_release_frames <= 0
            and self.auto_nav_curve_release_cooldown <= 0
            and float(np.dot(bend_direction, self.auto_nav_curve_direction)) < -0.75
        )
        if direction_flip:
            self.auto_nav_curve_release_frames = max(self.auto_nav_curve_release_frames, 8)
        if self.auto_nav_curve_error_rise_frames >= 3:
            self.auto_nav_curve_release_frames = max(self.auto_nav_curve_release_frames, 7)
            self.auto_nav_curve_error_rise_frames = 0

        if self.auto_nav_curve_release_frames > 0:
            # Stop cable force and use the robot's elasticity to return toward
            # neutral before building curvature in a different direction.
            self.auto_nav_curve_release_frames -= 1
            desired_curve_command = np.zeros(2, dtype=float)
            self.auto_nav_curve_command[:] = 0.0
            if self.auto_nav_curve_release_frames == 0:
                self.auto_nav_curve_direction[:] = 0.0
                self.auto_nav_curve_release_cooldown = 14
        else:
            if bend_magnitude > 1e-6:
                direction_alpha = 0.10 if curve_severity < 0.35 else 0.16
                if current_direction_norm < 0.5:
                    self.auto_nav_curve_direction = bend_direction.copy()
                else:
                    blended = (1.0 - direction_alpha) * self.auto_nav_curve_direction + direction_alpha * bend_direction
                    self.auto_nav_curve_direction = blended / max(float(np.linalg.norm(blended)), 1e-9)
            desired_curve_command = self.auto_nav_curve_direction * bend_magnitude
            max_command_delta = 0.012 if curve_severity < 0.35 else 0.018
            delta = desired_curve_command - self.auto_nav_curve_command
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_command_delta:
                delta *= max_command_delta / delta_norm
            self.auto_nav_curve_command += delta

        proximal = self.auto_nav_curve_command.copy()
        distal_ratio = float(np.clip(max(
            (curve_severity - 0.35) / 0.45,
            (math.degrees(heading_error) - 38.0) / 75.0,
        ), 0.0, 0.62))
        distal = proximal * distal_ratio
        if self.auto_nav_curve_release_frames > 0:
            proximal[:] = 0.0
            distal[:] = 0.0

        max_step = self._auto_param("max_step") * self._auto_param("speed_scale")
        heading_scale = float(np.clip(1.0 - heading_error / math.radians(95.0), 0.25, 1.0))
        cross_scale = float(np.clip(tolerance / max(cross_track, tolerance), 0.32, 1.0))
        curve_scale = float(np.clip(1.0 - 0.78 * curve_severity, 0.18, 1.0))
        release_scale = 0.20 if self.auto_nav_curve_release_frames > 0 else 1.0
        desired_insertion = max_step * max(0.14, heading_scale * cross_scale * curve_scale * release_scale)
        previous_insertion = float(getattr(self, "auto_nav_last_insertion_step", 0.0))
        insertion_step = previous_insertion + float(np.clip(
            desired_insertion - previous_insertion,
            -max_step * 0.20,
            max_step * 0.24,
        ))
        self.auto_nav_last_insertion_step = insertion_step

        target_index = int(np.clip(self.auto_nav_target_index, 1, len(path) - 1))
        target = np.asarray(path[target_index], dtype=float)
        target_distance = float(np.linalg.norm(target - position))
        threshold = self._auto_param("point_threshold")
        pass_tolerance = max(threshold * 1.5, tolerance)
        waypoint_reached = bool(
            target_distance <= threshold
            or (
                self.auto_nav_path_distance >= cumulative[target_index] - 0.3
                and cross_track <= pass_tolerance
            )
        )
        target_local = camera_rotation.T @ (target - position)
        all_path_local = (camera_rotation.T @ (path - position).T).T
        self.auto_nav_nearest_index = int(np.clip(near_segment + 1, 0, len(path) - 1))
        self.auto_nav_progress = 100.0 * self.auto_nav_path_distance / max(cumulative[-1], 1e-9)
        servo_phase = (
            "弹性释放，准备改变弯曲方向"
            if self.auto_nav_curve_release_frames > 0
            else ("大曲率连续建弯并限速" if curve_severity > 0.45 else "连续轨迹稳定跟踪")
        )
        visual_error = np.asarray(target_local[:2], dtype=float) / max(abs(float(target_local[2])), 1.0)
        return {
            "steering": distal.tolist(),
            "steering_proximal": proximal.tolist(),
            "steering_distal": distal.tolist(),
            "insertion_delta": float(insertion_step / 577.0),
            "insertion_step_mm": float(insertion_step),
            "target_index": target_index,
            "nearest_index": self.auto_nav_nearest_index,
            "target_model_mm": target.tolist(),
            "all_path_local_mm": all_path_local.tolist(),
            "distance_to_target_mm": target_distance,
            "waypoint_reached": waypoint_reached,
            "waypoint_local_mm": target_local.tolist(),
            "target_local_mm": target_local.tolist(),
            "guidance_target_model_mm": far_point.tolist(),
            "guidance_target_distance_mm": float(np.linalg.norm(far_point - position)),
            "relative_position_error_mm": target_local.tolist(),
            "relative_bearing_error_deg": math.degrees(heading_error),
            "visual_center_error": visual_error.tolist(),
            "visual_error_norm": float(np.linalg.norm(visual_error)),
            "visual_center_speed": heading_scale,
            "lateral_distance_mm": cross_track,
            "azimuth_deg": math.degrees(float(math.atan2(desired_local[0], max(desired_local[2], 1e-6)))),
            "elevation_deg": math.degrees(float(math.atan2(desired_local[1], max(desired_local[2], 1e-6)))),
            "heading_error_deg": math.degrees(heading_error),
            "servo_phase": servo_phase,
            "estimated_speed_mm_per_frame": float(getattr(self, "auto_nav_filtered_speed", 0.0)),
            "straight_segment": curve_severity < 0.12,
            "straight_path_mode": curve_severity < 0.12,
            "turn_active": curve_severity >= 0.12,
            "turn_severity": curve_severity,
            "curve_severity": curve_severity,
            "curvature_throttle": curve_scale,
            "distal_assist": distal_ratio,
            "proximal_utilization": float(np.linalg.norm(proximal) / 0.62),
            "cruise_boost": 1.0,
            "recovery_mode": self.auto_nav_curve_release_frames > 0,
            "retracting": False,
            "special_case": self.auto_nav_curve_release_frames > 0,
            "anti_curl_release": self.auto_nav_curve_release_frames > 0,
            "cross_track_error_mm": cross_track,
            "within_centerline_tolerance": cross_track <= tolerance,
            "global_curve_angle_deg": math.degrees(curve_angle),
            "global_preview_angle_deg": math.degrees(curve_angle),
            "global_speed_scale": curve_scale,
            "tangent_error_deg": math.degrees(heading_error),
            "forward_alignment": float(desired_local[2]),
            "overshot_waypoint": False,
            "stopping_distance_mm": 0.0,
            "elastic_release_active": self.auto_nav_curve_release_frames > 0,
        }

    def _compute_physics_model_navigation_action(self):
        """Track mandatory waypoints using the continuum robot's torque physics.

        The two elastic sections are controlled as absolute quasi-static torque
        states. Path curvature supplies feed-forward bending, while measured
        tip bearing and cross-track error close the loop. Commands are never
        accumulated per waypoint; this avoids exciting the elastic model.
        """
        path = self._get_auto_navigation_path()
        raw_position = np.asarray(getattr(self, "virtual_position", []), dtype=float)
        camera_rotation = np.asarray(getattr(self, "mapped_tip_camera_dir", []), dtype=float)
        if (
            path.ndim != 2 or len(path) < 2 or raw_position.shape != (3,)
            or camera_rotation.shape != (3, 3)
            or not np.isfinite(raw_position).all()
            or not np.isfinite(camera_rotation).all()
        ):
            return None

        # Position and forward-axis filtering reject NDI/MuJoCo jitter without
        # delaying the target-plane crossing decision, which uses raw position.
        if self.auto_nav_filtered_position is None:
            self.auto_nav_filtered_position = raw_position.copy()
        self.auto_nav_filtered_position += 0.42 * (
            raw_position - self.auto_nav_filtered_position
        )
        position = self.auto_nav_filtered_position.copy()
        raw_forward = np.asarray(camera_rotation[:, 2], dtype=float)
        raw_forward /= max(float(np.linalg.norm(raw_forward)), 1e-9)
        if self.auto_nav_filtered_forward is None:
            self.auto_nav_filtered_forward = raw_forward.copy()
        self.auto_nav_filtered_forward += 0.28 * (
            raw_forward - self.auto_nav_filtered_forward
        )
        self.auto_nav_filtered_forward /= max(
            float(np.linalg.norm(self.auto_nav_filtered_forward)), 1e-9
        )
        forward = self.auto_nav_filtered_forward.copy()

        target_index = int(np.clip(self.auto_nav_target_index, 1, len(path) - 1))
        target_changed = target_index != self.auto_nav_waypoint_controller_index
        if target_changed:
            self.auto_nav_waypoint_controller_index = target_index
            self.auto_nav_waypoint_reached_frames = 0
        previous_point = np.asarray(path[target_index - 1], dtype=float)
        target = np.asarray(path[target_index], dtype=float)
        next_point = np.asarray(path[min(target_index + 1, len(path) - 1)], dtype=float)

        incoming = target - previous_point
        incoming_length = max(float(np.linalg.norm(incoming)), 1e-9)
        incoming /= incoming_length
        outgoing = next_point - target
        if np.linalg.norm(outgoing) <= 1e-9:
            outgoing = incoming.copy()
        outgoing /= max(float(np.linalg.norm(outgoing)), 1e-9)
        curve_angle = math.acos(float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0)))
        curve_severity = float(np.clip(curve_angle / math.radians(70.0), 0.0, 1.0))
        self.auto_nav_curve_memory += (
            0.26 if curve_severity > self.auto_nav_curve_memory else 0.055
        ) * (curve_severity - self.auto_nav_curve_memory)
        if curve_severity < 0.025 and self.auto_nav_curve_memory < 0.035:
            self.auto_nav_curve_memory = 0.0
        effective_curve = float(max(curve_severity, self.auto_nav_curve_memory))

        threshold = self._auto_param("point_threshold")
        tolerance = self._auto_param("centerline_tolerance")
        error_world = target - position
        distance_error = float(np.linalg.norm(error_world))
        target_direction = error_world / max(distance_error, 1e-9)
        error_local = camera_rotation.T @ error_world
        lateral_error = float(np.linalg.norm(error_local[:2]))
        forward_error = float(error_local[2])
        path_lateral_error = float(np.linalg.norm(
            error_world - float(np.dot(error_world, incoming)) * incoming
        ))
        target_plane_progress = float(np.dot(position - target, incoming))
        raw_target_plane_progress = float(np.dot(raw_position - target, incoming))
        crossed_in_corridor = bool(
            target_index < len(path) - 1
            and raw_target_plane_progress >= 0.0
            and path_lateral_error <= max(threshold * 1.5, tolerance)
        )
        inside_target = distance_error <= threshold
        if inside_target or crossed_in_corridor:
            self.auto_nav_waypoint_reached_frames += 1
        else:
            self.auto_nav_waypoint_reached_frames = 0
        waypoint_reached = self.auto_nav_waypoint_reached_frames >= 2

        # The active point is mandatory. Blend toward the outgoing tangent
        # only inside its capture sphere, so the controller anticipates a bend
        # without cutting across unvisited path points.
        outgoing_blend = float(np.clip(
            1.0 - distance_error / max(threshold * 4.0, 5.0), 0.0, 0.32
        ))
        desired_world = (1.0 - outgoing_blend) * target_direction + outgoing_blend * outgoing
        desired_world /= max(float(np.linalg.norm(desired_world)), 1e-9)
        desired_local = camera_rotation.T @ desired_world
        desired_lateral = np.asarray(desired_local[:2], dtype=float)
        desired_lateral_norm = float(np.linalg.norm(desired_lateral))
        desired_direction = (
            desired_lateral / desired_lateral_norm
            if desired_lateral_norm > 1e-7 else np.zeros(2, dtype=float)
        )
        curvature_local = camera_rotation.T @ (outgoing - incoming)
        curvature_lateral = np.asarray(curvature_local[:2], dtype=float)
        curvature_lateral_norm = float(np.linalg.norm(curvature_lateral))
        curvature_direction = (
            curvature_lateral / curvature_lateral_norm
            if curvature_lateral_norm > 1e-7 else np.zeros(2, dtype=float)
        )
        if effective_curve > 0.08 and np.linalg.norm(curvature_direction) > 0.5:
            # Path curvature defines the missing steering direction when the
            # active waypoint is still centred in the camera. This establishes
            # the required bend before reaching the branch.
            curvature_direction_weight = float(np.clip(
                effective_curve * (0.72 if desired_lateral_norm < 0.08 else 0.38),
                0.0,
                0.72,
            ))
            desired_direction = (
                (1.0 - curvature_direction_weight) * desired_direction
                + curvature_direction_weight * curvature_direction
            )
            desired_direction /= max(float(np.linalg.norm(desired_direction)), 1e-9)
        bearing_angle = float(math.atan2(
            desired_lateral_norm, max(float(desired_local[2]), -0.15)
        ))
        bearing_angle_deg = math.degrees(bearing_angle)
        heading_error = math.acos(float(np.clip(np.dot(forward, desired_world), -1.0, 1.0)))

        previous_bearing = self.auto_nav_last_bearing_angle_deg
        raw_improvement = (
            0.0 if previous_bearing is None else previous_bearing - bearing_angle_deg
        )
        self.auto_nav_last_bearing_angle_deg = bearing_angle_deg
        self.auto_nav_bearing_improvement += 0.18 * (
            raw_improvement - self.auto_nav_bearing_improvement
        )

        # Reverse only after genuinely overshooting the target-normal plane
        # outside the centreline corridor. A tight bend may put the target
        # behind the camera before this plane is crossed; that requires more
        # bending, not a direction change.
        true_overshoot = bool(
            raw_target_plane_progress > max(threshold * 0.7, 0.8)
            and path_lateral_error > max(tolerance, threshold * 1.4)
            and not crossed_in_corridor
        )
        if true_overshoot:
            self.auto_nav_behind_frames += 1
            self.auto_nav_front_frames = 0
        else:
            self.auto_nav_behind_frames = max(0, self.auto_nav_behind_frames - 1)
            if raw_target_plane_progress <= 0.1:
                self.auto_nav_front_frames += 1
            else:
                self.auto_nav_front_frames = 0
        if self.auto_nav_motion_mode != "reverse" and self.auto_nav_behind_frames >= 5:
            self.auto_nav_motion_mode = "reverse"
            self.auto_nav_front_frames = 0
        elif self.auto_nav_motion_mode == "reverse" and self.auto_nav_front_frames >= 3:
            self.auto_nav_motion_mode = "forward"
            self.auto_nav_behind_frames = 0
            self.auto_nav_forward_reengage_frames = 6
        reversing = self.auto_nav_motion_mode == "reverse"
        reengaging = bool(not reversing and self.auto_nav_forward_reengage_frames > 0)
        if reengaging:
            self.auto_nav_forward_reengage_frames -= 1

        # XML physics: two 27 mm elastic segments, identical bend elasticity,
        # proximal normalized torque limit 1.0 and distal limit 0.7. Proximal
        # creates the route curvature; distal supplies residual tip alignment.
        bearing_ratio = float(np.clip(bearing_angle / math.radians(95.0), 0.0, 1.0))
        cross_ratio = float(np.clip(
            path_lateral_error / max(tolerance * 3.0, 1e-9), 0.0, 1.0
        ))
        equivalent_bend = float(np.clip(
            0.72 * bearing_ratio + 0.24 * effective_curve + 0.16 * cross_ratio,
            0.0,
            0.96,
        ))
        if bearing_angle_deg < 3.0 and effective_curve < 0.08 and path_lateral_error < tolerance * 0.45:
            equivalent_bend = 0.0

        proximal_fraction = float(np.clip(
            0.86 - 0.24 * effective_curve - 0.16 * bearing_ratio,
            0.52,
            0.86,
        ))
        proximal_magnitude = min(1.0, equivalent_bend * proximal_fraction)
        residual_equivalent = max(0.0, equivalent_bend - proximal_magnitude)
        distal_magnitude = min(
            0.7,
            residual_equivalent / 0.65
            + max(0.0, bearing_ratio - 0.24) * 0.24
            + effective_curve * 0.12,
        )

        current_combined = (
            self.auto_nav_proximal_command + 0.65 * self.auto_nav_distal_command
        )
        current_combined_norm = float(np.linalg.norm(current_combined))
        current_direction = (
            current_combined / current_combined_norm
            if current_combined_norm > 1e-6 else desired_direction.copy()
        )
        direction_opposed = bool(
            np.linalg.norm(desired_direction) > 0.5
            and current_combined_norm > 0.035
            and float(np.dot(current_direction, desired_direction)) < -0.12
        )
        if direction_opposed:
            # Cable torque can only pull. Unload first and let the elastic
            # sections recenter before establishing the opposite curvature.
            proximal_reference = np.zeros(2, dtype=float)
            distal_reference = np.zeros(2, dtype=float)
            servo_phase = "弯曲方向改变，弹性卸载后建立新曲率"
        elif reversing:
            proximal_reference = self.auto_nav_proximal_command.copy()
            distal_reference = self.auto_nav_distal_command.copy()
            servo_phase = "越过目标平面，保持曲率低速后退"
        else:
            direction_alpha = 0.10 if effective_curve < 0.25 else 0.16
            blended_direction = (
                (1.0 - direction_alpha) * current_direction
                + direction_alpha * desired_direction
            )
            blended_direction /= max(float(np.linalg.norm(blended_direction)), 1e-9)
            damping_scale = float(np.clip(
                1.0 - max(self.auto_nav_bearing_improvement, 0.0) / 3.0,
                0.72,
                1.0,
            ))
            proximal_reference = blended_direction * proximal_magnitude * damping_scale
            distal_reference = blended_direction * distal_magnitude * damping_scale
            servo_phase = (
                "大弯道近端主弯、远端补偿"
                if effective_curve > 0.42 or bearing_angle_deg > 40.0
                else "物理模型闭环路径跟踪"
            )

        def rate_limited_state(current, desired, alpha, max_delta):
            candidate = current + alpha * (desired - current)
            delta = candidate - current
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_delta:
                delta *= max_delta / delta_norm
            return current + delta

        # The simulator applies another 0.22 torque low-pass. These limits
        # keep the outer loop slower than the elastic plant and avoid ringing.
        self.auto_nav_proximal_command = rate_limited_state(
            self.auto_nav_proximal_command, proximal_reference, 0.16, 0.018
        )
        self.auto_nav_distal_command = rate_limited_state(
            self.auto_nav_distal_command, distal_reference, 0.14, 0.020
        )
        proximal = self.auto_nav_proximal_command.copy()
        distal = self.auto_nav_distal_command.copy()
        combined_bend = proximal + 0.65 * distal
        desired_combined = proximal_reference + 0.65 * distal_reference
        bend_tracking_error = float(np.linalg.norm(desired_combined - combined_bend))
        bend_readiness = float(np.clip(
            1.0 - bend_tracking_error / max(float(np.linalg.norm(desired_combined)) + 0.05, 1e-9),
            0.12,
            1.0,
        ))

        # If MuJoCo is available, include measured segment angles in the speed
        # gate. The angles are feedback only; command generation remains valid
        # for real NDI mode where this signal is unavailable.
        measured_proximal_angle = None
        measured_distal_angle = None
        panel = getattr(getattr(self, "mujoco_simulator", None), "physics_panel", None)
        if panel is not None and len(getattr(panel, "segments", [])) >= 2:
            measured_proximal_angle = float(panel.segments[0].get("real_angle", 0.0))
            measured_distal_angle = float(panel.segments[1].get("real_angle", 0.0))
            expected_angle = 7.6 * (
                float(np.linalg.norm(proximal)) + float(np.linalg.norm(distal))
            )
            measured_angle = measured_proximal_angle + measured_distal_angle
            if expected_angle > 1.0:
                bend_readiness *= float(np.clip(measured_angle / expected_angle, 0.25, 1.0))

        max_step = self._auto_param("max_step") * self._auto_param("speed_scale")
        alignment_scale = float(np.clip(1.0 - bearing_angle / math.radians(115.0), 0.12, 1.0))
        curve_scale = float(np.clip(1.0 - 0.72 * effective_curve, 0.22, 1.0))
        cross_scale = float(np.clip(
            tolerance / max(path_lateral_error, tolerance), 0.22, 1.0
        ))
        desired_insertion = max_step * max(
            0.08,
            alignment_scale * curve_scale * cross_scale * (0.38 + 0.62 * bend_readiness),
        )
        if reversing:
            desired_insertion = -max_step * float(np.clip(
                0.12 + raw_target_plane_progress / max(threshold * 16.0, 1.0),
                0.12,
                0.30,
            ))
        distance_to_plane = max(-target_plane_progress, 0.0)
        if not reversing and target_plane_progress < 0.0:
            desired_insertion = min(
                desired_insertion,
                max(max_step * 0.08, distance_to_plane * 0.45 + threshold * 0.10),
            )
        previous_insertion = float(getattr(self, "auto_nav_last_insertion_step", 0.0))
        insertion_step = previous_insertion + float(np.clip(
            desired_insertion - previous_insertion,
            -max_step * 0.10,
            max_step * 0.10,
        ))
        if reversing:
            insertion_step = min(insertion_step, -max_step * 0.05)
        elif not reengaging:
            insertion_step = max(insertion_step, max_step * 0.08)
        self.auto_nav_last_insertion_step = insertion_step

        previous_position = self.auto_nav_last_position
        frame_motion = 0.0 if previous_position is None else float(
            np.linalg.norm(raw_position - np.asarray(previous_position, dtype=float))
        )
        self.auto_nav_last_position = raw_position.copy()
        self.auto_nav_filtered_speed += 0.18 * (
            frame_motion - self.auto_nav_filtered_speed
        )
        self.auto_nav_nearest_index = target_index
        self.auto_nav_progress = 100.0 * self.auto_nav_passed_count / max(1, len(path))
        all_path_local = (camera_rotation.T @ (path - position).T).T
        visual_error = np.asarray(error_local[:2], dtype=float) / max(abs(forward_error), 1.0)
        return {
            "steering": distal.tolist(),
            "steering_proximal": proximal.tolist(),
            "steering_distal": distal.tolist(),
            "insertion_delta": float(insertion_step / 577.0),
            "insertion_step_mm": float(insertion_step),
            "target_index": target_index,
            "nearest_index": target_index,
            "target_model_mm": target.tolist(),
            "all_path_local_mm": all_path_local.tolist(),
            "distance_to_target_mm": distance_error,
            "waypoint_reached": waypoint_reached,
            "waypoint_hold_frames": self.auto_nav_waypoint_reached_frames,
            "waypoint_local_mm": error_local.tolist(),
            "target_local_mm": error_local.tolist(),
            "guidance_target_model_mm": target.tolist(),
            "guidance_target_distance_mm": distance_error,
            "relative_position_error_mm": error_local.tolist(),
            "relative_bearing_error_deg": bearing_angle_deg,
            "visual_center_error": visual_error.tolist(),
            "visual_error_norm": float(np.linalg.norm(visual_error)),
            "visual_center_speed": alignment_scale,
            "lateral_distance_mm": lateral_error,
            "forward_error_mm": forward_error,
            "azimuth_deg": math.degrees(float(math.atan2(error_local[0], max(forward_error, 1e-6)))),
            "elevation_deg": math.degrees(float(math.atan2(error_local[1], max(forward_error, 1e-6)))),
            "heading_error_deg": math.degrees(heading_error),
            "servo_phase": servo_phase,
            "estimated_speed_mm_per_frame": self.auto_nav_filtered_speed,
            "straight_segment": effective_curve < 0.12,
            "straight_path_mode": effective_curve < 0.12 and bearing_angle_deg < 12.0,
            "turn_active": effective_curve >= 0.12 or bearing_angle_deg >= 12.0,
            "turn_severity": max(effective_curve, bearing_ratio),
            "curve_severity": effective_curve,
            "curvature_throttle": curve_scale,
            "distal_assist": float(np.linalg.norm(distal) / 0.7),
            "proximal_utilization": float(np.linalg.norm(proximal)),
            "cruise_boost": 1.0,
            "recovery_mode": reversing or direction_opposed,
            "retracting": reversing,
            "motion_mode": self.auto_nav_motion_mode,
            "special_case": reversing or direction_opposed,
            "anti_curl_release": direction_opposed,
            "cross_track_error_mm": path_lateral_error,
            "within_centerline_tolerance": path_lateral_error <= tolerance,
            "global_curve_angle_deg": math.degrees(curve_angle),
            "global_preview_angle_deg": math.degrees(curve_angle),
            "forward_alignment": float(np.dot(forward, desired_world)),
            "overshot_waypoint": true_overshoot,
            "raw_target_plane_progress_mm": raw_target_plane_progress,
            "bend_tracking_error": bend_tracking_error,
            "bend_readiness": bend_readiness,
            "bearing_improvement_deg_per_frame": self.auto_nav_bearing_improvement,
            "equivalent_bend_demand": equivalent_bend,
            "measured_proximal_angle_deg": measured_proximal_angle,
            "measured_distal_angle_deg": measured_distal_angle,
            "target_transition": target_changed,
            "continuous_action": True,
        }

    def _compute_auto_navigation_action(self):
        position = np.asarray(getattr(self, "virtual_position", []), dtype=float)
        rotation = np.asarray(getattr(self, "mapped_tip_camera_dir", []), dtype=float)
        if position.shape != (3,) or rotation.shape != (3, 3):
            return None
        return self.auto_nav_v2.compute(
            position=position,
            rotation=rotation,
            target_index=self.auto_nav_target_index,
            point_threshold=self._auto_param("point_threshold"),
            centerline_tolerance=self._auto_param("centerline_tolerance"),
            max_step=self._auto_param("max_step"),
            speed_scale=self._auto_param("speed_scale"),
        )

        # Legacy curvature-preview controller retained below for reference.
        path = self._get_auto_navigation_path()
        position = getattr(self, "virtual_position", None)
        if path.ndim != 2 or len(path) < 2 or position is None:
            return None
        self.auto_nav_control_frames = int(getattr(self, "auto_nav_control_frames", 0)) + 1
        position = np.asarray(position, dtype=float)

        segments = np.diff(path, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))

        # Project onto forward path segments and track continuous arc length.
        # This prevents the lookahead target from jumping between discrete points.
        previous_nearest = int(np.clip(getattr(self, "auto_nav_nearest_index", 0), 0, len(path) - 1))
        search_start = min(len(path) - 2, max(0, previous_nearest - 2))
        search_end = min(len(path) - 1, previous_nearest + 30)
        best_distance = float("inf")
        candidate_distance = getattr(self, "auto_nav_path_distance", 0.0)
        for segment_index in range(search_start, max(search_start + 1, search_end)):
            length = segment_lengths[segment_index]
            if length <= 1e-9:
                continue
            alpha = float(np.clip(
                np.dot(position - path[segment_index], segments[segment_index]) / (length * length), 0.0, 1.0
            ))
            projected = path[segment_index] + alpha * segments[segment_index]
            distance = float(np.linalg.norm(position - projected))
            if distance < best_distance:
                best_distance = distance
                candidate_distance = cumulative[segment_index] + alpha * length
        self.auto_nav_path_distance = max(float(getattr(self, "auto_nav_path_distance", 0.0)), candidate_distance)
        nearest_index = int(np.clip(np.searchsorted(cumulative, self.auto_nav_path_distance), 0, len(path) - 1))
        self.auto_nav_nearest_index = nearest_index
        nearest_position = np.empty(3, dtype=float)
        nearest_segment = max(0, min(nearest_index - 1, len(segment_lengths) - 1))
        nearest_length = max(segment_lengths[nearest_segment], 1e-9)
        nearest_alpha = np.clip(
            (self.auto_nav_path_distance - cumulative[nearest_segment]) / nearest_length, 0.0, 1.0
        )
        nearest_position[:] = path[nearest_segment] + nearest_alpha * segments[nearest_segment]
        cross_track_error = float(np.linalg.norm(nearest_position - position))

        if self.auto_nav_last_position is not None:
            frame_motion = float(np.linalg.norm(position - self.auto_nav_last_position))
            self.auto_nav_stall_frames = self.auto_nav_stall_frames + 1 if frame_motion < 0.03 else 0
        else:
            frame_motion = 0.0
            self.auto_nav_stall_frames = 0
        self.auto_nav_last_position = position.copy()

        tolerance = self._auto_param("centerline_tolerance")
        off_path_recovery = cross_track_error > tolerance
        stall_recovery = self.auto_nav_stall_frames >= 20
        recovery_mode = off_path_recovery or stall_recovery
        if off_path_recovery:
            lookahead = self._auto_param("lookahead") * 0.70
        elif stall_recovery:
            lookahead = self._auto_param("lookahead") * 1.35
        else:
            lookahead = self._auto_param("lookahead") * 1.10

        nearest_tangent = segments[nearest_segment] / max(segment_lengths[nearest_segment], 1e-9)
        curve_probe_distance = min(
            float(cumulative[-1]),
            self.auto_nav_path_distance + max(self._auto_param("lookahead") * 1.8, 12.0),
        )
        curve_probe_segment = int(np.clip(
            np.searchsorted(cumulative, curve_probe_distance) - 1, nearest_segment, len(segments) - 1
        ))
        curve_probe_tangent = segments[curve_probe_segment] / max(segment_lengths[curve_probe_segment], 1e-9)
        local_curve_angle_deg = math.degrees(
            math.acos(float(np.clip(np.dot(nearest_tangent, curve_probe_tangent), -1.0, 1.0)))
        )

        # Scan only a bounded arc-length horizon. A fixed number of path
        # segments can include a distant bifurcation on densely sampled paths
        # and incorrectly trigger a hard bend while the tip is still straight.
        bend_preview_horizon = max(self._auto_param("lookahead") * 2.0, 10.0)
        forward_scan_distance = min(float(cumulative[-1]), self.auto_nav_path_distance + bend_preview_horizon)
        forward_scan_end = int(np.clip(
            np.searchsorted(cumulative, forward_scan_distance), nearest_segment + 1, len(segments)
        ))
        forward_turn_angles = []
        weighted_turn_scores = []
        nearest_bend_distance = float("inf")
        previous_scan_tangent = nearest_tangent
        for scan_index in range(nearest_segment + 1, forward_scan_end):
            scan_tangent = segments[scan_index] / max(segment_lengths[scan_index], 1e-9)
            segment_turn_deg = math.degrees(math.acos(float(np.clip(
                np.dot(previous_scan_tangent, scan_tangent), -1.0, 1.0
            ))))
            accumulated_turn_deg = math.degrees(math.acos(float(np.clip(
                np.dot(nearest_tangent, scan_tangent), -1.0, 1.0
            ))))
            turn_distance = max(0.0, float(cumulative[scan_index] - self.auto_nav_path_distance))
            distance_weight = float(np.clip(1.0 - turn_distance / bend_preview_horizon, 0.0, 1.0))
            forward_turn_angles.append(segment_turn_deg)
            weighted_turn_scores.append(max(segment_turn_deg, accumulated_turn_deg * 0.45) * distance_weight)
            if segment_turn_deg >= 5.0 or accumulated_turn_deg >= 10.0:
                nearest_bend_distance = min(nearest_bend_distance, turn_distance)
            previous_scan_tangent = scan_tangent
        peak_forward_turn_deg = max(forward_turn_angles, default=0.0)
        weighted_forward_turn_deg = max(weighted_turn_scores, default=0.0)
        bend_proximity = (
            float(np.clip(1.0 - nearest_bend_distance / bend_preview_horizon, 0.0, 1.0))
            if np.isfinite(nearest_bend_distance)
            else 0.0
        )
        probe_distance = max(0.0, curve_probe_distance - self.auto_nav_path_distance)
        probe_weight = float(np.clip(1.0 - probe_distance / bend_preview_horizon, 0.0, 1.0))
        sharp_turn_score = float(np.clip(
            max(
                local_curve_angle_deg * probe_weight / 52.0,
                weighted_forward_turn_deg / 24.0,
            ),
            0.0,
            1.0,
        ))
        # Explicit straight/turn state machine. Detecting a bend in the preview
        # horizon is not permission to bend yet; the bend must enter the
        # mechanical activation zone first. Hysteresis prevents mode chatter.
        turn_activation_distance = float(np.clip(
            max(self._auto_param("point_threshold") * 1.5, self._auto_param("lookahead") * 0.35),
            2.5,
            4.0,
        ))
        if self.auto_nav_turn_active:
            if (
                (not np.isfinite(nearest_bend_distance) or nearest_bend_distance > turn_activation_distance * 1.8)
                and local_curve_angle_deg < 8.0
            ):
                self.auto_nav_turn_active = False
        elif (
            np.isfinite(nearest_bend_distance)
            and nearest_bend_distance <= turn_activation_distance
            and sharp_turn_score >= 0.18
        ):
            self.auto_nav_turn_active = True

        effective_turn_score = sharp_turn_score if self.auto_nav_turn_active else 0.0
        severity_rate = 0.12 if effective_turn_score > self.auto_nav_turn_severity else 0.38
        self.auto_nav_turn_severity += severity_rate * (effective_turn_score - self.auto_nav_turn_severity)
        turn_severity = float(self.auto_nav_turn_severity)
        straight_path_mode = not self.auto_nav_turn_active
        lookahead *= float(np.clip(1.0 - local_curve_angle_deg / 85.0, 0.25, 1.0))

        target_index = int(np.clip(self.auto_nav_target_index, 1, len(path) - 1))
        target_segment = target_index - 1
        target = path[target_index]
        waypoint_delta = target - position
        waypoint_distance = float(np.linalg.norm(waypoint_delta))

        # Steer toward a forward point on the planned arc, while retaining the
        # current waypoint as the progress gate. Chasing a close waypoint after
        # it slips behind the tip creates a pursuit circle and can curl the
        # continuum robot even when the planned path is straight.
        guidance_lookahead = lookahead * (0.85 if straight_path_mode else 0.30)
        guidance_distance = min(
            float(cumulative[-1]),
            max(float(cumulative[target_index]), self.auto_nav_path_distance + guidance_lookahead),
        )
        if straight_path_mode and np.isfinite(nearest_bend_distance):
            # Never let a straight-mode guidance point cross into the bend.
            # It remains on the incoming tangent until the turn state activates.
            bend_arc_distance = self.auto_nav_path_distance + nearest_bend_distance
            guidance_distance = min(guidance_distance, max(
                self.auto_nav_path_distance + 1.0,
                bend_arc_distance - max(1.0, self._auto_param("point_threshold") * 0.6),
            ))
        guidance_segment = int(np.clip(
            np.searchsorted(cumulative, guidance_distance) - 1, 0, len(segments) - 1
        ))
        guidance_segment_length = max(segment_lengths[guidance_segment], 1e-9)
        guidance_alpha = float(np.clip(
            (guidance_distance - cumulative[guidance_segment]) / guidance_segment_length, 0.0, 1.0
        ))
        guidance_target = path[guidance_segment] + guidance_alpha * segments[guidance_segment]
        delta = guidance_target - position
        guidance_target_distance = float(np.linalg.norm(delta))
        distance_to_target = guidance_target_distance

        target_tangent = segments[guidance_segment] / max(segment_lengths[guidance_segment], 1e-9)
        preview_turn_deg = math.degrees(math.acos(float(np.clip(np.dot(nearest_tangent, target_tangent), -1.0, 1.0))))

        camera_rotation = getattr(self, "mapped_tip_camera_dir", None)
        if camera_rotation is not None and np.asarray(camera_rotation).shape == (3, 3):
            camera_rotation = np.asarray(camera_rotation, dtype=float)
            local_target = camera_rotation.T @ delta
            local_waypoint = camera_rotation.T @ waypoint_delta
            local_path_tangent = camera_rotation.T @ target_tangent
            local_curve_direction = camera_rotation.T @ (curve_probe_tangent - nearest_tangent)
        else:
            local_target = delta
            local_waypoint = waypoint_delta
            local_path_tangent = target_tangent
            local_curve_direction = curve_probe_tangent - nearest_tangent

        # Direct relative-pose controller: local X/Y determine bending
        # direction, while target distance and local Z determine insertion.
        local_x, local_y, local_z = [float(value) for value in local_target]
        lateral_distance = float(math.hypot(local_x, local_y))
        forward_alignment = float(local_z / max(distance_to_target, 1e-9))
        # The bend direction comes directly from the target's relative bearing.
        # For an exactly rearward target, path curvature supplies the only
        # geometrically meaningful turn direction and avoids a zero-action lock.
        heading_angle = float(math.atan2(lateral_distance, local_z))
        turn_direction = np.asarray([local_x, local_y], dtype=float)
        if np.linalg.norm(turn_direction) < 0.5 and local_z <= 0.0:
            turn_direction = np.asarray(local_curve_direction[:2], dtype=float)
        if np.linalg.norm(turn_direction) < 1e-6 and local_z <= 0.0:
            turn_direction = np.asarray(local_path_tangent[:2], dtype=float)
        if np.linalg.norm(turn_direction) < 1e-6:
            turn_direction = self.auto_nav_last_turn_direction.copy()

        # Build curvature progressively before reaching a bifurcation. The
        # anticipation grows only near the current waypoint, so the tip still
        # passes that waypoint instead of cutting across the centerline.
        curve_turn_direction = np.asarray(local_curve_direction[:2], dtype=float)
        curve_turn_norm = float(np.linalg.norm(curve_turn_direction))
        # Mechanical prebending needs a wider window than the target
        # lookahead, which intentionally shrinks in a sharp turn.
        prebend_distance = max(
            self._auto_param("lookahead") * 1.50,
            self._auto_param("point_threshold") * 4.0,
            bend_preview_horizon * 0.65,
        )
        # Prebend is governed by distance to the actual bend, not distance to
        # every intermediate waypoint on the preceding straight section.
        prebend_proximity = bend_proximity * float(np.clip(
            1.0 - nearest_bend_distance / max(prebend_distance, 1e-9), 0.0, 1.0
        )) if self.auto_nav_turn_active and np.isfinite(nearest_bend_distance) else 0.0
        prebend_angle = (
            math.radians(min(local_curve_angle_deg, 80.0))
            * turn_severity
            * prebend_proximity
            * 0.72
        )
        direct_turn_norm = float(np.linalg.norm(turn_direction))
        direct_turn_direction = (
            turn_direction / direct_turn_norm
            if direct_turn_norm > 1e-6
            else np.zeros(2, dtype=float)
        )
        desired_heading_vector = direct_turn_direction * heading_angle

        # Align with the planned path tangent as well as the next point.
        # Tangent alignment prevents pure-pursuit circles on long straight
        # sections when the tip has a temporary lateral heading error.
        tangent_lateral = np.asarray(local_path_tangent[:2], dtype=float)
        tangent_lateral_norm = float(np.linalg.norm(tangent_lateral))
        tangent_heading_angle = float(math.atan2(tangent_lateral_norm, float(local_path_tangent[2])))
        anti_curl_release = bool(straight_path_mode and float(local_path_tangent[2]) < 0.15)
        if tangent_lateral_norm > 1e-6:
            tangent_weight = 0.82 if straight_path_mode else 0.24
            desired_heading_vector = (
                (1.0 - tangent_weight) * desired_heading_vector
                + tangent_weight * tangent_lateral / tangent_lateral_norm * tangent_heading_angle
            )
        if curve_turn_norm > 1e-6 and prebend_angle > 1e-6:
            desired_heading_vector += curve_turn_direction / curve_turn_norm * prebend_angle
        desired_heading_angle = float(np.linalg.norm(desired_heading_vector))
        if desired_heading_angle > 1e-6:
            turn_direction = desired_heading_vector / desired_heading_angle
            heading_angle = desired_heading_angle

        if np.linalg.norm(turn_direction) > 1e-6:
            turn_direction /= np.linalg.norm(turn_direction)

            # Keep a stable steering side through a bifurcation. A transient
            # target crossing the optical axis must not reverse both sections.
            committed = self.auto_nav_committed_turn_direction
            if turn_severity > 0.32 and np.linalg.norm(committed) > 1e-6:
                agreement = float(np.dot(turn_direction, committed))
                if agreement < 0.15:
                    turn_direction = committed.copy()
                else:
                    blend = 0.88 if turn_severity > 0.65 else 0.70
                    turn_direction = blend * committed + (1.0 - blend) * turn_direction
                    turn_direction /= max(np.linalg.norm(turn_direction), 1e-9)
            if turn_severity > 0.24:
                committed_blend = 0.92 if np.linalg.norm(committed) > 1e-6 else 0.0
                self.auto_nav_committed_turn_direction = (
                    committed_blend * committed + (1.0 - committed_blend) * turn_direction
                )
                self.auto_nav_committed_turn_direction /= max(
                    np.linalg.norm(self.auto_nav_committed_turn_direction), 1e-9
                )
            elif math.degrees(heading_angle) < 8.0:
                self.auto_nav_committed_turn_direction *= 0.85
            self.auto_nav_last_turn_direction = turn_direction.copy()
        heading_error = turn_direction * heading_angle
        heading_angle_deg = math.degrees(heading_angle)
        straight_deadband_deg = 4.5 if cross_track_error <= tolerance else 2.5
        if straight_path_mode and heading_angle_deg <= straight_deadband_deg:
            heading_error[:] = 0.0
        else:
            heading_error[np.abs(heading_error) < math.radians(2.0)] = 0.0
        heading_error_rate = heading_error - self.auto_nav_previous_heading_error
        self.auto_nav_previous_heading_error = heading_error.copy()

        # Subtract the error-rate term to damp overshoot instead of amplifying
        # rapid bearing changes near a bifurcation.
        if straight_path_mode:
            damped_error = heading_error - 0.45 * heading_error_rate
        else:
            damped_error = heading_error - (0.28 + 0.18 * turn_severity) * heading_error_rate
        distance_factor = float(np.clip(distance_to_target / max(lookahead, 1e-6), 0.45, 1.0))
        # Proximal-first allocation: the proximal section performs the gross
        # bend. At a sharp bifurcation its usable range grows substantially;
        # the distal section then cooperates to reach the required curvature.
        proximal_limit = (0.10 if straight_path_mode else 0.24) + 0.46 * turn_severity
        distal_limit = (0.0 if straight_path_mode else 0.18) + 0.42 * turn_severity

        def limit_bend(vector, limit):
            vector = np.asarray(vector, dtype=float)
            magnitude = float(np.linalg.norm(vector))
            return vector if magnitude <= limit else vector * (limit / max(magnitude, 1e-9))

        straight_gain = 0.075 if straight_path_mode else 0.18
        proximal_gain = ((0.22 if recovery_mode else straight_gain) + 0.82 * turn_severity) * distance_factor
        proximal_target = limit_bend(damped_error * proximal_gain, proximal_limit)
        proximal_utilization = float(np.linalg.norm(proximal_target) / max(proximal_limit, 1e-9))
        if straight_path_mode:
            distal_assist = 0.0
        else:
            distal_assist = float(np.clip(
                max(
                    (proximal_utilization - 0.58) / 0.42,
                    heading_angle_deg / 48.0 - 0.18,
                    (turn_severity - 0.34) / 0.50,
                ),
                0.0,
                1.0,
            ))
        distal_gain = ((0.26 if recovery_mode else 0.20) + 0.62 * turn_severity) * distance_factor * distal_assist
        distal_target = limit_bend(damped_error * distal_gain, distal_limit)
        if straight_path_mode:
            distal_target[:] = 0.0

        # Once the target is near the optical axis, actively release both
        # sections on straight paths. Preserve the required bend while passing
        # through a curved bifurcation.
        if heading_angle_deg < 4.0 and turn_severity < 0.20:
            distal_target[:] = 0.0
            proximal_target[:] = 0.0
        if anti_curl_release:
            # On a geometrically straight path a rear-facing tangent indicates
            # that the continuum section has curled past the useful steering
            # range. Releasing cable torque unwinds it instead of completing a
            # pursuit circle around a waypoint.
            distal_target[:] = 0.0
            proximal_target[:] = 0.0
            self.auto_nav_committed_turn_direction *= 0.5

        proximal_rate_limit = (0.014 if recovery_mode else 0.010) + 0.026 * turn_severity
        distal_rate_limit = (0.012 if recovery_mode else 0.008) + 0.021 * turn_severity
        if straight_path_mode:
            proximal_rate_limit = min(proximal_rate_limit, 0.004)
            distal_rate_limit = max(distal_rate_limit, 0.012)
        if anti_curl_release:
            proximal_rate_limit = max(proximal_rate_limit, 0.025)
            distal_rate_limit = max(distal_rate_limit, 0.020)
        startup_steering_ramp = float(np.clip((self.auto_nav_control_frames - 2) / 24.0, 0.12, 1.0))
        proximal_target *= startup_steering_ramp
        distal_target *= startup_steering_ramp
        if not anti_curl_release:
            proximal_rate_limit *= startup_steering_ramp
            distal_rate_limit *= startup_steering_ramp
        self.auto_nav_proximal_command += np.clip(
            proximal_target - self.auto_nav_proximal_command, -proximal_rate_limit, proximal_rate_limit
        )
        self.auto_nav_distal_command += np.clip(
            distal_target - self.auto_nav_distal_command, -distal_rate_limit, distal_rate_limit
        )
        proximal_steering = self.auto_nav_proximal_command.copy()
        distal_steering = self.auto_nav_distal_command.copy()
        steering = distal_steering.copy()

        # act_slid_M spans 577 mm. Convert the requested millimetres per frame
        # to the actuator's normalized slider range instead of using /100.
        max_step_mm = self._auto_param("max_step")
        distance_speed = float(np.clip(
            distance_to_target / max(self._auto_param("point_threshold") * 3.0, 3.0), 0.08, 1.0
        ))
        alignment_speed = float(np.clip(math.cos(math.radians(heading_angle_deg)), 0.0, 1.0) ** 2)
        centerline_speed = float(np.clip(tolerance / max(cross_track_error, tolerance), 0.20, 1.0))
        anticipated_turn_deg = max(preview_turn_deg, local_curve_angle_deg)
        curvature_speed = float(np.clip(1.0 - anticipated_turn_deg / 70.0, 0.08, 1.0))
        visual_center_error = np.array([local_x, local_y], dtype=float) / max(abs(local_z), 1.0)
        visual_center_speed = float(np.clip(1.0 - np.linalg.norm(visual_center_error), 0.35, 1.0))
        recovery_speed = 0.45 if recovery_mode else 1.0
        cruise_boost = (
            1.45
            if heading_angle_deg < 8.0 and cross_track_error < tolerance * 0.6 and anticipated_turn_deg < 10.0
            else 1.0
        )
        adaptive_step_mm = (
            max_step_mm
            * self._auto_param("speed_scale")
            * distance_speed
            * alignment_speed
            * centerline_speed
            * curvature_speed
            * visual_center_speed
            * recovery_speed
            * cruise_boost
        )
        # Insertion is the throttle: keep a positive base feed while steering,
        # and only stop/retract for genuinely unsafe geometry.
        base_throttle_ratio = 0.32 - 0.17 * turn_severity
        base_throttle_mm = max_step_mm * self._auto_param("speed_scale") * base_throttle_ratio
        insertion_step_mm = max(base_throttle_mm, adaptive_step_mm)
        insertion_step_mm = min(insertion_step_mm, distance_to_target)
        special_case = bool(local_z < -1.0 or heading_angle_deg > 100.0 or anti_curl_release)
        if special_case:
            insertion_step_mm = max_step_mm * self._auto_param("speed_scale") * 0.05
        elif local_z <= 0.0 or heading_angle_deg > 82.0:
            insertion_step_mm = max_step_mm * self._auto_param("speed_scale") * 0.10
        insertion_step_mm = max(insertion_step_mm, 1e-4)
        insertion_delta = insertion_step_mm / 577.0

        waypoint_pass_tolerance = max(self._auto_param("point_threshold"), tolerance * 0.75)
        waypoint_reached = bool(
            waypoint_distance <= self._auto_param("point_threshold")
            or (
                self.auto_nav_path_distance >= float(cumulative[target_index])
                and cross_track_error <= waypoint_pass_tolerance
            )
        )
        self.auto_nav_progress = 100.0 * self.auto_nav_passed_count / max(1, len(path))
        return {
            "steering": steering.tolist(),
            "steering_proximal": proximal_steering.tolist(),
            "steering_distal": distal_steering.tolist(),
            "insertion_delta": float(insertion_delta),
            "insertion_step_mm": float(insertion_step_mm),
            "target_index": target_index,
            "nearest_index": nearest_index,
            "target_model_mm": target.tolist(),
            "distance_to_target_mm": waypoint_distance,
            "waypoint_reached": waypoint_reached,
            "waypoint_pass_tolerance_mm": waypoint_pass_tolerance,
            "waypoint_local_mm": local_waypoint.tolist(),
            "guidance_target_model_mm": guidance_target.tolist(),
            "guidance_target_distance_mm": guidance_target_distance,
            "target_local_mm": local_target.tolist(),
            "path_tangent_local": local_path_tangent.tolist(),
            "visual_center_error": visual_center_error.tolist(),
            "visual_center_speed": visual_center_speed,
            "lateral_distance_mm": lateral_distance,
            "azimuth_deg": math.degrees(float(heading_error[0])),
            "elevation_deg": math.degrees(float(heading_error[1])),
            "cruise_boost": cruise_boost,
            "preview_turn_deg": preview_turn_deg,
            "anticipated_turn_deg": anticipated_turn_deg,
            "peak_forward_turn_deg": peak_forward_turn_deg,
            "weighted_forward_turn_deg": weighted_forward_turn_deg,
            "nearest_bend_distance_mm": None if not np.isfinite(nearest_bend_distance) else nearest_bend_distance,
            "bend_proximity": bend_proximity,
            "turn_activation_distance_mm": turn_activation_distance,
            "turn_active": self.auto_nav_turn_active,
            "turn_severity": turn_severity,
            "straight_path_mode": straight_path_mode,
            "anti_curl_release": anti_curl_release,
            "prebend_angle_deg": math.degrees(prebend_angle),
            "prebend_proximity": prebend_proximity,
            "startup_steering_ramp": startup_steering_ramp,
            "curvature_speed": curvature_speed,
            "retracting": False,
            "special_case": special_case,
            "proximal_utilization": proximal_utilization,
            "distal_assist": distal_assist,
            "proximal_limit": proximal_limit,
            "distal_limit": distal_limit,
            "cross_track_error_mm": cross_track_error,
            "forward_alignment": forward_alignment,
            "heading_error_deg": heading_angle_deg,
            "frame_motion_mm": frame_motion,
            "stall_frames": self.auto_nav_stall_frames,
            "recovery_mode": recovery_mode,
            "within_centerline_tolerance": cross_track_error <= tolerance,
        }

    def _apply_simulation_auto_action(self, action):
        if getattr(self, "is_simulation_mode", False) and self.mujoco_simulator is not None:
            try:
                self.mujoco_simulator.set_auto_navigation_action(action)
            except Exception as e:
                print(f"MuJoCo 自动导航 action 应用失败: {e}")

    def _automation_frame_update(self):
        try:
            if self.auto_nav_state == "running":
                action = self._compute_auto_navigation_action()
                if action is None:
                    self.stop_auto_navigation()
                    return
                else:
                    self.auto_nav_action = action
                    self.auto_nav_action_cache.append({"timestamp": time.time(), **action})
                    self._update_navigation_target_window(action)
                    waypoint_reached = bool(action.get(
                        "waypoint_reached",
                        action["distance_to_target_mm"] <= self._auto_param("point_threshold"),
                    ))
                    if waypoint_reached:
                        path = self._get_auto_navigation_path()
                        # Strict point-by-point tracking: advance at most one
                        # index per frame, and the index is never allowed to
                        # move backward or jump across unconfirmed waypoints.
                        if self.auto_nav_target_index < len(path) - 1:
                            self.auto_nav_target_index += 1
                            self.auto_nav_passed_count = self.auto_nav_target_index
                            action = self._compute_auto_navigation_action()
                            self._refresh_auto_navigation_guides()
                            self.auto_nav_action = action
                            self._update_navigation_target_window(action)
                        else:
                            self.auto_nav_passed_count = len(path)
                            self.auto_nav_progress = 100.0
                            self.stop_auto_navigation()
                            self.auto_nav_status_label.setText("状态: 已依次通过全部规划路径点")
                            self.refresh_automation_path_status()
                            return
                    if getattr(self, "is_simulation_mode", False):
                        self._apply_simulation_auto_action(action)
                        panel = getattr(self.mujoco_simulator, "physics_panel", None)
                        slider_saturated = bool(panel is not None and panel.slider_val >= 0.999)
                        if slider_saturated and action["target_index"] < len(self._get_auto_navigation_path()) - 1:
                            self.auto_nav_status_label.setText("状态: 已达到最大进给，正在继续转向修正")
                        elif action.get("servo_phase"):
                            self.auto_nav_status_label.setText(
                                f"状态: {action['servo_phase']}，目标距离 {action['distance_to_target_mm']:.1f} mm，"
                                f"中心偏差 {action['visual_error_norm']:.3f}"
                            )
                        elif action.get("anti_curl_release", False):
                            self.auto_nav_status_label.setText(
                                "状态: 检测到直线路段卷曲趋势，正在快速卸载弯曲并回正"
                            )
                        elif action.get("motion_mode") == "unstick_reverse":
                            self.auto_nav_status_label.setText(
                                f"状态: 检测到运动受限，正在短距离后撤释放姿态，"
                                f"已后撤调整 {action.get('macro_action_frames', 0)} 帧"
                            )
                        elif action.get("motion_mode") == "unstick_realign":
                            self.auto_nav_status_label.setText(
                                "状态: 后撤完成，正在低速重新对准路径"
                            )
                        elif action["turn_severity"] > 0.55:
                            self.auto_nav_status_label.setText(
                                f"状态: 大角度分叉协同转向，目标距离 {action['distance_to_target_mm']:.1f} mm，"
                                f"弯曲强度 {action['turn_severity'] * 100:.0f}%"
                            )
                        elif action["recovery_mode"]:
                            self.auto_nav_status_label.setText(
                                f"状态: 路径恢复，目标距离 {action['distance_to_target_mm']:.1f} mm，"
                                f"方位 {action['azimuth_deg']:.1f}°/{action['elevation_deg']:.1f}°"
                            )
                        elif action.get("straight_path_mode", False):
                            self.auto_nav_status_label.setText(
                                f"状态: 直线稳定巡航，目标距离 {action['distance_to_target_mm']:.1f} mm，"
                                f"给进 {action['insertion_step_mm']:.2f} mm/帧"
                            )
                        elif action["cruise_boost"] > 1.0:
                            self.auto_nav_status_label.setText(
                                f"状态: 稳定巡航，目标距离 {action['distance_to_target_mm']:.1f} mm，"
                                f"给进 {action['insertion_step_mm']:.2f} mm/帧"
                            )
                        else:
                            self.auto_nav_status_label.setText(
                                f"状态: 自动跟踪，目标距离 {action['distance_to_target_mm']:.1f} mm，"
                                f"方位 {action['azimuth_deg']:.1f}°/{action['elevation_deg']:.1f}°"
                            )
                    else:
                        self.auto_nav_status_label.setText("状态: 真实模式，仅生成并缓存 action")
                self._update_auto_navigation_progress()
            self.refresh_automation_path_status()
            if self.vla_recording and self.vla_record_cb.isChecked():
                self._record_vla_sample()
        except Exception as e:
            print(f"自动化帧更新异常: {e}")

    def start_vla_episode(self):
        if self.vla_recording:
            return
        try:
            root = os.path.abspath(self.vla_save_path_edit.text().strip() or os.path.join(PROJECT_ROOT, "vla_dataset"))
            os.makedirs(root, exist_ok=True)
            episode_name = datetime.now().strftime("episode_%Y%m%d_%H%M%S_%f")
            self.vla_episode_dir = os.path.join(root, episode_name)
            for name in ("real_images", "mujoco_images", "masks"):
                os.makedirs(os.path.join(self.vla_episode_dir, name), exist_ok=True)
            self.vla_jsonl_file = open(os.path.join(self.vla_episode_dir, "vla_dataset.jsonl"), "w", encoding="utf-8")
            self.vla_csv_file = open(
                os.path.join(self.vla_episode_dir, "vla_dataset.csv"), "w", newline="", encoding="utf-8-sig"
            )
            fields = [
                "sample_index", "timestamp", "environment", "navigation_state", "progress",
                "target_index", "real_image_path", "mujoco_image_path", "mask_path",
                "raw_pose", "model_pose", "axis_control", "action", "path_state", "calibration",
            ]
            self.vla_csv_writer = csv.DictWriter(self.vla_csv_file, fieldnames=fields)
            self.vla_csv_writer.writeheader()
            self.vla_sample_index = 0
            self.vla_recording = True
            self.vla_record_cb.setChecked(True)
            self._write_vla_metadata(completed=False)
            self.vla_status_label.setText(f"正在采集: {episode_name}")
        except Exception as e:
            self.vla_recording = False
            self._close_vla_files()
            QMessageBox.critical(self, "VLA 采集启动失败", str(e))

    def stop_vla_episode(self):
        if not self.vla_recording:
            return
        try:
            self._write_vla_metadata(completed=True)
        except Exception as e:
            print(f"VLA metadata 保存失败: {e}")
        self.vla_recording = False
        self._close_vla_files()
        if hasattr(self, "vla_status_label"):
            self.vla_status_label.setText(f"采集完成，共 {self.vla_sample_index} 帧")

    def _close_vla_files(self):
        for attr in ("vla_jsonl_file", "vla_csv_file"):
            file_obj = getattr(self, attr, None)
            if file_obj is not None:
                try:
                    file_obj.flush()
                    file_obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        self.vla_csv_writer = None

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {key: MainWindow._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [MainWindow._json_safe(item) for item in value]
        return value

    def _write_vla_metadata(self, completed=False):
        if not self.vla_episode_dir:
            return
        metadata = {
            "episode": os.path.basename(self.vla_episode_dir),
            "created_at": datetime.now().isoformat(),
            "completed": completed,
            "sample_count": self.vla_sample_index,
            "environment": "mujoco" if self.is_simulation_mode else "real",
            "mujoco_xml": self.mujoco_xml_path if self.is_simulation_mode else None,
            "capture_selection": {key: check.isChecked() for key, check in self.vla_capture_checks.items()},
            "navigation_parameters": {key: self._auto_param(key) for key in self.auto_param_widgets},
            "path_point_count": len(getattr(self, "smoothpath", [])),
        }
        with open(os.path.join(self.vla_episode_dir, "metadata.json"), "w", encoding="utf-8") as file:
            json.dump(self._json_safe(metadata), file, ensure_ascii=False, indent=2)

    def _save_vla_image(self, category, image):
        if image is None:
            return None
        relative = os.path.join(category, f"{self.vla_sample_index:08d}.png")
        absolute = os.path.join(self.vla_episode_dir, relative)
        try:
            if cv2.imwrite(absolute, np.asarray(image)):
                return relative.replace("\\", "/")
        except Exception as e:
            print(f"VLA 图像保存失败 ({category}): {e}")
        return None

    def _record_vla_sample(self):
        if self.vla_jsonl_file is None or self.vla_csv_writer is None:
            return
        try:
            selected = {key: check.isChecked() for key, check in self.vla_capture_checks.items()}
            sample = {
                "sample_index": self.vla_sample_index,
                "timestamp": time.time(),
                "environment": "mujoco" if self.is_simulation_mode else "real",
                "navigation_state": self.auto_nav_state,
                "progress": self.auto_nav_progress,
                "target_index": self.auto_nav_target_index,
            }
            if selected["real_image"] and not self.is_simulation_mode:
                sample["real_image_path"] = self._save_vla_image("real_images", self.latest_real_frame_bgr)
            if selected["mujoco_image"] and self.is_simulation_mode and self.mujoco_simulator is not None:
                sample["mujoco_image_path"] = self._save_vla_image(
                    "mujoco_images", self.mujoco_simulator.latest_tip_frame_bgr
                )
            if selected["mask"]:
                sample["mask_path"] = self._save_vla_image("masks", self.latest_segmentation_mask)
            if selected["raw_pose"]:
                sample["raw_pose"] = {
                    "position": self.scope_pos,
                    "rotation": self.scope_dir,
                    "reference_position": self.ref_pos,
                    "tools": self.latest_tools_dict,
                }
            if selected["model_pose"]:
                sample["model_pose"] = {
                    "position": getattr(self, "virtual_position", None),
                    "rotation": getattr(self, "mapped_tip_camera_dir", None),
                }
            if selected["axis"]:
                sample["axis_control"] = list(self.current_axis_values)
            if selected["action"]:
                sample["action"] = self.auto_nav_action
            if selected["path_state"]:
                sample["path_state"] = {
                    "planned": len(getattr(self, "smoothpath", [])) >= 2,
                    "point_count": len(getattr(self, "smoothpath", [])),
                    "target_index": self.auto_nav_target_index,
                    "progress": self.auto_nav_progress,
                }
            if selected["calibration"]:
                sample["calibration"] = {
                    "R": self.R, "t": self.t, "rmse_mm": self.calibration_rmse,
                    "max_error_mm": self.calibration_max_error, "residuals_mm": self.calibration_residuals,
                }
            sample = self._json_safe(sample)
            self.vla_jsonl_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            csv_row = {key: sample.get(key, "") for key in self.vla_csv_writer.fieldnames}
            for key, value in csv_row.items():
                if isinstance(value, (dict, list)):
                    csv_row[key] = json.dumps(value, ensure_ascii=False)
            self.vla_csv_writer.writerow(csv_row)
            self.vla_jsonl_file.flush()
            self.vla_csv_file.flush()
            self.vla_sample_index += 1
            self.vla_status_label.setText(f"正在采集: {self.vla_sample_index} 帧")
        except Exception as e:
            print(f"VLA 样本保存异常: {e}")
