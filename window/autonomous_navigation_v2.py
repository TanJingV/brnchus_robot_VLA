"""Adaptive receding-horizon navigation for a two-section continuum robot."""

import math
from dataclasses import dataclass

import numpy as np


def _unit(vector, fallback=None):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    if fallback is None:
        return np.zeros_like(vector)
    return _unit(fallback)


def _rate_limit(current, target, limit):
    delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm > limit:
        delta *= limit / norm
    return np.asarray(current, dtype=float) + delta


@dataclass
class NavigationConfig:
    horizon: int = 9
    proximal_limit: float = 0.450
    distal_limit: float = 0.350
    proximal_state_step: float = 0.025
    distal_state_step: float = 0.020
    reverse_confirm_frames: int = 60
    reverse_release_frames: int = 3
    stall_confirm_frames: int = 18
    unstick_reverse_frames: int = 18
    unstick_realign_frames: int = 8
    minimum_state_frames: int = 8
    maximum_state_frames: int = 36
    state_cost_improvement_ratio: float = 0.16
    emergency_cross_track_ratio: float = 2.4


class AdaptiveRecedingHorizonNavigator:
    """Path-frame MPC with online continuum response adaptation.

    Commands are absolute normalized torque states for proximal/distal
    sections plus an insertion increment in millimetres per control frame.
    """

    def __init__(self, config=None):
        self.config = config or NavigationConfig()
        self.reset()

    def reset(self, path=None, position=None, rotation=None):
        self.path = np.asarray(path, dtype=float) if path is not None else np.empty((0, 3))
        self.segment_lengths = np.empty(0)
        self.cumulative = np.empty(0)
        self.tangents = np.empty((0, 3))
        self.curvatures = np.empty(0)
        if self.path.ndim == 2 and len(self.path) >= 2:
            self._prepare_path()
        self.position_filtered = None if position is None else np.asarray(position, dtype=float).copy()
        self.forward_filtered = None
        self.rotation_filtered = None
        self.kinematic_rotation = None
        if rotation is not None:
            rotation = np.asarray(rotation, dtype=float)
            if rotation.shape == (3, 3):
                self.forward_filtered = _unit(rotation[:, 2], [0.0, 0.0, 1.0])
                self.rotation_filtered = rotation.copy()
                self.kinematic_rotation = rotation.copy()
        self.proximal = np.zeros(2)
        self.distal = np.zeros(2)
        self.feed = 0.0
        self.previous_position = None
        self.previous_rotation = None
        self.frame_motion_mm = 0.0
        self.frame_motion_filtered_mm = 0.0
        self.previous_command = np.zeros(5)
        self.previous_command_delta = np.zeros(5)
        # Identified from the project MuJoCo model: the distal section is
        # substantially more sensitive than the proximal section.
        self.bend_gain_proximal = 2.45
        self.bend_gain_distal = 4.10
        self.lateral_gain_proximal = 137.5
        self.lateral_gain_distal = 173.0
        self.feed_gain = 0.90
        self.response_confidence = 0.0
        self.motion_mode = "forward"
        self.stall_frames = 0
        self.unstick_frames = 0
        self.unstick_events = 0
        self.recovery_reason = ""
        self.overshoot_frames = 0
        self.release_frames = 0
        self.distance_worsening_frames = 0
        self.recovery_improvement_frames = 0
        self.previous_target_distance = None
        self.target_distance_window = []
        self.progress_window = []
        self.active_target_index = -1
        self.waypoint_frames = 0
        self.frame = 0
        self.action_state_age = 0
        self.action_state_changes = 0
        self.held_action_cost = float("inf")
        self.held_motion_mode = "forward"
        self.straight_deviation_frames = 0
        self.straight_clear_frames = 0
        self.macro_action_active = False
        self.macro_target_index = -1
        self.macro_feed_remaining_mm = 0.0
        self.macro_feed_total_mm = 0.0
        self.macro_action_frames = 0
        self.macro_action_id = 0
        self.macro_hold_frames_required = 1
        self.macro_action_max_frames = 12
        self.macro_action_phase = "idle"
        self.pending_feed_total_mm = 0.0
        self.pending_feed_rate_mm = 0.0
        self.active_segment = None

    def _prepare_path(self):
        differences = np.diff(self.path, axis=0)
        self.segment_lengths = np.maximum(np.linalg.norm(differences, axis=1), 1e-6)
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.segment_lengths)))
        raw_tangents = np.gradient(self.path, axis=0)
        self.tangents = np.asarray([_unit(value, [0.0, 0.0, 1.0]) for value in raw_tangents])
        smooth = self.tangents.copy()
        for index in range(len(smooth)):
            lo, hi = max(0, index - 2), min(len(smooth), index + 3)
            smooth[index] = _unit(np.mean(self.tangents[lo:hi], axis=0), self.tangents[index])
        self.tangents = smooth
        self.curvatures = np.zeros(len(self.path))
        for index in range(1, len(self.path) - 1):
            angle = math.acos(float(np.clip(
                np.dot(self.tangents[index - 1], self.tangents[index + 1]), -1.0, 1.0
            )))
            span = max(self.cumulative[index + 1] - self.cumulative[index - 1], 1e-6)
            self.curvatures[index] = angle / span
        if len(self.path) > 2:
            self.curvatures[0] = self.curvatures[1]
            self.curvatures[-1] = self.curvatures[-2]

    def _sample_path(self, arc_length):
        arc_length = float(np.clip(arc_length, 0.0, self.cumulative[-1]))
        segment = int(np.clip(
            np.searchsorted(self.cumulative, arc_length, side="right") - 1,
            0,
            len(self.path) - 2,
        ))
        alpha = float(np.clip(
            (arc_length - self.cumulative[segment]) / self.segment_lengths[segment], 0.0, 1.0
        ))
        point = (1.0 - alpha) * self.path[segment] + alpha * self.path[segment + 1]
        tangent = _unit(
            (1.0 - alpha) * self.tangents[segment] + alpha * self.tangents[segment + 1],
            self.tangents[segment],
        )
        curvature = float(
            (1.0 - alpha) * self.curvatures[segment] + alpha * self.curvatures[segment + 1]
        )
        return point, tangent, curvature, segment

    def _update_observer(self, position, rotation):
        if self.position_filtered is None:
            self.position_filtered = position.copy()
        self.position_filtered += 0.44 * (position - self.position_filtered)
        forward_raw = _unit(rotation[:, 2], [0.0, 0.0, 1.0])
        if self.forward_filtered is None:
            self.forward_filtered = forward_raw.copy()
        self.forward_filtered = _unit(
            0.70 * self.forward_filtered + 0.30 * forward_raw, forward_raw
        )
        if self.rotation_filtered is None:
            self.rotation_filtered = rotation.copy()
        blended_rotation = 0.82 * self.rotation_filtered + 0.18 * rotation
        u, _, vh = np.linalg.svd(blended_rotation)
        self.rotation_filtered = u @ vh
        if np.linalg.det(self.rotation_filtered) < 0.0:
            u[:, -1] *= -1.0
            self.rotation_filtered = u @ vh

        if self.previous_position is not None and self.previous_rotation is not None:
            movement_world = position - self.previous_position
            self.frame_motion_mm = float(np.linalg.norm(movement_world))
            self.frame_motion_filtered_mm += 0.25 * (
                self.frame_motion_mm - self.frame_motion_filtered_mm
            )
            previous_local = self.previous_rotation.T @ movement_world
            proximal_delta = self.previous_command_delta[:2]
            distal_delta = self.previous_command_delta[2:4]
            proximal_energy = float(np.dot(proximal_delta, proximal_delta))
            distal_energy = float(np.dot(distal_delta, distal_delta))
            if max(proximal_energy, distal_energy) > 0.000025:
                if proximal_energy >= distal_energy:
                    bend_command = proximal_delta
                    bend_norm_sq = proximal_energy
                    gain_name = "bend_gain_proximal"
                    gain_limits = (1.0, 6.0)
                else:
                    bend_command = distal_delta
                    bend_norm_sq = distal_energy
                    gain_name = "bend_gain_distal"
                    gain_limits = (1.0, 8.0)
                relative_forward = self.previous_rotation.T @ rotation[:, 2]
                observed_turn = float(
                    np.dot(relative_forward[:2], bend_command) / bend_norm_sq
                )
                observed_turn = float(np.clip(abs(observed_turn), *gain_limits))
                current_gain = getattr(self, gain_name)
                setattr(
                    self,
                    gain_name,
                    float(np.clip(
                        current_gain + 0.018 * (observed_turn - current_gain),
                        *gain_limits,
                    )),
                )
                self.response_confidence = min(1.0, self.response_confidence + 0.015)
            forward_response = float(previous_local[2])
            if self.previous_command[4] > 0.03 and forward_response > 0.0:
                observed_feed_gain = forward_response / self.previous_command[4]
                self.feed_gain += 0.02 * (
                    float(np.clip(observed_feed_gain, 0.20, 1.30)) - self.feed_gain
                )
        self.previous_position = position.copy()
        self.previous_rotation = rotation.copy()

    def _nearest_forward_progress(self, position, target_index):
        start = max(0, target_index - 3)
        end = min(len(self.path) - 1, target_index + 8)
        best_distance = float("inf")
        best_s = self.cumulative[max(0, target_index - 1)]
        best_point = self.path[max(0, target_index - 1)]
        for segment in range(start, end):
            delta = self.path[segment + 1] - self.path[segment]
            length_sq = float(np.dot(delta, delta))
            alpha = float(np.clip(
                np.dot(position - self.path[segment], delta) / max(length_sq, 1e-9),
                0.0,
                1.0,
            ))
            point = self.path[segment] + alpha * delta
            distance = float(np.linalg.norm(position - point))
            if distance < best_distance:
                best_distance = distance
                best_s = self.cumulative[segment] + alpha * self.segment_lengths[segment]
                best_point = point
        return best_s, best_distance, best_point

    def _candidate_commands(self, direction, magnitude, curvature_severity, max_step, reverse):
        if reverse:
            # Retraction candidates progressively unload cable torque so the
            # elastic sections can recenter while backing out of an overshoot.
            for release_scale in (1.0, 0.65, 0.30, 0.0):
                for feed_scale in (0.14, 0.24, 0.32):
                    yield (
                        self.proximal * release_scale,
                        self.distal * release_scale,
                        -max_step * feed_scale,
                    )
            return

        direction_candidates = []
        if np.linalg.norm(direction) > 0.5:
            for offset_deg in (-18.0, -8.0, 0.0, 8.0, 18.0):
                angle = math.radians(offset_deg)
                rotation = np.array([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ])
                direction_candidates.append(_unit(rotation @ direction, direction))
        current_bend = self.proximal + 0.65 * self.distal
        if np.linalg.norm(current_bend) > 0.03:
            direction_candidates.append(_unit(current_bend))
        if not direction_candidates:
            direction_candidates.append(np.zeros(2))

        proximal_share = float(np.clip(0.82 - 0.22 * curvature_severity, 0.56, 0.82))
        for candidate_direction in direction_candidates:
            proximal_nominal = candidate_direction * min(
                self.config.proximal_limit, magnitude * proximal_share
            )
            residual = max(0.0, magnitude - float(np.linalg.norm(proximal_nominal)))
            distal_nominal = candidate_direction * min(
                self.config.distal_limit,
                residual / 0.65 + curvature_severity * 0.16,
            )
            for bend_scale in (0.76, 0.92, 1.06):
                proximal = proximal_nominal * bend_scale
                distal = distal_nominal * bend_scale
                pnorm, dnorm = float(np.linalg.norm(proximal)), float(np.linalg.norm(distal))
                if pnorm > self.config.proximal_limit:
                    proximal *= self.config.proximal_limit / pnorm
                if dnorm > self.config.distal_limit:
                    distal *= self.config.distal_limit / dnorm
                for feed_scale in (0.16, 0.34, 0.56, 0.82):
                    yield proximal, distal, max_step * feed_scale

    def _rollout_cost(
        self,
        proximal,
        distal,
        feed,
        position,
        forward,
        rotation,
        progress_s,
        target,
        target_tangent,
        tolerance,
        target_index,
    ):
        simulated_position = position.copy()
        simulated_forward = forward.copy()
        command_bend = proximal + 0.65 * distal
        action_change = (
            np.linalg.norm(proximal - self.proximal) ** 2
            + 0.8 * np.linalg.norm(distal - self.distal) ** 2
            + 0.08 * (feed - self.feed) ** 2
        )
        cost = 6.0 * action_change
        simulated_s = progress_s
        local_bend_world = rotation[:, :2] @ command_bend
        for step in range(1, self.config.horizon + 1):
            response = self.bend_gain_proximal * proximal + self.bend_gain_distal * distal
            bend_world = rotation[:, :2] @ response
            simulated_forward = _unit(
                simulated_forward + bend_world * (0.75 + 0.10 * step),
                simulated_forward,
            )
            simulated_position += simulated_forward * feed * self.feed_gain
            simulated_s = min(self.cumulative[-1], simulated_s + max(feed, 0.0) * self.feed_gain)
            reference, tangent, curvature, _ = self._sample_path(simulated_s)
            cross_track = float(np.linalg.norm(simulated_position - reference))
            tangent_error = 1.0 - float(np.clip(np.dot(simulated_forward, tangent), -1.0, 1.0))
            target_distance = float(np.linalg.norm(simulated_position - target))
            visual_error = 1.0 - float(np.clip(
                np.dot(simulated_forward, _unit(target - simulated_position, target_tangent)),
                -1.0,
                1.0,
            ))
            weight = 0.6 + 0.4 * step / self.config.horizon
            cost += weight * (
                2.8 * (cross_track / max(tolerance, 1e-6)) ** 2
                + 3.2 * tangent_error
                + 1.6 * visual_error
                + 0.06 * target_distance
                + 0.30 * curvature * float(np.dot(local_bend_world, local_bend_world))
            )
        # Reward forward progress but strongly penalize skipping the active gate.
        cost -= 0.55 * max(simulated_s - progress_s, 0.0)
        target_plane = float(np.dot(simulated_position - target, target_tangent))
        target_cross = float(np.linalg.norm(
            (simulated_position - target) - target_plane * target_tangent
        ))
        if target_plane > 0.0 and target_cross > tolerance * 1.4:
            cost += 30.0 + 4.0 * target_cross
        if target_index == len(self.path) - 1:
            final_distance = float(np.linalg.norm(simulated_position - target))
            cost += 2.4 * final_distance + 0.18 * final_distance ** 2
        return float(cost)

    def compute(
        self,
        position,
        rotation,
        target_index,
        point_threshold,
        centerline_tolerance,
        max_step,
        speed_scale=1.0,
    ):
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if (
            self.path.ndim != 2 or len(self.path) < 2
            or position.shape != (3,) or rotation.shape != (3, 3)
        ):
            return None
        self.frame += 1
        self._update_observer(position, rotation)
        filtered_position = self.position_filtered.copy()
        forward = self.forward_filtered.copy()
        if self.kinematic_rotation is None:
            self.kinematic_rotation = self.rotation_filtered.copy()
        # Slider insertion is fixed to the base axis in the revised MuJoCo
        # model. Do not rotate this axis with the bending tip camera.
        feed_axis = _unit(self.kinematic_rotation[:, 2])
        # Tendon bending axes rotate with the continuum tip, while slider
        # insertion remains fixed to the base axis. This hybrid frame keeps the
        # calibrated lateral Jacobian valid after entering a sharp branch.
        lateral_x = _unit(self.rotation_filtered[:, 0], self.kinematic_rotation[:, 0])
        lateral_y = _unit(self.rotation_filtered[:, 1], self.kinematic_rotation[:, 1])
        control_rotation = np.column_stack((lateral_x, lateral_y, feed_axis))
        target_index = int(np.clip(target_index, 1, len(self.path) - 1))
        target_changed = target_index != self.active_target_index
        if target_changed:
            self.active_target_index = target_index
            self.previous_target_distance = None
            self.target_distance_window = []
            self.progress_window = []
            self.distance_worsening_frames = 0
            self.overshoot_frames = 0
            self.waypoint_frames = 0
            self.macro_action_active = False
            self.macro_target_index = target_index
            self.macro_feed_remaining_mm = 0.0
            self.pending_feed_total_mm = 0.0
            self.pending_feed_rate_mm = 0.0
        target = self.path[target_index]
        target_tangent = self.tangents[target_index]
        target_delta = target - filtered_position
        target_distance = float(np.linalg.norm(target_delta))
        distance_change = (
            0.0
            if self.previous_target_distance is None
            else target_distance - self.previous_target_distance
        )
        self.previous_target_distance = target_distance
        self.target_distance_window.append(target_distance)
        if len(self.target_distance_window) > 20:
            self.target_distance_window.pop(0)
        stagnation_window_ready = len(self.target_distance_window) >= 20
        window_improvement = (
            self.target_distance_window[0] - min(self.target_distance_window)
            if stagnation_window_ready else float("inf")
        )
        target_local = control_rotation.T @ target_delta
        target_direction = _unit(target_delta, target_tangent)
        progress_s, cross_track, nearest_point = self._nearest_forward_progress(
            filtered_position, target_index
        )
        target_s = float(self.cumulative[target_index])
        target_arc_gap = max(0.0, target_s - float(progress_s))
        self.progress_window.append(float(progress_s))
        if len(self.progress_window) > 20:
            self.progress_window.pop(0)
        progress_window_ready = len(self.progress_window) >= 20
        progress_window_gain = (
            max(self.progress_window) - self.progress_window[0]
            if progress_window_ready else float("inf")
        )

        # Detect a constrained tip only when a meaningful forward command has
        # repeatedly produced almost no physical motion or target improvement.
        # This avoids confusing deliberate low-speed cornering with a wall hit.
        commanded_forward = bool(
            self.motion_mode == "forward"
            and self.previous_command[4] > max(max_step * speed_scale * 0.16, 0.08)
        )
        expected_motion = abs(float(self.previous_command[4])) * self.feed_gain
        motion_blocked = bool(
            self.frame_motion_filtered_mm < max(0.025, expected_motion * 0.10)
            and distance_change > -0.015
            and target_distance > point_threshold * 1.8
        )
        low_feed_deadlock = bool(
            self.motion_mode == "forward"
            and abs(float(self.previous_command[4])) <= max(max_step * 0.03, 0.08)
            and self.frame_motion_filtered_mm < 0.025
            and distance_change > -0.015
            and target_distance > point_threshold * 1.8
            and self.action_state_age >= self.config.minimum_state_frames
            and (
                target_distance > centerline_tolerance * 1.35
                or cross_track > centerline_tolerance
            )
        )
        progress_deadlock = bool(
            self.motion_mode == "forward"
            and progress_window_ready
            and progress_window_gain < 0.30
            and target_arc_gap > max(point_threshold * 0.45, 0.75)
            and target_distance > point_threshold
        )
        if (
            (commanded_forward and motion_blocked)
            or low_feed_deadlock
            or progress_deadlock
        ):
            self.stall_frames += 1
        elif self.motion_mode == "forward":
            self.stall_frames = max(0, self.stall_frames - 2)

        if (
            self.motion_mode == "forward"
            and self.stall_frames >= self.config.stall_confirm_frames
        ):
            # The model now uses a collision-free, bend-limited splice. Keep
            # following the route with bounded positive insertion instead of
            # introducing a reverse unstick action at a bifurcation.
            self.stall_frames = self.config.stall_confirm_frames
            self.recovery_reason = "motion_constrained_no_reverse"

        if self.motion_mode == "unstick_reverse":
            self.unstick_frames += 1
            if self.unstick_frames >= self.config.unstick_reverse_frames:
                self.motion_mode = "unstick_realign"
                self.unstick_frames = 0
                self.macro_action_active = False
        elif self.motion_mode == "unstick_realign":
            self.unstick_frames += 1
            if self.unstick_frames >= self.config.unstick_realign_frames:
                self.motion_mode = "forward"
                self.unstick_frames = 0
                self.stall_frames = 0
                self.recovery_reason = ""
                self.macro_action_active = False

        raw_plane = float(np.dot(position - target, target_tangent))
        target_cross = float(np.linalg.norm(
            (position - target) - raw_plane * target_tangent
        ))
        target_curve_hint = float(np.clip(
            self.curvatures[target_index] * 18.0, 0.0, 1.0
        ))
        crossing_tolerance = (
            max(point_threshold, centerline_tolerance * 0.45)
            if target_curve_hint >= 0.10 else
            max(centerline_tolerance, point_threshold * 1.5)
        )
        crossed_in_corridor = bool(
            raw_plane >= 0.0
            and target_cross <= crossing_tolerance
        )
        # A waypoint may only advance after the tip physically reaches its
        # path station.  Distance stagnation or actuator saturation must never
        # move the yellow target ahead while the robot remains at a junction.
        progress_gate_tolerance = max(0.60, point_threshold * 0.20)
        progress_gated_capture = bool(
            progress_s >= target_s - progress_gate_tolerance
            and target_distance <= centerline_tolerance
            and target_cross <= centerline_tolerance
        )
        final_waypoint = target_index == len(self.path) - 1
        # The compliant passive follower changes the static camera endpoint by
        # a few millimetres. Use the configured centerline corridor as the
        # physical final capture sphere; completion still requires the real tip
        # position and can never be inferred from a stalled waypoint index.
        final_capture_radius = max(point_threshold, centerline_tolerance)
        final_target_capture = bool(
            final_waypoint and target_distance <= final_capture_radius
        )
        if final_waypoint:
            crossed_in_corridor = False
            progress_gated_capture = False
        if (
            target_distance <= point_threshold
            or crossed_in_corridor
            or progress_gated_capture
            or final_target_capture
        ):
            self.waypoint_frames += 1
        else:
            self.waypoint_frames = 0
        # A flexible continuum tip may pass a dense waypoint slightly outside
        # its small capture sphere. Crossing that waypoint's normal plane while
        # still inside the centerline corridor is also a valid sequential pass;
        # otherwise the controller becomes locked on a point already behind it.
        waypoint_reached = bool(
            target_distance <= point_threshold
            or crossed_in_corridor
            or progress_gated_capture
            or final_target_capture
        )

        true_overshoot = bool(
            raw_plane > max(point_threshold * 5.0, 10.0)
            and target_cross > max(centerline_tolerance * 3.0, point_threshold * 4.0)
            and not crossed_in_corridor
        )
        if true_overshoot and distance_change > 0.025:
            self.distance_worsening_frames += 1
        elif distance_change < -0.02:
            self.distance_worsening_frames = max(0, self.distance_worsening_frames - 2)
        else:
            self.distance_worsening_frames = max(0, self.distance_worsening_frames - 1)
        confirmed_overshoot = bool(
            true_overshoot
            and (
                self.distance_worsening_frames >= 3
                or raw_plane > max(point_threshold * 3.0, 3.0)
            )
        )
        self.overshoot_frames = self.overshoot_frames + 1 if confirmed_overshoot else max(
            0, self.overshoot_frames - 1
        )
        freeze_motion_mode = self.macro_action_active
        if (
            not freeze_motion_mode
            and self.motion_mode == "forward"
            and self.overshoot_frames >= self.config.reverse_confirm_frames
        ):
            self.motion_mode = "reverse"
            self.release_frames = 0
        if self.motion_mode == "reverse" and not freeze_motion_mode:
            recovery_improving = bool(distance_change < -0.025)
            self.recovery_improvement_frames = (
                self.recovery_improvement_frames + 1
                if recovery_improving else max(0, self.recovery_improvement_frames - 1)
            )
            self.release_frames = self.release_frames + 1 if raw_plane <= 0.0 else 0
            if (
                self.release_frames >= self.config.reverse_release_frames
                or self.recovery_improvement_frames >= 3
            ):
                self.motion_mode = "forward"
                self.overshoot_frames = 0
                self.recovery_improvement_frames = 0
        reversing = self.motion_mode in ("reverse", "unstick_reverse")
        realigning = self.motion_mode == "unstick_realign"

        preview_distance = 14.0
        preview_origin_s = max(
            float(progress_s), float(self.cumulative[target_index])
        )
        preview_s = min(
            self.cumulative[-1],
            preview_origin_s + preview_distance,
        )
        preview_point, preview_tangent, preview_curvature, _ = self._sample_path(preview_s)
        current_path_point, current_path_tangent, current_path_curvature, _ = self._sample_path(
            min(float(progress_s), self.cumulative[-1])
        )
        preview_turn_angle = math.acos(float(np.clip(
            np.dot(current_path_tangent, preview_tangent), -1.0, 1.0
        )))
        preview_turn_severity = float(np.clip(
            preview_turn_angle / math.radians(70.0), 0.0, 1.0
        ))
        guidance_direction = _unit(preview_point - filtered_position, target_direction)
        route_direction = _unit(
            target_direction * (0.55 - 0.25 * preview_turn_severity)
            + guidance_direction * (0.30 + 0.45 * preview_turn_severity)
            + preview_tangent * 0.15,
            target_direction,
        )
        route_local = control_rotation.T @ route_direction
        lateral = np.asarray(route_local[:2], dtype=float)
        lateral_norm = float(np.linalg.norm(lateral))
        direction = lateral / lateral_norm if lateral_norm > 1e-7 else np.zeros(2)
        curvature_local = control_rotation.T @ (preview_tangent - target_tangent)
        curvature_direction = _unit(curvature_local[:2])
        curvature_severity = max(
            float(np.clip(max(preview_curvature, current_path_curvature) * 18.0, 0.0, 1.0)),
            preview_turn_severity,
        )
        if np.linalg.norm(curvature_direction) > 0.5:
            direction = _unit(
                direction
                + curvature_direction
                * (0.25 + 0.55 * curvature_severity),
                curvature_direction,
            )
        bearing_angle = float(math.atan2(lateral_norm, max(float(route_local[2]), -0.12)))
        bearing_ratio = float(np.clip(bearing_angle / math.radians(100.0), 0.0, 1.0))
        cross_ratio = float(np.clip(
            cross_track / max(centerline_tolerance * 3.0, 1e-9), 0.0, 1.0
        ))
        straight_route = bool(curvature_severity < 0.08)
        straight_mild_error = bool(
            straight_route
            and cross_track < centerline_tolerance * 1.30
        )
        straight_large_error = bool(
            straight_route
            and (
                (
                    bearing_angle > math.radians(20.0)
                    and cross_track > centerline_tolerance * 1.65
                )
                or cross_track > centerline_tolerance * 2.50
            )
        )
        if straight_mild_error:
            self.straight_clear_frames += 1
            self.straight_deviation_frames = max(0, self.straight_deviation_frames - 2)
        elif straight_large_error:
            self.straight_deviation_frames += 1
            self.straight_clear_frames = 0
        else:
            self.straight_deviation_frames = max(0, self.straight_deviation_frames - 1)
            self.straight_clear_frames = max(0, self.straight_clear_frames - 1)
        # Invert the identified torque-to-heading response. This produces an
        # absolute low-amplitude torque state instead of repeatedly adding a
        # large correction for every waypoint.
        desired_turn_angle = (
            bearing_angle + math.radians(24.0) * curvature_severity
        )
        bend_demand = float(np.clip(
            desired_turn_angle / 3.0 + 0.025 * cross_ratio,
            0.0,
            0.300,
        ))
        straight_correction_ready = bool(
            straight_route and self.straight_deviation_frames >= 16
        )
        straight_release_ready = bool(
            straight_route and self.straight_clear_frames >= 12
        )
        if straight_route and not straight_correction_ready:
            bend_demand = 0.0
        elif straight_correction_ready:
            bend_demand = min(bend_demand, 0.025)

        effective_max_step = float(max_step * speed_scale)
        fixed_state_speed_scale = float(np.clip(
            (1.0 - 0.72 * bearing_ratio) * (1.0 - 0.58 * curvature_severity),
            0.14,
            1.0,
        ))
        candidate_max_step = effective_max_step * fixed_state_speed_scale
        if target_distance < max(point_threshold * 2.5, 5.0):
            candidate_max_step = min(
                candidate_max_step,
                max(effective_max_step * 0.10, target_distance * 0.12),
            )
        if target_index == len(self.path) - 1 and not reversing:
            candidate_max_step = max(
                effective_max_step * 0.08,
                min(effective_max_step, target_distance * 0.24),
            )
        # Identified displacement inverse model. A fixed cable-torque state
        # establishes a fixed continuum shape; slider insertion then advances
        # that shape. Solve the cable state from required lateral tip
        # displacement and solve total insertion from forward displacement.
        guidance_local = control_rotation.T @ (preview_point - filtered_position)
        preview_blend = float(np.clip(
            0.20 + 0.70 * preview_turn_severity, 0.20, 0.90
        ))
        active_lateral_error = np.asarray(target_local[:2], dtype=float)
        preview_offset = np.asarray(guidance_local[:2], dtype=float) - active_lateral_error
        preview_offset_norm = float(np.linalg.norm(preview_offset))
        maximum_preview_offset = max(centerline_tolerance * 0.50, 1.5)
        if preview_offset_norm > maximum_preview_offset:
            preview_offset *= maximum_preview_offset / preview_offset_norm
        target_lateral_vector = active_lateral_error + preview_blend * preview_offset
        target_lateral_distance = float(np.linalg.norm(target_lateral_vector))
        target_forward_distance = float(target_local[2])
        macro_target_behind = bool(
            target_forward_distance < -max(point_threshold * 5.0, 8.0)
            and raw_plane > max(point_threshold * 6.0, 12.0)
            and target_distance > point_threshold
        )
        requested_active_segment = None
        alignment_only = False
        if reversing or macro_target_behind:
            if self.motion_mode == "unstick_reverse":
                # Preserve most of the current curvature while the slider
                # retreats. An abrupt straightening impulse can drive the tip
                # harder into the opposite wall and masquerade as translation.
                proximal_target = self.proximal * 0.75
                distal_target = self.distal * 0.75
            else:
                proximal_target = np.zeros(2, dtype=float)
                distal_target = np.zeros(2, dtype=float)
            feed_target = -max(candidate_max_step * 0.22, 0.10)
            planned_feed_total = max(
                abs(target_forward_distance) + point_threshold * 0.20,
                abs(feed_target) * 3.0,
            )
            solved_bend_angle = 0.0
        else:
            planned_feed_total = max(
                target_forward_distance,
                target_arc_gap,
                point_threshold * 0.45,
            )
            planned_feed_total = float(np.clip(
                planned_feed_total, point_threshold * 0.45, target_distance * 1.20
            ))
            lateral_deadband = max(point_threshold * 0.20, 0.25)
            current_command = np.concatenate((self.proximal, self.distal))
            correction_required = bool(
                target_lateral_distance > lateral_deadband
                or np.linalg.norm(current_command) > 0.004
                or curvature_severity >= 0.08
            )
            if not correction_required:
                proximal_target = np.zeros(2, dtype=float)
                distal_target = np.zeros(2, dtype=float)
                solved_bend_angle = 0.0
                planned_feed_total = max(target_forward_distance, point_threshold * 0.45)
            else:
                # Damped least-squares inverse of the measured revised-model
                # position Jacobian (millimetres per normalized compass unit).
                # Both manual and autonomous modes may coordinate the two
                # sections; this inverse distributes the requested correction.
                jacobian = np.array(
                    [
                        [20.2505, -36.5600, 5.9868, -35.0840],
                        [-0.3007, 52.6101, -0.2247, 40.8438],
                    ],
                    dtype=float,
                )
                estimated_displacement = jacobian @ current_command
                desired_displacement = estimated_displacement + target_lateral_vector
                damping = 16.0
                solved_command = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping * np.eye(2),
                    desired_displacement,
                )
                proximal_target = solved_command[:2]
                distal_target = solved_command[2:]
                proximal_norm = float(np.linalg.norm(proximal_target))
                distal_norm = float(np.linalg.norm(distal_target))
                if proximal_norm > self.config.proximal_limit:
                    proximal_target *= self.config.proximal_limit / proximal_norm
                if distal_norm > self.config.distal_limit:
                    distal_target *= self.config.distal_limit / distal_norm
                requested_active_segment = 2
                solved_bend_angle = (
                    np.linalg.norm(proximal_target) * self.bend_gain_proximal
                    + np.linalg.norm(distal_target) * self.bend_gain_distal
                )
            feed_target = float(np.clip(
                target_forward_distance * 0.12,
                -candidate_max_step * 0.18,
                candidate_max_step,
            ))
            lateral_slowdown = float(np.clip(
                np.linalg.norm(target_lateral_vector)
                / max(centerline_tolerance * 3.0, 1.0),
                0.0,
                1.0,
            ))
            feed_target *= float(np.clip(
                1.0
                - 0.72 * preview_turn_severity
                - 0.30 * lateral_slowdown,
                0.12,
                1.0,
            ))
            path_station_ahead = bool(
                target_arc_gap > progress_gate_tolerance
                and target_distance > point_threshold
            )
            safe_creep_corridor = bool(
                cross_track <= centerline_tolerance * 1.35
                and target_cross <= centerline_tolerance * 1.50
            )
            progress_feed_floor = max(
                0.08,
                effective_max_step
                * (0.05 + 0.05 * (1.0 - max(curvature_severity, bearing_ratio))),
            )
            if path_station_ahead and safe_creep_corridor:
                # In a bent branch the fixed slider axis can place the target
                # behind the camera's local Z even though it is still ahead on
                # the route. Keep the established curve and continue a gentle
                # insertion instead of settling at zero feed.
                feed_target = max(feed_target, progress_feed_floor)
            if realigning:
                # After retreat, rebuild most of the target curvature before
                # creeping forward. Re-entering straight repeats the same
                # geometric wedge at a bifurcation.
                proximal_target *= 0.75
                distal_target *= 0.75
                requested_active_segment = 2
                alignment_only = False
                feed_target = float(np.clip(
                    feed_target,
                    max(candidate_max_step * 0.02, 0.04),
                    max(candidate_max_step * 0.05, 0.08),
                ))
        best_cost = 0.0
        mode_changed = self.motion_mode != self.held_motion_mode
        start_macro_action = bool(
            not self.macro_action_active
            or target_changed
            or mode_changed
        )
        if start_macro_action:
            # One macro action contains one definite proximal bend, distal
            # bend, and insertion rate. Bending and insertion begin together;
            # a zero-feed settling phase caused elastic shaking without tip
            # progress in the real MuJoCo update loop.
            planned_proximal = np.round(
                np.asarray(proximal_target, dtype=float) / 0.005
            ) * 0.005
            planned_distal = np.round(
                np.asarray(distal_target, dtype=float) / 0.005
            ) * 0.005
            planned_proximal = _rate_limit(
                self.proximal,
                planned_proximal,
                self.config.proximal_state_step,
            )
            planned_distal = _rate_limit(
                self.distal,
                planned_distal,
                self.config.distal_state_step,
            )
            if requested_active_segment == 2:
                pass
            elif requested_active_segment == 0:
                planned_distal[:] = 0.0
            elif requested_active_segment == 1:
                planned_proximal[:] = 0.0
            else:
                planned_proximal[:] = 0.0
                planned_distal[:] = 0.0
            planned_feed = round(float(feed_target) / 0.02) * 0.02
            self.proximal = planned_proximal
            self.distal = planned_distal
            self.active_segment = requested_active_segment
            if reversing or macro_target_behind:
                self.feed = min(planned_feed, -candidate_max_step * 0.12)
                self.macro_action_phase = (
                    "unstick_reverse"
                    if self.motion_mode == "unstick_reverse"
                    else "execute_reverse"
                )
            else:
                turn_slowdown = max(curvature_severity, bearing_ratio)
                forward_feed_floor = candidate_max_step * (
                    0.08 + 0.17 * (1.0 - turn_slowdown)
                )
                if alignment_only:
                    # A continuum tip changes both position and heading while
                    # bending. Keep a small creep feed during alignment so the
                    # coupled pose error can converge instead of settling into
                    # a zero-insertion geometric deadlock.
                    self.feed = max(
                        effective_max_step * 0.10,
                        min(0.08, planned_feed_total),
                    )
                    self.macro_action_phase = (
                        "align_proximal"
                        if self.active_segment == 0 else "align_distal"
                    )
                elif self.active_segment == 2:
                    self.feed = planned_feed
                    self.macro_action_phase = "coordinated_inverse_kinematics"
                else:
                    self.feed = max(
                        planned_feed,
                        forward_feed_floor,
                        min(0.10, planned_feed_total),
                    )
                    self.macro_action_phase = (
                        "unstick_realign" if realigning else "execute_feed"
                    )
            self.macro_hold_frames_required = 1
            self.pending_feed_total_mm = 0.0
            self.pending_feed_rate_mm = 0.0
            self.action_state_age = 1
            self.action_state_changes += 1
            self.held_action_cost = best_cost
            self.held_motion_mode = self.motion_mode
            self.macro_action_active = True
            self.macro_target_index = target_index
            self.macro_action_frames = 0
            self.macro_action_id += 1
            self.macro_feed_total_mm = planned_feed_total
            self.macro_feed_remaining_mm = self.macro_feed_total_mm
            expected_frames = planned_feed_total / max(
                abs(self.feed) * self.feed_gain, 1e-6
            )
            self.macro_action_max_frames = int(np.clip(
                math.ceil(expected_frames * 0.75),
                5 if curvature_severity >= 0.10 else 8,
                14 if curvature_severity >= 0.10 else 24,
            ))
        else:
            self.action_state_age += 1
        self.macro_action_frames += 1
        self.macro_feed_remaining_mm = max(
            0.0,
            self.macro_feed_remaining_mm - abs(self.feed) * self.feed_gain,
        )
        # Keep one physical state until the waypoint is actually reached.
        # Between state transitions the action is constant. A bounded state
        # duration lets the next action correct real elastic/gravity drift
        # before the tip passes a closely spaced waypoint.
        macro_action_completed = bool(
            waypoint_reached or self.macro_action_frames >= self.macro_action_max_frames
        )
        if macro_action_completed:
            self.macro_action_active = False
        new_command = np.concatenate((self.proximal, self.distal, [self.feed]))
        self.previous_command_delta = new_command - self.previous_command
        self.previous_command = new_command

        all_path_local = (control_rotation.T @ (self.path - filtered_position).T).T
        visual_error = np.asarray(target_local[:2], dtype=float) / max(abs(float(target_local[2])), 1.0)
        heading_error = math.acos(float(np.clip(np.dot(forward, route_direction), -1.0, 1.0)))
        servo_phase = (
            "MPC越界恢复，保持曲率后退"
            if reversing else (
                (
                    "固定大弯曲状态保持"
                    if curvature_severity > 0.38 or bearing_angle > math.radians(35.0)
                    else (
                        "直线固定状态保持"
                        if straight_route else "固定导航状态保持"
                    )
                )
                if not start_macro_action else "执行新的路径点宏动作"
            )
        )
        return {
            "steering": self.distal.tolist(),
            "steering_proximal": self.proximal.tolist(),
            "steering_distal": self.distal.tolist(),
            "active_segment": self.active_segment,
            "coordinated_sections": self.active_segment == 2,
            "insertion_delta": float(self.feed / 577.0),
            "insertion_step_mm": float(self.feed),
            "target_index": target_index,
            "nearest_index": target_index,
            "target_model_mm": target.tolist(),
            "all_path_local_mm": all_path_local.tolist(),
            "distance_to_target_mm": target_distance,
            "waypoint_reached": waypoint_reached,
            "waypoint_hold_frames": self.waypoint_frames,
            "waypoint_local_mm": target_local.tolist(),
            "target_local_mm": target_local.tolist(),
            "guidance_target_model_mm": preview_point.tolist(),
            "guidance_target_distance_mm": float(np.linalg.norm(preview_point - filtered_position)),
            "relative_position_error_mm": target_local.tolist(),
            "relative_bearing_error_deg": math.degrees(bearing_angle),
            "visual_center_error": visual_error.tolist(),
            "visual_error_norm": float(np.linalg.norm(visual_error)),
            "visual_center_speed": float(np.clip(1.0 - bearing_ratio, 0.0, 1.0)),
            "lateral_distance_mm": float(np.linalg.norm(target_local[:2])),
            "forward_error_mm": float(target_local[2]),
            "azimuth_deg": math.degrees(float(math.atan2(target_local[0], max(target_local[2], 1e-6)))),
            "elevation_deg": math.degrees(float(math.atan2(target_local[1], max(target_local[2], 1e-6)))),
            "heading_error_deg": math.degrees(heading_error),
            "servo_phase": servo_phase,
            "estimated_speed_mm_per_frame": self.frame_motion_filtered_mm,
            "straight_segment": curvature_severity < 0.10,
            "straight_path_mode": straight_route,
            "turn_active": curvature_severity >= 0.10 or bearing_angle >= math.radians(10.0),
            "turn_severity": max(curvature_severity, bearing_ratio),
            "curve_severity": curvature_severity,
            "curvature_throttle": float(np.clip(1.0 - 0.72 * curvature_severity, 0.2, 1.0)),
            "distal_assist": float(np.linalg.norm(self.distal) / self.config.distal_limit),
            "proximal_utilization": float(np.linalg.norm(self.proximal) / self.config.proximal_limit),
            "cruise_boost": 1.0,
            "recovery_mode": reversing or realigning,
            "retracting": reversing,
            "motion_mode": self.motion_mode,
            "special_case": reversing or realigning,
            "anti_curl_release": False,
            "cross_track_error_mm": cross_track,
            "within_centerline_tolerance": cross_track <= centerline_tolerance,
            "global_curve_angle_deg": math.degrees(preview_curvature * 10.0),
            "global_preview_angle_deg": math.degrees(preview_curvature * 18.0),
            "path_preview_turn_deg": math.degrees(preview_turn_angle),
            "forward_alignment": float(np.dot(forward, route_direction)),
            "overshot_waypoint": true_overshoot,
            "confirmed_overshoot": confirmed_overshoot,
            "target_distance_change_mm": distance_change,
            "raw_target_plane_progress_mm": raw_plane,
            "mpc_cost": best_cost,
            "mpc_horizon": self.config.horizon,
            "adaptive_proximal_gain": self.bend_gain_proximal,
            "adaptive_distal_gain": self.bend_gain_distal,
            "adaptive_feed_gain": self.feed_gain,
            "response_confidence": self.response_confidence,
            "target_transition": target_changed,
            "continuous_action": False,
            "action_state_held": not start_macro_action,
            "action_state_age_frames": self.action_state_age,
            "action_state_changes": self.action_state_changes,
            "action_state_cost": self.held_action_cost,
            "action_state_candidate_improvement": 0.0,
            "straight_deviation_frames": self.straight_deviation_frames,
            "straight_clear_frames": self.straight_clear_frames,
            "straight_correction_ready": straight_correction_ready,
            "macro_action_id": self.macro_action_id,
            "macro_action_active": self.macro_action_active,
            "macro_action_completed": macro_action_completed,
            "macro_action_frames": self.macro_action_frames,
            "macro_action_phase": self.macro_action_phase,
            "macro_hold_frames_required": self.macro_hold_frames_required,
            "macro_action_max_frames": self.macro_action_max_frames,
            "macro_feed_total_mm": self.macro_feed_total_mm,
            "macro_feed_remaining_mm": self.macro_feed_remaining_mm,
            "macro_bend_angle_deg": math.degrees(solved_bend_angle),
            "macro_target_behind": macro_target_behind,
            "pending_feed_total_mm": self.pending_feed_total_mm,
            "motion_constrained": self.stall_frames >= self.config.stall_confirm_frames,
            "stagnant_corridor_capture": False,
            "progress_gated_capture": progress_gated_capture,
            "final_target_capture": final_target_capture,
            "final_capture_radius_mm": float(final_capture_radius),
            "path_progress_mm": float(progress_s),
            "target_path_station_mm": target_s,
            "target_arc_gap_mm": target_arc_gap,
            "progress_window_gain_mm": float(progress_window_gain),
            "target_window_improvement_mm": float(window_improvement),
            "stall_frames": self.stall_frames,
            "unstick_events": self.unstick_events,
            "recovery_reason": self.recovery_reason,
        }
