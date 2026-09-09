#!/usr/bin/env python3
"""Solve and validate two-section TDCR inverse kinematics on a tip trajectory.

The seven output controls are the absolute target lengths of Wire 1--6 in
metres followed by the insertion-axis target in metres. Input trajectory
coordinates are expressed in the initial ``base_frame`` coordinates; a
straight continuum tip, including its rigid connectors, is therefore at
approximately ``[0.054, 0, 0]`` metres.

The analytical constant-curvature IK is corrected by repeatable MuJoCo
pre-trials (global affine model-error inversion), then tracked by a bounded
damped-Jacobian Cartesian outer loop around the existing fast tendon PID.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
from scipy.optimize import least_squares

from tendon_compass_control import TendonCompassController


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE.parent / "meshes" / "cable_robot_bronch_final_seg2.xml"
SECTION_LENGTH_M = 0.021
MAX_BEND_RAD = math.radians(160.0)
CONTROL_NAMES = [*(f"wire_{wire}_length_m" for wire in range(1, 7)), "insertion_m"]


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def section_transform(bend_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Constant-curvature transform for a section whose straight axis is +X."""
    bend_y, bend_z = np.asarray(bend_vector, dtype=float)
    theta = float(math.hypot(bend_y, bend_z))
    if theta < 1e-9:
        return np.asarray([SECTION_LENGTH_M, 0.0, 0.0]), np.eye(3)

    cos_phi = bend_y / theta
    sin_phi = bend_z / theta
    position = np.asarray(
        [
            SECTION_LENGTH_M * math.sin(theta) / theta,
            SECTION_LENGTH_M * (1.0 - math.cos(theta)) / theta * cos_phi,
            SECTION_LENGTH_M * (1.0 - math.cos(theta)) / theta * sin_phi,
        ]
    )
    axis = np.asarray([0.0, -sin_phi, cos_phi])
    axis_skew = skew(axis)
    rotation = (
        np.eye(3)
        + math.sin(theta) * axis_skew
        + (1.0 - math.cos(theta)) * (axis_skew @ axis_skew)
    )
    return position, rotation


def forward_kinematics(ik_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return local tip position and tangent for [insertion, b1_yz, b2_yz]."""
    insertion = float(ik_state[0])
    position_1, rotation_1 = section_transform(ik_state[1:3])
    position_2, rotation_2 = section_transform(ik_state[3:5])
    tip_position = (
        np.asarray([insertion, 0.0, 0.0])
        + position_1
        + rotation_1 @ position_2
    )
    tip_rotation = rotation_1 @ rotation_2
    return tip_position, tip_rotation[:, 0]


def position_jacobian(ik_state: np.ndarray) -> np.ndarray:
    """Numerical 3-by-5 tip-position Jacobian of the analytical TDCR model."""
    state = np.asarray(ik_state, dtype=float)
    jacobian = np.empty((3, 5), dtype=float)
    steps = np.asarray([1e-5, 1e-4, 1e-4, 1e-4, 1e-4])
    for axis, step in enumerate(steps):
        upper = state.copy()
        lower = state.copy()
        upper[axis] += step
        lower[axis] -= step
        jacobian[:, axis] = (
            forward_kinematics(upper)[0] - forward_kinematics(lower)[0]
        ) / (2.0 * step)
    return jacobian


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    fallback = np.zeros_like(vectors)
    fallback[:, 0] = 1.0
    return np.where(norms > 1e-9, vectors / np.maximum(norms, 1e-9), fallback)


def demo_trajectory(points: int, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a smooth reachable 3-D trajectory from known continuum states."""
    parameter = np.linspace(0.0, 1.0, points)
    times = parameter * duration
    states = []
    positions = []
    tangents = []
    envelope = np.sin(math.pi * parameter) ** 2
    for phase, shape in zip(parameter, envelope):
        theta_1 = math.radians(32.0) * shape
        theta_2 = math.radians(24.0) * shape
        direction_1 = 2.0 * math.pi * phase
        direction_2 = 2.0 * math.pi * phase + math.radians(55.0)
        state = np.asarray(
            [
                0.045 * phase,
                theta_1 * math.cos(direction_1),
                theta_1 * math.sin(direction_1),
                theta_2 * math.cos(direction_2),
                theta_2 * math.sin(direction_2),
            ]
        )
        position, tangent = forward_kinematics(state)
        states.append(state)
        positions.append(position)
        tangents.append(tangent)
    return times, np.asarray(positions), np.asarray(tangents)


def load_trajectory(path: Path, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time,x,y,z[,tx,ty,tz] in metres from CSV."""
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError("Trajectory CSV must contain at least two rows.")
    required = ("x_m", "y_m", "z_m")
    saved_result_names = ("target_x_m", "target_y_m", "target_z_m")
    if all(name in rows[0] for name in required):
        position_names = required
    elif all(name in rows[0] for name in saved_result_names):
        # A previously exported validation result can be fed straight back in.
        position_names = saved_result_names
    else:
        raise ValueError(
            "Trajectory CSV requires x_m,y_m,z_m or "
            "target_x_m,target_y_m,target_z_m columns."
        )
    positions = np.asarray(
        [[float(row[name]) for name in position_names] for row in rows], dtype=float
    )
    if "time_s" in rows[0] and rows[0]["time_s"] not in (None, ""):
        times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
        times -= times[0]
    else:
        times = np.linspace(0.0, duration, len(rows))
    tangent_names = ("tx", "ty", "tz")
    if all(name in rows[0] for name in tangent_names):
        tangents = normalize_rows(
            np.asarray(
                [[float(row[name]) for name in tangent_names] for row in rows]
            )
        )
    else:
        tangents = normalize_rows(np.gradient(positions, times, axis=0))
    return times, positions, tangents


def solve_trajectory_ik(
    positions: np.ndarray,
    tangents: np.ndarray,
    *,
    maximum_insertion_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a smooth warm-started IK sequence for all trajectory samples."""
    states = []
    residual_norms = []
    previous = np.zeros(5, dtype=float)
    previous[0] = np.clip(
        float(positions[0, 0] - 2.0 * SECTION_LENGTH_M),
        0.0,
        maximum_insertion_m,
    )
    lower = np.asarray([0.0, -MAX_BEND_RAD, -MAX_BEND_RAD, -MAX_BEND_RAD, -MAX_BEND_RAD])
    upper = np.asarray(
        [maximum_insertion_m, MAX_BEND_RAD, MAX_BEND_RAD, MAX_BEND_RAD, MAX_BEND_RAD]
    )
    smooth_scales = np.asarray([0.05, MAX_BEND_RAD, MAX_BEND_RAD, MAX_BEND_RAD, MAX_BEND_RAD])

    for target_position, target_tangent in zip(positions, tangents):
        def residual(state: np.ndarray) -> np.ndarray:
            position, tangent = forward_kinematics(state)
            bend_limit_penalties = np.asarray(
                [
                    max(0.0, np.linalg.norm(state[1:3]) - MAX_BEND_RAD),
                    max(0.0, np.linalg.norm(state[3:5]) - MAX_BEND_RAD),
                ]
            )
            return np.concatenate(
                (
                    (position - target_position) / 0.001,
                    (tangent - target_tangent) * 7.5,
                    (state - previous) / smooth_scales * 0.003,
                    bend_limit_penalties * 100.0,
                )
            )

        solution = least_squares(
            residual,
            previous,
            bounds=(lower, upper),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=150,
        )
        if not solution.success:
            raise RuntimeError(f"IK failed: {solution.message}")
        previous = solution.x
        states.append(previous.copy())
        solved_position, _ = forward_kinematics(previous)
        residual_norms.append(float(np.linalg.norm(solved_position - target_position)))
    return np.asarray(states), np.asarray(residual_norms)


def states_to_controls(
    model: mujoco.MjModel, data: mujoco.MjData, states: np.ndarray
) -> np.ndarray:
    controller = TendonCompassController(model, data)
    controls = []
    for state in states:
        controls.append(state_to_control(controller, state))
    return np.asarray(controls)


def state_to_control(
    controller: TendonCompassController, state: np.ndarray
) -> np.ndarray:
    """Convert one insertion/two-section IK state to seven actuator targets."""
    state = np.asarray(state, dtype=float)
    for segment_index, bend_slice in ((0, slice(1, 3)), (1, slice(3, 5))):
        bend_vector = state[bend_slice].copy()
        magnitude = float(np.linalg.norm(bend_vector))
        if magnitude > MAX_BEND_RAD:
            bend_vector *= MAX_BEND_RAD / magnitude
        controller.set_segment_vector(segment_index, bend_vector / MAX_BEND_RAD)
    return np.concatenate((controller.target_lengths.copy(), [state[0]]))


def simulate_tracking(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    times: np.ndarray,
    targets: np.ndarray,
    controls: np.ndarray,
    *,
    viewer: bool,
    hold_viewer: bool,
    control_lead_s: float = 0.0,
    ik_states: Optional[np.ndarray] = None,
    cartesian_feedback_gain: float = 0.0,
) -> np.ndarray:
    """Apply interpolated seven-axis controls and return measured local tips."""
    tendon_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_t{wire}"
            )
            for wire in range(1, 7)
        ],
        dtype=int,
    )
    insertion_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_slid_M"
    )
    base_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "base_frame")
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")
    mujoco.mj_forward(model, data)
    initial_origin = data.site_xpos[base_site].copy()
    initial_rotation = data.site_xmat[base_site].reshape(3, 3).copy()
    measured = [
        initial_rotation.T @ (data.site_xpos[tip_site] - initial_origin)
    ]
    feedback_controller = (
        TendonCompassController(model, data)
        if ik_states is not None and cartesian_feedback_gain > 0.0
        else None
    )

    passive_viewer = None

    def refresh_overlays(trace_local: list[np.ndarray], target_index: int) -> None:
        if (
            passive_viewer is None
            or not passive_viewer.is_running()
            or passive_viewer.user_scn is None
        ):
            return
        scene = passive_viewer.user_scn
        scene.ngeom = 0
        identity = np.eye(3, dtype=np.float64).reshape(-1)

        def add_sphere(local_point, radius, rgba):
            if scene.ngeom >= scene.maxgeom:
                return
            world_point = initial_origin + initial_rotation @ np.asarray(local_point)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.asarray([radius, radius, radius], dtype=np.float64),
                np.asarray(world_point, dtype=np.float64),
                identity,
                np.asarray(rgba, dtype=np.float32),
            )
            scene.ngeom += 1

        for point in targets:
            add_sphere(point, 0.0010, [0.05, 0.40, 1.00, 0.90])
        for point in trace_local:
            add_sphere(point, 0.00125, [0.05, 1.00, 0.25, 1.00])
        add_sphere(targets[target_index], 0.0022, [1.00, 0.75, 0.00, 1.00])

    if viewer:
        import mujoco.viewer as mj_viewer

        passive_viewer = mj_viewer.launch_passive(model, data)
        target_world = (
            initial_origin[None, :]
            + (initial_rotation @ np.asarray(targets).T).T
        )
        trajectory_center = np.mean(target_world, axis=0)
        trajectory_extent = float(
            np.max(np.ptp(target_world, axis=0))
        )
        passive_viewer.cam.lookat[:] = trajectory_center
        passive_viewer.cam.azimuth = 135.0
        passive_viewer.cam.elevation = -28.0
        passive_viewer.cam.distance = max(0.11, trajectory_extent * 2.8)
        refresh_overlays(measured, 0)
        passive_viewer.sync()
    try:
        for waypoint in range(1, len(times)):
            interval = float(times[waypoint] - times[waypoint - 1])
            steps = max(1, math.ceil(interval / model.opt.timestep))
            for step in range(steps):
                alpha = (step + 1) / steps
                command_time = (
                    float(times[waypoint - 1]) + alpha * interval + control_lead_s
                )
                command = np.asarray(
                    [
                        np.interp(command_time, times, controls[:, axis])
                        for axis in range(controls.shape[1])
                    ]
                )
                if feedback_controller is not None:
                    nominal_state = np.asarray(
                        [
                            np.interp(command_time, times, ik_states[:, axis])
                            for axis in range(ik_states.shape[1])
                        ]
                    )
                    feedback_time = float(times[waypoint - 1]) + alpha * interval
                    desired_now = np.asarray(
                        [
                            np.interp(feedback_time, times, targets[:, axis])
                            for axis in range(3)
                        ]
                    )
                    actual_now = initial_rotation.T @ (
                        data.site_xpos[tip_site] - initial_origin
                    )
                    position_error = desired_now - actual_now
                    jacobian = position_jacobian(nominal_state)
                    state_scales = np.asarray([0.04, 1.0, 1.0, 1.0, 1.0])
                    scaled_jacobian = jacobian * state_scales[None, :]
                    damping_m = 0.0025
                    correction_scaled = scaled_jacobian.T @ np.linalg.solve(
                        scaled_jacobian @ scaled_jacobian.T
                        + damping_m**2 * np.eye(3),
                        position_error,
                    )
                    state_correction = (
                        float(cartesian_feedback_gain)
                        * state_scales
                        * correction_scaled
                    )
                    state_correction[0] = np.clip(
                        state_correction[0], -0.004, 0.004
                    )
                    state_correction[1:] = np.clip(
                        state_correction[1:], -0.30, 0.30
                    )
                    corrected_state = nominal_state + state_correction
                    corrected_state[0] = np.clip(
                        corrected_state[0],
                        model.actuator_ctrlrange[insertion_id, 0],
                        model.actuator_ctrlrange[insertion_id, 1],
                    )
                    command = state_to_control(feedback_controller, corrected_state)
                data.ctrl[tendon_ids] = command[:6]
                data.ctrl[insertion_id] = command[6]
                mujoco.mj_step(model, data)
                if passive_viewer is not None and step % 20 == 0:
                    passive_viewer.sync()
            measured.append(
                initial_rotation.T @ (data.site_xpos[tip_site] - initial_origin)
            )
            refresh_overlays(measured, waypoint)
            if passive_viewer is not None and passive_viewer.is_running():
                passive_viewer.sync()

        if hold_viewer and passive_viewer is not None:
            print(
                "Tracking finished. Blue=target, green=actual, yellow=current. "
                "Close the MuJoCo viewer to exit."
            )
            while passive_viewer.is_running():
                refresh_overlays(measured, len(targets) - 1)
                passive_viewer.sync()
                time.sleep(0.02)
    finally:
        if passive_viewer is not None:
            passive_viewer.close()
    return np.asarray(measured)


def fresh_home_data(model: mujoco.MjModel, key_id: int) -> mujoco.MjData:
    """Create an independent, repeatable simulation starting at ``home``."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return data


def compensated_inverse_kinematics(
    model: mujoco.MjModel,
    key_id: int,
    times: np.ndarray,
    desired_positions: np.ndarray,
    desired_tangents: np.ndarray,
    *,
    maximum_insertion_m: float,
    iterations: int,
    learning_gain: float,
    control_lead_s: float,
    cartesian_feedback_gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use repeatable MuJoCo trials to compensate analytical-model mismatch.

    The analytical constant-curvature solution is retained as the prior.  Each
    hidden trial measures the complete dynamic tip error and moves the IK
    reference in the opposite direction.  This compensates tendon routing,
    compliance, actuator bandwidth and repeatable phase lag without changing
    the robot's mechanical or PID parameters.
    """
    corrected_positions = np.asarray(desired_positions, dtype=float).copy()
    corrected_tangents = np.asarray(desired_tangents, dtype=float).copy()

    for iteration in range(iterations):
        trial_data = fresh_home_data(model, key_id)
        trial_states, _ = solve_trajectory_ik(
            corrected_positions,
            corrected_tangents,
            maximum_insertion_m=maximum_insertion_m,
        )
        trial_controls = states_to_controls(model, trial_data, trial_states)
        measured = simulate_tracking(
            model,
            trial_data,
            times,
            desired_positions,
            trial_controls,
            viewer=False,
            hold_viewer=False,
            control_lead_s=control_lead_s,
            ik_states=trial_states,
            cartesian_feedback_gain=cartesian_feedback_gain,
        )
        error = desired_positions - measured
        errors_mm = np.linalg.norm(error, axis=1) * 1000.0
        rmse_mm = math.sqrt(float(np.mean(errors_mm**2)))
        maximum_mm = float(np.max(errors_mm))
        print(
            f"IK compensation trial {iteration + 1}/{iterations}: "
            f"RMSE {rmse_mm:.4f} mm, maximum {maximum_mm:.4f} mm"
        )

        # Fit one global affine map from IK-reference coordinates to measured
        # coordinates.  Its inverse corrects the observed scale, rotation and
        # bias while keeping the warm-started IK on a single smooth branch.
        design = np.column_stack(
            (corrected_positions, np.ones(len(corrected_positions)))
        )
        affine = np.linalg.lstsq(design, measured, rcond=None)[0]
        linear = affine[:3, :]
        offset = affine[3, :]
        condition = float(np.linalg.cond(linear))
        if not np.isfinite(condition) or condition > 20.0:
            raise RuntimeError(
                "Trajectory does not excite enough independent directions "
                f"for model compensation (condition number {condition:.2f})."
            )
        inverse_reference = (
            desired_positions - offset[None, :]
        ) @ np.linalg.inv(linear)
        correction = inverse_reference - corrected_positions
        # Bound each trial update so an unusual contact or near-singular sample
        # cannot request a discontinuous, unreachable bend configuration.
        correction_norm = np.linalg.norm(correction, axis=1, keepdims=True)
        correction *= np.minimum(
            1.0, 0.012 / np.maximum(correction_norm, 1e-12)
        )
        correction[0] = 0.0
        corrected_positions += float(learning_gain) * correction
        print(f"  affine condition number: {condition:.3f}")

    final_data = fresh_home_data(model, key_id)
    final_states, ik_errors = solve_trajectory_ik(
        corrected_positions,
        corrected_tangents,
        maximum_insertion_m=maximum_insertion_m,
    )
    final_controls = states_to_controls(model, final_data, final_states)
    return final_controls, final_states, ik_errors, corrected_positions


def write_results(
    path: Path,
    times: np.ndarray,
    targets: np.ndarray,
    controls: np.ndarray,
    measured: np.ndarray,
    ik_errors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_s",
        "target_x_m",
        "target_y_m",
        "target_z_m",
        *CONTROL_NAMES,
        "actual_x_m",
        "actual_y_m",
        "actual_z_m",
        "ik_error_mm",
        "tracking_error_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(times)):
            row = {
                "time_s": times[index],
                "target_x_m": targets[index, 0],
                "target_y_m": targets[index, 1],
                "target_z_m": targets[index, 2],
                "actual_x_m": measured[index, 0],
                "actual_y_m": measured[index, 1],
                "actual_z_m": measured[index, 2],
                "ik_error_mm": ik_errors[index] * 1000.0,
                "tracking_error_mm": np.linalg.norm(
                    measured[index] - targets[index]
                )
                * 1000.0,
            }
            row.update(
                {name: controls[index, axis] for axis, name in enumerate(CONTROL_NAMES)}
            )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--trajectory-csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional CSV path for seven-axis commands and tracking data. "
            "By default no test data file is written."
        ),
    )
    parser.add_argument("--points", type=int, default=31)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--with-lung", action="store_true")
    parser.set_defaults(viewer=True)
    parser.add_argument(
        "--viewer",
        dest="viewer",
        action="store_true",
        help="Open the MuJoCo trajectory viewer (default).",
    )
    parser.add_argument(
        "--headless",
        dest="viewer",
        action="store_false",
        help="Run validation without opening a window.",
    )
    parser.add_argument(
        "--close-viewer-on-finish",
        action="store_true",
        help="Close the viewer immediately after the final waypoint.",
    )
    parser.add_argument(
        "--ik-compensation-iterations",
        type=int,
        default=3,
        help=(
            "Number of hidden MuJoCo iterative-learning trials before the "
            "visible run (default: 3; use 0 for analytical IK only)."
        ),
    )
    parser.add_argument(
        "--ik-learning-gain",
        type=float,
        default=0.8,
        help="Cartesian iterative-learning gain in (0, 1] (default: 0.8).",
    )
    parser.add_argument(
        "--control-lead-ms",
        type=float,
        default=40.0,
        help="Command preview used to cancel actuator phase lag (default: 40 ms).",
    )
    parser.add_argument(
        "--cartesian-feedback-gain",
        type=float,
        default=0.60,
        help=(
            "Outer-loop damped-Jacobian position feedback gain "
            "(default: 0.60; use 0 for open loop)."
        ),
    )
    parser.add_argument("--max-rmse-mm", type=float, default=4.0)
    args = parser.parse_args()
    if args.ik_compensation_iterations < 0:
        parser.error("--ik-compensation-iterations must be non-negative.")
    if not 0.0 < args.ik_learning_gain <= 1.0:
        parser.error("--ik-learning-gain must be in (0, 1].")
    if args.control_lead_ms < 0.0:
        parser.error("--control-lead-ms must be non-negative.")
    if not 0.0 <= args.cartesian_feedback_gain <= 1.0:
        parser.error("--cartesian-feedback-gain must be in [0, 1].")

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    model.opt.gravity[:] = 0.0
    if not args.with_lung and model.nflex:
        model.flex_contype[:] = 0
        model.flex_conaffinity[:] = 0
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise RuntimeError("Model has no home keyframe.")

    if args.trajectory_csv:
        times, targets, tangents = load_trajectory(
            args.trajectory_csv.resolve(), args.duration
        )
    else:
        times, targets, tangents = demo_trajectory(args.points, args.duration)

    insertion_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_slid_M"
    )
    maximum_insertion = float(model.actuator_ctrlrange[insertion_id, 1])
    control_lead_s = args.control_lead_ms / 1000.0
    controls, states, ik_errors, corrected_targets = compensated_inverse_kinematics(
        model,
        key_id,
        times,
        targets,
        tangents,
        maximum_insertion_m=maximum_insertion,
        iterations=args.ik_compensation_iterations,
        learning_gain=args.ik_learning_gain,
        control_lead_s=control_lead_s,
        cartesian_feedback_gain=args.cartesian_feedback_gain,
    )
    data = fresh_home_data(model, key_id)
    measured = simulate_tracking(
        model,
        data,
        times,
        targets,
        controls,
        viewer=args.viewer,
        hold_viewer=args.viewer and not args.close_viewer_on_finish,
        control_lead_s=control_lead_s,
        ik_states=states,
        cartesian_feedback_gain=args.cartesian_feedback_gain,
    )
    tracking_errors = np.linalg.norm(measured - targets, axis=1) * 1000.0
    ik_rmse = math.sqrt(float(np.mean((ik_errors * 1000.0) ** 2)))
    tracking_rmse = math.sqrt(float(np.mean(tracking_errors**2)))
    tracking_max = float(np.max(tracking_errors))

    if args.output is not None:
        output_path = args.output.resolve()
        write_results(
            output_path, times, targets, controls, measured, ik_errors
        )
        print(f"Wrote seven-axis commands and validation data: {output_path}")
    print(f"Waypoints: {len(times)}, duration: {times[-1]:.3f} s")
    print(f"Analytical IK RMSE: {ik_rmse:.4f} mm")
    compensation_mm = np.linalg.norm(corrected_targets - targets, axis=1) * 1000.0
    print(
        f"Model compensation: {args.ik_compensation_iterations} trial(s), "
        f"mean offset {float(np.mean(compensation_mm)):.4f} mm, "
        f"command lead {args.control_lead_ms:.1f} ms, "
        f"Cartesian feedback gain {args.cartesian_feedback_gain:.2f}"
    )
    print(
        f"MuJoCo tracking RMSE: {tracking_rmse:.4f} mm, "
        f"maximum: {tracking_max:.4f} mm"
    )
    print("Seven outputs: " + ", ".join(CONTROL_NAMES))
    if tracking_rmse > args.max_rmse_mm:
        raise AssertionError(
            f"Trajectory tracking RMSE {tracking_rmse:.3f} mm exceeds "
            f"limit {args.max_rmse_mm:.3f} mm."
        )
    print("Inverse-kinematics trajectory validation passed.")


if __name__ == "__main__":
    main()
