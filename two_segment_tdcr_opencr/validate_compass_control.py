#!/usr/bin/env python3
"""Validate compass direction mapping and tendon-servo step response."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from tendon_compass_control import TendonCompassController


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "meshes" / "cable_robot_bronch_final_seg2.xml"


def relative_angle_deg(data: mujoco.MjData, site_a: int, site_b: int) -> float:
    rotation_a = data.site_xmat[site_a].reshape(3, 3)
    rotation_b = data.site_xmat[site_b].reshape(3, 3)
    cosine = np.clip(
        (np.trace(rotation_a.T @ rotation_b) - 1.0) / 2.0, -1.0, 1.0
    )
    return math.degrees(math.acos(float(cosine)))


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.gravity[:] = 0.0
    if model.nflex:
        model.flex_contype[:] = 0
        model.flex_conaffinity[:] = 0
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    controller = TendonCompassController(model, data)

    controller.set_segment_vector(0, [1.0, 0.0])
    pull = controller.commanded_pull_mm()
    if not (pull[0] > 0 and pull[1] < 0 and pull[2] < 0):
        raise AssertionError(f"Proximal 0-degree mapping is wrong: {pull[:3]}")
    if not np.isclose(pull[1], pull[2]):
        raise AssertionError("Proximal symmetric tendon pair must change equally.")

    controller.set_segment_vector(0, [0.0, 1.0])
    pull = controller.commanded_pull_mm()
    if not (np.isclose(pull[0], 0.0) and pull[1] > 0 and pull[2] < 0):
        raise AssertionError(f"Proximal 90-degree mapping is wrong: {pull[:3]}")

    controller.set_segment_vector(1, [1.0, 0.0])
    pull = controller.commanded_pull_mm()
    if not (pull[3] > 0 and pull[4] < 0 and pull[5] > 0):
        raise AssertionError(f"Distal 0-degree mapping is wrong: {pull[3:]}")
    if not np.isclose(pull[3], pull[5]):
        raise AssertionError("Distal symmetric tendon pair must change equally.")
    single_section_distal_pull = pull[3:].copy()

    controller.set_segment_vector(1, [2.0, 0.0])
    combined_pull = controller.commanded_pull_mm()[3:]
    if not np.allclose(combined_pull, 2.0 * single_section_distal_pull):
        raise AssertionError(
            "Distal coupled tendons must retain two full sections of travel."
        )
    if not np.isclose(np.linalg.norm(controller.segment_vectors[1]), 2.0):
        raise AssertionError("Distal combined motor command was clamped below 2.0.")

    controller.reset()
    for _ in range(math.ceil(0.1 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    controller.set_segment_vector(0, [60.0 / 160.0, 0.0])

    base_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "base_frame"
    )
    interface_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "interface_center"
    )
    times = []
    angles = []
    for step in range(math.ceil(0.35 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if step % 5 == 0:
            times.append((step + 1) * model.opt.timestep)
            angles.append(relative_angle_deg(data, base_site, interface_site))

    times = np.asarray(times)
    angles = np.asarray(angles)
    final_angle = float(np.mean(angles[-50:]))
    peak_angle = float(np.max(angles))
    crossing = np.flatnonzero(angles >= 0.9 * final_angle)
    rise_time = float(times[crossing[0]]) if len(crossing) else math.inf
    overshoot = max(0.0, (peak_angle - final_angle) / final_angle * 100.0)
    tail_std = float(np.std(angles[-50:]))

    if rise_time >= 0.08:
        raise AssertionError(f"Tendon response is too slow: t90={rise_time:.4f} s")
    if overshoot >= 2.0:
        raise AssertionError(f"Tendon response overshoot is too large: {overshoot:.3f}%")
    if tail_std >= 0.03:
        raise AssertionError(f"Tendon response is oscillatory: std={tail_std:.5f} deg")
    if final_angle <= 45.0:
        raise AssertionError(f"Tendon response is too weak: {final_angle:.3f} deg")

    print("Compass mapping: Wire 1-3 proximal, Wire 4-6 distal")
    print(
        f"60 deg step: final={final_angle:.3f} deg, t90={rise_time:.4f} s, "
        f"overshoot={overshoot:.3f}%, tail_std={tail_std:.5f} deg"
    )
    print("Compass tendon-control validation passed.")


if __name__ == "__main__":
    main()
