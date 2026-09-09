#!/usr/bin/env python3
"""Compile the model and check proximal/distal tendon separation headlessly."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from control_demo import measurements, set_targets


HERE = Path(__file__).resolve().parent


def simulate_case(model: mujoco.MjModel, config: dict, proximal: float, distal: float):
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "straight_pretensioned"
    )
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    set_targets(model, data, config, proximal, 0.0, distal, 60.0)
    for _ in range(round(1.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)):
        raise AssertionError("Simulation produced non-finite qpos values.")
    return measurements(model, data)


def main() -> None:
    with (HERE / "config.json").open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    model = mujoco.MjModel.from_xml_path(str(HERE / "two_segment_tdcr.xml"))

    expected_actuators = [f"wire_{wire}" for wire in range(1, 7)]
    for name in expected_actuators:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) < 0:
            raise AssertionError(f"Missing actuator: {name}")

    straight = simulate_case(model, config, 0.0, 0.0)
    proximal = simulate_case(model, config, 30.0, 0.0)
    distal = simulate_case(model, config, 0.0, 30.0)
    maximum_command = config["tendons"][
        "maximum_commanded_bend_per_section_deg"
    ]
    maximum = simulate_case(model, config, maximum_command, maximum_command)

    if config["tendons"]["routing_mode"] == "independent":
        if proximal["proximal_angle_deg"] <= proximal["distal_angle_deg"] + 2.0:
            raise AssertionError(f"Proximal-only drive is not section-selective: {proximal}")
        if distal["distal_angle_deg"] <= distal["proximal_angle_deg"] + 2.0:
            raise AssertionError(f"Distal-only drive is not section-selective: {distal}")
    if maximum["proximal_angle_deg"] + maximum["distal_angle_deg"] <= 190.0:
        raise AssertionError(f"Combined active bending did not exceed 190 deg: {maximum}")

    print(f"Compiled MuJoCo model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"Straight:       {straight}")
    print(f"Proximal only:  {proximal}")
    print(f"Distal only:    {distal}")
    print(f"Maximum:        {maximum}")
    print("Validation passed.")


if __name__ == "__main__":
    main()
