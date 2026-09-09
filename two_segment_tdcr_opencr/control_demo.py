#!/usr/bin/env python3
"""Drive the two TDCR sections with constant-curvature tendon coordinates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, TextIO

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "two_segment_tdcr.xml"
DEFAULT_CONFIG = HERE / "config.json"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def section_commands(
    *,
    rest_length: float,
    angles_deg: Iterable[float],
    bend_deg: float,
    direction_deg: float,
    tendon_radius: float,
    pretension: float,
    kp: float,
) -> np.ndarray:
    """Map constant-curvature coordinates to three absolute tendon lengths.

    Positive bend direction points toward the tendon at the same polar angle.
    """
    theta = math.radians(bend_deg)
    phi = math.radians(direction_deg)
    baseline = rest_length - pretension / kp
    return np.asarray(
        [
            baseline - tendon_radius * theta * math.cos(math.radians(alpha) - phi)
            for alpha in angles_deg
        ],
        dtype=float,
    )


def set_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: Dict,
    proximal_bend_deg: float,
    proximal_direction_deg: float,
    distal_bend_deg: float,
    distal_direction_deg: float,
) -> None:
    geometry = config["geometry"]
    tendon = config["tendons"]
    section_length = geometry["section_length_m"]
    rigid = geometry["rigid_sections"]
    proximal_rest_length = (
        section_length
        + rigid["inter_section_length_m"]
    )
    distal_section_rest_length = section_length + rigid["distal_tip_length_m"]
    distal_rest_length = distal_section_rest_length
    if tendon["routing_mode"] == "coupled":
        distal_rest_length += proximal_rest_length
    common = {
        "tendon_radius": tendon["radius_from_center_m"],
        "pretension": tendon["pretension_n"],
        "kp": tendon["position_kp_n_per_m"],
    }
    proximal = section_commands(
        rest_length=proximal_rest_length,
        angles_deg=tendon["proximal_wire_angles_deg"],
        bend_deg=proximal_bend_deg,
        direction_deg=proximal_direction_deg,
        **common,
    )
    distal = section_commands(
        rest_length=distal_rest_length,
        angles_deg=tendon["distal_wire_angles_deg"],
        bend_deg=distal_bend_deg,
        direction_deg=distal_direction_deg,
        **common,
    )

    for wire, command in enumerate(np.concatenate((proximal, distal)), start=1):
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"wire_{wire}"
        )
        if actuator_id < 0:
            raise RuntimeError(f"Actuator wire_{wire} is missing from the model.")
        lower, upper = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(command, lower, upper)


def relative_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.reshape(3, 3).T @ b.reshape(3, 3)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def measurements(model: mujoco.MjModel, data: mujoco.MjData) -> Dict[str, float]:
    ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ("base_frame", "interface_center", "tip_center")
    }
    if min(ids.values()) < 0:
        raise RuntimeError("Required measurement sites are missing.")
    base = data.site_xmat[ids["base_frame"]]
    interface = data.site_xmat[ids["interface_center"]]
    tip = data.site_xmat[ids["tip_center"]]
    tip_position = data.site_xpos[ids["tip_center"]]
    return {
        "proximal_angle_deg": relative_angle_deg(base, interface),
        "distal_angle_deg": relative_angle_deg(interface, tip),
        "tip_x_m": float(tip_position[0]),
        "tip_y_m": float(tip_position[1]),
        "tip_z_m": float(tip_position[2]),
    }


def open_csv(path: Optional[Path]) -> tuple[Optional[TextIO], Optional[csv.DictWriter]]:
    if path is None:
        return None, None
    stream = path.open("w", newline="", encoding="utf-8")
    fields = [
        "time_s",
        "proximal_angle_deg",
        "distal_angle_deg",
        "tip_x_m",
        "tip_y_m",
        "tip_z_m",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    return stream, writer


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float,
    writer: Optional[csv.DictWriter],
) -> None:
    sample_interval = max(1, round(0.01 / model.opt.timestep))
    steps = math.ceil(duration / model.opt.timestep)
    for step in range(steps):
        mujoco.mj_step(model, data)
        if writer is not None and step % sample_interval == 0:
            writer.writerow({"time_s": data.time, **measurements(model, data)})


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float,
    writer: Optional[csv.DictWriter],
    render_fps: float,
) -> None:
    import mujoco.viewer

    if render_fps <= 0:
        raise ValueError("render_fps must be greater than zero.")

    sample_interval = max(1, round(0.01 / model.opt.timestep))
    # The physics step is 0.2 ms, so syncing the GUI after every mj_step would
    # request 5000 renders/s and make simulated time advance much slower than
    # wall time. Advance a frame's worth of physics first, then render at the
    # requested display rate. This changes only visualisation throughput, not
    # the model timestep or actuator dynamics.
    steps_per_frame = max(1, round((1.0 / render_fps) / model.opt.timestep))
    step = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.021, 0.0, 0.0)
        viewer.cam.distance = 0.085
        start = time.perf_counter()
        simulation_start = data.time
        while viewer.is_running() and data.time < duration:
            for _ in range(steps_per_frame):
                if data.time >= duration:
                    break
                mujoco.mj_step(model, data)
                if writer is not None and step % sample_interval == 0:
                    writer.writerow(
                        {"time_s": data.time, **measurements(model, data)}
                    )
                step += 1

            viewer.sync()
            # Keep simulated time close to real time when the computer is fast
            # enough. If rendering falls behind, skip sleeping and catch up.
            target_wall_time = start + (data.time - simulation_start)
            remaining = target_wall_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        print(f"Viewer wall time: {time.perf_counter() - start:.3f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--proximal-bend", type=float, default=45.0, metavar="DEG")
    parser.add_argument("--proximal-direction", type=float, default=0.0, metavar="DEG")
    parser.add_argument("--distal-bend", type=float, default=45.0, metavar="DEG")
    parser.add_argument("--distal-direction", type=float, default=90.0, metavar="DEG")
    parser.add_argument("--duration", type=float, default=2000.0, metavar="SECONDS")
    parser.add_argument(
        "--render-fps",
        type=float,
        default=60.0,
        metavar="FPS",
        help="Viewer refresh rate; physics is batched between frames (default: 60)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "straight_pretensioned"
    )
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    set_targets(
        model,
        data,
        config,
        args.proximal_bend,
        args.proximal_direction,
        args.distal_bend,
        args.distal_direction,
    )

    csv_stream, writer = open_csv(args.csv)
    try:
        if args.headless:
            run_headless(model, data, args.duration, writer)
        else:
            run_viewer(model, data, args.duration, writer, args.render_fps)
    finally:
        if csv_stream is not None:
            csv_stream.close()

    result = measurements(model, data)
    print(
        "Final shape: "
        f"proximal={result['proximal_angle_deg']:.2f} deg, "
        f"distal={result['distal_angle_deg']:.2f} deg, "
        f"tip=({result['tip_x_m']:.5f}, {result['tip_y_m']:.5f}, "
        f"{result['tip_z_m']:.5f}) m"
    )


if __name__ == "__main__":
    main()
