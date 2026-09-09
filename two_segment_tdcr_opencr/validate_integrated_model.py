#!/usr/bin/env python3
"""Validate the full bronchoscope after replacing its two active tip sections."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from contact_stabilization import ActiveTipContactRegularizer
from passive_joint_control import (
    PassiveActiveFollowerController,
    PassiveJointParameterController,
)


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "meshes" / "cable_robot_bronch_final_seg2.xml"


def object_id(model: mujoco.MjModel, object_type, name: str) -> int:
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise AssertionError(f"Missing MuJoCo object: {name}")
    return result


def rotation_error_deg(reference: np.ndarray, current: np.ndarray) -> float:
    delta = reference.T @ current
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    passive_controller = PassiveJointParameterController(model)
    passive_defaults = passive_controller.current_parameters()
    if passive_defaults["joint_count"] != 29:
        raise AssertionError("Passive parameter controller must cover 29 ball joints.")
    follower_stiffness = np.asarray([8.0, 6.0, 4.5, 3.0, 2.0, 1.0])
    follower_damping = np.asarray([1.6, 1.4, 1.2, 1.0, 0.8, 0.6])
    follower_bend_deg = np.asarray([6.0, 8.0, 10.0, 12.0, 15.0, 18.0])
    expected_stiffness = (23 * 500.0 + float(np.sum(follower_stiffness))) / 29.0
    expected_damping = (23 * 10.0 + float(np.sum(follower_damping))) / 29.0
    if not np.isclose(passive_defaults["stiffness"], expected_stiffness):
        raise AssertionError("Passive follower stiffness profile is incorrect.")
    if not np.isclose(passive_defaults["damping"], expected_damping):
        raise AssertionError("Passive follower damping profile is incorrect.")
    passive_scaled = passive_controller.set_scales(0.5, 0.5)
    if not np.isclose(passive_scaled["stiffness"], expected_stiffness * 0.5):
        raise AssertionError("Passive runtime stiffness scaling failed.")
    if not np.isclose(passive_scaled["damping"], expected_damping * 0.5):
        raise AssertionError("Passive runtime damping scaling failed.")
    passive_controller.reset()
    terminal_joint_names = [
        *(f"cable_stiffJ_{index}" for index in range(24, 29)),
        "cable_stiffJ_last",
    ]
    for joint_name, stiffness, damping, bend_deg in zip(
        terminal_joint_names,
        follower_stiffness,
        follower_damping,
        follower_bend_deg,
    ):
        joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        dof_id = int(model.jnt_dofadr[joint_id])
        if not np.isclose(model.jnt_stiffness[joint_id], stiffness):
            raise AssertionError(f"Follower stiffness is incorrect: {joint_name}")
        if not np.allclose(model.dof_damping[dof_id:dof_id + 3], damping):
            raise AssertionError(f"Follower damping is incorrect: {joint_name}")
        if not bool(model.jnt_limited[joint_id]) or not np.isclose(
            model.jnt_range[joint_id, 1], math.radians(float(bend_deg))
        ):
            raise AssertionError(f"Follower bend limit is incorrect: {joint_name}")

    data = mujoco.MjData(model)
    key_id = object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    first_passive_qpos = int(passive_controller.qpos_ids[0])
    first_passive_dof = int(model.jnt_dofadr[passive_controller.joint_ids[0]])
    data.qpos[first_passive_qpos:first_passive_qpos + 4] = (
        math.cos(0.15),
        math.sin(0.15),
        0.0,
        0.0,
    )
    data.qvel[first_passive_dof:first_passive_dof + 3] = (1.0, -2.0, 3.0)
    debug_state = passive_controller.set_debug_lock(data, True)
    if debug_state != {"enabled": True, "joint_count": 29}:
        raise AssertionError("Passive debug lock reports an invalid state.")
    for qpos_id in passive_controller.qpos_ids:
        if not np.allclose(
            data.qpos[qpos_id:qpos_id + 4],
            [1.0, 0.0, 0.0, 0.0],
            atol=1e-12,
        ):
            raise AssertionError("Passive debug lock did not straighten every joint.")
    if not np.allclose(data.qvel[passive_controller.dof_ids], 0.0, atol=1e-12):
        raise AssertionError("Passive debug lock did not zero every joint velocity.")
    passive_controller.set_debug_lock(data, False)
    if passive_controller.debug_locked:
        raise AssertionError("Passive debug lock could not be disabled.")
    mujoco.mj_forward(model, data)
    follower_controller = PassiveActiveFollowerController(model, data)
    data.qfrc_applied[:] = 0.0
    follower_controller.apply()
    follower_state = follower_controller.current_state()
    if follower_state["active_bend_deg"] > 1e-6:
        raise AssertionError("Passive follower must be neutral in the home pose.")
    if not np.allclose(data.qfrc_applied, 0.0, atol=1e-9):
        raise AssertionError("Passive follower applies torque in the home pose.")

    measurement_site_ids = [
        object_id(model, mujoco.mjtObj.mjOBJ_SITE, f"measurement_kp_{index}")
        for index in range(7)
    ]
    measurement_base_id = object_id(
        model, mujoco.mjtObj.mjOBJ_SITE, "base_frame"
    )
    measurement_rotation = data.site_xmat[measurement_base_id].reshape(3, 3)
    measurement_points = (
        measurement_rotation.T
        @ (
            data.site_xpos[measurement_site_ids]
            - data.site_xpos[measurement_base_id]
        ).T
    ).T
    # Physical positions include the 6 mm section spacer. The final 6 mm rigid
    # tip lies beyond measurement_kp_6.
    expected_measurement_x = np.asarray([0, 7, 14, 21, 34, 41, 48]) * 1e-3
    if not np.allclose(measurement_points[:, 0], expected_measurement_x, atol=1e-10):
        raise AssertionError(
            "D435 sites must include the configured rigid-section offsets."
        )
    if not np.allclose(measurement_points[:, 1:], 0.0, atol=1e-10):
        raise AssertionError("D435 measurement sites must lie on the neutral backbone.")

    required_bodies = (
        "cable_stiffB_last",
        "active_tdcr_base",
        "seg2_body",
        "end_6",
    )
    body_ids = {
        name: object_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in required_bodies
    }
    robot_root_id = object_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
    )
    if not np.allclose(
        model.body_pos[robot_root_id],
        [-0.38279, 0.00293, 1.00064],
        atol=1e-12,
    ):
        raise AssertionError("Robot root default world position is incorrect.")
    object_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "tip_camera")
    tip_site = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")
    tip_relative = measurement_rotation.T @ (
        data.site_xpos[tip_site] - data.site_xpos[measurement_base_id]
    )
    if not np.allclose(tip_relative, [0.054, 0.0, 0.0], atol=1e-10):
        raise AssertionError("Straight continuum length must be exactly 54 mm.")
    for rigid_name, expected_length in (
        ("inter_section_rigid", 0.006),
        ("distal_tip_rigid", 0.006),
    ):
        rigid_id = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, rigid_name)
        if not np.isclose(2.0 * model.geom_size[rigid_id, 1], expected_length):
            raise AssertionError(f"Unexpected rigid-section length: {rigid_name}")
    slider_actuator = object_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_slid_M"
    )
    if not bool(model.actuator_forcelimited[slider_actuator]):
        raise AssertionError("Insertion actuator must retain its force limit.")
    if not np.allclose(model.actuator_forcerange[slider_actuator], [-250.0, 250.0]):
        raise AssertionError("Insertion actuator must retain its +/-250 N limit.")
    if not np.isclose(model.actuator_gainprm[slider_actuator, 0], 6000.0):
        raise AssertionError("Insertion position gain must remain Kp=6000 N/m.")
    if not np.allclose(
        model.actuator_biasprm[slider_actuator, :3], [0.0, -6000.0, -180.0]
    ):
        raise AssertionError("Insertion velocity feedback must remain Kd=180 N*s/m.")
    for wire in range(1, 7):
        tendon_actuator = object_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_t{wire}"
        )
        object_id(model, mujoco.mjtObj.mjOBJ_TENDON, f"tendon{wire}")
        if not np.isclose(model.actuator_gainprm[tendon_actuator, 0], 30000.0):
            raise AssertionError(f"Wire {wire} must retain Kp=30000 N/m.")
        if not np.allclose(
            model.actuator_biasprm[tendon_actuator, :3],
            [0.0, -30000.0, -0.5],
        ):
            raise AssertionError(f"Wire {wire} must retain Kd=0.5 N*s/m.")
        command_sections = 2.0 if wire >= 4 else 1.0
        required_excursion = (
            command_sections * 0.00154 * math.radians(160.0)
        )
        baseline = float(model.key_ctrl[key_id, tendon_actuator])
        lower, upper = model.actuator_ctrlrange[tendon_actuator]
        if lower > baseline - required_excursion + 1e-9:
            raise AssertionError(f"Wire {wire} has insufficient pull travel.")
        if upper < baseline + required_excursion - 1e-9:
            raise AssertionError(f"Wire {wire} has insufficient release travel.")

    def is_body_descendant(body_id: int, ancestor_id: int) -> bool:
        while body_id > 0:
            if body_id == ancestor_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return False

    active_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if is_body_descendant(
            int(model.geom_bodyid[geom_id]), body_ids["active_tdcr_base"]
        )
        and model.geom_contype[geom_id] == 1
        and model.geom_conaffinity[geom_id] == 0
    ]
    if not active_geom_ids:
        raise AssertionError("No active TDCR collision geoms use mask 1/0.")
    fairing_geom = object_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "splice_collision_fairing"
    )
    if model.geom_type[fairing_geom] != mujoco.mjtGeom.mjGEOM_CAPSULE:
        raise AssertionError("Passive/active splice collision fairing must be a capsule.")
    if not np.isclose(model.geom_size[fairing_geom, 0], 0.00145):
        raise AssertionError("Splice collision fairing must retain its tapered radius.")
    passive_fairing_geom = object_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "splice_passive_collision"
    )
    if any(
        int(model.geom_contype[geom_id]) != 0
        for geom_id in (passive_fairing_geom, fairing_geom)
    ):
        raise AssertionError("Passive/active splice transition must be collision-free.")
    section_fairing_geom = object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "section_interface_collision_fairing",
    )
    if model.geom_type[section_fairing_geom] != mujoco.mjtGeom.mjGEOM_CAPSULE:
        raise AssertionError("Active-section interface fairing must be a capsule.")
    if not np.isclose(model.geom_size[section_fairing_geom, 0], 0.00155):
        raise AssertionError("Active-section interface fairing radius must be 1.55 mm.")
    passive_radii = []
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        ) or ""
        if geom_name.startswith("cable_stiffG"):
            passive_radii.append(float(model.geom_size[geom_id, 0]))
    if not passive_radii or not np.allclose(passive_radii, 0.00175):
        raise AssertionError("Passive cable diameter must remain 3.5 mm.")
    for visual_only_name in (
        "cable_stiffG29",
        "section_1_element_0",
        "section_1_element_1",
    ):
        visual_only_id = object_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, visual_only_name
        )
        if int(model.geom_contype[visual_only_id]) != 0:
            raise AssertionError(
                f"Flat splice geom must be visual-only: {visual_only_name}"
            )
    bronchial_flex_id = object_id(
        model, mujoco.mjtObj.mjOBJ_FLEX, "bronchial_wall_nonconvex"
    )
    if not bool(model.flex_rigid[bronchial_flex_id]):
        raise AssertionError("Bronchial collision flex must be rigid.")
    if int(model.flex_group[bronchial_flex_id]) != 4:
        raise AssertionError("Bronchial collision flex must use dedicated group 4.")
    if int(model.flex_condim[bronchial_flex_id]) != 1:
        raise AssertionError("Bronchial collision must use stable normal-only contact.")
    if not np.allclose(model.flex_friction[bronchial_flex_id], 0.0):
        raise AssertionError("Bronchial wall must remain completely frictionless.")
    if not np.allclose(model.flex_solref[bronchial_flex_id], [0.002, 1.0]):
        raise AssertionError("Bronchial collision must retain the smooth contact time constant.")
    if not np.allclose(
        model.flex_solimp[bronchial_flex_id, :3], [0.95, 0.99, 0.001]
    ):
        raise AssertionError("Bronchial collision impedance must remain smoothly graded.")
    if not np.isclose(model.flex_radius[bronchial_flex_id], 0.001):
        raise AssertionError(
            "Bronchial collision radius must remain 1 mm to avoid lumen narrowing."
        )
    active_joint_dofs = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        ) or ""
        if joint_name.startswith("section_"):
            active_joint_dofs.append(int(model.jnt_dofadr[joint_id]))
    if not active_joint_dofs:
        raise AssertionError("No active continuum bending joints were found.")
    if not np.allclose(model.dof_damping[active_joint_dofs], 0.006):
        raise AssertionError("Active continuum damping must remain 0.006 N*m*s/rad.")
    if not np.allclose(model.dof_armature[active_joint_dofs], 1e-10):
        raise AssertionError("Active continuum armature must remain 1e-10 kg*m^2.")
    if not np.isclose(model.flex_margin[bronchial_flex_id], 0.00015):
        raise AssertionError("Bronchial predictive contact margin must remain 0.15 mm.")
    if float(model.flex_rgba[bronchial_flex_id, 3]) <= 0:
        raise AssertionError("Bronchial collision flex must have visible alpha.")
    if (
        model.flex_contype[bronchial_flex_id] != 1
        or model.flex_conaffinity[bronchial_flex_id] != 1
    ):
        raise AssertionError("Bronchial collision flex must use mask 1/1.")
    active_id = active_geom_ids[0]
    collision_masks_overlap = bool(
        (
            model.geom_contype[active_id]
            & model.flex_conaffinity[bronchial_flex_id]
        )
        or (
            model.flex_contype[bronchial_flex_id]
            & model.geom_conaffinity[active_id]
        )
    )
    if not collision_masks_overlap:
        raise AssertionError("Active TDCR and bronchial collision masks do not overlap.")

    robot_geom_id_set = set(active_geom_ids)
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        ) or ""
        if body_name.startswith("cable_stiffB"):
            robot_geom_id_set.add(geom_id)
    if not np.isclose(model.opt.timestep, 0.0002):
        raise AssertionError(f"Unexpected integrated timestep: {model.opt.timestep}")
    if model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
        raise AssertionError("Integrated model must use the fast pyramidal cone.")
    contact_robot_geoms = [
        geom_id
        for geom_id in robot_geom_id_set
        if int(model.geom_contype[geom_id]) == 1
    ]
    if any(int(model.geom_condim[gid]) != 1 for gid in contact_robot_geoms):
        raise AssertionError("All robot/lung contacts must remain normal-only.")
    if any(not np.allclose(model.geom_friction[gid], 0.0) for gid in contact_robot_geoms):
        raise AssertionError("All robot/lung contacts must remain frictionless.")
    smooth_collision_geoms = [
        geom_id
        for geom_id in contact_robot_geoms
        if (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        ).startswith("smooth_collision_")
    ]
    if len(smooth_collision_geoms) != 49:
        raise AssertionError(
            "Expected 49 enabled rounded robot collision envelopes."
        )
    if any(
        model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_CAPSULE
        for geom_id in smooth_collision_geoms
    ):
        raise AssertionError("Every smooth robot collision envelope must be a capsule.")
    disabled_interface_capsules = (
        "smooth_collision_section_1_element_11",
        "smooth_collision_section_1_element_12",
        "smooth_collision_section_2_element_0",
        "smooth_collision_section_2_element_1",
    )
    if any(
        int(model.geom_contype[object_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)]) != 0
        for name in disabled_interface_capsules
    ):
        raise AssertionError(
            "Overlapping active-section interface capsules must remain disabled."
        )
    flat_robot_collision_geoms = [
        geom_id
        for geom_id in contact_robot_geoms
        if (
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
            .startswith(("cable_stiffG", "section_1_element_", "section_2_element_"))
        )
    ]
    if flat_robot_collision_geoms:
        raise AssertionError("Flat-ended robot cylinders must be visual-only.")
    if int(model.flex_elemnum[bronchial_flex_id]) > 20_000:
        raise AssertionError("Bronchial collision mesh exceeds the 20k face budget.")

    # Check the complete free-space step response, rather than accepting the
    # first tolerance crossing of an oscillatory response.
    response_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, response_data, key_id)
    slider_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "slid_M")
    slider_qpos_adr = int(model.jnt_qposadr[slider_joint])
    response_target = 0.20
    response_data.ctrl[slider_actuator] = response_target
    response_positions = []
    response_duration = 0.50
    original_disableflags = int(model.opt.disableflags)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    try:
        for _ in range(math.ceil(response_duration / model.opt.timestep)):
            mujoco.mj_step(model, response_data)
            response_positions.append(float(response_data.qpos[slider_qpos_adr]))
    finally:
        model.opt.disableflags = original_disableflags
    response_positions = np.asarray(response_positions)
    response_error = np.abs(response_positions - response_target)
    insertion_response_time = None
    for index in range(len(response_positions)):
        if np.all(response_error[index:] <= 0.001):
            insertion_response_time = (index + 1) * model.opt.timestep
            break
    response_overshoot = max(0.0, float(response_positions.max() - response_target))
    if insertion_response_time is None or insertion_response_time > 0.30:
        raise AssertionError("Insertion did not settle within 1 mm in 0.30 s.")
    if response_overshoot > 0.0002:
        raise AssertionError(
            f"Insertion step response overshot by {response_overshoot:.6f} m."
        )

    lengths_mm = []
    for wire in range(1, 7):
        tendon_id = object_id(model, mujoco.mjtObj.mjOBJ_TENDON, f"tendon{wire}")
        lengths_mm.append(float(data.ten_length[tendon_id]) * 1000.0)
    if not np.allclose(lengths_mm[:3], 27.0, atol=0.02):
        raise AssertionError(f"Unexpected proximal tendon lengths: {lengths_mm[:3]}")
    if not np.allclose(lengths_mm[3:], 54.0, atol=0.02):
        raise AssertionError(f"Unexpected distal tendon lengths: {lengths_mm[3:]}")

    passive_tip_site_id = object_id(
        model, mujoco.mjtObj.mjOBJ_SITE, "cable_stiffS_last"
    )
    # A MuJoCo cable composite's final body origin is one discretisation step
    # before its geometric endpoint. Compare against its terminal site instead.
    passive_tip = data.site_xpos[passive_tip_site_id]
    active_base = data.xpos[body_ids["active_tdcr_base"]]
    splice_error = float(np.linalg.norm(passive_tip - active_base))
    if splice_error > 0.001:
        raise AssertionError(f"Passive/active splice error is too large: {splice_error}")

    passive_rotation = data.site_xmat[passive_tip_site_id].reshape(3, 3)
    active_rotation = data.xmat[body_ids["active_tdcr_base"]].reshape(3, 3)
    reference_relative_rotation = passive_rotation.T @ active_rotation
    max_splice_translation = splice_error
    max_splice_rotation_deg = 0.0

    def update_splice_drift() -> None:
        nonlocal max_splice_translation, max_splice_rotation_deg
        translation = float(
            np.linalg.norm(
                data.site_xpos[passive_tip_site_id]
                - data.xpos[body_ids["active_tdcr_base"]]
            )
        )
        current_relative_rotation = (
            data.site_xmat[passive_tip_site_id].reshape(3, 3).T
            @ data.xmat[body_ids["active_tdcr_base"]].reshape(3, 3)
        )
        max_splice_translation = max(max_splice_translation, translation)
        max_splice_rotation_deg = max(
            max_splice_rotation_deg,
            rotation_error_deg(
                reference_relative_rotation, current_relative_rotation
            ),
        )

    # Settle under the full model's gravity/contact settings.
    settle_duration = 0.5
    for _ in range(math.ceil(settle_duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
        update_splice_drift()
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise AssertionError("Integrated simulation produced non-finite state.")
    if data.time < settle_duration * 0.99:
        raise AssertionError(
            f"Simulation time reset before settling completed: {data.time}"
        )

    tip_before = data.xpos[body_ids["end_6"]].copy()
    distal_actuator = object_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_t4"
    )
    lower, upper = model.actuator_ctrlrange[distal_actuator]
    data.ctrl[distal_actuator] = np.clip(
        data.ctrl[distal_actuator] - 0.001, lower, upper
    )
    driven_duration = 0.5
    driven_start = data.time
    for _ in range(math.ceil(driven_duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
        update_splice_drift()
    if data.time < driven_start + driven_duration * 0.99:
        raise AssertionError(
            f"Simulation time reset during tendon drive: {data.time}"
        )
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise AssertionError("Tendon drive produced non-finite state.")
    tip_motion = float(np.linalg.norm(data.xpos[body_ids["end_6"]] - tip_before))
    if tip_motion < 1e-5:
        raise AssertionError(f"Distal tendon produced negligible tip motion: {tip_motion}")
    if max_splice_translation > 1e-6:
        raise AssertionError(
            "Passive/active interface translated relative to itself: "
            f"{max_splice_translation} m"
        )
    if max_splice_rotation_deg > 0.01:
        raise AssertionError(
            "Passive/active interface rotated relative to itself: "
            f"{max_splice_rotation_deg} deg"
        )

    # Exercise the collision pipeline, not only its masks.  Advancing the
    # insertion axis toward 0.25 m brings the force-limited bronchoscope against
    # the bronchial wall in the supplied scene. A real robot/flex contact must
    # then appear.
    collision_data = mujoco.MjData(model)
    collision_regularizer = ActiveTipContactRegularizer(model, collision_data)
    mujoco.mj_resetDataKeyframe(model, collision_data, key_id)
    collision_data.ctrl[slider_actuator] = min(
        0.25, float(model.actuator_ctrlrange[slider_actuator, 1])
    )
    collision_data.ctrl[distal_actuator] = float(lower)
    max_lung_contacts = 0
    contact_test_steps = math.ceil(1.0 / model.opt.timestep)
    for _ in range(contact_test_steps):
        collision_regularizer.before_step()
        mujoco.mj_step(model, collision_data)
        lung_contacts = 0
        for contact_id in range(collision_data.ncon):
            contact = collision_data.contact[contact_id]
            for flex_side in (0, 1):
                if int(contact.flex[flex_side]) == bronchial_flex_id:
                    other_geom = int(contact.geom[1 - flex_side])
                    if other_geom in robot_geom_id_set:
                        lung_contacts += 1
                    break
        max_lung_contacts = max(max_lung_contacts, lung_contacts)
    if max_lung_contacts == 0:
        raise AssertionError(
            "No physical bronchoscope/bronchial-wall contact was generated."
        )
    if not np.all(np.isfinite(collision_data.qpos)) or not np.all(
        np.isfinite(collision_data.qvel)
    ):
        raise AssertionError("Bronchial contact test produced non-finite state.")

    # Regression test for wall tunnelling: command the full 0.577 m travel with
    # no software contact latch. The force-limited dynamic slider must remain
    # stopped at the physical non-convex wall.
    barrier_data = mujoco.MjData(model)
    barrier_regularizer = ActiveTipContactRegularizer(model, barrier_data)
    mujoco.mj_resetDataKeyframe(model, barrier_data, key_id)
    barrier_target = float(model.actuator_ctrlrange[slider_actuator, 1])
    barrier_data.ctrl[slider_actuator] = barrier_target
    barrier_contacts = 0
    first_barrier_contact_step = None
    peak_insertion_force = 0.0
    barrier_steps = math.ceil(1.0 / model.opt.timestep)
    for step_index in range(barrier_steps):
        barrier_regularizer.before_step()
        mujoco.mj_step(model, barrier_data)
        peak_insertion_force = max(
            peak_insertion_force,
            abs(float(barrier_data.actuator_force[slider_actuator])),
        )
        for contact_id in range(barrier_data.ncon):
            contact = barrier_data.contact[contact_id]
            if bronchial_flex_id in (int(contact.flex[0]), int(contact.flex[1])):
                barrier_contacts += 1
                if first_barrier_contact_step is None:
                    first_barrier_contact_step = step_index
                break
    stopped_insertion = float(barrier_data.qpos[slider_qpos_adr])
    hold_tip_positions = []
    hold_contact_counts = []
    for _ in range(math.ceil(0.3 / model.opt.timestep)):
        barrier_regularizer.before_step()
        mujoco.mj_step(model, barrier_data)
        hold_tip_positions.append(barrier_data.xpos[body_ids["end_6"]].copy())
        hold_contact_counts.append(
            sum(
                bronchial_flex_id
                in (
                    int(barrier_data.contact[contact_id].flex[0]),
                    int(barrier_data.contact[contact_id].flex[1]),
                )
                for contact_id in range(barrier_data.ncon)
            )
        )
    hold_drift = abs(
        float(barrier_data.qpos[slider_qpos_adr]) - stopped_insertion
    )
    hold_tip_velocity = np.diff(np.asarray(hold_tip_positions), axis=0) / model.opt.timestep
    hold_tip_speed_rms = float(
        np.sqrt(np.mean(np.sum(hold_tip_velocity * hold_tip_velocity, axis=1)))
    )
    hold_contact_std = float(np.std(hold_contact_counts))
    if barrier_contacts == 0:
        raise AssertionError("Insertion barrier test never touched the bronchial wall.")
    post_contact_steps = barrier_steps - int(first_barrier_contact_step)
    barrier_contact_fraction = barrier_contacts / post_contact_steps
    if barrier_contact_fraction < 0.90:
        raise AssertionError(
            "Insertion lost sustained physical wall contact: "
            f"contact fraction={barrier_contact_fraction:.3f}"
        )
    if stopped_insertion > barrier_target + 0.001:
        raise AssertionError(
            f"Insertion exceeded its commanded travel: q={stopped_insertion:.6f} m"
        )
    if peak_insertion_force > 250.001:
        raise AssertionError(
            "Insertion exceeded the 250 N limit: "
            f"peak={peak_insertion_force:.6f} N"
        )
    if hold_drift > 0.001:
        raise AssertionError(
            f"Insertion did not settle at the wall: drift={hold_drift:.6f} m"
        )
    if hold_tip_speed_rms > 0.025:
        raise AssertionError(
            "Continuum tip still chatters during deep wall contact: "
            f"RMS speed={hold_tip_speed_rms:.6f} m/s"
        )

    print(
        f"Compiled full model: nq={model.nq}, nv={model.nv}, nu={model.nu}, "
        f"dt={model.opt.timestep:g} s"
    )
    print(f"Tendon lengths (mm): {np.round(lengths_mm, 4).tolist()}")
    print("D435 measurement arclengths (mm): [0, 7, 14, 21, 28, 35, 42]")
    print(f"Passive-to-active splice error: {splice_error * 1e3:.6f} mm")
    print(
        "Splice transition: passive/active visual fairing is collision-free; "
        "active-section interface uses a 1.55 mm rounded capsule"
    )
    print(
        "Maximum splice drift: "
        f"{max_splice_translation * 1e3:.6f} mm, "
        f"{max_splice_rotation_deg:.8f} deg"
    )
    print(f"Tip motion from 1 mm Wire 4 pull: {tip_motion * 1e3:.4f} mm")
    print(
        "Collision: active TDCR geom=1/0, bronchial rigid non-convex flex=1/1, "
        "display group=4 "
        f"({int(model.flex_elemnum[bronchial_flex_id])} triangles)"
    )
    print(f"Physical lung contact test: max contacts={max_lung_contacts}")
    print(
        "Free insertion step to 0.20 m: "
        f"settled={insertion_response_time:.4f} s, "
        f"overshoot={response_overshoot * 1e3:.4f} mm"
    )
    print(
        "Wall barrier test: "
        f"target={barrier_target:.3f} m, stopped={stopped_insertion:.6f} m, "
        f"contact persistence={barrier_contact_fraction * 100:.2f}%, "
        f"hold drift={hold_drift * 1e3:.4f} mm, "
        f"tip speed RMS={hold_tip_speed_rms * 1e3:.4f} mm/s, "
        f"contact-count std={hold_contact_std:.3f}, "
        f"peak insertion force={peak_insertion_force:.3f} N"
    )
    print("Integrated model validation passed.")


if __name__ == "__main__":
    main()
