"""Runtime scaling of the passive bronchoscope's equivalent joint parameters."""

from __future__ import annotations

import mujoco
import numpy as np


class PassiveJointParameterController:
    """Scale passive ball-joint stiffness and damping from their MJCF values."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        joint_ids: list[int] = []
        dof_ids: list[int] = []
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            ) or ""
            if not name.startswith("cable_stiffJ"):
                continue
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_BALL:
                continue
            joint_ids.append(joint_id)
            dof_address = int(model.jnt_dofadr[joint_id])
            dof_ids.extend(range(dof_address, dof_address + 3))

        if not joint_ids:
            raise RuntimeError(
                "No passive cable_stiffJ_* ball joints were found in the model."
            )

        self.joint_ids = np.asarray(joint_ids, dtype=np.int32)
        self.qpos_ids = np.asarray(
            [int(model.jnt_qposadr[joint_id]) for joint_id in self.joint_ids],
            dtype=np.int32,
        )
        self.dof_ids = np.asarray(dof_ids, dtype=np.int32)
        self.base_stiffness = model.jnt_stiffness[self.joint_ids].copy()
        self.base_damping = model.dof_damping[self.dof_ids].copy()
        self.stiffness_scale = 1.0
        self.damping_scale = 1.0
        self.debug_locked = False
        self.guide_joint_offsets = None
        self.guide_front_x = None
        self.guide_slider_qpos = None
        reference = mujoco.MjData(model)
        mujoco.mj_forward(model, reference)
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "slid_base")
        slider = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slid_M")
        if base >= 0 and slider >= 0:
            rotation = reference.xmat[base].reshape(3, 3)
            origin = reference.xpos[base]
            front = []
            for geom in range(model.ngeom):
                if model.geom_bodyid[geom] != base or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mesh = model.geom_dataid[geom]
                start = model.mesh_vertadr[mesh]
                vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
                world = vertices @ reference.geom_xmat[geom].reshape(3, 3).T + reference.geom_xpos[geom]
                front.append(float(np.max(((world - origin) @ rotation)[:, 0])))
            if front:
                self.guide_front_x = max(front)
                self.guide_slider_qpos = int(model.jnt_qposadr[slider])
                self.guide_joint_offsets = ((reference.xanchor[self.joint_ids] - origin) @ rotation)[:, 0]

    def enforce_base_guide(self, data: mujoco.MjData) -> None:
        """Project joints inside the rigid guide onto their straight state.

        Use undeformed axial station plus insertion so a bent exposed tip
        cannot accidentally re-lock itself by pointing back towards the base.
        """
        if self.debug_locked:
            self.enforce_debug_lock(data)
            return
        if self.guide_joint_offsets is None:
            return
        inside = self.guide_joint_offsets + data.qpos[self.guide_slider_qpos] <= self.guide_front_x
        for index in np.flatnonzero(inside):
            qpos = self.qpos_ids[index]
            dof = int(self.model.jnt_dofadr[self.joint_ids[index]])
            data.qpos[qpos:qpos + 4] = (1.0, 0.0, 0.0, 0.0)
            data.qvel[dof:dof + 3] = 0.0
            data.qacc[dof:dof + 3] = 0.0
            data.qacc_warmstart[dof:dof + 3] = 0.0
            data.qfrc_applied[dof:dof + 3] = 0.0

    def enforce_debug_lock(self, data: mujoco.MjData) -> None:
        """Keep every passive ball joint exactly straight while debugging."""
        if not self.debug_locked:
            return
        for qpos_id in self.qpos_ids:
            data.qpos[qpos_id:qpos_id + 4] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[self.dof_ids] = 0.0
        data.qacc[self.dof_ids] = 0.0
        data.qacc_warmstart[self.dof_ids] = 0.0

    def set_debug_lock(
        self, data: mujoco.MjData, enabled: bool
    ) -> dict[str, bool | int]:
        """Enable a hard straight-joint lock without changing model tuning."""
        self.debug_locked = bool(enabled)
        if self.debug_locked:
            self.enforce_debug_lock(data)
            mujoco.mj_forward(self.model, data)
        return {
            "enabled": self.debug_locked,
            "joint_count": int(self.joint_ids.size),
        }

    def set_scales(
        self, stiffness_scale: float, damping_scale: float
    ) -> dict[str, float | int]:
        """Apply positive multipliers and return the effective mean values."""
        stiffness_scale = float(stiffness_scale)
        damping_scale = float(damping_scale)
        if not np.isfinite(stiffness_scale) or stiffness_scale <= 0:
            raise ValueError("Passive joint stiffness scale must be greater than 0.")
        if not np.isfinite(damping_scale) or damping_scale <= 0:
            raise ValueError("Passive joint damping scale must be greater than 0.")

        self.model.jnt_stiffness[self.joint_ids] = (
            self.base_stiffness * stiffness_scale
        )
        self.model.dof_damping[self.dof_ids] = self.base_damping * damping_scale
        self.stiffness_scale = stiffness_scale
        self.damping_scale = damping_scale
        return self.current_parameters()

    def current_parameters(self) -> dict[str, float | int]:
        return {
            "joint_count": int(self.joint_ids.size),
            "stiffness_scale": self.stiffness_scale,
            "damping_scale": self.damping_scale,
            "stiffness": float(
                np.mean(self.model.jnt_stiffness[self.joint_ids])
            ),
            "damping": float(np.mean(self.model.dof_damping[self.dof_ids])),
        }

    def reset(self) -> dict[str, float | int]:
        return self.set_scales(1.0, 1.0)


class PassiveActiveFollowerController:
    """Distribute measured proximal active curvature into the passive tip."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        follow_ratio: float = 0.55,
        coupling_stiffness: float = 18.0,
        coupling_damping: float = 0.35,
        maximum_torque: float = 2.5,
        filter_time_constant_s: float = 0.025,
    ) -> None:
        self.model = model
        self.data = data
        self.follow_ratio = float(follow_ratio)
        self.coupling_stiffness = float(coupling_stiffness)
        self.coupling_damping = float(coupling_damping)
        self.maximum_torque = float(maximum_torque)
        self.filter_time_constant_s = float(filter_time_constant_s)
        self.weights = np.asarray([0.10, 0.13, 0.16, 0.18, 0.20, 0.23])
        joint_names = [
            *(f"cable_stiffJ_{index}" for index in range(24, 29)),
            "cable_stiffJ_last",
        ]
        self.joint_ids = np.asarray([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
        ], dtype=np.int32)
        if np.any(self.joint_ids < 0):
            raise RuntimeError("Passive follower joints are missing from the model.")
        self.qpos_ids = np.asarray(
            [int(model.jnt_qposadr[joint_id]) for joint_id in self.joint_ids],
            dtype=np.int32,
        )
        self.dof_ids = np.asarray(
            [int(model.jnt_dofadr[joint_id]) for joint_id in self.joint_ids],
            dtype=np.int32,
        )
        self.maximum_joint_angles = model.jnt_range[self.joint_ids, 1].copy()
        self.base_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "active_tdcr_base"
        )
        self.proximal_end_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "seg2_body"
        )
        if self.base_body_id < 0 or self.proximal_end_body_id < 0:
            raise RuntimeError("Active proximal reference bodies are missing.")
        self.filtered_active_rotation = np.zeros(3, dtype=float)
        self.last_target_angles = np.zeros(6, dtype=float)

    def reset(self) -> None:
        """Clear coupling history when passive debug locking changes state."""
        self.filtered_active_rotation[:] = 0.0
        self.last_target_angles[:] = 0.0

    @staticmethod
    def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
        quaternion = np.zeros(4, dtype=float)
        rotation_vector = np.zeros(3, dtype=float)
        mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=float).reshape(9))
        mujoco.mju_quat2Vel(rotation_vector, quaternion, 1.0)
        return rotation_vector

    @staticmethod
    def _quaternion_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
        rotation_vector = np.zeros(3, dtype=float)
        mujoco.mju_quat2Vel(
            rotation_vector,
            np.asarray(quaternion, dtype=float),
            1.0,
        )
        return rotation_vector

    def apply(self) -> None:
        base_rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        end_rotation = self.data.xmat[self.proximal_end_body_id].reshape(3, 3)
        active_rotation = self._rotation_vector(base_rotation.T @ end_rotation)
        # The continuum bends about local Y/Z. Suppress numerical roll so the
        # passive follower cannot twist around its longitudinal axis.
        active_rotation[0] = 0.0
        alpha = float(np.clip(
            self.model.opt.timestep
            / (self.filter_time_constant_s + self.model.opt.timestep),
            0.0,
            1.0,
        ))
        self.filtered_active_rotation += alpha * (
            active_rotation - self.filtered_active_rotation
        )
        total_target = self.filtered_active_rotation * self.follow_ratio

        for index, (qpos_id, dof_id, weight, maximum_angle) in enumerate(zip(
            self.qpos_ids,
            self.dof_ids,
            self.weights,
            self.maximum_joint_angles,
        )):
            desired_rotation = total_target * weight
            desired_angle = float(np.linalg.norm(desired_rotation))
            if desired_angle > maximum_angle > 0.0:
                desired_rotation *= maximum_angle / desired_angle
                desired_angle = float(maximum_angle)
            current_rotation = self._quaternion_rotation_vector(
                self.data.qpos[qpos_id:qpos_id + 4]
            )
            torque = (
                self.coupling_stiffness * (desired_rotation - current_rotation)
                - self.coupling_damping * self.data.qvel[dof_id:dof_id + 3]
            )
            torque_norm = float(np.linalg.norm(torque))
            if torque_norm > self.maximum_torque:
                torque *= self.maximum_torque / torque_norm
            self.data.qfrc_applied[dof_id:dof_id + 3] += torque
            self.last_target_angles[index] = desired_angle

    def current_state(self) -> dict[str, object]:
        return {
            "active_bend_deg": float(
                np.degrees(np.linalg.norm(self.filtered_active_rotation))
            ),
            "target_joint_angles_deg": np.degrees(
                self.last_target_angles
            ).tolist(),
        }
