"""Contact-only numerical regularization for the lightweight active TDCR.

The free-space joint parameters remain exactly those stored in MJCF.  During
real active-tip/bronchial-wall contact, a small temporary armature is enabled
to keep the rigid non-convex constraint from exciting unresolved, sub-step
joint acceleration.  No position, velocity, control target, or contact state
is clamped or reset.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np


class ActiveTipContactRegularizer:
    """Regularize only active-tip contacts without slowing free bending."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        flex_name: str = "bronchial_wall_nonconvex",
        active_body_name: str = "active_tdcr_base",
        contact_armature: float = 1e-4,
        release_delay_s: float = 0.03,
    ) -> None:
        self.model = model
        self.data = data
        self.flex_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_FLEX, flex_name
        )
        active_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, active_body_name
        )

        self.dof_ids = np.asarray(
            [
                int(model.jnt_dofadr[joint_id])
                for joint_id in range(model.njnt)
                if (mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                ) or "").startswith("section_")
            ],
            dtype=np.int32,
        )
        self.active_geom_ids: set[int] = set()
        if active_body_id >= 0:
            for geom_id in range(model.ngeom):
                body_id = int(model.geom_bodyid[geom_id])
                while body_id > 0:
                    if body_id == active_body_id:
                        self.active_geom_ids.add(geom_id)
                        break
                    body_id = int(model.body_parentid[body_id])

        self.free_armature = model.dof_armature[self.dof_ids].copy()
        self.contact_armature = float(contact_armature)
        self.release_steps = max(
            1, math.ceil(float(release_delay_s) / float(model.opt.timestep))
        )
        self.remaining_steps = 0
        self.enabled = (
            self.flex_id >= 0
            and self.dof_ids.size > 0
            and bool(self.active_geom_ids)
        )

    def _touching_active_tip(self) -> bool:
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            if self.flex_id not in (int(contact.flex[0]), int(contact.flex[1])):
                continue
            if any(
                int(geom_id) in self.active_geom_ids
                for geom_id in contact.geom
                if int(geom_id) >= 0
            ):
                return True
        return False

    def before_step(self) -> None:
        """Update solver regularization before one ordinary dynamic step."""
        if not self.enabled:
            return
        if self._touching_active_tip():
            self.remaining_steps = self.release_steps
        elif self.remaining_steps > 0:
            self.remaining_steps -= 1

        if self.remaining_steps > 0:
            self.model.dof_armature[self.dof_ids] = self.contact_armature
        else:
            self.model.dof_armature[self.dof_ids] = self.free_armature

    def reset(self) -> None:
        """Restore the exact free-space model parameters."""
        if self.dof_ids.size:
            self.model.dof_armature[self.dof_ids] = self.free_armature
        self.remaining_steps = 0
