import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


@dataclass
class NavEnvConfig:
    xml_path: str
    max_episode_steps: int = 400
    frame_skip: int = 10
    bend_gain: float = 0.10
    planar_bend_gain: float = 20.0
    insert_gain: float = 0.01
    success_threshold_m: float = 0.004
    target_points: Optional[List[Sequence[float]]] = None
    render_camera: Optional[str] = "tip_camera"


def _default_xml_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "meshes", "cable_robot_bronch_final_seg2.xml")
    )


def _default_targets() -> List[List[float]]:
    # 默认的“肺部前方”目标点，可按你的模型再调整
    return [
        [-0.30, 0.00, 1.10],
        [-0.28, 0.02, 1.11],
        [-0.28, -0.02, 1.11],
    ]


def load_target_points(path: Optional[str]) -> List[List[float]]:
    if path is None:
        return _default_targets()
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    points = payload.get("target_points", [])
    if not points:
        raise ValueError("target_points 为空")
    return points


class BronchoscopeNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: Optional[NavEnvConfig] = None,
        render_mode: Optional[str] = None,
        fixed_target: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.config = config or NavEnvConfig(xml_path=_default_xml_path())
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        self.data = mujoco.MjData(self.model)

        self.tip_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_6")
        if self.tip_body_id < 0:
            raise RuntimeError("未找到 body: end_6")

        self.slider_act_id = self._actuator_id("act_slid_M")
        self.bend_mode, self.bend_act_ids = self._resolve_bend_actuators()
        self.ctrl_ids = [self.slider_act_id, *self.bend_act_ids]

        self.ctrl_min = self.model.actuator_ctrlrange[self.ctrl_ids, 0].copy()
        self.ctrl_max = self.model.actuator_ctrlrange[self.ctrl_ids, 1].copy()

        self.home_ctrl = self._infer_home_ctrl()
        self.current_ctrl = self.home_ctrl.copy()

        self.target_points = (
            [list(map(float, fixed_target))]
            if fixed_target is not None
            else [list(map(float, p)) for p in (self.config.target_points or _default_targets())]
        )
        self.target = np.array(self.target_points[0], dtype=np.float64)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # tip_pos(3), tip_vel(3), target(3), delta(3), current_ctrl(nu_selected), last_action(3)
        obs_dim = 12 + len(self.current_ctrl) + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._renderer = None
        self._step_count = 0
        self._last_dist = np.inf
        self._last_action = np.zeros(3, dtype=np.float64)
        self._render_camera = self._resolve_render_camera()

    def _actuator_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx < 0:
            raise RuntimeError(f"未找到 actuator: {name}")
        return idx

    def _try_actuator_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _resolve_bend_actuators(self) -> Tuple[str, List[int]]:
        tendon_names = [f"act_t{i}" for i in range(1, 7)]
        tendon_ids = [self._try_actuator_id(n) for n in tendon_names]
        if all(idx >= 0 for idx in tendon_ids):
            return "tendon", [int(i) for i in tendon_ids]

        up_id = self._try_actuator_id("up")
        right_id = self._try_actuator_id("right")
        if up_id >= 0 and right_id >= 0:
            return "planar", [int(up_id), int(right_id)]

        raise RuntimeError("未找到可用弯曲执行器：需要 act_t1..act_t6 或 up/right")

    def _resolve_render_camera(self):
        cam_name = self.config.render_camera
        if not cam_name:
            return None
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id >= 0:
            return cam_name
        return None

    def _infer_home_ctrl(self) -> np.ndarray:
        home = np.zeros(len(self.ctrl_ids), dtype=np.float64)
        if self.model.nkey > 0 and self.model.key_ctrl.size >= self.model.nu:
            home[:] = self.model.key_ctrl[0, self.ctrl_ids]
        else:
            home[0] = self.ctrl_min[0]
            home[1:] = self.ctrl_max[1:]
        return np.clip(home, self.ctrl_min, self.ctrl_max)

    def _tip_pos(self) -> np.ndarray:
        return self.data.xpos[self.tip_body_id].copy()

    def _tip_vel(self) -> np.ndarray:
        cvel = self.data.cvel[self.tip_body_id]
        return cvel[3:6].copy()  # 线速度

    def _select_target(self, seed: Optional[int]) -> None:
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(low=0, high=len(self.target_points)))
        self.target = np.array(self.target_points[idx], dtype=np.float64)

    def _build_obs(self) -> np.ndarray:
        tip = self._tip_pos()
        vel = self._tip_vel()
        delta = self.target - tip
        obs = np.concatenate([tip, vel, self.target, delta, self.current_ctrl, self._last_action], axis=0)
        return obs.astype(np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.clip(action, -1.0, 1.0)
        bend = action[:2].astype(np.float64)
        insertion = float(action[2])

        bend_norm = np.linalg.norm(bend)
        if bend_norm > 1.0:
            bend = bend / bend_norm

        slider_cmd = self.current_ctrl[0] + insertion * self.config.insert_gain

        self.current_ctrl[0] = np.clip(slider_cmd, self.ctrl_min[0], self.ctrl_max[0])

        if self.bend_mode == "tendon":
            angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300], dtype=np.float64))
            projection = bend[0] * np.cos(angles) + bend[1] * np.sin(angles)
            bend_cmd = self.home_ctrl[1:] - self.config.bend_gain * projection
            self.current_ctrl[1:] = np.clip(bend_cmd, self.ctrl_min[1:], self.ctrl_max[1:])
        else:
            # up/right 两个执行器分别响应 y/x 弯曲意图
            bend_cmd = self.home_ctrl[1:] + self.config.planar_bend_gain * np.array([bend[1], bend[0]], dtype=np.float64)
            self.current_ctrl[1:] = np.clip(bend_cmd, self.ctrl_min[1:], self.ctrl_max[1:])

        self.data.ctrl[self.ctrl_ids] = self.current_ctrl
        self._last_action = action.copy()

    def _contact_penalty(self) -> float:
        penalty = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            b1 = int(self.model.geom_bodyid[geom1])
            b2 = int(self.model.geom_bodyid[geom2])
            if b1 == self.tip_body_id or b2 == self.tip_body_id:
                penalty += 1.0
        return penalty

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._last_action[:] = 0.0
        self._select_target(seed=seed)

        self.current_ctrl = self.home_ctrl.copy()
        self.current_ctrl[0] = np.clip(self.current_ctrl[0] + np.random.uniform(0.0, 0.02), self.ctrl_min[0], self.ctrl_max[0])
        self.data.ctrl[self.ctrl_ids] = self.current_ctrl

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        self._last_dist = float(np.linalg.norm(self.target - self._tip_pos()))
        return self._build_obs(), {"target": self.target.copy()}

    def step(self, action):
        self._step_count += 1
        self._apply_action(np.asarray(action, dtype=np.float64))

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        tip = self._tip_pos()
        dist = float(np.linalg.norm(self.target - tip))
        progress = self._last_dist - dist
        self._last_dist = dist

        contact_pen = self._contact_penalty()
        ctrl_pen = float(np.linalg.norm(self.current_ctrl - self.home_ctrl))

        reward = (
            -4.0 * dist
            + 10.0 * progress
            - 0.03 * ctrl_pen
            - 0.02 * contact_pen
        )

        success = dist < self.config.success_threshold_m
        if success:
            reward += 20.0

        terminated = success
        truncated = self._step_count >= self.config.max_episode_steps

        info = {
            "distance_to_target": dist,
            "progress": progress,
            "contact_penalty": contact_pen,
            "is_success": success,
            "target": self.target.copy(),
        }
        return self._build_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        if self._render_camera is not None:
            self._renderer.update_scene(self.data, camera=self._render_camera)
        else:
            self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

