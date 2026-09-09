"""Dynamic MuJoCo input replay and seven-site shape comparison."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT
from .models import KEYPOINT_COUNT

try:
    import mujoco
except Exception:
    mujoco = None


class MujocoAlignmentBridge:
    def __init__(self, config: dict, model=None, data=None, owns_model: bool = True) -> None:
        if mujoco is None:
            raise RuntimeError("MuJoCo Python package is unavailable")
        self.config = config
        self.owns_model = owns_model
        if model is None or data is None:
            xml_path = Path(config["xml"])
            xml_path = (PROJECT_ROOT / xml_path).resolve() if not xml_path.is_absolute() else xml_path.resolve()
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
        self.model = model
        self.data = data
        self.base_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, config["base_site"])
        self.site_ids = np.asarray([
            self._id(mujoco.mjtObj.mjOBJ_SITE, name) for name in config["measurement_sites"]
        ], dtype=int)
        self.tendon_ids = np.asarray([
            self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in config["tendon_actuators"]
        ], dtype=int)
        self.slider_id = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, config["slider_actuator"])
        self.last_host_ns: int | None = None

    def _id(self, obj_type, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, obj_type, name)
        if identifier < 0:
            raise RuntimeError(f"MuJoCo model is missing {name}")
        return int(identifier)

    def apply_controls(self, controls_m: np.ndarray, host_ns: int | None = None, step: bool = True) -> None:
        controls = np.asarray(controls_m, dtype=float).reshape(7)
        if not np.isfinite(controls).all():
            return
        lower = self.model.actuator_ctrlrange[self.tendon_ids, 0]
        upper = self.model.actuator_ctrlrange[self.tendon_ids, 1]
        self.data.ctrl[self.tendon_ids] = np.clip(controls[:6], lower, upper)
        self.data.ctrl[self.slider_id] = float(np.clip(
            controls[6],
            self.model.actuator_ctrlrange[self.slider_id, 0],
            self.model.actuator_ctrlrange[self.slider_id, 1],
        ))
        if not step or not self.owns_model:
            return
        now = time.perf_counter_ns() if host_ns is None else int(host_ns)
        elapsed = self.model.opt.timestep if self.last_host_ns is None else float(np.clip((now - self.last_host_ns) * 1e-9, self.model.opt.timestep, 0.05))
        self.last_host_ns = now
        steps = max(1, int(round(elapsed / self.model.opt.timestep)))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def measurement_points_base_m(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        base_position = np.asarray(self.data.site_xpos[self.base_site_id], dtype=float)
        base_rotation = np.asarray(self.data.site_xmat[self.base_site_id], dtype=float).reshape(3, 3)
        world = np.asarray(self.data.site_xpos[self.site_ids], dtype=float)
        return (base_rotation.T @ (world - base_position).T).T

    def compare(self, real_points_base_m: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
        simulation = self.measurement_points_base_m()
        real = np.asarray(real_points_base_m, dtype=float).reshape(KEYPOINT_COUNT, 3)
        mask = np.asarray(valid, dtype=bool).reshape(KEYPOINT_COUNT) & np.isfinite(real).all(axis=1)
        errors = np.full(KEYPOINT_COUNT, np.nan)
        errors[mask] = np.linalg.norm(real[mask] - simulation[mask], axis=1) * 1000.0
        count = int(np.count_nonzero(mask))
        rmse = float(np.sqrt(np.mean(errors[mask] ** 2))) if count else float("nan")
        return simulation, errors, rmse, count


class ExistingMujocoBridge(MujocoAlignmentBridge):
    """View/control an already-running main-window simulator without stepping it."""

    def __init__(self, config: dict, simulator) -> None:
        self.simulator = simulator
        super().__init__(config, simulator.model, simulator.data, owns_model=False)

    def compare(self, real_points_base_m: np.ndarray, valid: np.ndarray):
        result = super().compare(real_points_base_m, valid)
        if hasattr(self.simulator, "set_tdcr_measurement_overlay"):
            self.simulator.set_tdcr_measurement_overlay(real_points_base_m, valid)
        return result
