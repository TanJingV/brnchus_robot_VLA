"""Offline staged identification from a recorded D435/Trio session."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from .config import PROJECT_ROOT

try:
    import mujoco
except Exception:
    mujoco = None


STATIC_NAMES = [
    *(f"wire_{index}_zero_offset_m" for index in range(1, 7)),
    "proximal_transmission_scale",
    "distal_transmission_scale",
    "proximal_stiffness_scale",
    "distal_stiffness_scale",
]
DYNAMIC_NAMES = [
    "proximal_damping_scale",
    "distal_damping_scale",
    "proximal_actuator_tau_s",
    "distal_actuator_tau_s",
    "input_delay_s",
]


def huber_cost(residual: np.ndarray, delta_m: float = 0.002) -> float:
    value = np.abs(np.asarray(residual, dtype=float))
    loss = np.where(value <= delta_m, 0.5 * value**2, delta_m * (value - 0.5 * delta_m))
    return float(np.mean(loss)) if loss.size else math.inf


class SessionData:
    def __init__(self, session: Path, maximum_frames: int = 180) -> None:
        self.session = session.resolve()
        archive = np.load(self.session / "keypoints.npz")
        count = len(archive["capture_host_ns"])
        if count < 8:
            raise RuntimeError("Session contains too few frames for identification")
        indices = np.unique(np.linspace(0, count - 1, min(count, maximum_frames)).astype(int))
        self.indices = indices
        host_ns = np.asarray(archive["capture_host_ns"], dtype=np.int64)[indices]
        self.time_s = (host_ns - host_ns[0]) * 1e-9
        self.controls_m = np.asarray(archive["controls_m"], dtype=float)[indices]
        self.points_m = np.asarray(archive["keypoints_base_m"], dtype=float)[indices]
        self.valid = np.asarray(archive["keypoint_valid"], dtype=bool)[indices]
        self.confidence = np.asarray(archive["keypoint_confidence"], dtype=float)[indices]
        self.valid &= np.isfinite(self.points_m).all(axis=2)
        if np.count_nonzero(self.valid) < 28:
            raise RuntimeError("Too few valid 3D keypoints for identification")
        manifest_path = self.session / "session.json"
        with manifest_path.open("r", encoding="utf-8") as stream:
            self.manifest = json.load(stream)


class TDCRIdentifier:
    def __init__(self, data: SessionData, xml_path: Path) -> None:
        if mujoco is None:
            raise RuntimeError("MuJoCo is required for parameter identification")
        self.session = data
        self.xml_path = xml_path.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.site_ids = np.asarray([
            self._id(mujoco.mjtObj.mjOBJ_SITE, f"measurement_kp_{index}") for index in range(7)
        ], dtype=int)
        self.base_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "base_frame")
        self.tendon_ids = np.asarray([
            self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_t{index}") for index in range(1, 7)
        ], dtype=int)
        self.slider_id = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "act_slid_M")
        self.section_joint_ids = {
            section: np.asarray([
                joint_id for joint_id in range(self.model.njnt)
                if (self._name(mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").startswith(f"section_{section}_joint_")
            ], dtype=int)
            for section in (1, 2)
        }
        self.baseline_stiffness = self.model.jnt_stiffness.copy()
        self.baseline_damping = self.model.dof_damping.copy()
        self.neutral = np.mean(self.model.actuator_ctrlrange[self.tendon_ids], axis=1)
        self.static_parameters = np.asarray([0.0] * 6 + [1.0, 1.0, 1.0, 1.0])
        self.dynamic_parameters = np.asarray([1.0, 1.0, 0.015, 0.015, 0.0])

    def _id(self, object_type, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, object_type, name)
        if identifier < 0:
            raise RuntimeError(f"MuJoCo model is missing {name}")
        return int(identifier)

    def _name(self, object_type, identifier: int) -> str | None:
        return mujoco.mj_id2name(self.model, object_type, int(identifier))

    def _set_joint_scale(self, stiffness_scales: tuple[float, float], damping_scales: tuple[float, float]) -> None:
        self.model.jnt_stiffness[:] = self.baseline_stiffness
        self.model.dof_damping[:] = self.baseline_damping
        for section, stiffness_scale, damping_scale in zip((1, 2), stiffness_scales, damping_scales):
            for joint_id in self.section_joint_ids[section]:
                self.model.jnt_stiffness[joint_id] = self.baseline_stiffness[joint_id] * stiffness_scale
                dof_address = int(self.model.jnt_dofadr[joint_id])
                dof_count = 1 if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE else 0
                if dof_count:
                    self.model.dof_damping[dof_address] = self.baseline_damping[dof_address] * damping_scale

    def _correct_controls(self, controls: np.ndarray, static: np.ndarray) -> np.ndarray:
        result = np.asarray(controls, dtype=float).copy()
        offsets = static[:6]
        scales = np.repeat(static[6:8], 3)
        result[:, :6] = self.neutral + scales * (result[:, :6] - self.neutral) + offsets
        lower = self.model.actuator_ctrlrange[self.tendon_ids, 0]
        upper = self.model.actuator_ctrlrange[self.tendon_ids, 1]
        result[:, :6] = np.clip(result[:, :6], lower, upper)
        return result

    def _delayed_controls(self, controls: np.ndarray, delay_s: float) -> np.ndarray:
        delayed_time = np.maximum(self.session.time_s - delay_s, self.session.time_s[0])
        result = np.empty_like(controls)
        for axis in range(7):
            result[:, axis] = np.interp(delayed_time, self.session.time_s, controls[:, axis])
        return result

    def _points_base(self) -> np.ndarray:
        base_position = self.data.site_xpos[self.base_site_id].copy()
        base_rotation = self.data.site_xmat[self.base_site_id].reshape(3, 3).copy()
        world = self.data.site_xpos[self.site_ids].copy()
        return (base_rotation.T @ (world - base_position).T).T

    def simulate(self, static: np.ndarray | None = None, dynamic: np.ndarray | None = None) -> np.ndarray:
        static = self.static_parameters if static is None else np.asarray(static, dtype=float)
        dynamic = self.dynamic_parameters if dynamic is None else np.asarray(dynamic, dtype=float)
        self._set_joint_scale((static[8], static[9]), (dynamic[0], dynamic[1]))
        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        controls = self._correct_controls(self.session.controls_m, static)
        controls = self._delayed_controls(controls, max(0.0, float(dynamic[4])))
        filtered = controls[0].copy()
        output = np.empty((len(controls), 7, 3), dtype=float)
        last_time = float(self.session.time_s[0])
        for index, (sample_time, target) in enumerate(zip(self.session.time_s, controls)):
            interval = max(float(sample_time - last_time), self.model.opt.timestep)
            last_time = float(sample_time)
            steps = max(1, int(round(interval / self.model.opt.timestep)))
            dt = interval / steps
            for _ in range(steps):
                for group, tau in ((slice(0, 3), dynamic[2]), (slice(3, 6), dynamic[3])):
                    alpha = 1.0 if tau <= 1e-5 else 1.0 - math.exp(-dt / tau)
                    filtered[group] += alpha * (target[group] - filtered[group])
                filtered[6] = target[6]
                self.data.ctrl[self.tendon_ids] = filtered[:6]
                self.data.ctrl[self.slider_id] = np.clip(
                    filtered[6],
                    self.model.actuator_ctrlrange[self.slider_id, 0],
                    self.model.actuator_ctrlrange[self.slider_id, 1],
                )
                mujoco.mj_step(self.model, self.data)
            output[index] = self._points_base()
        return output

    def residual(self, prediction: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
        valid = self.session.valid & frame_mask[:, None]
        confidence = np.clip(self.session.confidence, 0.05, 1.0)
        difference = (prediction - self.session.points_m) * np.sqrt(confidence)[..., None]
        return difference[valid].reshape(-1)

    def _observability(self, parameter: np.ndarray, residual_function) -> dict[str, Any]:
        baseline = residual_function(parameter)
        jacobian = np.empty((len(baseline), len(parameter)), dtype=float)
        for column in range(len(parameter)):
            step = max(abs(parameter[column]) * 1e-3, 1e-6)
            shifted = parameter.copy()
            shifted[column] += step
            jacobian[:, column] = (residual_function(shifted) - baseline) / step
        norms = np.linalg.norm(jacobian, axis=0)
        active = norms > max(np.max(norms) * 1e-5, 1e-9)
        condition = float(np.linalg.cond(jacobian[:, active])) if np.any(active) else math.inf
        return {"column_norms": norms.tolist(), "observable": active.tolist(), "condition_number": condition}

    def fit_static(self, train_mask: np.ndarray, maxiter: int) -> dict[str, Any]:
        bounds = [(-0.0015, 0.0015)] * 6 + [(0.70, 1.30), (0.70, 1.30), (0.40, 2.50), (0.40, 2.50)]

        def residual(value):
            return self.residual(self.simulate(static=value), train_mask)

        global_result = differential_evolution(
            lambda value: huber_cost(residual(value)), bounds, maxiter=maxiter,
            popsize=6, polish=False, seed=7, workers=1, updating="immediate",
        )
        local = least_squares(
            residual, global_result.x,
            bounds=(np.asarray([item[0] for item in bounds]), np.asarray([item[1] for item in bounds])),
            loss="huber", f_scale=0.002, max_nfev=max(20, maxiter * 3),
        )
        self.static_parameters = local.x.copy()
        return {
            "names": STATIC_NAMES,
            "values": local.x.tolist(),
            "cost": float(local.cost),
            "global_cost": float(global_result.fun),
            "observability": self._observability(local.x, residual),
        }

    def fit_dynamic(self, train_mask: np.ndarray, maxiter: int) -> dict[str, Any]:
        bounds = [(0.30, 3.0), (0.30, 3.0), (0.0, 0.15), (0.0, 0.15), (0.0, 0.20)]

        def residual(value):
            return self.residual(self.simulate(dynamic=value), train_mask)

        global_result = differential_evolution(
            lambda value: huber_cost(residual(value)), bounds, maxiter=maxiter,
            popsize=6, polish=False, seed=11, workers=1, updating="immediate",
        )
        local = least_squares(
            residual, global_result.x,
            bounds=(np.asarray([item[0] for item in bounds]), np.asarray([item[1] for item in bounds])),
            loss="huber", f_scale=0.002, max_nfev=max(20, maxiter * 3),
        )
        self.dynamic_parameters = local.x.copy()
        return {
            "names": DYNAMIC_NAMES,
            "values": local.x.tolist(),
            "cost": float(local.cost),
            "global_cost": float(global_result.fun),
            "observability": self._observability(local.x, residual),
        }

    def evaluate(self, frame_mask: np.ndarray) -> dict[str, float]:
        prediction = self.simulate()
        valid = self.session.valid & frame_mask[:, None]
        errors_mm = np.linalg.norm(prediction - self.session.points_m, axis=2)[valid] * 1000.0
        return {
            "samples": int(len(errors_mm)),
            "rmse_mm": float(np.sqrt(np.mean(errors_mm**2))),
            "median_mm": float(np.median(errors_mm)),
            "p95_mm": float(np.percentile(errors_mm, 95)),
        }


def write_results(session: Path, static: dict, dynamic: dict, train: dict, validation: dict, xml_path: Path) -> Path:
    profile = {
        "schema_version": 1,
        "source_session": str(session),
        "source_model": str(xml_path),
        "static": dict(zip(static["names"], static["values"])),
        "dynamic": dict(zip(dynamic["names"], dynamic["values"])),
        "observability": {"static": static["observability"], "dynamic": dynamic["observability"]},
        "metrics": {"train": train, "validation": validation},
        "applies_automatically": False,
    }
    profile_path = session / "identified_profile.json"
    with profile_path.open("w", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2, ensure_ascii=False)
    report = [
        "# D435–MuJoCo 参数辨识报告", "",
        f"- 数据会话：`{session}`", f"- 模型：`{xml_path}`", "",
        "## 验证误差", "",
        f"- 训练 RMSE：{train['rmse_mm']:.3f} mm", f"- 验证 RMSE：{validation['rmse_mm']:.3f} mm",
        f"- 验证中位误差：{validation['median_mm']:.3f} mm", f"- 验证 P95：{validation['p95_mm']:.3f} mm", "",
        "## 静态参数", "",
        *[f"- `{name}` = {value:.8g}" for name, value in zip(static["names"], static["values"])], "",
        f"静态灵敏度条件数：{static['observability']['condition_number']:.6g}", "",
        "## 动态参数", "",
        *[f"- `{name}` = {value:.8g}" for name, value in zip(dynamic["names"], dynamic["values"])], "",
        f"动态灵敏度条件数：{dynamic['observability']['condition_number']:.6g}", "",
        "> 该配置不会自动覆盖基础XML；请先检查验证集误差和可观测性。", "",
    ]
    (session / "identification_report.md").write_text("\n".join(report), encoding="utf-8")
    return profile_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=PROJECT_ROOT / "meshes" / "cable_robot_bronch_final_seg2.xml")
    parser.add_argument("--maximum-frames", type=int, default=180)
    parser.add_argument("--maxiter", type=int, default=10)
    args = parser.parse_args(argv)
    session_data = SessionData(args.session, args.maximum_frames)
    identifier = TDCRIdentifier(session_data, args.xml)
    validation_mask = np.arange(len(session_data.time_s)) % 5 == 0
    train_mask = ~validation_mask
    baseline_validation = identifier.evaluate(validation_mask)
    print(f"Baseline validation RMSE: {baseline_validation['rmse_mm']:.3f} mm")
    static = identifier.fit_static(train_mask, args.maxiter)
    dynamic = identifier.fit_dynamic(train_mask, args.maxiter)
    train = identifier.evaluate(train_mask)
    validation = identifier.evaluate(validation_mask)
    profile = write_results(args.session.resolve(), static, dynamic, train, validation, args.xml.resolve())
    improvement = 100.0 * (baseline_validation["rmse_mm"] - validation["rmse_mm"]) / max(baseline_validation["rmse_mm"], 1e-9)
    print(f"Validation RMSE: {validation['rmse_mm']:.3f} mm ({improvement:+.1f}% improvement)")
    print(f"Wrote: {profile}")


if __name__ == "__main__":
    main()
