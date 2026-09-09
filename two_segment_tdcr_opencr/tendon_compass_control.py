"""Map two planar compass commands to six physical tendon length targets."""

from __future__ import annotations

import math
from typing import Sequence

import mujoco
import numpy as np


class TendonCompassController:
    """Drive Wire 1--3 proximally and Wire 4--6 distally by tendon length."""

    WIRE_GROUPS = ((1, 2, 3), (4, 5, 6))
    WIRE_ANGLES_DEG = ((0.0, 120.0, 240.0), (60.0, 180.0, 300.0))

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        tendon_radius_m: float = 0.00154,
        max_bend_deg: float = 160.0,
        home_key_name: str = "home",
    ) -> None:
        self.model = model
        self.data = data
        self.tendon_radius_m = float(tendon_radius_m)
        self.max_bend_deg = float(max_bend_deg)
        # Wires 4--6 traverse both active sections, so their motor command must
        # accommodate proximal path compensation plus the distal bend itself.
        self.segment_command_limits = np.asarray([1.0, 2.0], dtype=float)
        self.actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_t{wire}"
                )
                for wire in range(1, 7)
            ],
            dtype=np.int32,
        )
        if np.any(self.actuator_ids < 0):
            missing = [
                str(wire)
                for wire, actuator_id in enumerate(self.actuator_ids, start=1)
                if actuator_id < 0
            ]
            raise RuntimeError(f"Missing tendon actuators act_t{','.join(missing)}.")

        key_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, home_key_name
        )
        if key_id >= 0:
            baseline = np.asarray(
                model.key_ctrl[key_id, self.actuator_ids], dtype=float
            )
        else:
            ranges = np.asarray(model.actuator_ctrlrange[self.actuator_ids])
            baseline = np.mean(ranges, axis=1)
        self.baseline_lengths = baseline.copy()
        self.target_lengths = baseline.copy()
        self.segment_vectors = np.zeros((2, 2), dtype=float)
        self.data.ctrl[self.actuator_ids] = self.target_lengths

    @staticmethod
    def _clamp_vector(
        vector_xy: Sequence[float], maximum_norm: float = 1.0
    ) -> np.ndarray:
        vector = np.asarray(vector_xy, dtype=float).reshape(2)
        magnitude = float(np.linalg.norm(vector))
        if magnitude > maximum_norm:
            vector = vector * (maximum_norm / magnitude)
        return vector

    def clamp_segment_vector(
        self, segment_index: int, vector_xy: Sequence[float]
    ) -> np.ndarray:
        if segment_index not in (0, 1):
            raise ValueError("segment_index must be 0 (proximal) or 1 (distal).")
        return self._clamp_vector(
            vector_xy, float(self.segment_command_limits[segment_index])
        )

    def lengths_for_vector(
        self, segment_index: int, vector_xy: Sequence[float]
    ) -> np.ndarray:
        """Return absolute tendon lengths for one compass vector.

        Compass +X is the 0-degree bending direction and +Y is 90 degrees.
        Radial magnitude maps linearly to the commanded bending angle.
        """
        if segment_index not in (0, 1):
            raise ValueError("segment_index must be 0 (proximal) or 1 (distal).")
        vector = self.clamp_segment_vector(segment_index, vector_xy)
        magnitude = float(np.linalg.norm(vector))
        direction = math.atan2(float(vector[1]), float(vector[0]))
        bend_angle = math.radians(self.max_bend_deg * magnitude)
        wire_angles = np.radians(self.WIRE_ANGLES_DEG[segment_index])
        wire_slice = slice(segment_index * 3, segment_index * 3 + 3)
        lengths = self.baseline_lengths[wire_slice] - (
            self.tendon_radius_m
            * bend_angle
            * np.cos(wire_angles - direction)
        )
        actuator_ids = self.actuator_ids[wire_slice]
        lower = self.model.actuator_ctrlrange[actuator_ids, 0]
        upper = self.model.actuator_ctrlrange[actuator_ids, 1]
        return np.clip(lengths, lower, upper)

    def set_segment_vector(
        self, segment_index: int, vector_xy: Sequence[float]
    ) -> np.ndarray:
        vector = self.clamp_segment_vector(segment_index, vector_xy)
        wire_slice = slice(segment_index * 3, segment_index * 3 + 3)
        self.segment_vectors[segment_index] = vector
        self.target_lengths[wire_slice] = self.lengths_for_vector(
            segment_index, vector
        )
        self.data.ctrl[self.actuator_ids[wire_slice]] = self.target_lengths[
            wire_slice
        ]
        return self.target_lengths[wire_slice].copy()

    def reset(self) -> None:
        self.segment_vectors[:] = 0.0
        self.target_lengths[:] = self.baseline_lengths
        self.data.ctrl[self.actuator_ids] = self.target_lengths

    def commanded_pull_mm(self) -> np.ndarray:
        """Positive values mean shortening/pulling relative to straight."""
        return (self.baseline_lengths - self.target_lengths) * 1000.0

    def segment_state(self, segment_index: int) -> dict[str, object]:
        vector = self.segment_vectors[segment_index]
        magnitude = float(np.linalg.norm(vector))
        direction_deg = (
            math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 360.0
            if magnitude > 1e-9
            else 0.0
        )
        wire_slice = slice(segment_index * 3, segment_index * 3 + 3)
        return {
            "bend_deg": self.max_bend_deg * magnitude,
            "direction_deg": direction_deg,
            "lengths_m": self.target_lengths[wire_slice].copy(),
            "pull_mm": self.commanded_pull_mm()[wire_slice].copy(),
        }
