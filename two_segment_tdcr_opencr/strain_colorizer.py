"""Real-time strain/load heat colouring for the active TDCR sections."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover - the host reports a clearer error later
    mujoco = None


_ELEMENT_RE = re.compile(r"^section_(\d+)_element_(\d+)$")

# Blue -> cyan -> green -> yellow -> red, close to the stress plot convention.
_COLOR_STOPS = np.asarray(
    [
        [0.00, 0.18, 1.00],
        [0.00, 0.82, 1.00],
        [0.08, 0.88, 0.30],
        [1.00, 0.82, 0.00],
        [1.00, 0.08, 0.00],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class _JointProbe:
    qpos_indices: tuple[int, int]
    dof_indices: tuple[int, int]
    bend_scale: float
    moment_scale: float


@dataclass(frozen=True)
class _ElementProbe:
    section: int
    geom_id: int
    joint_indices: tuple[int, ...]


def _heat_color(level: float) -> np.ndarray:
    level = float(np.clip(level, 0.0, 1.0))
    position = level * (_COLOR_STOPS.shape[0] - 1)
    lower = min(int(position), _COLOR_STOPS.shape[0] - 2)
    fraction = position - lower
    return (1.0 - fraction) * _COLOR_STOPS[lower] + fraction * _COLOR_STOPS[lower + 1]


class ContinuumStrainColorizer:
    """Colour active TDCR elements from local bend and generalized moment.

    ``mode`` may be ``"bend"``, ``"force"`` or ``"combined"``.  Combined
    mode uses the larger of the normalized bend and load signals, so a highly
    loaded straight section and a freely bent section are both visible.
    """

    VALID_MODES = {"bend", "force", "combined"}

    def __init__(self, model, data, mode: str = "combined", smoothing: float = 0.24):
        if mujoco is None:
            raise RuntimeError("MuJoCo is required for continuum heat colouring.")
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported colour mode: {mode}")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1].")

        self.model = model
        self.data = data
        self.mode = mode
        self.smoothing = float(smoothing)
        self.joints_by_section: dict[int, list[_JointProbe]] = {}
        self.elements: list[_ElementProbe] = []
        self.levels = np.empty(0, dtype=np.float64)
        self._discover_model_layout()

    @property
    def element_count(self) -> int:
        return len(self.elements)

    def _object_id(self, object_type, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, object_type, name))

    def _discover_model_layout(self) -> None:
        elements_by_section: dict[int, list[tuple[int, int]]] = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            match = _ELEMENT_RE.match(name or "")
            if match:
                section, element = map(int, match.groups())
                elements_by_section.setdefault(section, []).append((element, geom_id))

        for section, section_elements in sorted(elements_by_section.items()):
            joint_probes: list[_JointProbe] = []
            joint_index = 0
            while True:
                joint_ids = tuple(
                    self._object_id(
                        mujoco.mjtObj.mjOBJ_JOINT,
                        f"section_{section}_joint_{joint_index}_{axis}",
                    )
                    for axis in ("y", "z")
                )
                if min(joint_ids) < 0:
                    break

                qpos_indices = tuple(
                    int(self.model.jnt_qposadr[joint_id]) for joint_id in joint_ids
                )
                dof_indices = tuple(
                    int(self.model.jnt_dofadr[joint_id]) for joint_id in joint_ids
                )
                bend_scale = max(
                    max(abs(float(value)) for value in self.model.jnt_range[joint_id])
                    for joint_id in joint_ids
                )
                moment_scale = max(
                    float(self.model.jnt_stiffness[joint_id]) * bend_scale
                    for joint_id in joint_ids
                )
                joint_probes.append(
                    _JointProbe(
                        qpos_indices=qpos_indices,
                        dof_indices=dof_indices,
                        bend_scale=max(bend_scale, 1e-9),
                        moment_scale=max(moment_scale, 1e-9),
                    )
                )
                joint_index += 1

            if not joint_probes:
                continue
            self.joints_by_section[section] = joint_probes

            last_joint = len(joint_probes) - 1
            for element_index, geom_id in sorted(section_elements):
                if element_index <= 0:
                    neighbours = (0,)
                elif element_index >= len(joint_probes):
                    neighbours = (last_joint,)
                else:
                    neighbours = (element_index - 1, element_index)
                self.elements.append(
                    _ElementProbe(
                        section=section,
                        geom_id=geom_id,
                        joint_indices=neighbours,
                    )
                )

        if not self.elements:
            raise RuntimeError("No section_*_element_* TDCR geoms were found.")
        self.levels = np.zeros(len(self.elements), dtype=np.float64)

    def _joint_level(self, probe: _JointProbe) -> float:
        bend = float(np.linalg.norm(self.data.qpos[list(probe.qpos_indices)]))
        bend_level = bend / probe.bend_scale

        dofs = list(probe.dof_indices)
        generalized_moment = (
            np.asarray(self.data.qfrc_actuator[dofs], dtype=np.float64)
            + np.asarray(self.data.qfrc_constraint[dofs], dtype=np.float64)
            + np.asarray(self.data.qfrc_applied[dofs], dtype=np.float64)
        )
        force_level = float(np.linalg.norm(generalized_moment)) / probe.moment_scale

        if self.mode == "bend":
            return float(np.clip(bend_level, 0.0, 1.0))
        if self.mode == "force":
            return float(np.clip(force_level, 0.0, 1.0))
        return float(np.clip(max(bend_level, force_level), 0.0, 1.0))

    def update(self, immediate: bool = False) -> np.ndarray:
        joint_levels = {
            section: np.asarray(
                [self._joint_level(probe) for probe in probes], dtype=np.float64
            )
            for section, probes in self.joints_by_section.items()
        }

        targets = np.asarray(
            [
                float(
                    np.mean(
                        joint_levels[element.section][list(element.joint_indices)]
                    )
                )
                for element in self.elements
            ],
            dtype=np.float64,
        )
        if immediate:
            self.levels[:] = targets
        else:
            self.levels += self.smoothing * (targets - self.levels)

        for element, level in zip(self.elements, self.levels):
            self.model.geom_rgba[element.geom_id, :3] = _heat_color(level)
            self.model.geom_rgba[element.geom_id, 3] = 1.0
        return self.levels.copy()
