"""Motor-tendon PCC prior used to fill visual depth holes in legacy sessions."""

from __future__ import annotations

import math
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from Visual_information.d435_tdcr_capture.models import KEYPOINT_ARCLENGTHS_M, KEYPOINT_COUNT

from .models import SevenMarkerResult


SECTION_LENGTH_M = 0.021
TENDON_RADIUS_M = 0.00154
PROXIMAL_ANGLES_DEG = (0.0, 120.0, 240.0)
DISTAL_ANGLES_DEG = (60.0, 180.0, 300.0)
MAX_BEND_RAD = math.radians(160.0)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _section_transform(bend_vector: np.ndarray, length_m: float) -> tuple[np.ndarray, np.ndarray]:
    bend = np.asarray(bend_vector, dtype=float) * (length_m / SECTION_LENGTH_M)
    theta = float(np.linalg.norm(bend))
    if theta < 1e-9:
        return np.asarray([length_m, 0.0, 0.0]), np.eye(3)
    cos_phi, sin_phi = bend / theta
    position = np.asarray([
        length_m * math.sin(theta) / theta,
        length_m * (1.0 - math.cos(theta)) / theta * cos_phi,
        length_m * (1.0 - math.cos(theta)) / theta * sin_phi,
    ])
    axis = np.asarray([0.0, -sin_phi, cos_phi])
    axis_skew = _skew(axis)
    rotation = (
        np.eye(3)
        + math.sin(theta) * axis_skew
        + (1.0 - math.cos(theta)) * (axis_skew @ axis_skew)
    )
    return position, rotation


def _bend_from_lengths(lengths_m: np.ndarray, angles_deg: tuple[float, ...]) -> np.ndarray:
    lengths = np.asarray(lengths_m, dtype=float)
    if not np.isfinite(lengths).all():
        return np.full(2, np.nan)
    differential = lengths - float(np.mean(lengths))
    angles = np.radians(np.asarray(angles_deg, dtype=float))
    design = np.column_stack((np.cos(angles), np.sin(angles)))
    bend, *_ = np.linalg.lstsq(design, -differential / TENDON_RADIUS_M, rcond=None)
    magnitude = float(np.linalg.norm(bend))
    if magnitude > MAX_BEND_RAD:
        bend *= MAX_BEND_RAD / magnitude
    return bend


def motor_controls_to_local_points(controls_m: np.ndarray) -> np.ndarray:
    """Map six absolute tendon lengths to K0..K6 in the active-base frame."""

    controls = np.asarray(controls_m, dtype=float).reshape(7)
    proximal = _bend_from_lengths(controls[:3], PROXIMAL_ANGLES_DEG)
    total_distal_route = _bend_from_lengths(controls[3:6], DISTAL_ANGLES_DEG)
    if not np.isfinite(proximal).all() or not np.isfinite(total_distal_route).all():
        return np.full((KEYPOINT_COUNT, 3), np.nan)
    # Wires 4-6 run through the proximal section before reaching the distal
    # section.  Their differential length therefore contains both bends.
    distal = total_distal_route - proximal
    distal_norm = float(np.linalg.norm(distal))
    if distal_norm > MAX_BEND_RAD:
        distal *= MAX_BEND_RAD / distal_norm
    points = np.zeros((KEYPOINT_COUNT, 3), dtype=float)
    interface_position, interface_rotation = _section_transform(proximal, SECTION_LENGTH_M)
    for index, arclength in enumerate(KEYPOINT_ARCLENGTHS_M):
        if arclength <= SECTION_LENGTH_M + 1e-12:
            points[index] = _section_transform(proximal, float(arclength))[0]
        else:
            distal_position, _ = _section_transform(distal, float(arclength - SECTION_LENGTH_M))
            points[index] = interface_position + interface_rotation @ distal_position
    # Axis 6 is the insertion coordinate.  Keeping it in the same local frame
    # lets one camera alignment profile remain valid across insertion depths.
    if np.isfinite(controls[6]):
        points[:, 0] += float(controls[6])
    return points


def _kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(float(np.sum(weights)), 1e-9)
    source_center = np.sum(weights[:, None] * source, axis=0)
    target_center = np.sum(weights[:, None] * target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (weights[:, None] * source_zero).T @ target_zero
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


class MotorShapePriorFusion:
    def __init__(self, enabled: bool = True, alignment_profile: str | Path | None = None) -> None:
        self.enabled = bool(enabled)
        self.rotation: np.ndarray | None = None
        self.translation: np.ndarray | None = None
        self.alignment_updates = 0
        self.alignment_residual_m = float("nan")
        self._alignment_source: list[np.ndarray] = []
        self._alignment_target: list[np.ndarray] = []
        self._alignment_weight: list[np.ndarray] = []
        self.profile_path: Path | None = None
        if alignment_profile:
            self.load_alignment_profile(alignment_profile)

    def load_alignment_profile(self, path: str | Path) -> bool:
        profile_path = Path(path).expanduser().resolve()
        if not profile_path.exists():
            return False
        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
            rotation = np.asarray(payload["rotation_camera_from_motor"], dtype=float).reshape(3, 3)
            translation = np.asarray(payload["translation_camera_from_motor_m"], dtype=float).reshape(3)
            if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                return False
            u, _singular, vt = np.linalg.svd(rotation)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1
                rotation = u @ vt
            self.rotation = rotation
            self.translation = translation
            self.profile_path = profile_path
            self.alignment_updates = int(payload.get("alignment_updates", 0))
            self.alignment_residual_m = float(payload.get("median_residual_m", np.nan))
            return True
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def save_alignment_profile(self, path: str | Path, source_session: str = "") -> Path:
        if self.rotation is None or self.translation is None:
            raise ValueError("Motor-camera alignment has not been observed")
        rotation, translation = self.rotation, self.translation
        residual = self.alignment_residual_m
        if self._alignment_source:
            source = np.vstack(self._alignment_source)
            target = np.vstack(self._alignment_target)
            weight = np.concatenate(self._alignment_weight)
            initial_rotation, initial_translation = _kabsch(source, target, weight)
            error = np.linalg.norm(source @ initial_rotation.T + initial_translation - target, axis=1)
            median = float(np.median(error))
            mad = float(np.median(np.abs(error - median)))
            keep = error <= median + max(0.0015, 3.5 * mad)
            if np.count_nonzero(keep) >= 6:
                rotation, translation = _kabsch(source[keep], target[keep], weight[keep])
                error = np.linalg.norm(source[keep] @ rotation.T + translation - target[keep], axis=1)
                residual = float(np.median(error))
        target_path = Path(path).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "created_local": datetime.now().isoformat(),
            "source_session": source_session,
            "rotation_camera_from_motor": rotation.tolist(),
            "translation_camera_from_motor_m": translation.tolist(),
            "alignment_updates": int(self.alignment_updates),
            "median_residual_m": residual,
            "coordinate_note": "Motor local +X includes Axis 6 insertion; output is D435 optical frame.",
        }
        target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.rotation, self.translation = rotation, translation
        self.alignment_residual_m = residual
        self.profile_path = target_path
        return target_path

    @staticmethod
    def _controls(axis_row: dict) -> np.ndarray:
        return np.asarray([axis_row.get(f"control_{axis}_m", np.nan) for axis in range(7)], dtype=float)

    def fuse(
        self,
        result: SevenMarkerResult,
        axis_row: dict,
        transform_base_from_camera: np.ndarray,
    ) -> SevenMarkerResult:
        if not self.enabled or not axis_row:
            return result
        local = motor_controls_to_local_points(self._controls(axis_row))
        if not np.isfinite(local).all():
            return result
        measured = result.measured_valid & np.isfinite(result.fused_camera_m).all(axis=1)
        if np.count_nonzero(measured) >= 3:
            measured_weight = np.maximum(result.confidence[measured], 0.08)
            rotation, translation = _kabsch(
                local[measured], result.fused_camera_m[measured],
                measured_weight,
            )
            frame_error = np.linalg.norm(
                local[measured] @ rotation.T + translation - result.fused_camera_m[measured], axis=1
            )
            profile_error = float("nan")
            if self.rotation is not None and self.translation is not None:
                profile_error = float(np.median(np.linalg.norm(
                    local[measured] @ self.rotation.T + self.translation
                    - result.fused_camera_m[measured],
                    axis=1,
                )))
            accept_alignment = (
                self.profile_path is None
                or not np.isfinite(profile_error)
                or profile_error <= 0.010
            )
            if accept_alignment:
                self._alignment_source.append(local[measured].copy())
                self._alignment_target.append(result.fused_camera_m[measured].copy())
                self._alignment_weight.append(measured_weight.copy())
                self.alignment_updates += 1
                if self.rotation is None:
                    self.rotation, self.translation = rotation, translation
                else:
                    alpha = 0.18
                    blended = (1.0 - alpha) * self.rotation + alpha * rotation
                    u, _s, vt = np.linalg.svd(blended)
                    self.rotation = u @ vt
                    if np.linalg.det(self.rotation) < 0:
                        u[:, -1] *= -1
                        self.rotation = u @ vt
                    self.translation = (1.0 - alpha) * self.translation + alpha * translation
                self.alignment_residual_m = float(np.median(frame_error))
        if self.rotation is None or self.translation is None:
            return result
        prior_camera = local @ self.rotation.T + self.translation
        trusted_measured = measured.copy()
        if self.profile_path is not None and np.any(measured):
            disagreement = np.linalg.norm(
                result.fused_camera_m - prior_camera, axis=1
            )
            # A lone depth hit on a transparent lung wall can be formally valid
            # yet tens of millimetres away from the calibrated robot.  Do not
            # let it fold the entire 42 mm chain.  Raw/fused arrays remain in the
            # result for diagnostics; only the accepted smoothed solution uses
            # the calibrated prior at that marker.
            trusted_measured &= disagreement <= 0.012
        output = result.smoothed_camera_m.copy()
        covariance = result.covariance_m2.copy()
        predicted = result.predicted.copy()
        confidence = result.confidence.copy()
        sources = list(result.sources)
        rejected = measured & ~trusted_measured
        missing = ~trusted_measured
        output[missing] = prior_camera[missing]
        predicted[missing] = True
        confidence[missing] = np.maximum(confidence[missing], 0.22)
        covariance[missing] = np.eye(3) * (0.006 ** 2)
        for index in np.flatnonzero(missing):
            sources[index] = (
                "rgb_depth_outlier+motor_pcc" if rejected[index] else "motor_pcc_prior"
            )
        transform = np.asarray(transform_base_from_camera, dtype=float)
        base = np.full_like(output, np.nan)
        finite = np.isfinite(output).all(axis=1)
        base[finite] = output[finite] @ transform[:3, :3].T + transform[:3, 3]
        return replace(
            result,
            smoothed_camera_m=output,
            base_m=base,
            covariance_m2=covariance,
            measured_valid=trusted_measured,
            predicted=predicted,
            confidence=confidence,
            sources=tuple(sources),
        )
