"""Short-horizon robust shape smoother with TDCR geometry factors."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from Visual_information.d435_tdcr_capture.models import KEYPOINT_ARCLENGTHS_M, KEYPOINT_COUNT


class ShapeConstrainedSmoother:
    """Fuse measurement, motion and two-section shape priors.

    The constraints are deliberately soft: contact may violate constant
    curvature, while inextensibility still prevents isolated 3-D outliers.
    """

    def __init__(self, config: dict) -> None:
        self.settings = config.get("seven_marker_fusion", {})
        self.previous = np.full((KEYPOINT_COUNT, 3), np.nan)
        self.velocity = np.zeros((KEYPOINT_COUNT, 3))
        self.previous_ns: int | None = None
        self.missing_frames = np.zeros(KEYPOINT_COUNT, dtype=int)

    def reset(self) -> None:
        self.previous[:] = np.nan
        self.velocity[:] = 0.0
        self.previous_ns = None
        self.missing_frames[:] = 0

    @staticmethod
    def _fill_initial(measurements: np.ndarray, valid: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        initial = measurements.copy()
        for index in range(KEYPOINT_COUNT):
            if valid[index]:
                continue
            if np.isfinite(prediction[index]).all():
                initial[index] = prediction[index]
                continue
            available = np.flatnonzero(valid)
            if len(available):
                nearest = int(available[np.argmin(np.abs(available - index))])
                initial[index] = measurements[nearest]
        return initial

    def update(
        self,
        measurements: np.ndarray,
        covariance_m2: np.ndarray,
        valid: np.ndarray,
        confidence: np.ndarray,
        capture_host_ns: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        measurements = np.asarray(measurements, dtype=float).reshape(KEYPOINT_COUNT, 3)
        covariance_m2 = np.asarray(covariance_m2, dtype=float).reshape(KEYPOINT_COUNT, 3, 3)
        valid = np.asarray(valid, dtype=bool).reshape(KEYPOINT_COUNT)
        confidence = np.asarray(confidence, dtype=float).reshape(KEYPOINT_COUNT)
        now_ns = int(capture_host_ns)
        dt = (
            float(np.clip((now_ns - self.previous_ns) * 1e-9, 1.0 / 120.0, 0.10))
            if self.previous_ns is not None else 1.0 / 30.0
        )
        previous_valid = np.isfinite(self.previous).all(axis=1)
        prediction = self.previous + self.velocity * dt
        if not np.any(valid) and not np.any(previous_valid):
            return (
                np.full((KEYPOINT_COUNT, 3), np.nan),
                np.full((KEYPOINT_COUNT, 3, 3), np.nan),
                np.zeros(KEYPOINT_COUNT, dtype=bool),
                np.zeros(KEYPOINT_COUNT),
            )
        initial = self._fill_initial(measurements, valid, prediction)
        if not np.isfinite(initial).all():
            return initial, covariance_m2.copy(), ~valid, confidence.copy()

        measurement_sigma_floor = float(self.settings.get("measurement_sigma_floor_m", 0.00020))
        temporal_sigma = float(self.settings.get("temporal_prediction_sigma_m", 0.0018))
        length_sigma = float(self.settings.get("length_upper_sigma_m", 0.00045))
        curvature_sigma = float(self.settings.get("curvature_smooth_sigma_m", 0.0025))
        maximum_chord_scale = float(self.settings.get("maximum_chord_scale", 1.05))
        temporal_weight = float(self.settings.get("temporal_weight", 0.65))
        curvature_weight = float(self.settings.get("curvature_weight", 0.22))

        inverse_cholesky: dict[int, np.ndarray] = {}
        for index in np.flatnonzero(valid):
            covariance = covariance_m2[index]
            if not np.isfinite(covariance).all():
                covariance = np.eye(3) * measurement_sigma_floor**2
            covariance = 0.5 * (covariance + covariance.T) + np.eye(3) * measurement_sigma_floor**2
            inverse_cholesky[int(index)] = np.linalg.inv(np.linalg.cholesky(covariance))

        def residual(vector: np.ndarray) -> np.ndarray:
            points = vector.reshape(KEYPOINT_COUNT, 3)
            values: list[np.ndarray] = []
            for index in np.flatnonzero(valid):
                weight = math.sqrt(max(float(confidence[index]), 0.08))
                values.append(weight * inverse_cholesky[int(index)] @ (points[index] - measurements[index]))
            for index in np.flatnonzero(previous_valid):
                values.append(
                    np.asarray([temporal_weight])
                    * (points[index] - prediction[index])
                    / temporal_sigma
                )
            segment_lengths = np.diff(KEYPOINT_ARCLENGTHS_M)
            chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
            excess = np.maximum(chords - segment_lengths * maximum_chord_scale, 0.0)
            values.append(excess / length_sigma)
            # Each active section has its own bending state.  Do not force
            # curvature continuity across the K3 connection plane.
            for indices in ((0, 1, 2, 3), (3, 4, 5, 6)):
                section = points[np.asarray(indices)]
                second_difference = section[:-2] - 2.0 * section[1:-1] + section[2:]
                variation = second_difference[1:] - second_difference[:-1]
                values.append(curvature_weight * variation.reshape(-1) / curvature_sigma)
            return np.concatenate([np.ravel(value) for value in values])

        solution = least_squares(
            residual,
            initial.reshape(-1),
            loss="huber",
            f_scale=float(self.settings.get("huber_scale", 1.5)),
            max_nfev=int(self.settings.get("optimizer_max_evaluations", 18)),
            xtol=1e-5,
            ftol=1e-5,
            gtol=1e-5,
        )
        smoothed = solution.x.reshape(KEYPOINT_COUNT, 3)
        predicted_flags = ~valid & np.isfinite(smoothed).all(axis=1)
        output_confidence = confidence.copy()
        output_covariance = covariance_m2.copy()
        maximum_missing = int(self.settings.get("maximum_prediction_frames", 6))
        prediction_process_sigma = float(self.settings.get("prediction_process_sigma_m", 0.0015))
        for index in range(KEYPOINT_COUNT):
            if valid[index]:
                self.missing_frames[index] = 0
            else:
                self.missing_frames[index] += 1
                if self.missing_frames[index] > maximum_missing:
                    smoothed[index] = np.nan
                    predicted_flags[index] = False
                    output_confidence[index] = 0.0
                    output_covariance[index] = np.full((3, 3), np.nan)
                else:
                    output_confidence[index] = max(0.02, 0.35 * math.exp(-0.45 * self.missing_frames[index]))
                    output_covariance[index] = np.eye(3) * (
                        prediction_process_sigma * self.missing_frames[index]
                    ) ** 2
        newly_finite = np.isfinite(smoothed).all(axis=1)
        velocity_valid = newly_finite & previous_valid
        measured_velocity = (smoothed[velocity_valid] - self.previous[velocity_valid]) / dt
        velocity_alpha = float(self.settings.get("velocity_alpha", 0.35))
        self.velocity[velocity_valid] = (
            (1.0 - velocity_alpha) * self.velocity[velocity_valid]
            + velocity_alpha * measured_velocity
        )
        self.velocity[~newly_finite] = 0.0
        self.previous = smoothed.copy()
        self.previous_ns = now_ns
        return smoothed, output_covariance, predicted_flags, output_confidence
