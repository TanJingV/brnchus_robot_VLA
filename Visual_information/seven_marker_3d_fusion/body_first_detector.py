"""Strict body-first localisation of the seven transverse marker bands.

The legacy recordings contain a transparent lung phantom, specular metal and
several black drive rails.  Colour blobs alone are therefore not a safe marker
detector.  This module first traces the locally-dark catheter centreline from
the guide towards the distal tip, then searches a one-dimensional transverse
band signal on that centreline.  A seven-point result is returned only when the
whole ordered lattice is supported by the traced body.

The current experiment has a fixed top-view camera: the robot enters from the
right and advances towards the left.  The direction is explicit in the
configuration so a future camera arrangement can provide another tracer
without silently changing marker identities.  K0 is proximal (right) and K6 is
distal (left).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import cv2
import numpy as np

from Visual_information.d435_tdcr_capture.models import KEYPOINT_COUNT

from .neural_body_segmenter import NeuralContinuumSegmenter

try:  # Numba turns the centreline dynamic program into a few-millisecond step.
    from numba import njit
except Exception:  # pragma: no cover - the vectorised Python fallback remains valid.
    njit = None


if njit is not None:
    @njit(cache=True)
    def _trace_dp_numba(
        score: np.ndarray,
        anchor_x: int,
        distal_x: int,
        allowed_top: int,
        allowed_bottom: int,
        centre_y: float,
        anchor_sigma: float,
        maximum_dy: int,
    ) -> np.ndarray:
        height = score.shape[0]
        steps = anchor_x - distal_x
        dp = np.empty(height, dtype=np.float32)
        for y in range(height):
            if y < allowed_top or y >= allowed_bottom:
                dp[y] = 1e6
            else:
                relative = (y - centre_y) / anchor_sigma
                dp[y] = -4.0 * score[y, anchor_x] + relative * relative
        parents = np.zeros((steps, height), dtype=np.int8)
        next_dp = np.empty(height, dtype=np.float32)
        for step in range(steps):
            x = anchor_x - 1 - step
            for y in range(height):
                if y < allowed_top or y >= allowed_bottom:
                    next_dp[y] = 1e6
                    continue
                best_value = 1e20
                best_delta = 0
                for delta_y in range(-maximum_dy, maximum_dy + 1):
                    predecessor_y = y - delta_y
                    if predecessor_y < allowed_top or predecessor_y >= allowed_bottom:
                        continue
                    candidate = dp[predecessor_y] + 0.62 * delta_y * delta_y
                    if candidate < best_value:
                        best_value = candidate
                        best_delta = delta_y
                next_dp[y] = best_value - score[y, x]
                parents[step, y] = best_delta
            temporary = dp
            dp = next_dp
            next_dp = temporary
        current_y = int(np.argmin(dp))
        path = np.empty((steps + 1, 2), dtype=np.float32)
        path[0, 0] = distal_x
        path[0, 1] = current_y
        output_index = 1
        for step in range(steps - 1, -1, -1):
            current_y -= int(parents[step, current_y])
            path[output_index, 0] = anchor_x - step
            path[output_index, 1] = current_y
            output_index += 1
        return path
else:
    _trace_dp_numba = None


@dataclass(frozen=True)
class BodyFirstBandResult:
    accepted: bool
    pixels: np.ndarray
    confidence: np.ndarray
    body_mask: np.ndarray
    search_mask: np.ndarray
    centerline_xy: np.ndarray
    band_masks: tuple[np.ndarray, ...]
    score: float
    reason: str
    segmentation_backend: str = "classical"
    segmentation_confidence: float = 0.0
    temporally_predicted: bool = False


def _empty(shape: tuple[int, int], reason: str) -> BodyFirstBandResult:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    return BodyFirstBandResult(
        accepted=False,
        pixels=np.full((KEYPOINT_COUNT, 2), np.nan),
        confidence=np.zeros(KEYPOINT_COUNT, dtype=float),
        body_mask=mask,
        search_mask=mask.copy(),
        centerline_xy=np.empty((0, 2), dtype=np.float32),
        band_masks=tuple(mask.copy() for _ in range(KEYPOINT_COUNT)),
        score=0.0,
        reason=reason,
        segmentation_backend="none",
        segmentation_confidence=0.0,
    )


def _smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(1, -1)
    return cv2.GaussianBlur(vector, (0, 0), sigmaX=float(sigma)).reshape(-1)


def _shift_with_edge(values: np.ndarray, offset: int) -> np.ndarray:
    result = np.empty_like(values)
    if offset > 0:
        result[:offset] = values[0]
        result[offset:] = values[:-offset]
    elif offset < 0:
        result[offset:] = values[-1]
        result[:offset] = values[-offset:]
    else:
        result[:] = values
    return result


def _sample_image_bilinear(image: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    map_x = points[:, 0].reshape(1, -1)
    map_y = points[:, 1].reshape(1, -1)
    sampled = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(-1, image.shape[2] if image.ndim == 3 else 1)


class BodyFirstBandDetector:
    """Trace the complete visible catheter, then identify its seven bands."""

    def __init__(self, tracking_config: dict | None = None) -> None:
        cfg = dict(tracking_config or {})
        self.maximum_width = max(720, int(cfg.get("body_first_processing_width_px", 960)))
        self.anchor_fraction = float(cfg.get("body_first_anchor_x_fraction", 0.76))
        self.distal_fraction = float(cfg.get("body_first_distal_x_fraction", 0.08))
        self.minimum_gap = float(cfg.get("body_first_minimum_gap_px_at_960", 6.0))
        self.maximum_gap = float(cfg.get("body_first_maximum_gap_px_at_960", 24.0))
        self.minimum_contrast = float(cfg.get("body_first_minimum_band_contrast", 7.0))
        self.minimum_global_score = float(cfg.get("body_first_minimum_global_score", 1.00))
        self.minimum_temporal_score = float(cfg.get("body_first_minimum_temporal_score", 0.72))
        self.temporal_sigma = float(cfg.get("body_first_temporal_sigma_px_at_960", 22.0))
        self.maximum_temporal_error = float(
            cfg.get("body_first_maximum_temporal_error_px_at_960", 55.0)
        )
        self.maximum_marker_prediction_frames = max(
            0, int(cfg.get("body_first_maximum_marker_prediction_frames", 90))
        )
        self.neural_segmenter = NeuralContinuumSegmenter(cfg)
        self.marker_hues = np.asarray(
            cfg.get("body_marker_expected_hues", [0.0, 17.0, 31.0, 60.0, 91.0, 116.0, 150.0]),
            dtype=float,
        ).reshape(KEYPOINT_COUNT)
        # The printed stickers are strongly shifted by D435 auto exposure,
        # white balance and the transparent phantom.  Marker identity is
        # established by its K0..K6 arc order; after a confident lock we can
        # therefore learn the *observed* hue sequence without relabelling.
        self._observed_marker_hues = self.marker_hues.copy()
        self._observed_hue_weight = np.zeros(KEYPOINT_COUNT, dtype=float)
        self._last_pixels = np.full((KEYPOINT_COUNT, 2), np.nan)
        self._last_gray: np.ndarray | None = None
        self._reference_gap_full_px = float("nan")
        self._misses = 0
        self.last_band_prediction = False

    @property
    def target_locked(self) -> bool:
        return self.neural_segmenter.target_locked

    def set_target_lock(self, roi_normalized) -> None:
        """Condition the tracker on one explicitly selected terminal section."""

        self._last_pixels[:] = np.nan
        self._last_gray = None
        self._reference_gap_full_px = float("nan")
        self._misses = 0
        self.last_band_prediction = False
        self._observed_marker_hues = self.marker_hues.copy()
        self._observed_hue_weight[:] = 0.0
        self.neural_segmenter.set_target_lock(roi_normalized)

    def clear_target_lock(self) -> None:
        self._last_pixels[:] = np.nan
        self._last_gray = None
        self._reference_gap_full_px = float("nan")
        self._misses = 0
        self.last_band_prediction = False
        self._observed_marker_hues = self.marker_hues.copy()
        self._observed_hue_weight[:] = 0.0
        self.neural_segmenter.clear_target_lock()

    def condition_keypoints(self, image_bgr: np.ndarray, pixels: np.ndarray) -> None:
        """Seed lattice phase from the user's seven material identities.

        Internally BodyFirst uses proximal-to-distal order (right-to-left for
        the fixed top camera).  The pipeline restores the user's K0--K6 order
        on output, so clicking either endpoint first remains unambiguous.
        """

        image = np.asarray(image_bgr, dtype=np.uint8)
        points = np.asarray(pixels, dtype=float).reshape(KEYPOINT_COUNT, 2).copy()
        if not np.isfinite(points).all():
            raise ValueError("BodyFirst conditioning points must all be finite")
        if points[0, 0] < points[-1, 0]:
            points = points[::-1].copy()
        height, width = image.shape[:2]
        scale = min(1.0, self.maximum_width / max(width, 1))
        if scale < 0.999:
            image = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        self._last_pixels[:] = points
        self._last_gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2]
        self._reference_gap_full_px = float(np.median(
            np.linalg.norm(np.diff(points, axis=0), axis=1)
        ))
        self._misses = 0
        self.last_band_prediction = False

    def tracking_roi_normalized(
        self,
        shape: tuple[int, int],
        fallback: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        keypoint_roi = self._keypoint_tracking_roi(self._last_pixels, shape)
        if keypoint_roi is not None:
            return keypoint_roi
        return self.neural_segmenter.tracking_roi_normalized(shape, fallback)

    def reset(self) -> None:
        self._last_pixels[:] = np.nan
        self._last_gray = None
        self._reference_gap_full_px = float("nan")
        self._misses = 0
        self.last_band_prediction = False
        self._observed_marker_hues = self.marker_hues.copy()
        self._observed_hue_weight[:] = 0.0
        self.neural_segmenter.reset()

    def _optical_flow_prediction(
        self,
        gray: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        previous = self._last_pixels * scale
        if (
            self._last_gray is None
            or self._last_gray.shape != gray.shape
            or not np.isfinite(previous).all()
        ):
            return previous
        source = previous.astype(np.float32).reshape(-1, 1, 2)
        flowed, status, error = cv2.calcOpticalFlowPyrLK(
            self._last_gray,
            gray,
            source,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
        )
        if flowed is None or status is None:
            return previous
        flowed = flowed.reshape(-1, 2).astype(float)
        status = status.reshape(-1).astype(bool)
        error = np.zeros(KEYPOINT_COUNT) if error is None else error.reshape(-1)
        displacement = flowed - previous
        valid = (
            status
            & np.isfinite(flowed).all(axis=1)
            & (error <= 28.0)
            & (np.linalg.norm(displacement, axis=1) <= 40.0)
        )
        if np.count_nonzero(valid) < 3:
            return previous
        median_displacement = np.median(displacement[valid], axis=0)
        residual = np.linalg.norm(displacement - median_displacement, axis=1)
        inliers = valid & (residual <= max(3.0, 2.5 * float(np.median(residual[valid]))))
        if np.count_nonzero(inliers) >= 3:
            median_displacement = np.median(displacement[inliers], axis=0)
        # A common displacement is deliberate: it prevents one stationary lung
        # rib, accidentally close to K0, from pinning the whole seven-point
        # chain while the real colour bands translate with insertion.
        return previous + median_displacement

    @staticmethod
    def _roi_pixels(
        shape: tuple[int, int], roi_normalized: tuple[float, float, float, float]
    ) -> tuple[int, int, int, int]:
        height, width = shape
        x0n, y0n, x1n, y1n = map(float, roi_normalized)
        left, right = sorted((int(round(x0n * width)), int(round(x1n * width))))
        top, bottom = sorted((int(round(y0n * height)), int(round(y1n * height))))
        left, right = np.clip((left, right), 0, width)
        top, bottom = np.clip((top, bottom), 0, height)
        return int(left), int(top), int(right), int(bottom)

    @staticmethod
    def _expand_search_roi(
        roi: tuple[int, int, int, int],
        shape: tuple[int, int],
        minimum_width: int = 190,
        minimum_height: int = 40,
    ) -> tuple[int, int, int, int]:
        """Give the centreline tracer context while keeping SAM's lock tight.

        Seven 0.5--0.8 mm rings can occupy a very small user rectangle.  The
        legacy dynamic-programming tracer used to reject such a correct tight
        box because it required 160x60 processing pixels.  Expand only the
        *search* window around its centre; the original rectangle remains the
        SAM conditioning box and target identity cannot change.
        """

        height, width = shape
        left, top, right, bottom = roi
        centre_x = 0.5 * (left + right)
        centre_y = 0.5 * (top + bottom)
        target_width = min(width, max(right - left, int(minimum_width)))
        target_height = min(height, max(bottom - top, int(minimum_height)))
        left = int(round(centre_x - 0.5 * target_width))
        top = int(round(centre_y - 0.5 * target_height))
        left = int(np.clip(left, 0, max(0, width - target_width)))
        top = int(np.clip(top, 0, max(0, height - target_height)))
        return left, top, left + target_width, top + target_height

    @staticmethod
    def _keypoint_tracking_roi(
        points_xy: np.ndarray,
        shape: tuple[int, int],
    ) -> tuple[float, float, float, float] | None:
        """Moving ROI centred on the same seven material markers.

        A SAM mask can include the connected passive shaft, whose image
        position changes much less than the distal seven-ring section during
        insertion.  The optical-flow prediction of K0--K6 is therefore the
        authoritative moving search window once an initial lattice exists.
        """

        points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        valid = np.isfinite(points).all(axis=1)
        if np.count_nonzero(valid) < 4:
            return None
        points = points[valid]
        height, width = shape
        span_x = max(1.0, float(np.ptp(points[:, 0])))
        span_y = max(1.0, float(np.ptp(points[:, 1])))
        margin_x = max(32.0, 0.22 * span_x)
        margin_y = max(24.0, 0.20 * span_x, 0.75 * span_y)
        left = max(0.0, float(np.min(points[:, 0])) - margin_x)
        right = min(float(width), float(np.max(points[:, 0])) + margin_x)
        top = max(0.0, float(np.min(points[:, 1])) - margin_y)
        bottom = min(float(height), float(np.max(points[:, 1])) + margin_y)
        return left / width, top / height, right / width, bottom / height

    def _remember_flow_prediction(
        self,
        predicted_scaled: np.ndarray,
        gray: np.ndarray,
        scale: float,
    ) -> None:
        """Advance the marker identity through a short visual dropout."""

        if np.isfinite(predicted_scaled).all() and self._misses <= 6:
            self._last_pixels[:] = predicted_scaled / max(scale, 1e-9)
            self._last_gray = gray.copy()

    def _propagate_marker_points(
        self,
        previous_scaled: np.ndarray,
        path: np.ndarray,
        neural_mask: np.ndarray,
        search_radius: int,
    ) -> np.ndarray:
        distances = np.sum(
            (previous_scaled[:, None, :] - path[None, :, :]) ** 2,
            axis=2,
        )
        points = path[np.argmin(distances, axis=1)].copy()
        return self._refine_lattice_on_neural_body(
            points, path, neural_mask, search_radius
        )

    @staticmethod
    def _body_score(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[..., 1].astype(np.float32)
        value = hsv[..., 2].astype(np.float32)
        # Black-hat response isolates the catheter from both the white support
        # and the green surgical mat.  Absolute darkness is only a secondary
        # term because the green mat is itself dark in the value channel.
        close_size = 21
        local_close = cv2.morphologyEx(
            value.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
        ).astype(np.float32)
        black_hat = np.maximum(local_close - value, 0.0)
        neutral_dark = (
            np.clip((135.0 - value) / 80.0, 0.0, 1.0)
            * np.clip((100.0 - saturation) / 100.0, 0.0, 1.0)
        )
        very_dark = np.clip((60.0 - value) / 30.0, 0.0, 1.0)
        score = (
            np.clip(black_hat / 45.0, 0.0, 1.5)
            + 0.45 * neutral_dark
            + 0.30 * very_dark
        )
        score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), 1.2)
        return score, hsv

    def _trace_monotonic_body(
        self,
        score: np.ndarray,
        roi: tuple[int, int, int, int],
        previous_scaled: np.ndarray,
    ) -> np.ndarray:
        height, width = score.shape
        left, top, right, bottom = roi
        span_x = right - left
        span_y = bottom - top
        if span_x < 160 or span_y < 30:
            return np.empty((0, 2), dtype=np.float32)
        anchor_x = int(round(left + self.anchor_fraction * span_x))
        distal_x = int(round(left + self.distal_fraction * span_x))
        anchor_x = int(np.clip(anchor_x, left + 80, right - 2))
        distal_x = int(np.clip(distal_x, left + 1, anchor_x - 80))

        y_index = np.arange(height, dtype=np.float32)
        allowed_top = max(0, top)
        allowed_bottom = min(height, bottom)
        centre_y = 0.5 * (allowed_top + allowed_bottom - 1)
        anchor_sigma = max(18.0, 0.22 * span_y)
        if np.isfinite(previous_scaled[0]).all():
            centre_y = float(previous_scaled[0, 1])
            anchor_sigma = max(12.0, 0.10 * span_y)
        maximum_dy = max(2, int(round(width / 320.0)))
        if _trace_dp_numba is not None:
            path = _trace_dp_numba(
                score,
                anchor_x,
                distal_x,
                allowed_top,
                allowed_bottom,
                centre_y,
                anchor_sigma,
                maximum_dy,
            )
            path[:, 1] = _smooth_1d(path[:, 1], 2.2)
            return path

        dp = -4.0 * score[:, anchor_x] + ((y_index - centre_y) / anchor_sigma) ** 2
        dp[:allowed_top] = 1e6
        dp[allowed_bottom:] = 1e6

        parents: list[np.ndarray] = []
        target_xs = list(range(anchor_x - 1, distal_x - 1, -1))
        for x in target_xs:
            alternatives = []
            for delta_y in range(-maximum_dy, maximum_dy + 1):
                shifted = np.full(height, 1e6, dtype=np.float32)
                if delta_y < 0:
                    shifted[:delta_y] = dp[-delta_y:]
                elif delta_y > 0:
                    shifted[delta_y:] = dp[:-delta_y]
                else:
                    shifted[:] = dp
                # The catheter cannot change direction by several pixels in a
                # single image column.  A strong slope penalty prevents the
                # path from jumping from the black shaft to a nearby specular
                # bronchus edge at a marker gap.
                alternatives.append(shifted + 0.62 * float(delta_y * delta_y))
            stacked = np.stack(alternatives)
            choice = np.argmin(stacked, axis=0)
            dp = stacked[choice, np.arange(height)] - score[:, x]
            dp[:allowed_top] = 1e6
            dp[allowed_bottom:] = 1e6
            parents.append(choice.astype(np.int8) - maximum_dy)

        current_y = int(np.argmin(dp))
        coordinates = [(distal_x, current_y)]
        for index in range(len(parents) - 1, -1, -1):
            delta_y = int(parents[index][current_y])
            current_y -= delta_y
            coordinates.append((target_xs[index] + 1, current_y))
        path = np.asarray(coordinates, dtype=np.float32)
        path = path[np.argsort(path[:, 0])]
        path[:, 1] = _smooth_1d(path[:, 1], 2.2)
        return path

    @staticmethod
    def _sample_profile(hsv: np.ndarray, path: np.ndarray) -> np.ndarray:
        value = hsv[..., 2]
        height, width = value.shape
        profile = np.empty(len(path), dtype=np.float32)
        for index, point in enumerate(path):
            before = path[max(0, index - 3)]
            after = path[min(len(path) - 1, index + 3)]
            tangent = after - before
            tangent_norm = max(float(np.linalg.norm(tangent)), 1e-6)
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32) / tangent_norm
            samples = point[None, :] + np.arange(-3, 4, dtype=np.float32)[:, None] * normal
            sample_x = np.clip(np.rint(samples[:, 0]).astype(int), 0, width - 1)
            sample_y = np.clip(np.rint(samples[:, 1]).astype(int), 0, height - 1)
            profile[index] = float(np.median(value[sample_y, sample_x]))
        return _smooth_1d(profile, 1.0)

    def _unwrap_body_features(
        self,
        hsv: np.ndarray,
        path: np.ndarray,
        half_width: int = 7,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Straighten a bent catheter into arc-length x transverse strips."""

        path = np.asarray(path, dtype=np.float32)
        tangent = np.gradient(path, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        offsets = np.arange(-half_width, half_width + 1, dtype=np.float32)
        coordinates = path[:, None, :] + offsets[None, :, None] * normal[:, None, :]
        samples = _sample_image_bilinear(hsv, coordinates.reshape(-1, 2)).reshape(
            len(path), len(offsets), 3
        )
        hue = samples[..., 0].astype(np.float32)
        saturation = samples[..., 1].astype(np.float32)
        value = samples[..., 2].astype(np.float32)
        # A transverse sticker is bright/chromatic across the tube, not merely
        # one specular pixel.  Percentiles are robust to the black borders.
        bright = np.percentile(value, 75, axis=1) - np.percentile(value, 20, axis=1)
        chroma = np.percentile(saturation, 75, axis=1)
        band_signal = np.clip(bright / 55.0, 0.0, 1.8) + 0.35 * np.clip(chroma / 100.0, 0.0, 1.5)
        band_signal = _smooth_1d(band_signal.astype(np.float32), 1.0)
        weights = np.maximum(saturation, 12.0)
        angles = hue * (2.0 * np.pi / 180.0)
        mean_hue = np.mod(
            np.arctan2(np.sum(weights * np.sin(angles), axis=1), np.sum(weights * np.cos(angles), axis=1))
            * 180.0 / (2.0 * np.pi),
            180.0,
        )
        return band_signal, mean_hue, chroma, coordinates

    def _refine_lattice_on_unwrapped_body(
        self,
        hsv: np.ndarray,
        path: np.ndarray,
        pixels: np.ndarray,
        gap: float,
        previous_scaled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Joint seven-ring structured matching in curved arc-length space."""

        signal, observed_hue, chroma, _coordinates = self._unwrap_body_features(hsv, path)
        path = np.asarray(path, dtype=np.float32)
        # The tracer is x-monotone, so nearest path index is also arc order.
        seed_indices = np.asarray(
            [int(np.argmin(np.sum((path - point) ** 2, axis=1))) for point in pixels],
            dtype=int,
        )
        # K0->K6 is proximal->distal, while path index increases distal->proximal.
        expected_step = max(4.0, float(gap))
        candidates: list[list[tuple[int, float]]] = []
        for marker in range(KEYPOINT_COUNT):
            centre = seed_indices[marker]
            radius = max(3, int(round(0.34 * expected_step)))
            lo, hi = max(0, centre - radius), min(len(path), centre + radius + 1)
            row = []
            for index in range(lo, hi):
                expected_hue = (
                    self._observed_marker_hues[marker]
                    if self._observed_hue_weight[marker] >= 0.5
                    else self.marker_hues[marker]
                )
                hue_delta = abs(float(observed_hue[index]) - float(expected_hue))
                hue_delta = min(hue_delta, 180.0 - hue_delta)
                colour_quality = math.exp(-0.5 * (hue_delta / 22.0) ** 2) * min(1.0, chroma[index] / 55.0)
                temporal = 0.0
                if np.isfinite(previous_scaled[marker]).all():
                    temporal = math.exp(-0.5 * (float(np.linalg.norm(path[index] - previous_scaled[marker])) / 12.0) ** 2)
                score = float(signal[index]) + 0.55 * colour_quality + 0.40 * temporal
                row.append((index, score))
            candidates.append(row)
        # Dynamic programming enforces order and near-equal arc spacing for all
        # seven markers jointly.  One reflection cannot independently move K3.
        states = {index: (score, [index]) for index, score in candidates[0]}
        for marker in range(1, KEYPOINT_COUNT):
            next_states = {}
            for index, score in candidates[marker]:
                best = None
                for previous_index, (previous_score, chain) in states.items():
                    step = previous_index - index
                    if step <= 0:
                        continue
                    spacing_cost = 1.7 * ((step - expected_step) / max(expected_step, 1.0)) ** 2
                    total = previous_score + score - spacing_cost
                    if best is None or total > best[0]:
                        best = (total, chain + [index])
                if best is not None:
                    next_states[index] = best
            states = next_states
            if not states:
                return None
        best_score, indices = max(states.values(), key=lambda item: item[0])
        indices = np.asarray(indices, dtype=int)
        refined = path[indices].copy()
        local_confidence = np.asarray([signal[index] for index in indices], dtype=float)
        confidence = np.clip(0.20 + 0.45 * local_confidence, 0.18, 0.94)
        spacing_cv = float(np.std(-np.diff(indices)) / max(np.mean(-np.diff(indices)), 1e-6))
        if spacing_cv > 0.28 or float(np.mean(local_confidence)) < 0.32:
            return None
        # Online photometric adaptation is deliberately slow and is gated by
        # transverse chroma.  The ordered dynamic-programming solution fixes
        # identity first, so this cannot swap two marker labels.
        for marker, index in enumerate(indices):
            if chroma[index] < 16.0 or local_confidence[marker] < 0.34:
                continue
            alpha = 0.22 if self._observed_hue_weight[marker] < 0.5 else 0.045
            old = self._observed_marker_hues[marker] * (2.0 * np.pi / 180.0)
            new = observed_hue[index] * (2.0 * np.pi / 180.0)
            vector = (1.0 - alpha) * np.asarray([np.cos(old), np.sin(old)]) + alpha * np.asarray([np.cos(new), np.sin(new)])
            self._observed_marker_hues[marker] = (
                np.arctan2(vector[1], vector[0]) * 180.0 / (2.0 * np.pi)
            ) % 180.0
            self._observed_hue_weight[marker] = min(
                1.0, self._observed_hue_weight[marker] + alpha
            )
        return refined, confidence, float(best_score)

    @staticmethod
    def _distal_body_onset(path: np.ndarray, profile: np.ndarray) -> float:
        """Locate where the dark catheter starts when scanning from the tip.

        Narrow colour stickers interrupt the low-value centreline, so a short
        one-dimensional closing bridges those gaps.  Fixed clear-airway ribs
        farther to the right cannot move this distal onset towards the drive
        mechanism and are therefore excluded before lattice scoring.
        """

        dark = np.where(np.asarray(profile) < 155.0, 255, 0).astype(np.uint8).reshape(1, -1)
        closed = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            np.ones((1, 21), dtype=np.uint8),
        ).reshape(-1) != 0
        last = len(closed) - 1
        while last >= 0 and not closed[last]:
            last -= 1
        if last < 0:
            return float(path[0, 0])
        first = last
        while first >= 0 and closed[first]:
            first -= 1
        return float(path[first + 1, 0])

    @staticmethod
    def _robust_smooth_band_centres(pixels: np.ndarray, gap: float) -> np.ndarray:
        """Reject a centreline jump while preserving the seven x locations.

        A transparent airway edge can have a stronger black-hat response than
        the marker exactly at the distal tip.  The seven material points,
        however, lie on one smooth 42 mm backbone.  A small deterministic
        RANSAC quadratic detects such a transverse outlier; only points outside
        the inlier tube are replaced by the smooth backbone prediction.
        """

        points = np.asarray(pixels, dtype=np.float32).copy()
        x = points[:, 0].astype(float)
        y = points[:, 1].astype(float)
        threshold = max(2.5, 0.22 * float(gap))
        best: tuple[int, float, np.ndarray, np.ndarray] | None = None
        for indices in combinations(range(KEYPOINT_COUNT), 3):
            selected = np.asarray(indices, dtype=int)
            if np.ptp(x[selected]) < max(5.0, gap):
                continue
            coefficients = np.polyfit(x[selected], y[selected], 2)
            prediction = np.polyval(coefficients, x)
            residual = np.abs(y - prediction)
            inliers = residual <= threshold
            candidate = (int(np.count_nonzero(inliers)), -float(np.mean(residual[inliers])), coefficients, inliers)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None or best[0] < 5:
            return points
        inliers = best[3]
        degree = 2 if np.count_nonzero(inliers) >= 3 else 1
        coefficients = np.polyfit(x[inliers], y[inliers], degree)
        prediction = np.polyval(coefficients, x)
        residual = np.abs(y - prediction)
        replace = residual > threshold
        points[replace, 1] = prediction[replace]
        return points

    def _find_lattice(
        self,
        path: np.ndarray,
        profile: np.ndarray,
        hsv: np.ndarray,
        distal_onset_x: float,
        reference_gap_scaled: float,
        previous_scaled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, str] | None:
        path_x = path[:, 0]
        first_x = int(np.ceil(path_x[0]))
        last_x = int(np.floor(path_x[-1]))
        values = np.interp(np.arange(first_x, last_x + 1), path_x, profile)
        previous_valid = np.isfinite(previous_scaled).all(axis=1)
        has_history = bool(np.count_nonzero(previous_valid) == KEYPOINT_COUNT and self._misses <= 6)
        previous_gap = float("nan")
        if has_history:
            previous_gap = float(np.median(np.linalg.norm(np.diff(previous_scaled, axis=0), axis=1)))

        shortlist: list[
            tuple[float, np.ndarray, np.ndarray, np.ndarray, float, float, float]
        ] = []
        allowed_minimum_gap = self.minimum_gap
        allowed_maximum_gap = self.maximum_gap
        if np.isfinite(reference_gap_scaled):
            allowed_minimum_gap = max(allowed_minimum_gap, 0.45 * reference_gap_scaled)
            allowed_maximum_gap = min(allowed_maximum_gap, 1.15 * reference_gap_scaled)
        if has_history:
            gap_values = np.arange(
                max(allowed_minimum_gap, 0.70 * previous_gap),
                min(allowed_maximum_gap, 1.05 * previous_gap) + 0.01,
                0.5,
            )
            expected_distal_x = float(previous_scaled[-1, 0])
            temporal_start_radius = max(12.0, 0.85 * previous_gap)
        else:
            gap_values = np.arange(allowed_minimum_gap, allowed_maximum_gap + 0.01, 0.5)
            expected_distal_x = float("nan")
            temporal_start_radius = float("inf")
        for gap in gap_values:
            half = max(2, int(round(0.42 * gap)))
            left_value = _shift_with_edge(values, half)
            right_value = _shift_with_edge(values, -half)
            response = np.maximum(values - 0.5 * (left_value + right_value), 0.0)
            start_max = last_x - 6.0 * gap - max(4.0, 0.35 * gap)
            if start_max <= first_x:
                continue
            start_min = first_x + 2.0
            # The active 42 mm section terminates at the distal end of the
            # complete black-body run.  Permit perspective/occlusion slack of
            # four marker spacings towards the tip, but never search the fixed
            # mechanism and transparent tracheal ribs far behind it.
            start_min = max(start_min, distal_onset_x - 4.2 * gap)
            start_max = min(start_max, distal_onset_x + 1.2 * gap)
            if has_history:
                start_min = max(start_min, expected_distal_x - temporal_start_radius)
                start_max = min(start_max, expected_distal_x + temporal_start_radius)
            for start in np.arange(start_min, start_max, 1.0):
                nominal_x = start + np.arange(KEYPOINT_COUNT) * gap
                nominal_index = np.rint(nominal_x).astype(int) - first_x
                refine_radius = max(1, int(round(0.18 * gap)))
                selected_index = []
                for candidate_index in nominal_index:
                    lo = max(0, candidate_index - refine_radius)
                    hi = min(len(response), candidate_index + refine_radius + 1)
                    selected_index.append(lo + int(np.argmax(response[lo:hi])))
                selected_index = np.asarray(selected_index, dtype=int)
                if np.any(np.diff(selected_index) <= max(2, int(round(0.45 * gap)))):
                    continue
                selected_x = selected_index.astype(np.float32) + first_x
                selected_response = response[selected_index]
                clear_count = int(np.count_nonzero(selected_response >= self.minimum_contrast))
                # Never keep an alternating lung-rib pattern in which only
                # three or four of the seven forced lattice sites have visual
                # evidence.  Missing sites remain missing instead of being
                # manufactured by temporal continuity.
                minimum_clear = 5
                if clear_count < minimum_clear:
                    continue
                midpoint = np.rint(0.5 * (selected_x[:-1] + selected_x[1:])).astype(int) - first_x
                valley_quality = np.clip((150.0 - values[midpoint]) / 70.0, 0.0, 1.0)
                continuation_start = int(round(selected_x[-1] + 0.35 * gap)) - first_x
                continuation_stop = int(round(selected_x[-1] + 2.5 * gap)) - first_x
                continuation = values[
                    max(0, continuation_start):min(len(values), continuation_stop)
                ]
                continuation_quality = (
                    float(np.mean(np.clip((150.0 - continuation) / 70.0, 0.0, 1.0)))
                    if len(continuation) else 0.0
                )
                response_quality = np.clip(selected_response / 38.0, 0.0, 1.5)
                score = (
                    float(np.mean(response_quality))
                    + 0.50 * float(np.mean(valley_quality))
                    + 0.25 * continuation_quality
                    - 0.15 * float(np.std(response_quality))
                )

                selected_path_index = np.searchsorted(path_x, selected_x)
                selected_path_index = np.clip(selected_path_index, 0, len(path) - 1)
                distal_to_proximal = path[selected_path_index]
                proximal_to_distal = distal_to_proximal[::-1].copy()
                temporal_quality = 0.0
                temporal_error = float("nan")
                if has_history:
                    temporal_error = float(np.mean(np.linalg.norm(
                        proximal_to_distal - previous_scaled, axis=1
                    )))
                    if temporal_error > self.maximum_temporal_error:
                        continue
                    temporal_quality = float(np.exp(
                        -0.5 * (temporal_error / max(self.temporal_sigma, 1.0)) ** 2
                    ))
                    gap_quality = float(np.exp(
                        -0.5 * ((gap - previous_gap) / max(1.4, 0.16 * previous_gap)) ** 2
                    ))
                    score += 1.15 * temporal_quality + 0.30 * gap_quality
                shortlist.append((
                        score,
                        proximal_to_distal.copy(),
                        selected_response[::-1].copy(),
                        selected_x[::-1].copy(),
                        temporal_quality,
                        temporal_error,
                        float(gap),
                    ))
                if len(shortlist) > 32:
                    shortlist.sort(key=lambda item: item[0], reverse=True)
                    del shortlist[32:]
        if not shortlist:
            return None
        best = None
        for candidate in shortlist:
            preliminary_score, candidate_pixels, *_rest, candidate_gap = candidate
            transverse_quality = self._transverse_compactness(
                hsv,
                path,
                candidate_pixels,
                candidate_gap,
            )
            compact_count = int(np.count_nonzero(transverse_quality >= 0.28))
            if compact_count < 5:
                continue
            final_score = preliminary_score + 0.65 * float(np.mean(transverse_quality))
            reranked = (final_score,) + candidate[1:]
            if best is None or reranked[0] > best[0]:
                best = reranked
        if best is None:
            return None
        score, pixels, response, selected_x, temporal_quality, temporal_error, gap = best
        threshold = self.minimum_temporal_score if has_history else self.minimum_global_score
        if score < threshold:
            return None
        reason = (
            f"body+lattice temporal score={score:.2f} error={temporal_error:.1f}px"
            if has_history
            else f"body+lattice global score={score:.2f}"
        )
        pixels = self._robust_smooth_band_centres(pixels, gap)
        confidence = np.clip(0.20 + 0.55 * response / 38.0, 0.20, 0.92)
        if has_history:
            confidence *= 0.70 + 0.30 * temporal_quality
        return pixels, confidence, score, reason

    @staticmethod
    def _transverse_compactness(
        hsv: np.ndarray,
        path: np.ndarray,
        pixels: np.ndarray,
        gap: float,
    ) -> np.ndarray:
        """Score whether a stripe is confined to the narrow black catheter.

        A real marker crosses only the 3.5 mm body.  A transparent phantom rib
        or metal highlight continues far beyond that width.  Comparing matched
        band response inside and outside the expected body radius separates
        these otherwise similar one-dimensional peaks.
        """

        value = hsv[..., 2].astype(np.float32)
        height, width = value.shape
        radius = max(3, int(round(0.32 * gap)))
        half = max(2.0, 0.42 * gap)
        qualities = np.zeros(KEYPOINT_COUNT, dtype=float)
        for marker_index, pixel in enumerate(pixels):
            path_index = int(np.argmin(np.sum((path - pixel) ** 2, axis=1)))
            before = path[max(0, path_index - 4)]
            after = path[min(len(path) - 1, path_index + 4)]
            tangent = after - before
            tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
            offsets = np.arange(-4 * radius, 4 * radius + 1, dtype=np.float32)
            transverse = pixel[None, :] + offsets[:, None] * normal[None, :]
            centre = transverse
            flank0 = transverse - half * tangent[None, :]
            flank1 = transverse + half * tangent[None, :]

            def sample(points: np.ndarray) -> np.ndarray:
                xx = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
                yy = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
                return value[yy, xx]

            response = np.maximum(sample(centre) - 0.5 * (sample(flank0) + sample(flank1)), 0.0)
            inside = np.abs(offsets) <= radius
            outside = (np.abs(offsets) >= 2 * radius) & (np.abs(offsets) <= 4 * radius)
            inside_mean = float(np.mean(response[inside]))
            outside_mean = float(np.mean(response[outside])) if np.any(outside) else 0.0
            compactness = float(np.clip(1.0 - outside_mean / (inside_mean + 5.0), 0.0, 1.0))
            strength = float(np.clip(inside_mean / max(1.0, 1.5 * 7.0), 0.0, 1.0))
            qualities[marker_index] = compactness * strength
        return qualities

    @staticmethod
    def _snap_path_to_neural_mask(
        path_xy: np.ndarray,
        neural_mask: np.ndarray,
        search_radius: int,
    ) -> np.ndarray:
        """Move the coarse path to the medial row of the accepted SAM mask."""
        path = np.asarray(path_xy, dtype=np.float32).copy()
        height, width = neural_mask.shape
        radius = max(3, int(search_radius))
        valid = np.zeros(len(path), dtype=bool)
        for index, point in enumerate(path):
            x = int(np.clip(round(float(point[0])), 0, width - 1))
            y = int(np.clip(round(float(point[1])), 0, height - 1))
            top = max(0, y - radius)
            bottom = min(height, y + radius + 1)
            rows = np.flatnonzero(neural_mask[top:bottom, x] != 0) + top
            if len(rows) == 0:
                continue
            # Use the contiguous run nearest to the seed, not the mean of two
            # unrelated masks that happen to cross the same image column.
            split = np.flatnonzero(np.diff(rows) > 1) + 1
            runs = np.split(rows, split)
            run = min(runs, key=lambda values: abs(float(np.mean(values)) - y))
            path[index, 1] = float(np.mean(run))
            valid[index] = True
        if np.count_nonzero(valid) >= max(6, len(path) // 3):
            known_x = path[valid, 0]
            known_y = path[valid, 1]
            order = np.argsort(known_x)
            path[:, 1] = np.interp(path[:, 0], known_x[order], known_y[order])
            path[:, 1] = _smooth_1d(path[:, 1], 1.6)
        return path

    def _material_prompt_backbone(
        self,
        hsv: np.ndarray,
        traced_path: np.ndarray,
        seed_pixels: np.ndarray,
        gap: float,
    ) -> np.ndarray:
        """Build a smooth SAM prompt while rejecting bright phantom edges.

        The preliminary lattice may put one or two distal centres on a clear
        airway reflection.  Such pixels are normally brighter than the other
        five transverse observations.  They may define the x positions, but
        they must not bend the positive SAM prompt away from the black body.
        """
        pixels = np.asarray(seed_pixels, dtype=np.float32)
        x_index = np.clip(np.rint(pixels[:, 0]).astype(int), 0, hsv.shape[1] - 1)
        y_index = np.clip(np.rint(pixels[:, 1]).astype(int), 0, hsv.shape[0] - 1)
        value = hsv[y_index, x_index, 2].astype(float)
        compactness = self._transverse_compactness(hsv, traced_path, pixels, gap)
        brightness_limit = min(225.0, float(np.median(value)) + 22.0)
        reliable = (value <= brightness_limit) & (compactness >= 0.15)
        if np.count_nonzero(reliable) < 4:
            ranking = np.argsort(value - 22.0 * compactness)
            reliable[ranking[:4]] = True
        fit_x = pixels[reliable, 0].astype(float)
        fit_y = pixels[reliable, 1].astype(float)
        degree = 2 if len(fit_x) >= 5 and np.ptp(fit_x) >= 3.0 * gap else 1
        coefficients = np.polyfit(fit_x, fit_y, degree)
        predicted = np.polyval(coefficients, pixels[:, 0])
        corrected_y = pixels[:, 1].astype(float).copy()
        corrected_y[~reliable] = predicted[~reliable]
        # A second fit through the corrected material points gives a dense,
        # differentiable prompt and preserves genuine distributed bending.
        coefficients = np.polyfit(pixels[:, 0], corrected_y, min(2, degree + 1))
        left = float(np.min(pixels[:, 0]) - 0.55 * gap)
        right = float(np.max(pixels[:, 0]) + 0.70 * gap)
        sample_x = np.arange(left, right + 0.5, 1.0, dtype=np.float32)
        sample_y = np.polyval(coefficients, sample_x).astype(np.float32)
        sample_y = np.clip(sample_y, 0, hsv.shape[0] - 1)
        return np.column_stack((sample_x, sample_y)).astype(np.float32)

    @staticmethod
    def _refine_lattice_on_neural_body(
        seed_pixels: np.ndarray,
        snapped_path: np.ndarray,
        neural_mask: np.ndarray,
        search_radius: int,
    ) -> np.ndarray:
        """Keep transverse-band x evidence but measure y on the SAM body."""
        pixels = np.asarray(seed_pixels, dtype=np.float32).copy()
        order = np.argsort(snapped_path[:, 0])
        predicted_y = np.interp(
            pixels[:, 0], snapped_path[order, 0], snapped_path[order, 1]
        )
        height, width = neural_mask.shape
        radius = max(4, int(search_radius))
        for index, (x_value, y_value) in enumerate(zip(pixels[:, 0], predicted_y)):
            x = int(np.clip(round(float(x_value)), 0, width - 1))
            y = int(np.clip(round(float(y_value)), 0, height - 1))
            top = max(0, y - radius)
            bottom = min(height, y + radius + 1)
            rows = np.flatnonzero(neural_mask[top:bottom, x] != 0) + top
            if len(rows):
                split = np.flatnonzero(np.diff(rows) > 1) + 1
                runs = np.split(rows, split)
                run = min(runs, key=lambda values: abs(float(np.mean(values)) - y))
                pixels[index, 1] = float(np.mean(run))
            else:
                pixels[index, 1] = float(y_value)
        return pixels

    def detect(
        self,
        image_bgr: np.ndarray,
        roi_normalized: tuple[float, float, float, float],
    ) -> BodyFirstBandResult:
        image = np.asarray(image_bgr, dtype=np.uint8)
        height, width = image.shape[:2]
        self.last_band_prediction = False
        if not self.target_locked:
            return _empty(
                (height, width),
                "TARGET UNLOCKED: drag a box around the seven-ring terminal section and click lock",
            )
        scale = min(1.0, self.maximum_width / max(width, 1))
        if scale < 0.999:
            small = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = image
        small_height, small_width = small.shape[:2]
        score_map, hsv = self._body_score(small)
        current_gray = hsv[..., 2]
        previous_scaled = self._optical_flow_prediction(current_gray, scale)
        tracking_roi = self._keypoint_tracking_roi(
            previous_scaled, (small_height, small_width)
        )
        if tracking_roi is None:
            tracking_roi = self.neural_segmenter.tracking_roi_normalized(
                (small_height, small_width), roi_normalized
            )
        roi = self._roi_pixels((small_height, small_width), tracking_roi)
        roi = self._expand_search_roi(roi, (small_height, small_width))
        if roi[2] - roi[0] < 160 or roi[3] - roi[1] < 30:
            return _empty((height, width), "body-first ROI too small")
        path = self._trace_monotonic_body(score_map, roi, previous_scaled)
        if len(path) < 100:
            self._misses += 1
            self._remember_flow_prediction(previous_scaled, current_gray, scale)
            return _empty((height, width), "black body centreline not found")
        # First establish the identity and complete pixel region of the black
        # terminal section.  Colour-ring positions are deliberately *not*
        # needed to initialise SAM2: the explicit lock box and coarse black
        # centreline are the only conditioning prompts.
        seed_radius = 4.0
        neural = self.neural_segmenter.segment(small, path, seed_radius, roi)
        neural_mask_small: np.ndarray | None = None
        if neural.accepted:
            neural_mask_small = neural.mask
            path = self._snap_path_to_neural_mask(
                path,
                neural_mask_small,
                max(6, int(round(2.5 * seed_radius))),
            )
        elif self.neural_segmenter.enabled and self.neural_segmenter.required:
            self._misses += 1
            self._remember_flow_prediction(previous_scaled, current_gray, scale)
            return _empty((height, width), f"neural body segmentation rejected: {neural.reason}")

        # Only after the locked body mask exists do we localise the seven
        # transverse colour bands along that body's medial path.
        profile = self._sample_profile(hsv, path)
        distal_onset_x = self._distal_body_onset(path, profile)
        lattice = self._find_lattice(
            path,
            profile,
            hsv,
            distal_onset_x,
            self._reference_gap_full_px * scale,
            previous_scaled,
        )
        temporally_predicted = False
        if lattice is None:
            self._misses += 1
            if (
                neural_mask_small is not None
                and np.isfinite(previous_scaled).all()
                and self._misses <= self.maximum_marker_prediction_frames
            ):
                # Preserve K0--K6 identity through glare/occlusion.  These are
                # deliberately labelled predictions downstream and therefore
                # never counted as direct RGB/depth measurements.
                pixels_scaled = self._propagate_marker_points(
                    previous_scaled,
                    path,
                    neural_mask_small,
                    max(6, int(round(2.5 * seed_radius))),
                )
                confidence = np.full(
                    KEYPOINT_COUNT,
                    max(0.08, 0.34 * (0.97 ** self._misses)),
                    dtype=float,
                )
                lattice_score = 0.0
                reason = (
                    f"seven bands temporarily occluded; locked-body/optical-flow "
                    f"prediction age={self._misses}"
                )
                temporally_predicted = True
            else:
                self._remember_flow_prediction(previous_scaled, current_gray, scale)
                return _empty((height, width), "locked body found but seven transverse bands were not confirmed")
        else:
            pixels_scaled, confidence, lattice_score, reason = lattice
            structured = self._refine_lattice_on_unwrapped_body(
                hsv,
                path,
                pixels_scaled,
                float(np.median(np.linalg.norm(np.diff(pixels_scaled, axis=0), axis=1))),
                previous_scaled,
            )
            if structured is not None:
                pixels_scaled, structured_confidence, structured_score = structured
                confidence = np.sqrt(np.clip(confidence, 0.0, 1.0) * structured_confidence)
                lattice_score += 0.25 * structured_score / KEYPOINT_COUNT
                reason += "; arc-unwrapped seven-colour structured match"
        if neural_mask_small is not None:
            pixels_scaled = self._refine_lattice_on_neural_body(
                pixels_scaled,
                path,
                neural_mask_small,
                max(6, int(round(2.5 * seed_radius))),
            )

        if neural_mask_small is not None:
            support = cv2.dilate(
                neural_mask_small,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            )
            support_x = np.clip(
                np.rint(pixels_scaled[:, 0]).astype(int), 0, small_width - 1
            )
            support_y = np.clip(
                np.rint(pixels_scaled[:, 1]).astype(int), 0, small_height - 1
            )
            marker_support = support[support_y, support_x] != 0
            supported_count = int(np.count_nonzero(marker_support))
            if supported_count < KEYPOINT_COUNT and not temporally_predicted:
                self._misses += 1
                if (
                    np.isfinite(previous_scaled).all()
                    and self._misses <= self.maximum_marker_prediction_frames
                ):
                    pixels_scaled = self._propagate_marker_points(
                        previous_scaled,
                        path,
                        neural_mask_small,
                        max(6, int(round(2.5 * seed_radius))),
                    )
                    confidence = np.full(
                        KEYPOINT_COUNT,
                        max(0.08, 0.34 * (0.97 ** self._misses)),
                        dtype=float,
                    )
                    lattice_score = 0.0
                    reason = (
                        f"colour lattice left locked mask; locked-body/optical-flow "
                        f"prediction age={self._misses}"
                    )
                    temporally_predicted = True
                    support_x = np.clip(
                        np.rint(pixels_scaled[:, 0]).astype(int), 0, small_width - 1
                    )
                    support_y = np.clip(
                        np.rint(pixels_scaled[:, 1]).astype(int), 0, small_height - 1
                    )
                    supported_count = int(np.count_nonzero(support[support_y, support_x]))
                else:
                    self._remember_flow_prediction(previous_scaled, current_gray, scale)
                    return _empty((height, width), "one or more bands lie outside the SAM2.1 body mask")
            if temporally_predicted and supported_count < 3:
                self._remember_flow_prediction(previous_scaled, current_gray, scale)
                return _empty((height, width), "predicted marker chain left the locked SAM2.1 body")

        gap = float(np.median(np.linalg.norm(np.diff(pixels_scaled, axis=0), axis=1)))
        # SAM deliberately follows the black object, so at the proximal end it
        # can also include the long insertion shaft and the metal connector.
        # That is useful for identity tracking, but it is not the 42 mm active
        # section.  Build the exported/displayed mask strictly from K6..K0
        # plus less than half a marker pitch at either end.
        active_left = float(np.min(pixels_scaled[:, 0]))
        active_right = float(np.max(pixels_scaled[:, 0]))
        keep = (
            (path[:, 0] >= active_left - 0.45 * gap)
            & (path[:, 0] <= active_right + 0.45 * gap)
        )
        body_path = path[keep]
        if len(body_path) < 2:
            self._misses += 1
            return _empty((height, width), "black body path terminated before marker chain")

        # Replace the centreline inside the active 42 mm section by the smooth
        # material-point interpolation.  This keeps the displayed body mask on
        # the catheter rather than a neighbouring clear-airway reflection.
        order = np.argsort(pixels_scaled[:, 0])
        band_x = pixels_scaled[order, 0]
        band_y = pixels_scaled[order, 1]
        active = (body_path[:, 0] >= band_x[0]) & (body_path[:, 0] <= band_x[-1])
        body_path[active, 1] = np.interp(body_path[active, 0], band_x, band_y)

        body_radius = max(3, int(round(0.36 * gap)))
        if neural_mask_small is not None:
            neural_distance = cv2.distanceTransform(neural_mask_small, cv2.DIST_L2, 3)
            xx = np.clip(np.rint(body_path[:, 0]).astype(int), 0, small_width - 1)
            yy = np.clip(np.rint(body_path[:, 1]).astype(int), 0, small_height - 1)
            radius_samples = neural_distance[yy, xx]
            radius_samples = radius_samples[radius_samples > 0.5]
            if len(radius_samples):
                body_radius = int(np.clip(round(float(np.median(radius_samples))), 3, 0.60 * gap))
        body_mask_small = np.zeros((small_height, small_width), dtype=np.uint8)
        body_polyline = np.rint(body_path).astype(np.int32)
        body_corridor = np.zeros_like(body_mask_small)
        cv2.polylines(
            body_corridor,
            [body_polyline],
            False,
            255,
            4 * body_radius + 1,
            cv2.LINE_8,
        )
        if neural_mask_small is not None:
            body_mask_small = cv2.bitwise_and(neural_mask_small, body_corridor)
            body_mask_small = cv2.morphologyEx(
                body_mask_small,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            # Coloured rings intentionally interrupt the black appearance but
            # are physically part of the same body.  Preserve a narrow neural
            # centreline core so the final body mask remains one component.
            cv2.polylines(
                body_mask_small,
                [body_polyline],
                False,
                255,
                max(3, body_radius),
                cv2.LINE_8,
            )
            body_mask_small = cv2.bitwise_and(body_mask_small, body_corridor)
        else:
            cv2.polylines(
                body_mask_small,
                [body_polyline],
                False,
                255,
                2 * body_radius + 1,
                cv2.LINE_8,
            )
        search_mask_small = cv2.dilate(
            body_mask_small,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * body_radius + 3, 2 * body_radius + 3)
            ),
        )
        band_masks_small = []
        for pixel in pixels_scaled:
            index = int(np.argmin(np.sum((body_path - pixel) ** 2, axis=1)))
            before = body_path[max(0, index - 3)]
            after = body_path[min(len(body_path) - 1, index + 3)]
            tangent = after - before
            tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
            normal = np.asarray([-tangent[1], tangent[0]])
            endpoint0 = pixel - normal * body_radius
            endpoint1 = pixel + normal * body_radius
            marker_mask = np.zeros_like(body_mask_small)
            cv2.line(
                marker_mask,
                tuple(np.rint(endpoint0).astype(int)),
                tuple(np.rint(endpoint1).astype(int)),
                255,
                max(3, int(round(0.30 * gap))),
                cv2.LINE_8,
            )
            band_masks_small.append(marker_mask)

        output_size = (width, height)
        body_mask = cv2.resize(body_mask_small, output_size, interpolation=cv2.INTER_NEAREST)
        search_mask = cv2.resize(search_mask_small, output_size, interpolation=cv2.INTER_NEAREST)
        band_masks = tuple(
            cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
            for mask in band_masks_small
        )
        pixels = pixels_scaled / scale
        centerline = body_path / scale
        if not np.isfinite(self._reference_gap_full_px):
            self._reference_gap_full_px = float(np.median(
                np.linalg.norm(np.diff(pixels, axis=0), axis=1)
            ))
        self._last_pixels[:] = pixels
        self._last_gray = current_gray.copy()
        if not temporally_predicted:
            self._misses = 0
        self.last_band_prediction = temporally_predicted
        return BodyFirstBandResult(
            accepted=True,
            pixels=pixels,
            confidence=confidence,
            body_mask=body_mask,
            search_mask=search_mask,
            centerline_xy=centerline.astype(np.float32),
            band_masks=band_masks,
            score=float(lattice_score),
            reason=(f"{reason}; {neural.reason}" if neural.accepted else reason),
            segmentation_backend=(neural.backend if neural.accepted else "classical"),
            segmentation_confidence=(neural.confidence if neural.accepted else 0.0),
            temporally_predicted=temporally_predicted,
        )
