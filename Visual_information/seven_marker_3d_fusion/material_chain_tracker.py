"""Fast identity-preserving tracker for the seven coloured material bands.

The seven rings are not seven unrelated image points.  They are ordered
material coordinates on one 42 mm continuum section.  This tracker therefore
combines three pieces of evidence on every RGB frame:

* forward/backward pyramidal Lucas--Kanade flow for local image motion;
* the seven fixed Lab colour templates for the common lattice phase;
* the six calibrated material intervals for topology and smoothness.

SAM remains useful for coarse target identity, but an incomplete thin-object
mask is not allowed to veto a clearly observed coloured ring.  That distinction
is essential in the long free-space capture where SAM coverage decreases while
all seven rings remain visible.
"""

from __future__ import annotations

import cv2
import numpy as np

from Visual_information.d435_tdcr_capture.models import KEYPOINT_COUNT

from .body_first_detector import BodyFirstBandResult
from .cotracker3_tracker import TrackedPointRefiner


class MaterialColorChainTracker:
    """Track K0--K6 as one colour-identified material lattice."""

    def __init__(self, tracking_config: dict | None = None) -> None:
        cfg = dict(tracking_config or {})
        self.window = max(21, int(cfg.get("material_lk_window_px", 31))) | 1
        self.levels = max(2, int(cfg.get("material_lk_pyramid_levels", 4)))
        self.maximum_fb_error = float(cfg.get("material_lk_maximum_fb_error_px", 2.0))
        self.maximum_lk_error = float(cfg.get("material_lk_maximum_error", 45.0))
        self.maximum_phase_delta = float(cfg.get("material_colour_maximum_lab_delta", 60.0))
        self.minimum_direct_points = max(
            3, int(cfg.get("material_minimum_direct_points", 3))
        )
        self._photometric = TrackedPointRefiner(cfg)
        self._previous_gray: np.ndarray | None = None
        self._points = np.full((KEYPOINT_COUNT, 2), np.nan, dtype=np.float32)
        self._reference_gaps = np.full(KEYPOINT_COUNT - 1, np.nan, dtype=np.float32)
        self._initialised = False

    @property
    def initialised(self) -> bool:
        return self._initialised

    def reset(self) -> None:
        self._photometric.reset()
        self._previous_gray = None
        self._points[:] = np.nan
        self._reference_gaps[:] = np.nan
        self._initialised = False

    def initialise(self, image_bgr: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        image = np.asarray(image_bgr, dtype=np.uint8)
        points = np.asarray(pixels, dtype=np.float32).reshape(KEYPOINT_COUNT, 2)
        if not np.isfinite(points).all():
            raise ValueError("Material colour-chain initialisation requires seven finite points")
        self._photometric.reset()
        # Keep the user's colour identities exactly.  The upstream
        # initialiser has already snapped coarse clicks onto the centreline.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        for index, point in enumerate(points):
            self._photometric.templates[index] = self._photometric._median_patch(
                lab, point, 2
            )
        projected_gap = float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        self._reference_gaps[:] = max(projected_gap, 4.0)
        self._photometric.reference_gaps[:] = self._reference_gaps
        self._photometric.maximum_delta = self.maximum_phase_delta
        self._points[:] = points
        self._previous_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._initialised = True
        return points.copy()

    @staticmethod
    def _dense_chain(points: np.ndarray, samples_per_interval: int = 14) -> np.ndarray:
        pieces = []
        for index in range(KEYPOINT_COUNT - 1):
            alpha = np.linspace(
                0.0, 1.0, samples_per_interval, endpoint=False, dtype=np.float32
            )[:, None]
            pieces.append((1.0 - alpha) * points[index] + alpha * points[index + 1])
        return np.vstack((*pieces, points[-1:])).astype(np.float32)

    @staticmethod
    def _masks(
        shape: tuple[int, int], points: np.ndarray, gap: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        height, width = shape
        centreline = MaterialColorChainTracker._dense_chain(points)
        polyline = np.rint(centreline).astype(np.int32)
        body = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(
            body,
            [polyline],
            False,
            255,
            max(5, int(round(0.42 * gap)) | 1),
            cv2.LINE_AA,
        )
        search = cv2.dilate(
            body,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (max(9, int(round(0.72 * gap)) | 1),) * 2,
            ),
        )
        bands = []
        half_width = max(4.0, 0.23 * gap)
        thickness = max(3, int(round(0.20 * gap)) | 1)
        for index, point in enumerate(points):
            before = points[max(0, index - 1)]
            after = points[min(KEYPOINT_COUNT - 1, index + 1)]
            tangent = after - before
            tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
            band = np.zeros((height, width), dtype=np.uint8)
            cv2.line(
                band,
                tuple(np.rint(point - half_width * normal).astype(int)),
                tuple(np.rint(point + half_width * normal).astype(int)),
                255,
                thickness,
                cv2.LINE_AA,
            )
            bands.append(band)
        return body, search, centreline, tuple(bands)

    def update(self, image_bgr: np.ndarray) -> BodyFirstBandResult:
        image = np.asarray(image_bgr, dtype=np.uint8)
        height, width = image.shape[:2]
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        if not self._initialised or self._previous_gray is None:
            return BodyFirstBandResult(
                False,
                np.full((KEYPOINT_COUNT, 2), np.nan),
                np.zeros(KEYPOINT_COUNT),
                empty_mask,
                empty_mask.copy(),
                np.empty((0, 2), np.float32),
                tuple(empty_mask.copy() for _ in range(KEYPOINT_COUNT)),
                0.0,
                "material colour chain is not initialised",
                "material-colour-chain",
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        previous = self._points.copy()
        p0 = previous.reshape(-1, 1, 2)
        lk_options = dict(
            winSize=(self.window, self.window),
            maxLevel=self.levels,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
            minEigThreshold=1e-5,
        )
        forward, status, error = cv2.calcOpticalFlowPyrLK(
            self._previous_gray, gray, p0, None, **lk_options
        )
        if forward is None:
            forward = p0.copy()
            status = np.zeros((KEYPOINT_COUNT, 1), dtype=np.uint8)
            error = np.full((KEYPOINT_COUNT, 1), np.inf, dtype=np.float32)
        backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._previous_gray, forward, None, **lk_options
        )
        if backward is None:
            backward = p0.copy()
            reverse_status = np.zeros((KEYPOINT_COUNT, 1), dtype=np.uint8)
        candidates = forward.reshape(KEYPOINT_COUNT, 2)
        fb_error = np.linalg.norm(
            backward.reshape(KEYPOINT_COUNT, 2) - previous, axis=1
        )
        direct = (
            (status.reshape(-1) != 0)
            & (reverse_status.reshape(-1) != 0)
            & (fb_error <= self.maximum_fb_error)
            & (error.reshape(-1) <= self.maximum_lk_error)
            & np.isfinite(candidates).all(axis=1)
        )
        direct_count = int(np.count_nonzero(direct))
        if direct_count:
            common_motion = np.median(candidates[direct] - previous[direct], axis=0)
        else:
            common_motion = np.zeros(2, dtype=np.float32)
        candidates[~direct] = previous[~direct] + common_motion

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        gap = float(np.median(self._reference_gaps))
        phase_shift, phase_quality = self._photometric._joint_band_shift(
            lab, candidates, gap
        )
        phase_confirmed = bool(
            np.isfinite(phase_quality) and phase_quality <= self.maximum_phase_delta
        )
        if phase_confirmed:
            candidates += phase_shift
        points = TrackedPointRefiner._regularise_material_chain(
            candidates, previous, self._reference_gaps
        )
        inside = (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= 0.0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0.0)
            & (points[:, 1] < height)
        )
        accepted = bool(direct_count >= self.minimum_direct_points and np.all(inside))
        self._previous_gray = gray
        if accepted:
            self._points[:] = points
        confidence = np.exp(-0.5 * (fb_error / max(self.maximum_fb_error, 0.25)) ** 2)
        confidence *= np.exp(-np.minimum(error.reshape(-1), 100.0) / 90.0)
        confidence = np.where(direct, confidence, 0.28)
        phase_confidence = (
            float(np.exp(-0.5 * (phase_quality / max(self.maximum_phase_delta, 1.0)) ** 2))
            if np.isfinite(phase_quality)
            else 0.0
        )
        confidence = np.clip(confidence * max(0.35, phase_confidence), 0.05, 1.0)
        body, search, centreline, band_masks = self._masks(
            (height, width), points if accepted else previous, gap
        )
        temporally_predicted = bool(direct_count < 4 or not phase_confirmed)
        reason = (
            f"seven-colour material chain: LK {direct_count}/7, "
            f"phase Lab={phase_quality:.1f}, topology locked"
        )
        return BodyFirstBandResult(
            accepted=accepted,
            pixels=(points if accepted else np.full((KEYPOINT_COUNT, 2), np.nan)),
            confidence=(confidence if accepted else np.zeros(KEYPOINT_COUNT)),
            body_mask=body,
            search_mask=search,
            centerline_xy=centreline,
            band_masks=band_masks,
            score=float(phase_confidence),
            reason=reason,
            segmentation_backend="lk+lab-seven-colour+material-topology",
            segmentation_confidence=float(np.mean(confidence)) if accepted else 0.0,
            temporally_predicted=temporally_predicted,
        )

