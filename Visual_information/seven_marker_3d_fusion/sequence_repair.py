"""Repair one-missed/one-distractor ring sequences using the 7 mm chain prior."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from Visual_information.d435_tdcr_capture.models import KEYPOINT_COUNT


@dataclass
class SequenceRepair:
    pixels: np.ndarray
    confidence: np.ndarray
    changed: np.ndarray
    accepted: bool


def _robust_spacing(points: list[np.ndarray]) -> float:
    if len(points) < 2:
        return float("nan")
    gaps = np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1)
    if len(gaps) >= 4:
        gaps = np.sort(gaps)[: max(2, int(np.ceil(len(gaps) * 0.70)))]
    return float(np.median(gaps))


def _refine_missing_band(
    image_bgr: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Search RGB brightness/chroma across the local tube cross-section."""

    vector = second - first
    length = float(np.linalg.norm(vector))
    if length < 2.0:
        return first + fraction * vector
    tangent = vector / length
    normal = np.asarray([-tangent[1], tangent[0]])
    predicted = first + fraction * vector
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    best_point = predicted
    best_score = -np.inf
    search_radius = max(2.0, min(length * 0.18, 12.0))
    normal_radius = max(2, int(round(min(length * 0.10, 5.0))))
    for tangent_offset in np.linspace(-search_radius, search_radius, int(2 * search_radius) + 1):
        center = predicted + tangent_offset * tangent
        samples = center[None, :] + np.arange(-normal_radius, normal_radius + 1)[:, None] * normal[None, :]
        x = np.rint(samples[:, 0]).astype(int)
        y = np.rint(samples[:, 1]).astype(int)
        inside = (x >= 0) & (x < image_bgr.shape[1]) & (y >= 0) & (y < image_bgr.shape[0])
        if np.count_nonzero(inside) < 3:
            continue
        x, y = x[inside], y[inside]
        value = hsv[y, x, 2].astype(float)
        saturation = hsv[y, x, 1].astype(float)
        chroma = np.linalg.norm(lab[y, x, 1:].astype(float) - 128.0, axis=1)
        # A ring crosses the tube, so several normal samples become brighter
        # or more chromatic together.  The median rejects a single specular dot.
        score = float(
            np.median(value)
            + 0.12 * np.median(saturation)
            + 0.18 * np.median(chroma)
            - 0.08 * abs(tangent_offset)
        )
        if score > best_score:
            best_score = score
            best_point = center
    return best_point


def repair_ordered_ring_sequence(
    image_bgr: np.ndarray,
    marker_pixels: np.ndarray,
    marker_confidence: np.ndarray,
) -> SequenceRepair:
    """Return a seven-point, tip-to-base regularized marker sequence.

    Input/output remain in K0..K6 identity order.  The repair is only accepted
    when a clear geometric outlier and a compensating missing interval exist;
    normal seven-point detections are left untouched.
    """

    original = np.asarray(marker_pixels, dtype=float).reshape(KEYPOINT_COUNT, 2)
    original_confidence = np.asarray(marker_confidence, dtype=float).reshape(KEYPOINT_COUNT)
    finite = np.isfinite(original).all(axis=1)
    if np.count_nonzero(finite) < 5:
        return SequenceRepair(original.copy(), original_confidence.copy(), np.zeros(7, bool), False)
    # K6 is the free tip and K0 the active-section base.
    points = [original[index].copy() for index in range(6, -1, -1) if finite[index]]
    qualities = [float(original_confidence[index]) for index in range(6, -1, -1) if finite[index]]
    original_points = [point.copy() for point in points]
    original_gaps = np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1)
    original_spacing = _robust_spacing(points)
    if not np.isfinite(original_spacing) or original_spacing < 2.0:
        return SequenceRepair(original.copy(), original_confidence.copy(), np.zeros(7, bool), False)

    removed = False
    # Base/tip reflections are the common failure mode.  Reject an endpoint
    # only when its gap is much larger than the robust spacing of the chain.
    while len(points) > 5:
        gaps = np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1)
        spacing = _robust_spacing(points)
        if gaps[-1] > 2.45 * spacing:
            points.pop()
            qualities.pop()
            removed = True
            continue
        if gaps[0] > 2.45 * spacing:
            points.pop(0)
            qualities.pop(0)
            removed = True
            continue
        break
    if not removed:
        return SequenceRepair(original.copy(), original_confidence.copy(), np.zeros(7, bool), False)

    inserted = False
    while len(points) < KEYPOINT_COUNT and len(points) >= 2:
        gaps = np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1)
        spacing = _robust_spacing(points)
        gap_index = int(np.argmax(gaps))
        gap = float(gaps[gap_index])
        if gap > 1.55 * spacing:
            missing_count = max(1, int(round(gap / spacing)) - 1)
            missing_count = min(missing_count, KEYPOINT_COUNT - len(points))
            for local_index in range(missing_count):
                fraction = (local_index + 1) / (missing_count + 1)
                new_point = _refine_missing_band(
                    image_bgr, points[gap_index], points[gap_index + 1], fraction
                )
                points.insert(gap_index + 1 + local_index, new_point)
                qualities.insert(
                    gap_index + 1 + local_index,
                    0.55 * min(qualities[gap_index], qualities[gap_index + 1 + local_index]),
                )
                inserted = True
            continue
        # If the missing ring is at an endpoint, extrapolate by the local
        # tangent but keep confidence low.  This remains explicitly marked as
        # a repaired 2-D observation and still needs real metric depth.
        if len(points) < KEYPOINT_COUNT:
            direction = points[-1] - points[-2]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                break
            predicted = points[-1] + direction / norm * spacing
            points.append(predicted)
            qualities.append(0.20 * qualities[-1])
            inserted = True
    if len(points) != KEYPOINT_COUNT or not inserted:
        return SequenceRepair(original.copy(), original_confidence.copy(), np.zeros(7, bool), False)
    repaired_gaps = np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1)
    repaired_cv = float(np.std(repaired_gaps) / max(np.mean(repaired_gaps), 1e-6))
    original_cv = float(np.std(original_gaps) / max(np.mean(original_gaps), 1e-6))
    if repaired_cv > min(0.36, original_cv * 0.72):
        return SequenceRepair(original.copy(), original_confidence.copy(), np.zeros(7, bool), False)
    repaired = np.asarray(points[::-1], dtype=float)
    confidence = np.asarray(qualities[::-1], dtype=float)
    changed = (~finite) | (
        np.linalg.norm(repaired - original, axis=1) > max(2.5, 0.12 * original_spacing)
    )
    return SequenceRepair(repaired, confidence, changed, True)
