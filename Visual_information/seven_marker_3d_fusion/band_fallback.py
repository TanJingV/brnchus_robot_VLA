"""Periodic bright-band fallback for washed-out seven-colour recordings."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from Visual_information.d435_tdcr_capture.models import KEYPOINT_COUNT


@dataclass(frozen=True)
class PeriodicBandResult:
    accepted: bool
    pixels: np.ndarray
    confidence: np.ndarray
    mask: np.ndarray
    reason: str


def _odd(value: float, minimum: int = 3) -> int:
    result = max(minimum, int(round(value)))
    return result if result % 2 else result + 1


def extract_periodic_bands(
    image_bgr: np.ndarray,
    roi_normalized: tuple[float, float, float, float],
) -> PeriodicBandResult:
    """Find seven transverse rings by local contrast and lattice consistency.

    Colour is unreliable in the old MP4 recordings because exposure clips the
    narrow stickers.  Their repeated 7 mm topology is much more stable: each
    ring is a compact bright component touching the dark tube, and the seven
    centroids form a smooth, approximately periodic image-space chain.
    """

    image = np.asarray(image_bgr, dtype=np.uint8)
    height, width = image.shape[:2]
    x0n, y0n, x1n, y1n = map(float, roi_normalized)
    x0, x1 = sorted((int(round(x0n * width)), int(round(x1n * width))))
    y0, y1 = sorted((int(round(y0n * height)), int(round(y1n * height))))
    x0, x1 = np.clip((x0, x1), 0, width)
    y0, y1 = np.clip((y0, y1), 0, height)
    empty_pixels = np.full((KEYPOINT_COUNT, 2), np.nan)
    empty_confidence = np.zeros(KEYPOINT_COUNT)
    empty_mask = np.zeros((height, width), dtype=np.uint8)
    if x1 - x0 < 80 or y1 - y0 < 30:
        return PeriodicBandResult(False, empty_pixels, empty_confidence, empty_mask, "ROI too small")
    crop = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2]
    scale = max(width / 1920.0, 0.35)
    dark_limit = float(np.clip(np.percentile(value, 30) + 8.0, 68.0, 115.0))
    dark = np.where(value <= dark_limit, 255, 0).astype(np.uint8)
    support_radius = _odd(25 * scale)
    support = cv2.dilate(
        dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (support_radius, support_radius))
    )
    top_hat_size = _odd(19 * scale)
    top_hat = cv2.morphologyEx(
        value,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (top_hat_size, top_hat_size)),
    )
    contrast_limit = max(14.0, float(np.percentile(top_hat, 78)))
    mask = np.where(
        (support != 0) & (top_hat >= contrast_limit) & (value >= max(82.0, dark_limit + 4.0)),
        255,
        0,
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates: list[dict] = []
    minimum_area = max(6.0, 9.0 * scale * scale)
    maximum_area = 1500.0 * scale * scale
    maximum_extent = max(28.0, 72.0 * scale)
    for label in range(1, count):
        left, top, component_width, component_height, area = stats[label]
        if not minimum_area <= area <= maximum_area:
            continue
        if max(component_width, component_height) > maximum_extent:
            continue
        center = centers[label] + np.asarray([x0, y0], dtype=float)
        candidates.append({
            "center": center,
            "area": float(area),
            "label": label,
            "height": float(component_height),
        })
    if len(candidates) < KEYPOINT_COUNT:
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = mask
        return PeriodicBandResult(False, empty_pixels, empty_confidence, full_mask, "fewer than seven bands")
    candidates = sorted(candidates, key=lambda item: item["area"], reverse=True)[:24]
    positions = np.asarray([item["center"] for item in candidates])
    weights = np.asarray([item["area"] for item in candidates])
    centroid = np.average(positions, axis=0, weights=weights)
    covariance = ((positions - centroid) * weights[:, None]).T @ (positions - centroid)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if axis[0] < 0:
        axis *= -1.0
    projection = positions @ axis
    order = np.argsort(projection)
    positions = positions[order]
    projection = projection[order]
    candidate_area = weights[order]
    candidate_height = np.asarray([item["height"] for item in candidates])[order]

    best: tuple[float, np.ndarray, float] | None = None
    minimum_step, maximum_step = 12.0 * scale, 92.0 * scale
    for first in range(len(positions)):
        for last in range(first + KEYPOINT_COUNT - 1, len(positions)):
            step = float((projection[last] - projection[first]) / (KEYPOINT_COUNT - 1))
            if not minimum_step <= step <= maximum_step:
                continue
            selected: list[int] = []
            residuals: list[float] = []
            used: set[int] = set()
            valid_chain = True
            for marker in range(KEYPOINT_COUNT):
                target = projection[first] + marker * step
                ranked = np.argsort(np.abs(projection - target))
                chosen = next((int(index) for index in ranked if int(index) not in used), -1)
                residual = abs(float(projection[chosen] - target)) if chosen >= 0 else float("inf")
                if chosen < 0 or residual > 0.43 * step:
                    valid_chain = False
                    break
                selected.append(chosen)
                residuals.append(residual)
                used.add(chosen)
            if not valid_chain or len(set(selected)) != KEYPOINT_COUNT:
                continue
            sequence = positions[selected]
            gaps = np.linalg.norm(np.diff(sequence, axis=0), axis=1)
            if np.min(gaps) < 0.38 * step or np.max(gaps) > 1.65 * step:
                continue
            smoothness = float(np.mean(np.linalg.norm(np.diff(sequence, n=2, axis=0), axis=1)))
            area_quality = float(np.mean(np.log1p(candidate_area[selected])))
            score = (
                float(np.mean(residuals)) / step
                + 0.16 * smoothness / step
                + 0.10 * float(np.std(gaps)) / step
                - 0.035 * area_quality
            )
            if best is None or score < best[0]:
                best = score, np.asarray(selected, dtype=int), step
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = mask
    if best is None:
        return PeriodicBandResult(False, empty_pixels, empty_confidence, full_mask, "no periodic seven-band chain")

    score, selected, step = best
    chain = positions[selected]
    chain_height = float(np.median(candidate_height[selected]))
    # Determine which end continues into the black shaft.  That end is K0
    # (active-section base); the free end is K6.  This cue is independent of
    # washed-out sticker hue.
    dark_y, dark_x = np.nonzero(dark)
    dark_points = np.column_stack((dark_x + x0, dark_y + y0)).astype(float)
    direction = chain[-1] - chain[0]
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    normal = np.asarray([-direction[1], direction[0]])

    def continuation(endpoint: np.ndarray, sign: float) -> int:
        relative = dark_points - endpoint
        longitudinal = relative @ direction * sign
        lateral = np.abs(relative @ normal)
        return int(np.count_nonzero(
            (longitudinal >= max(4.0, 0.15 * step))
            & (longitudinal <= max(70.0 * scale, 2.4 * step))
            & (lateral <= max(8.0 * scale, 0.65 * chain_height))
        ))

    continuation_at_start = continuation(chain[0], -1.0)
    continuation_at_end = continuation(chain[-1], 1.0)
    if continuation_at_end >= continuation_at_start:
        chain = chain[::-1]
    lattice_quality = float(np.clip(np.exp(-max(0.0, score + 0.12)), 0.25, 0.92))
    confidence = np.full(KEYPOINT_COUNT, lattice_quality)
    return PeriodicBandResult(True, chain, confidence, full_mask, "periodic bright-band fallback")
