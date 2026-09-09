"""Seven-marker RGB/IR tracking, stereo triangulation and low-latency filtering."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .models import KEYPOINT_COUNT, CameraFrame, KeypointObservation


try:
    from numba import cuda as _numba_cuda
except Exception:
    _numba_cuda = None


def numba_cuda_available() -> bool:
    try:
        return _numba_cuda is not None and bool(_numba_cuda.is_available())
    except Exception:
        return False


if _numba_cuda is not None:
    @_numba_cuda.jit(device=True, inline=True)
    def _bilinear_channel(source, y0, x0, y1, x1, wx, wy, channel):
        top = source[y0, x0, channel] * (1.0 - wx) + source[y0, x1, channel] * wx
        bottom = source[y1, x0, channel] * (1.0 - wx) + source[y1, x1, channel] * wx
        return (top * (1.0 - wy) + bottom * wy) / 255.0

    @_numba_cuda.jit
    def _bgr_to_hsv_lab_kernel(source, hsv, lab):
        y_out, x_out = _numba_cuda.grid(2)
        height_out, width_out = hsv.shape[0], hsv.shape[1]
        if y_out >= height_out or x_out >= width_out:
            return
        height_in, width_in = source.shape[0], source.shape[1]
        x_source = (x_out + 0.5) * width_in / width_out - 0.5
        y_source = (y_out + 0.5) * height_in / height_out - 0.5
        x0 = max(0, min(width_in - 1, int(math.floor(x_source))))
        y0 = max(0, min(height_in - 1, int(math.floor(y_source))))
        x1 = min(width_in - 1, x0 + 1)
        y1 = min(height_in - 1, y0 + 1)
        wx = max(0.0, min(1.0, x_source - x0))
        wy = max(0.0, min(1.0, y_source - y0))
        blue = _bilinear_channel(source, y0, x0, y1, x1, wx, wy, 0)
        green = _bilinear_channel(source, y0, x0, y1, x1, wx, wy, 1)
        red = _bilinear_channel(source, y0, x0, y1, x1, wx, wy, 2)
        maximum = max(red, green, blue)
        minimum = min(red, green, blue)
        difference = maximum - minimum
        hue = 0.0
        if difference > 1e-7:
            if maximum == red:
                hue = ((green - blue) / difference) % 6.0
            elif maximum == green:
                hue = (blue - red) / difference + 2.0
            else:
                hue = (red - green) / difference + 4.0
            hue = (hue * 30.0) % 180.0
        saturation = difference / maximum if maximum > 1e-7 else 0.0
        hsv[y_out, x_out, 0] = int(max(0.0, min(179.0, hue)) + 0.5)
        hsv[y_out, x_out, 1] = int(max(0.0, min(255.0, saturation * 255.0)) + 0.5)
        hsv[y_out, x_out, 2] = int(max(0.0, min(255.0, maximum * 255.0)) + 0.5)

        red_linear = red / 12.92 if red <= 0.04045 else math.pow((red + 0.055) / 1.055, 2.4)
        green_linear = green / 12.92 if green <= 0.04045 else math.pow((green + 0.055) / 1.055, 2.4)
        blue_linear = blue / 12.92 if blue <= 0.04045 else math.pow((blue + 0.055) / 1.055, 2.4)
        xyz_x = (0.4124564 * red_linear + 0.3575761 * green_linear + 0.1804375 * blue_linear) / 0.95047
        xyz_y = 0.2126729 * red_linear + 0.7151522 * green_linear + 0.0721750 * blue_linear
        xyz_z = (0.0193339 * red_linear + 0.1191920 * green_linear + 0.9503041 * blue_linear) / 1.08883
        threshold = math.pow(6.0 / 29.0, 3.0)
        linear_scale = 1.0 / (3.0 * math.pow(6.0 / 29.0, 2.0))
        fx = math.pow(xyz_x, 1.0 / 3.0) if xyz_x > threshold else xyz_x * linear_scale + 4.0 / 29.0
        fy = math.pow(xyz_y, 1.0 / 3.0) if xyz_y > threshold else xyz_y * linear_scale + 4.0 / 29.0
        fz = math.pow(xyz_z, 1.0 / 3.0) if xyz_z > threshold else xyz_z * linear_scale + 4.0 / 29.0
        lab[y_out, x_out, 0] = int(max(0.0, min(255.0, (116.0 * fy - 16.0) * 2.55)) + 0.5)
        lab[y_out, x_out, 1] = int(max(0.0, min(255.0, 500.0 * (fx - fy) + 128.0)) + 0.5)
        lab[y_out, x_out, 2] = int(max(0.0, min(255.0, 200.0 * (fy - fz) + 128.0)) + 0.5)


class NumbaCudaColourPreprocessor:
    def __init__(self) -> None:
        if not numba_cuda_available():
            raise RuntimeError("Numba CUDA is unavailable")
        self.input_shape = None
        self.output_shape = None
        self.device_input = None
        self.device_hsv = None
        self.device_lab = None
        self.host_hsv = None
        self.host_lab = None

    def process(
        self, image_bgr: np.ndarray, target_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        source = np.ascontiguousarray(image_bgr, dtype=np.uint8)
        output_shape = (target_size[1], target_size[0], 3)
        if self.input_shape != source.shape:
            self.device_input = _numba_cuda.device_array(source.shape, dtype=np.uint8)
            self.input_shape = source.shape
        if self.output_shape != output_shape:
            self.device_hsv = _numba_cuda.device_array(output_shape, dtype=np.uint8)
            self.device_lab = _numba_cuda.device_array(output_shape, dtype=np.uint8)
            self.host_hsv = _numba_cuda.pinned_array(output_shape, dtype=np.uint8)
            self.host_lab = _numba_cuda.pinned_array(output_shape, dtype=np.uint8)
            self.output_shape = output_shape
        self.device_input.copy_to_device(source)
        threads = (16, 16)
        blocks = (
            (output_shape[0] + threads[0] - 1) // threads[0],
            (output_shape[1] + threads[1] - 1) // threads[1],
        )
        _bgr_to_hsv_lab_kernel[blocks, threads](self.device_input, self.device_hsv, self.device_lab)
        self.device_hsv.copy_to_host(self.host_hsv)
        self.device_lab.copy_to_host(self.host_lab)
        return self.host_hsv, self.host_lab


def _project(point: np.ndarray, intr: dict[str, Any]) -> np.ndarray:
    if not np.isfinite(point).all() or point[2] <= 0:
        return np.full(2, np.nan)
    return np.asarray([
        intr["fx"] * point[0] / point[2] + intr["ppx"],
        intr["fy"] * point[1] / point[2] + intr["ppy"],
    ])


def _deproject(pixel: np.ndarray, depth_m: float, intr: dict[str, Any]) -> np.ndarray:
    return np.asarray([
        (pixel[0] - intr["ppx"]) * depth_m / intr["fx"],
        (pixel[1] - intr["ppy"]) * depth_m / intr["fy"],
        depth_m,
    ])


def _transform(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ point + transform[:3, 3]


@dataclass
class ContinuumSegmentation:
    valid: bool
    body_mask: np.ndarray
    search_mask: np.ndarray
    centerline_xy: np.ndarray
    ordered_band_pixels: np.ndarray
    ordered_band_confidence: np.ndarray
    ordered_band_masks: list[np.ndarray]
    ordered_band_colours: np.ndarray


def _normalized_roi_mask(
    shape: tuple[int, int], roi_normalized: tuple[float, float, float, float]
) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = roi_normalized
    left, top = int(np.floor(x0 * width)), int(np.floor(y0 * height))
    right, bottom = int(np.ceil(x1 * width)), int(np.ceil(y1 * height))
    mask = np.zeros(shape, dtype=np.uint8)
    mask[max(0, top):min(height, bottom), max(0, left):min(width, right)] = 255
    return mask


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    image = np.where(mask != 0, 255, 0).astype(np.uint8)
    if not np.any(image):
        return image
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(image)
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.any(image):
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = eroded
    return skeleton


def _longest_skeleton_path(skeleton: np.ndarray) -> np.ndarray:
    yy, xx = np.nonzero(skeleton)
    if len(xx) < 2:
        return np.empty((0, 2), dtype=np.float32)
    index_map = np.full(skeleton.shape, -1, dtype=np.int32)
    index_map[yy, xx] = np.arange(len(xx), dtype=np.int32)
    coordinates = np.column_stack((xx, yy)).astype(np.int32)

    def breadth_first(start: int, keep_parent: bool = False):
        distance = np.full(len(coordinates), -1, dtype=np.int32)
        parent = np.full(len(coordinates), -1, dtype=np.int32) if keep_parent else None
        distance[start] = 0
        queue = deque([start])
        while queue:
            current = queue.popleft()
            x, y = coordinates[current]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]):
                        continue
                    neighbor = int(index_map[ny, nx])
                    if neighbor < 0 or distance[neighbor] >= 0:
                        continue
                    distance[neighbor] = distance[current] + 1
                    if parent is not None:
                        parent[neighbor] = current
                    queue.append(neighbor)
        farthest = int(np.argmax(distance))
        return farthest, distance, parent

    visited = np.zeros(len(coordinates), dtype=bool)
    longest_indices: list[int] = []
    for seed in range(len(coordinates)):
        if visited[seed]:
            continue
        first, component_distance, _parent = breadth_first(seed)
        visited |= component_distance >= 0
        last, _distance, parent = breadth_first(first, keep_parent=True)
        path_indices = []
        current = last
        while current >= 0:
            path_indices.append(current)
            if current == first:
                break
            current = int(parent[current])
        if path_indices and path_indices[-1] == first and len(path_indices) > len(longest_indices):
            longest_indices = path_indices[::-1]
    if not longest_indices:
        return np.empty((0, 2), dtype=np.float32)
    return coordinates[np.asarray(longest_indices, dtype=int)].astype(np.float32)


def _empty_continuum_segmentation(shape: tuple[int, int]) -> ContinuumSegmentation:
    empty = np.zeros(shape, dtype=np.uint8)
    return ContinuumSegmentation(
        valid=False,
        body_mask=empty,
        search_mask=empty.copy(),
        centerline_xy=np.empty((0, 2), dtype=np.float32),
        ordered_band_pixels=np.full((KEYPOINT_COUNT, 2), np.nan),
        ordered_band_confidence=np.zeros(KEYPOINT_COUNT, dtype=float),
        ordered_band_masks=[empty.copy() for _ in range(KEYPOINT_COUNT)],
        ordered_band_colours=np.full((KEYPOINT_COUNT, 3), np.nan),
    )


def scaled_tracking_config(tracking_config: dict, scale: float) -> dict:
    """Scale pixel-domain gates while retaining enough support for thin rings."""
    if scale >= 0.999:
        return tracking_config
    result = dict(tracking_config)
    result["minimum_blob_area_px"] = max(
        1.0, float(tracking_config["minimum_blob_area_px"]) * scale**2
    )
    result["maximum_blob_area_px"] = max(
        64.0, float(tracking_config["maximum_blob_area_px"]) * scale**2
    )
    result["temporal_sigma_px"] = float(
        tracking_config.get("temporal_sigma_px", 42.0)
    ) * scale
    result["maximum_temporal_jump_px"] = float(
        tracking_config.get("maximum_temporal_jump_px", 110.0)
    ) * scale
    radius_minimums = {
        "body_open_radius_px": 1,
        "body_gap_close_radius_px": 5,
        "body_marker_support_radius_px": 6,
        "body_component_link_radius_px": 4,
        "body_min_centerline_px": 12,
        "marker_band_merge_distance_px": 6,
        "body_tight_band_support_radius_px": 2,
    }
    radius_defaults = {
        "body_open_radius_px": 1,
        "body_gap_close_radius_px": 7,
        "body_marker_support_radius_px": 22,
        "body_component_link_radius_px": 6,
        "body_min_centerline_px": 35,
        "marker_band_merge_distance_px": 8,
        "body_tight_band_support_radius_px": 10,
    }
    for key, minimum in radius_minimums.items():
        result[key] = max(
            minimum,
            int(round(float(tracking_config.get(key, radius_defaults[key])) * scale)),
        )
    result["body_min_component_area_px"] = max(
        24.0,
        float(tracking_config.get("body_min_component_area_px", 120)) * scale**2,
    )
    result["marker_band_max_area_px"] = max(
        100.0,
        float(tracking_config.get("marker_band_max_area_px", 600)) * scale**2,
    )
    result["body_selection_min_band_area_px"] = max(
        2.0,
        float(tracking_config.get("body_selection_min_band_area_px", 12)) * scale**2,
    )
    return result


def segment_black_continuum(
    hsv: np.ndarray,
    lab: np.ndarray,
    marker_config: list[dict],
    prototypes: list[dict[str, float]],
    tracking_config: dict,
    previous_pixels: np.ndarray,
    roi_normalized: tuple[float, float, float, float],
) -> ContinuumSegmentation:
    """Lock the dark continuum first, then recover seven ordered ring bands."""

    height, width = hsv.shape[:2]
    empty = _empty_continuum_segmentation((height, width))
    if not bool(tracking_config.get("body_segmentation_enabled", True)):
        return empty
    roi_mask = _normalized_roi_mask((height, width), roi_normalized)
    roi_values = hsv[..., 2][roi_mask != 0]
    if roi_values.size < 64:
        return empty
    otsu_threshold, _unused = cv2.threshold(
        roi_values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    value_limit = int(min(
        float(tracking_config.get("body_value_max", 92)),
        max(float(tracking_config.get("body_value_min", 18)), float(otsu_threshold)),
    ))
    dark = np.where((hsv[..., 2] <= value_limit) & (roi_mask != 0), 255, 0).astype(np.uint8)
    dark_seed = dark.copy()
    # The seven bright rings interrupt the dark tube.  Bridge only bright pixels
    # touching the dark seed so the complete tube becomes one component without
    # admitting unrelated bright background or metal elsewhere in the image.
    bridge_radius = max(
        1, int(tracking_config.get("body_band_bridge_radius_px", 6))
    )
    bridge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * bridge_radius + 1, 2 * bridge_radius + 1)
    )
    near_dark = cv2.dilate(dark, bridge_kernel) != 0
    bridge_value = min(
        255,
        value_limit + int(tracking_config.get("body_band_bridge_value_margin", 10)),
    )
    bridge_saturation = int(
        tracking_config.get("body_band_bridge_min_saturation", 8)
    )
    band_bridge = (
        near_dark
        & (roi_mask != 0)
        & (hsv[..., 2] >= bridge_value)
        & (
            (hsv[..., 1] >= bridge_saturation)
            | (hsv[..., 2] >= min(255, bridge_value + 24))
        )
    )
    dark[band_bridge] = 255
    open_radius = max(0, int(tracking_config.get("body_open_radius_px", 1)))
    close_radius = max(1, int(tracking_config.get("body_gap_close_radius_px", 7)))
    if open_radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * open_radius + 1, 2 * open_radius + 1)
        )
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1)
    )
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, close_kernel)
    link_radius = max(1, int(tracking_config.get("body_component_link_radius_px", 6)))
    link_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * link_radius + 1, 2 * link_radius + 1)
    )
    linked = cv2.dilate(closed, link_kernel)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(linked, 8)
    minimum_area = max(24.0, float(tracking_config.get("body_min_component_area_px", 120)))
    maximum_fraction = float(tracking_config.get("body_max_area_fraction", 0.30))
    minimum_elongation = float(tracking_config.get("body_min_elongation", 2.0))
    previous_valid = previous_pixels[np.isfinite(previous_pixels).all(axis=1)]
    best_label = -1
    best_score = -math.inf
    best_elongation = 0.0
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area or area > maximum_fraction * height * width:
            continue
        component_y, component_x = np.nonzero(labels == label)
        if len(component_x) < 3:
            continue
        coordinates = np.column_stack((component_x, component_y)).astype(np.float32)
        covariance = np.cov(coordinates, rowvar=False)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 1e-6))
        elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
        box_area = max(
            1.0,
            float(stats[label, cv2.CC_STAT_WIDTH] * stats[label, cv2.CC_STAT_HEIGHT]),
        )
        fill = area / box_area
        component = labels == label
        absolute_band_value = int(tracking_config.get("marker_band_absolute_value_min", 135))
        band_evidence_mask = (
            component
            & (
                hsv[..., 2]
                >= max(
                    absolute_band_value,
                    min(255, value_limit + int(tracking_config.get("marker_band_value_margin", 14))),
                )
            )
            & (
                hsv[..., 1]
                >= int(tracking_config.get("body_selection_min_saturation", 25))
            )
        ).astype(np.uint8) * 255
        evidence_count, _evidence_labels, evidence_stats, _evidence_centroids = (
            cv2.connectedComponentsWithStats(band_evidence_mask, 8)
        )
        evidence_maximum_area = float(tracking_config.get("marker_band_max_area_px", 600))
        selection_minimum_area = float(
            tracking_config.get("body_selection_min_band_area_px", 12)
        )
        evidence_areas = [
            float(evidence_stats[evidence_index, cv2.CC_STAT_AREA])
            for evidence_index in range(1, evidence_count)
            if selection_minimum_area
            <= float(evidence_stats[evidence_index, cv2.CC_STAT_AREA])
            <= evidence_maximum_area
        ]
        band_count = len(evidence_areas)
        band_count_quality = min(band_count / KEYPOINT_COUNT, 1.0)
        colour_fraction = min(float(np.sum(evidence_areas)) / max(area, 1.0), 0.25)
        previous_hits = 0
        for pixel in previous_valid:
            px, py = map(int, np.round(pixel))
            if 0 <= py < height and 0 <= px < width and labels[py, px] == label:
                previous_hits += 1
        score = math.sqrt(area) * min(elongation, 15.0) * (1.15 - min(fill, 1.0))
        score *= (1.0 + 8.0 * band_count_quality) * (1.0 + 60.0 * colour_fraction)
        score *= 1.0 + 0.8 * previous_hits
        if elongation >= minimum_elongation and score > best_score:
            best_label = label
            best_score = score
            best_elongation = elongation
    if best_label < 0:
        return empty

    body_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    body_fraction = float(np.count_nonzero(body_mask)) / max(height * width, 1)
    if body_fraction > maximum_fraction or best_elongation < minimum_elongation:
        return empty
    # Use the connected body itself for the geodesic diameter.  Compute topology
    # on a reduced mask and map the path back: final band centroids remain at full
    # resolution while Python graph traversal stays inexpensive at video rate.
    path_scale = float(np.clip(tracking_config.get("body_path_scale", 0.40), 0.20, 1.0))
    if path_scale < 0.999:
        path_source = cv2.dilate(
            body_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        path_mask = cv2.resize(
            path_source,
            (
                max(2, int(round(width * path_scale))),
                max(2, int(round(height * path_scale))),
            ),
            interpolation=cv2.INTER_NEAREST,
        )
        path_mask = cv2.morphologyEx(
            path_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        centerline = _longest_skeleton_path(path_mask)
        if len(centerline):
            centerline = (centerline + 0.5) / path_scale - 0.5
    else:
        centerline = _longest_skeleton_path(body_mask)
    if len(centerline) < int(tracking_config.get("body_min_centerline_px", 35)):
        return empty
    support_radius = max(1, int(tracking_config.get("body_marker_support_radius_px", 22)))
    search_mask = np.zeros((height, width), dtype=np.uint8)
    centerline_i32 = np.round(centerline).astype(np.int32)
    cv2.polylines(
        search_mask,
        [centerline_i32],
        False,
        255,
        2 * support_radius + 1,
        cv2.LINE_8,
    )
    if len(centerline_i32):
        cv2.circle(search_mask, tuple(centerline_i32[0]), support_radius, 255, -1)
        cv2.circle(search_mask, tuple(centerline_i32[-1]), support_radius, 255, -1)
    search_mask = cv2.bitwise_and(search_mask, roi_mask)

    value_margin = int(tracking_config.get("marker_band_value_margin", 14))
    bright_margin = int(tracking_config.get("marker_band_bright_margin", 34))
    minimum_saturation = int(tracking_config.get("marker_band_min_saturation", 10))
    # Generic ring extraction uses the closed body interior, not the dilated
    # search ROI.  The latter also contains a bright background rim along both
    # body edges, which can connect all seven rings into one false component.
    band_domain = body_mask != 0
    if bool(tracking_config.get("body_tight_band_support_enabled", False)):
        tight_radius = max(
            1, int(tracking_config.get("body_tight_band_support_radius_px", 10))
        )
        tight_support = cv2.dilate(
            dark_seed,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * tight_radius + 1, 2 * tight_radius + 1)
            ),
        )
        band_domain &= tight_support != 0
    band_mask = (
        band_domain
        & (
            hsv[..., 2]
            >= max(
                int(tracking_config.get("marker_band_absolute_value_min", 135)),
                min(255, value_limit + value_margin),
            )
        )
        & (
            (hsv[..., 1] >= minimum_saturation)
            | (hsv[..., 2] >= min(255, value_limit + bright_margin))
        )
    ).astype(np.uint8) * 255
    band_mask = cv2.morphologyEx(
        band_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    band_count, band_labels, band_stats, _band_centroids = cv2.connectedComponentsWithStats(
        band_mask, 8
    )
    minimum_band_area = max(2.0, float(tracking_config.get("minimum_blob_area_px", 3)))
    maximum_band_area = float(tracking_config.get("marker_band_max_area_px", 600))
    path = centerline
    path_step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(path_step)))
    body_distance = cv2.distanceTransform(body_mask, cv2.DIST_L2, 3)
    candidates = []
    for label in range(1, band_count):
        area = float(band_stats[label, cv2.CC_STAT_AREA])
        if not minimum_band_area <= area <= maximum_band_area:
            continue
        component = band_labels == label
        yy, xx = np.nonzero(component)
        center = np.asarray([float(np.mean(xx)), float(np.mean(yy))])
        distances = np.sum((path - center) ** 2, axis=1)
        path_index = int(np.argmin(distances))
        if float(np.sqrt(distances[path_index])) > support_radius + 3.0:
            continue
        # Bright reflections often expose only one side of a narrow ring.  Keep
        # its tangent/arclength coordinate, but move the transverse coordinate
        # to the ridge of the reconstructed tube envelope.
        tangent_radius = min(4, max(1, len(path) // 20))
        tangent = (
            path[min(len(path) - 1, path_index + tangent_radius)]
            - path[max(0, path_index - tangent_radius)]
        )
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > 1e-6:
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32) / tangent_norm
            offsets = np.linspace(-support_radius, support_radius, 2 * support_radius + 1)
            samples = center[None, :] + offsets[:, None] * normal[None, :]
            sample_x = np.clip(np.rint(samples[:, 0]).astype(int), 0, width - 1)
            sample_y = np.clip(np.rint(samples[:, 1]).astype(int), 0, height - 1)
            ridge_score = body_distance[sample_y, sample_x] - 0.025 * np.abs(offsets)
            center = samples[int(np.argmax(ridge_score))]
        values_h = hsv[..., 0][component].astype(np.float32)
        values_s = hsv[..., 1][component].astype(np.float32)
        values_a = lab[..., 1][component].astype(np.float32)
        values_b = lab[..., 2][component].astype(np.float32)
        weights = values_s / 255.0 + 0.12
        observed = np.asarray([
            MarkerTracker._circular_hue_mean(values_h, weights),
            float(np.average(values_a, weights=weights)),
            float(np.average(values_b, weights=weights)),
        ])
        component_mask = np.where(component, 255, 0).astype(np.uint8)
        quality = float(np.clip(area / max(minimum_band_area * 6.0, 1.0), 0.25, 1.0))
        candidates.append({
            "arc": float(arc[path_index]),
            "center": center,
            "area": area,
            "quality": quality,
            "observed": observed,
            "mask": component_mask,
        })
    if len(candidates) < KEYPOINT_COUNT:
        empty.valid = True
        empty.body_mask = body_mask
        empty.search_mask = search_mask
        empty.centerline_xy = centerline
        return empty

    # The seven rings occupy the distal end.  Orient the centerline so K6 is
    # closest to the free tip; after the first frame, K6 history resolves any
    # remaining endpoint ambiguity.
    total_arc = float(arc[-1])
    orientation_mode = str(
        tracking_config.get("marker_tip_orientation", "endpoint_proximity")
    ).lower()
    if np.isfinite(previous_pixels[KEYPOINT_COUNT - 1]).all():
        distance_start = float(np.linalg.norm(centerline[0] - previous_pixels[-1]))
        distance_end = float(np.linalg.norm(centerline[-1] - previous_pixels[-1]))
        tip_at_start = distance_start <= distance_end
    elif orientation_mode in ("auto_color", "auto_hybrid"):
        # On a tube that continues through a guide, an unrelated reflection can
        # sit closer to the base endpoint than the real free-tip ring.  Resolve
        # the first-frame direction with the known K6 colour identity instead
        # of endpoint distance alone.  Later frames still use temporal history.
        # A body that continues into the drive/guide normally crosses the ROI
        # boundary at its base, whereas the free tip terminates inside the ROI.
        # Use this very strong cue before colour, whose hue can be washed out by
        # the green surgical mat and specular ring reflections.
        roi_y, roi_x = np.nonzero(roi_mask)
        roi_left = float(np.min(roi_x)) if len(roi_x) else 0.0
        roi_right = float(np.max(roi_x)) if len(roi_x) else width - 1.0
        roi_top = float(np.min(roi_y)) if len(roi_y) else 0.0
        roi_bottom = float(np.max(roi_y)) if len(roi_y) else height - 1.0

        def boundary_clearance(point: np.ndarray) -> float:
            return min(
                float(point[0]) - roi_left,
                roi_right - float(point[0]),
                float(point[1]) - roi_top,
                roi_bottom - float(point[1]),
            )

        start_clearance = boundary_clearance(centerline[0])
        end_clearance = boundary_clearance(centerline[-1])
        boundary_decisive = (
            orientation_mode == "auto_hybrid"
            and abs(start_clearance - end_clearance) > max(5.0, support_radius * 0.55)
        )
        tip_prototype = prototypes[KEYPOINT_COUNT - 1]
        hue_sigma = max(float(tracking_config.get("hue_sigma", 9.0)), 1.0)
        lab_sigma = max(float(tracking_config.get("lab_chroma_sigma", 32.0)), 1.0)

        def endpoint_identity_cost(at_start: bool) -> float:
            best = math.inf
            nominal_spacing = max(total_arc / 7.0, 1.0)
            for candidate in candidates:
                endpoint_distance = (
                    candidate["arc"] if at_start else total_arc - candidate["arc"]
                )
                if endpoint_distance > total_arc * 0.45:
                    continue
                hue_delta = abs(
                    float(candidate["observed"][0]) - float(tip_prototype["hue"])
                )
                hue_delta = min(hue_delta, 180.0 - hue_delta)
                lab_delta = float(np.linalg.norm(
                    candidate["observed"][1:]
                    - np.asarray([tip_prototype["lab_a"], tip_prototype["lab_b"]])
                ))
                colour_cost = hue_delta / hue_sigma + lab_delta / lab_sigma
                endpoint_cost = 0.20 * endpoint_distance / nominal_spacing
                best = min(best, colour_cost + endpoint_cost)
            return best

        tip_at_start = (
            start_clearance > end_clearance
            if boundary_decisive
            else endpoint_identity_cost(True) <= endpoint_identity_cost(False)
        )
    else:
        closest_start = min(candidate["arc"] for candidate in candidates)
        closest_end = min(total_arc - candidate["arc"] for candidate in candidates)
        tip_at_start = closest_start <= closest_end
    for candidate in candidates:
        candidate["tip_arc"] = (
            candidate["arc"] if tip_at_start else total_arc - candidate["arc"]
        )
    candidates.sort(key=lambda item: item["tip_arc"])
    merge_distance = max(1.0, float(tracking_config.get("marker_band_merge_distance_px", 8.0)))
    clusters: list[list[dict]] = []
    for candidate in candidates:
        if not clusters or candidate["tip_arc"] - clusters[-1][-1]["tip_arc"] > merge_distance:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)
    merged = []
    for cluster in clusters:
        areas = np.asarray([item["area"] for item in cluster], dtype=float)
        weights = areas / max(float(np.sum(areas)), 1e-6)
        merged_mask = np.zeros((height, width), dtype=np.uint8)
        for item in cluster:
            merged_mask = cv2.bitwise_or(merged_mask, item["mask"])
        merged.append({
            "tip_arc": float(np.sum(weights * np.asarray([item["tip_arc"] for item in cluster]))),
            "center": np.sum(weights[:, None] * np.asarray([item["center"] for item in cluster]), axis=0),
            "quality": float(np.clip(np.sum(areas) / max(minimum_band_area * 6.0, 1.0), 0.25, 1.0)),
            "observed": np.sum(weights[:, None] * np.asarray([item["observed"] for item in cluster]), axis=0),
            "mask": merged_mask,
        })
    if len(merged) < KEYPOINT_COUNT:
        empty.valid = True
        empty.body_mask = body_mask
        empty.search_mask = search_mask
        empty.centerline_xy = centerline
        return empty

    maximum_windows = min(len(merged) - KEYPOINT_COUNT + 1, 8)
    best_window = None
    best_window_cost = math.inf
    hue_sigma = max(float(tracking_config.get("hue_sigma", 9.0)), 1.0)
    lab_sigma = max(float(tracking_config.get("lab_chroma_sigma", 32.0)), 1.0)
    for start in range(maximum_windows):
        window = merged[start:start + KEYPOINT_COUNT]
        positions = np.asarray([item["tip_arc"] for item in window])
        gaps = np.diff(positions)
        if np.any(gaps <= 0):
            continue
        spacing_cost = float(np.std(gaps) / max(np.mean(gaps), 1e-6))
        tip_cost = float(positions[0] / max(np.mean(gaps), 1.0)) * float(
            tracking_config.get("marker_tip_anchor_weight", 2.5)
        )
        color_cost = 0.0
        temporal_cost = 0.0
        for tip_order, item in enumerate(window):
            marker_index = KEYPOINT_COUNT - 1 - tip_order
            prototype = prototypes[marker_index]
            hue_delta = abs(float(item["observed"][0]) - float(prototype["hue"]))
            hue_delta = min(hue_delta, 180.0 - hue_delta)
            lab_delta = float(np.linalg.norm(
                item["observed"][1:] - np.asarray([prototype["lab_a"], prototype["lab_b"]])
            ))
            color_cost += min(3.0, hue_delta / hue_sigma + lab_delta / lab_sigma)
            previous = previous_pixels[marker_index]
            if np.isfinite(previous).all():
                temporal_cost += min(3.0, float(np.linalg.norm(item["center"] - previous)) / 30.0)
        cost = (
            spacing_cost * 6.0
            + tip_cost
            + color_cost / KEYPOINT_COUNT
            * float(tracking_config.get("marker_sequence_color_weight", 0.08))
            + temporal_cost / KEYPOINT_COUNT
        )
        if cost < best_window_cost:
            best_window_cost = cost
            best_window = window
    if best_window is None:
        empty.valid = True
        empty.body_mask = body_mask
        empty.search_mask = search_mask
        empty.centerline_xy = centerline
        return empty

    ordered_pixels = np.full((KEYPOINT_COUNT, 2), np.nan)
    ordered_confidence = np.zeros(KEYPOINT_COUNT, dtype=float)
    ordered_masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(KEYPOINT_COUNT)]
    ordered_colours = np.full((KEYPOINT_COUNT, 3), np.nan)
    sequence_quality = float(np.clip(1.0 / (1.0 + best_window_cost), 0.50, 1.0))
    for tip_order, item in enumerate(best_window):
        marker_index = KEYPOINT_COUNT - 1 - tip_order
        ordered_pixels[marker_index] = item["center"]
        ordered_confidence[marker_index] = item["quality"] * sequence_quality
        ordered_masks[marker_index] = item["mask"]
        ordered_colours[marker_index] = item["observed"]
    return ContinuumSegmentation(
        valid=True,
        body_mask=body_mask,
        search_mask=search_mask,
        centerline_xy=centerline,
        ordered_band_pixels=ordered_pixels,
        ordered_band_confidence=ordered_confidence,
        ordered_band_masks=ordered_masks,
        ordered_band_colours=ordered_colours,
    )


def detect_color_bands(
    hsv: np.ndarray,
    lab: np.ndarray,
    marker_config: list[dict],
    prototypes: list[dict[str, float]],
    tracking_config: dict,
    previous_pixels: np.ndarray,
    roi_normalized: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, ContinuumSegmentation]:
    """Segment the black body first, then identify the seven coloured bands."""

    height, width = hsv.shape[:2]
    segmentation = segment_black_continuum(
        hsv,
        lab,
        marker_config,
        prototypes,
        tracking_config,
        previous_pixels,
        roi_normalized,
    )
    saturation = int(tracking_config.get(
        "segmented_minimum_saturation" if segmentation.valid else "minimum_saturation",
        18 if segmentation.valid else 55,
    ))
    value = int(tracking_config.get("minimum_value", 35))
    mask = cv2.inRange(
        hsv,
        np.asarray([0, saturation, value], np.uint8),
        np.asarray([179, 255, 255], np.uint8),
    )
    x0, y0, x1, y1 = roi_normalized
    if x0 > 0.0 or y0 > 0.0 or x1 < 1.0 or y1 < 1.0:
        left, top = int(np.floor(x0 * width)), int(np.floor(y0 * height))
        right, bottom = int(np.ceil(x1 * width)), int(np.ceil(y1 * height))
        roi_mask = np.zeros_like(mask)
        roi_mask[max(0, top):min(height, bottom), max(0, left):min(width, right)] = mask[
            max(0, top):min(height, bottom), max(0, left):min(width, right)
        ]
        mask = roi_mask
    if segmentation.valid:
        mask = cv2.bitwise_and(mask, segmentation.search_mask)
    minimum = float(tracking_config["minimum_blob_area_px"])
    maximum = float(tracking_config["maximum_blob_area_px"])
    hue_sigma = max(float(tracking_config.get("hue_sigma", 9.0)), 1e-3)
    lab_sigma = max(float(tracking_config.get("lab_chroma_sigma", 32.0)), 1e-3)
    temporal_sigma = max(float(tracking_config.get("temporal_sigma_px", 42.0)), 1e-3)
    maximum_jump = float(tracking_config.get("maximum_temporal_jump_px", 110.0))
    hue_gate = max(18.0, 2.0 * float(tracking_config.get("adaptive_hue_half_width", 12.0)))
    pixels = np.full((KEYPOINT_COUNT, 2), np.nan)
    confidence = np.zeros(KEYPOINT_COUNT, dtype=float)
    component_masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(KEYPOINT_COUNT)]
    observed_colours = np.full((KEYPOINT_COUNT, 3), np.nan)
    ordered_valid = np.isfinite(segmentation.ordered_band_pixels).all(axis=1)
    if np.count_nonzero(ordered_valid) == KEYPOINT_COUNT:
        return (
            segmentation.ordered_band_pixels.copy(),
            segmentation.ordered_band_confidence.copy(),
            [mask.copy() for mask in segmentation.ordered_band_masks],
            segmentation.ordered_band_colours.copy(),
            segmentation,
        )
    # Classify hue once for the entire image.  This keeps touching adjacent colour
    # rings separate, unlike connected components on a union saturation mask.
    hue_values = np.arange(180, dtype=np.float32)[:, None]
    prototype_hues = np.asarray([prototype["hue"] for prototype in prototypes], dtype=np.float32)[None, :]
    hue_distances = np.abs(hue_values - prototype_hues)
    hue_distances = np.minimum(hue_distances, 180.0 - hue_distances)
    hue_labels = np.argmin(hue_distances, axis=1).astype(np.uint8)
    hue_minimum = np.min(hue_distances, axis=1)
    hue_labels[hue_minimum > hue_gate] = 255
    hue_lut = np.full(256, 255, dtype=np.uint8)
    hue_lut[:180] = hue_labels
    classified = cv2.LUT(hsv[..., 0], hue_lut)
    radius = int(tracking_config.get("morphology_radius_px", 1))
    kernel = None
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    for marker_index, prototype in enumerate(prototypes):
        marker_mask = np.where((classified == marker_index) & (mask != 0), 255, 0).astype(np.uint8)
        if kernel is not None:
            marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_OPEN, kernel)
            marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_CLOSE, kernel)
        contours, _hierarchy = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: tuple[float, np.ndarray, float, np.ndarray, np.ndarray] | None = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not minimum <= area <= maximum:
                continue
            component_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, -1)
            component = component_mask != 0
            yy, xx = np.nonzero(component)
            hues = hsv[..., 0][component].astype(np.float32)
            lab_a = lab[..., 1][component].astype(np.float32)
            lab_b = lab[..., 2][component].astype(np.float32)
            hue_delta = np.abs(hues - float(prototype["hue"]))
            hue_distance = np.minimum(hue_delta, 180.0 - hue_delta)
            lab_distance = np.sqrt(
                (lab_a - float(prototype["lab_a"])) ** 2
                + (lab_b - float(prototype["lab_b"])) ** 2
            )
            likelihood = np.exp(-0.5 * (hue_distance / hue_sigma) ** 2) * np.exp(
                -0.5 * (lab_distance / lab_sigma) ** 2
            )
            saturation_weight = np.sqrt(
                np.clip(hsv[..., 1][component].astype(np.float32) / 255.0, 0.05, 1.0)
            )
            weights = likelihood * saturation_weight + 1e-6
            center = np.asarray([
                float(np.sum(weights * xx) / np.sum(weights)),
                float(np.sum(weights * yy) / np.sum(weights)),
            ])
            previous = previous_pixels[marker_index]
            distance = float(np.linalg.norm(center - previous)) if np.isfinite(previous).all() else 0.0
            if np.isfinite(previous).all() and distance > maximum_jump:
                continue
            temporal_quality = math.exp(-0.5 * (distance / temporal_sigma) ** 2) if np.isfinite(previous).all() else 1.0
            color_quality = float(np.clip(np.mean(likelihood), 0.0, 1.0))
            area_quality = float(np.clip(area / max(minimum * 8.0, 1.0), 0.15, 1.0))
            score = (
                (0.25 + 0.75 * area_quality)
                * (0.10 + 0.90 * color_quality) ** 2
                * (0.15 + 0.85 * temporal_quality)
            )
            quality = float(np.clip(
                area_quality * (0.25 + 0.75 * color_quality) * math.sqrt(max(temporal_quality, 0.0)),
                0.0,
                1.0,
            ))
            observed = np.asarray([
                MarkerTracker._circular_hue_mean(hues, weights),
                float(np.sum(weights * lab_a) / np.sum(weights)),
                float(np.sum(weights * lab_b) / np.sum(weights)),
            ])
            if best is None or score > best[0]:
                best = score, center, quality, observed, component_mask
        if best is not None:
            _score, center, quality, observed, component_mask = best
            pixels[marker_index] = center
            confidence[marker_index] = quality
            component_masks[marker_index] = component_mask
            observed_colours[marker_index] = observed
    return pixels, confidence, component_masks, observed_colours, segmentation


class ConstantVelocityFilter:
    def __init__(self, config: dict) -> None:
        self.accel_std = float(config["kalman_acceleration_std_m_s2"])
        initial_std = float(config["kalman_initial_position_std_m"])
        self.timeout_s = float(config["prediction_timeout_s"])
        self.gate = float(config["mahalanobis_gate"])
        self.x = np.zeros(6, dtype=float)
        self.P = np.eye(6, dtype=float) * initial_std**2
        self.initialized = False
        self.last_ns: int | None = None
        self.last_measurement_ns: int | None = None

    def update(self, measurement: np.ndarray | None, covariance: np.ndarray, now_ns: int) -> tuple[np.ndarray, bool]:
        dt = 0.0 if self.last_ns is None else float(np.clip((now_ns - self.last_ns) * 1e-9, 0.0, 0.2))
        self.last_ns = now_ns
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.accel_std**2
        G = np.vstack((np.eye(3) * (0.5 * dt**2), np.eye(3) * dt))
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ (np.eye(3) * q) @ G.T

        accepted = measurement is not None and np.isfinite(measurement).all()
        if accepted and not self.initialized:
            self.x[:3] = measurement
            self.x[3:] = 0.0
            self.P[:3, :3] = covariance
            self.initialized = True
            self.last_measurement_ns = now_ns
        elif accepted:
            H = np.hstack((np.eye(3), np.zeros((3, 3))))
            innovation = measurement - H @ self.x
            S = H @ self.P @ H.T + covariance
            try:
                mahalanobis = float(innovation.T @ np.linalg.solve(S, innovation))
            except np.linalg.LinAlgError:
                mahalanobis = math.inf
            if mahalanobis <= self.gate:
                K = self.P @ H.T @ np.linalg.inv(S)
                self.x += K @ innovation
                self.P = (np.eye(6) - K @ H) @ self.P
                self.last_measurement_ns = now_ns
            else:
                accepted = False
        if not self.initialized:
            return np.full(3, np.nan), False
        age_s = math.inf if self.last_measurement_ns is None else (now_ns - self.last_measurement_ns) * 1e-9
        if age_s > self.timeout_s:
            return np.full(3, np.nan), False
        return self.x[:3].copy(), not accepted


class MarkerTracker:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.marker_config = config["markers"]
        self.camera_config = config["camera"]
        self.tracking_config = config["tracking"]
        self.performance_config = config.get("performance", {})
        self._numba_cuda_preprocessor = None
        self.filters = [ConstantVelocityFilter(self.tracking_config) for _ in range(KEYPOINT_COUNT)]
        self.last_color_pixels = np.full((KEYPOINT_COUNT, 2), np.nan)
        self.last_continuum_segmentation: ContinuumSegmentation | None = None
        self._last_tracking_gray: np.ndarray | None = None
        self.color_roi_normalized = (0.0, 0.0, 1.0, 1.0)
        self.color_prototypes = self._load_color_prototypes()
        self.transform_base_from_camera = np.asarray(
            config["calibration"]["transform_base_from_camera"], dtype=float
        )

    def _preprocess_colour_spaces(
        self, image_bgr: np.ndarray, target_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resize and convert colour spaces on the selected CPU/GPU backend."""

        backend = str(self.performance_config.get("resolved_preprocess_backend", "cpu"))
        try:
            if backend == "numba_cuda":
                if self._numba_cuda_preprocessor is None:
                    self._numba_cuda_preprocessor = NumbaCudaColourPreprocessor()
                return self._numba_cuda_preprocessor.process(image_bgr, target_size)
            if backend == "cuda":
                uploaded = cv2.cuda_GpuMat()
                uploaded.upload(image_bgr)
                resized = (
                    cv2.cuda.resize(uploaded, target_size, interpolation=cv2.INTER_AREA)
                    if image_bgr.shape[1::-1] != target_size
                    else uploaded
                )
                hsv = cv2.cuda.cvtColor(resized, cv2.COLOR_BGR2HSV).download()
                lab = cv2.cuda.cvtColor(resized, cv2.COLOR_BGR2LAB).download()
                return hsv, lab
            if backend == "opencl":
                uploaded = cv2.UMat(image_bgr)
                resized = (
                    cv2.resize(uploaded, target_size, interpolation=cv2.INTER_AREA)
                    if image_bgr.shape[1::-1] != target_size
                    else uploaded
                )
                hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).get()
                lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).get()
                return hsv, lab
        except Exception:
            # A driver may disappear at runtime.  Fall back permanently instead
            # of losing the camera stream.
            self.performance_config["resolved_preprocess_backend"] = "cpu"
        resized = (
            cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
            if image_bgr.shape[1::-1] != target_size
            else image_bgr
        )
        return (
            cv2.cvtColor(resized, cv2.COLOR_BGR2HSV),
            cv2.cvtColor(resized, cv2.COLOR_BGR2LAB),
        )

    def _load_color_prototypes(self) -> list[dict[str, Any]]:
        saved = self.tracking_config.get("color_prototypes", {})
        prototypes: list[dict[str, Any]] = []
        for marker in self.marker_config:
            bgr = np.asarray(marker["display_bgr"], dtype=np.uint8).reshape(1, 1, 3)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0]
            value = saved.get(marker["name"], {}) if isinstance(saved, dict) else {}
            prototypes.append({
                "hue": float(value.get("hue", hsv[0])),
                "lab_a": float(value.get("lab_a", lab[1])),
                "lab_b": float(value.get("lab_b", lab[2])),
            })
        return prototypes

    def _save_color_prototypes(self) -> None:
        self.tracking_config["color_prototypes"] = {
            marker["name"]: {key: float(value) for key, value in prototype.items()}
            for marker, prototype in zip(self.marker_config, self.color_prototypes)
        }

    @staticmethod
    def _circular_hue_mean(hue: np.ndarray, weights: np.ndarray) -> float:
        angle = np.asarray(hue, dtype=float) * (2.0 * np.pi / 180.0)
        weights = np.asarray(weights, dtype=float)
        sine = float(np.sum(weights * np.sin(angle)))
        cosine = float(np.sum(weights * np.cos(angle)))
        return float((np.arctan2(sine, cosine) * 180.0 / (2.0 * np.pi)) % 180.0)

    def _update_color_prototype(self, index: int, observed: np.ndarray, alpha: float) -> None:
        if not np.isfinite(observed).all() or alpha <= 0.0:
            return
        prototype = self.color_prototypes[index]
        current_angle = float(prototype["hue"]) * (2.0 * np.pi / 180.0)
        observed_angle = float(observed[0]) * (2.0 * np.pi / 180.0)
        vector = (1.0 - alpha) * np.asarray([np.cos(current_angle), np.sin(current_angle)]) + alpha * np.asarray([np.cos(observed_angle), np.sin(observed_angle)])
        prototype["hue"] = float((np.arctan2(vector[1], vector[0]) * 180.0 / (2.0 * np.pi)) % 180.0)
        prototype["lab_a"] = (1.0 - alpha) * float(prototype["lab_a"]) + alpha * float(observed[1])
        prototype["lab_b"] = (1.0 - alpha) * float(prototype["lab_b"]) + alpha * float(observed[2])

    def reset_filters(self) -> None:
        self.filters = [ConstantVelocityFilter(self.tracking_config) for _ in range(KEYPOINT_COUNT)]
        self.last_color_pixels[:] = np.nan
        self.last_continuum_segmentation = None
        self._last_tracking_gray = None

    def set_transform_base_from_camera(self, transform: np.ndarray, ready: bool = True) -> None:
        value = np.asarray(transform, dtype=float).reshape(4, 4)
        if not np.isfinite(value).all():
            raise ValueError("Calibration transform contains non-finite values")
        self.transform_base_from_camera = value.copy()
        self.config["calibration"]["transform_base_from_camera"] = value.tolist()
        self.config["calibration"]["ready"] = bool(ready)

    def set_color_roi(self, roi_normalized: tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = map(float, roi_normalized)
        x0, x1 = sorted((float(np.clip(x0, 0.0, 1.0)), float(np.clip(x1, 0.0, 1.0))))
        y0, y1 = sorted((float(np.clip(y0, 0.0, 1.0)), float(np.clip(y1, 0.0, 1.0))))
        if x1 - x0 < 1e-4 or y1 - y0 < 1e-4:
            raise ValueError("Tracking ROI must have a non-zero area")
        self.color_roi_normalized = (x0, y0, x1, y1)

    def _apply_color_roi(self, mask: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.color_roi_normalized
        if x0 <= 0.0 and y0 <= 0.0 and x1 >= 1.0 and y1 >= 1.0:
            return mask
        height, width = mask.shape
        left = int(np.floor(x0 * width))
        top = int(np.floor(y0 * height))
        right = int(np.ceil(x1 * width))
        bottom = int(np.ceil(y1 * height))
        restricted = np.zeros_like(mask)
        restricted[max(0, top):min(height, bottom), max(0, left):min(width, right)] = mask[
            max(0, top):min(height, bottom), max(0, left):min(width, right)
        ]
        return restricted

    def _color_blob(
        self,
        hsv: np.ndarray,
        lab: np.ndarray,
        marker: dict,
        previous: np.ndarray,
        marker_index: int,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in marker["hsv_ranges"]:
            mask |= cv2.inRange(hsv, np.asarray(low, np.uint8), np.asarray(high, np.uint8))
        prototype = self.color_prototypes[marker_index]
        expected_hue = float(prototype["hue"])
        hue_half_width = float(self.tracking_config.get("adaptive_hue_half_width", 12.0))
        hue_low, hue_high = expected_hue - hue_half_width, expected_hue + hue_half_width
        saturation = int(self.tracking_config.get("minimum_saturation", 55))
        value = int(self.tracking_config.get("minimum_value", 35))
        if hue_low < 0.0:
            adaptive = cv2.inRange(hsv, np.asarray([0, saturation, value], np.uint8), np.asarray([int(np.ceil(hue_high)), 255, 255], np.uint8))
            adaptive |= cv2.inRange(hsv, np.asarray([int(np.floor(180 + hue_low)), saturation, value], np.uint8), np.asarray([179, 255, 255], np.uint8))
        elif hue_high >= 180.0:
            adaptive = cv2.inRange(hsv, np.asarray([int(np.floor(hue_low)), saturation, value], np.uint8), np.asarray([179, 255, 255], np.uint8))
            adaptive |= cv2.inRange(hsv, np.asarray([0, saturation, value], np.uint8), np.asarray([int(np.ceil(hue_high - 180)), 255, 255], np.uint8))
        else:
            adaptive = cv2.inRange(hsv, np.asarray([int(np.floor(hue_low)), saturation, value], np.uint8), np.asarray([int(np.ceil(hue_high)), 255, 255], np.uint8))
        mask |= adaptive
        mask = self._apply_color_roi(mask)
        radius = int(self.tracking_config.get("morphology_radius_px", 1))
        if radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = self._apply_color_roi(mask)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        min_area = float(self.tracking_config["minimum_blob_area_px"])
        max_area = float(self.tracking_config["maximum_blob_area_px"])
        hue_sigma = max(float(self.tracking_config.get("hue_sigma", 9.0)), 1e-3)
        lab_sigma = max(float(self.tracking_config.get("lab_chroma_sigma", 32.0)), 1e-3)
        temporal_sigma = max(float(self.tracking_config.get("temporal_sigma_px", 42.0)), 1e-3)
        maximum_jump = float(self.tracking_config.get("maximum_temporal_jump_px", 110.0))
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if min_area <= area <= max_area:
                component = labels == label
                component_hue = hsv[..., 0][component].astype(np.float32)
                component_hue_delta = np.abs(component_hue - expected_hue)
                component_hue_distance = np.minimum(component_hue_delta, 180.0 - component_hue_delta)
                lab_a = lab[..., 1][component].astype(np.float32)
                lab_b = lab[..., 2][component].astype(np.float32)
                lab_distance = np.sqrt(
                    (lab_a - float(prototype["lab_a"])) ** 2
                    + (lab_b - float(prototype["lab_b"])) ** 2
                )
                likelihood = np.exp(-0.5 * (component_hue_distance / hue_sigma) ** 2) * np.exp(-0.5 * (lab_distance / lab_sigma) ** 2)
                saturation_weight = np.clip(hsv[..., 1][component].astype(np.float32) / 255.0, 0.05, 1.0)
                weights = likelihood * np.sqrt(saturation_weight) + 1e-6
                yy, xx = np.nonzero(component)
                center = np.asarray([
                    float(np.sum(weights * xx) / np.sum(weights)),
                    float(np.sum(weights * yy) / np.sum(weights)),
                ])
                distance = float(np.linalg.norm(center - previous)) if np.isfinite(previous).all() else 0.0
                if np.isfinite(previous).all() and distance > maximum_jump:
                    continue
                temporal_quality = math.exp(-0.5 * (distance / temporal_sigma) ** 2) if np.isfinite(previous).all() else 1.0
                color_quality = float(np.clip(np.mean(likelihood), 0.0, 1.0))
                area_quality = float(np.clip(area / max(min_area * 8.0, 1.0), 0.15, 1.0))
                score = (0.25 + 0.75 * area_quality) * (0.10 + 0.90 * color_quality) ** 2 * (0.15 + 0.85 * temporal_quality)
                observed = np.asarray([
                    self._circular_hue_mean(hsv[..., 0][component], weights),
                    float(np.sum(weights * lab_a) / np.sum(weights)),
                    float(np.sum(weights * lab_b) / np.sum(weights)),
                ])
                candidates.append((score, area, center, label, color_quality, temporal_quality, observed))
        if not candidates:
            return np.full(2, np.nan), 0.0, mask, np.full(3, np.nan)
        _score, area, center, label, color_quality, temporal_quality, observed = max(candidates, key=lambda item: item[0])
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        area_quality = float(np.clip(area / max(min_area * 8.0, 1.0), 0.15, 1.0))
        confidence = float(np.clip(area_quality * (0.25 + 0.75 * color_quality) * math.sqrt(max(temporal_quality, 0.0)), 0.0, 1.0))
        return np.asarray(center, dtype=float), confidence, component_mask, observed

    def _robust_depth(self, depth: np.ndarray, pixel: np.ndarray, component_mask: np.ndarray) -> tuple[float, int]:
        if not np.isfinite(pixel).all():
            return math.nan, 0
        radius = int(self.camera_config["depth_roi_radius_px"])
        u, v = map(int, np.round(pixel))
        y0, y1 = max(0, v - radius), min(depth.shape[0], v + radius + 1)
        x0, x1 = max(0, u - radius), min(depth.shape[1], u + radius + 1)
        values = depth[y0:y1, x0:x1]
        selected = None
        if component_mask.shape == depth.shape:
            selected = component_mask[y0:y1, x0:x1] > 0
        elif component_mask.size:
            scale_x = component_mask.shape[1] / float(depth.shape[1])
            scale_y = component_mask.shape[0] / float(depth.shape[0])
            mx0 = int(np.floor(x0 * scale_x))
            mx1 = int(np.ceil(x1 * scale_x))
            my0 = int(np.floor(y0 * scale_y))
            my1 = int(np.ceil(y1 * scale_y))
            reduced = component_mask[
                max(0, my0):min(component_mask.shape[0], my1),
                max(0, mx0):min(component_mask.shape[1], mx1),
            ]
            if reduced.size:
                selected = cv2.resize(
                    reduced,
                    (values.shape[1], values.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
        if selected is not None:
            marked_values = values[selected]
            if np.count_nonzero(np.isfinite(marked_values) & (marked_values > 0)) >= 3:
                values = marked_values
        minimum = float(self.camera_config["depth_min_m"])
        maximum = float(self.camera_config["depth_max_m"])
        valid = values[np.isfinite(values) & (values >= minimum) & (values <= maximum)]
        if len(valid) < 2:
            return math.nan, int(len(valid))
        median = float(np.median(valid))
        mad = float(np.median(np.abs(valid - median)))
        if mad > 0:
            valid = valid[np.abs(valid - median) <= 3.5 * 1.4826 * mad]
        return (float(np.median(valid)), int(len(valid))) if len(valid) else (math.nan, 0)

    @staticmethod
    def _ir_centroid(image: np.ndarray, predicted: np.ndarray, radius: int) -> tuple[np.ndarray, float]:
        if not np.isfinite(predicted).all():
            return np.full(2, np.nan), 0.0
        u, v = map(int, np.round(predicted))
        y0, y1 = max(0, v - radius), min(image.shape[0], v + radius + 1)
        x0, x1 = max(0, u - radius), min(image.shape[1], u + radius + 1)
        patch = image[y0:y1, x0:x1].astype(float)
        if patch.size < 9:
            return np.full(2, np.nan), 0.0
        median = float(np.median(patch))
        dark = np.clip(median - patch, 0.0, None)
        bright = np.clip(patch - median, 0.0, None)
        weights = dark if dark.sum() >= bright.sum() else bright
        contrast = float(np.percentile(patch, 90) - np.percentile(patch, 10))
        if weights.sum() <= 1e-6 or contrast < 4.0:
            return predicted.copy(), 0.15
        yy, xx = np.mgrid[y0:y1, x0:x1]
        center = np.asarray([(weights * xx).sum(), (weights * yy).sum()]) / weights.sum()
        return center, float(np.clip(contrast / 80.0, 0.15, 1.0))

    def _match_right_epipolar(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        left_pixel: np.ndarray,
        right_prediction: np.ndarray,
        search_radius_override: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Match the same IR texture along the rectified epipolar line using ZNCC."""

        invalid = (np.full(2, np.nan), 0.0)
        if not np.isfinite(left_pixel).all() or not np.isfinite(right_prediction).all():
            return invalid
        patch_radius = int(self.camera_config.get("ir_patch_radius_px", 4))
        search_radius = int(
            self.camera_config.get("ir_epipolar_search_radius_px", 10)
            if search_radius_override is None
            else search_radius_override
        )
        vertical_radius = int(self.camera_config.get("ir_epipolar_vertical_radius_px", 2))
        lu, lv = map(int, np.round(left_pixel))
        if (
            lu - patch_radius < 0 or lu + patch_radius >= left_image.shape[1]
            or lv - patch_radius < 0 or lv + patch_radius >= left_image.shape[0]
        ):
            return invalid
        template = left_image[
            lv - patch_radius:lv + patch_radius + 1,
            lu - patch_radius:lu + patch_radius + 1,
        ].astype(np.uint8)
        if float(np.std(template)) < 2.0:
            return invalid
        ru, rv = map(int, np.round(right_prediction))
        x0 = max(0, ru - search_radius - patch_radius)
        x1 = min(right_image.shape[1], ru + search_radius + patch_radius + 1)
        y0 = max(0, rv - vertical_radius - patch_radius)
        y1 = min(right_image.shape[0], rv + vertical_radius + patch_radius + 1)
        search = right_image[y0:y1, x0:x1].astype(np.uint8)
        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            return invalid
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _minimum, quality, _min_location, location = cv2.minMaxLoc(scores)
        if not np.isfinite(quality) or quality < float(self.camera_config.get("ir_min_zncc", 0.45)):
            return invalid
        best_x, best_y = location
        delta_x = 0.0
        if 0 < best_x < scores.shape[1] - 1:
            left_score = float(scores[best_y, best_x - 1])
            center_score = float(scores[best_y, best_x])
            right_score = float(scores[best_y, best_x + 1])
            denominator = left_score - 2.0 * center_score + right_score
            if abs(denominator) > 1e-6:
                delta_x = float(np.clip(0.5 * (left_score - right_score) / denominator, -0.5, 0.5))
        center = np.asarray([
            x0 + best_x + patch_radius + delta_x,
            y0 + best_y + patch_radius,
        ], dtype=float)
        return center, float(np.clip(quality, 0.0, 1.0))

    def _measure_one(
        self,
        frame: CameraFrame,
        color_pixel: np.ndarray,
        blob_confidence: float,
        component_mask: np.ndarray,
        previous_depth_m: float = math.nan,
        depth_measurement: tuple[float, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
        invalid = (np.full(3, np.nan), np.full((3, 3), np.nan), np.full(2, np.nan), np.full(2, np.nan), 0.0, "invalid")
        depth_m, depth_count = (
            self._robust_depth(frame.depth_m, color_pixel, component_mask)
            if depth_measurement is None
            else depth_measurement
        )
        measured_depth = bool(np.isfinite(depth_m))
        if not measured_depth:
            depth_m = (
                float(previous_depth_m)
                if np.isfinite(previous_depth_m)
                else float(self.camera_config.get("depth_seed_m", 0.30))
            )
        if not np.isfinite(depth_m) or depth_m <= 0.0:
            return invalid
        point_color = _deproject(color_pixel, depth_m, frame.color_intrinsics)
        point_left_seed = _transform(frame.transform_left_from_color, point_color)
        point_right_seed = _transform(frame.transform_right_from_left, point_left_seed)
        left_prediction = _project(point_left_seed, frame.left_intrinsics)
        right_prediction = _project(point_right_seed, frame.right_intrinsics)
        radius = int(self.camera_config["ir_refine_radius_px"])
        left_pixel, left_quality = self._ir_centroid(frame.infrared_left, left_prediction, radius)
        right_pixel, right_quality = self._match_right_epipolar(
            frame.infrared_left,
            frame.infrared_right,
            left_pixel,
            right_prediction,
            None
            if measured_depth
            else int(self.camera_config.get("ir_epipolar_fallback_search_radius_px", 28)),
        )
        if not np.isfinite(right_pixel).all():
            right_pixel, right_quality = self._ir_centroid(frame.infrared_right, right_prediction, radius)
            right_quality *= 0.60

        vertical_error = abs(float(left_pixel[1] - right_pixel[1])) if np.isfinite(left_pixel).all() and np.isfinite(right_pixel).all() else math.inf
        disparity = float(left_pixel[0] - right_pixel[0]) if np.isfinite(left_pixel).all() and np.isfinite(right_pixel).all() else math.nan
        baseline = float(np.linalg.norm(frame.transform_right_from_left[:3, 3]))
        fx = float(frame.left_intrinsics["fx"])
        use_stereo = disparity > 0.2 and baseline > 1e-4 and vertical_error <= float(self.camera_config["max_vertical_stereo_error_px"])
        if use_stereo:
            z = fx * baseline / disparity
            point = _deproject(left_pixel, z, frame.left_intrinsics)
            disagreement = abs(float(z - point_left_seed[2]))
            stereo_quality = min(left_quality, right_quality)
            # A high-quality IR correspondence is the primary keypoint depth.
            # Aligned depth remains a sanity check, with a wider gate when ZNCC is strong.
            allowed_disagreement = float(self.camera_config["max_depth_disagreement_m"]) + 0.018 * stereo_quality
            use_stereo = (not measured_depth) or disagreement <= allowed_disagreement
        if use_stereo:
            disparity_std = 0.35 / max(min(left_quality, right_quality), 0.15)
            sigma_z = max(0.00025, point[2] ** 2 * disparity_std / (fx * baseline))
            sigma_xy = max(0.00015, point[2] * 0.35 / fx)
            covariance = np.diag([sigma_xy**2, sigma_xy**2, sigma_z**2])
            confidence = blob_confidence * min(left_quality, right_quality) * math.exp(-vertical_error)
            return point, covariance, left_pixel, right_pixel, float(np.clip(confidence, 0.0, 1.0)), "stereo_ir"

        if not measured_depth:
            return invalid
        sigma_z = max(0.0010, 0.004 * depth_m) / math.sqrt(max(depth_count, 1))
        sigma_xy = max(0.0003, depth_m / float(frame.color_intrinsics["fx"]))
        point = point_left_seed
        covariance = np.diag([sigma_xy**2, sigma_xy**2, sigma_z**2])
        confidence = blob_confidence * min(1.0, depth_count / 8.0) * 0.65
        return point, covariance, left_prediction, right_prediction, float(np.clip(confidence, 0.0, 1.0)), "aligned_depth"

    def process(self, frame: CameraFrame) -> KeypointObservation:
        color_height, color_width = frame.color_bgr.shape[:2]
        processing_max_width = max(320, int(self.tracking_config.get("processing_max_width_px", 416)))
        processing_scale = min(1.0, processing_max_width / float(color_width))
        processing_size = (
            int(round(color_width * processing_scale)),
            int(round(color_height * processing_scale)),
        )
        if processing_scale < 0.999:
            previous_pixels = self.last_color_pixels * processing_scale
            processing_tracking_config = scaled_tracking_config(
                self.tracking_config, processing_scale
            )
        else:
            previous_pixels = self.last_color_pixels
            processing_tracking_config = self.tracking_config
        hsv, lab = self._preprocess_colour_spaces(frame.color_bgr, processing_size)
        raw = np.full((KEYPOINT_COUNT, 3), np.nan)
        filtered = np.full_like(raw, np.nan)
        base = np.full_like(raw, np.nan)
        covariance = np.full((KEYPOINT_COUNT, 3, 3), np.nan)
        valid = np.zeros(KEYPOINT_COUNT, dtype=bool)
        predicted = np.zeros(KEYPOINT_COUNT, dtype=bool)
        confidence = np.zeros(KEYPOINT_COUNT, dtype=float)
        color_pixels = np.full((KEYPOINT_COUNT, 2), np.nan)
        left_pixels = np.full_like(color_pixels, np.nan)
        right_pixels = np.full_like(color_pixels, np.nan)
        sources: list[str] = []

        detected_pixels, blob_confidences, component_masks, observed_colours, segmentation = detect_color_bands(
            hsv,
            lab,
            self.marker_config,
            self.color_prototypes,
            processing_tracking_config,
            previous_pixels,
            self.color_roi_normalized,
        )
        tracking_gray = hsv[..., 2]
        previous_valid = np.isfinite(previous_pixels).all(axis=1)
        if (
            bool(self.tracking_config.get("optical_flow_fallback_enabled", True))
            and self._last_tracking_gray is not None
            and self._last_tracking_gray.shape == tracking_gray.shape
            and np.any(previous_valid)
        ):
            previous_indices = np.flatnonzero(previous_valid)
            previous_points = previous_pixels[previous_indices].astype(np.float32).reshape(-1, 1, 2)
            flowed, flow_status, flow_error = cv2.calcOpticalFlowPyrLK(
                self._last_tracking_gray,
                tracking_gray,
                previous_points,
                None,
                winSize=(21, 21),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
            )
            if flowed is not None and flow_status is not None:
                maximum_error = float(
                    self.tracking_config.get("optical_flow_max_error", 28.0)
                )
                replacement_distance = float(
                    self.tracking_config.get("optical_flow_replacement_distance_px", 18.0)
                ) * processing_scale
                flow_confidence = float(
                    self.tracking_config.get("optical_flow_confidence", 0.32)
                )
                for local_index, marker_index in enumerate(previous_indices):
                    candidate = flowed[local_index, 0].astype(float)
                    error = float(flow_error[local_index, 0]) if flow_error is not None else 0.0
                    if (
                        not bool(flow_status[local_index, 0])
                        or not np.isfinite(candidate).all()
                        or error > maximum_error
                    ):
                        continue
                    u, v = map(int, np.round(candidate))
                    if not (0 <= u < tracking_gray.shape[1] and 0 <= v < tracking_gray.shape[0]):
                        continue
                    if segmentation.valid and not bool(segmentation.search_mask[v, u]):
                        continue
                    detected = detected_pixels[marker_index]
                    should_replace = (
                        not np.isfinite(detected).all()
                        or blob_confidences[marker_index] < 0.20
                        or float(np.linalg.norm(detected - candidate)) > replacement_distance
                    )
                    if not should_replace:
                        continue
                    detected_pixels[marker_index] = candidate
                    blob_confidences[marker_index] = flow_confidence
                    marker_mask = np.zeros(tracking_gray.shape, dtype=np.uint8)
                    cv2.circle(
                        marker_mask,
                        (u, v),
                        max(2, int(processing_tracking_config.get("morphology_radius_px", 1)) + 2),
                        255,
                        -1,
                    )
                    component_masks[marker_index] = marker_mask
        self._last_tracking_gray = tracking_gray.copy()
        self.last_continuum_segmentation = segmentation
        if processing_scale < 0.999:
            detected_pixels /= processing_scale

        depth_measurements = [
            self._robust_depth(frame.depth_m, detected_pixels[index], component_masks[index])
            for index in range(KEYPOINT_COUNT)
        ]
        measured_depths = np.asarray(
            [value for value, _count in depth_measurements if np.isfinite(value)], dtype=float
        )
        frame_depth_seed = (
            float(np.median(measured_depths))
            if len(measured_depths)
            else float(self.camera_config.get("depth_seed_m", 0.30))
        )

        for index, marker in enumerate(self.marker_config):
            pixel = detected_pixels[index]
            blob_confidence = blob_confidences[index]
            component_mask = component_masks[index]
            if np.isfinite(pixel).all():
                self.last_color_pixels[index] = pixel
            point, point_cov, left_pixel, right_pixel, point_conf, source = self._measure_one(
                frame,
                pixel,
                blob_confidence,
                component_mask,
                self.filters[index].x[2]
                if self.filters[index].initialized
                else frame_depth_seed,
                depth_measurements[index],
            )
            is_valid = np.isfinite(point).all() and point_conf > 0.02
            filter_point, was_predicted = self.filters[index].update(
                point if is_valid else None,
                point_cov if is_valid else np.eye(3) * 1e-4,
                frame.capture_host_ns,
            )
            raw[index] = point
            filtered[index] = filter_point
            covariance[index] = point_cov
            valid[index] = is_valid
            predicted[index] = bool(was_predicted and np.isfinite(filter_point).all())
            confidence[index] = point_conf
            color_pixels[index] = pixel
            left_pixels[index] = left_pixel
            right_pixels[index] = right_pixel
            sources.append(source)
            if np.isfinite(filter_point).all():
                base[index] = _transform(self.transform_base_from_camera, filter_point)
            if np.isfinite(pixel).all() and blob_confidence >= 0.25:
                alpha = float(self.tracking_config.get("adaptive_color_alpha", 0.025))
                self._update_color_prototype(index, observed_colours[index], alpha)

        self._save_color_prototypes()

        return KeypointObservation(
            raw_camera_m=raw,
            filtered_camera_m=filtered,
            base_m=base,
            covariance_m2=covariance,
            valid=valid,
            predicted=predicted,
            confidence=confidence,
            color_pixels=color_pixels,
            left_pixels=left_pixels,
            right_pixels=right_pixels,
            source=tuple(sources),
            color_confidence=blob_confidences,
            fusion_mode="d435_internal",
        )

    def calibrate_color_prototypes(
        self,
        frame: CameraFrame,
        observation: KeypointObservation,
    ) -> int:
        """Calibrate the seven colour models from the currently matched marker patches."""

        hsv = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2LAB)
        calibrated = 0
        radius = max(3, int(self.camera_config.get("depth_roi_radius_px", 4)) + 2)
        for index, pixel in enumerate(observation.color_pixels):
            if (
                not np.isfinite(pixel).all()
                or observation.color_confidence[index] < 0.05
            ):
                continue
            u, v = map(int, np.round(pixel))
            y0, y1 = max(0, v - radius), min(hsv.shape[0], v + radius + 1)
            x0, x1 = max(0, u - radius), min(hsv.shape[1], u + radius + 1)
            hsv_patch = hsv[y0:y1, x0:x1]
            lab_patch = lab[y0:y1, x0:x1]
            selected = (
                (
                    hsv_patch[..., 1]
                    >= int(self.tracking_config.get("segmented_minimum_saturation", 12))
                )
                & (hsv_patch[..., 2] >= int(self.tracking_config.get("minimum_value", 35)))
            )
            if np.count_nonzero(selected) < 4:
                continue
            weights = hsv_patch[..., 1][selected].astype(float) / 255.0 + 0.05
            observed = np.asarray([
                self._circular_hue_mean(hsv_patch[..., 0][selected], weights),
                float(np.average(lab_patch[..., 1][selected], weights=weights)),
                float(np.average(lab_patch[..., 2][selected], weights=weights)),
            ])
            self._update_color_prototype(index, observed, 1.0)
            calibrated += 1
        self._save_color_prototypes()
        return calibrated

    def draw_overlay(self, frame: CameraFrame, observation: KeypointObservation) -> np.ndarray:
        image = frame.color_bgr.copy()
        segmentation = self.last_continuum_segmentation
        if segmentation is not None and segmentation.valid:
            body_mask = segmentation.body_mask
            if body_mask.shape != image.shape[:2]:
                body_mask = cv2.resize(
                    body_mask, image.shape[1::-1], interpolation=cv2.INTER_NEAREST
                )
            contours, _hierarchy = cv2.findContours(
                body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours, -1, (70, 220, 255), 1, cv2.LINE_AA)
            centerline = segmentation.centerline_xy.copy()
            if len(centerline):
                centerline[:, 0] *= image.shape[1] / float(segmentation.body_mask.shape[1])
                centerline[:, 1] *= image.shape[0] / float(segmentation.body_mask.shape[0])
                cv2.polylines(
                    image,
                    [np.round(centerline).astype(np.int32)],
                    False,
                    (70, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                image,
                "BLACK BODY LOCK",
                (12, 47),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (70, 220, 255),
                1,
                cv2.LINE_AA,
            )
        x0, y0, x1, y1 = self.color_roi_normalized
        if x0 > 0.0 or y0 > 0.0 or x1 < 1.0 or y1 < 1.0:
            height, width = image.shape[:2]
            corner0 = (int(round(x0 * width)), int(round(y0 * height)))
            corner1 = (int(round(x1 * width)) - 1, int(round(y1 * height)) - 1)
            cv2.rectangle(image, corner0, corner1, (255, 210, 70), 2, cv2.LINE_AA)
            cv2.putText(image, "ACTIVE TRACKING ROI", (corner0[0] + 7, max(20, corner0[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 210, 70), 1, cv2.LINE_AA)
        for index, marker in enumerate(self.marker_config):
            pixel = observation.color_pixels[index]
            if not np.isfinite(pixel).all():
                continue
            center = tuple(int(round(value)) for value in pixel)
            colour = tuple(int(value) for value in marker["display_bgr"])
            cv2.circle(image, center, 9, colour, 2, cv2.LINE_AA)
            label = f"K{index} C{observation.color_confidence[index]:.2f} 3D{observation.confidence[index]:.2f} {observation.source[index]}"
            cv2.putText(image, label, (center[0] + 10, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        valid_count = int(np.count_nonzero(observation.valid))
        mode_label = "D435 RGB+IR+Depth" if observation.fusion_mode == "d435_internal" else "D435 + side RGB"
        cv2.putText(image, f"frame {frame.sequence} | valid {valid_count}/7 | {mode_label}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        return image
