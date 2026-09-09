"""Continuous 3-D shape reconstruction from the seven arc-length markers.

The visual/multimodal pipeline estimates seven material points.  This module
turns those sparse observations into a C1-continuous centreline and a tube mesh
without assuming that the robot is a single constant-curvature arc.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from Visual_information.d435_tdcr_capture.models import KEYPOINT_ARCLENGTHS_M, KEYPOINT_COUNT

from .models import SevenMarkerResult


TOTAL_ACTIVE_LENGTH_M = float(KEYPOINT_ARCLENGTHS_M[-1])
DEFAULT_DIAMETER_M = 0.0035
MAX_MARKER_CHORD_M = 0.007001
MIN_MARKER_CHORD_M = 0.0055


def _two_section_c1_basis() -> np.ndarray:
    """Linear C1 piecewise-quadratic basis at the seven material markers.

    The first and second 21 mm steering sections have independent curvature
    vectors, while sharing position and tangent at K3.  This is the lowest
    dimensional shape family that represents two-section TDCR bending without
    allowing the physically impossible point-to-point alternating snake.
    """

    rows = []
    for index in range(KEYPOINT_COUNT):
        if index <= 3:
            x = float(index)
            rows.append([1.0, x, x * x, 0.0])
        else:
            u = float(index - 3)
            # p3=p0+3v+9c1 and p'(3)=v+6c1.
            rows.append([1.0, 3.0 + u, 9.0 + 6.0 * u, u * u])
    return np.asarray(rows, dtype=float)


def project_two_section_c1(
    points_m: np.ndarray,
    confidence: np.ndarray | None = None,
    measured_valid: np.ndarray | None = None,
    robust_iterations: int = 5,
) -> np.ndarray:
    """Robustly remove non-physical zig-zag while retaining two-section bend.

    Direct depth observations receive more weight than temporal/model points,
    but no single observation can create alternating curvature.  A small
    curvature ridge prevents an under-observed section from curling solely to
    interpolate depth holes.
    """

    anchors = _fill_missing_markers(points_m)
    if not np.isfinite(anchors).all():
        return anchors
    base = _two_section_c1_basis()
    confidence_array = (
        np.full(KEYPOINT_COUNT, 0.25)
        if confidence is None
        else np.asarray(confidence, dtype=float).reshape(KEYPOINT_COUNT)
    )
    measured = (
        np.zeros(KEYPOINT_COUNT, dtype=bool)
        if measured_valid is None
        else np.asarray(measured_valid, dtype=bool).reshape(KEYPOINT_COUNT)
    )
    data_weight = np.clip(confidence_array, 0.05, 1.0) * np.where(measured, 4.0, 0.9)
    robust_weight = np.ones(KEYPOINT_COUNT, dtype=float)
    coefficients = np.zeros((4, 3), dtype=float)
    # Penalise independent curvature coefficients, not translation/tangent.
    ridge = np.diag([0.0, 0.0, 0.08, 0.08])
    for _ in range(max(2, int(robust_iterations))):
        weight = np.sqrt(np.maximum(data_weight * robust_weight, 1e-5))
        design = np.vstack((base * weight[:, None], ridge))
        target = np.vstack((anchors * weight[:, None], np.zeros((4, 3))))
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        fitted = base @ coefficients
        residual = np.linalg.norm(fitted - anchors, axis=1)
        scale = max(0.00045, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        robust_weight = np.minimum(1.0, 1.5 * scale / np.maximum(residual, 1e-9))
    return base @ coefficients


def _readonly(value: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape is not None:
        array = array.reshape(shape)
    array = array.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ContinuumShape3D:
    marker_points_m: np.ndarray
    centerline_m: np.ndarray
    arclength_m: np.ndarray
    tangent: np.ndarray
    curvature_1_m: np.ndarray
    total_length_m: float
    maximum_curvature_1_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "marker_points_m", _readonly(self.marker_points_m, (KEYPOINT_COUNT, 3)))
        samples = len(np.asarray(self.centerline_m))
        object.__setattr__(self, "centerline_m", _readonly(self.centerline_m, (samples, 3)))
        object.__setattr__(self, "arclength_m", _readonly(self.arclength_m, (samples,)))
        object.__setattr__(self, "tangent", _readonly(self.tangent, (samples, 3)))
        object.__setattr__(self, "curvature_1_m", _readonly(self.curvature_1_m, (samples,)))


def _fill_missing_markers(points: np.ndarray) -> np.ndarray:
    output = np.asarray(points, dtype=float).reshape(KEYPOINT_COUNT, 3).copy()
    valid = np.isfinite(output).all(axis=1)
    if np.count_nonzero(valid) < 2:
        return output
    indices = np.arange(KEYPOINT_COUNT, dtype=float)
    valid_indices = indices[valid]
    for coordinate in range(3):
        output[:, coordinate] = np.interp(
            indices, valid_indices, output[valid, coordinate]
        )
    # Linear extrapolation is preferable to np.interp's constant endpoint when
    # an end marker is temporarily missing.
    first, last = np.flatnonzero(valid)[[0, -1]]
    if first > 0:
        delta = output[first + 1] - output[first]
        for index in range(first - 1, -1, -1):
            output[index] = output[index + 1] - delta
    if last < KEYPOINT_COUNT - 1:
        delta = output[last] - output[last - 1]
        for index in range(last + 1, KEYPOINT_COUNT):
            output[index] = output[index - 1] + delta
    return output


def project_material_keypoints(
    points_m: np.ndarray,
    confidence: np.ndarray | None = None,
    measured_valid: np.ndarray | None = None,
    iterations: int = 36,
) -> np.ndarray:
    """Project seven points onto the physically possible material-chain set.

    The points are 7 mm apart in *arc length*.  Their Euclidean chord can be a
    little shorter during strong bending, but can never exceed 7 mm.  A very
    short chord is also a reliable sign that two rings were assigned to the
    same reflection.  Position-based constraints correct those failures while
    weighting direct measurements more strongly than predictions.
    """

    anchors = project_two_section_c1(points_m, confidence, measured_valid)
    if not np.isfinite(anchors).all():
        return anchors
    output = anchors.copy()
    confidence_array = np.ones(KEYPOINT_COUNT) * 0.25 if confidence is None else np.asarray(
        confidence, dtype=float
    ).reshape(KEYPOINT_COUNT)
    measured = np.zeros(KEYPOINT_COUNT, dtype=bool) if measured_valid is None else np.asarray(
        measured_valid, dtype=bool
    ).reshape(KEYPOINT_COUNT)
    anchor_weight = np.clip(confidence_array, 0.04, 1.0) * np.where(measured, 5.0, 0.7)
    mobility = 1.0 / (0.20 + anchor_weight)

    def constrain_segment(first: int, second: int) -> None:
        delta = output[second] - output[first]
        distance = float(np.linalg.norm(delta))
        if distance < 1e-10:
            fallback = anchors[second] - anchors[first]
            fallback_norm = float(np.linalg.norm(fallback))
            direction = fallback / fallback_norm if fallback_norm > 1e-10 else np.asarray([1.0, 0.0, 0.0])
            distance = 1e-10
        else:
            direction = delta / distance
        if distance > MAX_MARKER_CHORD_M:
            target = 0.00698
            gain = 0.92
        elif distance < MIN_MARKER_CHORD_M:
            target = 0.00635
            gain = 0.48
        else:
            return
        correction = direction * (distance - target) * gain
        total_mobility = mobility[first] + mobility[second]
        output[first] += correction * mobility[first] / total_mobility
        output[second] -= correction * mobility[second] / total_mobility

    iterations = max(8, int(iterations))
    for iteration in range(iterations):
        # Data attraction is disabled in the final passes so the returned state
        # satisfies the hard no-stretch limit rather than ending on a data step.
        if iteration < iterations - 10:
            attraction = 0.018 * np.clip(anchor_weight, 0.0, 5.0)[:, None]
            output += attraction * (anchors - output)
        if iteration < iterations - 8:
            # Pull back towards the C1 two-section family after every length
            # correction.  This prevents the constraint solver itself from
            # reintroducing an alternating zig-zag.
            c1_target = project_two_section_c1(output, confidence_array, measured)
            output += 0.18 * (c1_target - output)
        for first in range(KEYPOINT_COUNT - 1):
            constrain_segment(first, first + 1)
        for first in range(KEYPOINT_COUNT - 2, -1, -1):
            constrain_segment(first, first + 1)
    # Final projection has a much larger visual/physical benefit than forcing
    # every noisy chord to pass exactly through an isolated depth observation.
    return project_two_section_c1(output, confidence_array, measured)


def constrain_result_shape(
    result: SevenMarkerResult,
    transform_base_from_camera: np.ndarray,
) -> SevenMarkerResult:
    """Apply material spacing to the displayed/exported smoothed solution."""

    constrained = project_material_keypoints(
        result.smoothed_camera_m, result.confidence, result.measured_valid
    )
    transform = np.asarray(transform_base_from_camera, dtype=float).reshape(4, 4)
    base = np.full_like(constrained, np.nan)
    finite = np.isfinite(constrained).all(axis=1)
    base[finite] = constrained[finite] @ transform[:3, :3].T + transform[:3, 3]
    return replace(result, smoothed_camera_m=constrained, base_m=base)


def _marker_tangents(points: np.ndarray) -> np.ndarray:
    s = np.asarray(KEYPOINT_ARCLENGTHS_M, dtype=float)
    tangents = np.zeros_like(points)
    tangents[0] = (points[1] - points[0]) / (s[1] - s[0])
    tangents[-1] = (points[-1] - points[-2]) / (s[-1] - s[-2])
    for index in range(1, KEYPOINT_COUNT - 1):
        before = (points[index] - points[index - 1]) / (s[index] - s[index - 1])
        after = (points[index + 1] - points[index]) / (s[index + 1] - s[index])
        # Harmonic-like direction averaging suppresses a single noisy chord.
        direction = before + after
        norm = float(np.linalg.norm(direction))
        if norm < 1e-10:
            direction = after
            norm = float(np.linalg.norm(direction))
        speed = np.clip(0.5 * (np.linalg.norm(before) + np.linalg.norm(after)), 0.25, 1.25)
        tangents[index] = direction / max(norm, 1e-10) * speed
    for index in (0, KEYPOINT_COUNT - 1):
        norm = float(np.linalg.norm(tangents[index]))
        if norm > 1e-10:
            tangents[index] *= np.clip(norm, 0.25, 1.25) / norm
    # The two active sections share one physical interface; averaging both
    # one-sided directions gives a continuous tangent at K3.
    interface = 3
    direction = (
        points[interface] - points[interface - 1]
        + points[interface + 1] - points[interface]
    )
    norm = float(np.linalg.norm(direction))
    if norm > 1e-10:
        tangents[interface] = direction / norm * np.clip(
            np.linalg.norm(tangents[interface]), 0.25, 1.25
        )
    return tangents


def _hermite_dense(points: np.ndarray, tangents: np.ndarray, per_interval: int = 16) -> np.ndarray:
    """Build exact 7 mm material arcs guided by the Hermite tangent field."""

    dense: list[np.ndarray] = []
    s = np.asarray(KEYPOINT_ARCLENGTHS_M, dtype=float)
    for index in range(KEYPOINT_COUNT - 1):
        ds = s[index + 1] - s[index]
        delta = points[index + 1] - points[index]
        chord = float(np.linalg.norm(delta))
        values = np.linspace(0.0, 1.0, per_interval, endpoint=index == KEYPOINT_COUNT - 2)
        if chord < 1e-9:
            segment = np.repeat(points[index][None, :], len(values), axis=0)
            dense.append(segment)
            continue
        chord_direction = delta / chord
        ratio = float(np.clip(chord / ds, 1e-4, 1.0))
        if ratio >= 0.9998:
            segment = points[index] + values[:, None] * delta
            dense.append(segment)
            continue
        # Solve 2 sin(theta/2) / theta = chord / arc_length.
        lower, upper = 1e-8, 2.0 * np.pi - 1e-5
        for _ in range(48):
            theta = 0.5 * (lower + upper)
            estimate = 2.0 * np.sin(theta / 2.0) / theta
            if estimate > ratio:
                lower = theta
            else:
                upper = theta
        theta = 0.5 * (lower + upper)
        radius = ds / theta
        # Endpoint tangent change determines the bending side.  Remove its
        # chord component so the constructed arc remains in a valid plane.
        bend_direction = tangents[index] - tangents[index + 1]
        bend_direction -= chord_direction * np.dot(bend_direction, chord_direction)
        bend_norm = float(np.linalg.norm(bend_direction))
        if bend_norm < 1e-8:
            # Use the nearest material point as a geometric side hint.
            hints: list[np.ndarray] = []
            if index > 0:
                hints.append(points[index - 1] - points[index])
            if index + 2 < KEYPOINT_COUNT:
                hints.append(points[index + 2] - points[index + 1])
            bend_direction = np.sum(hints, axis=0) if hints else np.asarray([0.0, 1.0, 0.0])
            bend_direction -= chord_direction * np.dot(bend_direction, chord_direction)
            bend_norm = float(np.linalg.norm(bend_direction))
        if bend_norm < 1e-8:
            reference = np.asarray([0.0, 0.0, 1.0])
            if abs(float(np.dot(reference, chord_direction))) > 0.9:
                reference = np.asarray([0.0, 1.0, 0.0])
            bend_direction = np.cross(chord_direction, reference)
            bend_norm = float(np.linalg.norm(bend_direction))
        bend_direction /= max(bend_norm, 1e-10)
        phi = (values - 0.5) * theta
        midpoint = 0.5 * (points[index] + points[index + 1])
        segment = (
            midpoint
            + (radius * np.sin(phi))[:, None] * chord_direction
            + (radius * (np.cos(phi) - np.cos(theta / 2.0)))[:, None] * bend_direction
        )
        dense.append(segment)
    return np.vstack(dense)


def _resample_by_arclength(path: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray]:
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    keep = np.r_[True, np.diff(cumulative) > 1e-10]
    path = path[keep]
    cumulative = cumulative[keep]
    if len(path) < 2 or cumulative[-1] <= 1e-10:
        return np.full((samples, 3), np.nan), np.full(samples, np.nan)
    target = np.linspace(0.0, cumulative[-1], max(13, int(samples)))
    output = np.column_stack([
        np.interp(target, cumulative, path[:, coordinate]) for coordinate in range(3)
    ])
    return output, target


def fit_continuum_shape(points_m: np.ndarray, samples: int = 85) -> ContinuumShape3D:
    """Fit a local, two-section compatible C1 curve through K0..K6.

    Tangent-guided local material arcs are deliberately used instead of one
    global high-order polynomial: every interval has exactly 7 mm arc length,
    and a bad marker can only perturb neighbouring intervals.
    """

    markers = project_material_keypoints(points_m)
    if np.count_nonzero(np.isfinite(markers).all(axis=1)) < 2:
        empty = np.full((max(13, int(samples)), 3), np.nan)
        scalar = np.full(len(empty), np.nan)
        return ContinuumShape3D(markers, empty, scalar, empty, scalar, float("nan"), float("nan"))
    tangents_at_markers = _marker_tangents(markers)
    raw_dense = _hermite_dense(markers, tangents_at_markers)
    centerline, actual_s = _resample_by_arclength(raw_dense, samples)
    # The polyline approximation of a circular arc is infinitesimally shorter
    # than its analytic material length.  Parameterise the returned curve by
    # the known 42 mm material coordinate rather than accumulating that
    # discretisation error.
    material_s = actual_s / max(float(actual_s[-1]), 1e-10) * TOTAL_ACTIVE_LENGTH_M
    tangent = np.gradient(centerline, material_s, axis=0, edge_order=2)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-10)
    # Use the analytic curvature of each material arc.  A raw numerical second
    # derivative would report artificial spikes at finite sample boundaries.
    interval_curvature = np.zeros(KEYPOINT_COUNT - 1)
    for index, chord in enumerate(np.linalg.norm(np.diff(markers, axis=0), axis=1)):
        ratio = float(np.clip(chord / 0.007, 1e-4, 1.0))
        if ratio >= 0.9998:
            continue
        lower, upper = 1e-8, 2.0 * np.pi - 1e-5
        for _ in range(40):
            theta = 0.5 * (lower + upper)
            if 2.0 * np.sin(theta / 2.0) / theta > ratio:
                lower = theta
            else:
                upper = theta
        interval_curvature[index] = 0.5 * (lower + upper) / 0.007
    interval_index = np.clip(
        np.floor(material_s / 0.007).astype(int), 0, KEYPOINT_COUNT - 2
    )
    curvature = interval_curvature[interval_index]
    interior = interval_curvature
    maximum = float(np.nanpercentile(interior, 95)) if len(interior) else float("nan")
    return ContinuumShape3D(
        markers, centerline, material_s, tangent, curvature,
        TOTAL_ACTIVE_LENGTH_M, maximum,
    )


def tube_mesh(
    centerline_m: np.ndarray,
    radius_m: float = DEFAULT_DIAMETER_M / 2.0,
    radial_samples: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a parallel-transport tube mesh around a reconstructed centreline."""

    centerline = np.asarray(centerline_m, dtype=float).reshape(-1, 3)
    if len(centerline) < 2 or not np.isfinite(centerline).all():
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32), np.empty((0, 4))
    tangent = np.gradient(centerline, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-10)
    normals = np.zeros_like(tangent)
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, tangent[0]))) > 0.88:
        reference = np.asarray([0.0, 1.0, 0.0])
    normals[0] = np.cross(tangent[0], reference)
    normals[0] /= max(float(np.linalg.norm(normals[0])), 1e-10)
    for index in range(1, len(centerline)):
        transported = normals[index - 1] - tangent[index] * np.dot(
            normals[index - 1], tangent[index]
        )
        norm = float(np.linalg.norm(transported))
        if norm < 1e-8:
            transported = np.cross(tangent[index], reference)
            norm = float(np.linalg.norm(transported))
        normals[index] = transported / max(norm, 1e-10)
    binormals = np.cross(tangent, normals)
    angles = np.linspace(0.0, 2.0 * np.pi, max(6, int(radial_samples)), endpoint=False)
    rings = (
        centerline[:, None, :]
        + float(radius_m)
        * (
            np.cos(angles)[None, :, None] * normals[:, None, :]
            + np.sin(angles)[None, :, None] * binormals[:, None, :]
        )
    )
    vertices = rings.reshape(-1, 3)
    radial = len(angles)
    faces: list[tuple[int, int, int]] = []
    for row in range(len(centerline) - 1):
        for column in range(radial):
            current = row * radial + column
            following = row * radial + (column + 1) % radial
            next_current = (row + 1) * radial + column
            next_following = (row + 1) * radial + (column + 1) % radial
            faces.extend(((current, next_current, following), (following, next_current, next_following)))
    progress = np.repeat(np.linspace(0.0, 1.0, len(centerline)), radial)
    colors = np.column_stack((
        0.08 + 0.12 * progress,
        0.48 + 0.34 * progress,
        0.98 - 0.18 * progress,
        np.full_like(progress, 0.92),
    ))
    return vertices, np.asarray(faces, dtype=np.int32), colors
