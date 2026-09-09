"""Non-causal trajectory and continuous-shape refinement for recorded data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from Visual_information.d435_tdcr_capture.models import KEYPOINT_ARCLENGTHS_M, KEYPOINT_COUNT

from .curve_model import fit_continuum_shape, project_material_keypoints


def _zero_phase_smooth(values: np.ndarray, weights: np.ndarray, strength: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    count = len(values)
    if count < 4:
        return values.copy()
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(finite) < 2:
        return values.copy()
    target = np.where(finite, values, 0.0)
    effective_weight = np.where(finite, weights, 0.0)
    second = sparse.diags(
        (np.ones(count - 2), -2.0 * np.ones(count - 2), np.ones(count - 2)),
        (0, 1, 2),
        shape=(count - 2, count),
        format="csc",
    )
    matrix = (
        sparse.diags(effective_weight, format="csc")
        + max(0.0, float(strength)) * (second.T @ second)
        + sparse.eye(count, format="csc") * 1e-9
    )
    return np.asarray(spsolve(matrix, effective_weight * target), dtype=float)


def refine_output_trajectory(
    output_dir: str | Path,
    temporal_strength: float = 5.0,
    curve_samples: int = 85,
) -> dict:
    """Add future-aware keypoints and dense 3-D centreline arrays to an output.

    This pass is deliberately non-causal.  It uses samples before and after the
    current frame, so it reduces jitter without introducing the phase delay of a
    one-direction low-pass filter.  Direct measurements retain much larger data
    weights than motor/depth-curve predictions.
    """

    target = Path(output_dir).expanduser().resolve()
    archive_path = target / "seven_marker_3d.npz"
    if not archive_path.exists():
        return {"offline_refined": False, "reason": "seven_marker_3d.npz missing"}
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    points = np.asarray(arrays.get("smoothed_camera_m"), dtype=float)
    if points.ndim != 3 or points.shape[1:] != (KEYPOINT_COUNT, 3) or not len(points):
        return {"offline_refined": False, "reason": "no keypoint trajectory"}
    measured = np.asarray(arrays.get("measured_valid", np.zeros(points.shape[:2])), dtype=bool)
    predicted = np.asarray(arrays.get("predicted", ~measured), dtype=bool)
    confidence = np.asarray(arrays.get("confidence", np.ones(points.shape[:2]) * 0.2), dtype=float)
    data_weight = np.clip(confidence, 0.02, 1.0) * np.where(measured, 7.0, 0.85)
    data_weight *= np.where(predicted & ~measured, 0.65, 1.0)
    refined = points.copy()
    for marker in range(KEYPOINT_COUNT):
        for coordinate in range(3):
            refined[:, marker, coordinate] = _zero_phase_smooth(
                points[:, marker, coordinate], data_weight[:, marker], temporal_strength
            )

    samples = max(25, int(curve_samples))
    marker_locations = np.rint(
        np.asarray(KEYPOINT_ARCLENGTHS_M) / float(KEYPOINT_ARCLENGTHS_M[-1]) * (samples - 1)
    ).astype(int)
    centerlines = np.full((len(points), samples, 3), np.nan)
    tangents = np.full_like(centerlines, np.nan)
    curvature = np.full((len(points), samples), np.nan)
    arclength = np.full((len(points), samples), np.nan)
    total_length = np.full(len(points), np.nan)
    maximum_curvature = np.full(len(points), np.nan)
    for frame in range(len(points)):
        refined[frame] = project_material_keypoints(
            refined[frame], confidence[frame], measured[frame]
        )
        first_shape = fit_continuum_shape(refined[frame], samples=samples)
        shape_markers = first_shape.centerline_m[marker_locations]
        # Measurements stay near the sensor solution; predictions are drawn
        # more strongly onto the smooth material curve.
        blend = np.where(measured[frame], 0.12, 0.58)[:, None]
        finite = np.isfinite(shape_markers).all(axis=1) & np.isfinite(refined[frame]).all(axis=1)
        refined[frame, finite] = (
            (1.0 - blend[finite]) * refined[frame, finite]
            + blend[finite] * shape_markers[finite]
        )
        refined[frame] = project_material_keypoints(
            refined[frame], confidence[frame], measured[frame]
        )
        shape = fit_continuum_shape(refined[frame], samples=samples)
        centerlines[frame] = shape.centerline_m
        tangents[frame] = shape.tangent
        curvature[frame] = shape.curvature_1_m
        arclength[frame] = shape.arclength_m
        total_length[frame] = shape.total_length_m
        maximum_curvature[frame] = shape.maximum_curvature_1_m

    arrays.update({
        "offline_refined_camera_m": refined,
        "curve_centerline_camera_m": centerlines,
        "curve_tangent": tangents,
        "curve_curvature_1_m": curvature,
        "curve_arclength_m": arclength,
        "curve_total_length_m": total_length,
        "curve_maximum_curvature_1_m": maximum_curvature,
    })
    np.savez_compressed(archive_path, **arrays)

    fields = ["frame", "sequence", "capture_host_ns", "measured_count", "curve_length_mm", "max_curvature_1_m"]
    for marker in range(KEYPOINT_COUNT):
        fields.extend([
            f"kp{marker}_measured", f"kp{marker}_predicted", f"kp{marker}_confidence",
            f"kp{marker}_x_m", f"kp{marker}_y_m", f"kp{marker}_z_m",
        ])
    sequence = np.asarray(arrays.get("sequence", np.arange(len(points))))
    timestamps = np.asarray(arrays.get("capture_host_ns", np.zeros(len(points), dtype=np.int64)))
    with (target / "seven_marker_refined.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame in range(len(points)):
            row = {
                "frame": frame,
                "sequence": int(sequence[frame]),
                "capture_host_ns": int(timestamps[frame]),
                "measured_count": int(np.count_nonzero(measured[frame])),
                "curve_length_mm": total_length[frame] * 1000.0,
                "max_curvature_1_m": maximum_curvature[frame],
            }
            for marker in range(KEYPOINT_COUNT):
                row.update({
                    f"kp{marker}_measured": int(measured[frame, marker]),
                    f"kp{marker}_predicted": int(predicted[frame, marker]),
                    f"kp{marker}_confidence": confidence[frame, marker],
                    f"kp{marker}_x_m": refined[frame, marker, 0],
                    f"kp{marker}_y_m": refined[frame, marker, 1],
                    f"kp{marker}_z_m": refined[frame, marker, 2],
                })
            writer.writerow(row)

    metrics = {
        "offline_refined": True,
        "temporal_method": "global weighted second-difference regularization (zero phase)",
        "temporal_strength": float(temporal_strength),
        "curve_model": "tangent-guided 7 mm material arcs with local continuity",
        "curve_samples": samples,
        "median_curve_length_mm": float(np.nanmedian(total_length) * 1000.0),
        "p95_maximum_curvature_1_m": float(np.nanpercentile(maximum_curvature, 95)),
    }
    summary_path = target / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["offline_shape_refinement"] = metrics
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics
