"""Session readers and deterministic CSV/NPZ result writers."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from Visual_information.d435_tdcr_capture.config import VISUAL_ROOT
from Visual_information.d435_tdcr_capture.models import KEYPOINT_COUNT

from .models import SevenMarkerResult
from .smoother import ShapeConstrainedSmoother


class ResultWriter:
    def __init__(self, output_dir: str | Path, save_overlay: bool = True, fps: float = 30.0) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = (self.output_dir / "seven_marker_3d.csv").open(
            "w", newline="", encoding="utf-8-sig"
        )
        fields = ["sequence", "capture_host_ns", "processing_ms", "measured_count"]
        for index in range(KEYPOINT_COUNT):
            fields.extend([
                f"kp{index}_valid", f"kp{index}_predicted", f"kp{index}_confidence",
                f"kp{index}_source", f"kp{index}_u_px", f"kp{index}_v_px",
                f"kp{index}_camera_x_m", f"kp{index}_camera_y_m", f"kp{index}_camera_z_m",
                f"kp{index}_base_x_m", f"kp{index}_base_y_m", f"kp{index}_base_z_m",
                f"kp{index}_sigma_x_mm", f"kp{index}_sigma_y_mm", f"kp{index}_sigma_z_mm",
            ])
        self.csv = csv.DictWriter(self.csv_file, fieldnames=fields)
        self.csv.writeheader()
        self.save_overlay = bool(save_overlay)
        self.fps = float(fps)
        self.video: cv2.VideoWriter | None = None
        self.buffers: dict[str, list] = {name: [] for name in (
            "sequence", "capture_host_ns", "raw_camera_m", "fused_camera_m",
            "smoothed_camera_m", "base_m", "covariance_m2", "measured_valid",
            "predicted", "confidence", "color_pixels", "left_pixels", "right_pixels",
            "sources", "rgb_depth_camera_m", "ir_stereo_camera_m", "processing_ms",
        )}

    def append(self, result: SevenMarkerResult, overlay: np.ndarray | None = None) -> None:
        row = {
            "sequence": result.sequence,
            "capture_host_ns": result.capture_host_ns,
            "processing_ms": result.processing_ms,
            "measured_count": int(np.count_nonzero(result.measured_valid)),
        }
        for index in range(KEYPOINT_COUNT):
            covariance = result.covariance_m2[index]
            sigma = (
                np.sqrt(np.maximum(np.diag(covariance), 0.0)) * 1000.0
                if np.isfinite(covariance).all() else np.full(3, np.nan)
            )
            row.update({
                f"kp{index}_valid": int(result.measured_valid[index]),
                f"kp{index}_predicted": int(result.predicted[index]),
                f"kp{index}_confidence": result.confidence[index],
                f"kp{index}_source": result.sources[index],
                f"kp{index}_u_px": result.color_pixels[index, 0],
                f"kp{index}_v_px": result.color_pixels[index, 1],
                f"kp{index}_camera_x_m": result.smoothed_camera_m[index, 0],
                f"kp{index}_camera_y_m": result.smoothed_camera_m[index, 1],
                f"kp{index}_camera_z_m": result.smoothed_camera_m[index, 2],
                f"kp{index}_base_x_m": result.base_m[index, 0],
                f"kp{index}_base_y_m": result.base_m[index, 1],
                f"kp{index}_base_z_m": result.base_m[index, 2],
                f"kp{index}_sigma_x_mm": sigma[0],
                f"kp{index}_sigma_y_mm": sigma[1],
                f"kp{index}_sigma_z_mm": sigma[2],
            })
        self.csv.writerow(row)
        for name in self.buffers:
            value = result.sources if name == "sources" else getattr(result, name)
            self.buffers[name].append(np.asarray(value).copy())
        if self.save_overlay and overlay is not None:
            if self.video is None:
                height, width = overlay.shape[:2]
                path = self.output_dir / "seven_marker_overlay.mp4"
                self.video = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
                )
                if not self.video.isOpened():
                    self.video.release()
                    self.video = None
                    raise RuntimeError(f"Unable to create overlay video: {path}")
            self.video.write(np.asarray(overlay, dtype=np.uint8))
        if len(self.buffers["sequence"]) % 30 == 0:
            self.csv_file.flush()

    def close(self, metadata: dict | None = None) -> Path:
        self.csv_file.flush()
        self.csv_file.close()
        if self.video is not None:
            self.video.release()
            self.video = None
        arrays = {name: np.asarray(values) for name, values in self.buffers.items()}
        np.savez_compressed(self.output_dir / "seven_marker_3d.npz", **arrays)
        valid = arrays["measured_valid"] if len(arrays["measured_valid"]) else np.zeros((0, 7), bool)
        summary = {
            "created_local": datetime.now().isoformat(),
            "frames": int(len(arrays["sequence"])),
            "mean_measured_points": float(np.mean(np.sum(valid, axis=1))) if len(valid) else 0.0,
            "all_seven_fraction": float(np.mean(np.all(valid, axis=1))) if len(valid) else 0.0,
            "median_processing_ms": float(np.median(arrays["processing_ms"])) if len(valid) else None,
            "coordinate_frames": {
                "camera": "D435 left infrared optical frame: +X right, +Y down, +Z forward",
                "base": "T_base_from_camera from project calibration",
            },
            "metadata": metadata or {},
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return self.output_dir


def default_output_dir(root: str | Path = VISUAL_ROOT / "outputs") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(root).resolve() / stamp


def refine_existing_npz(
    archive_path: str | Path,
    output_dir: str | Path,
    config: dict,
) -> Path:
    """Apply geometry/temporal factors to an already recorded keypoints.npz."""

    source_path = Path(archive_path).expanduser().resolve()
    with np.load(source_path, allow_pickle=False) as archive:
        names = set(archive.files)
        point_name = next(
            (name for name in ("keypoints_raw_camera_m", "fused_camera_m", "smoothed_camera_m") if name in names),
            None,
        )
        if point_name is None:
            raise ValueError("NPZ lacks keypoints_raw_camera_m/fused_camera_m/smoothed_camera_m")
        points = np.asarray(archive[point_name], dtype=float)
        timestamps = np.asarray(
            archive["capture_host_ns"] if "capture_host_ns" in names else np.arange(len(points)) * 33_333_333,
            dtype=np.int64,
        )
        valid = np.asarray(
            archive["keypoint_valid"] if "keypoint_valid" in names else np.isfinite(points).all(axis=2),
            dtype=bool,
        )
        confidence = np.asarray(
            archive["keypoint_confidence"] if "keypoint_confidence" in names else valid.astype(float),
            dtype=float,
        )
        covariance = np.asarray(
            archive["keypoint_covariance_m2"]
            if "keypoint_covariance_m2" in names
            else np.tile(np.eye(3) * 1e-6, (len(points), KEYPOINT_COUNT, 1, 1)),
            dtype=float,
        )
    if points.shape[1:] != (KEYPOINT_COUNT, 3):
        raise ValueError(f"Expected (N, 7, 3) keypoints, got {points.shape}")
    smoother = ShapeConstrainedSmoother(config)
    smoothed, predicted, output_covariance, output_confidence = [], [], [], []
    for frame_index in range(len(points)):
        result = smoother.update(
            points[frame_index], covariance[frame_index], valid[frame_index],
            confidence[frame_index], int(timestamps[frame_index]),
        )
        smooth_frame, covariance_frame, predicted_frame, confidence_frame = result
        smoothed.append(smooth_frame)
        output_covariance.append(covariance_frame)
        predicted.append(predicted_frame)
        output_confidence.append(confidence_frame)
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / "seven_marker_3d.npz",
        capture_host_ns=timestamps,
        input_camera_m=points,
        smoothed_camera_m=np.asarray(smoothed),
        covariance_m2=np.asarray(output_covariance),
        measured_valid=valid,
        predicted=np.asarray(predicted),
        confidence=np.asarray(output_confidence),
    )
    fields = ["frame", "capture_host_ns"]
    for index in range(KEYPOINT_COUNT):
        fields.extend([
            f"kp{index}_valid", f"kp{index}_predicted", f"kp{index}_confidence",
            f"kp{index}_x_m", f"kp{index}_y_m", f"kp{index}_z_m",
        ])
    with (target / "seven_marker_3d.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame_index, frame_points in enumerate(smoothed):
            row = {"frame": frame_index, "capture_host_ns": int(timestamps[frame_index])}
            for index in range(KEYPOINT_COUNT):
                row.update({
                    f"kp{index}_valid": int(valid[frame_index, index]),
                    f"kp{index}_predicted": int(predicted[frame_index][index]),
                    f"kp{index}_confidence": output_confidence[frame_index][index],
                    f"kp{index}_x_m": frame_points[index, 0],
                    f"kp{index}_y_m": frame_points[index, 1],
                    f"kp{index}_z_m": frame_points[index, 2],
                })
            writer.writerow(row)
    summary = {
        "source": str(source_path),
        "frames": len(points),
        "mode": "saved_measurements_shape_refinement",
        "note": "No RGB/depth/IR images were present in NPZ; visual measurements were not recomputed.",
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
