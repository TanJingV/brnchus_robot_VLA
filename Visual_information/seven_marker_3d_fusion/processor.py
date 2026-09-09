"""Headless session processing shared by the UI and tests."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from Visual_information.d435_tdcr_capture.camera import RealSenseSource

from .curve_model import constrain_result_shape
from .cotracker3_tracker import CoTracker3SequenceTracker
from .em_constraint import EmChordConstraintFusion
from .io import ResultWriter, refine_existing_npz
from .legacy_refinement import LocalDepthCurveFusion
from .motor_prior import MotorShapePriorFusion
from .offline_refinement import refine_output_trajectory
from .pipeline import SevenMarkerFusionPipeline
from .session import LegacyFramePacket, LegacySessionReader, SessionKind, describe_session


ProgressCallback = Callable[[int, int, object | None, dict], None]
RecordingCallback = Callable[[], bool]


def process_session(
    session_path: str | Path,
    output_dir: str | Path,
    config: dict,
    parameters: dict | None = None,
    progress: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    save_overlay: bool = True,
    recording_enabled: RecordingCallback | None = None,
) -> Path:
    descriptor = describe_session(session_path)
    parameters = parameters or {}
    stop_event = stop_event or threading.Event()
    output = Path(output_dir).expanduser().resolve()
    if descriptor.kind == SessionKind.KEYPOINTS:
        target = refine_existing_npz(descriptor.path / "keypoints.npz", output, config)
        if bool(parameters.get("offline_refine_enabled", True)):
            refine_output_trajectory(
                target,
                temporal_strength=float(parameters.get("offline_temporal_strength", 5.0)),
                curve_samples=int(parameters.get("curve_samples", 85)),
            )
        return target
    if descriptor.kind == SessionKind.LEGACY_VIDEO:
        return _process_legacy(
            descriptor.path, output, config, parameters, progress, stop_event,
            start_frame, end_frame, save_overlay, recording_enabled,
        )
    if descriptor.kind == SessionKind.BAG:
        return _process_bag(
            descriptor.path / "realsense.bag", output, config, parameters, progress,
            stop_event, end_frame, save_overlay,
        )
    raise ValueError(descriptor.quality_detail)


def _process_legacy(
    session: Path,
    output: Path,
    config: dict,
    parameters: dict,
    progress: ProgressCallback | None,
    stop_event: threading.Event,
    start_frame: int,
    end_frame: int | None,
    save_overlay: bool,
    recording_enabled: RecordingCallback | None,
) -> Path:
    reader = LegacySessionReader(session)
    total = reader.descriptor.frame_count
    first = max(0, int(start_frame))
    final = min(total, int(end_frame) if end_frame is not None else total)
    pipeline = SevenMarkerFusionPipeline(config)
    alignment_profile = (
        parameters.get("motor_alignment_profile")
        if bool(parameters.get("use_motor_alignment_profile", True))
        else None
    )
    motor_prior = MotorShapePriorFusion(
        enabled=bool(parameters.get("motor_prior_enabled", True)),
        alignment_profile=alignment_profile,
    )
    depth_curve = LocalDepthCurveFusion(
        enabled=bool(parameters.get("local_depth_curve_enabled", True)),
        search_radius_px=int(parameters.get("local_depth_search_radius_px", 22)),
        depth_gate_m=float(parameters.get("local_depth_gate_m", 0.018)),
    )
    em_constraint = EmChordConstraintFusion(
        enabled=bool(parameters.get("em_chord_constraint_enabled", True)),
        first_port=int(parameters.get("em_first_port", 10)),
        second_port=int(parameters.get("em_second_port", 11)),
        gain=float(parameters.get("em_chord_gain", 0.92)),
        endpoint_offset_m=float(parameters.get("em_chord_offset_m", 0.002)),
    )
    roi = parameters.get("roi", (0.0, 0.0, 1.0, 1.0))
    pipeline.set_roi(tuple(map(float, roi)))
    target_lock_roi = parameters.get("target_lock_roi")
    initial_keypoints = parameters.get("initial_keypoints_px")
    use_cotracker = bool(config.get("tracking", {}).get("cotracker3_enabled", True))
    if not pipeline.yolo_pose.ready and pipeline.yolo_pose.strict and not use_cotracker:
        raise ValueError(
            "Strict learned-keypoint mode is enabled, but Visual_information/models/tdcr_yolo_pose/best.pt "
            "does not exist. Label and train the custom seven-keypoint model first."
        )
    if (
        target_lock_roi is None
        and initial_keypoints is None
        and not pipeline.yolo_pose.ready
        and not use_cotracker
    ):
        raise ValueError(
            "The seven-ring terminal target is not locked. Drag a tight RGB box "
            "around the terminal section and click '锁定末端并开始跟踪' before processing."
        )
    if target_lock_roi is not None:
        pipeline.set_target_lock(tuple(map(float, target_lock_roi)))
    cotracker_result = None
    if initial_keypoints is not None:
        initialization_frame = int(parameters.get("keypoint_init_frame", first))
        if initialization_frame != first:
            raise ValueError(
                f"The processing start frame ({first}) must equal the K0-K6 "
                f"conditioning frame ({initialization_frame})."
            )
        conditioning = reader.read(first, parameters)
        raw_initial_keypoints = np.asarray(initial_keypoints, dtype=float)
        initial_keypoints = pipeline.initialise_keypoints(
            conditioning.rgb_bgr, raw_initial_keypoints
        )
        parameters["manual_initial_keypoints_px"] = raw_initial_keypoints.tolist()
        parameters["initial_keypoints_px"] = np.asarray(initial_keypoints).tolist()
        parameters["initialization_mode"] = "manual-order+body-lattice-primary"
    elif use_cotracker:
        raise ValueError(
            "CoTracker3 requires one conditioning frame: click K0 through K6 once in the RGB view."
        )
    if use_cotracker:
        tracker3 = CoTracker3SequenceTracker(config.get("tracking", {}))
        cotracker_result = tracker3.track_video(
            session / "rgb.mp4",
            first,
            final,
            np.asarray(initial_keypoints, dtype=float),
            progress=(
                (lambda done, total_frames: progress(
                    done, total_frames, None,
                    {"phase": "cotracker3", "frame_index": first + max(0, done - 1)},
                ))
                if progress is not None else None
            ),
            stop_requested=stop_event.is_set,
        )
    # ``None`` preserves the headless/CLI behaviour of recording the complete
    # processing interval.  The workstation supplies an independent gate so
    # tracking can continue while disk recording is started or stopped.
    automatic_recording = recording_enabled is None
    writer: ResultWriter | None = None
    recording_segment = 0
    recorded_in_segment = 0
    recorded_total = 0
    recorded_targets: list[Path] = []

    def recording_requested() -> bool:
        return automatic_recording or bool(recording_enabled and recording_enabled())

    def open_writer() -> ResultWriter:
        nonlocal recording_segment, recorded_in_segment
        recording_segment += 1
        recorded_in_segment = 0
        target = output if recording_segment == 1 else output / f"recording_{recording_segment:03d}"
        return ResultWriter(target, save_overlay=save_overlay, fps=reader.descriptor.fps)

    def close_recording(active: ResultWriter, stopped_at_frame: int) -> Path:
        target = active.close({
            "source_session": str(session),
            "session_kind": "legacy_video",
            "recording_segment": int(recording_segment),
            "recorded_frames": int(recorded_in_segment),
            "stopped_at_frame": int(stopped_at_frame),
            "tracking_continued_after_recording": not bool(stop_event.is_set()),
            "parameters": parameters,
        })
        recorded_targets.append(target)
        return target
    preview_stride = max(1, int(parameters.get("preview_stride", 3)))
    processed = 0
    alignment_saved = ""
    alignment_save_error = ""
    try:
        for index in range(first, final):
            if stop_event.is_set():
                break
            packet = reader.read(index, parameters)
            if cotracker_result is not None:
                trajectory_index = index - first
                pipeline.set_external_keypoints(
                    cotracker_result.pixels[trajectory_index],
                    cotracker_result.confidence[trajectory_index],
                    cotracker_result.visible[trajectory_index],
                    cotracker_result.backend,
                )
            result = pipeline.process(packet.camera_frame)
            result = motor_prior.fuse(
                result, packet.axis_row, pipeline.tracker.transform_base_from_camera
            )
            result = depth_curve.fuse(
                result, packet.camera_frame, pipeline.tracker.transform_base_from_camera
            )
            result = constrain_result_shape(
                result, pipeline.tracker.transform_base_from_camera
            )
            result = em_constraint.fuse(
                result, packet.em_row, pipeline.tracker.transform_base_from_camera
            )
            overlay = pipeline.draw_overlay(packet.camera_frame, result)
            wants_recording = recording_requested()
            if wants_recording and writer is None:
                writer = open_writer()
            elif not wants_recording and writer is not None:
                close_recording(writer, index)
                writer = None
            if writer is not None:
                writer.append(result, overlay)
                recorded_in_segment += 1
                recorded_total += 1
            processed += 1
            if progress is not None and (processed == 1 or processed % preview_stride == 0):
                progress(processed, final - first, result, {
                    "overlay": overlay,
                    "depth": packet.depth_visual_bgr,
                    "pointcloud": packet.pointcloud_bgr,
                    "endoscope": packet.endoscope_bgr,
                    "axis": packet.axis_row,
                    "em": packet.em_row,
                    "decode_confidence": packet.depth_decode_confidence,
                    "frame_index": index,
                })
    finally:
        reader.close()
        if bool(parameters.get("save_motor_alignment_profile", False)):
            try:
                if motor_prior.alignment_updates < 5:
                    raise ValueError("At least five frames with >=3 direct 3-D markers are required")
                alignment_saved = str(motor_prior.save_alignment_profile(
                    parameters.get("motor_alignment_profile", output / "motor_camera_alignment.json"),
                    source_session=str(session),
                ))
            except Exception as exc:
                alignment_save_error = str(exc)
        metadata = {
            "source_session": str(session),
            "session_kind": "legacy_video",
            "metric_quality": "approximate",
            "warning": "Depth was reconstructed from a lossy TURBO MP4 and registered to RGB with an editable approximate transform.",
            "parameters": parameters,
            "target_lock_roi": list(map(float, target_lock_roi)) if target_lock_roi is not None else None,
            "keypoint_initialization_frame": int(parameters.get("keypoint_init_frame", first)),
            "keypoint_initialization_px": (
                np.asarray(initial_keypoints, dtype=float).tolist()
                if initial_keypoints is not None else None
            ),
            "rgb_point_tracker": (
                cotracker_result.backend
                if cotracker_result is not None
                else "pyramidal-lk+seven-lab-identities+material-topology"
            ),
            "motor_pcc_prior_enabled": bool(parameters.get("motor_prior_enabled", True)),
            "motor_alignment_profile_loaded": str(motor_prior.profile_path or ""),
            "motor_alignment_updates": int(motor_prior.alignment_updates),
            "motor_alignment_residual_mm": (
                float(motor_prior.alignment_residual_m * 1000.0)
                if np.isfinite(motor_prior.alignment_residual_m) else None
            ),
            "motor_alignment_profile_saved": alignment_saved,
            "motor_alignment_save_error": alignment_save_error,
            "rgb_local_depth_curve_enabled": bool(parameters.get("local_depth_curve_enabled", True)),
            "em_chord_constraint_enabled": bool(parameters.get("em_chord_constraint_enabled", True)),
            "em_ports": [
                int(parameters.get("em_first_port", 10)),
                int(parameters.get("em_second_port", 11)),
            ],
            "em_chord_offset_m": float(parameters.get("em_chord_offset_m", 0.002)),
            "processed_frames": processed,
            "recorded_frames": recorded_total,
            "recording_segments": [str(path) for path in recorded_targets],
            "stopped_by_user": bool(stop_event.is_set()),
        }
        if writer is not None:
            final_metadata = dict(metadata)
            final_metadata.update({
                "recording_segment": int(recording_segment),
                "recorded_frames_in_segment": int(recorded_in_segment),
                "stopped_at_frame": int(first + processed),
            })
            recorded_targets.append(writer.close(final_metadata))
            writer = None
            metadata["recording_segments"] = [str(path) for path in recorded_targets]
        output.mkdir(parents=True, exist_ok=True)
        import json
        (output / "tracking_summary.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if bool(parameters.get("offline_refine_enabled", True)):
        for target in recorded_targets:
            refine_output_trajectory(
                target,
                temporal_strength=float(parameters.get("offline_temporal_strength", 5.0)),
                curve_samples=int(parameters.get("curve_samples", 85)),
            )
    return output


def _process_bag(
    bag_path: Path,
    output: Path,
    config: dict,
    parameters: dict,
    progress: ProgressCallback | None,
    stop_event: threading.Event,
    end_frame: int | None,
    save_overlay: bool,
) -> Path:
    source = RealSenseSource(config["camera"], bag_path=bag_path, repeat=False)
    pipeline = SevenMarkerFusionPipeline(config)
    roi = parameters.get("roi", (0.0, 0.0, 1.0, 1.0))
    pipeline.set_roi(tuple(map(float, roi)))
    target_lock_roi = parameters.get("target_lock_roi")
    use_cotracker = bool(config.get("tracking", {}).get("cotracker3_enabled", True))
    initial_keypoints = parameters.get("initial_keypoints_px")
    if not pipeline.yolo_pose.ready and pipeline.yolo_pose.strict and not use_cotracker:
        raise ValueError(
            "Strict learned-keypoint mode is enabled, but the trained TDCR YOLO Pose model is missing."
        )
    if target_lock_roi is None and not pipeline.yolo_pose.ready and not use_cotracker:
        raise ValueError(
            "The seven-ring terminal target is not locked. Set target_lock_roi before BAG processing."
        )
    if target_lock_roi is not None:
        pipeline.set_target_lock(tuple(map(float, target_lock_roi)))
    if use_cotracker:
        raise ValueError(
            "Direct BAG CoTracker3 preprocessing is not available yet. "
            "Use the capture session containing rgb.mp4, or disable CoTracker3 for BAG replay."
        )
    depth_curve = LocalDepthCurveFusion(
        enabled=bool(parameters.get("local_depth_curve_enabled", True)),
        search_radius_px=int(parameters.get("local_depth_search_radius_px", 22)),
        depth_gate_m=float(parameters.get("local_depth_gate_m", 0.018)),
    )
    writer = ResultWriter(output, save_overlay=save_overlay, fps=float(config["camera"]["color"][2]))
    source.start()
    count = 0
    last_frame_at = time.monotonic()
    try:
        while not stop_event.is_set():
            frame = source.poll()
            if frame is None:
                if count and time.monotonic() - last_frame_at > 1.5:
                    break
                time.sleep(0.001)
                continue
            last_frame_at = time.monotonic()
            result = pipeline.process(frame)
            result = depth_curve.fuse(
                result, frame, pipeline.tracker.transform_base_from_camera
            )
            result = constrain_result_shape(
                result, pipeline.tracker.transform_base_from_camera
            )
            overlay = pipeline.draw_overlay(frame, result)
            writer.append(result, overlay)
            count += 1
            if progress is not None and (count == 1 or count % 3 == 0):
                depth = frame.native_depth_color_bgr
                progress(count, int(end_frame or 0), result, {
                    "overlay": overlay,
                    "depth": depth,
                    "pointcloud": None,
                    "endoscope": None,
                    "axis": {},
                    "em": {},
                    "decode_confidence": 1.0,
                    "frame_index": count - 1,
                })
            if end_frame is not None and count >= int(end_frame):
                break
    finally:
        source.stop()
        target = writer.close({
            "source_bag": str(bag_path),
            "session_kind": "bag",
            "metric_quality": "factory_calibrated",
            "target_lock_roi": list(map(float, target_lock_roi)) if target_lock_roi is not None else None,
            "processed_frames": count,
        })
    if bool(parameters.get("offline_refine_enabled", True)) and count:
        refine_output_trajectory(
            target,
            temporal_strength=float(parameters.get("offline_temporal_strength", 5.0)),
            curve_samples=int(parameters.get("curve_samples", 85)),
        )
    return target
