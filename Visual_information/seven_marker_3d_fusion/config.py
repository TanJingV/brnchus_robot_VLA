"""Load the project configuration plus fusion-specific defaults."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from Visual_information.d435_tdcr_capture.config import load_config as load_project_config


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_FUSION_CONFIG = PACKAGE_DIR / "fusion_config.json"


def _merge(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_fusion_config(
    project_config_path: str | Path | None = None,
    fusion_config_path: str | Path | None = None,
) -> dict:
    config = load_project_config(project_config_path)
    with DEFAULT_FUSION_CONFIG.open("r", encoding="utf-8") as stream:
        extension = json.load(stream)
    if fusion_config_path is not None:
        with Path(fusion_config_path).expanduser().open("r", encoding="utf-8") as stream:
            extension = _merge(extension, json.load(stream))
    config = _merge(config, extension)
    # Metric fusion requires all three D435 sensing paths.  These changes are
    # local to this program and never overwrite the main UI configuration.
    config["tracking"]["enabled"] = True
    config["tracking"]["marker_tip_orientation"] = "auto_hybrid"
    config["tracking"]["marker_sequence_color_weight"] = 0.65
    config["tracking"]["body_tight_band_support_enabled"] = True
    config["tracking"]["body_tight_band_support_radius_px"] = 4
    # The legacy colour rings are only a few pixels apart at the old 416 px
    # processing width.  The strict detector owns a separate 960 px body-first
    # path; metric depth is still sampled at the original RGB resolution.
    config["tracking"]["body_first_processing_width_px"] = 960
    config["tracking"]["body_first_anchor_x_fraction"] = 0.76
    config["tracking"]["body_first_distal_x_fraction"] = 0.08
    config["tracking"]["body_first_minimum_band_contrast"] = 7.0
    config["tracking"]["body_first_minimum_global_score"] = 1.0
    config["tracking"]["body_first_minimum_temporal_score"] = 0.72
    config["tracking"]["body_first_maximum_marker_prediction_frames"] = 90
    config["seven_marker_fusion"]["chain_depth_maximum_chord_m"] = 0.0082
    config["seven_marker_fusion"]["chain_depth_interpolation_gate_m"] = 0.0060
    # SAM2.1 video memory is the authoritative body identity.  The user must
    # lock the seven-ring terminal section once; the program never silently
    # replaces a lost target with a visually similar lung edge or drive rail.
    config["tracking"]["body_neural_segmentation_enabled"] = True
    config["tracking"]["body_neural_segmentation_required"] = True
    config["tracking"]["body_neural_device"] = "cuda"
    config["tracking"]["body_target_lock_required"] = True
    config["tracking"]["body_target_lock_roi"] = None
    config["tracking"]["body_target_tracking_margin"] = 0.65
    config["tracking"]["body_neural_positive_points"] = 9
    config["tracking"]["body_neural_negative_offset_radii"] = 3.2
    config["tracking"]["body_neural_corridor_radii"] = 2.0
    config["tracking"]["body_neural_minimum_confidence"] = 0.50
    config["tracking"]["body_neural_minimum_prompt_coverage"] = 0.72
    config["tracking"]["body_neural_minimum_temporal_iou"] = 0.18
    config["tracking"]["body_neural_minimum_area_ratio"] = 0.38
    config["tracking"]["body_neural_maximum_area_ratio"] = 2.4
    config["tracking"]["body_neural_maximum_propagation_frames"] = 2
    # SAM2 corrects identity at 10 Hz; pyramidal flow propagates the accepted
    # material mask on the two intervening 30 Hz frames.
    config["tracking"]["body_neural_inference_stride"] = 3
    # The fixed transparent lung and guide produce a large foreground-
    # difference response.  That mask is useful diagnostically but must never
    # replace a lost neural object identity in quantitative tracking.
    config["tracking"]["material_chain_foreground_fallback_enabled"] = False
    # CoTracker queries lock onto the stationary transparent guide/reflections
    # in this experiment while the coloured material lattice slides beneath
    # them.  The arc-unwrapped body+lattice tracker is therefore authoritative;
    # seven manual clicks define identity/order only.  CoTracker remains an
    # explicit opt-in diagnostic backend.
    config["tracking"]["cotracker3_enabled"] = False
    config["tracking"].setdefault(
        "cotracker3_checkpoint", "Visual_information/models/cotracker3/scaled_offline.pth"
    )
    config["tracking"].setdefault("cotracker3_device", "cuda")
    config["tracking"].setdefault("cotracker3_processing_width", 960)
    config["tracking"].setdefault("cotracker3_window_frames", 60)
    config["tracking"].setdefault("cotracker3_overlap_frames", 16)
    # Search around the propagated material state, not around the stale
    # conditioning-frame CoTracker query.  Twenty-two native RGB pixels still
    # remain below one marker pitch in the 1080p experiment.
    config["tracking"]["cotracker3_colour_search_radius_px"] = 22
    config["tracking"].setdefault("cotracker3_colour_maximum_lab_delta", 52.0)
    config["tracking"].setdefault("cotracker3_maximum_gap_scale", 2.8)
    config["tracking"].setdefault("tdcr_yolo_pose_enabled", True)
    config["tracking"].setdefault(
        "tdcr_yolo_pose_model", "Visual_information/models/tdcr_yolo_pose/best.pt"
    )
    config["tracking"].setdefault("tdcr_yolo_pose_strict", False)
    config["tracking"].setdefault("tdcr_yolo_pose_image_size", 960)
    config["tracking"].setdefault("tdcr_yolo_pose_box_confidence", 0.28)
    config["tracking"].setdefault("tdcr_yolo_pose_keypoint_confidence", 0.30)
    config["tracking"].setdefault("tdcr_yolo_pose_minimum_keypoints", 7)
    config["tracking"].setdefault("tdcr_yolo_pose_device", "0")
    config["tracking"].setdefault("tdcr_yolo_pose_half", True)
    config["camera"]["enable_infrared_streams"] = True
    config["camera"]["copy_infrared_frames"] = True
    config["camera"]["align_depth_to_color"] = True
    config["camera"]["runtime_align_depth_to_color"] = True
    return config
