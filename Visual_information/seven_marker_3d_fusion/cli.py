"""Command-line entry points for live, BAG, session and NPZ processing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from Visual_information.d435_tdcr_capture.camera import RealSenseSource, SyntheticCameraSource

from .config import load_fusion_config
from .io import ResultWriter, default_output_dir, refine_existing_npz
from .pipeline import SevenMarkerFusionPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m Visual_information.seven_marker_3d_fusion",
        description="Reconstruct seven TDCR marker coordinates from RGB, aligned depth and left/right IR.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--live", action="store_true", help="Use the connected D435")
    source.add_argument("--bag", type=Path, help="Replay a RealSense BAG")
    source.add_argument("--session", type=Path, help="Read realsense.bag or keypoints.npz in a session")
    source.add_argument("--npz", type=Path, help="Refine an existing keypoints.npz")
    source.add_argument("--synthetic", action="store_true", help="Run the deterministic no-hardware demo")
    parser.add_argument("--config", type=Path, help="Main d435_tdcr_capture JSON override")
    parser.add_argument("--fusion-config", type=Path, help="Fusion-only JSON override")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means until Q/EOF")
    parser.add_argument("--roi", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"), default=(0, 0, 1, 1))
    parser.add_argument(
        "--target-roi",
        nargs=4,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Tight first-frame box around the black seven-ring terminal section (required for real RGB)",
    )
    parser.add_argument("--headless", action="store_true", help="Do not open the OpenCV preview")
    parser.add_argument("--no-overlay-video", action="store_true", help="Do not save the diagnostic MP4")
    parser.add_argument("--repeat", action="store_true", help="Repeat BAG playback")
    return parser


def _resolve_source(args) -> tuple[str, Path | None]:
    if args.session is not None:
        session = args.session.expanduser().resolve()
        bag = session / "realsense.bag"
        archive = session / "keypoints.npz"
        if bag.exists():
            return "bag", bag
        if archive.exists():
            return "npz", archive
        raise FileNotFoundError(f"No realsense.bag or keypoints.npz in {session}")
    if args.bag is not None:
        return "bag", args.bag.expanduser().resolve()
    if args.npz is not None:
        return "npz", args.npz.expanduser().resolve()
    if args.synthetic:
        return "synthetic", None
    return "live", None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_fusion_config(args.config, args.fusion_config)
    mode, path = _resolve_source(args)
    output = args.output.expanduser().resolve() if args.output else default_output_dir()
    if mode == "npz":
        target = refine_existing_npz(path, output, config)
        print(f"Seven-marker refinement complete: {target}")
        return 0

    source = (
        SyntheticCameraSource(config)
        if mode == "synthetic"
        else RealSenseSource(config["camera"], bag_path=path, repeat=args.repeat)
    )
    pipeline = SevenMarkerFusionPipeline(config)
    pipeline.set_roi(tuple(args.roi))
    if mode != "synthetic":
        if not pipeline.yolo_pose.ready and pipeline.yolo_pose.strict:
            raise SystemExit(
                "Strict learned-keypoint mode requires Visual_information/models/tdcr_yolo_pose/best.pt. "
                "Use the UI annotation and training tools first."
            )
        if args.target_roi is None and not pipeline.yolo_pose.ready:
            raise SystemExit(
                "Real RGB processing requires --target-roi X0 Y0 X1 Y1. "
                "The graphical UI is recommended because it lets you draw and verify this lock box."
            )
        if args.target_roi is not None:
            pipeline.set_target_lock(tuple(args.target_roi))
    fps = float(config["camera"]["color"][2])
    writer = ResultWriter(output, save_overlay=not args.no_overlay_video, fps=fps)
    metadata = {
        "mode": mode,
        "source": str(path) if path else "connected D435",
        "roi": list(args.roi),
        "target_lock_roi": list(args.target_roi) if args.target_roi is not None else None,
    }
    source.start()
    frame_count = 0
    last_frame_time = time.monotonic()
    startup_time = last_frame_time
    bag_depth_only_fallback_used = False
    try:
        while True:
            frame = source.poll()
            if frame is None:
                # BAG playback has no uniform Python EOF event.  A completed
                # file is considered ended after the producer stays idle.
                if mode == "bag" and frame_count and time.monotonic() - last_frame_time > 1.5:
                    break
                if mode == "bag" and not frame_count and time.monotonic() - startup_time > 2.0:
                    if bag_depth_only_fallback_used:
                        raise RuntimeError(
                            "The BAG yielded no RGB+Depth frames. Check that it contains D435 streams."
                        )
                    # Some older capture sessions recorded RGB+Depth but not
                    # explicit left/right IR streams.  Reopen once without an
                    # IR requirement and retain the depth-only fusion route.
                    source.stop()
                    config["camera"]["enable_infrared_streams"] = False
                    config["camera"]["copy_infrared_frames"] = False
                    source = RealSenseSource(config["camera"], bag_path=path, repeat=args.repeat)
                    source.start()
                    bag_depth_only_fallback_used = True
                    metadata["bag_depth_only_fallback"] = True
                    startup_time = time.monotonic()
                    continue
                if mode in ("live", "synthetic") and not frame_count and time.monotonic() - startup_time > 10.0:
                    raise RuntimeError("No complete camera frame arrived within 10 seconds")
                time.sleep(0.001)
                continue
            last_frame_time = time.monotonic()
            result = pipeline.process(frame)
            overlay = pipeline.draw_overlay(frame, result)
            writer.append(result, overlay)
            frame_count += 1
            if not args.headless:
                preview_width = min(1280, overlay.shape[1])
                preview_height = int(round(overlay.shape[0] * preview_width / overlay.shape[1]))
                preview = cv2.resize(overlay, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
                cv2.imshow("Seven-marker 3D fusion - Q to stop", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        cv2.destroyAllWindows()
        target = writer.close(metadata)
    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    print(
        f"Seven-marker reconstruction complete: {target}\n"
        f"frames={summary['frames']}, mean measured={summary['mean_measured_points']:.2f}/7, "
        f"all-seven={summary['all_seven_fraction']:.1%}"
    )
    return 0
