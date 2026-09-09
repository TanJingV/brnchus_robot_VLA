from __future__ import annotations

import ast
import csv
import json
from dataclasses import replace
import tempfile
import threading
import time
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import cv2
import numpy as np

from Visual_information.d435_tdcr_capture.axes import NullAxisSource, SyntheticAxisSource, map_native_axes, sample_trio_connection
from Visual_information.d435_tdcr_capture.app import D435CaptureWindow, LatestCaptureWorker, discover_side_rgb_cameras
from Visual_information.d435_tdcr_capture.camera import RealSenseSource, SyntheticCameraSource, SyntheticSideRgbSource
from Visual_information.d435_tdcr_capture.config import PROJECT_ROOT, load_config
from Visual_information.d435_tdcr_capture.em import EmFrame, EmMotorCsvRecorder, EmToolPose
from Visual_information.d435_tdcr_capture.engine import CaptureEngine
from Visual_information.d435_tdcr_capture.multiview import D435_INTERNAL_MODE, DUAL_VIEW_MODE, SideViewFusion
from Visual_information.d435_tdcr_capture.recording import AuxiliaryRgbRecorder, ViewerStreamRecorder
from Visual_information.d435_tdcr_capture.sync import AxisSampler, estimate_delay_ms
from Visual_information.d435_tdcr_capture.tracking import MarkerTracker, detect_color_bands


class FakeTrio:
    def GetAxisParameter_DPOS(self, axis):
        return 10.0 + axis

    def GetAxisParameter_MPOS(self, axis):
        if axis == 4:
            raise RuntimeError("encoder unavailable")
        return 9.5 + axis


class FakeCameraInfo:
    def __init__(self, description: str, device_name: str):
        self._description = description
        self._device_name = device_name

    def description(self):
        return self._description

    def deviceName(self):
        return self._device_name


class FakeEmSource:
    state = "tracking"

    def __init__(self):
        matrix = np.eye(4, dtype=float)
        matrix[:3, 3] = (12.0, -3.0, 45.0)
        self.frame = EmFrame(
            sequence=7,
            host_ns=time.perf_counter_ns(),
            tools=(EmToolPose(10, time.time(), 1234, matrix, 0.2, True),),
        )

    def latest(self):
        return self.frame

    def device_rate_hz(self):
        return 40.0


class D435CaptureTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_mujoco_alignment_is_disabled_by_default(self):
        self.assertFalse(self.config["mujoco"]["enabled"])
        self.assertEqual(self.config["ui"]["workflow_mode"], "capture")
        self.assertFalse(self.config["tracking"]["enabled"])
        self.assertEqual(self.config["camera"]["color"], [1920, 1080, 30])
        self.assertEqual(self.config["camera"]["depth"], [1280, 720, 30])
        self.assertTrue(self.config["camera"]["align_depth_to_color"])
        self.assertFalse(self.config["camera"]["runtime_align_depth_to_color"])
        self.assertTrue(self.config["camera"]["depth_postprocess_enabled"])
        self.assertTrue(self.config["camera"]["native_depth_colorizer_enabled"])
        self.assertEqual(self.config["camera"]["visual_preset"], "high_density")
        self.assertEqual(self.config["ui"]["depth_view_space"], "native")
        engine = CaptureEngine(self.config, SyntheticCameraSource(self.config))
        self.assertIsNone(engine.bridge)

    def test_synthetic_frame_keeps_native_and_aligned_depth_paths_separate(self):
        source = SyntheticCameraSource(self.config)
        source.start()
        frame = source.poll()
        source.stop()
        self.assertIsNotNone(frame)
        self.assertIsNotNone(frame.raw_depth_z16)
        self.assertIsNotNone(frame.native_depth_m)
        self.assertEqual(frame.native_depth_m.shape, frame.raw_depth_z16.shape)
        np.testing.assert_allclose(frame.native_depth_m, frame.depth_m)
        self.assertEqual(frame.depth_intrinsics["width"], frame.native_depth_m.shape[1])

    def test_side_camera_discovery_keeps_directshow_index_and_excludes_d435(self):
        devices = discover_side_rgb_cameras([
            FakeCameraInfo("Intel RealSense D435 RGB", "realsense-serial"),
            FakeCameraInfo("USB Camera", "usb-camera-a"),
            FakeCameraInfo("Document Camera", "usb-camera-b"),
        ])
        self.assertEqual([device["index"] for device in devices], [1, 2])
        self.assertEqual([device["name"] for device in devices], ["USB Camera", "Document Camera"])

    def test_d435_usb2_connection_is_rejected_with_actionable_message(self):
        source = RealSenseSource(self.config["camera"])
        with patch.object(RealSenseSource, "devices", return_value=[{
            "name": "RealSense D435",
            "serial": "TEST",
            "firmware": "test",
            "usb": "2.1",
        }]):
            with self.assertRaisesRegex(RuntimeError, "USB 3.x"):
                source.start()

    def test_direct_seven_axis_mapping_and_dpos_mpos_separation(self):
        sample = sample_trio_connection(FakeTrio(), self.config["axes"])
        np.testing.assert_allclose(sample.demand_native, np.arange(7) + 10.0)
        self.assertFalse(sample.measured_valid[4])
        self.assertTrue(sample.measured_valid[0])
        expected_source = sample.measured_native.copy()
        expected_source[4] = sample.demand_native[4]
        mapped = map_native_axes(sample.demand_native, sample.measured_native, self.config["axes"])
        np.testing.assert_allclose(mapped[:6], np.asarray([0.021] * 3 + [0.042] * 3) - expected_source[:6] * 0.001)
        self.assertAlmostEqual(mapped[6], expected_source[6] * 0.001)

    def test_synthetic_rgb_ir_tracking_returns_all_points(self):
        self.config["tracking"]["enabled"] = True
        source = SyntheticCameraSource(self.config)
        engine = CaptureEngine(self.config, source)
        engine.start()
        sample = None
        deadline = time.time() + 2.0
        while sample is None and time.time() < deadline:
            sample = engine.process_once()
            time.sleep(0.005)
        engine.stop()
        self.assertIsNotNone(sample)
        self.assertEqual(np.count_nonzero(sample.keypoints.valid), 7)
        self.assertTrue(all(source_name == "stereo_ir" for source_name in sample.keypoints.source))
        self.assertLess(np.max(np.abs(sample.keypoints.base_m[:, 0] - np.linspace(0, 0.042, 7))), 0.0015)

    def test_black_body_segmentation_orders_seven_pale_bands_and_rejects_distractor(self):
        image = np.full((360, 640, 3), (96, 126, 121), dtype=np.uint8)
        tip = np.asarray([120.0, 140.0])
        base = np.asarray([570.0, 225.0])
        cv2.line(image, tuple(tip.astype(int)), tuple(base.astype(int)), (18, 18, 18), 14, cv2.LINE_AA)
        tangent = (base - tip) / np.linalg.norm(base - tip)
        normal = np.asarray([-tangent[1], tangent[0]])
        expected = np.zeros((7, 2), dtype=float)
        for tip_order in range(7):
            marker_index = 6 - tip_order
            center = tip + tangent * (12.0 + tip_order * 28.0)
            expected[marker_index] = center
            colour = np.asarray(self.config["markers"][marker_index]["display_bgr"])
            pale_colour = tuple(map(int, np.round(0.7 * colour + 0.3 * 255.0)))
            start = tuple(map(int, np.round(center - normal * 9.0)))
            end = tuple(map(int, np.round(center + normal * 9.0)))
            cv2.line(image, start, end, pale_colour, 5, cv2.LINE_AA)
        cv2.circle(
            image,
            (60, 60),
            12,
            tuple(map(int, self.config["markers"][0]["display_bgr"])),
            -1,
        )
        tracker = MarkerTracker(self.config)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        pixels, confidence, _masks, _colours, segmentation = detect_color_bands(
            hsv,
            lab,
            self.config["markers"],
            tracker.color_prototypes,
            self.config["tracking"],
            np.full((7, 2), np.nan),
            (0.0, 0.0, 1.0, 1.0),
        )
        self.assertTrue(segmentation.valid)
        self.assertEqual(np.count_nonzero(np.isfinite(pixels).all(axis=1)), 7)
        self.assertLess(float(np.max(np.linalg.norm(pixels - expected, axis=1))), 1.0)
        self.assertGreater(float(np.min(confidence)), 0.45)

    def test_ir_stereo_recovers_keypoint_when_aligned_depth_has_a_hole(self):
        source = SyntheticCameraSource(self.config)
        source.start()
        frame = source.poll()
        reference = MarkerTracker(self.config).process(frame)
        depth_with_hole = frame.depth_m.copy()
        u, v = np.round(reference.color_pixels[3]).astype(int)
        cv2.circle(depth_with_hole, (int(u), int(v)), 12, 0.0, -1)
        recovered = MarkerTracker(self.config).process(
            replace(frame, depth_m=depth_with_hole)
        )
        source.stop()
        self.assertTrue(recovered.valid[3])
        self.assertEqual(recovered.source[3], "stereo_ir")
        self.assertLess(
            abs(float(recovered.raw_camera_m[3, 2] - reference.raw_camera_m[3, 2])),
            0.001,
        )

    def test_background_capture_worker_exposes_only_the_latest_completed_frame(self):
        source = SyntheticCameraSource(self.config)
        engine = CaptureEngine(self.config, source)
        worker = LatestCaptureWorker(engine)
        engine.start()
        worker.start()
        deadline = time.time() + 1.0
        version = 0
        packet = None
        while packet is None and time.time() < deadline:
            time.sleep(0.01)
            version, packet, error = worker.latest_after(version)
            self.assertIsNone(error)
        worker.stop()
        engine.stop()
        self.assertIsNotNone(packet)
        self.assertGreater(version, 0)
        self.assertGreater(packet.processing_ms, 0.0)

    def test_high_resolution_colour_tracking_preserves_full_resolution_pixels(self):
        source = SyntheticCameraSource(self.config)
        tracker = MarkerTracker(self.config)
        tracker.set_transform_base_from_camera(source.transform_base_from_camera, ready=True)
        source.start()
        frame = source.poll()
        scaled_intrinsics = dict(frame.color_intrinsics)
        for field in ("fx", "fy", "ppx", "ppy"):
            scaled_intrinsics[field] *= 2.0
        scaled_intrinsics["width"] = frame.color_bgr.shape[1] * 2
        scaled_intrinsics["height"] = frame.color_bgr.shape[0] * 2
        high_resolution = replace(
            frame,
            color_bgr=cv2.resize(frame.color_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST),
            depth_m=cv2.resize(frame.depth_m, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST),
            color_intrinsics=scaled_intrinsics,
        )
        observation = tracker.process(high_resolution)
        source.stop()
        self.assertEqual(np.count_nonzero(observation.valid), 7)
        self.assertGreater(float(np.nanmax(observation.color_pixels[:, 0])), 640.0)

    def test_colour_tracker_rejects_large_same_colour_distractor_by_temporal_identity(self):
        source = SyntheticCameraSource(self.config)
        tracker = MarkerTracker(self.config)
        tracker.set_transform_base_from_camera(source.transform_base_from_camera, ready=True)
        source.start()
        first_frame = source.poll()
        first = tracker.process(first_frame)
        deadline = time.time() + 1.0
        second_frame = None
        while second_frame is None and time.time() < deadline:
            second_frame = source.poll()
            time.sleep(0.005)
        corrupted_color = second_frame.color_bgr.copy()
        cv2.circle(corrupted_color, (90, 90), 14, tuple(self.config["markers"][0]["display_bgr"]), -1)
        corrupted = replace(second_frame, color_bgr=corrupted_color)
        second = tracker.process(corrupted)
        source.stop()
        self.assertTrue(second.valid[0])
        self.assertLess(np.linalg.norm(second.color_pixels[0] - first.color_pixels[0]), 8.0)
        self.assertGreater(np.linalg.norm(second.color_pixels[0] - np.asarray([90.0, 90.0])), 100.0)

    def test_current_frame_colour_calibration_updates_all_seven_prototypes(self):
        source = SyntheticCameraSource(self.config)
        tracker = MarkerTracker(self.config)
        source.start()
        frame = source.poll()
        observation = tracker.process(frame)
        calibrated = tracker.calibrate_color_prototypes(frame, observation)
        source.stop()
        self.assertEqual(calibrated, 7)
        prototypes = self.config["tracking"]["color_prototypes"]
        self.assertEqual(set(prototypes), {marker["name"] for marker in self.config["markers"]})
        self.assertTrue(all(set(value) == {"hue", "lab_a", "lab_b"} for value in prototypes.values()))

    def test_keypoint_capture_switch_skips_tracking_and_returns_explicit_invalid_points(self):
        source = SyntheticCameraSource(self.config)
        engine = CaptureEngine(self.config, source)
        engine.set_keypoint_tracking_enabled(False)
        engine.start()
        with patch.object(engine.tracker, "process", side_effect=AssertionError("tracker must stay idle")):
            sample = None
            deadline = time.time() + 1.0
            while sample is None and time.time() < deadline:
                sample = engine.process_once()
                time.sleep(0.005)
        engine.stop()
        self.assertIsNotNone(sample)
        self.assertEqual(np.count_nonzero(sample.keypoints.valid), 0)
        self.assertTrue(np.isnan(sample.keypoints.raw_camera_m).all())
        self.assertTrue(all(source_name == "tracking_disabled" for source_name in sample.keypoints.source))
        self.assertFalse(self.config["tracking"]["enabled"])

    def test_tracking_roi_follows_current_view_and_preserves_global_pixels(self):
        self.config["tracking"]["enabled"] = True
        source = SyntheticCameraSource(self.config)
        engine = CaptureEngine(self.config, source)
        roi = (0.48, 0.44, 0.52, 0.56)
        engine.set_tracking_rois(roi, (0.0, 0.0, 1.0, 1.0))
        engine.start()
        sample = None
        deadline = time.time() + 1.0
        while sample is None and time.time() < deadline:
            sample = engine.process_once()
            time.sleep(0.005)
        self.assertIsNotNone(sample)
        np.testing.assert_allclose(sample.d435_tracking_roi, roi)
        self.assertTrue(sample.keypoints.valid[0])
        self.assertLess(np.count_nonzero(sample.keypoints.valid), 7)
        self.assertGreater(sample.keypoints.color_pixels[0, 0], 300.0)
        engine.set_tracking_rois((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0))
        restored = None
        deadline = time.time() + 1.0
        while restored is None and time.time() < deadline:
            restored = engine.process_once()
            time.sleep(0.005)
        engine.stop()
        self.assertEqual(np.count_nonzero(restored.keypoints.valid), 7)

    def test_dual_view_mode_fuses_all_seven_points_and_switches_cleanly(self):
        self.config["tracking"]["enabled"] = True
        self.config["fusion"]["mode"] = DUAL_VIEW_MODE
        primary = SyntheticCameraSource(self.config)
        side = SyntheticSideRgbSource(self.config, primary)
        engine = CaptureEngine(self.config, primary, side_camera_source=side)
        engine.start()
        sample = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            candidate = engine.process_once()
            if candidate is not None and candidate.side_frame is not None and all(
                source.startswith("dual_") for source in candidate.keypoints.source
            ):
                sample = candidate
                break
            time.sleep(0.005)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.fusion_mode, DUAL_VIEW_MODE)
        self.assertEqual(np.count_nonzero(sample.keypoints.valid), 7)
        self.assertTrue(np.isfinite(sample.keypoints.side_pixels).all())
        self.assertLess(np.max(np.abs(sample.keypoints.base_m[:, 0] - np.linspace(0, 0.042, 7))), 0.0015)

        engine.set_fusion_mode(D435_INTERNAL_MODE)
        switched = None
        deadline = time.time() + 1.0
        while switched is None and time.time() < deadline:
            switched = engine.process_once()
            time.sleep(0.005)
        engine.stop()
        self.assertIsNotNone(switched)
        self.assertEqual(switched.fusion_mode, D435_INTERNAL_MODE)
        self.assertTrue(all(source == "stereo_ir" for source in switched.keypoints.source))

    def test_dual_view_session_records_side_timing_and_pixels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.config["tracking"]["enabled"] = True
            self.config["recording"]["synchronized_video_enabled"] = True
            self.config["recording"]["root"] = temp_dir
            self.config["fusion"]["mode"] = DUAL_VIEW_MODE
            primary = SyntheticCameraSource(self.config)
            side = SyntheticSideRgbSource(self.config, primary)
            engine = CaptureEngine(self.config, primary, side_camera_source=side)
            engine.start()
            session = Path(engine.start_recording())
            deadline = time.time() + 0.30
            while time.time() < deadline:
                engine.process_once()
                time.sleep(0.005)
            engine.stop_recording()
            engine.stop()
            with np.load(session / "keypoints.npz") as archive:
                self.assertIn("side_pixels", archive.files)
                self.assertIn("side_capture_host_ns", archive.files)
                self.assertEqual(archive["side_pixels"].shape[1:], (7, 2))
                self.assertTrue(np.any(archive["side_capture_host_ns"] > 0))
            self.assertTrue((session / "side_overlay.mp4").exists())
            for name in (
                "d435_rgb.mp4",
                "side_rgb_synchronized.mp4",
                "synchronized_rgb.mp4",
                "video_timestamps.csv",
            ):
                self.assertTrue((session / name).exists(), name)
                self.assertGreater((session / name).stat().st_size, 0, name)
            with (session / "video_timestamps.csv").open("r", encoding="utf-8-sig", newline="") as stream:
                video_rows = list(csv.DictReader(stream))
            self.assertGreater(len(video_rows), 2)
            self.assertTrue(all(row["side_valid"] == "1" for row in video_rows))
            self.assertEqual(
                [int(row["video_frame_index"]) for row in video_rows],
                list(range(len(video_rows))),
            )
            video_counts = []
            for name in ("d435_rgb.mp4", "side_rgb_synchronized.mp4", "synchronized_rgb.mp4"):
                capture = cv2.VideoCapture(str(session / name))
                video_counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                capture.release()
            self.assertEqual(video_counts, [len(video_rows)] * 3)

    def test_dual_view_rejects_corrupted_d435_depth(self):
        primary = SyntheticCameraSource(self.config)
        side = SyntheticSideRgbSource(self.config, primary)
        primary.start()
        side.start()
        frame = primary.poll()
        side_frame = side.poll_nearest(frame.capture_host_ns)
        tracker = MarkerTracker(self.config)
        tracker.set_transform_base_from_camera(primary.transform_base_from_camera, ready=True)
        internal = tracker.process(frame)
        corrupted = internal.raw_camera_m.copy()
        corrupted[:, 2] += 0.030
        bad_observation = replace(internal, raw_camera_m=corrupted)
        fusion = SideViewFusion(self.config)
        fusion.set_d435_transform(primary.transform_base_from_camera)
        fusion.set_side_calibration(side.transform_base_from_camera, side.intrinsics, ready=True)
        fused = fusion.fuse(frame, bad_observation, side_frame)
        primary.stop()
        side.stop()
        self.assertTrue(all(source == "dual_triangulated_depth_rejected" for source in fused.source))
        self.assertLess(np.max(np.abs(fused.base_m[:, 2])), 0.012)

    def test_recording_writes_replayable_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.config["recording"]["root"] = temp_dir
            engine = CaptureEngine(self.config, SyntheticCameraSource(self.config))
            engine.start()
            session = Path(engine.start_recording())
            deadline = time.time() + 0.35
            while time.time() < deadline:
                engine.process_once()
                time.sleep(0.005)
            engine.stop_recording()
            engine.stop()
            for name in ("samples.csv", "keypoints.npz", "session.json", "calibration.json", "summary.json"):
                self.assertTrue((session / name).exists(), name)
            with np.load(session / "keypoints.npz") as archive:
                self.assertGreater(len(archive["capture_host_ns"]), 3)
                self.assertEqual(archive["keypoints_base_m"].shape[1:], (7, 3))
                self.assertEqual(archive["keypoint_covariance_m2"].shape[1:], (7, 3, 3))
                self.assertEqual(archive["keypoint_color_pixels"].shape[1:], (7, 2))
                self.assertEqual(archive["keypoint_left_pixels"].shape[1:], (7, 2))
                self.assertEqual(archive["keypoint_right_pixels"].shape[1:], (7, 2))
                self.assertEqual(archive["keypoint_source"].shape[1:], (7,))
                self.assertEqual(archive["keypoint_color_confidence"].shape[1:], (7,))
                self.assertEqual(archive["d435_tracking_roi"].shape[1:], (4,))
                self.assertEqual(archive["side_tracking_roi"].shape[1:], (4,))
            with (session / "samples.csv").open("r", encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream))
            for field in (
                "kp0_color_u_px", "kp0_left_u_px", "kp0_right_u_px",
                "kp0_disparity_px", "kp0_sigma_x_mm", "kp0_sigma_z_mm",
                "kp0_color_confidence",
            ):
                self.assertIn(field, header)

    def test_viewer_recorder_writes_selected_windows_and_camera_axis_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.config["recording"]["root"] = temp_dir
            recorder = ViewerStreamRecorder(self.config)
            axis_sampler = AxisSampler(
                SyntheticAxisSource(self.config["axes"]), sample_rate_hz=100.0
            )
            axis_sampler.start()
            session = recorder.start({"rgb", "depth"}, axis_sampler=axis_sampler)
            source = SyntheticCameraSource(self.config)
            source.start()
            for _ in range(6):
                frame = None
                while frame is None:
                    frame = source.poll()
                    time.sleep(0.002)
                recorder.append_camera(
                    frame, axis_sampler.interpolate(frame.capture_host_ns)
                )
            source.stop()
            recorder.stop()
            axis_sampler.stop()
            for name in (
                "rgb.mp4", "depth.mp4", "camera_axes.csv",
                "motor_axes_100hz.csv", "capture.json", "summary.json",
            ):
                self.assertTrue((session / name).exists(), name)
                self.assertGreater((session / name).stat().st_size, 0, name)
            with (session / "camera_axes.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertIn("axis_6_measured_native", rows[0])
            summary = json.loads((session / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(recorder.output_frame_counts, summary["encoded_frames"])
            self.assertEqual(
                summary["encoded_frames"]["rgb"],
                summary["encoded_frames"]["depth"],
            )
            self.assertGreater(summary["motor_sample_rate_hz"], 60.0)
            self.assertFalse(summary["errors"])
            decoded_counts = []
            for name in ("rgb.mp4", "depth.mp4"):
                capture = cv2.VideoCapture(str(session / name))
                self.assertTrue(capture.isOpened())
                self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 30.0)
                decoded_counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                capture.release()
            self.assertEqual(decoded_counts, [summary["encoded_frames"]["rgb"]] * 2)

    def test_endoscope_recorder_writes_independent_video_and_timestamps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir)
            primary = SyntheticCameraSource(self.config)
            source = SyntheticSideRgbSource(self.config, primary)
            primary.start()
            source.start()
            recorder = AuxiliaryRgbRecorder(self.config)
            recorder.start(session, source)
            deadline = time.monotonic() + 2.0
            while recorder.frame_count < 6 and time.monotonic() < deadline:
                time.sleep(0.01)
            recorder.stop()
            source.stop()
            primary.stop()
            self.assertIsNone(recorder.error)
            for name in (
                "endoscope_rgb.mp4",
                "endoscope_timestamps.csv",
                "endoscope_summary.json",
            ):
                self.assertTrue((session / name).exists(), name)
                self.assertGreater((session / name).stat().st_size, 0, name)
            summary = json.loads(
                (session / "endoscope_summary.json").read_text(encoding="utf-8")
            )
            self.assertGreaterEqual(summary["source_frames"], 6)
            self.assertEqual(summary["source_frames"], summary["encoded_frames"])
            self.assertEqual(
                summary["encoded_frames"], summary["decoded_validation"]["frames"]
            )

    def test_em_csv_has_one_row_per_motor_tick_and_preserves_native_frame_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            axis_sampler = AxisSampler(
                SyntheticAxisSource(self.config["axes"]), sample_rate_hz=100.0
            )
            axis_sampler.start()
            recorder = EmMotorCsvRecorder(self.config["em"])
            recorder.start(temp_dir, axis_sampler, FakeEmSource())
            time.sleep(0.18)
            recorder.stop()
            axis_sampler.stop()
            path = Path(temp_dir) / "em_data_100hz.csv"
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreater(len(rows), 10)
            self.assertEqual(len(rows), recorder.tick_count)
            self.assertEqual({row["em_frame_number"] for row in rows}, {"1234"})
            self.assertEqual(sum(int(row["new_em_frame"]) for row in rows), 1)
            self.assertTrue(all(float(row["tx_mm"]) == 12.0 for row in rows))
            summary = json.loads(
                (Path(temp_dir) / "em_summary.json").read_text(encoding="utf-8")
            )
            self.assertGreater(summary["motor_clock_rate_hz"], 60.0)
            self.assertEqual(summary["csv_rows"], summary["motor_ticks"])

    def test_depth_display_filter_fills_transient_surface_holes_without_crossing_edges(self):
        preview = SimpleNamespace(
            config=self.config,
            _depth_preview_m=None,
            _depth_preview_age=None,
            _cloud_depth_preview_m=None,
            _cloud_depth_preview_age=None,
            _depth_preview_filter_lock=threading.Lock(),
            _cloud_depth_filter_lock=threading.Lock(),
        )

        flat = np.full((80, 120), 0.30, dtype=np.float32)
        D435CaptureWindow._filtered_depth_preview(preview, flat)
        holes = flat.copy()
        missing_pixels = ((20, 20), (35, 65), (60, 95))
        for row, column in missing_pixels:
            holes[row, column] = 0.0
        filtered, valid, _, _ = D435CaptureWindow._filtered_depth_preview(preview, holes)
        for row, column in missing_pixels:
            self.assertTrue(valid[row, column])
            self.assertAlmostEqual(float(filtered[row, column]), 0.30, places=4)

        D435CaptureWindow._reset_depth_display_filters(preview)
        stepped = np.full((80, 120), 0.30, dtype=np.float32)
        stepped[:, 60:] = 0.45
        D435CaptureWindow._filtered_depth_preview(preview, stepped)
        edge_hole = stepped.copy()
        edge_hole[40, 60] = 0.0
        _, edge_valid, _, _ = D435CaptureWindow._filtered_depth_preview(preview, edge_hole)
        self.assertFalse(edge_valid[40, 60])

    def test_measurement_sites_are_exactly_seven_mm_apart(self):
        xml_path = PROJECT_ROOT / "meshes" / "cable_robot_bronch_final_seg2.xml"
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"measurement_kp_{i}") for i in range(7)]
        self.assertTrue(all(identifier >= 0 for identifier in site_ids))
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "base_frame")
        rotation = data.site_xmat[base_id].reshape(3, 3)
        points = (rotation.T @ (data.site_xpos[site_ids] - data.site_xpos[base_id]).T).T
        np.testing.assert_allclose(points[:, 0], np.linspace(0, 0.042, 7), atol=1e-10)
        np.testing.assert_allclose(points[:, 1:], 0.0, atol=1e-10)

    def test_delay_estimator(self):
        values = np.sin(np.linspace(0, 12, 400)) + 0.2 * np.sin(np.linspace(0, 41, 400))
        response = np.concatenate((np.zeros(12), values[:-12]))
        delay = estimate_delay_ms(values, response, 0.01)
        self.assertAlmostEqual(delay, 120.0, delta=15.0)

    def test_main_gui_remains_parseable_without_writing_pyc(self):
        real_path = PROJECT_ROOT / "window" / "界面1220（可以使用版本+数据记录）双探子版.py"
        simulation_path = PROJECT_ROOT / "window" / "界面1220（可以使用版本+数据记录）双探子版_仿真版.py"
        automation_path = PROJECT_ROOT / "window" / "simulation_automation_mixin.py"
        runtime_path = PROJECT_ROOT / "window" / "simulation_runtime_mixin.py"
        ast.parse(real_path.read_text(encoding="utf-8-sig"), filename=str(real_path))
        simulation_source = simulation_path.read_text(encoding="utf-8-sig")
        simulation_tree = ast.parse(simulation_source, filename=str(simulation_path))
        self.assertIn("load_real_environment_window_class", simulation_source)
        self.assertIn("create_environment_window", simulation_source)
        main_window = next(
            node for node in simulation_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
        )
        self.assertEqual(
            [ast.unparse(base) for base in main_window.bases[:2]],
            ["SimulationAutomationMixin", "SimulationRuntimeMixin"],
        )
        mixin_expectations = (
            (automation_path, "SimulationAutomationMixin", 30),
            (runtime_path, "SimulationRuntimeMixin", 26),
        )
        for path, class_name, expected_methods in mixin_expectations:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            mixin = next(
                node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            self.assertEqual(
                len([node for node in mixin.body if isinstance(node, ast.FunctionDef)]),
                expected_methods,
            )
        self.assertLess(len(simulation_source.splitlines()), 5000)


if __name__ == "__main__":
    unittest.main()
