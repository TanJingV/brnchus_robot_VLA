from __future__ import annotations

import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import cv2

from Visual_information.d435_tdcr_capture.camera import SyntheticCameraSource
from Visual_information.seven_marker_3d_fusion.band_fallback import extract_periodic_bands
from Visual_information.seven_marker_3d_fusion.body_first_detector import BodyFirstBandDetector, BodyFirstBandResult
from Visual_information.seven_marker_3d_fusion.config import load_fusion_config
from Visual_information.seven_marker_3d_fusion.cotracker3_tracker import CoTracker3SequenceTracker, TrackedPointRefiner
from Visual_information.seven_marker_3d_fusion.curve_model import (
    fit_continuum_shape,
    project_material_keypoints,
    tube_mesh,
)
from Visual_information.seven_marker_3d_fusion.em_constraint import extract_em_pair
from Visual_information.seven_marker_3d_fusion.io import refine_existing_npz
from Visual_information.seven_marker_3d_fusion.legacy_refinement import LocalDepthCurveFusion
from Visual_information.seven_marker_3d_fusion.measurement import covariance_intersection
from Visual_information.seven_marker_3d_fusion.material_chain_tracker import MaterialColorChainTracker
from Visual_information.seven_marker_3d_fusion.models import ModalityMeasurement, SevenMarkerResult
from Visual_information.seven_marker_3d_fusion.motor_prior import (
    MotorShapePriorFusion,
    motor_controls_to_local_points,
)
from Visual_information.seven_marker_3d_fusion.neural_body_segmenter import NeuralContinuumSegmenter
from Visual_information.seven_marker_3d_fusion.offline_refinement import refine_output_trajectory
from Visual_information.seven_marker_3d_fusion.pipeline import SevenMarkerFusionPipeline
from Visual_information.seven_marker_3d_fusion.sequence_repair import repair_ordered_ring_sequence
from Visual_information.seven_marker_3d_fusion.session import TurboDepthDecoder
from Visual_information.seven_marker_3d_fusion.smoother import ShapeConstrainedSmoother
from Visual_information.seven_marker_3d_fusion.yolo_pose_detector import TdcrYoloPoseDetector
from Visual_information.d435_tdcr_capture.tracking import MarkerTracker, detect_color_bands, scaled_tracking_config


class SevenMarkerFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_fusion_config()

    def test_cotracker_checkpoint_and_no_training_default(self):
        tracker = CoTracker3SequenceTracker(self.config["tracking"])
        self.assertTrue(tracker.ready)
        self.assertFalse(self.config["tracking"]["cotracker3_enabled"])
        self.assertFalse(self.config["tracking"]["tdcr_yolo_pose_strict"])

    def test_material_colour_chain_tracks_translation_and_bending(self):
        height, width = 360, 640
        base_x = np.linspace(150.0, 390.0, 7)
        display_colours = [
            tuple(map(int, marker["display_bgr"])) for marker in self.config["markers"]
        ]

        def render(frame_index):
            phase = frame_index / 20.0
            points = np.column_stack((
                base_x + 0.9 * frame_index,
                180.0 + 17.0 * phase * np.sin(np.linspace(0.0, np.pi, 7)),
            )).astype(np.float32)
            image = np.full((height, width, 3), (82, 126, 112), dtype=np.uint8)
            cv2.polylines(
                image, [np.rint(points).astype(np.int32)], False,
                (24, 25, 24), 15, cv2.LINE_AA,
            )
            for point, colour in zip(points, display_colours):
                cv2.circle(image, tuple(np.rint(point).astype(int)), 7, colour, -1, cv2.LINE_AA)
                cv2.circle(image, tuple(np.rint(point).astype(int)), 7, (240, 240, 240), 1, cv2.LINE_AA)
            return image, points

        first_image, first_points = render(0)
        tracker = MaterialColorChainTracker(self.config["tracking"])
        tracker.initialise(first_image, first_points)
        latest = None
        truth = None
        for frame_index in range(21):
            image, truth = render(frame_index)
            latest = tracker.update(image)
            self.assertTrue(latest.accepted, latest.reason)
            self.assertTrue(np.isfinite(latest.pixels).all())
        error = np.linalg.norm(latest.pixels - truth, axis=1)
        gaps = np.linalg.norm(np.diff(latest.pixels, axis=0), axis=1)
        self.assertLess(float(np.max(error)), 6.0)
        self.assertLess(float(np.std(gaps) / np.mean(gaps)), 0.025)
    def test_body_lattice_keeps_manual_identity_order_and_repairs_collapse(self):
        pipeline = SevenMarkerFusionPipeline(self.config)
        reference = np.column_stack((80.0 + np.arange(7) * 32.0, np.full(7, 70.0)))
        pipeline._keypoint_order_reference = reference.copy()
        mask = np.zeros((140, 340), np.uint8)
        body = BodyFirstBandResult(
            accepted=True,
            pixels=(reference + [4.0, 2.0])[::-1].copy(),
            confidence=np.ones(7), body_mask=mask, search_mask=mask,
            centerline_xy=reference.astype(np.float32),
            band_masks=tuple(mask.copy() for _ in range(7)),
            score=1.0, reason="synthetic",
        )
        aligned = pipeline._align_body_result_to_manual_order(body)
        self.assertTrue(np.all(np.diff(aligned.pixels[:, 0]) > 0.0))
        collapsed = aligned.pixels.copy()
        collapsed[4] = collapsed[3] + [2.0, 0.0]
        repaired = pipeline._align_body_result_to_manual_order(
            replace(aligned, pixels=collapsed)
        )
        self.assertTrue(repaired.temporally_predicted)
        gaps = np.linalg.norm(np.diff(repaired.pixels, axis=0), axis=1)
        self.assertGreater(float(np.min(gaps)), 0.90 * float(np.median(gaps)))

    def test_tracked_point_refiner_preserves_seven_ordered_colour_points(self):
        image = np.full((120, 320, 3), (70, 115, 90), dtype=np.uint8)
        points = np.column_stack((np.linspace(260, 80, 7), np.full(7, 60.0)))
        colours = ((40, 40, 230), (40, 130, 230), (40, 220, 220),
                   (40, 210, 60), (230, 210, 40), (230, 90, 40), (210, 40, 180))
        for point, colour in zip(points, colours):
            cv2.circle(image, tuple(np.rint(point).astype(int)), 5, colour, -1)
        refiner = TrackedPointRefiner({"cotracker3_colour_search_radius_px": 8})
        refiner.initialise(image, points)
        shifted = points + np.asarray([3.0, -2.0])
        refined, confidence, valid = refiner.refine(
            image, shifted, np.ones(7), np.full(image.shape[:2], 255, np.uint8)
        )
        self.assertTrue(np.all(valid))
        self.assertTrue(np.all(confidence > 0.25))
        self.assertTrue(np.all(np.diff(refined[:, 0]) < 0.0))

    def test_manual_seed_is_automatically_snapped_to_black_centerline(self):
        image = np.full((180, 420, 3), (72, 118, 92), dtype=np.uint8)
        x = np.linspace(80.0, 320.0, 7)
        centre_y = 74.0 + 0.0008 * (x - 200.0) ** 2
        centreline = np.column_stack((
            np.linspace(55.0, 345.0, 240),
            74.0 + 0.0008 * (np.linspace(55.0, 345.0, 240) - 200.0) ** 2,
        ))
        cv2.polylines(
            image, [np.rint(centreline).astype(np.int32)], False,
            (18, 20, 19), 13, cv2.LINE_AA,
        )
        coarse_clicks = np.column_stack((x, centre_y + 14.0))
        refiner = TrackedPointRefiner()
        corrected = refiner.initialise(image, coarse_clicks)
        expected_y = 74.0 + 0.0008 * (corrected[:, 0] - 200.0) ** 2
        self.assertLess(float(np.median(np.abs(corrected[:, 1] - expected_y))), 4.0)
        gaps = np.linalg.norm(np.diff(corrected, axis=0), axis=1)
        self.assertLess(float(np.max(gaps) / np.min(gaps)), 1.04)

    def test_refiner_fails_closed_when_black_centerline_is_lost(self):
        image = np.full((140, 360, 3), (72, 118, 92), dtype=np.uint8)
        points = np.column_stack((80.0 + np.arange(7) * 34.0, np.full(7, 70.0)))
        cv2.line(image, (55, 70), (315, 70), (16, 18, 17), 13, cv2.LINE_AA)
        refiner = TrackedPointRefiner()
        corrected = refiner.initialise(image, points)
        # Consume the conditioning frame before testing a genuine tracking frame.
        refiner.refine(image, corrected, np.ones(7), None)
        blank = np.full_like(image, 255)
        refined, confidence, valid = refiner.refine(
            blank, corrected, np.ones(7), None
        )
        self.assertFalse(np.isfinite(refined).any())
        self.assertTrue(np.all(confidence == 0.0))
        self.assertFalse(np.any(valid))
        self.assertEqual(refiner.tracking_source, "centerline-lost-reacquiring")

    def test_material_chain_regularizer_rejects_marker_collapse(self):
        previous = np.column_stack((np.arange(7) * 34.0, np.zeros(7))).astype(np.float32)
        observed = previous + np.asarray([5.0, 2.0], np.float32)
        observed[4] = observed[3] + np.asarray([3.0, 0.0], np.float32)
        observed[5] += np.asarray([20.0, -8.0], np.float32)
        regular = TrackedPointRefiner._regularise_material_chain(
            observed, previous, 34.0
        )
        gaps = np.linalg.norm(np.diff(regular, axis=0), axis=1)
        self.assertTrue(np.all(gaps > 0.97 * 34.0))
        self.assertTrue(np.all(gaps < 1.01 * 34.0))
        self.assertTrue(np.all(np.diff(regular[:, 0]) > 0.0))
        second_difference = regular[:-2] - 2.0 * regular[1:-1] + regular[2:]
        self.assertLess(float(np.max(np.linalg.norm(second_difference, axis=1))), 8.0)

    def test_seven_bands_share_one_tangential_phase(self):
        image = np.full((160, 420, 3), 170, np.uint8)
        initial = np.column_stack((80.0 + np.arange(7) * 38.0, np.full(7, 80.0)))
        colours = (
            (20, 30, 230), (20, 140, 245), (20, 220, 245),
            (40, 200, 70), (230, 190, 70), (225, 90, 30), (190, 50, 170),
        )
        for point, colour in zip(initial, colours):
            cv2.circle(image, tuple(np.rint(point).astype(int)), 5, colour, -1)
        refiner = TrackedPointRefiner()
        refiner.initialise(image, initial)
        moved = np.full_like(image, 170)
        expected_shift = np.asarray([-13.0, 6.0])
        for point, colour in zip(initial + expected_shift, colours):
            cv2.circle(moved, tuple(np.rint(point).astype(int)), 5, colour, -1)
        lab = cv2.cvtColor(moved, cv2.COLOR_BGR2LAB).astype(np.float32)
        shift, quality = refiner._joint_band_shift(lab, initial, 38.0)
        self.assertLess(float(np.linalg.norm(shift - expected_shift)), 2.5)
        self.assertTrue(np.isfinite(quality))

    def test_synthetic_rgb_depth_ir_pipeline_recovers_all_seven(self):
        source = SyntheticCameraSource(self.config)
        pipeline = SevenMarkerFusionPipeline(self.config)
        source.start()
        frame = source.poll()
        result = pipeline.process(frame)
        source.stop()
        self.assertEqual(np.count_nonzero(result.measured_valid), 7)
        self.assertTrue(all(source == "rgb_depth+ir_stereo_ci" for source in result.sources))
        self.assertTrue(np.isfinite(result.base_m).all())
        self.assertLess(float(np.max(np.linalg.norm(
            result.rgb_depth_camera_m - result.ir_stereo_camera_m, axis=1
        ))), 0.003)

    def test_yolo_pose_fails_closed_without_project_specific_weights(self):
        detector = TdcrYoloPoseDetector({
            "tdcr_yolo_pose_enabled": True,
            "tdcr_yolo_pose_model": "Visual_information/models/tdcr_yolo_pose/definitely_missing.pt",
        })
        result = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertFalse(result.accepted)
        self.assertFalse(np.isfinite(result.pixels).any())
        self.assertIn("MODEL MISSING", result.reason)

    def test_body_first_detector_stays_on_real_catheter_not_drive_mechanism(self):
        asset_root = Path(__file__).resolve().parent / "assets"
        lock_boxes = {
            "body_first_real_000.png": (0.45, 0.18, 0.70, 0.48),
            "body_first_real_755.png": (0.055, 0.40, 0.24, 0.70),
        }
        for name, target_lock in lock_boxes.items():
            image = cv2.imread(str(asset_root / name))
            self.assertIsNotNone(image, name)
            detector = BodyFirstBandDetector(self.config["tracking"])
            detector.set_target_lock(target_lock)
            result = detector.detect(image, (0.0, 0.0, 1.0, 1.0))
            self.assertTrue(result.accepted, f"{name}: {result.reason}")
            self.assertTrue(result.segmentation_backend.startswith("sam2.1-video"))
            self.assertGreaterEqual(result.segmentation_confidence, 0.50)
            self.assertTrue(np.all(np.diff(result.pixels[:, 0]) < 0.0))
            # K0 is the proximal/right marker and K6 the distal/left marker.
            self.assertGreater(result.pixels[0, 0], result.pixels[-1, 0])
            for pixel in result.pixels:
                u, v = np.rint(pixel).astype(int)
                self.assertNotEqual(int(result.body_mask[v, u]), 0)
            # The crop contains the drive guide on the far right.  No marker is
            # allowed to jump onto that mechanism.
            self.assertLess(float(np.max(result.pixels[:, 0])), 0.72 * image.shape[1])
            mask_y, mask_x = np.nonzero(result.body_mask)
            self.assertGreater(len(mask_x), 300)
            self.assertLess(float(np.max(mask_x)), 0.74 * image.shape[1])

    def test_body_first_detector_rejects_plain_drive_rail_without_seven_bands(self):
        image = np.full((300, 1100, 3), (72, 92, 66), dtype=np.uint8)
        cv2.line(image, (80, 150), (1050, 150), (25, 28, 25), 14)
        # Three metal highlights are not a seven-marker observation.
        for x in (730, 850, 970):
            cv2.rectangle(image, (x - 5, 135), (x + 5, 165), (210, 210, 210), -1)
        detector = BodyFirstBandDetector(self.config["tracking"])
        detector.set_target_lock((0.55, 0.35, 0.99, 0.65))
        result = detector.detect(image, (0.0, 0.0, 1.0, 1.0))
        self.assertFalse(result.accepted)
        self.assertFalse(np.isfinite(result.pixels).any())

    def test_real_detector_requires_explicit_target_identity_lock(self):
        image = cv2.imread(str(Path(__file__).resolve().parent / "assets" / "body_first_real_000.png"))
        self.assertIsNotNone(image)
        detector = BodyFirstBandDetector(self.config["tracking"])
        result = detector.detect(image, (0.0, 0.0, 1.0, 1.0))
        self.assertFalse(result.accepted)
        self.assertIn("TARGET UNLOCKED", result.reason)
        self.assertFalse(np.isfinite(result.pixels).any())

    def test_tracking_box_normalises_reduced_mask_in_its_own_coordinate_system(self):
        segmenter = NeuralContinuumSegmenter(self.config["tracking"])
        segmenter.set_target_lock((0.2, 0.2, 0.4, 0.4))
        segmenter._last_mask = np.zeros((100, 200), dtype=np.uint8)
        segmenter._last_mask[30:50, 80:120] = 255
        # The overlay is full HD, but the mask was produced at reduced size.
        roi = segmenter.tracking_roi_normalized((1080, 1920), (0, 0, 1, 1))
        center_x = 0.5 * (roi[0] + roi[2])
        center_y = 0.5 * (roi[1] + roi[3])
        self.assertAlmostEqual(center_x, 0.5, delta=0.02)
        self.assertAlmostEqual(center_y, 0.4, delta=0.03)

    def test_pipeline_does_not_turn_drive_rail_highlights_into_rgb_keypoints(self):
        source = SyntheticCameraSource(self.config)
        source.start()
        frame = source.poll()
        source.stop()
        image = np.full_like(frame.color_bgr, (72, 92, 66))
        height, width = image.shape[:2]
        y = height // 2
        cv2.line(image, (width // 20, y), (width - 20, y), (25, 28, 25), 14)
        for x in (int(width * 0.68), int(width * 0.80), int(width * 0.92)):
            cv2.rectangle(image, (x - 5, y - 15), (x + 5, y + 15), (210, 210, 210), -1)
        frame = replace(
            frame,
            color_bgr=image,
            metadata={"device": {"serial": "REAL-D435"}},
        )
        result = SevenMarkerFusionPipeline(self.config).process(frame)
        self.assertFalse(np.isfinite(result.color_pixels).any())
        self.assertEqual(int(np.count_nonzero(result.measured_valid)), 0)

    def test_covariance_intersection_does_not_claim_independent_sensor_variance(self):
        first = ModalityMeasurement(
            np.asarray([0.0, 0.0, 0.300]), np.eye(3) * 1e-6, 0.8, True, "depth"
        )
        second = ModalityMeasurement(
            np.asarray([0.0, 0.0, 0.302]), np.eye(3) * 1e-6, 0.8, True, "stereo"
        )
        fused = covariance_intersection(first, second)
        self.assertTrue(fused.valid)
        self.assertGreaterEqual(float(np.min(np.diag(fused.covariance_m2))), 0.99e-6)
        self.assertGreaterEqual(fused.point_camera_m[2], 0.300)
        self.assertLessEqual(fused.point_camera_m[2], 0.302)

    def test_shape_factor_reduces_an_isolated_three_dimensional_outlier(self):
        reference = np.column_stack((
            np.arange(7) * 0.007,
            np.zeros(7),
            np.full(7, 0.30),
        ))
        corrupted = reference.copy()
        corrupted[3] += np.asarray([0.025, 0.020, 0.0])
        covariance = np.tile(np.eye(3) * 1e-6, (7, 1, 1))
        smoother = ShapeConstrainedSmoother(self.config)
        smoothed, _covariance, _predicted, _confidence = smoother.update(
            corrupted, covariance, np.ones(7, dtype=bool), np.ones(7), 1_000_000_000
        )
        before = float(np.linalg.norm(corrupted[3] - reference[3]))
        after = float(np.linalg.norm(smoothed[3] - reference[3]))
        self.assertLess(after, before * 0.35)

    def test_saved_real_rgb_frame_repairs_tip_glare_and_base_reflection(self):
        image = cv2.imread(str(Path(__file__).resolve().parent / "fixtures" / "live_tracking_raw.png"))
        self.assertIsNotNone(image)
        processing_width = 416
        scale = processing_width / image.shape[1]
        reduced = cv2.resize(
            image, (processing_width, int(round(image.shape[0] * scale))), cv2.INTER_AREA
        )
        hsv = cv2.cvtColor(reduced, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(reduced, cv2.COLOR_BGR2LAB)
        tracker = MarkerTracker(self.config)
        pixels, confidence, _masks, _colours, _segmentation = detect_color_bands(
            hsv,
            lab,
            self.config["markers"],
            tracker.color_prototypes,
            scaled_tracking_config(self.config["tracking"], scale),
            np.full((7, 2), np.nan),
            (0.0, 0.0, 1.0, 1.0),
        )
        pixels /= scale
        repaired = repair_ordered_ring_sequence(image, pixels, confidence)
        self.assertTrue(repaired.accepted)
        self.assertEqual(np.count_nonzero(np.isfinite(repaired.pixels).all(axis=1)), 7)
        gaps = np.linalg.norm(np.diff(repaired.pixels[::-1], axis=0), axis=1)
        self.assertLess(float(np.max(gaps)), 1.5 * float(np.median(gaps)))
        self.assertLess(float(np.max(repaired.pixels[:, 0])), 760.0)

    def test_existing_keypoint_archive_can_be_refined_without_images(self):
        points = np.tile(
            np.column_stack((np.arange(7) * 0.007, np.zeros(7), np.full(7, 0.30))),
            (4, 1, 1),
        )
        points[2, 4] = np.nan
        valid = np.isfinite(points).all(axis=2)
        covariance = np.tile(np.eye(3) * 1e-6, (4, 7, 1, 1))
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "keypoints.npz"
            target = Path(temporary) / "result"
            np.savez_compressed(
                source,
                keypoints_raw_camera_m=points,
                keypoint_valid=valid,
                keypoint_confidence=valid.astype(float),
                keypoint_covariance_m2=covariance,
                capture_host_ns=np.arange(4, dtype=np.int64) * 33_333_333,
            )
            refine_existing_npz(source, target, self.config)
            self.assertTrue((target / "seven_marker_3d.csv").exists())
            with np.load(target / "seven_marker_3d.npz") as archive:
                self.assertEqual(archive["smoothed_camera_m"].shape, (4, 7, 3))
                self.assertTrue(archive["predicted"][2, 4])
                self.assertTrue(np.isfinite(archive["smoothed_camera_m"][2, 4]).all())

    def test_legacy_turbo_depth_palette_is_inverted_to_metric_depth(self):
        decoder = TurboDepthDecoder()
        palette_index = 128
        visual = decoder.palette[palette_index].astype(np.uint8).reshape(1, 1, 3)
        depth, confidence = decoder.decode(visual, 0.18, 0.60)
        expected = 0.18 + (255 - palette_index) * (0.60 - 0.18) / 255.0
        self.assertAlmostEqual(float(depth[0, 0]), expected, places=3)
        self.assertGreater(float(confidence[0, 0]), 0.9)

    def test_neutral_tendon_controls_create_a_straight_7mm_chain(self):
        points = motor_controls_to_local_points(np.zeros(7))
        np.testing.assert_allclose(points[:, 0], np.arange(7) * 0.007, atol=1e-10)
        np.testing.assert_allclose(points[:, 1:], 0.0, atol=1e-10)

    def test_motor_prior_fills_occluded_points_without_relabeling_measurements(self):
        local = motor_controls_to_local_points(np.zeros(7))
        target = local + np.asarray([0.10, 0.02, 0.30])
        measured = np.zeros(7, dtype=bool)
        measured[4:] = True
        sparse = np.full((7, 3), np.nan)
        sparse[measured] = target[measured]
        covariance = np.tile(np.eye(3) * 1e-6, (7, 1, 1))
        result = SevenMarkerResult(
            sequence=0,
            capture_host_ns=0,
            raw_camera_m=sparse,
            fused_camera_m=sparse,
            smoothed_camera_m=sparse,
            base_m=sparse,
            covariance_m2=covariance,
            measured_valid=measured,
            predicted=np.zeros(7, dtype=bool),
            confidence=measured.astype(float),
            color_pixels=np.full((7, 2), np.nan),
            left_pixels=np.full((7, 2), np.nan),
            right_pixels=np.full((7, 2), np.nan),
            sources=tuple("rgb_aligned_depth" if value else "invalid" for value in measured),
            rgb_depth_camera_m=sparse,
            ir_stereo_camera_m=np.full((7, 3), np.nan),
            processing_ms=0.0,
        )
        axis = {f"control_{index}_m": 0.0 for index in range(7)}
        fused = MotorShapePriorFusion(True).fuse(result, axis, np.eye(4))
        self.assertEqual(np.count_nonzero(fused.measured_valid), 3)
        self.assertEqual(np.count_nonzero(fused.predicted), 4)
        self.assertTrue(np.isfinite(fused.smoothed_camera_m).all())
        self.assertTrue(all(fused.sources[index] == "motor_pcc_prior" for index in range(4)))
        self.assertTrue(all(fused.sources[index] == "rgb_aligned_depth" for index in range(4, 7)))

    def test_curve_model_reconstructs_a_smooth_42mm_centerline_and_tube(self):
        angle = np.linspace(0.0, np.pi / 2.0, 7)
        radius = 0.042 / (np.pi / 2.0)
        points = np.column_stack((
            radius * np.sin(angle),
            radius * (1.0 - np.cos(angle)),
            np.full(7, 0.30),
        ))
        shape = fit_continuum_shape(points, samples=85)
        self.assertEqual(shape.centerline_m.shape, (85, 3))
        self.assertAlmostEqual(shape.total_length_m, 0.042, delta=0.0015)
        self.assertTrue(np.isfinite(shape.curvature_1_m).all())
        vertices, faces, colors = tube_mesh(shape.centerline_m)
        self.assertGreater(len(vertices), 500)
        self.assertGreater(len(faces), 800)
        self.assertEqual(len(colors), len(vertices))

    def test_two_section_projection_removes_alternating_depth_snake(self):
        # Captured failure pattern: each local chord looks plausible, but the
        # depth samples alternate between two surfaces and form a snake.
        points_mm = np.asarray(
            [
                [0.00, 0.00, 0.00],
                [-5.84, -0.37, 0.00],
                [-10.84, 2.13, 0.00],
                [-15.34, -1.31, 0.00],
                [-20.84, -2.94, 0.00],
                [-24.45, 1.63, 0.00],
                [-28.19, -2.91, 0.00],
            ]
        )
        projected = project_material_keypoints(
            points_mm / 1000.0,
            np.full(7, 0.25),
            np.zeros(7, dtype=bool),
        )
        chords = np.diff(projected, axis=0)
        unit = chords / np.linalg.norm(chords, axis=1, keepdims=True)
        turn_deg = np.degrees(
            np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0))
        )
        second_difference_mm = 1000.0 * np.linalg.norm(
            np.diff(projected, n=2, axis=0), axis=1
        )
        self.assertLess(float(np.max(turn_deg)), 25.0)
        self.assertLess(float(np.max(second_difference_mm)), 2.5)
        self.assertGreater(float(np.min(np.linalg.norm(chords, axis=1))), 0.0053)
        self.assertLess(float(np.max(np.linalg.norm(chords, axis=1))), 0.00701)

    def test_local_depth_uses_neighbourhood_when_marker_center_is_a_hole(self):
        depth = np.full((61, 61), 0.302, dtype=float)
        yy, xx = np.mgrid[:61, :61]
        depth[(xx - 30) ** 2 + (yy - 30) ** 2 < 5 ** 2] = 0.0
        fusion = LocalDepthCurveFusion(True, search_radius_px=16, depth_gate_m=0.015)
        value, confidence = fusion._local_depth(depth, np.asarray([30.0, 30.0]), 0.300)
        self.assertAlmostEqual(value, 0.302, places=4)
        self.assertGreater(confidence, 0.35)

    def test_offline_refinement_adds_zero_phase_trajectory_and_dense_curve(self):
        frames = 31
        base = np.column_stack((
            np.arange(7) * 0.007,
            np.zeros(7),
            np.full(7, 0.30),
        ))
        rng = np.random.default_rng(8)
        points = np.tile(base, (frames, 1, 1))
        points[:, :, 1] += np.sin(np.linspace(0, 2 * np.pi, frames))[:, None] * 0.002
        points += rng.normal(0.0, 0.0007, points.shape)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            np.savez_compressed(
                target / "seven_marker_3d.npz",
                sequence=np.arange(frames),
                capture_host_ns=np.arange(frames, dtype=np.int64) * 33_333_333,
                smoothed_camera_m=points,
                measured_valid=np.ones((frames, 7), dtype=bool),
                predicted=np.zeros((frames, 7), dtype=bool),
                confidence=np.ones((frames, 7), dtype=float),
            )
            (target / "summary.json").write_text("{}", encoding="utf-8")
            metrics = refine_output_trajectory(target, temporal_strength=6.0, curve_samples=85)
            self.assertTrue(metrics["offline_refined"])
            self.assertTrue((target / "seven_marker_refined.csv").exists())
            with np.load(target / "seven_marker_3d.npz") as archive:
                refined = archive["offline_refined_camera_m"]
                self.assertEqual(archive["curve_centerline_camera_m"].shape, (frames, 85, 3))
                raw_jitter = np.std(np.diff(points[:, 3, 2]))
                refined_jitter = np.std(np.diff(refined[:, 3, 2]))
                self.assertLess(refined_jitter, raw_jitter * 0.65)

    def test_em_pair_extracts_coordinate_invariant_chord_and_relative_angle(self):
        first = np.eye(4)
        second = np.eye(4)
        angle = np.radians(30.0)
        second[:3, :3] = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        payload = [
            {"port_handle": 10, "valid": True, "translation_mm": [0, 0, 0], "transform_row_major": first.ravel().tolist()},
            {"port_handle": 11, "valid": True, "translation_mm": [0, 30, 40], "transform_row_major": second.ravel().tolist()},
        ]
        observation = extract_em_pair({"all_tools_json": json.dumps(payload)}, 10, 11)
        self.assertTrue(observation.valid)
        self.assertAlmostEqual(observation.chord_m, 0.05, places=6)
        self.assertAlmostEqual(np.degrees(observation.relative_angle_rad), 30.0, places=5)

    def test_periodic_band_fallback_recovers_seven_washed_out_rings(self):
        image = np.full((1080, 1920, 3), (72, 118, 92), dtype=np.uint8)
        cv2.rectangle(image, (820, 530), (1300, 558), (22, 25, 24), -1)
        positions = np.arange(890, 1135, 40)
        colours = [(210, 210, 245), (190, 215, 245), (150, 230, 235),
                   (180, 225, 190), (225, 225, 205), (235, 210, 205), (225, 195, 235)]
        for x, colour in zip(positions, colours):
            cv2.rectangle(image, (int(x - 5), 526), (int(x + 5), 562), colour, -1)
        result = extract_periodic_bands(image, (0.40, 0.45, 0.70, 0.60))
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(result.pixels.shape, (7, 2))
        gaps = np.linalg.norm(np.diff(result.pixels, axis=0), axis=1)
        self.assertLess(float(np.std(gaps)), 3.0)
        # The dark shaft continues to the right, so the rightmost ring is K0.
        self.assertGreater(result.pixels[0, 0], result.pixels[-1, 0])


if __name__ == "__main__":
    unittest.main()
