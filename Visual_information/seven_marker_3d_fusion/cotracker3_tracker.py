"""Annotation-free seven-point tracking with Meta CoTracker3.

The seven semantic identities are supplied once on the conditioning frame.
CoTracker3 then follows those material points jointly.  Long recordings are
processed in overlapping windows so a 1080p session never has to reside in
GPU memory in its entirety.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from Visual_information.d435_tdcr_capture.config import PROJECT_ROOT, VISUAL_ROOT
from Visual_information.d435_tdcr_capture.models import KEYPOINT_COUNT


DEFAULT_CHECKPOINT = VISUAL_ROOT / "models" / "cotracker3" / "scaled_offline.pth"


@dataclass(frozen=True)
class CoTrackerSequenceResult:
    pixels: np.ndarray
    confidence: np.ndarray
    visible: np.ndarray
    backend: str


def _validate_points(points: np.ndarray) -> np.ndarray:
    result = np.asarray(points, dtype=np.float32).reshape(KEYPOINT_COUNT, 2)
    if not np.isfinite(result).all():
        raise ValueError("K0-K6 initialization contains invalid coordinates")
    return result


class CoTracker3SequenceTracker:
    """Run pretrained CoTracker3 on a bounded RGB video range."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        configured = str(cfg.get("cotracker3_checkpoint", "")).strip()
        path = Path(configured) if configured else DEFAULT_CHECKPOINT
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        self.checkpoint = path.resolve()
        self.device_requested = str(cfg.get("cotracker3_device", "cuda")).lower()
        self.processing_width = max(512, int(cfg.get("cotracker3_processing_width", 960)))
        # CoTracker3's released offline model has a fixed 60-frame temporal
        # embedding.  Short clips are padded by repeating the final frame.
        self.window_frames = 60
        self.overlap_frames = int(np.clip(
            cfg.get("cotracker3_overlap_frames", 16), 4, self.window_frames // 2
        ))
        self._model = None
        self._torch = None
        self.device = "cpu"

    @property
    def ready(self) -> bool:
        return self.checkpoint.is_file() and self.checkpoint.stat().st_size > 20_000_000

    @property
    def status(self) -> str:
        if not self.ready:
            return f"CoTracker3 权重缺失: {self.checkpoint}"
        return f"CoTracker3 {self.device if self._model is not None else '待加载'}"

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.ready:
            raise FileNotFoundError(
                f"CoTracker3 checkpoint is missing or incomplete: {self.checkpoint}"
            )
        import torch
        from cotracker.predictor import CoTrackerPredictor

        self.device = (
            "cuda"
            if self.device_requested == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self._model = CoTrackerPredictor(
            checkpoint=str(self.checkpoint), offline=True, window_len=60
        ).to(self.device).eval()
        self._torch = torch

    @staticmethod
    def _read_window(
        capture: cv2.VideoCapture,
        first: int,
        last: int,
        processing_width: int,
    ) -> tuple[list[np.ndarray], float, float]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(first))
        frames: list[np.ndarray] = []
        scale_x = scale_y = 1.0
        for _index in range(first, last):
            ok, bgr = capture.read()
            if not ok:
                break
            height, width = bgr.shape[:2]
            scale = min(1.0, processing_width / max(width, 1))
            if scale < 0.999:
                resized = cv2.resize(
                    bgr,
                    (int(round(width * scale)), int(round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                resized = bgr
            scale_x = resized.shape[1] / width
            scale_y = resized.shape[0] / height
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        return frames, float(scale_x), float(scale_y)

    def track_video(
        self,
        video_path: str | Path,
        first_frame: int,
        final_frame: int,
        initial_pixels: np.ndarray,
        progress: Callable[[int, int], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> CoTrackerSequenceResult:
        """Track K0-K6 on ``[first_frame, final_frame)``.

        Output coordinates are always in the native RGB resolution.  Window
        boundaries share an overlap and the next query is taken from an
        interior prediction rather than extrapolated from the last frame.
        """

        self._load()
        initial = _validate_points(initial_pixels)
        capture = cv2.VideoCapture(str(Path(video_path).expanduser().resolve()))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open RGB video: {video_path}")
        total = max(0, int(final_frame) - int(first_frame))
        pixels = np.full((total, KEYPOINT_COUNT, 2), np.nan, dtype=np.float32)
        confidence = np.zeros((total, KEYPOINT_COUNT), dtype=np.float32)
        visible = np.zeros((total, KEYPOINT_COUNT), dtype=bool)
        cursor = int(first_frame)
        seed = initial.copy()
        try:
            while cursor < int(final_frame):
                if stop_requested is not None and stop_requested():
                    raise InterruptedError("CoTracker3 preprocessing stopped by user")
                window_end = min(int(final_frame), cursor + self.window_frames)
                frames, scale_x, scale_y = self._read_window(
                    capture, cursor, window_end, self.processing_width
                )
                if len(frames) < 2:
                    break
                actual_count = len(frames)
                if actual_count < self.window_frames:
                    frames.extend([frames[-1]] * (self.window_frames - actual_count))
                scaled_seed = seed * np.asarray([scale_x, scale_y], dtype=np.float32)
                video = self._torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
                video = video.unsqueeze(0).float().to(self.device, non_blocking=True)
                queries = np.column_stack((
                    np.zeros(KEYPOINT_COUNT, dtype=np.float32), scaled_seed
                ))[None]
                queries = self._torch.from_numpy(queries).to(self.device)
                with self._torch.inference_mode():
                    tracks_t, visible_t = self._model(
                        video, queries=queries, backward_tracking=False
                    )
                tracks = tracks_t[0].float().cpu().numpy()
                vis = visible_t[0].float().cpu().numpy()
                tracks /= np.asarray([scale_x, scale_y], dtype=np.float32)
                # The predictor exposes thresholded visibility.  Use 0.96 for
                # visible learned tracks and zero for occluded points; later
                # RGB/SAM/depth gates provide the fine-grained confidence.
                vis_bool = vis.astype(bool)
                tracks = tracks[:actual_count]
                vis_bool = vis_bool[:actual_count]
                is_last = window_end >= int(final_frame)
                commit = actual_count if is_last else actual_count - self.overlap_frames
                commit = max(1, commit)
                out0 = cursor - int(first_frame)
                out1 = min(total, out0 + commit)
                count = out1 - out0
                pixels[out0:out1] = tracks[:count]
                visible[out0:out1] = vis_bool[:count]
                confidence[out0:out1] = np.where(vis_bool[:count], 0.96, 0.0)
                if is_last:
                    cursor = int(final_frame)
                else:
                    # The next chunk begins at this committed boundary.
                    seed_index = min(commit, len(tracks) - 1)
                    next_seed = tracks[seed_index]
                    next_vis = vis_bool[seed_index]
                    seed = np.where(next_vis[:, None], next_seed, tracks[max(0, seed_index - 1)])
                    cursor += commit
                if progress is not None:
                    progress(min(total, cursor - int(first_frame)), total)
                del video, queries, tracks_t, visible_t
                if self.device == "cuda":
                    self._torch.cuda.empty_cache()
        finally:
            capture.release()
        if not np.isfinite(pixels[0]).all():
            pixels[0] = initial
            visible[0] = True
            confidence[0] = 1.0
        return CoTrackerSequenceResult(
            pixels=pixels,
            confidence=confidence,
            visible=visible,
            backend=f"cotracker3-offline-{self.device}",
        )


class TrackedPointRefiner:
    """Fuse CoTracker, SAM2 body motion and ordered colour-band evidence.

    CoTracker is intentionally only a *proposal* here.  In the lung phantom a
    query can lock onto a stationary transparent-wall highlight while still
    reporting high visibility.  The material state is therefore propagated
    from the previous accepted K0--K6 chain, moved with the SAM2 mask, and only
    then locally re-identified from colour.  This makes translation accumulate
    frame by frame instead of repeatedly searching around the (possibly stale)
    CoTracker position.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        self.radius = max(8, int(cfg.get("cotracker3_colour_search_radius_px", 22)))
        self.maximum_delta = float(cfg.get("cotracker3_colour_maximum_lab_delta", 52.0))
        self.maximum_step_scale = float(cfg.get("cotracker3_maximum_gap_scale", 2.8))
        self.foreground_fallback_enabled = bool(
            cfg.get("material_chain_foreground_fallback_enabled", False)
        )
        self.templates = np.full((KEYPOINT_COUNT, 3), np.nan, dtype=np.float32)
        self.last_points = np.full((KEYPOINT_COUNT, 2), np.nan, dtype=np.float32)
        self.velocity = np.zeros((KEYPOINT_COUNT, 2), dtype=np.float32)
        self.reference_gaps = np.full(KEYPOINT_COUNT - 1, np.nan, dtype=np.float32)
        self.last_body_mask: np.ndarray | None = None
        self.last_motion_mask: np.ndarray | None = None
        self.reference_bgr: np.ndarray | None = None
        self.tracking_source = "cotracker3"
        self.frames_since_initialisation = 0

    def reset(self) -> None:
        self.templates[:] = np.nan
        self.last_points[:] = np.nan
        self.velocity[:] = 0.0
        self.reference_gaps[:] = np.nan
        self.last_body_mask = None
        self.last_motion_mask = None
        self.reference_bgr = None
        self.tracking_source = "cotracker3"
        self.frames_since_initialisation = 0

    @staticmethod
    def _median_patch(lab: np.ndarray, point: np.ndarray, radius: int = 2) -> np.ndarray:
        height, width = lab.shape[:2]
        x, y = np.rint(point).astype(int)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return np.full(3, np.nan, dtype=np.float32)
        return np.median(lab[y0:y1, x0:x1].reshape(-1, 3), axis=0)

    def initialise(self, image_bgr: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(np.asarray(image_bgr, np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        points = _validate_points(pixels)
        raw_gap = float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        body = self._material_centerline_mask(
            image_bgr, points, max(raw_gap, 4.0), None, corridor_scale=1.8
        )
        if body is not None:
            centred = self._snap_to_mask_medial_axis(
                points, body, max(7, int(round(0.85 * raw_gap))), strength=1.0
            )
            # Manual clicks define identities and approximate phase only.  The
            # accepted initialization is an equal-material-interval smooth
            # curve on the extracted black backbone.
            points = self._regularise_material_chain(
                centred, points, max(raw_gap, 4.0)
            )
        for index in range(KEYPOINT_COUNT):
            self.templates[index] = self._median_patch(lab, points[index], 2)
        self.last_points[:] = points
        self.velocity[:] = 0.0
        # The rings are installed every 7 mm.  A single robust projected pitch
        # prevents click placement error from becoming a permanent material
        # distortion in all later frames.
        projected_gap = float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        self.reference_gaps[:] = projected_gap
        self.last_body_mask = None
        self.last_motion_mask = None
        self.reference_bgr = np.asarray(image_bgr, np.uint8).copy()
        self.frames_since_initialisation = 0
        return points.copy()

    def predict(
        self,
        pixels: np.ndarray,
        confidence: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the material-chain prediction used to prompt SAM2.

        A learned proposal is blended only while it remains close to the
        propagated material point.  A stationary high-confidence proposal can
        no longer pull a moving chain back to its conditioning-frame pixels.
        """

        learned = np.asarray(pixels, dtype=np.float32).reshape(KEYPOINT_COUNT, 2)
        if not np.isfinite(self.last_points).all():
            return learned.copy()
        predicted = self.last_points + self.velocity
        gap = float(np.nanmedian(self.reference_gaps))
        if not np.isfinite(gap):
            gap = float(np.nanmedian(np.linalg.norm(np.diff(self.last_points, axis=0), axis=1)))
        gate = max(8.0, 0.65 * gap)
        learned_valid = np.isfinite(learned).all(axis=1)
        if confidence is not None:
            learned_valid &= np.asarray(confidence, dtype=float).reshape(KEYPOINT_COUNT) > 0.05
        residual = np.linalg.norm(learned - predicted, axis=1)
        compatible = learned_valid & (residual <= gate)
        predicted[compatible] = 0.88 * predicted[compatible] + 0.12 * learned[compatible]
        return predicted

    @staticmethod
    def _mask_centroid(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None or not np.any(mask):
            return None
        moments = cv2.moments(np.asarray(mask != 0, dtype=np.uint8))
        if moments["m00"] <= 1.0:
            return None
        return np.asarray(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float32,
        )

    def _body_motion(self, body_mask: np.ndarray | None, gap: float) -> np.ndarray:
        current = self._mask_centroid(body_mask)
        previous = self._mask_centroid(self.last_body_mask)
        if current is None or previous is None:
            return np.zeros(2, dtype=np.float32)
        displacement = current - previous
        maximum = max(5.0, 0.48 * gap)
        length = float(np.linalg.norm(displacement))
        if length > maximum:
            displacement *= maximum / max(length, 1e-6)
        return displacement

    def _foreground_motion_mask(
        self,
        image_bgr: np.ndarray,
        points: np.ndarray,
        gap: float,
    ) -> np.ndarray | None:
        """Fixed-camera fallback when SAM2 is unavailable or temporarily lost."""

        if self.reference_bgr is None or self.reference_bgr.shape != image_bgr.shape:
            return None
        reference = cv2.GaussianBlur(self.reference_bgr, (5, 5), 0)
        current = cv2.GaussianBlur(np.asarray(image_bgr, np.uint8), (5, 5), 0)
        difference = cv2.cvtColor(cv2.absdiff(current, reference), cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(np.asarray(image_bgr, np.uint8), cv2.COLOR_BGR2HSV)
        darkness = (255.0 - hsv[..., 2].astype(np.float32)) * np.clip(
            (175.0 - hsv[..., 1].astype(np.float32)) / 120.0, 0.20, 1.0
        )
        mask = ((difference >= 9) & (darkness >= 24.0)).astype(np.uint8) * 255
        corridor = np.zeros(mask.shape, dtype=np.uint8)
        path = np.rint(points).astype(np.int32)
        cv2.polylines(
            corridor,
            [path],
            False,
            255,
            max(19, int(round(1.8 * gap)) | 1),
            cv2.LINE_8,
        )
        # A per-frame material displacement is small, so the moving corridor
        # can safely be dilated without admitting the distant guide rails.
        corridor = cv2.dilate(
            corridor,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (max(9, int(round(gap))) | 1, max(9, int(round(gap))) | 1),
            ),
        )
        mask = cv2.bitwise_and(mask, corridor)
        close_width = max(7, int(round(0.85 * gap)) | 1)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_width, 7)),
        )
        return mask if np.count_nonzero(mask) >= 18 else None

    @staticmethod
    def _material_centerline_mask(
        image_bgr: np.ndarray,
        points: np.ndarray,
        gap: float,
        neural_mask: np.ndarray | None = None,
        corridor_scale: float = 1.0,
    ) -> np.ndarray | None:
        """Extract only the dark continuum backbone near the material chain.

        SAM2 supplies object identity, but transparent lung reflections can
        occasionally make its mask wider than the black continuum.  This mask
        is therefore rebuilt from low-light/low-chroma pixels inside a narrow
        temporal corridor, with gaps bridged only along the chain tangent.
        The returned component must span most of the six material intervals.
        """

        full_image = np.asarray(image_bgr, np.uint8)
        full_height, full_width = full_image.shape[:2]
        chain = np.asarray(points, dtype=np.float32).reshape(KEYPOINT_COUNT, 2)
        if not np.isfinite(chain).all():
            return None
        corridor_scale = float(np.clip(corridor_scale, 0.8, 3.5))
        corridor_width = max(15, int(round(1.12 * gap * corridor_scale)) | 1)
        margin = corridor_width + max(8, int(round(0.45 * gap)))
        x0 = max(0, int(np.floor(np.min(chain[:, 0]))) - margin)
        y0 = max(0, int(np.floor(np.min(chain[:, 1]))) - margin)
        x1 = min(full_width, int(np.ceil(np.max(chain[:, 0]))) + margin + 1)
        y1 = min(full_height, int(np.ceil(np.max(chain[:, 1]))) + margin + 1)
        if x1 - x0 < 12 or y1 - y0 < 12:
            return None
        image = full_image[y0:y1, x0:x1]
        chain = chain - np.asarray([x0, y0], dtype=np.float32)
        height, width = image.shape[:2]
        corridor = np.zeros((height, width), dtype=np.uint8)
        path = np.rint(chain).astype(np.int32)
        cv2.polylines(corridor, [path], False, 255, corridor_width, cv2.LINE_AA)
        for endpoint in (path[0], path[-1]):
            cv2.circle(corridor, tuple(endpoint), corridor_width // 2, 255, -1)
        if neural_mask is not None and np.asarray(neural_mask).shape == (full_height, full_width):
            neural_support = cv2.dilate(
                np.asarray(neural_mask[y0:y1, x0:x1] != 0, np.uint8) * 255,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (max(9, int(round(0.55 * gap))) | 1,) * 2,
                ),
            )
            # Do not trust the neural mask as a hard boundary: retain the
            # temporal corridor when SAM loses a thin or strongly curved tip.
            corridor = cv2.bitwise_and(
                corridor,
                cv2.bitwise_or(neural_support, cv2.erode(corridor, np.ones((3, 3), np.uint8))),
            )
        lab_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
        values = lab_l[corridor != 0]
        if values.size < 30:
            return None
        # A broad re-acquisition corridor may contain much more green table or
        # transparent phantom than catheter.  A percentile-only threshold then
        # classifies the whole corridor as "dark" and its medial axis stays at
        # the manual click offset.  Otsu supplies the local foreground/background
        # valley; the percentile term remains as a conservative upper bound for
        # low-contrast frames.
        otsu_threshold, _ = cv2.threshold(
            np.rint(values).astype(np.uint8), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        percentile_threshold = float(np.percentile(values, 34.0) + 18.0)
        threshold = float(np.clip(
            min(percentile_threshold, float(otsu_threshold) + 12.0),
            42.0, 122.0,
        ))
        # Absolute darkness is insufficient through a transparent airway: the
        # green table can be darker than a colour ring.  A black catheter is a
        # narrow dark ridge relative to pixels immediately across its axis.
        # Morphological closing estimates that local background and suppresses
        # broad low-frequency table/phantom shading.
        background_width = max(11, int(round(0.62 * gap)) | 1)
        local_background = cv2.morphologyEx(
            np.rint(lab_l).astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (background_width, background_width)
            ),
        ).astype(np.float32)
        ridge = np.maximum(local_background - lab_l, 0.0)
        ridge_values = ridge[corridor != 0]
        ridge_otsu, _ = cv2.threshold(
            np.rint(np.clip(ridge_values, 0.0, 255.0)).astype(np.uint8),
            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        ridge_threshold = float(np.clip(0.72 * float(ridge_otsu), 4.0, 24.0))
        dark = (
            (corridor != 0)
            & (lab_l <= threshold)
            & (ridge >= ridge_threshold)
            & ((saturation <= 125.0) | (lab_l <= 58.0))
        ).astype(np.uint8) * 255
        # Bridge the narrow coloured rings along the catheter direction without
        # expanding into the orthogonal transparent airway wall.
        tangent = chain[-1] - chain[0]
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length <= 1e-4:
            return None
        tangent /= tangent_length
        half = max(4, int(round(0.34 * gap)))
        kernel_size = 2 * half + 3
        centre = kernel_size // 2
        bridge = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        offset = np.rint(tangent * half).astype(int)
        cv2.line(
            bridge,
            tuple((np.asarray([centre, centre]) - offset).astype(int)),
            tuple((np.asarray([centre, centre]) + offset).astype(int)),
            1,
            3,
            cv2.LINE_8,
        )
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, bridge)
        # Once the distal section bends, its local tangent can differ greatly
        # from the endpoint chord.  A second compact isotropic closure joins
        # neighbouring coloured intervals around that bend; its diameter stays
        # below one material pitch, so it cannot shortcut a genuine branch.
        curved_bridge = max(7, int(round(0.86 * gap)) | 1)
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (curved_bridge, curved_bridge)
            ),
        )
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
        best_label = 0
        best_score = -np.inf
        minimum_span = 2.7 * gap
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(18, int(0.30 * gap * gap)):
                continue
            yy, xx = np.nonzero(labels == label)
            projection = xx.astype(np.float32) * tangent[0] + yy.astype(np.float32) * tangent[1]
            span = float(np.percentile(projection, 98) - np.percentile(projection, 2))
            if span < minimum_span:
                continue
            hit_radius2 = float(0.48 * gap) ** 2
            hits = sum(
                int(np.min((xx - point[0]) ** 2 + (yy - point[1]) ** 2) <= hit_radius2)
                for point in chain
            )
            score = 5.0 * hits + span / max(gap, 1.0) + 0.002 * area
            if score > best_score:
                best_label, best_score = label, score
        if best_label == 0:
            return None
        selected = np.where(labels == best_label, 255, 0).astype(np.uint8)
        if np.count_nonzero(selected) < 18:
            return None
        output = np.zeros((full_height, full_width), dtype=np.uint8)
        output[y0:y1, x0:x1] = selected
        return output

    def _joint_band_shift(
        self,
        lab: np.ndarray,
        points: np.ndarray,
        gap: float,
    ) -> tuple[np.ndarray, float]:
        """Jointly align all seven colour templates to one material phase.

        The black backbone is longer than the marked 42 mm section, so a dark
        centreline alone cannot determine tangential position.  Searching one
        shared shift for all seven rings preserves their identities and cannot
        accumulate seven independent along-shaft errors.
        """

        chain = np.asarray(points, dtype=np.float32).reshape(KEYPOINT_COUNT, 2)
        if not np.isfinite(chain).all() or not np.isfinite(self.templates).all():
            return np.zeros(2, dtype=np.float32), np.inf
        tangent = chain[-1] - chain[0]
        length = float(np.linalg.norm(tangent))
        if length <= 1e-4:
            return np.zeros(2, dtype=np.float32), np.inf
        tangent /= length
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
        longitudinal = np.arange(
            -max(5, int(round(0.62 * gap))),
            max(5, int(round(0.62 * gap))) + 1,
            1.0,
            dtype=np.float32,
        )
        transverse = np.arange(
            -max(3, int(round(0.34 * gap))),
            max(3, int(round(0.34 * gap))) + 1,
            1.0,
            dtype=np.float32,
        )
        along, across = np.meshgrid(longitudinal, transverse, indexing="xy")
        shifts = (
            along.reshape(-1, 1) * tangent
            + across.reshape(-1, 1) * normal
        ).astype(np.float32)
        smooth_lab = cv2.GaussianBlur(np.asarray(lab, np.float32), (5, 5), 0)
        height, width = smooth_lab.shape[:2]
        deltas = np.full((len(shifts), KEYPOINT_COUNT), 500.0, dtype=np.float32)
        for index in range(KEYPOINT_COUNT):
            locations = chain[index] + shifts
            x = np.rint(locations[:, 0]).astype(int)
            y = np.rint(locations[:, 1]).astype(int)
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            deltas[inside, index] = np.linalg.norm(
                smooth_lab[y[inside], x[inside]] - self.templates[index], axis=1
            )
        # Two rings may be washed out by glare; the five best template matches
        # still define a unique periodic phase.
        photometric = np.mean(np.sort(deltas, axis=1)[:, :5], axis=1)
        temporal_penalty = 0.018 * np.sum(shifts * shifts, axis=1)
        cost = photometric + temporal_penalty
        best = int(np.argmin(cost))
        quality = float(photometric[best])
        if not np.isfinite(quality) or quality > 0.92 * self.maximum_delta:
            return np.zeros(2, dtype=np.float32), quality
        return shifts[best].copy(), quality

    @staticmethod
    def _snap_to_mask_medial_axis(
        points: np.ndarray,
        mask: np.ndarray | None,
        radius: int,
        strength: float = 0.68,
    ) -> np.ndarray:
        if mask is None or not np.any(mask):
            return points
        binary = np.asarray(mask != 0, dtype=np.uint8)
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        height, width = binary.shape
        output = points.copy()
        for index, point in enumerate(points):
            x, y = np.rint(point).astype(int)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            local = distance[y0:y1, x0:x1]
            if local.size == 0 or float(np.max(local)) < 0.75:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            spatial = (xx - point[0]) ** 2 + (yy - point[1]) ** 2
            score = local - 0.018 * spatial
            location = np.unravel_index(int(np.argmax(score)), score.shape)
            candidate = np.asarray([x0 + location[1], y0 + location[0]], np.float32)
            gain = float(np.clip(strength, 0.0, 1.0))
            output[index] = gain * candidate + (1.0 - gain) * point
        return output

    @staticmethod
    def _regularise_material_chain(
        points: np.ndarray,
        previous: np.ndarray,
        reference_gap: float | np.ndarray,
    ) -> np.ndarray:
        """Apply a topology-preserving material-chain state transition.

        The previous seven material points are moved by a robust, smoothly
        varying displacement field.  Per-marker colour observations contribute
        only when their motion agrees with the other points, after which six
        distance constraints restore the calibrated ring pitches.  Unlike a
        fresh curve fit, this cannot swap marker identities or walk the two
        endpoints along a longer black shaft.
        """

        observed = np.asarray(points, dtype=np.float32).reshape(KEYPOINT_COUNT, 2)
        old = np.asarray(previous, dtype=np.float32).reshape(KEYPOINT_COUNT, 2)
        if not np.isfinite(observed).all() or not np.isfinite(old).all():
            return observed
        reference = np.asarray(reference_gap, dtype=np.float32).reshape(-1)
        if reference.size == 1:
            reference = np.repeat(reference, KEYPOINT_COUNT - 1)
        if reference.size != KEYPOINT_COUNT - 1 or not np.isfinite(reference).all():
            reference = np.linalg.norm(np.diff(old, axis=0), axis=1)
        reference = np.maximum(reference, 2.0)
        indices = np.arange(KEYPOINT_COUNT, dtype=np.float32)
        displacement = observed - old
        median_motion = np.median(displacement, axis=0)
        residual = np.linalg.norm(displacement - median_motion, axis=1)
        median_residual = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median_residual)))
        threshold = max(2.5, median_residual + 3.0 * mad + 0.5)
        inliers = residual <= threshold
        if np.count_nonzero(inliers) >= 4:
            degree = 2 if np.count_nonzero(inliers) >= 5 else 1
            smooth_displacement = np.column_stack((
                np.polyval(np.polyfit(indices[inliers], displacement[inliers, 0], degree), indices),
                np.polyval(np.polyfit(indices[inliers], displacement[inliers, 1], degree), indices),
            )).astype(np.float32)
        else:
            smooth_displacement = np.repeat(median_motion[None, :], KEYPOINT_COUNT, axis=0)
        # Only the jointly fitted displacement field may update the material
        # state.  Writing seven local observations back independently is what
        # creates the alternating snake visible after long sequences.
        gain = 1.0
        old_edges = np.diff(old, axis=0)
        while gain > 0.0625:
            candidate = old + gain * smooth_displacement
            candidate_edges = np.diff(candidate, axis=0)
            orientation = np.sum(candidate_edges * old_edges, axis=1)
            if np.all(orientation > 0.20 * reference ** 2):
                break
            gain *= 0.5
        constrained = old + gain * smooth_displacement
        # Differential tracker drift must not move the centre of the physical
        # 42 mm section.  Its centre follows only the robust whole-chain motion.
        target_centre = np.mean(old + median_motion, axis=0)
        constrained += target_centre - np.mean(constrained, axis=0)
        # Fit one smooth material curve, restore its calibrated total projected
        # length, then sample the six original material intervals along it.
        # This construction cannot form alternating PBD zigzags.
        material = np.concatenate(([0.0], np.cumsum(reference))).astype(np.float32)
        total_reference = float(material[-1])
        parameter = material / max(total_reference, 1e-6)
        degree = min(3, KEYPOINT_COUNT - 1)
        polynomial_x = np.polyfit(parameter, constrained[:, 0], degree)
        polynomial_y = np.polyfit(parameter, constrained[:, 1], degree)
        dense_parameter = np.linspace(0.0, 1.0, 401, dtype=np.float32)
        dense = np.column_stack((
            np.polyval(polynomial_x, dense_parameter),
            np.polyval(polynomial_y, dense_parameter),
        )).astype(np.float32)
        cumulative = np.concatenate((
            [0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))
        ))
        dense_length = float(cumulative[-1])
        if dense_length <= 1e-4:
            return (old + median_motion).astype(np.float32)
        dense_centre = np.mean(dense, axis=0)
        dense = dense_centre + (dense - dense_centre) * (total_reference / dense_length)
        dense += target_centre - np.mean(dense, axis=0)
        cumulative = np.concatenate((
            [0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))
        ))
        query = material * (float(cumulative[-1]) / max(total_reference, 1e-6))
        sampled = np.column_stack((
            np.interp(query, cumulative, dense[:, 0]),
            np.interp(query, cumulative, dense[:, 1]),
        )).astype(np.float32)
        return sampled

    def refine(
        self,
        image_bgr: np.ndarray,
        pixels: np.ndarray,
        confidence: np.ndarray,
        body_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        learned = np.asarray(pixels, dtype=np.float32).reshape(KEYPOINT_COUNT, 2).copy()
        scores = np.asarray(confidence, dtype=np.float32).reshape(KEYPOINT_COUNT).copy()
        points = self.predict(learned, scores)
        valid = np.isfinite(points).all(axis=1) & (scores > 0.05)
        lab = cv2.cvtColor(np.asarray(image_bgr, np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        height, width = lab.shape[:2]
        if not np.isfinite(self.templates).all():
            self.initialise(image_bgr, points)
        gap = float(np.nanmedian(self.reference_gaps))
        if not np.isfinite(gap):
            gap = float(np.nanmedian(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        gap = max(gap, 4.0)
        band_shift = np.zeros(2, dtype=np.float32)
        band_quality = np.inf
        if self.frames_since_initialisation > 0:
            band_shift, band_quality = self._joint_band_shift(lab, points, gap)
            points += band_shift
        accepted_body = (
            body_mask is not None
            and np.asarray(body_mask).shape == (height, width)
            and np.count_nonzero(body_mask) >= 18
        )
        active_mask = np.asarray(body_mask, np.uint8) if accepted_body else None
        material_mask = self._material_centerline_mask(
            image_bgr, points, gap, active_mask
        )
        if material_mask is None and self.frames_since_initialisation > 0:
            material_mask = self._material_centerline_mask(
                image_bgr, points, gap, active_mask, corridor_scale=2.8
            )
        material_identified = material_mask is not None
        if self.frames_since_initialisation == 0:
            # K0--K6 clicks are the identity ground truth on the conditioning
            # frame.  SAM may include a nearby transparent boundary while its
            # video memory is being created, so never move those seven labels
            # on frame zero.
            self.frames_since_initialisation = 1
            self.last_body_mask = (
                None if material_mask is None else np.asarray(material_mask, np.uint8).copy()
            )
            self.tracking_source = "conditioning-frame"
            initial_valid = np.isfinite(self.last_points).all(axis=1)
            return self.last_points.copy(), scores, initial_valid
        if not material_identified:
            # Fail closed.  Keep the internal state for a subsequent wide
            # re-acquisition, but never publish off-centre or colour-only 2-D
            # coordinates as measurements.
            self.tracking_source = "centerline-lost-reacquiring"
            self.frames_since_initialisation += 1
            return (
                np.full((KEYPOINT_COUNT, 2), np.nan, dtype=np.float32),
                np.zeros(KEYPOINT_COUNT, dtype=np.float32),
                np.zeros(KEYPOINT_COUNT, dtype=bool),
            )
        if material_identified:
            # SAM2 object identity survives a temporary point-level CoTracker
            # occlusion.  Do not throw away the material chain merely because
            # the stale proposal reports visibility=False.
            valid |= np.isfinite(points).all(axis=1)
            scores = np.maximum(scores, 0.62)
        tracking_mask = material_mask
        if tracking_mask is None and self.foreground_fallback_enabled:
            tracking_mask = self._foreground_motion_mask(image_bgr, points, gap)
        mask_motion = self._body_motion(tracking_mask, gap)
        # The seven-ring phase owns tangential motion.  The dark mask may only
        # correct the normal direction, otherwise its centroid slides toward
        # the longer unmarked shaft.
        chain_axis = points[-1] - points[0]
        chain_axis_length = float(np.linalg.norm(chain_axis))
        if chain_axis_length > 1e-4:
            chain_axis /= chain_axis_length
            mask_motion -= float(mask_motion @ chain_axis) * chain_axis
        if np.linalg.norm(mask_motion) > 0.15:
            points += mask_motion
            self.tracking_source = (
                "band-phase+black-centerline"
                if material_mask is not None
                else "foreground-material-mask"
            )
        else:
            self.tracking_source = (
                "band-phase+black-centerline"
                if material_mask is not None
                else "centerline-lost-reacquiring"
            )
        # Only a temporally identified SAM2 object is strong enough to define
        # the medial axis.  Fixed-background difference is a translation cue,
        # not a segmentation mask: specular changes on the lung wall must not
        # pull individual K points away from the material chain.
        points = self._snap_to_mask_medial_axis(
            points,
            material_mask,
            max(5, int(round(0.50 * gap))),
        )
        support = None
        support_mask = material_mask
        if support_mask is not None:
            support = cv2.dilate(
                np.asarray(support_mask, np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (max(9, int(round(0.70 * gap))) | 1,) * 2,
                ),
            ) != 0
        previous = self.last_points.copy()
        for index in range(KEYPOINT_COUNT):
            if not valid[index]:
                continue
            x, y = np.rint(points[index]).astype(int)
            x0, x1 = max(0, x - self.radius), min(width, x + self.radius + 1)
            y0, y1 = max(0, y - self.radius), min(height, y + self.radius + 1)
            if x1 <= x0 or y1 <= y0:
                valid[index] = False
                scores[index] = 0.0
                continue
            patch = lab[y0:y1, x0:x1]
            delta = np.linalg.norm(patch - self.templates[index], axis=2)
            # Five-pixel support rejects a single transparent-wall highlight.
            delta = cv2.boxFilter(delta, -1, (5, 5), normalize=True)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            spatial = ((xx - points[index, 0]) ** 2 + (yy - points[index, 1]) ** 2)
            cost = delta + 0.035 * spatial
            if support is not None:
                cost = np.where(support[y0:y1, x0:x1], cost, np.inf)
            location = np.unravel_index(int(np.argmin(cost)), cost.shape)
            best_delta = float(delta[location])
            if not np.isfinite(cost[location]) or best_delta > self.maximum_delta:
                valid[index] = False
                scores[index] = 0.0
                continue
            corrected = np.asarray([x0 + location[1], y0 + location[0]], np.float32)
            correction = corrected - points[index]
            left = points[max(0, index - 1)]
            right = points[min(KEYPOINT_COUNT - 1, index + 1)]
            local_tangent = right - left
            local_length = float(np.linalg.norm(local_tangent))
            if local_length > 1e-4:
                local_tangent /= local_length
                correction -= float(correction @ local_tangent) * local_tangent
            correction_length = float(np.linalg.norm(correction))
            maximum_correction = max(4.0, 0.48 * gap)
            if correction_length > maximum_correction:
                correction *= maximum_correction / max(correction_length, 1e-6)
            points[index] += 0.72 * correction
            colour_quality = np.exp(-0.5 * (best_delta / max(self.maximum_delta * 0.45, 1.0)) ** 2)
            scores[index] *= float(colour_quality)
            # Keep the conditioning-frame colour identities fixed.  Updating a
            # template after a reflection-induced miss causes irreversible
            # identity drift along the black shaft.
        # The reported points are hard-projected back to the extracted black
        # backbone after colour correction, then constrained by the persistent
        # K0--K6 material topology.  No off-centre candidate reaches 3-D.
        if material_mask is not None:
            points = self._snap_to_mask_medial_axis(
                points, material_mask, max(5, int(round(0.55 * gap)))
            )
        points = self._regularise_material_chain(
            points, previous, self.reference_gaps
        )
        inside_image = (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= 0.0) & (points[:, 0] < float(width))
            & (points[:, 1] >= 0.0) & (points[:, 1] < float(height))
        )
        if not np.all(inside_image):
            # A malformed segmentation/photometric proposal must never be
            # allowed to poison the persistent material state.
            self.tracking_source = "centerline-lost-reacquiring"
            self.frames_since_initialisation += 1
            return (
                np.full((KEYPOINT_COUNT, 2), np.nan, dtype=np.float32),
                np.zeros(KEYPOINT_COUNT, dtype=np.float32),
                np.zeros(KEYPOINT_COUNT, dtype=bool),
            )
        gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        reliable_gaps = gaps[np.isfinite(gaps) & (gaps > 1.0)]
        if len(reliable_gaps) >= 3:
            median = float(np.median(reliable_gaps))
            for index, interval in enumerate(gaps):
                if not np.isfinite(interval) or interval > self.maximum_step_scale * median:
                    # Mark both endpoints conservatively; the 3-D smoother may
                    # predict them but they are not claimed as observations.
                    valid[index:index + 2] = False
                    scores[index:index + 2] = 0.0
        if np.isfinite(previous).all():
            displacement = points - previous
            if not material_identified:
                maximum_step = max(3.0, 0.30 * gap)
                length = np.linalg.norm(displacement, axis=1)
                excessive = length > maximum_step
                if np.any(excessive):
                    displacement[excessive] *= (
                        maximum_step / np.maximum(length[excessive], 1e-6)
                    )[:, None]
                    points[excessive] = previous[excessive] + displacement[excessive]
                    scores[excessive] *= 0.72
                learned_residual = np.linalg.norm(points - learned, axis=1)
                far = learned_residual > max(10.0, 1.15 * gap)
                points[far] = 0.72 * points[far] + 0.28 * learned[far]
                displacement = points - previous
            self.velocity = 0.68 * self.velocity + 0.32 * displacement
            velocity_length = np.linalg.norm(self.velocity, axis=1)
            maximum_velocity = max(1.5, 0.16 * gap)
            fast = velocity_length > maximum_velocity
            self.velocity[fast] *= (
                maximum_velocity / np.maximum(velocity_length[fast], 1e-6)
            )[:, None]
            if not material_identified:
                self.velocity *= 0.82
        else:
            self.velocity[:] = 0.0
        self.last_points[:] = points
        self.last_body_mask = (
            None if tracking_mask is None else np.asarray(tracking_mask, np.uint8).copy()
        )
        self.frames_since_initialisation += 1
        return points, scores, valid
