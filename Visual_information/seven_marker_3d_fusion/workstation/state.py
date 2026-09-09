"""Single source of truth for the workstation UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from Visual_information.d435_tdcr_capture.config import VISUAL_ROOT

from ..session import SessionDescriptor


@dataclass
class WorkstationState:
    descriptor: SessionDescriptor | None = None
    current_frame: int = 0
    initial_frame: int = 0
    initial_keypoints_px: np.ndarray | None = None
    target_roi: tuple[float, float, float, float] | None = None
    output_root: Path = field(default_factory=lambda: (VISUAL_ROOT / "outputs").resolve())
    latest_output: Path | None = None
    processing: bool = False

    @property
    def source_ready(self) -> bool:
        return bool(
            self.descriptor is not None
            and self.descriptor.has_rgb
            and self.descriptor.has_depth
            and self.descriptor.frame_count > 0
        )

    @property
    def initialization_ready(self) -> bool:
        return bool(
            self.initial_keypoints_px is not None
            and np.asarray(self.initial_keypoints_px).shape == (7, 2)
            and np.isfinite(self.initial_keypoints_px).all()
        )

    def reset_initialization(self) -> None:
        self.initial_frame = self.current_frame
        self.initial_keypoints_px = None
        self.target_roi = None

    def set_initialization(self, frame: int, pixels: np.ndarray, image_shape: tuple[int, int]) -> None:
        points = np.asarray(pixels, dtype=float).reshape(7, 2)
        if not np.isfinite(points).all():
            raise ValueError("七个初始化点必须全部有效")
        height, width = image_shape
        gap = float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        margin_x = max(8.0, 0.55 * gap)
        margin_y = max(10.0, 1.8 * margin_x)
        self.initial_frame = int(frame)
        self.initial_keypoints_px = points.copy()
        self.target_roi = (
            float(np.clip((np.min(points[:, 0]) - margin_x) / width, 0.0, 1.0)),
            float(np.clip((np.min(points[:, 1]) - margin_y) / height, 0.0, 1.0)),
            float(np.clip((np.max(points[:, 0]) + margin_x) / width, 0.0, 1.0)),
            float(np.clip((np.max(points[:, 1]) + margin_y) / height, 0.0, 1.0)),
        )
