"""D435-based shape capture and MuJoCo alignment for the two-section TDCR."""

from .config import DEFAULT_CONFIG_PATH, load_config
from .em import EmFrame, EmToolPose, NdiEmSource
from .models import AlignedSample, AxisSample, CameraFrame, KeypointObservation, RgbCameraFrame

__all__ = [
    "AlignedSample",
    "AxisSample",
    "CameraFrame",
    "EmFrame",
    "EmToolPose",
    "RgbCameraFrame",
    "DEFAULT_CONFIG_PATH",
    "KeypointObservation",
    "NdiEmSource",
    "load_config",
]

__version__ = "0.1.0"
