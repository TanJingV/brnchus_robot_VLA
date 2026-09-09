"""Seven-marker RGB/depth/IR fusion for the two-section TDCR."""

from .curve_model import ContinuumShape3D, fit_continuum_shape
from .models import SevenMarkerResult
from .pipeline import SevenMarkerFusionPipeline

__all__ = [
    "ContinuumShape3D",
    "SevenMarkerFusionPipeline",
    "SevenMarkerResult",
    "fit_continuum_shape",
]

__version__ = "0.2.0"
