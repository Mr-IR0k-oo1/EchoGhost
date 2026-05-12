"""Signal processing primitives used by the dashboard."""

from .adaptive import AdaptationResult, WaveformAdapter
from .classifier import ACTIVITY_LABELS, ActivityClassifier, ActivityResult
from .motion import MotionDetector, MotionMetrics
from .range_doppler import RangeHeatmap, RangeHeatmapResult
from .vitals import BreathingEstimator, BreathingEstimate

__all__ = [
    "ACTIVITY_LABELS",
    "ActivityClassifier",
    "ActivityResult",
    "AdaptationResult",
    "BreathingEstimate",
    "BreathingEstimator",
    "MotionDetector",
    "MotionMetrics",
    "RangeHeatmap",
    "RangeHeatmapResult",
    "WaveformAdapter",
]
