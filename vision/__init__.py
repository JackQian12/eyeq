"""
EyeQ Vision 模块
=================
提供基于 MediaPipe Face Mesh 的眼部追踪、眨眼检测与泪膜间接评估能力。

Provides eye tracking, blink detection, and indirect tear-film assessment
powered by MediaPipe Face Mesh.
"""

from .eye_tracker import EyeTracker, EyeLandmarks
from .blink_detector import BlinkDetector, BlinkEvent, BlinkType
from .tear_film import TearFilmEstimator, TearFilmMetrics

__all__ = [
    "EyeTracker",
    "EyeLandmarks",
    "BlinkDetector",
    "BlinkEvent",
    "BlinkType",
    "TearFilmEstimator",
    "TearFilmMetrics",
]
