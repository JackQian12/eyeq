"""
aggregator.py
=============
将 EyeTracker → BlinkDetector → TearFilmEstimator 连接为单一
流水线，并维护一个用于仪表盘渲染的滑动窗口快照列表。

Chains EyeTracker → BlinkDetector → TearFilmEstimator into a single
pipeline and maintains a sliding-window snapshot list for dashboard use.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from vision.eye_tracker import EyeTracker, EyeLandmarks
from vision.blink_detector import BlinkDetector, BlinkEvent
from vision.tear_film import TearFilmEstimator, TearFilmMetrics


@dataclass
class FrameSnapshot:
    """每帧的轻量级快照，用于实时图表。Lightweight per-frame snapshot for live charts."""
    timestamp: float
    ear: float
    blink_event: Optional[BlinkEvent]
    risk_score: Optional[float]
    risk_level: Optional[str]


class MetricsAggregator:
    """
    一站式流水线管理器。
    All-in-one pipeline manager.

    使用方法 / Usage::

        agg = MetricsAggregator(config)
        while cap.isOpened():
            ret, frame = cap.read()
            snapshot, metrics = agg.process_frame(frame)
    """

    def __init__(
        self,
        ear_threshold: float = 0.20,
        ear_incomplete_threshold: float = 0.23,
        closed_frames_min: int = 2,
        closed_frames_max: int = 90,
        refractory_frames: int = 5,
        fps: float = 30.0,
        window_seconds: float = 60.0,
        normal_blink_rate_min: float = 15.0,
        normal_blink_rate_max: float = 25.0,
        incomplete_blink_risk_threshold: float = 0.40,
        nibut_long_risk_seconds: float = 6.0,
        snapshot_buffer_size: int = 1800,   # 约 60s @30fps
    ) -> None:
        self.tracker = EyeTracker()
        self.detector = BlinkDetector(
            ear_threshold=ear_threshold,
            ear_incomplete_threshold=ear_incomplete_threshold,
            closed_frames_min=closed_frames_min,
            closed_frames_max=closed_frames_max,
            refractory_frames=refractory_frames,
            fps=fps,
        )
        self.estimator = TearFilmEstimator(
            window_seconds=window_seconds,
            nibut_long_risk_seconds=nibut_long_risk_seconds,
            normal_blink_rate_min=normal_blink_rate_min,
            normal_blink_rate_max=normal_blink_rate_max,
            incomplete_blink_risk_threshold=incomplete_blink_risk_threshold,
        )
        self._snapshots: Deque[FrameSnapshot] = deque(maxlen=snapshot_buffer_size)
        self._last_metrics: Optional[TearFilmMetrics] = None
        self._session_start: float = time.time()

    # ── 核心处理循环 / Core processing loop ──────────────────────────────────

    def process_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[FrameSnapshot, Optional[TearFilmMetrics]]:
        """
        处理一帧图像，返回 (快照, 泪膜指标)。
        Process one BGR frame; returns (FrameSnapshot, TearFilmMetrics | None).

        泪膜指标每秒更新一次（非每帧），以减少抖动。
        Tear-film metrics are recomputed at most once per second to reduce jitter.
        """
        lm: EyeLandmarks = self.tracker.process_frame(frame_bgr)
        blink_event: Optional[BlinkEvent] = self.detector.update(lm)

        # 每秒重新计算泪膜指标
        now = time.time()
        if (
            self._last_metrics is None
            or (now - self._last_metrics.timestamp) >= 1.0
        ):
            self._last_metrics = self.estimator.compute(self.detector)

        snap = FrameSnapshot(
            timestamp=lm.timestamp,
            ear=lm.ear_avg,
            blink_event=blink_event,
            risk_score=self._last_metrics.risk_score if self._last_metrics else None,
            risk_level=self._last_metrics.risk_level if self._last_metrics else None,
        )
        self._snapshots.append(snap)
        return snap, self._last_metrics

    def draw_overlay(self, frame_bgr: np.ndarray, lm: EyeLandmarks) -> np.ndarray:
        """在帧上绘制关键点+风险等级覆盖层。Draw landmarks + risk overlay on frame."""
        out = self.tracker.draw_eye_landmarks(frame_bgr, lm)
        if self._last_metrics:
            from vision.tear_film import TearFilmEstimator
            color = TearFilmEstimator.risk_color(self._last_metrics.risk_level)
            import cv2
            cv2.putText(
                out,
                f"Risk: {self._last_metrics.risk_level.upper()} "
                f"({self._last_metrics.risk_score:.0f}/100)",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
            cv2.putText(
                out,
                f"BR: {self._last_metrics.blink_rate_bpm:.1f} bpm  "
                f"IBR: {self._last_metrics.incomplete_blink_ratio * 100:.0f}%  "
                f"NIBUT~: {self._last_metrics.estimated_nibut_s:.1f}s",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )
        return out

    # ── 属性 / Properties ─────────────────────────────────────────────────────

    @property
    def snapshots(self) -> List[FrameSnapshot]:
        return list(self._snapshots)

    @property
    def latest_metrics(self) -> Optional[TearFilmMetrics]:
        return self._last_metrics

    @property
    def session_duration_s(self) -> float:
        return time.time() - self._session_start

    def release(self) -> None:
        """释放所有资源。Release all resources."""
        self.tracker.release()
