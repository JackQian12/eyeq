"""
blink_detector.py
=================
基于 EAR 时间序列的实时眨眼检测器。
Real-time blink detector operating on the EAR time-series.

眨眼类型 / Blink types
-----------------------
- COMPLETE   : EAR 从张眼状态下降并越过 ear_threshold，再回升
- INCOMPLETE : EAR 仅下降到 incomplete 区间（未完全闭合）
- PROLONGED  : 眼睛闭合时间超过 closed_frames_max（长时间闭眼）
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, List, Optional

from .eye_tracker import EyeLandmarks


class BlinkType(Enum):
    COMPLETE   = auto()  # 完全眨眼
    INCOMPLETE = auto()  # 不完全眨眼
    PROLONGED  = auto()  # 长时间闭眼（超过阈值帧数）


@dataclass
class BlinkEvent:
    """一次眨眼事件的记录。A record of a single blink event."""
    blink_type: BlinkType
    start_time: float       # 闭眼开始时刻
    end_time: float         # 睁眼恢复时刻
    min_ear: float          # 本次眨眼 EAR 最低值
    duration_ms: float      # 持续时间 (毫秒)
    frame_start: int
    frame_end: int

    @property
    def inter_blink_interval(self) -> Optional[float]:
        """由 BlinkDetector 在发现后设置。Set externally after detection."""
        return self._ibi

    @inter_blink_interval.setter
    def inter_blink_interval(self, v: float) -> None:
        self._ibi = v

    def __post_init__(self) -> None:
        self._ibi: Optional[float] = None


class BlinkDetector:
    """
    状态机式眨眼检测器。
    State-machine blink detector.

    工作原理
    --------
    1. 每帧输入 EyeLandmarks，更新状态机。
    2. 当 EAR 连续 >= closed_frames_min 帧低于 ear_threshold 时，进入 CLOSED 状态。
    3. EAR 恢复后，根据最低 EAR 和持续帧数分类眨眼类型，触发回调。
    """

    def __init__(
        self,
        ear_threshold: float = 0.20,
        ear_incomplete_threshold: float = 0.23,
        closed_frames_min: int = 2,
        closed_frames_max: int = 90,
        refractory_frames: int = 5,
        fps: float = 30.0,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.ear_incomplete_threshold = ear_incomplete_threshold
        self.closed_frames_min = closed_frames_min
        self.closed_frames_max = closed_frames_max
        self.refractory_frames = refractory_frames
        self.fps = fps

        # 状态变量 / State variables
        self._below_threshold_count: int = 0     # 连续低于阈值帧计数
        self._below_incomplete_count: int = 0    # 连续进入 incomplete 区间计数
        self._in_blink: bool = False             # 是否正在眨眼中
        self._blink_start_time: float = 0.0
        self._blink_start_frame: int = 0
        self._min_ear_during_blink: float = 1.0
        self._refractory_counter: int = 0        # 不应期计数

        # 历史记录 / History
        self.blink_history: List[BlinkEvent] = []
        self._last_blink_end_time: Optional[float] = None

    # ── 主接口 / Main interface ───────────────────────────────────────────────

    def update(self, lm: EyeLandmarks) -> Optional[BlinkEvent]:
        """
        使用新一帧的眼部特征更新检测器状态。
        Update detector with a new frame's eye landmarks.

        参数 / Args:
            lm: 当前帧 EyeLandmarks

        返回 / Returns:
            BlinkEvent — 若本帧触发了完整眨眼事件，否则 None
        """
        if not lm.face_detected:
            self._reset_blink_state()
            return None

        if self._refractory_counter > 0:
            self._refractory_counter -= 1
            return None

        ear = lm.ear_avg
        event: Optional[BlinkEvent] = None

        if ear < self.ear_threshold:
            # ── 完全闭眼区间 ─────────────────────────────────────────
            if not self._in_blink:
                self._below_threshold_count += 1
                if self._below_threshold_count >= self.closed_frames_min:
                    self._in_blink = True
                    self._blink_start_time = lm.timestamp
                    self._blink_start_frame = lm.frame_index
                    self._min_ear_during_blink = ear
            else:
                self._min_ear_during_blink = min(self._min_ear_during_blink, ear)
                self._below_threshold_count += 1

                if self._below_threshold_count > self.closed_frames_max:
                    # 长时间闭眼 → 发出 PROLONGED 事件并结束
                    event = self._finalize_blink(
                        lm, BlinkType.PROLONGED
                    )
        else:
            # ── EAR 恢复（眼睛重新睁开）──────────────────────────────
            if self._in_blink:
                blink_type = (
                    BlinkType.COMPLETE
                    if self._min_ear_during_blink <= self.ear_threshold
                    else BlinkType.INCOMPLETE
                )
                event = self._finalize_blink(lm, blink_type)
            elif (
                ear < self.ear_incomplete_threshold
                and self._below_incomplete_count == 0
            ):
                # 进入 incomplete 区间但未达到完全闭合
                self._below_incomplete_count += 1
            elif (
                ear >= self.ear_incomplete_threshold
                and self._below_incomplete_count > 0
            ):
                # 从 incomplete 区间恢复 — 记录不完全眨眼
                event = BlinkEvent(
                    blink_type=BlinkType.INCOMPLETE,
                    start_time=lm.timestamp,
                    end_time=lm.timestamp,
                    min_ear=ear,
                    duration_ms=0.0,
                    frame_start=lm.frame_index,
                    frame_end=lm.frame_index,
                )
                self._register_event(event)
                self._below_incomplete_count = 0
            else:
                self._below_threshold_count = 0
                self._below_incomplete_count = 0

        return event

    @property
    def total_blinks(self) -> int:
        return len(self.blink_history)

    @property
    def complete_blinks(self) -> List[BlinkEvent]:
        return [e for e in self.blink_history if e.blink_type == BlinkType.COMPLETE]

    @property
    def incomplete_blinks(self) -> List[BlinkEvent]:
        return [e for e in self.blink_history
                if e.blink_type == BlinkType.INCOMPLETE]

    def recent_blinks(self, window_seconds: float = 60.0) -> List[BlinkEvent]:
        """返回最近 window_seconds 秒内的眨眼事件。"""
        cutoff = time.time() - window_seconds
        return [e for e in self.blink_history if e.end_time >= cutoff]

    def blink_rate_per_minute(self, window_seconds: float = 60.0) -> float:
        """计算最近窗口内的眨眼频率（次/分钟）。"""
        recent = self.recent_blinks(window_seconds)
        if window_seconds <= 0:
            return 0.0
        return len(recent) / (window_seconds / 60.0)

    def incomplete_blink_ratio(self, window_seconds: float = 60.0) -> float:
        """不完全眨眼 / 总眨眼 比例（0~1）。"""
        recent = self.recent_blinks(window_seconds)
        if not recent:
            return 0.0
        incomplete = sum(1 for e in recent if e.blink_type == BlinkType.INCOMPLETE)
        return incomplete / len(recent)

    def reset(self) -> None:
        """清除所有历史，重置为初始状态。"""
        self.blink_history.clear()
        self._last_blink_end_time = None
        self._reset_blink_state()

    # ── 内部方法 / Internal ───────────────────────────────────────────────────

    def _finalize_blink(
        self,
        lm: EyeLandmarks,
        blink_type: BlinkType,
    ) -> BlinkEvent:
        duration_ms = (lm.timestamp - self._blink_start_time) * 1000.0
        event = BlinkEvent(
            blink_type=blink_type,
            start_time=self._blink_start_time,
            end_time=lm.timestamp,
            min_ear=self._min_ear_during_blink,
            duration_ms=duration_ms,
            frame_start=self._blink_start_frame,
            frame_end=lm.frame_index,
        )
        self._register_event(event)
        self._reset_blink_state()
        self._refractory_counter = self.refractory_frames
        return event

    def _register_event(self, event: BlinkEvent) -> None:
        if self._last_blink_end_time is not None:
            event.inter_blink_interval = event.end_time - self._last_blink_end_time
        self._last_blink_end_time = event.end_time
        self.blink_history.append(event)

    def _reset_blink_state(self) -> None:
        self._in_blink = False
        self._below_threshold_count = 0
        self._below_incomplete_count = 0
        self._min_ear_during_blink = 1.0
        self._blink_start_time = 0.0
        self._blink_start_frame = 0
