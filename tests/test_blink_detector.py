"""
tests/test_blink_detector.py
============================
单元测试：BlinkDetector 状态机正确性。
Unit tests for BlinkDetector state machine correctness.

运行 / Run:
    pytest tests/ -v
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.eye_tracker import EyeLandmarks
from vision.blink_detector import BlinkDetector, BlinkType


# ── 工具函数 / Helpers ─────────────────────────────────────────────────────────

def _make_lm(ear: float, ts: float = None, frame: int = 0) -> EyeLandmarks:
    """构造给定 EAR 的虚拟 EyeLandmarks。"""
    dummy = np.zeros((6, 2))
    return EyeLandmarks(
        timestamp=ts or time.time(),
        frame_index=frame,
        ear_right=ear,
        ear_left=ear,
        ear_avg=ear,
        right_pts=dummy,
        left_pts=dummy,
        face_detected=True,
    )


def _feed_sequence(detector: BlinkDetector, ear_sequence: list, fps: float = 30.0):
    """将 EAR 序列逐帧馈入检测器，返回所有眨眼事件。"""
    events = []
    dt = 1.0 / fps
    t = time.time()
    for i, ear in enumerate(ear_sequence):
        lm = _make_lm(ear, ts=t + i * dt, frame=i)
        ev = detector.update(lm)
        if ev:
            events.append(ev)
    return events


# ── 测试用例 / Test cases ──────────────────────────────────────────────────────

class TestCompleteBlink:
    """完全眨眼检测。"""

    def test_single_complete_blink(self):
        """标准单次完全眨眼应被检测到。"""
        det = BlinkDetector(closed_frames_min=2, refractory_frames=2)
        # 开眼 → 闭眼（5帧）→ 开眼
        seq = [0.30] * 5 + [0.10] * 5 + [0.30] * 5
        events = _feed_sequence(det, seq)
        assert len(events) == 1
        assert events[0].blink_type == BlinkType.COMPLETE

    def test_blink_below_min_frames_ignored(self):
        """仅1帧低于阈值不应触发眨眼（闪烁噪声）。"""
        det = BlinkDetector(closed_frames_min=2, refractory_frames=2)
        seq = [0.30] * 5 + [0.10] * 1 + [0.30] * 5
        events = _feed_sequence(det, seq)
        assert len(events) == 0

    def test_multiple_blinks(self):
        """连续三次眨眼均应被检测到。"""
        det = BlinkDetector(
            closed_frames_min=2, refractory_frames=2, closed_frames_max=30
        )
        blink_unit = [0.10] * 3 + [0.30] * 15
        seq = blink_unit * 3
        events = _feed_sequence(det, seq)
        assert len(events) == 3
        for ev in events:
            assert ev.blink_type == BlinkType.COMPLETE

    def test_ibi_computed_on_second_blink(self):
        """第二次眨眼应有合理的眨眼间期。"""
        det = BlinkDetector(closed_frames_min=2, refractory_frames=2)
        blink_unit = [0.10] * 3 + [0.30] * 20
        seq = blink_unit * 2
        events = _feed_sequence(det, seq, fps=30.0)
        assert len(events) == 2
        ibi = events[1].inter_blink_interval
        assert ibi is not None and 0.5 < ibi < 3.0

    def test_blink_duration_reasonable(self):
        """眨眼持续时间应在合理范围内（100–400ms）。"""
        det = BlinkDetector(closed_frames_min=2)
        # 6 帧闭眼 @30fps = 200ms
        seq = [0.30] * 5 + [0.10] * 6 + [0.30] * 5
        events = _feed_sequence(det, seq, fps=30.0)
        assert len(events) == 1
        assert 100 < events[0].duration_ms < 500


class TestIncompleteBlink:
    """不完全眨眼检测。"""

    def test_incomplete_blink_detected(self):
        """EAR 仅进入 incomplete 区间应被识别为不完全眨眼。"""
        det = BlinkDetector(
            ear_threshold=0.20,
            ear_incomplete_threshold=0.23,
            closed_frames_min=2,
        )
        # EAR 在 0.21（incomplete 区间）徘徊
        seq = [0.30] * 5 + [0.21] * 4 + [0.30] * 5
        events = _feed_sequence(det, seq)
        assert len(events) >= 1
        assert any(e.blink_type == BlinkType.INCOMPLETE for e in events)


class TestProlongedClosure:
    """长时间闭眼检测。"""

    def test_prolonged_closure(self):
        """超过 closed_frames_max 的闭眼应生成 PROLONGED 事件。"""
        det = BlinkDetector(
            closed_frames_min=2,
            closed_frames_max=10,
        )
        seq = [0.30] * 3 + [0.10] * 15 + [0.30] * 3
        events = _feed_sequence(det, seq)
        assert len(events) >= 1
        assert events[0].blink_type == BlinkType.PROLONGED


class TestNoFace:
    """人脸未检测到时的鲁棒性。"""

    def test_no_face_resets_state(self):
        """无人脸帧应重置状态，不产生虚假眨眼。"""
        det = BlinkDetector(closed_frames_min=2)
        dummy = np.zeros((6, 2))

        # 开始 2 帧低于阈值
        for i in range(2):
            lm = _make_lm(0.10, frame=i)
            det.update(lm)

        # 无人脸帧重置
        no_face = EyeLandmarks(
            timestamp=time.time(), frame_index=3,
            ear_right=0.0, ear_left=0.0, ear_avg=0.0,
            right_pts=dummy, left_pts=dummy,
            face_detected=False,
        )
        ev = det.update(no_face)
        assert ev is None

        # 之后恢复不应误报眨眼
        lm = _make_lm(0.30, frame=4)
        ev = det.update(lm)
        assert ev is None


class TestBlinkRate:
    """眨眼频率统计。"""

    def test_blink_rate_approx(self):
        """
        以 ~15 bpm 的速率注入眨眼事件，
        blink_rate_per_minute() 应返回接近 15 的值。
        """
        det = BlinkDetector(closed_frames_min=2, refractory_frames=2)
        fps = 30.0
        # 15 bpm → 每 2s 一次眨眼 → 60s 共 30 帧一次眨眼 @30fps
        # 使用 60s 窗口，注入 15 次眨眼
        blink_unit = [0.10] * 3 + [0.30] * (int(fps * 4) - 3)  # 4s 间隔
        seq = blink_unit * 15
        _feed_sequence(det, seq, fps=fps)
        rate = det.blink_rate_per_minute(window_seconds=60.0)
        # 允许 ±30% 误差（由不应期引起的轻微偏差）
        assert 10 <= rate <= 20, f"Blink rate {rate:.1f} bpm out of range"
