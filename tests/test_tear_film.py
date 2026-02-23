"""
tests/test_tear_film.py
=======================
单元测试：TearFilmEstimator 评分逻辑。
Unit tests for TearFilmEstimator scoring logic.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.blink_detector import BlinkDetector, BlinkEvent, BlinkType
from vision.tear_film import TearFilmEstimator
from tests.test_blink_detector import _feed_sequence


def _inject_blinks_with_ibi(detector: BlinkDetector, n: int, ibi_s: float, fps: float = 30.0):
    """以指定间期注入 n 次完整眨眼。"""
    frames_per_blink = int(fps * 3)  # 3帧闭眼
    gap_frames = max(1, int(fps * ibi_s) - frames_per_blink)
    blink_unit = [0.10] * frames_per_blink + [0.30] * gap_frames
    seq = blink_unit * n
    _feed_sequence(detector, seq, fps=fps)


class TestTearFilmEstimator:

    def test_returns_none_below_min_blinks(self):
        """不足 3 次眨眼时应返回 None。"""
        det = BlinkDetector(closed_frames_min=2, refractory_frames=2)
        est = TearFilmEstimator(window_seconds=60.0)
        _inject_blinks_with_ibi(det, n=2, ibi_s=4.0)
        result = est.compute(det)
        assert result is None

    def test_normal_blink_rate_low_risk(self):
        """
        正常眨眼频率 (15–25 bpm)、短 IBI、低 IBR → 综合风险应较低。
        """
        det = BlinkDetector(closed_frames_min=2, refractory_frames=2)
        est = TearFilmEstimator(window_seconds=60.0)
        # ~20 bpm → IBI ≈ 3s
        _inject_blinks_with_ibi(det, n=20, ibi_s=3.0)
        result = est.compute(det)
        assert result is not None
        assert result.risk_level in ("low", "moderate")

    def test_low_blink_rate_raises_risk(self):
        """
        极低眨眼频率 (< 10 bpm) 应提高风险评分。
        """
        det = BlinkDetector(closed_frames_min=2, refractory_frames=2)
        est = TearFilmEstimator(window_seconds=60.0)
        # ~5 bpm → IBI ≈ 12s
        _inject_blinks_with_ibi(det, n=5, ibi_s=12.0)
        result = est.compute(det)
        assert result is not None
        assert result.risk_score > 30

    def test_high_ibi_raises_risk(self):
        """长 IBI (> 6s) 应提高 IBI 子评分。"""
        det = BlinkDetector(closed_frames_min=2, refractory_frames=2)
        est = TearFilmEstimator(window_seconds=60.0, nibut_long_risk_seconds=6.0)
        _inject_blinks_with_ibi(det, n=8, ibi_s=9.0)
        result = est.compute(det)
        assert result is not None
        assert result.score_ibi > 5

    def test_risk_score_in_range(self):
        """风险评分应始终在 0–100 之间。"""
        det = BlinkDetector(closed_frames_min=2, refractory_frames=2)
        est = TearFilmEstimator(window_seconds=60.0)
        _inject_blinks_with_ibi(det, n=10, ibi_s=4.0)
        result = est.compute(det)
        assert result is not None
        assert 0.0 <= result.risk_score <= 100.0

    def test_nibut_estimate_matches_ibi(self):
        """估算 NIBUT 应与眨眼间期均值接近。"""
        det = BlinkDetector(closed_frames_min=2, refractory_frames=2)
        est = TearFilmEstimator(window_seconds=60.0)
        _inject_blinks_with_ibi(det, n=12, ibi_s=5.0)
        result = est.compute(det)
        assert result is not None
        # 估算 NIBUT 应在目标 IBI 的 ±50% 以内
        assert abs(result.estimated_nibut_s - 5.0) < 3.0

    def test_sub_scores_sum_to_total(self):
        """子评分之和应等于总分。"""
        det = BlinkDetector(closed_frames_min=2, refractory_frames=2)
        est = TearFilmEstimator(window_seconds=60.0)
        _inject_blinks_with_ibi(det, n=15, ibi_s=4.0)
        result = est.compute(det)
        assert result is not None
        expected = (result.score_blink_rate + result.score_ibr
                    + result.score_ibi + result.score_ibi_cv)
        assert abs(result.risk_score - expected) < 0.01
