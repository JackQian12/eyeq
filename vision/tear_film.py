"""
tear_film.py
============
基于眨眼行为模式的泪膜状态间接评估模块。
Indirect tear-film status estimator based on blink behavioral patterns.

科学背景 / Scientific background
---------------------------------
泪膜破裂时间 (TBUT / NIBUT) 是诊断干眼症的核心指标。临床金标准需要荧光素
染色与裂隙灯，消费级设备无法直接测量。

本模块采用以下行为指标间接估算泪膜状态：

1. **眨眼频率 (Blink Rate, BR)**  
   - 正常参考：15–25 次/分钟  
   - 干眼患者通常有代偿性眨眼频率升高（神经感觉刺激）

2. **不完全眨眼比例 (Incomplete Blink Ratio, IBR)**  
   - 眼睛未完全闭合的眨眼比例  
   - IBR > 40% 与睑缘腺功能障碍 (MGD) 强相关  
   - 不完全眨眼减少泪液涂布，加速泪膜破裂

3. **眨眼间期 (Inter-Blink Interval, IBI)**  
   - IBI 均值：估算 NIBUT 的间接代理指标  
   - 用户通常在泪膜即将或刚刚破裂时才眨眼  
   - 正常 IBI ≈ 3–6 s；长于 6 s 可能指示反射性眨眼减弱

4. **IBI 变异系数 (IBI-CV)**  
   - 高变异性 → 眨眼模式不稳定 → 泪膜稳定性差

5. **综合风险评分 (Risk Score, 0–100)**  
   - 基于上述四项指标的加权评分
   - 0–30 = 低风险（绿），31–60 = 中风险（黄），61–100 = 高风险（红）
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .blink_detector import BlinkDetector, BlinkEvent, BlinkType


@dataclass
class TearFilmMetrics:
    """当前泪膜评估指标快照。Snapshot of current tear-film assessment metrics."""

    timestamp: float

    # ── 眨眼行为指标 / Blink behavior metrics ────────────────────────────────
    blink_rate_bpm: float          # 眨眼频率 (次/分钟)
    incomplete_blink_ratio: float  # 不完全眨眼比例 (0–1)
    ibi_mean_s: float              # 平均眨眼间期 (秒)
    ibi_std_s: float               # 眨眼间期标准差
    ibi_cv: float                  # 变异系数 = std / mean

    # ── 泪膜估算 / Tear film estimates ───────────────────────────────────────
    estimated_nibut_s: float       # 估算 NIBUT (秒) — 基于 IBI 均值
    risk_score: float              # 综合风险评分 (0–100)
    risk_level: str                # "low" | "moderate" | "high"

    # ── 样本量 / Sample counts ───────────────────────────────────────────────
    n_blinks_in_window: int        # 计算窗口内的眨眼次数
    window_seconds: float          # 计算窗口长度

    # ── 各项子评分 (用于可视化) / Sub-scores for visualization ──────────────
    score_blink_rate: float        # 0–25
    score_ibr: float               # 0–25
    score_ibi: float               # 0–25
    score_ibi_cv: float            # 0–25
    # ── 泪河高度指标 / Tear Meniscus Height (TMH) ───────────────────
    tmh_avg_mm: float = 0.0        # 滚动平均下角膜缘暴露量 (mm, 有符号)
    tmh_status: str = "unknown"    # "normal" | "borderline" | "low" | "unknown"

class TearFilmEstimator:
    """
    从 BlinkDetector 的历史记录中计算泪膜评估指标。
    Computes tear-film metrics from BlinkDetector's blink history.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        nibut_long_risk_seconds: float = 6.0,
        normal_blink_rate_min: float = 15.0,
        normal_blink_rate_max: float = 25.0,
        incomplete_blink_risk_threshold: float = 0.40,
    ) -> None:
        self.window_seconds = window_seconds
        self.nibut_long_risk_seconds = nibut_long_risk_seconds
        self.normal_br_min = normal_blink_rate_min
        self.normal_br_max = normal_blink_rate_max
        self.ibr_risk_thresh = incomplete_blink_risk_threshold

    # ── 主计算接口 / Main computation ────────────────────────────────────────

    def compute(self, detector: BlinkDetector) -> Optional[TearFilmMetrics]:
        """
        根据最近 window_seconds 的眨眼事件计算泪膜指标。
        Compute metrics from recent blink events.

        返回 None 若数据量不足（< 3 次眨眼）。
        Returns None if insufficient data (< 3 blinks).
        """
        recent = detector.recent_blinks(self.window_seconds)
        if len(recent) < 3:
            return None

        # 1. 眨眼频率
        br = len(recent) / (self.window_seconds / 60.0)

        # 2. 不完全眨眼比例
        ibr = detector.incomplete_blink_ratio(self.window_seconds)

        # 3. IBI 统计
        ibis = self._extract_ibis(recent)
        if len(ibis) < 2:
            return None

        ibi_arr = np.array(ibis)
        ibi_mean = float(ibi_arr.mean())
        ibi_std  = float(ibi_arr.std(ddof=1))
        ibi_cv   = ibi_std / ibi_mean if ibi_mean > 0 else 0.0

        # 4. 估算 NIBUT
        # IBI 均值是泪膜破裂时间的保守下界估计：
        # 若用户不需要眨眼就不会眨，所以 IBI ≈ 平均泪膜破裂时间
        estimated_nibut = ibi_mean

        # 5. 计算各项子评分 (越高 = 越高风险，每项 0–25)
        score_br  = self._score_blink_rate(br)
        score_ibr = self._score_ibr(ibr)
        score_ibi = self._score_ibi(ibi_mean)
        score_cv  = self._score_ibi_cv(ibi_cv)
        total_risk = score_br + score_ibr + score_ibi + score_cv

        if total_risk < 30:
            level = "low"
        elif total_risk < 60:
            level = "moderate"
        else:
            level = "high"

        return TearFilmMetrics(
            timestamp=time.time(),
            blink_rate_bpm=br,
            incomplete_blink_ratio=ibr,
            ibi_mean_s=ibi_mean,
            ibi_std_s=ibi_std,
            ibi_cv=ibi_cv,
            estimated_nibut_s=estimated_nibut,
            risk_score=total_risk,
            risk_level=level,
            n_blinks_in_window=len(recent),
            window_seconds=self.window_seconds,
            score_blink_rate=score_br,
            score_ibr=score_ibr,
            score_ibi=score_ibi,
            score_ibi_cv=score_cv,
        )

    # ── 子评分函数 / Sub-score functions (0–25 each, higher = worse) ─────────

    def _score_blink_rate(self, br: float) -> float:
        """
        眨眼率评分：
        - 低于正常 → 高风险（泪液分布不足）
        - 过高 → 中风险（代偿性反射）
        - 正常范围 → 低分
        """
        if br < self.normal_br_min:
            # 线性插值：br=0 → 25 分，br=normal_min → 0 分
            return min(25.0, 25.0 * (1 - br / self.normal_br_min))
        elif br > self.normal_br_max:
            # 适度惩罚
            excess = br - self.normal_br_max
            return min(25.0, excess * 0.8)
        else:
            return 0.0

    def _score_ibr(self, ibr: float) -> float:
        """不完全眨眼比例评分：IBR > risk_threshold → 25 分。"""
        if ibr <= self.ibr_risk_thresh:
            return 25.0 * (ibr / self.ibr_risk_thresh) ** 2
        else:
            return min(25.0, 25.0 + (ibr - self.ibr_risk_thresh) * 20.0)

    def _score_ibi(self, ibi_mean: float) -> float:
        """
        IBI 均值评分：
        - IBI < 2s → 高补偿眨眼，轻度风险
        - IBI 3–5s → 正常，低分
        - IBI > 6s → 泪膜破裂前延迟长，高风险
        """
        if ibi_mean < 2.0:
            return min(25.0, (2.0 - ibi_mean) * 5.0)
        elif ibi_mean <= self.nibut_long_risk_seconds:
            normal_range = self.nibut_long_risk_seconds - 2.0
            pos = ibi_mean - 2.0
            # 中间低，两端略高
            return max(0.0, 5.0 * abs(pos / normal_range - 0.5))
        else:
            excess = ibi_mean - self.nibut_long_risk_seconds
            return min(25.0, excess * 3.0)

    def _score_ibi_cv(self, cv: float) -> float:
        """IBI 变异系数评分：CV > 0.5 认为高变异。"""
        return min(25.0, cv * 35.0)

    @staticmethod
    def classify_tmh(tmh_mm: float) -> str:
        """
        泪河代理指标分级（下角膜缘暴露量，有符号）。
        Classify lower limbus exposure (signed TMH proxy).

        内底逻辑 / Logic:
          暴露量 ≤0 mm  → 正常：眼睑覆盖虹膜下缘，泪河区良好
          暴露量 0–1 mm → 临界：巩膜少量暴露
          暴露量 > 1 mm  → 偏高风险：巩膜显著暴露，提示睑裂小 / 干眼
        """
        if tmh_mm == 0.0:
            return "unknown"
        if tmh_mm <= 0.0:
            return "normal"
        if tmh_mm <= 1.0:
            return "borderline"
        return "low"

    # ── 工具函数 / Utilities ─────────────────────────────────────────────────

    @staticmethod
    def _extract_ibis(events: List[BlinkEvent]) -> List[float]:
        """从眨眼事件列表中提取有效的眨眼间期 (秒)。"""
        ibis: List[float] = []
        prev_end: Optional[float] = None
        for ev in events:
            if prev_end is not None:
                ibi = ev.end_time - prev_end
                if 0.2 < ibi < 30.0:   # 过滤异常值
                    ibis.append(ibi)
            prev_end = ev.end_time
        return ibis

    @staticmethod
    def risk_color(level: str) -> tuple[int, int, int]:
        """返回风险等级对应的 BGR 颜色。"""
        return {
            "low":      (80, 200, 80),
            "moderate": (0, 165, 255),
            "high":     (0, 60, 220),
        }.get(level, (128, 128, 128))
