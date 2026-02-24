"""
eye_tracker.py
==============
使用 MediaPipe Face Mesh 从视频帧中提取眼部关键点，并计算
眼睛纵横比 (EAR — Eye Aspect Ratio)。

Eye Aspect Ratio (Soukupová & Čech, 2016):
    EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2·‖p1-p4‖)

MediaPipe Face Mesh 右眼关键点索引 (landmark indices):
    p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
左眼:
    p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
import numpy as np

# 默认模型路径（与 main.py 同级目录）
_DEFAULT_MODEL = Path(__file__).parent.parent / "face_landmarker.task"

# ── MediaPipe landmark indices ────────────────────────────────────────────────
# 右眼 Right eye  (viewer's left)
_RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
# 左眼 Left eye   (viewer's right)
_LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]

# 眼部轮廓 (用于可视化) Eye contour for visualization
_RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                       173, 157, 158, 159, 160, 161, 246]
_LEFT_EYE_CONTOUR  = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                       466, 388, 387, 386, 385, 384, 398]

# 虹膜关键点 Iris landmarks (MediaPipe FaceLandmarker with iris model)
# 实际顺序 (经实测验证): [center, right, UPPER, left, LOWER]
# Empirically verified order: 468=center, 469=right, 470=UPPER, 471=left, 472=LOWER
_RIGHT_IRIS_IDX = [468, 469, 470, 471, 472]
_LEFT_IRIS_IDX  = [473, 474, 475, 476, 477]

# 下眼睑缘关键点 (用于泪河位置代理指标)
# Lower eyelid MARGIN landmarks (for TMH proxy measurement)
# MediaPipe face mesh: 145 = right center lower lid margin, 374 = left
_RIGHT_LOWER_LID_MARGIN = 145
_LEFT_LOWER_LID_MARGIN  = 374

# 人类虹膜平均直径 (mm) — 用作像素到毫米换算尺度参照
_IRIS_DIAMETER_MM: float = 11.7


@dataclass
class EyeLandmarks:
    """单帧眼部特征 / Per-frame eye feature snapshot."""
    timestamp: float            # UNIX 时间戳
    frame_index: int
    ear_right: float            # 右眼 EAR
    ear_left: float             # 左眼 EAR
    ear_avg: float              # 双眼均值 EAR
    # 关键点坐标 (像素) / Landmark pixel coords
    right_pts: np.ndarray       # shape (6, 2)
    left_pts: np.ndarray        # shape (6, 2)
    face_detected: bool = True
    # 虹膜关键点 (像素) / Iris landmark pixel coords — None 若模型不支持
    iris_right_pts: Optional[np.ndarray] = None   # shape (5, 2)
    iris_left_pts: Optional[np.ndarray]  = None   # shape (5, 2)
    # 泪河高度估算 / Tear Meniscus Height estimates (mm)
    tmh_right_mm: float = 0.0
    tmh_left_mm:  float = 0.0
    tmh_avg_mm:   float = 0.0
    # 泪河高度代理：下角膜缘暴露量 (mm)，负值=正常覆盖，正值=巩膜暴露
    # TMH proxy: signed lower limbus exposure; negative = normal lid coverage
    tmh_exposure_right: float = 0.0   # 右 signed exposure (mm)
    tmh_exposure_left:  float = 0.0   # 左 signed exposure (mm)
    tmh_exposure_avg:   float = 0.0   # 双眼均值


def _limbus_exposure_mm(
    lower_lid_y_px: float,
    iris_pts: np.ndarray,
) -> float:
    """
    计算下角膜缘暴露量（泪河高度代理指标）。
    Compute lower limbus exposure as a TMH proxy (in mm).

    参数 / Args:
      lower_lid_y_px : 下眼睑缘关键点的 y 像素坐标 (landmark 145 / 374)
      iris_pts       : 虹膜 5 点 [center, right, bottom, left, top] 像素坐标

    返回 / Returns:
      正值 = 下巩膜暴露量，提示睑裂不良 / 干眼风险
      负值 = 眼睑正常覆盖虹膜下缘 (tear film apposition OK)
      magnitude 越大、正值越大 → 风险越高

    临床参考 / Clinical reference:
      ≤ -0.5 mm : 正常（eyelid covers > 0.5mm of lower iris）
      -0.5–0 mm : 临界（minimal or no coverage）
      > 0 mm    : 巩膜暴露，干眼风险
    """
    if iris_pts is None or len(iris_pts) < 5:
        return 0.0
    iris_bottom_y  = float(iris_pts[4, 1])          # landmark 472/477 = LOWER limbus
    iris_center_y  = float(iris_pts[0, 1])           # landmark 468/473 = center
    iris_upper_y   = float(iris_pts[2, 1])           # landmark 470/475 = UPPER limbus
    iris_radius_px = abs(iris_center_y - iris_upper_y)
    if iris_radius_px < 2.0:
        return 0.0
    # 正值 = 眼睑在虹膜下缘以下 (scleral show)
    exposure_px = lower_lid_y_px - iris_bottom_y
    return float((exposure_px / (iris_radius_px * 2.0)) * _IRIS_DIAMETER_MM)


# ── 旧函数保留兼容性 / Legacy wrapper kept for compatibility ─────────────────
def _tmh_mm(
    lower_lid_pts: np.ndarray,
    iris_pts: np.ndarray,
) -> float:
    """Deprecated: use _limbus_exposure_mm with landmark 145/374 instead."""
    return 0.0


def _ear(pts: np.ndarray) -> float:
    """
    计算眼睛纵横比 (EAR)。
    pts: 6×2 array — [p1, p2, p3, p4, p5, p6]
    """
    vertical1 = np.linalg.norm(pts[1] - pts[5])  # p2-p6
    vertical2 = np.linalg.norm(pts[2] - pts[4])  # p3-p5
    horizontal = np.linalg.norm(pts[0] - pts[3]) # p1-p4
    if horizontal < 1e-6:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


class EyeTracker:
    """
    封装 MediaPipe FaceLandmarker Tasks API，逐帧输出双眼 EAR 及关键点。
    Wraps MediaPipe FaceLandmarker Tasks API; yields per-frame EAR and landmarks.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ) -> None:
        path = str(model_path or _DEFAULT_MODEL)
        base_options = mp_python.BaseOptions(model_asset_path=path)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._frame_idx = 0

    # ── 公开方法 / Public API ────────────────────────────────────────────────

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> EyeLandmarks:
        """
        处理单帧 BGR 图像，返回眼部特征。
        Process one BGR frame and return eye landmarks + EAR.
        """
        ts = timestamp or time.time()
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)
        self._frame_idx += 1

        if not results.face_landmarks:
            dummy = np.zeros((6, 2))
            return EyeLandmarks(
                timestamp=ts,
                frame_index=self._frame_idx,
                ear_right=0.0,
                ear_left=0.0,
                ear_avg=0.0,
                right_pts=dummy,
                left_pts=dummy,
                face_detected=False,
            )

        lm = results.face_landmarks[0]

        right_pts = self._extract_pts(lm, _RIGHT_EYE_IDX, w, h)
        left_pts  = self._extract_pts(lm, _LEFT_EYE_IDX,  w, h)

        ear_r = _ear(right_pts)
        ear_l = _ear(left_pts)
        ear_avg = (ear_r + ear_l) / 2.0

        # 虹膜关键点 + 泪河高度代理 / Iris landmarks + TMH proxy
        iris_r: Optional[np.ndarray] = None
        iris_l: Optional[np.ndarray] = None
        tmh_r = tmh_l = tmh_a = 0.0
        exp_r = exp_l = exp_a = 0.0
        if len(lm) >= 478:
            iris_r = self._extract_pts(lm, _RIGHT_IRIS_IDX, w, h)
            iris_l = self._extract_pts(lm, _LEFT_IRIS_IDX,  w, h)
            # 使用下眼睑缘关键点 145/374 计算暴露量
            lid_r_y = float(lm[_RIGHT_LOWER_LID_MARGIN].y * h)
            lid_l_y = float(lm[_LEFT_LOWER_LID_MARGIN].y  * h)
            exp_r = _limbus_exposure_mm(lid_r_y, iris_r)
            exp_l = _limbus_exposure_mm(lid_l_y, iris_l)
            exp_a = (exp_r + exp_l) / 2.0
            # 为向后兼容保留 tmh_*_mm 字段（设为同一值）
            tmh_r, tmh_l, tmh_a = exp_r, exp_l, exp_a

        return EyeLandmarks(
            timestamp=ts,
            frame_index=self._frame_idx,
            ear_right=ear_r,
            ear_left=ear_l,
            ear_avg=ear_avg,
            right_pts=right_pts,
            left_pts=left_pts,
            face_detected=True,
            iris_right_pts=iris_r,
            iris_left_pts=iris_l,
            tmh_right_mm=tmh_r,
            tmh_left_mm=tmh_l,
            tmh_avg_mm=tmh_a,
            tmh_exposure_right=exp_r,
            tmh_exposure_left=exp_l,
            tmh_exposure_avg=exp_a,
        )

    def draw_eye_landmarks(
        self,
        frame_bgr: np.ndarray,
        landmarks: EyeLandmarks,
        color: Tuple[int, int, int] = (0, 255, 200),
        radius: int = 2,
    ) -> np.ndarray:
        """在帧上绘制眼部关键点及 EAR 数值。Draw landmarks + EAR on frame."""
        out = frame_bgr.copy()
        if not landmarks.face_detected:
            cv2.putText(out, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return out

        for pts in (landmarks.right_pts, landmarks.left_pts):
            for x, y in pts.astype(int):
                cv2.circle(out, (x, y), radius, color, -1)

        # 绘制虹膜轮廓 / Draw iris outlines
        for iris_pts in (landmarks.iris_right_pts, landmarks.iris_left_pts):
            if iris_pts is not None:
                cx, cy = iris_pts[0].astype(int)
                r = int(abs(iris_pts[0, 1] - iris_pts[4, 1]))  # center-to-lower
                if r > 1:
                    cv2.circle(out, (cx, cy), r, (200, 200, 0), 1)

        cv2.putText(
            out,
            f"EAR: {landmarks.ear_avg:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        if landmarks.tmh_avg_mm != 0:
            # 暴露量解读：负 = 正常覆盖；正 = 巩膜暴露
            exp = landmarks.tmh_exposure_avg
            if exp <= -0.5:
                tmh_color = (80, 200, 80)    # 正常
            elif exp <= 0.0:
                tmh_color = (0, 165, 255)    # 临界
            else:
                tmh_color = (0, 60, 220)     # 风险
            cv2.putText(
                out,
                f"TMH-idx: {exp:+.2f}mm",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                tmh_color,
                2,
            )
        return out

    def release(self) -> None:
        """释放 MediaPipe 资源。Release MediaPipe resources."""
        self._landmarker.close()

    # ── 私有方法 / Internal ──────────────────────────────────────────────────

    @staticmethod
    def _extract_pts(
        landmarks,
        indices: list[int],
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        return np.array(
            [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in indices],
            dtype=np.float32,
        )
