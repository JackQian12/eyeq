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
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

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
    封装 MediaPipe Face Mesh，逐帧输出双眼 EAR 及关键点。
    Wraps MediaPipe Face Mesh; yields per-frame EAR and landmarks.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
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
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        self._frame_idx += 1

        if not results.multi_face_landmarks:
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

        lm = results.multi_face_landmarks[0].landmark

        right_pts = self._extract_pts(lm, _RIGHT_EYE_IDX, w, h)
        left_pts  = self._extract_pts(lm, _LEFT_EYE_IDX,  w, h)

        ear_r = _ear(right_pts)
        ear_l = _ear(left_pts)
        ear_avg = (ear_r + ear_l) / 2.0

        return EyeLandmarks(
            timestamp=ts,
            frame_index=self._frame_idx,
            ear_right=ear_r,
            ear_left=ear_l,
            ear_avg=ear_avg,
            right_pts=right_pts,
            left_pts=left_pts,
            face_detected=True,
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

        cv2.putText(
            out,
            f"EAR: {landmarks.ear_avg:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        return out

    def release(self) -> None:
        """释放 MediaPipe 资源。Release MediaPipe resources."""
        self._face_mesh.close()

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
