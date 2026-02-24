"""
dashboard/app.py
================
Streamlit 实时仪表盘。
Real-time Streamlit dashboard for EyeQ monitoring.

启动方式 / Launch:
    streamlit run dashboard/app.py

或通过 / or via:
    python main.py --mode dashboard
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from metrics.aggregator import MetricsAggregator
from metrics.storage import SessionStorage, BlinkRecord, TearFilmRecord
from vision.blink_detector import BlinkType
import yaml


# ── 页面配置 / Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="EyeQ — 眼部健康监测",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 常量 / Constants ──────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).parent.parent / "config.yaml"
RISK_COLOR = {"low": "#4CAF50", "moderate": "#FF9800", "high": "#F44336"}
RISK_LABEL = {"low": "低风险 LOW", "moderate": "中风险 MODERATE", "high": "高风险 HIGH"}


def _load_config() -> dict:
    if CFG_PATH.exists():
        with open(CFG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def _make_gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30],  "color": "#1B5E20"},
                {"range": [30, 60], "color": "#E65100"},
                {"range": [60, 100],"color": "#B71C1C"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def _ear_chart(timestamps: list, ears: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=ears,
        mode="lines", name="EAR",
        line=dict(color="#00BCD4", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,188,212,0.15)",
    ))
    fig.add_hline(y=0.20, line_dash="dash", line_color="#F44336",
                  annotation_text="眨眼阈值", annotation_position="top right")
    fig.add_hline(y=0.23, line_dash="dot", line_color="#FF9800",
                  annotation_text="不完全阈值", annotation_position="top right")
    fig.update_layout(
        title="实时 EAR (Eye Aspect Ratio)",
        xaxis_title="时间 (s)",
        yaxis_title="EAR",
        yaxis_range=[0, 0.45],
        height=220,
        margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.2)",
        font={"color": "white"},
    )
    return fig


def _ibi_hist(ibis: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ibis, nbinsx=15,
        marker_color="#7E57C2",
        opacity=0.8, name="IBI"
    ))
    fig.add_vline(x=6.0, line_dash="dash", line_color="#F44336",
                  annotation_text="高风险 6s", annotation_position="top right")
    fig.update_layout(
        title="眨眼间期分布 (IBI Distribution)",
        xaxis_title="间期 (秒)",
        yaxis_title="频次",
        height=220,
        margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.2)",
        font={"color": "white"},
    )
    return fig


def run_dashboard() -> None:
    cfg = _load_config()
    cam_cfg   = cfg.get("camera",    {})
    blink_cfg = cfg.get("blink",     {})
    tf_cfg    = cfg.get("tear_film", {})
    st_cfg    = cfg.get("storage",   {})

    st.markdown("""
        <style>
        .main {background-color: #0E1117;}
        .risk-badge {
            display:inline-block; padding:6px 18px; border-radius:20px;
            font-size:1.2rem; font-weight:700; letter-spacing:1px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("👁️ EyeQ — 智能眼镜眼部健康监测 PoC")

    # ── 侧边栏 / Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 设置 Settings")
        cam_src = st.number_input(
            "摄像头索引 Camera Index",
            value=cam_cfg.get("source", 0), min_value=0, step=1,
        )
        fps = st.slider(
            "目标帧率 Target FPS",
            min_value=15, max_value=60,
            value=cam_cfg.get("fps", 30),
        )
        window_s = st.slider(
            "评估窗口 (秒) Assessment Window",
            min_value=30, max_value=120,
            value=tf_cfg.get("rolling_window_seconds", 60),
        )
        ear_thr = st.slider(
            "EAR 闭眼阈值",
            min_value=0.10, max_value=0.30,
            value=blink_cfg.get("ear_threshold", 0.20),
            step=0.01,
        )
        st.divider()
        use_db = st.toggle("💾 保存会话 Save Session", value=True)
        start_btn = st.button("▶ 开始监测 Start", type="primary", use_container_width=True)
        stop_btn  = st.button("⏹ 停止 Stop", use_container_width=True)

    # ── 会话状态 / Session state ──────────────────────────────────────────────
    if "running"    not in st.session_state: st.session_state.running = False
    if "aggregator" not in st.session_state: st.session_state.aggregator = None
    if "storage"    not in st.session_state: st.session_state.storage = None
    if "session_id" not in st.session_state: st.session_state.session_id = None
    if "cap"        not in st.session_state: st.session_state.cap = None
    if "ear_buf"    not in st.session_state: st.session_state.ear_buf = []
    if "ts_buf"     not in st.session_state: st.session_state.ts_buf = []
    if "ibi_buf"    not in st.session_state: st.session_state.ibi_buf = []
    if "tmh_buf"    not in st.session_state: st.session_state.tmh_buf = []
    if "tf_save_ts" not in st.session_state: st.session_state.tf_save_ts = 0.0

    if start_btn and not st.session_state.running:
        agg = MetricsAggregator(
            ear_threshold=ear_thr,
            ear_incomplete_threshold=blink_cfg.get("ear_incomplete_threshold", 0.23),
            closed_frames_min=blink_cfg.get("closed_frames_min", 2),
            closed_frames_max=blink_cfg.get("closed_frames_max", 90),
            refractory_frames=blink_cfg.get("refractory_frames", 5),
            fps=float(fps),
            window_seconds=float(window_s),
            normal_blink_rate_min=tf_cfg.get("normal_blink_rate_min", 15.0),
            normal_blink_rate_max=tf_cfg.get("normal_blink_rate_max", 25.0),
            incomplete_blink_risk_threshold=tf_cfg.get(
                "incomplete_blink_risk_threshold", 0.40),
            nibut_long_risk_seconds=tf_cfg.get("nibut_long_risk_seconds", 6.0),
        )
        cap = cv2.VideoCapture(int(cam_src))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))
        cap.set(cv2.CAP_PROP_FPS, fps)

        db = None
        sid = None
        if use_db:
            db = SessionStorage(st_cfg.get("db_path", "data/eyeq_sessions.db"))
            sid = db.start_session(note="dashboard session")

        st.session_state.aggregator = agg
        st.session_state.cap        = cap
        st.session_state.storage    = db
        st.session_state.session_id = sid
        st.session_state.running    = True
        st.session_state.ear_buf    = []
        st.session_state.ts_buf     = []
        st.session_state.ibi_buf    = []
        st.session_state.tmh_buf    = []

    if stop_btn and st.session_state.running:
        if st.session_state.cap:
            st.session_state.cap.release()
        if st.session_state.aggregator:
            st.session_state.aggregator.release()
        if st.session_state.storage:
            st.session_state.storage.end_session(st.session_state.session_id)
        st.session_state.running = False

    # ── 主显示区 / Main display ───────────────────────────────────────────────
    if not st.session_state.running:
        st.info("👈 在侧边栏点击 **▶ 开始监测** 启动摄像头。\n"
                "Click **▶ Start** in the sidebar to begin monitoring.")
        return

    # 帧采集
    cap: cv2.VideoCapture = st.session_state.cap
    agg: MetricsAggregator = st.session_state.aggregator

    ret, frame = cap.read()
    if not ret:
        st.error("❌ 无法读取摄像头帧。Cannot read camera frame.")
        st.session_state.running = False
        return

    snap, metrics = agg.process_frame(frame)

    # 更新缓冲
    t0 = st.session_state.ts_buf[0] if st.session_state.ts_buf else snap.timestamp
    st.session_state.ts_buf.append(snap.timestamp - t0)
    st.session_state.ear_buf.append(snap.ear)

    # 更新 TMH 缓冲
    if metrics and metrics.tmh_avg_mm != 0:
        st.session_state.tmh_buf.append({
            "t": snap.timestamp - t0,
            "v": metrics.tmh_avg_mm,
            "s": metrics.tmh_status,
        })
        st.session_state.tmh_buf = st.session_state.tmh_buf[-int(fps * 120):]

    # 裁剪缓冲区到最近 120s
    max_buf = int(fps * 120)
    st.session_state.ts_buf  = st.session_state.ts_buf[-max_buf:]
    st.session_state.ear_buf = st.session_state.ear_buf[-max_buf:]

    if snap.blink_event:
        ev = snap.blink_event
        # 持久化眨眼
        if st.session_state.storage and st.session_state.session_id:
            rec = BlinkRecord(
                session_id=st.session_state.session_id,
                blink_type=ev.blink_type.name.lower(),
                start_time=ev.start_time,
                end_time=ev.end_time,
                duration_ms=ev.duration_ms,
                min_ear=ev.min_ear,
                ibi_s=ev.inter_blink_interval,
            )
            st.session_state.storage.save_blink(rec)
        # 更新 IBI 缓冲
        if ev.inter_blink_interval:
            st.session_state.ibi_buf.append(ev.inter_blink_interval)
            st.session_state.ibi_buf = st.session_state.ibi_buf[-200:]

    # 持久化泪膜日志 (每 5s 一次)
    now = time.time()
    if (
        metrics
        and st.session_state.storage
        and st.session_state.session_id
        and (now - st.session_state.tf_save_ts) >= 5.0
    ):
        tfrec = TearFilmRecord(
            session_id=st.session_state.session_id,
            timestamp=metrics.timestamp,
            blink_rate_bpm=metrics.blink_rate_bpm,
            incomplete_blink_ratio=metrics.incomplete_blink_ratio,
            ibi_mean_s=metrics.ibi_mean_s,
            ibi_std_s=metrics.ibi_std_s,
            ibi_cv=metrics.ibi_cv,
            estimated_nibut_s=metrics.estimated_nibut_s,
            risk_score=metrics.risk_score,
            risk_level=metrics.risk_level,
        )
        st.session_state.storage.save_tear_film(tfrec)
        st.session_state.tf_save_ts = now

    # ── 绘制关键点并显示视频帧 (复用 last_lm，避免重复处理)
    lm = agg.last_lm or agg.tracker.process_frame(frame)
    annotated = agg.draw_overlay(frame, lm)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # ── 布局 / Layout ─────────────────────────────────────────────────────────
    top_left, top_right = st.columns([1, 2])

    with top_left:
        st.image(annotated_rgb, channels="RGB", use_container_width=True)
        st.caption(f"会话时长: {agg.session_duration_s:.0f}s  |  "
                   f"总眨眼: {agg.detector.total_blinks} 次")

    with top_right:
        if metrics:
            risk_color = RISK_COLOR.get(metrics.risk_level, "#9E9E9E")
            risk_label = RISK_LABEL.get(metrics.risk_level, "—")

            st.markdown(
                f'<div class="risk-badge" style="background:{risk_color}20;'
                f'border:2px solid {risk_color};color:{risk_color}">'
                f'泪膜风险 {risk_label} &nbsp;|&nbsp; {metrics.risk_score:.0f}/100</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("眨眼频率 BR", f"{metrics.blink_rate_bpm:.1f} /min",
                      delta="正常" if 15 <= metrics.blink_rate_bpm <= 25 else "异常")
            m2.metric("不完全眨眼 IBR", f"{metrics.incomplete_blink_ratio*100:.0f}%")
            m3.metric("估算 NIBUT", f"{metrics.estimated_nibut_s:.1f} s")
            m4.metric("IBI 变异 CV", f"{metrics.ibi_cv:.2f}")

            # TMH 代理指标徽章
            if metrics.tmh_avg_mm != 0:
                _TMH_COLOR = {
                    "normal":     "#4CAF50",
                    "borderline": "#FF9800",
                    "low":        "#F44336",
                    "unknown":    "#9E9E9E",
                }
                _TMH_LABEL = {
                    "normal":     "✓ 正常（眼睑覆盖充分）",
                    "borderline": "⚠ 临界（巩膜少量暴露）",
                    "low":        "⚠ 偏高风险（巩膜显著暴露）",
                    "unknown":    "—",
                }
                tc = _TMH_COLOR.get(metrics.tmh_status, "#9E9E9E")
                tl = _TMH_LABEL.get(metrics.tmh_status, metrics.tmh_status)
                st.markdown(
                    f'<div style="background:{tc}18;border:1.5px solid {tc};'
                    f'border-radius:8px;padding:8px 14px;margin-top:6px;color:{tc};'
                    f'font-size:0.95em">'
                    f'💧 <b>泪河区位置代理指标 TMH-idx</b>: '
                    f'{metrics.tmh_avg_mm:+.2f} mm &nbsp;—&nbsp; {tl}'
                    f'<span style="font-size:0.78em;opacity:0.7;margin-left:10px">'
                    f'（≤ 0 mm = 正常覆盖；> 0 mm = 巩膜暴露）</span></div>',
                    unsafe_allow_html=True,
                )

            st.plotly_chart(_make_gauge(metrics.risk_score, "综合风险评分", risk_color),
                            use_container_width=True)
        else:
            st.info("⏳ 数据积累中，至少需要 3 次完整眨眼…\n"
                    "Collecting data — need at least 3 blinks.")

    st.divider()
    col_ear, col_ibi = st.columns(2)

    with col_ear:
        if len(st.session_state.ear_buf) > 10:
            st.plotly_chart(
                _ear_chart(st.session_state.ts_buf, st.session_state.ear_buf),
                use_container_width=True,
            )

    with col_ibi:
        if len(st.session_state.ibi_buf) > 3:
            st.plotly_chart(
                _ibi_hist(st.session_state.ibi_buf),
                use_container_width=True,
            )

    # TMH 趋势图
    if len(st.session_state.tmh_buf) > 5:
        _ts = [r["t"] for r in st.session_state.tmh_buf]
        _vs = [r["v"] for r in st.session_state.tmh_buf]
        fig_tmh = go.Figure()
        fig_tmh.add_trace(go.Scatter(
            x=_ts, y=_vs,
            mode="lines",
            name="TMH-idx",
            line=dict(color="#00BCD4", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0,188,212,0.12)",
        ))
        fig_tmh.add_hline(y=0, line_dash="dash", line_color="#4CAF50",
                          annotation_text="正常覆盖线", annotation_position="top right")
        fig_tmh.add_hline(y=1.0, line_dash="dot", line_color="#F44336",
                          annotation_text="巩膜暴露阈值", annotation_position="top right")
        fig_tmh.update_layout(
            title="泪河区位置代理指标趋势 (TMH-idx)",
            xaxis_title="时间 (s)",
            yaxis_title="下角膜缘暴露量 (mm)",
            height=220,
            margin=dict(l=40, r=20, t=40, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
        )
        st.plotly_chart(fig_tmh, use_container_width=True)

    if metrics:
        with st.expander("📊 详细指标 Detailed Metrics"):
            sub_scores = {
                "眨眼频率评分 BR Score":     metrics.score_blink_rate,
                "不完全眨眼评分 IBR Score":  metrics.score_ibr,
                "IBI 时长评分 IBI Score":    metrics.score_ibi,
                "IBI 变异评分 CV Score":     metrics.score_ibi_cv,
            }
            df = pd.DataFrame([sub_scores]).T.rename(columns={0: "评分 (0–25)"})
            df["评分 (0–25)"] = df["评分 (0–25)"].round(2)
            st.dataframe(df, use_container_width=True)

            st.json({
                "完全眨眼 complete_blinks":     len(agg.detector.complete_blinks),
                "不完全眨眼 incomplete_blinks":  len(agg.detector.incomplete_blinks),
                "窗口内眨眼 window_blinks":      metrics.n_blinks_in_window,
                "IBI均值 ibi_mean_s":            round(metrics.ibi_mean_s, 3),
                "IBI标准差 ibi_std_s":           round(metrics.ibi_std_s, 3),
                "TMH代理 tmh_idx_mm":            round(metrics.tmh_avg_mm, 3),
                "TMH状态 tmh_status":            metrics.tmh_status,
            })

    # 持续刷新
    time.sleep(1.0 / fps)
    st.rerun()


if __name__ == "__main__":
    run_dashboard()
