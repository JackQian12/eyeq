"""
main.py
=======
EyeQ PoC 入口点。两种运行模式：
EyeQ PoC entry point. Two run modes:

    python main.py --mode headless   # 纯命令行，输出指标到终端
    python main.py --mode dashboard  # 启动 Streamlit 仪表盘

默认 mode = headless。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import yaml
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# 确保包路径正确
sys.path.insert(0, str(Path(__file__).parent))

from metrics.aggregator import MetricsAggregator
from metrics.storage import SessionStorage, BlinkRecord, TearFilmRecord
from vision.blink_detector import BlinkType

console = Console()

CFG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    if CFG_PATH.exists():
        with open(CFG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def _build_aggregator(cfg: dict) -> MetricsAggregator:
    cam_cfg   = cfg.get("camera",    {})
    blink_cfg = cfg.get("blink",     {})
    tf_cfg    = cfg.get("tear_film", {})
    return MetricsAggregator(
        ear_threshold=blink_cfg.get("ear_threshold", 0.20),
        ear_incomplete_threshold=blink_cfg.get("ear_incomplete_threshold", 0.23),
        closed_frames_min=blink_cfg.get("closed_frames_min", 2),
        closed_frames_max=blink_cfg.get("closed_frames_max", 90),
        refractory_frames=blink_cfg.get("refractory_frames", 5),
        fps=float(cam_cfg.get("fps", 30)),
        window_seconds=float(tf_cfg.get("rolling_window_seconds", 60)),
        normal_blink_rate_min=tf_cfg.get("normal_blink_rate_min", 15.0),
        normal_blink_rate_max=tf_cfg.get("normal_blink_rate_max", 25.0),
        incomplete_blink_risk_threshold=tf_cfg.get(
            "incomplete_blink_risk_threshold", 0.40),
        nibut_long_risk_seconds=tf_cfg.get("nibut_long_risk_seconds", 6.0),
    )


def _risk_color(level: str) -> str:
    return {"low": "green", "moderate": "yellow", "high": "red"}.get(level, "white")


def _build_status_table(agg: MetricsAggregator) -> Table:
    """构建 Rich 控制台状态表格。"""
    metrics = agg.latest_metrics
    table = Table(
        title="EyeQ 实时监测 Real-time Monitor",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("指标 Metric",      style="cyan", width=30)
    table.add_column("当前值 Value",     style="white", width=20)
    table.add_column("参考范围 Ref",     style="dim",   width=25)
    table.add_column("状态 Status",      width=15)

    table.add_row(
        "会话时长 Session Duration",
        f"{agg.session_duration_s:.0f}s",
        "—", "—"
    )
    table.add_row(
        "总眨眼数 Total Blinks",
        str(agg.detector.total_blinks),
        "—", "—"
    )

    if metrics:
        def _status(ok: bool) -> Text:
            return Text("✓ 正常", style="green") if ok else Text("⚠ 异常", style="yellow")

        table.add_row(
            "眨眼频率 Blink Rate",
            f"{metrics.blink_rate_bpm:.1f} /min",
            "15–25 /min",
            _status(15 <= metrics.blink_rate_bpm <= 25),
        )
        table.add_row(
            "不完全眨眼比 IBR",
            f"{metrics.incomplete_blink_ratio * 100:.0f}%",
            "< 40%",
            _status(metrics.incomplete_blink_ratio < 0.40),
        )
        table.add_row(
            "估算 NIBUT",
            f"{metrics.estimated_nibut_s:.1f}s",
            "3–6s",
            _status(3.0 <= metrics.estimated_nibut_s <= 6.0),
        )
        table.add_row(
            "IBI 变异系数 CV",
            f"{metrics.ibi_cv:.2f}",
            "< 0.5",
            _status(metrics.ibi_cv < 0.5),
        )
        # 泪河高度 TMH 代理指标
        if metrics.tmh_avg_mm != 0:
            tmh_style = {
                "normal":     "green",
                "borderline": "yellow",
                "low":        "red",
            }.get(metrics.tmh_status, "white")
            tmh_label = {
                "normal":     "✓ 正常 (眼睑覆盖充分)",
                "borderline": "⚠ 临界 (巩膜少量暴露)",
                "low":        "⚠ 偏高风险 (巩膜显著暴露)",
            }.get(metrics.tmh_status, metrics.tmh_status)
            table.add_row(
                "泪河区位置 TMH代理",
                f"{metrics.tmh_avg_mm:+.2f} mm",
                "≤ 0 mm (正常覆盖)",
                Text(tmh_label, style=tmh_style),
            )
        else:
            table.add_row("泪河区位置 TMH代理", "检测中…", "≤ 0 mm (正常覆盖)", "—")
        risk_txt = Text(
            f"{metrics.risk_level.upper()} ({metrics.risk_score:.0f}/100)",
            style=_risk_color(metrics.risk_level) + " bold",
        )
        table.add_row("综合风险 Risk Score", "", "", risk_txt)
    else:
        table.add_row("⏳ 数据积累中…", "需至少 3 次眨眼", "—", "—")

    return table


def run_headless(cfg: dict) -> None:
    """命令行无头模式：摄像头捕获 + 终端实时输出。"""
    cam_cfg = cfg.get("camera", {})
    st_cfg  = cfg.get("storage", {})

    agg = _build_aggregator(cfg)
    cap = cv2.VideoCapture(int(cam_cfg.get("source", 0)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))
    cap.set(cv2.CAP_PROP_FPS,          cam_cfg.get("fps", 30))

    storage = SessionStorage(st_cfg.get("db_path", "data/eyeq_sessions.db"))
    sid = storage.start_session(note="headless session")

    tf_save_ts = 0.0

    console.print(Panel(
        "[bold cyan]EyeQ PoC — Headless Mode[/bold cyan]\n"
        "按 [yellow]Ctrl+C[/yellow] 停止。Press Ctrl+C to stop.",
        border_style="cyan",
    ))

    try:
        with Live(console=console, refresh_per_second=2) as live:
            while True:
                ret, frame = cap.read()
                if not ret:
                    console.print("[red]无法读取摄像头帧。[/red]")
                    break

                snap, metrics = agg.process_frame(frame)

                # 持久化眨眼事件
                if snap.blink_event:
                    ev = snap.blink_event
                    storage.save_blink(BlinkRecord(
                        session_id=sid,
                        blink_type=ev.blink_type.name.lower(),
                        start_time=ev.start_time,
                        end_time=ev.end_time,
                        duration_ms=ev.duration_ms,
                        min_ear=ev.min_ear,
                        ibi_s=ev.inter_blink_interval,
                    ))

                now = time.time()
                if metrics and (now - tf_save_ts) >= 5.0:
                    storage.save_tear_film(TearFilmRecord(
                        session_id=sid,
                        timestamp=metrics.timestamp,
                        blink_rate_bpm=metrics.blink_rate_bpm,
                        incomplete_blink_ratio=metrics.incomplete_blink_ratio,
                        ibi_mean_s=metrics.ibi_mean_s,
                        ibi_std_s=metrics.ibi_std_s,
                        ibi_cv=metrics.ibi_cv,
                        estimated_nibut_s=metrics.estimated_nibut_s,
                        risk_score=metrics.risk_score,
                        risk_level=metrics.risk_level,
                    ))
                    tf_save_ts = now

                live.update(_build_status_table(agg))

    except KeyboardInterrupt:
        console.print("\n[yellow]监测已停止。[/yellow]")
    finally:
        cap.release()
        agg.release()
        storage.end_session(sid)
        console.print(f"[green]会话 #{sid} 已保存。Session #{sid} saved.[/green]")


def run_dashboard_mode() -> None:
    """启动 Streamlit 仪表盘子进程。"""
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    console.print(Panel(
        "[bold cyan]启动 Streamlit 仪表盘…[/bold cyan]\n"
        f"[dim]{dashboard_path}[/dim]",
        border_style="cyan",
    ))
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
         "--server.headless", "true"],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EyeQ — 智能眼镜眨眼/泪膜监测 PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式 / Run modes:
  headless   纯命令行输出实时指标 (默认)
  dashboard  启动 Streamlit 可视化仪表盘
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["headless", "dashboard"],
        default="headless",
        help="运行模式 (默认: headless)",
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.mode == "dashboard":
        run_dashboard_mode()
    else:
        run_headless(cfg)


if __name__ == "__main__":
    main()
