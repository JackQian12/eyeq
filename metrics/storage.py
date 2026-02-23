"""
storage.py
==========
SQLite 持久化存储，使用 SQLAlchemy Core（无 ORM 依赖的轻量方案）。
SQLite persistence using SQLAlchemy Core (lightweight, no heavy ORM).

表结构 / Schema
---------------
sessions        — 每次会话记录
blink_events    — 每次眨眼事件
tear_film_log   — 每秒泪膜指标快照
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import sqlalchemy as sa
from sqlalchemy import (
    Column, Float, Integer, String, Text, Boolean,
    MetaData, Table, create_engine, insert, select
)

# ── 数据模型 (轻量 dataclass) / Data models (lightweight dataclass) ──────────

@dataclass
class Session:
    id: int
    started_at: float
    ended_at: Optional[float]
    note: str


@dataclass
class BlinkRecord:
    session_id: int
    blink_type: str           # "complete" | "incomplete" | "prolonged"
    start_time: float
    end_time: float
    duration_ms: float
    min_ear: float
    ibi_s: Optional[float]


@dataclass
class TearFilmRecord:
    session_id: int
    timestamp: float
    blink_rate_bpm: float
    incomplete_blink_ratio: float
    ibi_mean_s: float
    ibi_std_s: float
    ibi_cv: float
    estimated_nibut_s: float
    risk_score: float
    risk_level: str


# ── DDL ──────────────────────────────────────────────────────────────────────

def _build_metadata() -> MetaData:
    meta = MetaData()

    Table(
        "sessions", meta,
        Column("id",         Integer, primary_key=True, autoincrement=True),
        Column("started_at", Float,   nullable=False),
        Column("ended_at",   Float,   nullable=True),
        Column("note",       Text,    default=""),
    )

    Table(
        "blink_events", meta,
        Column("id",          Integer, primary_key=True, autoincrement=True),
        Column("session_id",  Integer, nullable=False, index=True),
        Column("blink_type",  String(16), nullable=False),
        Column("start_time",  Float,   nullable=False),
        Column("end_time",    Float,   nullable=False),
        Column("duration_ms", Float,   nullable=False),
        Column("min_ear",     Float,   nullable=False),
        Column("ibi_s",       Float,   nullable=True),
    )

    Table(
        "tear_film_log", meta,
        Column("id",                      Integer, primary_key=True, autoincrement=True),
        Column("session_id",              Integer, nullable=False, index=True),
        Column("timestamp",               Float,   nullable=False),
        Column("blink_rate_bpm",          Float,   nullable=False),
        Column("incomplete_blink_ratio",  Float,   nullable=False),
        Column("ibi_mean_s",              Float,   nullable=False),
        Column("ibi_std_s",              Float,   nullable=False),
        Column("ibi_cv",                  Float,   nullable=False),
        Column("estimated_nibut_s",       Float,   nullable=False),
        Column("risk_score",              Float,   nullable=False),
        Column("risk_level",              String(16), nullable=False),
    )

    return meta


class SessionStorage:
    """SQLite 会话存储。Thread-safe SQLite session storage."""

    def __init__(self, db_path: str = "data/eyeq_sessions.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        self._meta = _build_metadata()
        self._meta.create_all(self._engine)
        self._current_session_id: Optional[int] = None

    # ── 会话管理 / Session management ────────────────────────────────────────

    def start_session(self, note: str = "") -> int:
        """开始新会话，返回 session_id。"""
        with self._engine.begin() as conn:
            result = conn.execute(
                insert(self._meta.tables["sessions"]).values(
                    started_at=time.time(), note=note
                )
            )
            sid = result.inserted_primary_key[0]
        self._current_session_id = sid
        return sid

    def end_session(self, session_id: Optional[int] = None) -> None:
        """标记会话结束时间。"""
        sid = session_id or self._current_session_id
        if sid is None:
            return
        with self._engine.begin() as conn:
            conn.execute(
                self._meta.tables["sessions"]
                .update()
                .where(self._meta.tables["sessions"].c.id == sid)
                .values(ended_at=time.time())
            )

    # ── 数据写入 / Write ──────────────────────────────────────────────────────

    def save_blink(
        self,
        record: BlinkRecord,
        session_id: Optional[int] = None,
    ) -> None:
        sid = session_id or self._current_session_id
        if sid is None:
            raise RuntimeError("No active session. Call start_session() first.")
        with self._engine.begin() as conn:
            conn.execute(
                insert(self._meta.tables["blink_events"]).values(
                    session_id=sid,
                    blink_type=record.blink_type,
                    start_time=record.start_time,
                    end_time=record.end_time,
                    duration_ms=record.duration_ms,
                    min_ear=record.min_ear,
                    ibi_s=record.ibi_s,
                )
            )

    def save_tear_film(
        self,
        record: TearFilmRecord,
        session_id: Optional[int] = None,
    ) -> None:
        sid = session_id or self._current_session_id
        if sid is None:
            raise RuntimeError("No active session. Call start_session() first.")
        with self._engine.begin() as conn:
            conn.execute(
                insert(self._meta.tables["tear_film_log"]).values(
                    session_id=sid,
                    timestamp=record.timestamp,
                    blink_rate_bpm=record.blink_rate_bpm,
                    incomplete_blink_ratio=record.incomplete_blink_ratio,
                    ibi_mean_s=record.ibi_mean_s,
                    ibi_std_s=record.ibi_std_s,
                    ibi_cv=record.ibi_cv,
                    estimated_nibut_s=record.estimated_nibut_s,
                    risk_score=record.risk_score,
                    risk_level=record.risk_level,
                )
            )

    # ── 数据读取 / Read ───────────────────────────────────────────────────────

    def list_sessions(self) -> List[Session]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(self._meta.tables["sessions"]).order_by(
                    self._meta.tables["sessions"].c.started_at.desc()
                )
            ).fetchall()
        return [Session(id=r[0], started_at=r[1], ended_at=r[2], note=r[3]) for r in rows]

    def get_tear_film_log(self, session_id: int) -> List[TearFilmRecord]:
        t = self._meta.tables["tear_film_log"]
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(t).where(t.c.session_id == session_id)
                .order_by(t.c.timestamp)
            ).fetchall()
        return [
            TearFilmRecord(
                session_id=r[1], timestamp=r[2],
                blink_rate_bpm=r[3], incomplete_blink_ratio=r[4],
                ibi_mean_s=r[5], ibi_std_s=r[6], ibi_cv=r[7],
                estimated_nibut_s=r[8], risk_score=r[9], risk_level=r[10],
            )
            for r in rows
        ]
