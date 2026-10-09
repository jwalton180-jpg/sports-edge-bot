"""Immutable first-observation Tennis reversal ledger.

Designed for a configured durable SQLite volume. Local Streamlit disk is not
assumed durable. Outcomes are graded only from a verified external settlement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS tennis_reversal_signals (
    signal_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    selection TEXT NOT NULL,
    side TEXT NOT NULL,
    lane TEXT NOT NULL,
    model_version TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    entry_ask REAL NOT NULL,
    model_fair REAL NOT NULL,
    score_state TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    settlement TEXT,
    settled_at TEXT,
    settlement_source TEXT,
    CHECK(entry_ask > 0 AND entry_ask < 1),
    CHECK(model_fair >= 0 AND model_fair <= 1),
    CHECK(settlement IS NULL OR settlement IN ('WIN','LOSS','VOID')),
    UNIQUE(event_id,ticker,side,lane,model_version)
);
"""


@dataclass(frozen=True)
class SignalObservation:
    event_id: str
    ticker: str
    selection: str
    side: str
    lane: str
    model_version: str
    observed_at: datetime
    entry_ask: float
    model_fair: float
    score_state: str
    evidence: dict


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    return dt.astimezone(timezone.utc).isoformat()


def _signal_id(record: SignalObservation) -> str:
    key = (record.event_id,record.ticker,record.side,record.lane,record.model_version)
    return sha256(json.dumps(key,separators=(",",":")).encode()).hexdigest()


class TennisSignalLedger:
    def __init__(self, db_path: str | Path):
        self.path=Path(db_path)
        if str(self.path)==":memory:":
            raise ValueError("Use a persistent, operator-configured SQLite path")
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self):
        conn=sqlite3.connect(self.path,timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def record_first(self, row: SignalObservation) -> tuple[str,bool]:
        """INSERT OR IGNORE: later higher prices never overwrite first observation."""
        if row.side not in {"YES","NO"}:
            raise ValueError("Invalid contract side")
        if row.lane not in {"EARLY WATCH — RESEARCH","WATCH","REVERSAL SIGNAL","DEEP REVERSAL"}:
            raise ValueError("Invalid lane")
        if not (0<float(row.entry_ask)<1 and 0<=float(row.model_fair)<=1):
            raise ValueError("Invalid probabilities")
        signal_id=_signal_id(row)
        payload=(
            signal_id,row.event_id,row.ticker,row.selection,row.side,row.lane,
            row.model_version,_iso(row.observed_at),float(row.entry_ask),
            float(row.model_fair),row.score_state,json.dumps(row.evidence,sort_keys=True,default=str),
        )
        with self._connect() as conn:
            result=conn.execute(
                """INSERT OR IGNORE INTO tennis_reversal_signals
                (signal_id,event_id,ticker,selection,side,lane,model_version,
                 observed_at,entry_ask,model_fair,score_state,evidence_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",payload
            )
            return signal_id, bool(result.rowcount)

    def settle_verified(self, signal_id: str, outcome: str, *,source: str,settled_at: datetime) -> bool:
        """Immutable grade; caller must verify authoritative Kalshi settlement."""
        if outcome not in {"WIN","LOSS","VOID"} or not source.strip():
            raise ValueError("Authoritative outcome and source required")
        with self._connect() as conn:
            result=conn.execute(
                """UPDATE tennis_reversal_signals
                SET settlement=?,settled_at=?,settlement_source=?
                WHERE signal_id=? AND settlement IS NULL""",
                (outcome,_iso(settled_at),source,signal_id),
            )
            return bool(result.rowcount)

    def snapshot(self) -> tuple[dict,...]:
        with self._connect() as conn:
            conn.row_factory=sqlite3.Row
            return tuple(dict(r) for r in conn.execute(
                "SELECT * FROM tennis_reversal_signals ORDER BY observed_at, signal_id"
            ))

    def metrics(self) -> dict:
        """All unsettled and void observations remain visible outside W/L."""
        rows=self.snapshot()
        wins=sum(r["settlement"]=="WIN" for r in rows)
        losses=sum(r["settlement"]=="LOSS" for r in rows)
        return {
            "total":len(rows),"settled":wins+losses,
            "wins":wins,"losses":losses,
            "void":sum(r["settlement"]=="VOID" for r in rows),
            "pending":sum(r["settlement"] is None for r in rows),
            "hit_rate":wins/(wins+losses) if wins+losses else None,
            "entry_benchmark":(
                sum(r["entry_ask"] for r in rows if r["settlement"] in ("WIN","LOSS"))/(wins+losses)
                if wins+losses else None
            ),
        }
