"""Durable, append-only Tennis research observations and official Kalshi grading.

The companion GitHub Actions workflow commits these PUBLIC-market-only JSONL
records to a dedicated data branch. There are no orders or user positions.
Never infer an outcome from the live score, market price or an unfinalized market.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


DATA_FILES = ("signals.jsonl", "settlements.jsonl", "snapshots.jsonl", "runs.jsonl")
SIGNAL_VERSION = "tennis-reversal-prospective-v1"
GRADED_STATES = frozenset({"settled", "finalized"})


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("a timezone-aware UTC timestamp is required")
    return value.astimezone(timezone.utc).isoformat()


def stable_id(*fields: object) -> str:
    return sha256(json.dumps(fields, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    results = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_no} is not a JSON object")
        results.append(value)
    return results


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")


def first_signal_record(
    *, event_id: str, ticker: str, selection: str, side: str,
    lane: str, price: float, fair: float, score: str,
    observed_at: datetime, model_version: str,
    candle_end_ts: int, spread_pp: float, evidence: dict,
) -> dict[str, Any]:
    if side not in {"YES", "NO"} or not event_id or not ticker or not lane:
        raise ValueError("invalid signal identity")
    if not (0 < price < 1 and 0 <= fair <= 1):
        raise ValueError("invalid price or probability")
    if not (math.isfinite(spread_pp) and 0 <= spread_pp <= 100):
        raise ValueError("invalid spread")
    return {
        "signal_id": stable_id(event_id, ticker, side, lane, model_version),
        "event_id": event_id, "ticker": ticker, "selection": selection,
        "side": side, "lane": lane, "model_version": model_version,
        "first_observed_at": utc_iso(observed_at), "candle_end_ts": int(candle_end_ts),
        "entry_ask": round(price, 5), "model_fair": round(fair, 6),
        "spread_pp": round(spread_pp, 4),
        "score_state": score, "evidence": evidence,
        "disclaimer": "observed quoted ask; not a guaranteed fill, no order was placed",
    }


def verified_kalshi_settlement(
    payload: dict, *, ticker: str, side: str, signal_id: str,
    checked_at: datetime,
) -> dict[str, Any] | None:
    """Only grade from that exact, terminal Kalshi market and its YES/NO result."""
    market = payload.get("market") if isinstance(payload, dict) else None
    if not isinstance(market, dict):
        return None
    if str(market.get("ticker") or "") != ticker:
        return None
    status = str(market.get("status") or "").lower()
    result = str(market.get("result") or "").lower()
    if status not in GRADED_STATES or result not in {"yes", "no"}:
        return None
    if side not in {"YES", "NO"}:
        return None
    won = (result == "yes") == (side == "YES")
    return {
        "signal_id": signal_id, "ticker": ticker, "side": side,
        "outcome": "WIN" if won else "LOSS",
        "kalshi_result": result, "market_status": status,
        "source": "Kalshi public GET /markets/{ticker}",
        "checked_at": utc_iso(checked_at),
        "kalshi_settled_time": market.get("settled_time") or None,
    }


class ProspectiveResearchStore:
    """Never rewrite old observation/grade lines, even when prices change."""
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.signals = read_jsonl(self.root / DATA_FILES[0])
        self.settlements = read_jsonl(self.root / DATA_FILES[1])
        self.snapshots = read_jsonl(self.root / DATA_FILES[2])
        self.runs = read_jsonl(self.root / DATA_FILES[3])
        self._signal_ids = {r["signal_id"] for r in self.signals}
        self._grade_ids = {r["signal_id"] for r in self.settlements}
        if len(self._signal_ids) != len(self.signals) or len(self._grade_ids) != len(self.settlements):
            raise ValueError("Duplicate first signals or settlement grades in immutable ledger")
        if not self._grade_ids.issubset(self._signal_ids):
            raise ValueError("Orphan grades found: settlement lacks first observation")
        self._snapshot_ids = {r["snapshot_id"] for r in self.snapshots}

    def add_signal(self, row: dict) -> bool:
        signal_id = row["signal_id"]
        if signal_id in self._signal_ids:
            return False
        append_jsonl(self.root / DATA_FILES[0], [row])
        self.signals.append(row)
        self._signal_ids.add(signal_id)
        return True

    def add_settlement(self, row: dict) -> bool:
        sid = row["signal_id"]
        if sid not in self._signal_ids:
            raise ValueError("Cannot grade an unknown signal")
        if sid in self._grade_ids:
            return False
        if row.get("outcome") not in {"WIN", "LOSS", "VOID"} or not row.get("source"):
            raise ValueError("Invalid settlement outcome or missing official source")
        append_jsonl(self.root / DATA_FILES[1], [row])
        self.settlements.append(row)
        self._grade_ids.add(sid)
        return True

    def add_snapshots(self, rows: list[dict]) -> int:
        seen = set(self._snapshot_ids)
        unseen = []
        for r in rows:
            if r["snapshot_id"] not in seen:
                unseen.append(r)
                seen.add(r["snapshot_id"])
        append_jsonl(self.root / DATA_FILES[2], unseen)
        self.snapshots.extend(unseen)
        self._snapshot_ids.update(r["snapshot_id"] for r in unseen)
        return len(unseen)

    def add_run(self, row: dict, *, hourly_only: bool = True) -> bool:
        # Keep a durable hourly heartbeat, not 288 Git commits per day in
        # markets where no candidates or settlements changed.
        hour = str(row["observed_at"])[:13]
        # A separate mutable heartbeat improves observability even when no
        # qualifying signals/snapshots change during the current hour.
        (self.root / "latest_poll.json").write_text(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        if hourly_only and any(str(r.get("observed_at", ""))[:13] == hour for r in self.runs):
            return False
        append_jsonl(self.root / DATA_FILES[3], [row])
        self.runs.append(row)
        return True

    def pending(self) -> list[dict]:
        return [r for r in self.signals if r["signal_id"] not in self._grade_ids]

    def report(self, *, min_train: int = 30) -> dict:
        grades = {r["signal_id"]: r for r in self.settlements}
        cohorts = defaultdict(list)
        for signal in self.signals:
            cohorts[signal["lane"]].append((signal, grades.get(signal["signal_id"])))
        by_lane = {}
        for lane, items in sorted(cohorts.items()):
            settled = [(s, g) for s, g in items if g and g["outcome"] in {"WIN", "LOSS"}]
            ordered = sorted(settled, key=lambda z: (z[0]["first_observed_at"], z[0]["signal_id"]))
            won = sum(g["outcome"] == "WIN" for _, g in ordered)
            n = len(ordered)
            # Observed quote benchmark is gross payout on a hypothetical single
            # full-settlement contract. Fees, slippage, fills are NOT included.
            gross = sum((1.0 if g["outcome"] == "WIN" else 0.0) - s["entry_ask"]
                        for s, g in ordered)
            brier = mean((s["model_fair"] - (1 if g["outcome"] == "WIN" else 0)) ** 2
                         for s, g in ordered) if n else None
            market_brier = mean((s["entry_ask"] - (1 if g["outcome"] == "WIN" else 0)) ** 2
                                for s, g in ordered) if n else None
            lane_metrics = {
                "observations": len(items), "settled": n, "wins": won, "losses": n-won,
                "pending": sum(g is None for _, g in items),
                "win_rate": won/n if n else None,
                "model_brier": brier, "quote_brier": market_brier,
                "hypothetical_gross_dollars_per_contract": round(gross, 4),
                "fees_fills_slippage": "unknown; not realized ROI",
                "first_half_win_rate": (
                    sum(g["outcome"] == "WIN" for _,g in ordered[:n//2])/(n//2)
                    if n>=2 else None
                ),
                "second_half_win_rate": (
                    sum(g["outcome"] == "WIN" for _,g in ordered[n//2:])/(n-n//2)
                    if n>=2 else None
                ),
                "chronological_out_of_sample_ready": n >= min_train+20,
            }
            for label,lo,hi in (
                ("1-4c", .01,.04), ("4-10c", .04,.10), ("10-15c",.10,.15),
                ("15-20c",.15,.20), ("20-25c",.20,.25),
            ):
                group = [(s,g) for s,g in ordered if lo <= s["entry_ask"] < hi or
                         (label=="20-25c" and s["entry_ask"]==hi)]
                lane_metrics[label] = {
                    "settled":len(group),
                    "wins":sum(g["outcome"]=="WIN" for _,g in group),
                }
            # Prequential evaluation: an earlier result counts as training
            # only if Kalshi's verification was already recorded BEFORE the
            # test observation. This cannot use future settlement labels.
            forward = []
            for test_signal, test_grade in ordered:
                known_prior = [
                    (train_signal, train_grade)
                    for train_signal, train_grade in ordered
                    if train_signal["event_id"] != test_signal["event_id"]
                    and train_signal["first_observed_at"] < test_signal["first_observed_at"]
                    and train_grade["checked_at"] < test_signal["first_observed_at"]
                ]
                if len(known_prior) < min_train:
                    continue
                target = int(test_grade["outcome"] == "WIN")
                forward.append((
                    (test_signal["model_fair"]-target)**2,
                    (test_signal["entry_ask"]-target)**2,
                ))
            lane_metrics["causal_forward"] = {
                "minimum_prior_verified": min_train,
                "evaluated": len(forward),
                "model_brier": round(mean(x[0] for x in forward), 6) if forward else None,
                "quote_brier": round(mean(x[1] for x in forward), 6) if forward else None,
                "sufficient_for_exploration": len(forward) >= 20,
                "trained_on_future_results": False,
            }
            by_lane[lane]=lane_metrics
        latest_file = self.root / "latest_poll.json"
        latest_poll = (
            json.loads(latest_file.read_text(encoding="utf-8"))
            if latest_file.exists() else (self.runs[-1] if self.runs else None)
        )
        research_snapshots = [
            row for row in self.snapshots
            if str(row.get("research_observation", "")).startswith("EXTREME DIP")
        ]
        report = {
            "schema_version":1,
            "extreme_dip_quote_snapshots":len(research_snapshots),
            "recovery_building_quote_snapshots":sum(
                row.get("research_observation") == "RECOVERY BUILDING — RESEARCH"
                for row in self.snapshots
            ),
            "total_observations":len(self.signals),
            "total_settled":len(self.settlements),
            "total_pending":len(self.pending()),
            "recorded_snapshots":len(self.snapshots),
            "last_worker_run":latest_poll,
            "by_lane":by_lane,
            "status":"prospective observational research, no confirmed model uplift",
            "no_hindsight":"first observations are immutable; no holdout was reopened",
        }
        return report

    def save_report(self) -> dict:
        report = self.report()
        (self.root / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8"
        )
        return report
