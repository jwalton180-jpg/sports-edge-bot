"""Append-only, read-only-to-markets prospective evidence and reproducible exports."""
from __future__ import annotations
import csv
import hashlib
import io
import json
from datetime import datetime, timezone

FIELDS = ("id", "observed_at", "sport", "ticker", "side", "outcome",
          "model", "model_probability", "market_ask", "status",
          "source_id", "source_as_of", "settled_at", "settled_win")

def record(*, sport: str, ticker: str, side: str, outcome: str,
           model: str, probability: float | None, ask: float | None,
           source_id: str, source_as_of: str, observed_at: str,
           status: str = "UNVALIDATED_RESEARCH") -> dict:
    timestamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    source_time = datetime.fromisoformat(source_as_of.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or source_time.tzinfo is None or source_time > timestamp:
        raise ValueError("source must be timestamped before the prediction")
    if side not in ("YES", "NO") or not ticker or not source_id:
        raise ValueError("missing immutable evidence identity")
    if probability is not None and (not isinstance(probability, (int, float)) or not __import__("math").isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("invalid probability")
    if ask is not None and (not isinstance(ask, (int, float)) or not __import__("math").isfinite(ask) or not 0 < ask < 1):
        raise ValueError("invalid ask")
    key = json.dumps([sport, ticker, side, observed_at, model, source_id], separators=(",", ":"))
    return dict(zip(FIELDS, (hashlib.sha256(key.encode()).hexdigest()[:24],
                            observed_at, sport, ticker, side, outcome, model,
                            probability, ask, status, source_id, source_as_of, None, None)))

def settle(row: dict, *, settled_at: str, won: bool) -> dict:
    if row.get("settled_at") is not None or row.get("settled_win") is not None:
        raise ValueError("immutable settlement already recorded")
    settlement_time = datetime.fromisoformat(settled_at.replace("Z", "+00:00"))
    observation_time = datetime.fromisoformat(row["observed_at"].replace("Z", "+00:00"))
    if settlement_time.tzinfo is None or observation_time.tzinfo is None or settlement_time <= observation_time:
        raise ValueError("settlement must follow prediction")
    if not isinstance(won, bool):
        raise ValueError("settlement outcome must be boolean")
    return {**row, "settled_at": settled_at, "settled_win": won}

def export_jsonl(rows) -> str:
    return "".join(json.dumps({k: row.get(k) for k in FIELDS}, sort_keys=True) + "\n" for row in rows)

def export_csv(rows) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()

def evaluate(rows: list[dict]) -> dict:
    graded = [r for r in rows if isinstance(r.get("settled_win"), bool)
              and isinstance(r.get("model_probability"), (int, float))
              and __import__("math").isfinite(r["model_probability"])
              and 0 <= r["model_probability"] <= 1
              and r.get("settled_at") is not None]
    if not graded:
        return {"graded": 0, "brier": None, "accuracy": None, "qualified": False}
    brier = sum((float(r["model_probability"]) - int(r["settled_win"])) ** 2 for r in graded) / len(graded)
    accuracy = sum((float(r["model_probability"]) >= .5) == bool(r["settled_win"]) for r in graded) / len(graded)
    return {"graded": len(graded), "brier": brier, "accuracy": accuracy,
            "qualified": False}  # Requires preregistered holdout, calibration and coverage.
