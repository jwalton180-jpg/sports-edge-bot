"""Strict chronological model promotion evaluation, never retrospective auto-promotion."""
from __future__ import annotations
from math import sqrt, isfinite

def assess_holdout(rows: list[dict], *, min_games: int = 200,
                   max_brier: float = .22, max_ece: float = .06) -> dict:
    """Rows must be out-of-sample predictions with immutable earlier timestamps.

    A pass is a statistical research gate, not authorization to place bets.
    """
    clean = []
    seen = set()
    for r in rows:
        try:
            from datetime import datetime
            event = datetime.fromisoformat(r["event_time"].replace("Z","+00:00"))
            predicted = datetime.fromisoformat(r["predicted_at"].replace("Z","+00:00"))
            if event.tzinfo is None or predicted.tzinfo is None or predicted >= event:
                continue
            p = float(r["probability"])
            if not isfinite(p) or not 0 <= p <= 1 or type(r.get("won")) is not bool:
                continue
            key = (r["sport"],r["event_id"],r["outcome"])
            if key in seen or not r.get("source_id") or not r.get("model_version"):
                continue
            seen.add(key)
            clean.append((p,int(r["won"])))
        except (KeyError,TypeError,ValueError):
            continue
    n = len(clean)
    if not n:
        return {"graded":0,"brier":None,"ece":None,"accuracy":None,"qualified":False,
                "reason":"no_valid_prospective_holdout"}
    brier = sum((p-y)**2 for p,y in clean)/n
    acc = sum((p>=.5)==bool(y) for p,y in clean)/n
    bins = [[] for _ in range(10)]
    for p,y in clean:
        bins[min(9,int(p*10))].append((p,y))
    ece = sum(len(b)/n*abs(sum(p for p,_ in b)/len(b)-sum(y for _,y in b)/len(b))
              for b in bins if b)
    qualified = n >= min_games and brier <= max_brier and ece <= max_ece
    return {"graded":n,"brier":brier,"ece":ece,"accuracy":acc,"qualified":qualified,
            "reason":"holdout_gates_passed" if qualified else "insufficient_or_uncalibrated_holdout"}
