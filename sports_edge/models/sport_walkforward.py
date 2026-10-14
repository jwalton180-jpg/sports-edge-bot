"""Causal historical walk-forward evaluation for independently sourced sports data.

Historical backtests are NEVER prospective evidence and NEVER promote live bets.
Only pre-event game logs and completed official results are admitted.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from math import isfinite, sqrt

from sports_edge.data.historical_sport_evidence import (
    historical_ncaaf_projection, historical_soccer_projection, historical_nhl_projection,
)

def _date(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timezone required")
    return dt.astimezone(timezone.utc)

def _validate_team_games(games: list[dict], sport: str) -> list[dict]:
    seen = set()
    out = []
    for g in games:
        if g.get("sport") != sport or g.get("source") != "ESPN":
            raise ValueError("unexpected source or sport")
        key = g["event_id"]
        if key in seen or not key or not g.get("home_id") or not g.get("away_id"):
            raise ValueError("duplicate or incomplete event")
        seen.add(key)
        if g["home_id"] == g["away_id"] or any(type(g[k]) is not int or g[k] < 0 for k in ("home_score", "away_score")):
            raise ValueError("invalid final score")
        _date(g["event_time"])
        out.append(g)
    return sorted(out, key=lambda x: (x["event_time"], x["event_id"]))

def team_walkforward(games: list[dict], *, sport: str, threshold: int = 2,
                     min_history: int = 0) -> list[dict]:
    """Every prediction uses strictly earlier games; no market prices, no future stats.

    NCAAF uses home-team win probability as the uncalibrated rating prior;
    no favorites are inferred from the actual result or sportsbook lines.
    """
    if sport not in ("NCAAF", "Soccer"):
        raise ValueError("unsupported team model")
    rows = _validate_team_games(games, sport)
    predictions = []
    for game in rows:
        before = game["event_time"]
        past = tuple(g for g in rows if _date(g["event_time"]) < _date(before))
        if len(past) < min_history:
            continue
        if sport == "Soccer":
            estimate = historical_soccer_projection(
                past, home_id=game["home_id"], away_id=game["away_id"],
                before_time=before, threshold=threshold)
            won = game["home_score"] + game["away_score"] >= threshold
            outcome = f"total_goals_{threshold}_plus"
        else:
            if game["home_score"] == game["away_score"]:
                continue
            estimate = historical_ncaaf_projection(
                past, favorite_id=game["home_id"], underdog_id=game["away_id"],
                before_time=before, favorite_home=True, upset_red_flags=())
            won = game["home_score"] > game["away_score"]
            outcome = "home_win"
        if estimate.probability is None:
            continue
        predictions.append({
            "sport": sport, "event_id": game["event_id"], "event_time": before,
            "predicted_at": max(g["event_time"] for g in past),
            "source_id": "ESPN:" + game["event_id"], "model_version": estimate.model,
            "outcome": outcome, "probability": estimate.probability, "won": won,
            "evaluation_kind": "historical_walkforward", "status": "RESEARCH_ONLY",
        })
    return predictions

def nhl_player_walkforward(log: list[dict], *, threshold: int = 2) -> list[dict]:
    """Historical NHL game logs, excluding same-day and future observations."""
    if not log:
        return []
    seen = set()
    for g in log:
        if g.get("source") != "NHL" or not g.get("game_id") or type(g.get("goals")) is not int or g["goals"] < 0:
            raise ValueError("invalid NHL game log")
        if g["game_id"] in seen:
            raise ValueError("duplicate NHL game")
        seen.add(g["game_id"])
        _date(g["date"] + "T00:00:00Z")
    ordered = sorted(log, key=lambda g: (g["date"], g["game_id"]))
    out = []
    for g in ordered:
        prior = tuple(r for r in ordered if r["date"] < g["date"])
        projection = historical_nhl_projection(prior, before_date=g["date"], threshold=threshold)
        if projection.probability is None:
            continue
        out.append({
            "sport": "NHL", "event_id": str(g["game_id"]) + ":" + str(g["player_id"]),
            "event_time": g["date"] + "T12:00:00Z",
            "predicted_at": prior[-1]["date"] + "T23:59:59Z",
            "source_id": "NHL:" + str(g["game_id"]),
            "model_version": projection.model, "outcome": f"player_goals_{threshold}_plus",
            "probability": projection.probability, "won": g["goals"] >= threshold,
            "evaluation_kind": "historical_walkforward", "status": "RESEARCH_ONLY",
        })
    return out

def calibration_report(rows: list[dict], *, bins: int = 10) -> dict:
    """Evaluation and Wilson accuracy bounds; no auto-promotion on retrospective data."""
    if bins < 2:
        raise ValueError("bins")
    if len({(r["sport"], r["event_id"], r["outcome"]) for r in rows}) != len(rows):
        raise ValueError("duplicate prediction")
    groups = defaultdict(list)
    for r in rows:
        if r.get("evaluation_kind") != "historical_walkforward" or r.get("status") != "RESEARCH_ONLY":
            raise ValueError("mixed evidence classes")
        p = r["probability"]
        if not isinstance(p, (int, float)) or not isfinite(p) or not 0 <= p <= 1 or type(r["won"]) is not bool:
            raise ValueError("invalid probability or outcome")
        if _date(r["predicted_at"]) >= _date(r["event_time"]):
            raise ValueError("noncausal prediction")
        groups[r["sport"]].append(r)
    result = {}
    for sport, group in groups.items():
        n = len(group)
        brier = sum((r["probability"] - int(r["won"]))**2 for r in group) / n
        prevalence = sum(r["won"] for r in group) / n
        baseline_brier = sum((prevalence - int(r["won"]))**2 for r in group) / n
        bucket = [[] for _ in range(bins)]
        for r in group:
            bucket[min(bins - 1, int(r["probability"] * bins))].append(r)
        ece = sum(len(b)/n * abs(sum(r["probability"] for r in b)/len(b) -
                    sum(r["won"] for r in b)/len(b)) for b in bucket if b)
        accuracy = sum((r["probability"] >= .5) == r["won"] for r in group)/n
        z = 1.96
        center = (accuracy + z*z/(2*n))/(1+z*z/n)
        half = z*sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/(1+z*z/n)
        positives = sum(r["won"] for r in group)
        predicted_yes = [r for r in group if r["probability"] >= .5]
        result[sport] = {"n": n, "positives": positives, "positive_rate": positives/n,
                         "always_no_accuracy": 1-positives/n,
                         "predicted_yes": len(predicted_yes),
                         "predicted_yes_precision": (sum(r["won"] for r in predicted_yes)/len(predicted_yes)
                                                     if predicted_yes else None),
                         "brier": brier, "baseline_brier": baseline_brier,
                         "ece": ece, "accuracy": accuracy,
                         "accuracy_wilson95": [max(0,center-half), min(1,center+half)],
                         "beats_prevalence_baseline": brier < baseline_brier,
                         "qualified_for_live": False,
                         "reason": "retrospective_walkforward_not_prospective_validation"}
    return result
