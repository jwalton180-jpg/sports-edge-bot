"""Deterministic, auditable performance calculations for research records.

This module deliberately does not create predictions, settle outcomes, or store
records. Callers must supply immutable, timestamped records and trustworthy
settlement outcomes. Unknown outcomes are excluded rather than guessed.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import mean
from typing import Iterable, Mapping, Any

VALID_RESULTS = {"win", "loss", "push", "void", "pending"}


def american_to_decimal(odds: int | float) -> float:
    """Convert American odds to decimal odds; reject zero and non-finite values."""
    value = float(odds)
    if not isfinite(value) or value == 0:
        raise ValueError("American odds must be finite and non-zero")
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))


def implied_probability(odds: int | float) -> float:
    """Raw price-implied probability (not de-vigged)."""
    return 1.0 / american_to_decimal(odds)


def net_units(result: str, decimal_odds: float, stake_units: float = 1.0) -> float:
    """Net profit in stake units, assuming ordinary single-ticket settlement."""
    if result not in VALID_RESULTS:
        raise ValueError("unknown settlement result")
    if not isfinite(float(stake_units)) or float(stake_units) <= 0:
        raise ValueError("stake_units must be positive and finite")
    if not isfinite(float(decimal_odds)) or float(decimal_odds) < 1:
        raise ValueError("decimal_odds must be finite and at least 1")
    if result == "win":
        return float(stake_units) * (float(decimal_odds) - 1.0)
    if result == "loss":
        return -float(stake_units)
    return 0.0


def summarize_performance(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize settled records; pending, push and void are not wins/losses.

    Records may use decimal_odds or odds_american. For calibration, binary
    outcome must be 1 for win and 0 for loss and probability in [0, 1] must
    have been captured before the event. No backfill is performed.
    """
    rows = list(records)
    counts = {key: 0 for key in VALID_RESULTS}
    risked = profit = 0.0
    resolved_odds: list[float] = []
    brier_terms: list[float] = []
    for row in rows:
        result = str(row.get("result", "pending")).lower()
        if result not in VALID_RESULTS:
            raise ValueError("unknown settlement result")
        counts[result] += 1
        if result in {"win", "loss"}:
            stake = float(row.get("stake_units", 1.0))
            if not isfinite(stake) or stake <= 0:
                raise ValueError("stake_units must be positive and finite")
            if row.get("decimal_odds") is not None:
                dec = float(row["decimal_odds"])
            elif row.get("odds_american") is not None:
                dec = american_to_decimal(row["odds_american"])
            else:
                raise ValueError("settled win/loss requires recorded odds")
            profit += net_units(result, dec, stake)
            risked += stake
            resolved_odds.append(dec)
            probability = row.get("probability")
            outcome = row.get("outcome", 1 if result == "win" else 0)
            if probability is not None:
                p = float(probability)
                if not isfinite(p) or not 0 <= p <= 1:
                    raise ValueError("probability must be finite and in [0, 1]")
                if outcome not in (0, 1):
                    raise ValueError("calibration outcome must be binary")
                brier_terms.append((p - int(outcome)) ** 2)

    decided = counts["win"] + counts["loss"]
    return {
        "wins": counts["win"],
        "losses": counts["loss"],
        "pushes": counts["push"],
        "voids": counts["void"],
        "pending": counts["pending"],
        "total_records": len(rows),
        "decided": decided,
        "hit_rate": counts["win"] / decided if decided else None,
        "hit_rate_denominator": decided,
        "net_units": profit,
        "risked_units": risked,
        "roi": profit / risked if risked else None,
        "average_decimal_odds": mean(resolved_odds) if resolved_odds else None,
        "brier_score": mean(brier_terms) if brier_terms else None,
        "brier_denominator": len(brier_terms),
    }


def summarize_parlays(records: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Return separate performance groups by whole-parlay leg count."""
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if str(row.get("record_type", "parlay")) != "parlay":
            continue
        count = row.get("leg_count")
        if type(count) is not int or count < 1:
            raise ValueError("parlay record requires positive integer leg_count")
        groups[count].append(row)
    return {legs: summarize_performance(group) for legs, group in sorted(groups.items())}
