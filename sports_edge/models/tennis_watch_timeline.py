"""Chronological presentation of immutable Tennis reversal observations.

This is an explanatory watchboard, not a betting/entry engine. No alert is
described as current solely because a historical signal happened recently.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite


EARLY_LANES = frozenset({
    "EARLY WATCH — RESEARCH",
    "RECOVERY BUILDING — RESEARCH",
    "REVERSAL SIGNAL",
    "DEEP REVERSAL",
})
KNOWN_LANES = EARLY_LANES | {"WATCH"}


def _utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (TypeError, ValueError, OverflowError):
        return None


def _num(value) -> float | None:
    try:
        n = float(value)
        return n if isfinite(n) else None
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class TennisWatchTimeline:
    event_id: str
    ticker: str
    selection: str
    first_seen: datetime
    latest_seen: datetime
    first_price: float
    latest_price: float
    latest_model_fair: float
    earliest_positive_edge_price: float | None
    earliest_positive_edge_at: datetime | None
    sequence: tuple[str, ...]
    status: str
    latest_score: str
    age_minutes: float
    positive_edge: bool
    confirmed_live: bool
    verified_grade: str | None


def build_tennis_watch_timelines(
    signals: list[dict] | tuple[dict, ...],
    *,
    now: datetime,
    live_event_ids: set[str] | frozenset[str] | None = None,
    grades: list[dict] | tuple[dict, ...] = (),
    lookback_minutes: int = 180,
    max_rows: int = 30,
) -> tuple[TennisWatchTimeline, ...]:
    """Group staged observations by exact event/ticker/selection, without hindsight.

    A WATCH with negative modeled edge is explicitly NO EDGE. Only a fresh
    research watch with positive independent fair-minus-ask is highlighted.
    LIVE status requires current structural match confirmation supplied by the
    caller; clock time alone is insufficient.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(timezone.utc)
    live = live_event_ids or set()
    grade_index = {g.get("signal_id"): g.get("outcome") for g in grades
                   if isinstance(g, dict) and g.get("signal_id")}
    grouped = defaultdict(list)
    for raw in signals:
        if not isinstance(raw, dict) or raw.get("lane") not in KNOWN_LANES:
            continue
        stamped = _utc(raw.get("first_observed_at"))
        if stamped is None:
            continue
        age = (now - stamped).total_seconds() / 60
        if age < -2 or age > lookback_minutes:
            continue
        ticker = str(raw.get("ticker") or "").strip()
        eid = str(raw.get("event_id") or "").strip()
        player = str(raw.get("selection") or "").strip()
        side = str(raw.get("side") or "YES").upper()
        price = _num(raw.get("entry_ask"))
        fair = _num(raw.get("model_fair"))
        if not eid or not ticker or not player or side not in {"YES", "NO"}:
            continue
        if price is None or fair is None or not (0 < price < 1 and 0 <= fair <= 1):
            continue
        grouped[(eid, ticker, player, side)].append((stamped, raw, price, fair))

    output = []
    for (eid, ticker, player, side), observations in grouped.items():
        observations.sort(key=lambda x: (x[0], str(x[1].get("signal_id") or "")))
        first = observations[0]
        last = observations[-1]
        age = (now - last[0]).total_seconds() / 60
        edge = last[3] - last[2]
        positive_edge = edge >= .04
        first_positive = next(
            (x for x in observations
             if (x[3] - x[2] >= .04 and x[1]["lane"] in EARLY_LANES)),
            None
        )
        event_live = eid in live
        graded = next(
            (grade_index.get(x[1].get("signal_id"))
             for x in reversed(observations)
             if x[1].get("signal_id") in grade_index),
            None,
        )
        latest_lane = last[1]["lane"]
        if graded:
            status = f"VERIFIED {graded} — HISTORICAL"
        elif not event_live:
            status = "HISTORICAL — LIVE NOT VERIFIED"
        elif age > 12:
            status = "OLDER WATCH — RECHECK PRICE"
        elif latest_lane == "WATCH" and not positive_edge:
            status = "WATCH — NO ESTIMATED EDGE"
        elif first_positive and positive_edge:
            status = "RECENT RESEARCH SIGNAL — VERIFY LIVE QUOTE"
        else:
            status = "WATCH — CHECK LIVE CONDITIONS"
        output.append(TennisWatchTimeline(
            event_id=eid, ticker=ticker, selection=player,
            first_seen=first[0], latest_seen=last[0],
            first_price=first[2], latest_price=last[2],
            latest_model_fair=last[3],
            earliest_positive_edge_price=first_positive[2] if first_positive else None,
            earliest_positive_edge_at=first_positive[0] if first_positive else None,
            sequence=tuple(str(x[1]["lane"]) for x in observations),
            status=status,
            latest_score=str(last[1].get("score_state") or ""),
            age_minutes=max(0., age),
            positive_edge=positive_edge,
            confirmed_live=event_live,
            verified_grade=str(graded) if graded else None,
        ))
    status_order = {
        "RECENT RESEARCH SIGNAL — VERIFY LIVE QUOTE": 0,
        "WATCH — CHECK LIVE CONDITIONS": 1,
        "WATCH — NO ESTIMATED EDGE": 2,
        "OLDER WATCH — RECHECK PRICE": 3,
        "HISTORICAL — LIVE NOT VERIFIED": 4,
    }
    output.sort(key=lambda x: (
        status_order.get(x.status, 5),
        -x.latest_seen.timestamp(), x.ticker
    ))
    return tuple(output[:max(0, int(max_rows))])
