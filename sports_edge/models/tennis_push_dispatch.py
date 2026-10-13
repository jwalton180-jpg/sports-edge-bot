"""Debounced push dispatch from immutable, fresh Tennis first observations.

Publish only after independent SportsEdge live quality gates created a signal,
or when a verified 1–4c dip is deliberately labeled TRACKING ONLY. One push
per ticker per confidence stage; later stages may escalate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os

from sports_edge.data.tennis_push import NtfySettings, PushEvent, post_ntfy
from sports_edge.models.tennis_prospective_store import stable_id, utc_iso


RANK = {
    "EXTREME DIP — TRACKING ONLY": 1,
    "WATCH": 2,
    "EARLY WATCH — RESEARCH": 2,
    "RECOVERY BUILDING — RESEARCH": 3,
    "REVERSAL SIGNAL": 4,
    "DEEP REVERSAL": 5,
}


def send_tennis_pushes(
    store, *,
    now: datetime,
    extreme_observations=(),
    model_candidates=(),
    environ=None,
    session=None,
    max_per_poll: int = 5,
) -> dict:
    """Returns counters, not provider URLs/tokens, and never places orders.

    Only prior + current fresh evidence is used. Historical events are not
    backfilled as "live" when a user configures push for the first time.
    """
    env = environ if environ is not None else os.environ
    result = {"configured": False, "accepted": 0, "failed": 0,
              "eligible": 0, "errors": []}
    try:
        cfg = NtfySettings.from_environ(env)
    except ValueError:
        result["errors"].append("invalid ntfy push configuration")
        result["failed"] += 1
        return result
    if cfg is None:
        return result
    result["configured"] = True
    allow_extreme = str(env.get("SPORTSEDGE_NTFY_EXTREME_DIPS", "true")).lower() in {
        "1", "true", "yes", "on",
    }

    queue = []
    for signal in store.signals:
        lane = str(signal.get("lane") or "")
        if lane not in RANK:
            continue
        try:
            event = PushEvent(
                alert_id=stable_id("signal-push", signal["signal_id"]),
                selection=str(signal["selection"]),
                ticker=str(signal["ticker"]),
                lane=lane,
                price=float(signal["entry_ask"]),
                fair=float(signal["model_fair"]),
                score=str(signal.get("score_state") or "score unavailable"),
                observed_at=str(signal["first_observed_at"]),
            )
            if event.is_eligible(now=now, allow_extreme=allow_extreme):
                queue.append(event)
        except (TypeError, ValueError, KeyError):
            continue

    # Extremely low quotes are tracked separately from prospective scored
    # signals and should never pollute the win/loss denominator.
    if allow_extreme:
        for row in extreme_observations:
            if row.lane != "EXTREME DIP — TRACKING ONLY" or row.current_ask > .04:
                continue
            event = PushEvent(
                alert_id=stable_id("extreme-push", row.ticker),
                ticker=row.ticker,
                selection=row.selection,
                lane=row.lane,
                price=row.current_ask,
                fair=row.live_fair,
                score=row.score,
                observed_at=utc_iso(now),
            )
            if event.is_eligible(now=now, allow_extreme=True):
                queue.append(event)

    queue.sort(key=lambda item: (
        -RANK.get(item.lane, 0),
        item.price,
        item.ticker,
    ))
    seen_tickers: set[str] = set()
    for event in queue:
        if event.ticker in seen_tickers:
            continue
        if store.has_accepted_push(event.alert_id):
            continue
        if store.highest_accepted_push_rank(event.ticker) >= RANK[event.lane]:
            continue
        seen_tickers.add(event.ticker)
        if result["eligible"] >= max(0, int(max_per_poll)):
            break
        result["eligible"] += 1
        try:
            provider_id = post_ntfy(event, cfg, session=session)
            store.record_push_acceptance(
                alert_id=event.alert_id, ticker=event.ticker, lane=event.lane,
                accepted_at=now, provider_id=provider_id,
            )
            result["accepted"] += 1
        except RuntimeError as exc:
            # No exception may contain the topic name, token or message body.
            result["failed"] += 1
            result["errors"].append(str(exc))
    return result
