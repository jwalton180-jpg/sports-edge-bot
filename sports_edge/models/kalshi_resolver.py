from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SportsLiveBinding:
    event_ticker: str
    milestone_id: str
    milestone_type: str
    title: str
    source: str


def _event_ticker(market: dict[str, Any], event: dict[str, Any] | None) -> str:
    market_event = str(market.get("event_ticker") or "").strip()
    event_ticker = str((event or {}).get("event_ticker") or (event or {}).get("ticker") or "").strip()
    if market_event and event_ticker and market_event != event_ticker:
        return ""
    return market_event or event_ticker


def _linked(milestone: dict[str, Any], event_ticker: str) -> bool:
    primary = {str(x) for x in (milestone.get("primary_event_tickers") or [])}
    related = {str(x) for x in (milestone.get("related_event_tickers") or [])}
    return event_ticker in primary or event_ticker in related


def resolve_sports_live_binding(
    market: dict[str, Any],
    *,
    event: dict[str, Any] | None = None,
    milestones: list[dict[str, Any]] | None = None,
) -> SportsLiveBinding | None:
    """Resolve a market to one explicit live-data milestone; never guess."""
    event_ticker = _event_ticker(market, event)
    if not event_ticker:
        return None

    candidates: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for source, rows in (("event", (event or {}).get("milestones") or []), ("lookup", milestones or [])):
        for row in rows:
            if not isinstance(row, dict) or not _linked(row, event_ticker):
                continue
            milestone_id = str(row.get("id") or "").strip()
            if not milestone_id:
                continue
            candidates[milestone_id] = row
            sources[milestone_id] = source if milestone_id not in sources else "event+lookup"

    if len(candidates) != 1:
        return None

    milestone_id, milestone = next(iter(candidates.items()))
    return SportsLiveBinding(
        event_ticker=event_ticker,
        milestone_id=milestone_id,
        milestone_type=str(milestone.get("type") or ""),
        title=str(milestone.get("title") or ""),
        source=sources[milestone_id],
    )
