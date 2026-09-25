from __future__ import annotations
from dataclasses import dataclass
from typing import Any

class KalshiResolutionError(ValueError):
    """Raised when market -> event/live identity cannot be established uniquely."""

@dataclass(frozen=True)
class KalshiMarketContext:
    market_ticker: str
    event_ticker: str
    milestone_id: str | None = None

def _event_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("events", []) if isinstance(payload, dict) else []
    return [r for r in rows if isinstance(r, dict)]

def _market_tickers(event: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for m in event.get("markets", []) or []:
        if isinstance(m, dict) and m.get("ticker"):
            out.add(str(m["ticker"]))
    return out

def resolve_event_ticker(market: dict[str, Any], events_payload: dict[str, Any]) -> str:
    ticker = str(market.get("ticker") or "").strip()
    if not ticker:
        raise KalshiResolutionError("market ticker missing")
    events = _event_rows(events_payload)
    explicit = str(market.get("event_ticker") or "").strip()
    nested = [str(e.get("event_ticker") or e.get("ticker")) for e in events if ticker in _market_tickers(e)]
    nested = [x for x in nested if x and x != "None"]
    unique_nested = sorted(set(nested))
    if len(unique_nested) > 1:
        raise KalshiResolutionError("market appears in multiple events")
    if explicit:
        if unique_nested and unique_nested[0] != explicit:
            raise KalshiResolutionError("market event metadata conflicts with nested event")
        return explicit
    if len(unique_nested) == 1:
        return unique_nested[0]
    raise KalshiResolutionError("event identity unavailable")

def _collect_milestones(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            k = str(key).lower()
            if k in {"milestone_id", "milestoneid"} and value not in (None, ""):
                out.add(str(value))
            elif k == "milestone" and isinstance(value, (str, int)):
                out.add(str(value))
            _collect_milestones(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _collect_milestones(value, out)

def resolve_unique_milestone_id(live_payload: dict[str, Any]) -> str:
    found: set[str] = set()
    _collect_milestones(live_payload, found)
    if len(found) != 1:
        why = "missing" if not found else "ambiguous"
        raise KalshiResolutionError(f"milestone identity {why}")
    return next(iter(found))

def resolve_market_context(market: dict[str, Any], events_payload: dict[str, Any], live_payload: dict[str, Any] | None = None) -> KalshiMarketContext:
    ticker = str(market.get("ticker") or "").strip()
    event_ticker = resolve_event_ticker(market, events_payload)
    milestone = resolve_unique_milestone_id(live_payload) if live_payload is not None else None
    return KalshiMarketContext(ticker, event_ticker, milestone)
