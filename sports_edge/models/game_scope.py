from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any


@dataclass(frozen=True)
class GameEvent:
    event_id: str
    sport_key: str
    sport: str
    home_team: str
    away_team: str
    commence_time: datetime
    state: str


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def normalize(value: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def team_aliases(name: str) -> tuple[str, ...]:
    n = normalize(name)
    if not n:
        return ()
    parts = n.split()
    aliases = {n}
    if len(parts) >= 2:
        aliases.add(parts[-1])
    if len(parts) >= 3:
        aliases.add(" ".join(parts[-2:]))
    return tuple(sorted((a for a in aliases if len(a) >= 4), key=len, reverse=True))


def _contains_team(context: str, team: str) -> bool:
    hay = normalize(context)
    for alias in team_aliases(team):
        if re.search(rf"\b{re.escape(alias)}\b", hay):
            return True
    return False


def market_context(market: dict) -> str:
    return " ".join(
        str(market.get(k) or "")
        for k in (
            "title",
            "subtitle",
            "event_title",
            "yes_sub_title",
            "no_sub_title",
            "ticker",
            "event_ticker",
            "series_ticker",
        )
    )


def market_matches_game(market: dict, game: GameEvent) -> bool:
    """Strict game identity gate: both participants must be present."""
    context = market_context(market)
    return _contains_team(context, game.home_team) and _contains_team(context, game.away_team)


def event_state(commence_time: datetime, now: datetime | None = None) -> str:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    delta = (commence_time - now).total_seconds()
    if delta <= 0:
        return "LIVE"
    if delta <= 6 * 3600:
        return "SOON"
    return "UPCOMING"


def build_game_events(
    raw_events: list[dict],
    *,
    sport_key: str,
    sport: str,
    now: datetime | None = None,
    past_hours: float = 8.0,
    future_hours: float = 72.0,
) -> list[GameEvent]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    out: list[GameEvent] = []
    for event in raw_events:
        start = _parse_iso(event.get("commence_time"))
        home = str(event.get("home_team") or "").strip()
        away = str(event.get("away_team") or "").strip()
        event_id = str(event.get("id") or "").strip()
        if not start or not home or not away or not event_id:
            continue
        age_h = (now - start).total_seconds() / 3600.0
        lead_h = (start - now).total_seconds() / 3600.0
        if age_h > past_hours or lead_h > future_hours:
            continue
        out.append(
            GameEvent(
                event_id=event_id,
                sport_key=sport_key,
                sport=sport,
                home_team=home,
                away_team=away,
                commence_time=start,
                state=event_state(start, now),
            )
        )
    return sorted(out, key=lambda g: g.commence_time)


def game_scoped_markets(markets: list[dict], games: list[GameEvent]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {g.event_id: [] for g in games}
    for game in games:
        result[game.event_id] = [m for m in markets if market_matches_game(m, game)]
    return result


FUTURE_TERMS = (
    "championship",
    "super bowl",
    "world series",
    "before 20",
    "by 20",
    "season wins",
    "win the division",
    "win the conference",
    "mvp",
    "award",
)


def looks_like_future(market: dict) -> bool:
    text = normalize(market_context(market))
    return any(term in text for term in FUTURE_TERMS)
