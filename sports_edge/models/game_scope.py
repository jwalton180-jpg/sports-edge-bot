from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import unicodedata
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
    raw = (value or "").lower()
    raw = raw.translate(str.maketrans({
        "ł": "l", "ø": "o", "đ": "d", "ð": "d", "þ": "th",
        "æ": "ae", "œ": "oe", "ß": "ss",
    }))
    raw = unicodedata.normalize("NFKD", raw)
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", raw))


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
            "series_title",
            "series_category",
            "series_tags",
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
    """Map sportsbook games to exact Kalshi event markets.

    Some Kalshi match-winner contracts name only the YES participant in each
    individual market. Combine sibling markets sharing event_ticker before
    applying the strict two-participant gate; this recovers legitimate Tennis
    matches without relaxing cross-event identity.
    """
    result: dict[str, list[dict]] = {g.event_id: [] for g in games}

    event_contexts: dict[str, str] = {}
    for market in markets:
        event_ticker = str(market.get("event_ticker") or "").strip()
        if not event_ticker:
            continue
        context = market_context(market)
        if context:
            event_contexts[event_ticker] = (event_contexts.get(event_ticker, "") + " " + context).strip()

    for game in games:
        matched: list[dict] = []
        for market in markets:
            if market_matches_game(market, game):
                matched.append(market)
                continue

            event_ticker = str(market.get("event_ticker") or "").strip()
            combined = event_contexts.get(event_ticker, "") if event_ticker else ""
            if combined and _contains_team(combined, game.home_team) and _contains_team(combined, game.away_team):
                matched.append(market)

        result[game.event_id] = matched
    return result


FUTURE_TERMS = (
    "before 20",
    "by 20",
    "season wins",
    "win total",
    "win totals",
    "division winner",
    "conference winner",
    "playoff qualifier",
    "playoff qualification",
    "playoff seed",
    "season home runs",
    "season record",
    "season stats",
    "best record",
    "worst record",
    "league leader",
    "stat leader",
    "mvp",
    "rookie of the year",
    "player of the year",
    "coach of the year",
    "defensive player of the year",
    "offensive player of the year",
    "award",
    "draft pick",
    "drafted",
    "next team",
    "next club",
    "retirement",
    "hall of fame",
    "series winner",
    "series exact",
    "series total games",
    "championship series score",
    "tournament winner",
    "win tournament",
    "stage qualifiers",
    "stage of tournament",
    "round of elimination",
    "qualify for atp finals",
    "qualify for wta finals",
    "ranked player",
    "player to compete",
    "player to return",
    "most wins",
    "highest win total",
    "exact wins",
    "all star selections",
    "draft top",
)


def looks_like_future(market: dict) -> bool:
    text = normalize(market_context(market))
    if any(term in text for term in FUTURE_TERMS):
        return True
    patterns = (
        r"\b(?:american|national) league (?:east|west|central) winner\b",
        r"\b(?:american|national) football conference (?:east|west|north|south) winner\b",
        r"\bwill .+ win .+ tournament\b",
        r"\bwill .+ qualify for (?:atp|wta) finals\b",
        r"\bwill .+ reach (?:round|stage)\b",
        r"\bwill .+ retire\b",
        r"\bwill .+ be drafted\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)
