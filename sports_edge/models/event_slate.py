from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from statistics import median
from typing import Any

from sports_edge.core.math import american_to_implied, no_vig_two_way


FUTURES_TERMS = (
    "championship",
    "championships",
    "super bowl",
    "world series",
    "stanley cup",
    "nba finals",
    "conference",
    "division",
    "playoffs",
    "postseason",
    "mvp",
    "award",
    "rookie of the year",
    "cy young",
    "season wins",
    "regular season wins",
    "make the playoffs",
    "win the league",
    "win the title",
    "before 20",
    "by 20",
    "draft",
    "cast in",
    "feature-length",
    "movie",
    "film",
)

GAME_MARKET_TERMS = (
    "moneyline",
    "money line",
    "spread",
    "total",
    "over",
    "under",
    "1st quarter",
    "first quarter",
    "1st half",
    "first half",
    "1st 5",
    "first 5",
    "inning",
    "game",
    "match",
    "wins",
    "win",
)

PROP_TERMS = (
    "hit",
    "hits",
    "home run",
    "homer",
    "strikeout",
    "strikeouts",
    "total bases",
    "rbi",
    "run scored",
    "stolen base",
    "passing",
    "pass yards",
    "pass touchdowns",
    "rushing",
    "rush yards",
    "receiving",
    "receptions",
    "receiving yards",
    "touchdown",
    "touchdowns",
    "td",
    "sacks",
    "tackles",
)


@dataclass(frozen=True)
class EventMatch:
    event_id: str
    sport_key: str
    home_team: str
    away_team: str
    commence_time: datetime | None
    kalshi_markets: tuple[dict, ...]


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if out.tzinfo is None:
            out = out.replace(tzinfo=timezone.utc)
        return out.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def normalize(value: str | None) -> str:
    text = (value or "").lower().replace("&", " and ")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _participant_aliases(name: str) -> tuple[str, ...]:
    n = normalize(name)
    if not n:
        return ()
    parts = n.split()
    aliases = {n}
    if len(parts) >= 3:
        aliases.add(" ".join(parts[-2:]))
    if len(parts) == 2 and len(parts[0]) >= 4 and len(parts[1]) >= 4:
        aliases.add(n)
    return tuple(sorted(aliases, key=len, reverse=True))


def participant_present(name: str, text: str) -> bool:
    hay = normalize(text)
    for alias in _participant_aliases(name):
        if len(alias) >= 7 and re.search(rf"\b{re.escape(alias)}\b", hay):
            return True
    return False


def market_context(market: dict) -> str:
    return " ".join(
        str(market.get(key) or "")
        for key in (
            "title",
            "subtitle",
            "event_title",
            "yes_sub_title",
            "no_sub_title",
            "yes_title",
            "no_title",
            "ticker",
            "event_ticker",
            "series_ticker",
            "rules_primary",
        )
    )


def is_futures_or_non_game_market(market: dict) -> bool:
    text = normalize(market_context(market))
    if any(term in text for term in FUTURES_TERMS):
        return True

    # Long-dated market language is a strong futures signal.
    if re.search(r"\b(20\d{2})\b", text) and any(k in text for k in ("before", "after", "by", "through")):
        return True
    return False


def event_identity_matches(market: dict, event: dict) -> bool:
    if is_futures_or_non_game_market(market):
        return False

    home = str(event.get("home_team") or "").strip()
    away = str(event.get("away_team") or "").strip()
    if not home or not away:
        return False

    context = market_context(market)
    if not (participant_present(home, context) and participant_present(away, context)):
        return False

    commence = _dt(event.get("commence_time"))
    if commence is None:
        return True

    # If Kalshi supplies an occurrence/expiry timestamp, require it to belong
    # to the same game window. Absence of a timestamp is allowed only because
    # the two-participant identity check above is already strict.
    candidate_times = [
        _dt(market.get("occurrence_datetime")),
        _dt(market.get("expected_expiration_time")),
        _dt(market.get("close_time")),
    ]
    candidate_times = [x for x in candidate_times if x is not None]
    if not candidate_times:
        return True

    return any(abs((t - commence).total_seconds()) <= 60 * 60 * 36 for t in candidate_times)


def kalshi_markets_for_event(event: dict, markets: list[dict]) -> list[dict]:
    matched = [m for m in markets if event_identity_matches(m, event)]
    return sorted(
        matched,
        key=lambda m: float(m.get("volume_fp", m.get("volume", 0)) or 0),
        reverse=True,
    )


def classify_kalshi_market(market: dict) -> str:
    text = normalize(market_context(market))
    if any(term in text for term in PROP_TERMS):
        return "Player Prop"
    if "spread" in text:
        return "Spread"
    if any(term in text for term in ("total", "over", "under")):
        return "Total"
    if any(term in text for term in ("1st quarter", "first quarter", "1st half", "first half", "1st 5", "first 5")):
        return "Period"
    if any(term in text for term in ("win", "wins", "moneyline", "money line")):
        return "Moneyline"
    return "Game Market"


def _book_market(bookmaker: dict, key: str) -> dict | None:
    return next((m for m in bookmaker.get("markets", []) or [] if m.get("key") == key), None)


def game_line_summary(event: dict) -> dict[str, Any]:
    """Compact consensus view for h2h/spread/total from the current event payload."""
    home = str(event.get("home_team") or "")
    away = str(event.get("away_team") or "")
    h2h: dict[str, list[float]] = {home: [], away: []}
    spreads: dict[str, list[tuple[float, float]]] = {home: [], away: []}
    totals: dict[str, list[tuple[float, float]]] = {"Over": [], "Under": []}

    for book in event.get("bookmakers", []) or []:
        market = _book_market(book, "h2h")
        if market:
            outcomes = market.get("outcomes", []) or []
            if len(outcomes) == 2:
                try:
                    p1 = american_to_implied(float(outcomes[0]["price"]))
                    p2 = american_to_implied(float(outcomes[1]["price"]))
                    f1, f2 = no_vig_two_way(p1, p2)
                    h2h.setdefault(str(outcomes[0].get("name") or ""), []).append(f1)
                    h2h.setdefault(str(outcomes[1].get("name") or ""), []).append(f2)
                except Exception:
                    pass

        market = _book_market(book, "spreads")
        if market:
            for out in market.get("outcomes", []) or []:
                try:
                    spreads.setdefault(str(out.get("name") or ""), []).append((float(out.get("point")), float(out.get("price"))))
                except Exception:
                    pass

        market = _book_market(book, "totals")
        if market:
            for out in market.get("outcomes", []) or []:
                try:
                    totals.setdefault(str(out.get("name") or ""), []).append((float(out.get("point")), float(out.get("price"))))
                except Exception:
                    pass

    def med_prob(values: list[float]) -> float | None:
        return float(median(values)) if values else None

    def med_line(values: list[tuple[float, float]]) -> tuple[float, float] | None:
        if not values:
            return None
        return float(median(v[0] for v in values)), float(median(v[1] for v in values))

    return {
        "home_team": home,
        "away_team": away,
        "home_fair": med_prob(h2h.get(home, [])),
        "away_fair": med_prob(h2h.get(away, [])),
        "home_spread": med_line(spreads.get(home, [])),
        "away_spread": med_line(spreads.get(away, [])),
        "total_over": med_line(totals.get("Over", [])),
        "total_under": med_line(totals.get("Under", [])),
        "book_count": len(event.get("bookmakers", []) or []),
    }


PROP_PRESETS: dict[str, tuple[str, ...]] = {
    "MLB Hits": ("batter_hits", "batter_hits_alternate"),
    "MLB Home Runs": ("batter_home_runs", "batter_home_runs_alternate", "batter_first_home_run"),
    "MLB Strikeouts": ("pitcher_strikeouts", "pitcher_strikeouts_alternate"),
    "NFL Passing": ("player_pass_yds", "player_pass_tds", "player_pass_completions", "player_pass_attempts"),
    "NFL Rushing": ("player_rush_yds", "player_rush_attempts", "player_rush_tds"),
    "NFL Receiving": ("player_receptions", "player_reception_yds", "player_reception_tds"),
    "NFL Touchdowns": ("player_anytime_td", "player_tds", "player_rush_reception_tds"),
}


def prop_consensus_rows(payload: dict, market_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Aggregate per-event player props across books without inventing history."""
    samples: dict[tuple[str, str, float | None, str], list[float]] = {}
    books: dict[tuple[str, str, float | None, str], set[str]] = {}

    for book in payload.get("bookmakers", []) or []:
        book_key = str(book.get("key") or book.get("title") or "book")
        for market in book.get("markets", []) or []:
            key = str(market.get("key") or "")
            if key not in market_keys:
                continue
            for out in market.get("outcomes", []) or []:
                desc = str(out.get("description") or out.get("name") or "").strip()
                side = str(out.get("name") or "").strip()
                point_raw = out.get("point")
                try:
                    point = float(point_raw) if point_raw is not None else None
                    price = float(out.get("price"))
                except Exception:
                    continue
                k = (key, desc, point, side)
                samples.setdefault(k, []).append(price)
                books.setdefault(k, set()).add(book_key)

    rows = []
    for (key, desc, point, side), prices in samples.items():
        rows.append(
            {
                "Market": key,
                "Player / Outcome": desc,
                "Side": side,
                "Line": point,
                "Median odds": round(float(median(prices))),
                "Books": len(books[(key, desc, point, side)]),
            }
        )

    return sorted(rows, key=lambda r: (-r["Books"], r["Player / Outcome"], r["Market"], str(r["Side"])))
