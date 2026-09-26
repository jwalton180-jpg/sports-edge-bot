from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import requests

from sports_edge.models.game_scope import normalize


PLAYER_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/"
    "stats_player_week_{season}.csv"
)
SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
_HEADERS = {
    "User-Agent": "SportsEdgeReadOnly/1.0",
    "Accept": "text/csv,*/*",
}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
_KALSHI_TEAM_CODES = {
    "JAX": "JAC",
    "LA": "LAR",
    "WSH": "WAS",
}


@dataclass(frozen=True)
class NFLPlayerContext:
    player_id: str
    player_name: str
    position: str
    team: str
    opponent: str
    event_date: date
    game_title: str
    current_rows: tuple[dict, ...]
    prior_rows: tuple[dict, ...]


def _aliases(value: str | None) -> set[str]:
    n = normalize(value)
    if not n:
        return set()
    tokens = n.split()
    if tokens and tokens[-1] in _SUFFIXES:
        tokens = tokens[:-1]
    if not tokens:
        return set()
    out = {" ".join(tokens)}
    if len(tokens) >= 3:
        out.add(f"{tokens[0]} {tokens[-1]}")
    return out


def _safe_int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=4)
def player_week_rows(season: int) -> tuple[dict, ...]:
    response = requests.get(
        PLAYER_WEEK_URL.format(season=int(season)),
        headers=_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return tuple(csv.DictReader(io.StringIO(response.text)))


@lru_cache(maxsize=1)
def schedule_rows() -> tuple[dict, ...]:
    response = requests.get(SCHEDULE_URL, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    return tuple(csv.DictReader(io.StringIO(response.text)))


def nfl_season(event_date: date) -> int:
    # NFL regular season labeled by the calendar year in which it starts.
    return event_date.year if event_date.month >= 3 else event_date.year - 1


def _resolve_player(
    player_name: str,
    current_rows: tuple[dict, ...],
    prior_rows: tuple[dict, ...],
) -> tuple[str, str, str] | None:
    wanted = _aliases(player_name)
    if not wanted:
        return None

    candidates: dict[str, tuple[str, str, str]] = {}
    for row in (*current_rows, *prior_rows):
        pid = str(row.get("player_id") or "").strip()
        display = str(row.get("player_display_name") or row.get("player_name") or "").strip()
        if not pid or not display:
            continue
        aliases = _aliases(display)
        if not (wanted & aliases):
            continue
        candidates[pid] = (
            pid,
            display,
            str(row.get("position") or "").strip(),
        )

    if len(candidates) == 1:
        return next(iter(candidates.values()))

    exact = [
        item for item in candidates.values()
        if normalize(item[1]) == normalize(player_name)
    ]
    return exact[0] if len(exact) == 1 else None


def _season_rows_for_player(rows: tuple[dict, ...], player_id: str) -> list[dict]:
    out = [
        row for row in rows
        if str(row.get("player_id") or "").strip() == player_id
        and str(row.get("season_type") or "").strip().upper() in {"REG", ""}
    ]
    return sorted(out, key=lambda r: _safe_int(r.get("week")) or 0)


def _latest_team(rows: list[dict]) -> str:
    for row in reversed(rows):
        team = str(row.get("team") or "").strip()
        if team:
            return team
    return ""


def _scheduled_opponent(team: str, event_date: date, season: int) -> tuple[str, str] | None:
    matches: list[tuple[str, str]] = []
    for row in schedule_rows():
        if str(row.get("season") or "") != str(season):
            continue
        if str(row.get("game_type") or "").strip().upper() not in {"REG", "POST"}:
            continue
        if str(row.get("gameday") or "")[:10] != event_date.isoformat():
            continue
        home = str(row.get("home_team") or "").strip()
        away = str(row.get("away_team") or "").strip()
        if team == home:
            matches.append((away, f"{away} @ {home}"))
        elif team == away:
            matches.append((home, f"{away} @ {home}"))
    return matches[0] if len(matches) == 1 else None


@lru_cache(maxsize=512)
def player_context(
    player_name: str,
    event_date: date,
    event_ticker: str | None = None,
) -> NFLPlayerContext | None:
    season = nfl_season(event_date)
    try:
        current_all = player_week_rows(season)
        prior_all = player_week_rows(season - 1)
    except requests.RequestException:
        return None

    resolved = _resolve_player(player_name, current_all, prior_all)
    if resolved is None:
        return None
    player_id, canonical_name, position = resolved

    current = _season_rows_for_player(current_all, player_id)
    prior = _season_rows_for_player(prior_all, player_id)
    if not current and not prior:
        return None

    team = _latest_team(current) or _latest_team(prior)
    if not team:
        return None

    scheduled = _scheduled_opponent(team, event_date, season)
    if scheduled is None:
        return None
    opponent, game_title = scheduled

    if event_ticker:
        team_code = _KALSHI_TEAM_CODES.get(team, team)
        opp_code = _KALSHI_TEAM_CODES.get(opponent, opponent)
        ticker = str(event_ticker).upper()
        if f"{team_code}{opp_code}" not in ticker and f"{opp_code}{team_code}" not in ticker:
            return None

    return NFLPlayerContext(
        player_id=player_id,
        player_name=canonical_name,
        position=position,
        team=team,
        opponent=opponent,
        event_date=event_date,
        game_title=game_title,
        current_rows=tuple(current),
        prior_rows=tuple(prior),
    )


@lru_cache(maxsize=256)
def defensive_week_values(
    *,
    season: int,
    opponent_team: str,
    stat_key: str,
) -> tuple[float, ...]:
    """Aggregate an opponent's allowed stat by week from player-week rows.

    Passing yards/TDs are summed over passers; receiving yards are summed over
    receivers. The result is used only as a conservative, shrunk matchup factor.
    """
    try:
        rows = player_week_rows(season)
    except requests.RequestException:
        return ()

    grouped: dict[int, float] = {}
    for row in rows:
        if str(row.get("opponent_team") or "").strip() != opponent_team:
            continue
        week = _safe_int(row.get("week"))
        if week is None:
            continue
        try:
            value = float(row.get(stat_key) or 0.0)
        except (TypeError, ValueError):
            continue

        position = str(row.get("position") or "").strip().upper()
        if stat_key.startswith("passing_") and position != "QB":
            continue
        grouped[week] = grouped.get(week, 0.0) + value

    return tuple(grouped[w] for w in sorted(grouped))


@lru_cache(maxsize=32)
def league_defensive_week_values(*, season: int, stat_key: str) -> tuple[float, ...]:
    """Return team-game allowed values across the league for a matchup prior."""
    try:
        rows = player_week_rows(season)
    except requests.RequestException:
        return ()

    grouped: dict[tuple[int, str], float] = {}
    for row in rows:
        opp = str(row.get("opponent_team") or "").strip()
        week = _safe_int(row.get("week"))
        if not opp or week is None:
            continue
        try:
            value = float(row.get(stat_key) or 0.0)
        except (TypeError, ValueError):
            continue
        position = str(row.get("position") or "").strip().upper()
        if stat_key.startswith("passing_") and position != "QB":
            continue
        key = (week, opp)
        grouped[key] = grouped.get(key, 0.0) + value
    return tuple(grouped.values())
