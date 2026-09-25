from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import requests

from sports_edge.models.game_scope import normalize


PLAYER_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv"
)
TEAM_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_team/stats_team_week_{season}.csv"
)
ROSTER_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "weekly_rosters/roster_weekly_{season}.csv"
)
SCHEDULE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "schedules/games.csv"
)

_HEADERS = {
    "User-Agent": "SportsEdgeReadOnly/1.0",
    "Accept": "text/csv,text/plain,*/*",
}


@dataclass(frozen=True)
class NFLPassingContext:
    player_id: str
    player_name: str
    team: str
    opponent: str
    week: int
    game_date: date
    game_id: str
    home: bool
    roster_status: str
    roster_week: int
    scheduled_qb_name: str | None
    player_rows: tuple[dict, ...]
    prior_player_rows: tuple[dict, ...]
    opponent_allowed_rows: tuple[dict, ...]
    league_rows: tuple[dict, ...]


def _download_csv(url: str, timeout: int = 20) -> tuple[dict, ...]:
    response = requests.get(url, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    return tuple(csv.DictReader(io.StringIO(response.text)))


@lru_cache(maxsize=4)
def player_week_rows(season: int) -> tuple[dict, ...]:
    return _download_csv(PLAYER_WEEK_URL.format(season=int(season)))


@lru_cache(maxsize=4)
def team_week_rows(season: int) -> tuple[dict, ...]:
    return _download_csv(TEAM_WEEK_URL.format(season=int(season)))


@lru_cache(maxsize=4)
def roster_week_rows(season: int) -> tuple[dict, ...]:
    return _download_csv(ROSTER_WEEK_URL.format(season=int(season)))


@lru_cache(maxsize=1)
def schedule_rows() -> tuple[dict, ...]:
    return _download_csv(SCHEDULE_URL)


def _i(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _d(value) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _name_aliases(value: str | None) -> set[str]:
    name = normalize(value)
    if not name:
        return set()
    parts = name.split()
    out = {name}
    if len(parts) >= 2:
        out.add(f"{parts[0]} {parts[-1]}")
    return out


def _same_player(a: str | None, b: str | None) -> bool:
    aa = _name_aliases(a)
    bb = _name_aliases(b)
    return bool(aa and bb and aa.intersection(bb))


def _season_for_date(event_date: date) -> int:
    return event_date.year - 1 if event_date.month <= 2 else event_date.year


def _schedule_game(event_date: date, team: str) -> dict | None:
    season = _season_for_date(event_date)
    hits = []
    for row in schedule_rows():
        if _i(row.get("season")) != season:
            continue
        if str(row.get("game_type") or "") != "REG":
            continue
        if _d(row.get("gameday")) != event_date:
            continue
        if team in {str(row.get("away_team") or ""), str(row.get("home_team") or "")}:
            hits.append(row)
    return hits[0] if len(hits) == 1 else None


def _candidate_players(player_name: str, event_date: date) -> list[tuple[str, str, str]]:
    season = _season_for_date(event_date)
    by_id: dict[str, tuple[str, str, str]] = {}

    for row in player_week_rows(season):
        if _i(row.get("season")) != season:
            continue
        if str(row.get("season_type") or "") != "REG":
            continue
        if str(row.get("position") or "") != "QB":
            continue
        display = str(row.get("player_display_name") or row.get("player_name") or "")
        if not _same_player(player_name, display):
            continue
        pid = str(row.get("player_id") or "")
        team = str(row.get("team") or "")
        if pid and team:
            by_id[pid] = (pid, display, team)

    if by_id:
        return list(by_id.values())

    # A current QB can have no stat row yet, so use the weekly roster as a
    # secondary identity source. The model still requires prior passing stats.
    for row in roster_week_rows(season):
        if _i(row.get("season")) != season:
            continue
        if str(row.get("game_type") or "") != "REG":
            continue
        if str(row.get("position") or "") != "QB":
            continue
        display = str(row.get("full_name") or "")
        if not _same_player(player_name, display):
            continue
        pid = str(row.get("gsis_id") or "")
        team = str(row.get("team") or "")
        if pid and team:
            by_id[pid] = (pid, display, team)

    return list(by_id.values())


def _latest_roster_status(
    player_id: str,
    team: str,
    *,
    season: int,
    target_week: int,
) -> tuple[str, int] | None:
    candidates: list[tuple[int, str]] = []
    for row in roster_week_rows(season):
        if _i(row.get("season")) != season:
            continue
        if str(row.get("game_type") or "") != "REG":
            continue
        if str(row.get("gsis_id") or "") != player_id:
            continue
        if str(row.get("team") or "") != team:
            continue
        week = _i(row.get("week"))
        if week is None or week > target_week:
            continue
        status = str(row.get("status") or "").upper().strip()
        candidates.append((week, status))

    if not candidates:
        return None
    week, status = max(candidates, key=lambda x: x[0])
    return status, week


def resolve_passing_context(
    *,
    player_name: str,
    event_date: date,
) -> NFLPassingContext | None:
    season = _season_for_date(event_date)
    candidates = _candidate_players(player_name, event_date)
    resolved: list[NFLPassingContext] = []

    for player_id, display_name, team in candidates:
        game = _schedule_game(event_date, team)
        if game is None:
            continue

        week = _i(game.get("week"))
        if week is None:
            continue
        away = str(game.get("away_team") or "")
        home = str(game.get("home_team") or "")
        if team == away:
            opponent = home
            home_flag = False
            scheduled_qb = str(game.get("away_qb_name") or "").strip() or None
        elif team == home:
            opponent = away
            home_flag = True
            scheduled_qb = str(game.get("home_qb_name") or "").strip() or None
        else:
            continue

        # If nflverse identifies the upcoming starting QB, treat a mismatch as
        # a hard identity/role conflict.
        if scheduled_qb and not _same_player(display_name, scheduled_qb):
            continue

        roster = _latest_roster_status(
            player_id,
            team,
            season=season,
            target_week=week,
        )
        if roster is None:
            continue
        roster_status, roster_week = roster
        if roster_status != "ACT":
            continue

        p_rows = []
        for row in player_week_rows(season):
            if _i(row.get("season")) != season:
                continue
            if str(row.get("season_type") or "") != "REG":
                continue
            if str(row.get("player_id") or "") != player_id:
                continue
            prior_week = _i(row.get("week"))
            attempts = _i(row.get("attempts"))
            if prior_week is None or prior_week >= week:
                continue
            if attempts is None or attempts < 10:
                continue
            p_rows.append(dict(row))
        p_rows.sort(key=lambda r: _i(r.get("week")) or 0)

        prior_rows = []
        prior_season = season - 1
        try:
            for row in player_week_rows(prior_season):
                if _i(row.get("season")) != prior_season:
                    continue
                if str(row.get("season_type") or "") != "REG":
                    continue
                if str(row.get("player_id") or "") != player_id:
                    continue
                attempts = _i(row.get("attempts"))
                if attempts is None or attempts < 10:
                    continue
                prior_rows.append(dict(row))
            prior_rows.sort(key=lambda r: _i(r.get("week")) or 0)
        except requests.RequestException:
            prior_rows = []

        allowed = []
        league = []
        for row in team_week_rows(season):
            if _i(row.get("season")) != season:
                continue
            if str(row.get("season_type") or "") != "REG":
                continue
            prior_week = _i(row.get("week"))
            if prior_week is None or prior_week >= week:
                continue
            attempts = _i(row.get("attempts"))
            if attempts is None or attempts <= 0:
                continue
            league.append(dict(row))
            if str(row.get("opponent_team") or "") == opponent:
                allowed.append(dict(row))

        resolved.append(
            NFLPassingContext(
                player_id=player_id,
                player_name=display_name,
                team=team,
                opponent=opponent,
                week=week,
                game_date=event_date,
                game_id=str(game.get("game_id") or ""),
                home=home_flag,
                roster_status=roster_status,
                roster_week=roster_week,
                scheduled_qb_name=scheduled_qb,
                player_rows=tuple(p_rows),
                prior_player_rows=tuple(prior_rows),
                opponent_allowed_rows=tuple(allowed),
                league_rows=tuple(league),
            )
        )

    # Cross-source identity must be unique.
    return resolved[0] if len(resolved) == 1 else None
