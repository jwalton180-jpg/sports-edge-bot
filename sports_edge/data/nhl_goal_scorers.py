"""Read-only NHL slate and player game-log retrieval for SportEdge research.

Uses NHL's public web API. This module retrieves source data only; it does not
place wagers and does not promote unvalidated model outputs to recommendations.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import requests

BASE = "https://api-web.nhle.com/v1"
HEADERS = {
    "User-Agent": "SportEdgeBot/1.0 (read-only analytics)",
    "Accept": "application/json",
}


def _get_json(path: str, *, params: dict | None = None) -> dict:
    response = requests.get(
        f"{BASE}/{path.lstrip('/')}",
        params=params,
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("NHL API returned an unexpected payload")
    return payload


@lru_cache(maxsize=16)
def schedule_for_day(day_iso: str) -> tuple[dict, ...]:
    """Return games scheduled on a calendar date; invalid dates fail closed."""
    date.fromisoformat(day_iso)
    payload = _get_json(f"schedule/{day_iso}")
    games: list[dict] = []
    for week in payload.get("gameWeek", []) or []:
        if week.get("date") != day_iso:
            continue
        games.extend(week.get("games", []) or [])
    return tuple(games)


@lru_cache(maxsize=256)
def player_game_log(player_id: int, season: str, game_type: int = 2) -> tuple[dict, ...]:
    """Fetch NHL player game logs; season must be explicit (e.g. 20252026)."""
    pid = int(player_id)
    if pid <= 0:
        raise ValueError("player_id must be positive")
    if len(season) != 8 or not season.isdigit():
        raise ValueError("season must be YYYYyyyy, e.g. 20252026")
    if game_type not in (2, 3):
        raise ValueError("game_type must be 2 (regular season) or 3 (playoffs)")
    payload = _get_json(f"player/{pid}/game-log/{season}/{game_type}")
    rows = payload.get("gameLog", [])
    if not isinstance(rows, list):
        raise ValueError("NHL API gameLog was not a list")
    normalized: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            goals = int(row["goals"])
            assists = int(row["assists"])
            points = int(row["points"])
            shots = int(row["shots"])
            pp_goals = int(row.get("powerPlayGoals", 0))
            game_id = str(row["gameId"])
            game_date = str(row["gameDate"])
        except (KeyError, TypeError, ValueError):
            # Missing stats are not fabricated as zero.
            continue
        if min(goals, assists, points, shots, pp_goals) < 0:
            continue
        if points != goals + assists:
            continue
        normalized.append({
            "source": "NHL",
            "player_id": pid,
            "game_id": game_id,
            "date": game_date,
            "goals": goals,
            "assists": assists,
            "points": points,
            "shots": shots,
            "power_play_goals": pp_goals,
        })
    return tuple(sorted(normalized, key=lambda r: (r["date"], r["game_id"])))


def candidate_player_ids(games: tuple[dict, ...]) -> tuple[int, ...]:
    """Extract unique, explicitly identified players only when present in API data."""
    found: set[int] = set()
    for game in games:
        for side in ("homeTeam", "awayTeam"):
            team = game.get(side) or {}
            for key in ("goalie", "players"):
                values = team.get(key)
                if isinstance(values, dict):
                    values = values.values()
                if isinstance(values, (list, tuple, type({}.values()))):
                    for value in values:
                        if isinstance(value, dict):
                            raw = value.get("playerId", value.get("id"))
                            try:
                                pid = int(raw)
                            except (TypeError, ValueError):
                                continue
                            if pid > 0:
                                found.add(pid)
    return tuple(sorted(found))
