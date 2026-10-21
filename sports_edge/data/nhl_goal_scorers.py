"""Read-only NHL schedule, roster and goal-scorer research pipeline.

Official NHL API inputs only. Probabilities are unvalidated research estimates,
not recommendations or claims of positive expected value.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from math import exp
import requests

BASE = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "SportEdgeBot/1.0 (read-only analytics)", "Accept": "application/json"}


def _get_json(path: str, *, params: dict | None = None) -> dict:
    response = requests.get(f"{BASE}/{path.lstrip('/')}", params=params, headers=HEADERS, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("NHL API returned an unexpected payload")
    return payload


def _localized_name(value) -> str:
    """Read NHL API localized-name objects as well as legacy string fields."""
    if isinstance(value, dict):
        value = value.get("default") or next(iter(value.values()), "")
    return value.strip() if isinstance(value, str) else ""


@lru_cache(maxsize=16)
def schedule_for_day(day_iso: str) -> tuple[dict, ...]:
    date.fromisoformat(day_iso)
    payload = _get_json(f"schedule/{day_iso}")
    games = []
    for week in payload.get("gameWeek", []) or []:
        if week.get("date") == day_iso:
            games.extend(week.get("games", []) or [])
    return tuple(g for g in games if isinstance(g, dict))


@lru_cache(maxsize=64)
def roster_for_team(team_abbrev: str, season: str) -> tuple[dict, ...]:
    team = str(team_abbrev).strip().upper()
    if not team.isalpha() or not 2 <= len(team) <= 4:
        raise ValueError("invalid NHL team abbreviation")
    if len(season) != 8 or not season.isdigit():
        raise ValueError("season must be YYYYyyyy")
    payload = _get_json(f"roster/{team}/{season}")
    players = []
    for group in ("forwards", "defensemen"):
        rows = payload.get(group, []) or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                pid = int(row.get("id", row.get("playerId")))
            except (TypeError, ValueError):
                continue
            # The current NHL API schema supplies firstName/lastName as
            # localized objects; legacy fullName/name fields are not reliable.
            name = (
                _localized_name(row.get("fullName") or row.get("name"))
                or " ".join(
                    part for part in (
                        _localized_name(row.get("firstName")),
                        _localized_name(row.get("lastName")),
                    ) if part
                )
            )
            if pid > 0 and name:
                players.append({"player_id": pid, "name": name, "team": team,
                                "position": str(row.get("positionCode") or group[:-1]).upper()})
    return tuple(players)


@lru_cache(maxsize=256)
def player_game_log(player_id: int, season: str, game_type: int = 2) -> tuple[dict, ...]:
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
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            goals, assists, points, shots = (int(row[k]) for k in ("goals", "assists", "points", "shots"))
            pp_goals = int(row.get("powerPlayGoals", 0))
            game_id, game_date = str(row["gameId"]), str(row["gameDate"])
        except (KeyError, TypeError, ValueError):
            continue
        if min(goals, assists, points, shots, pp_goals) < 0 or points != goals + assists:
            continue
        normalized.append({"source": "NHL", "player_id": pid, "game_id": game_id, "date": game_date,
                           "goals": goals, "assists": assists, "points": points,
                           "shots": shots, "power_play_goals": pp_goals})
    return tuple(sorted(normalized, key=lambda r: (r["date"], r["game_id"])))


def candidate_player_ids(games: tuple[dict, ...]) -> tuple[int, ...]:
    found = set()
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
                            try:
                                pid = int(value.get("playerId", value.get("id")))
                            except (TypeError, ValueError):
                                continue
                            if pid > 0:
                                found.add(pid)
    return tuple(sorted(found))


def probability_goal_1plus(game_log: tuple[dict, ...], *, minimum_games: int = 20) -> dict:
    """Conservative Poisson baseline from official game logs; explicitly unvalidated."""
    if len(game_log) < minimum_games:
        return {"probability": None, "status": "INSUFFICIENT_HISTORY", "games": len(game_log)}
    if any(r.get("source") != "NHL" or type(r.get("goals")) is not int or r["goals"] < 0
           for r in game_log):
        return {"probability": None, "status": "INVALID_SOURCE_DATA", "games": len(game_log)}
    recent = game_log[-10:]
    season_rate = sum(r["goals"] for r in game_log) / len(game_log)
    recent_rate = sum(r["goals"] for r in recent) / len(recent)
    rate = 0.6 * recent_rate + 0.4 * season_rate
    return {"probability": max(0.0, min(1.0, 1.0 - exp(-rate))),
            "status": "UNVALIDATED_RESEARCH", "games": len(game_log), "rate": rate}


def goal_scorer_candidates(day_iso: str, *, current_season: str, prior_season: str,
                           max_per_game: int = 2) -> tuple[dict, ...]:
    """Return up to N research candidates per scheduled game; no fabricated fallbacks."""
    if not 1 <= max_per_game <= 5:
        raise ValueError("max_per_game must be between 1 and 5")
    output = []
    for game in schedule_for_day(day_iso):
        home = game.get("homeTeam") or {}
        away = game.get("awayTeam") or {}
        home_abbr = home.get("abbrev")
        away_abbr = away.get("abbrev")
        game_id = game.get("id")
        if not home_abbr or not away_abbr or game_id is None:
            continue
        for team_abbr, opponent in ((home_abbr, away_abbr), (away_abbr, home_abbr)):
            try:
                roster = roster_for_team(str(team_abbr), current_season)
            except (requests.RequestException, ValueError):
                continue
            candidates = []
            # Keep API work bounded, while evaluating forwards before defensemen.
            skaters = [p for p in roster if p.get("position") != "D"]
            skaters += [p for p in roster if p.get("position") == "D"]
            for player in skaters[:18]:
                pid = player["player_id"]
                try:
                    logs = player_game_log(pid, current_season)
                    season_used = current_season
                    if len(logs) < 20:
                        prior_logs = player_game_log(pid, prior_season)
                        if len(prior_logs) > len(logs):
                            logs = prior_logs
                            season_used = prior_season
                    estimate = probability_goal_1plus(logs)
                except (requests.RequestException, ValueError):
                    continue
                if estimate["probability"] is None:
                    continue
                candidates.append({**player, "opponent": str(opponent), "game_id": game_id,
                                   "game_date": day_iso, "season_used": season_used,
                                   **estimate,
                                   "model": "official_game_log_poisson_v1",
                                   "warnings": ["UNVALIDATED: no goalie/line/ice-time adjustment",
                                                "No executable odds/value check performed"]})
            candidates.sort(key=lambda r: (r["probability"], r.get("rate", 0)), reverse=True)
            output.extend(candidates[:max_per_game])
    return tuple(output)
