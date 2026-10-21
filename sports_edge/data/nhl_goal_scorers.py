"""Read-only NHL goal-scorer research pipeline using official NHL API data.

Outputs are unvalidated research estimates, not betting advice or claims of edge.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from functools import lru_cache
from math import exp
import requests

BASE = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "SportEdgeBot/1.0 (read-only analytics)", "Accept": "application/json"}


def _get_json(path: str, *, params: dict | None = None) -> dict:
    response = requests.get(f"{BASE}/{path.lstrip('/')}", params=params, headers=HEADERS, timeout=12)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("NHL API returned an unexpected payload")
    return payload


def _localized_name(value) -> str:
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
            name = (
                _localized_name(row.get("fullName") or row.get("name"))
                or " ".join(part for part in (
                    _localized_name(row.get("firstName")),
                    _localized_name(row.get("lastName")),
                ) if part)
            )
            if pid > 0 and name:
                players.append({"player_id": pid, "name": name, "team": team,
                                "position": str(row.get("positionCode") or group[:-1]).upper()})
    return tuple(players)


@lru_cache(maxsize=512)
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
    """Extract explicit player IDs from NHL game/team payloads without guessing."""
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
                        if not isinstance(value, dict):
                            continue
                        try:
                            pid = int(value.get("playerId", value.get("id")))
                        except (TypeError, ValueError):
                            continue
                        if pid > 0:
                            found.add(pid)
    return tuple(sorted(found))

def probability_goal_1plus(game_log: tuple[dict, ...], *, minimum_games: int = 10) -> dict:
    """Poisson baseline; lower history floor is explicitly flagged as unvalidated."""
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


def _player_history(player: dict, seasons: tuple[str, ...], minimum_games: int) -> tuple[dict | None, str | None, str | None]:
    best_logs: tuple[dict, ...] = ()
    best_season = seasons[0]
    errors = []
    for season in seasons:
        try:
            logs = player_game_log(player["player_id"], season)
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{season}: {type(exc).__name__}")
            continue
        if len(logs) > len(best_logs):
            best_logs, best_season = logs, season
        if len(logs) >= minimum_games:
            # Prefer current-season history. Otherwise accept the first prior
            # season with sufficient history; only query an older season when
            # the newer season has too little data. This keeps API work bounded.
            return {"logs": logs, "season": season}, None, None
    estimate = probability_goal_1plus(best_logs, minimum_games=minimum_games)
    if estimate["probability"] is None:
        return None, best_season, "; ".join(errors) if errors else None
    return {"logs": best_logs, "season": best_season}, None, None


def goal_scorer_report(day_iso: str, *, current_season: str, prior_season: str,
                       max_per_game: int = 2, minimum_games: int = 10,
                       max_skaters_per_team: int = 24) -> dict:
    """Return candidates plus actionable diagnostics so an empty state is explainable."""
    if not 1 <= max_per_game <= 5:
        raise ValueError("max_per_game must be between 1 and 5")
    if not 5 <= minimum_games <= 30:
        raise ValueError("minimum_games must be between 5 and 30")
    games = schedule_for_day(day_iso)
    report = {"date": day_iso, "schedule_games": len(games), "teams_seen": 0,
              "roster_players": 0, "players_checked": 0, "history_qualified": 0,
              "api_errors": 0, "candidates": [], "diagnostic": ""}
    season_start = int(current_season[:4])
    seasons = tuple(dict.fromkeys((current_season, prior_season, f"{season_start - 2}{season_start - 1}")))
    for game in games:
        home, away = game.get("homeTeam") or {}, game.get("awayTeam") or {}
        if not home.get("abbrev") or not away.get("abbrev") or game.get("id") is None:
            continue
        for team_abbr, opponent in ((home["abbrev"], away["abbrev"]), (away["abbrev"], home["abbrev"])):
            report["teams_seen"] += 1
            try:
                roster = roster_for_team(str(team_abbr), current_season)
            except (requests.RequestException, ValueError):
                report["api_errors"] += 1
                continue
            skaters = [p for p in roster if p.get("position") != "D"]
            skaters += [p for p in roster if p.get("position") == "D"]
            skaters = skaters[:max_skaters_per_team]
            report["roster_players"] += len(skaters)
            team_candidates = []
            # Bound concurrency to avoid a serial multi-minute scan or request burst.
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = {pool.submit(_player_history, p, seasons, minimum_games): p for p in skaters}
                for future in as_completed(futures):
                    player = futures[future]
                    report["players_checked"] += 1
                    try:
                        history, error_season, error = future.result()
                    except Exception:
                        report["api_errors"] += 1
                        continue
                    if history is None:
                        if error:
                            report["api_errors"] += 1
                        continue
                    estimate = probability_goal_1plus(history["logs"], minimum_games=minimum_games)
                    if estimate["probability"] is None:
                        continue
                    report["history_qualified"] += 1
                    team_candidates.append({**player, "opponent": str(opponent), "game_id": game["id"],
                        "game_date": day_iso, "season_used": history["season"], **estimate,
                        "model": "official_game_log_poisson_v1",
                        "warnings": ["UNVALIDATED: no goalie/line/ice-time adjustment",
                                     "No executable odds/value check performed"]})
            team_candidates.sort(key=lambda r: (r["probability"], r.get("rate", 0)), reverse=True)
            report["candidates"].extend(team_candidates[:max_per_game])
    if report["candidates"]:
        report["diagnostic"] = f"{len(report['candidates'])} research candidates from {report['history_qualified']} history-qualified skaters."
    elif report["schedule_games"] == 0:
        report["diagnostic"] = "Official schedule returned no games for this date."
    elif report["roster_players"] == 0:
        report["diagnostic"] = "No rosters loaded; official NHL roster API failed or returned no skaters."
    elif report["history_qualified"] == 0:
        report["diagnostic"] = "No skaters met the history floor across current and two prior seasons; inspect API error count and history coverage."
    else:
        report["diagnostic"] = "Players had sufficient history, but no candidate rows were emitted; inspect candidate ranking/filters."
    return report


def goal_scorer_candidates(day_iso: str, *, current_season: str, prior_season: str,
                           max_per_game: int = 2) -> tuple[dict, ...]:
    report = goal_scorer_report(day_iso, current_season=current_season, prior_season=prior_season,
                                max_per_game=max_per_game)
    return tuple(report["candidates"])
