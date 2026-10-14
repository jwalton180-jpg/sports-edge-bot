"""Historical sports evidence ingestion. Only finalized, timestamped official records.

The adapters do not infer market outcomes or use Kalshi prices. Callers must
supply event-time cutoffs and persist immutable predictions separately.
"""
from __future__ import annotations
from datetime import datetime, timezone
from functools import lru_cache
from math import isfinite
import requests

ESPN_ENDPOINTS = {
    "NCAAF": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
    "Soccer": "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard",
}
NHL_LOG = "https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/2"

def _datetime(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone required")
    return result.astimezone(timezone.utc)

def _get_json(url: str, params: dict | None = None) -> dict:
    response = requests.get(url, params=params, timeout=12,
                            headers={"User-Agent": "SportEdge-Research/1.0",
                                     "Accept": "application/json"})
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("invalid source response")
    return data

@lru_cache(maxsize=256)
def completed_team_games(sport: str, day: str) -> tuple[dict, ...]:
    """Read-only ESPN daily schedule. Reject unfinished or ambiguous contests."""
    if sport not in ESPN_ENDPOINTS:
        raise ValueError("unsupported sport")
    if len(day) != 8 or not day.isdigit():
        raise ValueError("date must be YYYYMMDD")
    data = _get_json(ESPN_ENDPOINTS[sport], {"dates": day, "limit": 500})
    rows = []
    for event in data.get("events", []):
        try:
            comp = event["competitions"][0]
            if len(event["competitions"]) != 1:
                continue
            if not event["status"]["type"]["completed"]:
                continue
            kickoff = _datetime(event["date"])
            competitors = comp["competitors"]
            if len(competitors) != 2:
                continue
            home, away = next(x for x in competitors if x["homeAway"] == "home"), next(x for x in competitors if x["homeAway"] == "away")
            home_score, away_score = int(home["score"]), int(away["score"])
            if min(home_score, away_score) < 0:
                continue
            rows.append({"source": "ESPN", "sport": sport, "event_id": str(event["id"]),
                         "event_time": kickoff.isoformat(), "home_id": str(home["team"]["id"]),
                         "away_id": str(away["team"]["id"]), "home_score": home_score,
                         "away_score": away_score})
        except (KeyError, IndexError, StopIteration, TypeError, ValueError):
            continue
    return tuple(rows)

@lru_cache(maxsize=256)
def nhl_player_goal_log(player_id: int, season: int) -> tuple[dict, ...]:
    """NHL official completed regular-season player game logs."""
    if player_id <= 0 or not 2000 <= season <= 2100:
        raise ValueError("invalid player or season")
    data = _get_json(NHL_LOG.format(player_id=player_id, season=season))
    rows = []
    for item in data.get("gameLog", []):
        try:
            goals = int(item["goals"])
            day = str(item["gameDate"])
            if goals < 0 or len(day) != 10:
                continue
            rows.append({"source": "NHL", "player_id": player_id,
                         "game_id": str(item["gameId"]), "date": day, "goals": goals})
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(rows)

def historical_nhl_projection(log: tuple[dict, ...], *, before_date: str, threshold: int = 2):
    """No future game logs may enter the player model."""
    from sports_edge.models.new_sport_research import hockey_goals
    prior = [r for r in log if r["date"] < before_date]
    if len({r["game_id"] for r in prior}) != len(prior):
        raise ValueError("duplicate NHL game")
    return hockey_goals(goals_per_game=(sum(r["goals"] for r in prior) / len(prior) if prior else None),
                        games=len(prior), threshold=threshold,
                        source_verified=all(r.get("source") == "NHL" for r in prior),
                        as_of_before_event=bool(prior))

def historical_soccer_projection(games: tuple[dict, ...], *, home_id: str,
                                 away_id: str, before_time: str, threshold: int = 2):
    from sports_edge.models.new_sport_research import soccer_goals
    cutoff = _datetime(before_time)
    prior = [r for r in games if r.get("sport") == "Soccer"
             and _datetime(r["event_time"]) < cutoff]
    home = [r for r in prior if r["home_id"] == home_id]
    away = [r for r in prior if r["away_id"] == away_id]
    def avg(rows, field):
        return sum(r[field] for r in rows) / len(rows) if rows else None
    return soccer_goals(home_scored=avg(home,"home_score"),home_conceded=avg(home,"away_score"),
                        away_scored=avg(away,"away_score"),away_conceded=avg(away,"home_score"),
                        home_games=len(home),away_games=len(away),threshold=threshold,
                        source_verified=all(r.get("source") == "ESPN" for r in prior),
                        as_of_before_event=bool(prior))

def historical_ncaaf_projection(games: tuple[dict, ...], *, favorite_id: str,
                                underdog_id: str, before_time: str,
                                favorite_home: bool, upset_red_flags: tuple[str,...] | None):
    from sports_edge.models.new_sport_research import college_football_win
    cutoff = _datetime(before_time)
    prior = [r for r in games if r.get("sport") == "NCAAF"
             and _datetime(r["event_time"]) < cutoff]
    def margins(team_id):
        return [(r["home_score"] - r["away_score"]) * (1 if r["home_id"] == team_id else -1)
                for r in prior if team_id in (r["home_id"], r["away_id"])]
    favorite, underdog = margins(favorite_id), margins(underdog_id)
    return college_football_win(
        favorite_rating=sum(favorite)/len(favorite) if favorite else None,
        underdog_rating=sum(underdog)/len(underdog) if underdog else None,
        favorite_games=len(favorite),underdog_games=len(underdog),
        home_adjustment_points=2.5 if favorite_home else -2.5,
        upset_red_flags=upset_red_flags,
        source_verified=all(r.get("source") == "ESPN" for r in prior),
        as_of_before_event=bool(prior))
