from __future__ import annotations

from datetime import date
from functools import lru_cache
import re

import requests

from sports_edge.models.game_scope import normalize


BASE = "https://statsapi.mlb.com/api/v1"
_HEADERS = {
    "User-Agent": "SportsEdgeReadOnly/1.0",
    "Accept": "application/json",
}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _name_aliases(value: str | None) -> set[str]:
    n = normalize(value)
    if not n:
        return set()
    parts = n.split()
    if parts and parts[-1] in _SUFFIXES:
        parts = parts[:-1]
    if not parts:
        return set()
    out = {" ".join(parts)}
    if len(parts) >= 3:
        out.add(f"{parts[0]} {parts[-1]}")
    return out


@lru_cache(maxsize=8)
def season_players(season: int) -> tuple[dict, ...]:
    response = requests.get(
        f"{BASE}/sports/1/players",
        params={"season": int(season)},
        headers=_HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    return tuple(response.json().get("people", []) or [])


def resolve_player(name: str, season: int) -> dict | None:
    wanted = _name_aliases(name)
    if not wanted:
        return None

    hits: list[dict] = []
    for player in season_players(season):
        aliases: set[str] = set()
        for key in (
            "fullName",
            "nameFirstLast",
            "firstLastName",
            "fullFMLName",
        ):
            aliases.update(_name_aliases(player.get(key)))
        use_name = str(player.get("useName") or "")
        use_last = str(player.get("useLastName") or "")
        aliases.update(_name_aliases(f"{use_name} {use_last}".strip()))
        if wanted & aliases:
            hits.append(player)

    if len(hits) == 1:
        return hits[0]

    exact = [
        player for player in hits
        if normalize(str(player.get("fullName") or "")) == normalize(name)
    ]
    return exact[0] if len(exact) == 1 else None


@lru_cache(maxsize=64)
def schedule_for_day(day_iso: str) -> tuple[dict, ...]:
    response = requests.get(
        f"{BASE}/schedule",
        params={
            "sportId": 1,
            "date": day_iso,
            "hydrate": "probablePitcher,team",
        },
        headers=_HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    games: list[dict] = []
    for block in response.json().get("dates", []) or []:
        games.extend(block.get("games", []) or [])
    return tuple(games)


def _game_number_from_ticker(event_ticker: str | None) -> int | None:
    match = re.search(r"G([12])(?:$|-)", str(event_ticker or "").upper())
    return int(match.group(1)) if match else None


def find_player_game(
    *,
    team_id: int,
    event_date: date,
    event_ticker: str | None,
) -> dict | None:
    candidates: list[dict] = []
    for game in schedule_for_day(event_date.isoformat()):
        teams = game.get("teams") or {}
        try:
            home_id = int(((teams.get("home") or {}).get("team") or {}).get("id"))
            away_id = int(((teams.get("away") or {}).get("team") or {}).get("id"))
        except (TypeError, ValueError):
            continue
        if team_id in {home_id, away_id}:
            candidates.append(game)

    if len(candidates) == 1:
        return candidates[0]

    game_no = _game_number_from_ticker(event_ticker)
    if game_no is not None:
        numbered = [
            game for game in candidates
            if int(game.get("gameNumber") or 0) == game_no
        ]
        if len(numbered) == 1:
            return numbered[0]

    # Ambiguous doubleheaders fail closed rather than joining the wrong game.
    return None


@lru_cache(maxsize=1024)
def season_stat(
    player_id: int,
    group: str,
    season: int,
) -> dict | None:
    response = requests.get(
        f"{BASE}/people/{int(player_id)}/stats",
        params={
            "stats": "season",
            "group": group,
            "season": int(season),
        },
        headers=_HEADERS,
        timeout=12,
    )
    response.raise_for_status()
    stats = response.json().get("stats", []) or []
    if not stats:
        return None
    splits = stats[0].get("splits", []) or []
    if not splits:
        return None
    return dict(splits[0].get("stat") or {})


@lru_cache(maxsize=2048)
def date_range_stat(
    player_id: int,
    group: str,
    start_iso: str,
    end_iso: str,
) -> dict | None:
    response = requests.get(
        f"{BASE}/people/{int(player_id)}/stats",
        params={
            "stats": "byDateRange",
            "group": group,
            "startDate": start_iso,
            "endDate": end_iso,
        },
        headers=_HEADERS,
        timeout=12,
    )
    response.raise_for_status()
    stats = response.json().get("stats", []) or []
    if not stats:
        return None

    # The API can return MLB plus an aggregate "All" split. Prefer sportId 1.
    for split in stats[0].get("splits", []) or []:
        sport = split.get("sport") or {}
        if int(sport.get("id") or 0) == 1:
            return dict(split.get("stat") or {})

    splits = stats[0].get("splits", []) or []
    return dict((splits[0].get("stat") or {})) if splits else None


@lru_cache(maxsize=256)
def team_season_stat(
    team_id: int,
    group: str,
    season: int,
) -> dict | None:
    response = requests.get(
        f"{BASE}/teams/{int(team_id)}/stats",
        params={
            "stats": "season",
            "group": group,
            "season": int(season),
        },
        headers=_HEADERS,
        timeout=12,
    )
    response.raise_for_status()
    stats = response.json().get("stats", []) or []
    if not stats:
        return None
    splits = stats[0].get("splits", []) or []
    if not splits:
        return None
    return dict(splits[0].get("stat") or {})


def game_context_for_pitcher(player: dict, game: dict) -> dict | None:
    try:
        team_id = int((player.get("currentTeam") or {}).get("id"))
        player_id = int(player.get("id"))
    except (TypeError, ValueError):
        return None

    teams = game.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    home_team = home.get("team") or {}
    away_team = away.get("team") or {}

    try:
        home_id = int(home_team.get("id"))
        away_id = int(away_team.get("id"))
    except (TypeError, ValueError):
        return None

    if team_id == home_id:
        own, opp = home, away
    elif team_id == away_id:
        own, opp = away, home
    else:
        return None

    probable = own.get("probablePitcher") or {}
    try:
        probable_id = int(probable.get("id"))
    except (TypeError, ValueError):
        probable_id = None

    return {
        "own_team_id": team_id,
        "own_team_name": str((own.get("team") or {}).get("name") or ""),
        "opponent_team_id": int((opp.get("team") or {}).get("id") or 0),
        "opponent_team_name": str((opp.get("team") or {}).get("name") or ""),
        "probable_pitcher_id": probable_id,
        "probable_pitcher_name": str(probable.get("fullName") or ""),
        "is_probable_starter": probable_id == player_id if probable_id is not None else False,
        "game_pk": game.get("gamePk"),
        "game_status": str((game.get("status") or {}).get("abstractGameState") or ""),
    }


def game_context_for_hitter(player: dict, game: dict) -> dict | None:
    try:
        team_id = int((player.get("currentTeam") or {}).get("id"))
    except (TypeError, ValueError):
        return None

    teams = game.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    home_team = home.get("team") or {}
    away_team = away.get("team") or {}

    try:
        home_id = int(home_team.get("id"))
        away_id = int(away_team.get("id"))
    except (TypeError, ValueError):
        return None

    if team_id == home_id:
        own, opp = home, away
    elif team_id == away_id:
        own, opp = away, home
    else:
        return None

    pitcher = opp.get("probablePitcher") or {}
    return {
        "own_team_id": team_id,
        "own_team_name": str((own.get("team") or {}).get("name") or ""),
        "opponent_team_id": int((opp.get("team") or {}).get("id") or 0),
        "opponent_team_name": str((opp.get("team") or {}).get("name") or ""),
        "probable_pitcher_id": pitcher.get("id"),
        "probable_pitcher_name": str(pitcher.get("fullName") or ""),
        "game_pk": game.get("gamePk"),
        "game_status": str((game.get("status") or {}).get("abstractGameState") or ""),
    }
