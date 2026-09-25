from __future__ import annotations

import csv
import io
from dataclasses import replace
from datetime import date
from functools import lru_cache

import requests

from sports_edge.data.public_player_data import nba_player_rows
from sports_edge.models.game_scope import normalize
from sports_edge.models.model_evidence import ModelEvidence, team_record_model


_GENERIC_LOCATION_TOKENS = {
    "city", "united", "state", "university", "club", "team",
}

NFLVERSE_SCHEDULE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
)

BASKETBALL_SCHEDULE_URLS = {
    "NBA": "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json",
}

WNBA_SCHEDULE_CSV_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_wnba_schedules/wnba_schedule_2026.csv"
)

NFL_TEAM_ALIASES: dict[str, tuple[str, ...]] = {
    "ARI": ("Arizona", "Arizona Cardinals", "Cardinals"),
    "ATL": ("Atlanta", "Atlanta Falcons", "Falcons"),
    "BAL": ("Baltimore", "Baltimore Ravens", "Ravens"),
    "BUF": ("Buffalo", "Buffalo Bills", "Bills"),
    "CAR": ("Carolina", "Carolina Panthers", "Panthers"),
    "CHI": ("Chicago", "Chicago Bears", "Bears"),
    "CIN": ("Cincinnati", "Cincinnati Bengals", "Bengals"),
    "CLE": ("Cleveland", "Cleveland Browns", "Browns"),
    "DAL": ("Dallas", "Dallas Cowboys", "Cowboys"),
    "DEN": ("Denver", "Denver Broncos", "Broncos"),
    "DET": ("Detroit", "Detroit Lions", "Lions"),
    "GB": ("Green Bay", "Green Bay Packers", "Packers"),
    "HOU": ("Houston", "Houston Texans", "Texans"),
    "IND": ("Indianapolis", "Indianapolis Colts", "Colts"),
    "JAX": ("Jacksonville", "Jacksonville Jaguars", "Jaguars"),
    "KC": ("Kansas City", "Kansas City Chiefs", "Chiefs"),
    "LV": ("Las Vegas", "Las Vegas Raiders", "Raiders"),
    "LAC": ("Los Angeles C", "Los Angeles Chargers", "LA Chargers", "Chargers"),
    "LAR": ("Los Angeles R", "Los Angeles Rams", "LA Rams", "Rams"),
    "MIA": ("Miami", "Miami Dolphins", "Dolphins"),
    "MIN": ("Minnesota", "Minnesota Vikings", "Vikings"),
    "NE": ("New England", "New England Patriots", "Patriots"),
    "NO": ("New Orleans", "New Orleans Saints", "Saints"),
    "NYG": ("New York G", "New York Giants", "NY Giants", "Giants"),
    "NYJ": ("New York J", "New York Jets", "NY Jets", "Jets"),
    "PHI": ("Philadelphia", "Philadelphia Eagles", "Eagles"),
    "PIT": ("Pittsburgh", "Pittsburgh Steelers", "Steelers"),
    "SEA": ("Seattle", "Seattle Seahawks", "Seahawks"),
    "SF": ("San Francisco", "San Francisco 49ers", "49ers"),
    "TB": ("Tampa Bay", "Tampa Bay Buccaneers", "Buccaneers"),
    "TEN": ("Tennessee", "Tennessee Titans", "Titans"),
    "WAS": ("Washington", "Washington Commanders", "Commanders"),
}


def _aliases(name: str) -> set[str]:
    n = normalize(name)
    if not n:
        return set()
    parts = n.split()
    out = {n}
    if len(parts) >= 2:
        last = parts[-1]
        if len(last) >= 3 and last not in _GENERIC_LOCATION_TOKENS:
            out.add(last)
    if len(parts) >= 3:
        out.add(" ".join(parts[-2:]))
    return {x for x in out if len(x) >= 3 and x not in _GENERIC_LOCATION_TOKENS}


def _metadata_aliases(team: dict) -> set[str]:
    values = {
        str(team.get("displayName") or ""),
        str(team.get("shortDisplayName") or ""),
        str(team.get("name") or ""),
        str(team.get("abbreviation") or ""),
        str(team.get("location") or ""),
        str(team.get("nickname") or ""),
        str(team.get("clubName") or ""),
        str(team.get("teamName") or ""),
        str(team.get("shortName") or ""),
        str(team.get("teamCity") or ""),
        str(team.get("teamTricode") or ""),
        str(team.get("teamSlug") or ""),
    }
    aliases: set[str] = set()
    for value in values:
        aliases.update(_aliases(value))

    location = normalize(str(team.get("location") or team.get("teamCity") or ""))
    nickname = normalize(
        str(team.get("name") or team.get("nickname") or team.get("teamName") or "")
    )
    if location:
        aliases.add(location)
    if location and nickname:
        aliases.add(f"{location} {nickname}")
        aliases.add(f"{location} {nickname[0]}")

    return {x for x in aliases if x and x not in _GENERIC_LOCATION_TOKENS}


def _identity_score(query: str, team: dict) -> int:
    q = normalize(query)
    if not q:
        return 0
    aliases = _metadata_aliases(team)
    if q in aliases:
        return 100
    overlap = _aliases(query) & aliases
    if not overlap:
        return 0
    return max(60 + min(30, len(alias)) for alias in overlap)


def _resolve_competitors(
    team_a: str,
    team_b: str,
    competitors: list[dict],
) -> tuple[dict, dict] | None:
    scored_a = [(comp, _identity_score(team_a, comp.get("team") or {})) for comp in competitors]
    scored_b = [(comp, _identity_score(team_b, comp.get("team") or {})) for comp in competitors]
    scored_a = [(c, s) for c, s in scored_a if s > 0]
    scored_b = [(c, s) for c, s in scored_b if s > 0]
    if not scored_a or not scored_b:
        return None
    max_a = max(s for _, s in scored_a)
    max_b = max(s for _, s in scored_b)
    best_a = [c for c, s in scored_a if s == max_a]
    best_b = [c for c, s in scored_b if s == max_b]
    if len(best_a) != 1 or len(best_b) != 1 or best_a[0] is best_b[0]:
        return None
    return best_a[0], best_b[0]


def _pct(wins: int, losses: int) -> float:
    total = wins + losses
    return wins / total if total > 0 else 0.5


def _safe_float(value) -> float | None:
    if value in (None, "", "NA", "NaN", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# MLB — official MLB Stats API

@lru_cache(maxsize=8)
def _mlb_teams(season: int) -> tuple[dict, ...]:
    r = requests.get(
        "https://statsapi.mlb.com/api/v1/teams",
        params={"sportId": 1, "season": season},
        headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "application/json"},
        timeout=10,
    )
    r.raise_for_status()
    return tuple(r.json().get("teams", []) or [])


@lru_cache(maxsize=8)
def _mlb_standings(season: int) -> tuple[dict, ...]:
    r = requests.get(
        "https://statsapi.mlb.com/api/v1/standings",
        params={"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"},
        headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "application/json"},
        timeout=10,
    )
    r.raise_for_status()
    rows: list[dict] = []
    for block in r.json().get("records", []) or []:
        rows.extend(block.get("teamRecords", []) or [])
    dedup: dict[int, dict] = {}
    for row in rows:
        try:
            tid = int((row.get("team") or {}).get("id"))
        except (TypeError, ValueError):
            continue
        dedup[tid] = row
    return tuple(dedup.values())


@lru_cache(maxsize=64)
def _mlb_schedule(day_iso: str) -> tuple[dict, ...]:
    r = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": day_iso, "hydrate": "team"},
        headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "application/json"},
        timeout=10,
    )
    r.raise_for_status()
    games: list[dict] = []
    for block in r.json().get("dates", []) or []:
        games.extend(block.get("games", []) or [])
    return tuple(games)


def _mlb_team_names(season: int) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for team in _mlb_teams(season):
        try:
            tid = int(team.get("id"))
        except (TypeError, ValueError):
            continue
        metadata = {
            "displayName": team.get("name"),
            "shortDisplayName": team.get("shortName"),
            "name": team.get("teamName"),
            "abbreviation": team.get("abbreviation"),
            "location": team.get("locationName"),
            "clubName": team.get("clubName"),
            "teamName": team.get("teamName"),
            "shortName": team.get("shortName"),
        }
        out[tid] = _metadata_aliases(metadata)
    return out


def _mlb_match_team_id(name: str, aliases: dict[int, set[str]]) -> int | None:
    q = normalize(name)
    wanted = _aliases(name)
    exact = [tid for tid, values in aliases.items() if q and q in values]
    if len(exact) == 1:
        return exact[0]
    hits = [tid for tid, values in aliases.items() if wanted & values]
    return hits[0] if len(hits) == 1 else None


def _last_ten_pct(row: dict) -> float | None:
    for rec in ((row.get("records") or {}).get("splitRecords") or []):
        if str(rec.get("type") or "") == "lastTen":
            try:
                return float(rec.get("pct"))
            except (TypeError, ValueError):
                return None
    return None


def mlb_team_model(team_a: str, team_b: str, *, event_date: date) -> ModelEvidence | None:
    season = event_date.year
    try:
        aliases = _mlb_team_names(season)
        aid = _mlb_match_team_id(team_a, aliases)
        bid = _mlb_match_team_id(team_b, aliases)
        if aid is None or bid is None or aid == bid:
            return None
        standings = {int((r.get("team") or {}).get("id")): r for r in _mlb_standings(season)}
        a, b = standings.get(aid), standings.get(bid)
        if not a or not b:
            return None
        home_a: bool | None = None
        for game in _mlb_schedule(event_date.isoformat()):
            teams = game.get("teams") or {}
            home_id = int(((((teams.get("home") or {}).get("team") or {}).get("id")) or 0))
            away_id = int(((((teams.get("away") or {}).get("team") or {}).get("id")) or 0))
            if {home_id, away_id} == {aid, bid}:
                home_a = home_id == aid
                break
        ga = max(1, int(a.get("gamesPlayed") or 0))
        gb = max(1, int(b.get("gamesPlayed") or 0))
        return team_record_model(
            sport="MLB",
            team_a=team_a,
            team_b=team_b,
            win_pct_a=float(a.get("winningPercentage") or 0.5),
            win_pct_b=float(b.get("winningPercentage") or 0.5),
            games_a=ga,
            games_b=gb,
            home_a=home_a,
            recent_pct_a=_last_ten_pct(a),
            recent_pct_b=_last_ten_pct(b),
            differential_per_game_a=float(a.get("runDifferential") or 0.0) / ga,
            differential_per_game_b=float(b.get("runDifferential") or 0.0) / gb,
        )
    except (requests.RequestException, TypeError, ValueError):
        return None


# NFL — nflverse public schedules/results

@lru_cache(maxsize=1)
def _nflverse_games() -> tuple[dict, ...]:
    r = requests.get(
        NFLVERSE_SCHEDULE_URL,
        headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "text/csv,*/*"},
        timeout=20,
    )
    r.raise_for_status()
    return tuple(csv.DictReader(io.StringIO(r.text)))


def _nfl_team_code(name: str) -> str | None:
    q = normalize(name)
    if not q:
        return None
    hits: list[str] = []
    for code, values in NFL_TEAM_ALIASES.items():
        aliases = {normalize(code), *(normalize(v) for v in values)}
        if q in aliases:
            hits.append(code)
    return hits[0] if len(hits) == 1 else None


def _nfl_season(event_date: date) -> int:
    return event_date.year - 1 if event_date.month <= 2 else event_date.year


def nfl_team_model(team_a: str, team_b: str, *, event_date: date) -> ModelEvidence | None:
    try:
        aid, bid = _nfl_team_code(team_a), _nfl_team_code(team_b)
        if aid is None or bid is None or aid == bid:
            return None
        season = _nfl_season(event_date)
        season_rows = [
            row for row in _nflverse_games()
            if str(row.get("season") or "") == str(season)
            and str(row.get("game_type") or "") in {"REG", "POST"}
        ]
        completed: list[tuple[date, dict, float, float]] = []
        home_a: bool | None = None
        for row in season_rows:
            try:
                d = date.fromisoformat(str(row.get("gameday") or ""))
            except ValueError:
                continue
            home, away = str(row.get("home_team") or ""), str(row.get("away_team") or "")
            if d == event_date and {home, away} == {aid, bid}:
                home_a = home == aid
            hs, aws = _safe_float(row.get("home_score")), _safe_float(row.get("away_score"))
            if d >= event_date or hs is None or aws is None:
                continue
            completed.append((d, row, hs, aws))

        stats = {
            aid: {"wins": 0, "losses": 0, "pf": 0.0, "pa": 0.0, "recent": []},
            bid: {"wins": 0, "losses": 0, "pf": 0.0, "pa": 0.0, "recent": []},
        }
        for d, row, hs, aws in completed:
            home, away = str(row.get("home_team") or ""), str(row.get("away_team") or "")
            for code in (aid, bid):
                if code not in {home, away}:
                    continue
                pf, pa = (hs, aws) if code == home else (aws, hs)
                won = pf > pa
                stats[code]["wins"] += int(won)
                stats[code]["losses"] += int(not won)
                stats[code]["pf"] += pf
                stats[code]["pa"] += pa
                stats[code]["recent"].append((d, 1.0 if won else 0.0))

        a, b = stats[aid], stats[bid]
        ga, gb = int(a["wins"] + a["losses"]), int(b["wins"] + b["losses"])
        if min(ga, gb) < 2:
            return None
        recent_a = [v for _, v in sorted(a["recent"], key=lambda x: x[0])[-5:]]
        recent_b = [v for _, v in sorted(b["recent"], key=lambda x: x[0])[-5:]]
        return team_record_model(
            sport="NFL",
            team_a=team_a,
            team_b=team_b,
            win_pct_a=_pct(int(a["wins"]), int(a["losses"])),
            win_pct_b=_pct(int(b["wins"]), int(b["losses"])),
            games_a=ga,
            games_b=gb,
            home_a=home_a,
            recent_pct_a=(sum(recent_a) / len(recent_a)) if recent_a else None,
            recent_pct_b=(sum(recent_b) / len(recent_b)) if recent_b else None,
            differential_per_game_a=(a["pf"] - a["pa"]) / ga,
            differential_per_game_b=(b["pf"] - b["pa"]) / gb,
        )
    except (requests.RequestException, TypeError, ValueError):
        return None


# WNBA — SportsDataverse GitHub release (cloud-safe public CSV)

@lru_cache(maxsize=1)
def _wnba_schedule_rows() -> tuple[dict, ...]:
    r = requests.get(
        WNBA_SCHEDULE_CSV_URL,
        headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "text/csv,*/*"},
        timeout=20,
    )
    r.raise_for_status()
    return tuple(csv.DictReader(io.StringIO(r.text)))


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def _wnba_team_index(rows: tuple[dict, ...]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for row in rows:
        for side in ("home", "away"):
            try:
                tid = int(row.get(f"{side}_id"))
            except (TypeError, ValueError):
                continue
            meta = {
                "displayName": row.get(f"{side}_display_name"),
                "shortDisplayName": row.get(f"{side}_short_display_name"),
                "name": row.get(f"{side}_name"),
                "abbreviation": row.get(f"{side}_abbreviation"),
                "location": row.get(f"{side}_location"),
            }
            out.setdefault(tid, set()).update(_metadata_aliases(meta))
    return out


def wnba_team_model(
    team_a: str,
    team_b: str,
    *,
    event_date: date,
) -> ModelEvidence | None:
    try:
        rows = _wnba_schedule_rows()
        index = _wnba_team_index(rows)
        aid = _resolve_team_id(team_a, index)
        bid = _resolve_team_id(team_b, index)
        if aid is None or bid is None or aid == bid:
            return None

        stats = {
            aid: {"wins": 0, "losses": 0, "pf": 0.0, "pa": 0.0, "recent": []},
            bid: {"wins": 0, "losses": 0, "pf": 0.0, "pa": 0.0, "recent": []},
        }
        home_a: bool | None = None

        for row in rows:
            raw_date = str(row.get("game_date") or row.get("start_date") or "")[:10]
            try:
                gd = date.fromisoformat(raw_date)
            except ValueError:
                continue

            try:
                home_id = int(row.get("home_id"))
                away_id = int(row.get("away_id"))
            except (TypeError, ValueError):
                continue

            if gd == event_date and {home_id, away_id} == {aid, bid}:
                home_a = home_id == aid

            if gd >= event_date:
                continue
            if str(row.get("season_type") or row.get("type_id") or "") not in {"2", "3"}:
                continue
            if not _truthy(row.get("status_type_completed")):
                continue

            hs = _safe_float(row.get("home_score"))
            aws = _safe_float(row.get("away_score"))
            if hs is None or aws is None:
                continue

            for tid in (aid, bid):
                if tid not in {home_id, away_id}:
                    continue
                pf, pa = (hs, aws) if tid == home_id else (aws, hs)
                won = pf > pa
                stats[tid]["wins"] += int(won)
                stats[tid]["losses"] += int(not won)
                stats[tid]["pf"] += pf
                stats[tid]["pa"] += pa
                stats[tid]["recent"].append((gd, 1.0 if won else 0.0))

        a, b = stats[aid], stats[bid]
        ga = int(a["wins"] + a["losses"])
        gb = int(b["wins"] + b["losses"])
        if min(ga, gb) < 5:
            return None

        recent_a = [v for _, v in sorted(a["recent"], key=lambda x: x[0])[-10:]]
        recent_b = [v for _, v in sorted(b["recent"], key=lambda x: x[0])[-10:]]
        return team_record_model(
            sport="WNBA",
            team_a=team_a,
            team_b=team_b,
            win_pct_a=_pct(int(a["wins"]), int(a["losses"])),
            win_pct_b=_pct(int(b["wins"]), int(b["losses"])),
            games_a=ga,
            games_b=gb,
            home_a=home_a,
            recent_pct_a=(sum(recent_a) / len(recent_a)) if recent_a else None,
            recent_pct_b=(sum(recent_b) / len(recent_b)) if recent_b else None,
            differential_per_game_a=(a["pf"] - a["pa"]) / ga,
            differential_per_game_b=(b["pf"] - b["pa"]) / gb,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None


# NBA / WNBA — official league schedule/results JSON

@lru_cache(maxsize=2)
def _basketball_schedule(sport: str) -> dict:
    r = requests.get(
        BASKETBALL_SCHEDULE_URLS[sport],
        headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "application/json"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def _basketball_games(payload: dict) -> list[dict]:
    out: list[dict] = []
    for block in ((payload.get("leagueSchedule") or {}).get("gameDates", []) or []):
        out.extend(block.get("games", []) or [])
    return out


def _basketball_team_metadata(team: dict) -> dict:
    city, name = str(team.get("teamCity") or ""), str(team.get("teamName") or "")
    return {
        "displayName": f"{city} {name}".strip(),
        "shortDisplayName": name,
        "name": name,
        "abbreviation": team.get("teamTricode"),
        "location": city,
        "teamCity": city,
        "teamName": name,
        "teamTricode": team.get("teamTricode"),
        "teamSlug": team.get("teamSlug"),
    }


def _basketball_team_index(games: list[dict]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for game in games:
        for key in ("homeTeam", "awayTeam"):
            team = game.get(key) or {}
            try:
                tid = int(team.get("teamId"))
            except (TypeError, ValueError):
                continue
            out.setdefault(tid, set()).update(_metadata_aliases(_basketball_team_metadata(team)))
    return out


def _resolve_team_id(name: str, index: dict[int, set[str]]) -> int | None:
    q = normalize(name)
    exact = [tid for tid, aliases in index.items() if q and q in aliases]
    if len(exact) == 1:
        return exact[0]
    wanted = _aliases(name)
    hits = [tid for tid, aliases in index.items() if wanted & aliases]
    return hits[0] if len(hits) == 1 else None


def basketball_team_model(
    sport: str,
    team_a: str,
    team_b: str,
    *,
    event_date: date,
) -> ModelEvidence | None:
    try:
        games = _basketball_games(_basketball_schedule(sport))
        index = _basketball_team_index(games)
        aid, bid = _resolve_team_id(team_a, index), _resolve_team_id(team_b, index)
        if aid is None or bid is None or aid == bid:
            return None
        stats = {
            aid: {"wins": 0, "losses": 0, "pf": 0.0, "pa": 0.0, "recent": []},
            bid: {"wins": 0, "losses": 0, "pf": 0.0, "pa": 0.0, "recent": []},
        }
        home_a: bool | None = None
        for game in games:
            try:
                gd = date.fromisoformat(str(game.get("gameDateEst") or "")[:10])
            except ValueError:
                continue
            home, away = game.get("homeTeam") or {}, game.get("awayTeam") or {}
            try:
                home_id, away_id = int(home.get("teamId")), int(away.get("teamId"))
            except (TypeError, ValueError):
                continue
            if gd == event_date and {home_id, away_id} == {aid, bid}:
                home_a = home_id == aid
            if gd >= event_date or int(game.get("gameStatus") or 0) != 3:
                continue
            if "preseason" in str(game.get("gameLabel") or "").lower():
                continue
            hs, aws = _safe_float(home.get("score")), _safe_float(away.get("score"))
            if hs is None or aws is None:
                continue
            for tid in (aid, bid):
                if tid not in {home_id, away_id}:
                    continue
                pf, pa = (hs, aws) if tid == home_id else (aws, hs)
                won = pf > pa
                stats[tid]["wins"] += int(won)
                stats[tid]["losses"] += int(not won)
                stats[tid]["pf"] += pf
                stats[tid]["pa"] += pa
                stats[tid]["recent"].append((gd, 1.0 if won else 0.0))

        a, b = stats[aid], stats[bid]
        ga, gb = int(a["wins"] + a["losses"]), int(b["wins"] + b["losses"])
        if min(ga, gb) < 5:
            return None
        recent_a = [v for _, v in sorted(a["recent"], key=lambda x: x[0])[-10:]]
        recent_b = [v for _, v in sorted(b["recent"], key=lambda x: x[0])[-10:]]
        return team_record_model(
            sport=sport,
            team_a=team_a,
            team_b=team_b,
            win_pct_a=_pct(int(a["wins"]), int(a["losses"])),
            win_pct_b=_pct(int(b["wins"]), int(b["losses"])),
            games_a=ga,
            games_b=gb,
            home_a=home_a,
            recent_pct_a=(sum(recent_a) / len(recent_a)) if recent_a else None,
            recent_pct_b=(sum(recent_b) / len(recent_b)) if recent_b else None,
            differential_per_game_a=(a["pf"] - a["pa"]) / ga,
            differential_per_game_b=(b["pf"] - b["pa"]) / gb,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None


def nba_prior_season_team_model(
    team_a: str,
    team_b: str,
) -> ModelEvidence | None:
    """Low-confidence season-open fallback from the completed 2025-26 season.

    This is independent team-performance evidence, not sportsbook consensus.
    It is used only when the current 2026-27 NBA schedule has too little
    completed-game evidence for the normal current-season model.
    """
    try:
        rows = nba_player_rows()
        aliases: dict[int, set[str]] = {}
        for row in rows:
            try:
                tid = int(row.get("team_id"))
            except (TypeError, ValueError):
                continue
            meta = {
                "displayName": row.get("team_display_name"),
                "shortDisplayName": row.get("team_short_display_name"),
                "name": row.get("team_name"),
                "abbreviation": row.get("team_abbreviation"),
                "location": row.get("team_location"),
            }
            aliases.setdefault(tid, set()).update(_metadata_aliases(meta))

        aid = _resolve_team_id(team_a, aliases)
        bid = _resolve_team_id(team_b, aliases)
        if aid is None or bid is None or aid == bid:
            return None

        stats = {
            aid: {"wins": 0, "losses": 0, "pf": 0.0, "pa": 0.0},
            bid: {"wins": 0, "losses": 0, "pf": 0.0, "pa": 0.0},
        }
        seen: set[tuple[str, int]] = set()
        for row in rows:
            try:
                tid = int(row.get("team_id"))
            except (TypeError, ValueError):
                continue
            if tid not in stats:
                continue
            # Keep the regular-season baseline clean when the source labels it.
            season_type = str(row.get("season_type") or "").strip()
            if season_type and season_type != "2":
                continue
            game_id = str(row.get("game_id") or "")
            key = (game_id, tid)
            if not game_id or key in seen:
                continue
            seen.add(key)

            pf = _safe_float(row.get("team_score"))
            pa = _safe_float(row.get("opponent_team_score"))
            if pf is None or pa is None:
                continue
            won_raw = str(row.get("team_winner") or "").strip().lower()
            won = won_raw in {"1", "true", "t", "yes"} or pf > pa
            stats[tid]["wins"] += int(won)
            stats[tid]["losses"] += int(not won)
            stats[tid]["pf"] += pf
            stats[tid]["pa"] += pa

        a, b = stats[aid], stats[bid]
        ga = int(a["wins"] + a["losses"])
        gb = int(b["wins"] + b["losses"])
        if min(ga, gb) < 20:
            return None

        base = team_record_model(
            sport="NBA",
            team_a=team_a,
            team_b=team_b,
            win_pct_a=_pct(int(a["wins"]), int(a["losses"])),
            win_pct_b=_pct(int(b["wins"]), int(b["losses"])),
            games_a=ga,
            games_b=gb,
            home_a=None,
            differential_per_game_a=(a["pf"] - a["pa"]) / ga,
            differential_per_game_b=(b["pf"] - b["pa"]) / gb,
        )
        return replace(
            base,
            model_name="NBA prior-season team-strength baseline",
            confidence=min(base.confidence, 0.52),
            warnings=tuple(base.warnings) + (
                "2025-26 baseline; current 2026-27 season sample not established",
            ),
        )
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None


def team_game_model(
    sport: str,
    team_a: str,
    team_b: str,
    *,
    event_date: date,
) -> ModelEvidence | None:
    if sport == "MLB":
        return mlb_team_model(team_a, team_b, event_date=event_date)
    if sport == "NFL":
        return nfl_team_model(team_a, team_b, event_date=event_date)
    if sport == "WNBA":
        return wnba_team_model(team_a, team_b, event_date=event_date)
    if sport == "NBA":
        current = basketball_team_model(sport, team_a, team_b, event_date=event_date)
        if current is not None:
            return current
        return nba_prior_season_team_model(team_a, team_b)
    return None
