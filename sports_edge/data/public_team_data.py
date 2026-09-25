from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Any

import requests

from sports_edge.models.game_scope import normalize
from sports_edge.models.model_evidence import ModelEvidence, team_record_model


ESPN_SLUGS = {
    "NFL": ("football", "nfl"),
    "NBA": ("basketball", "nba"),
    "WNBA": ("basketball", "wnba"),
}


def _aliases(name: str) -> set[str]:
    n = normalize(name)
    if not n:
        return set()
    parts = n.split()
    out = {n}
    if len(parts) >= 2:
        out.add(parts[-1])
    if len(parts) >= 3:
        out.add(" ".join(parts[-2:]))
    return {x for x in out if len(x) >= 3}


def _same_team(a: str, b: str) -> bool:
    aa = _aliases(a)
    bb = _aliases(b)
    return bool(aa and bb and (aa & bb))


def _record(summary: str | None) -> tuple[int, int] | None:
    raw = str(summary or "").strip()
    parts = raw.split("-")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _pct(wins: int, losses: int) -> float:
    total = wins + losses
    return wins / total if total > 0 else 0.5


@lru_cache(maxsize=128)
def _espn_scoreboard(sport: str, yyyymmdd: str) -> dict:
    category, league = ESPN_SLUGS[sport]
    url = f"https://site.api.espn.com/apis/site/v2/sports/{category}/{league}/scoreboard"
    r = requests.get(
        url,
        params={"dates": yyyymmdd},
        headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "application/json"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def _overall_record(competitor: dict) -> tuple[int, int] | None:
    for rec in competitor.get("records", []) or []:
        if str(rec.get("name") or "").lower() == "overall" or str(rec.get("type") or "").lower() == "total":
            parsed = _record(rec.get("summary"))
            if parsed:
                return parsed
    return None


def espn_team_model(
    sport: str,
    team_a: str,
    team_b: str,
    *,
    event_date: date,
) -> ModelEvidence | None:
    if sport not in ESPN_SLUGS:
        return None
    try:
        payload = _espn_scoreboard(sport, event_date.strftime("%Y%m%d"))
    except requests.RequestException:
        return None

    for event in payload.get("events", []) or []:
        comps = event.get("competitions", []) or []
        if not comps:
            continue
        competitors = comps[0].get("competitors", []) or []
        if len(competitors) < 2:
            continue

        by_name = {}
        for comp in competitors:
            team = comp.get("team") or {}
            display = str(team.get("displayName") or team.get("name") or "").strip()
            if display:
                by_name[display] = comp

        match_a = next((comp for name, comp in by_name.items() if _same_team(team_a, name)), None)
        match_b = next((comp for name, comp in by_name.items() if _same_team(team_b, name)), None)
        if match_a is None or match_b is None:
            continue

        ra = _overall_record(match_a)
        rb = _overall_record(match_b)
        if not ra or not rb:
            return None
        wa, la = ra
        wb, lb = rb
        return team_record_model(
            sport=sport,
            team_a=team_a,
            team_b=team_b,
            win_pct_a=_pct(wa, la),
            win_pct_b=_pct(wb, lb),
            games_a=wa + la,
            games_b=wb + lb,
            home_a=str(match_a.get("homeAway") or "").lower() == "home",
        )
    return None


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
        params={
            "leagueId": "103,104",
            "season": season,
            "standingsTypes": "regularSeason",
        },
        headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "application/json"},
        timeout=10,
    )
    r.raise_for_status()
    rows: list[dict] = []
    for block in r.json().get("records", []) or []:
        rows.extend(block.get("teamRecords", []) or [])
    # division blocks repeat the same teams; last copy is equivalent enough.
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
        names = {
            str(team.get("name") or ""),
            str(team.get("teamName") or ""),
            str(team.get("locationName") or ""),
            str(team.get("shortName") or ""),
            str(team.get("clubName") or ""),
        }
        aliases: set[str] = set()
        for name in names:
            aliases.update(_aliases(name))
        out[tid] = aliases
    return out


def _mlb_match_team_id(name: str, aliases: dict[int, set[str]]) -> int | None:
    wanted = _aliases(name)
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


def mlb_team_model(
    team_a: str,
    team_b: str,
    *,
    event_date: date,
) -> ModelEvidence | None:
    season = event_date.year
    try:
        aliases = _mlb_team_names(season)
        aid = _mlb_match_team_id(team_a, aliases)
        bid = _mlb_match_team_id(team_b, aliases)
        if aid is None or bid is None:
            return None

        standings = {int((r.get("team") or {}).get("id")): r for r in _mlb_standings(season)}
        a = standings.get(aid)
        b = standings.get(bid)
        if not a or not b:
            return None

        home_a: bool | None = None
        for game in _mlb_schedule(event_date.isoformat()):
            teams = game.get("teams") or {}
            home_id = int((((teams.get("home") or {}).get("team") or {}).get("id")) or 0)
            away_id = int((((teams.get("away") or {}).get("team") or {}).get("id")) or 0)
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


def team_game_model(
    sport: str,
    team_a: str,
    team_b: str,
    *,
    event_date: date,
) -> ModelEvidence | None:
    if sport == "MLB":
        return mlb_team_model(team_a, team_b, event_date=event_date)
    if sport in ESPN_SLUGS:
        return espn_team_model(sport, team_a, team_b, event_date=event_date)
    return None
