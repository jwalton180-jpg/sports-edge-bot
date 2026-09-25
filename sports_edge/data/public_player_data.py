from __future__ import annotations

import csv
import io
import unicodedata
from functools import lru_cache
from typing import Iterable

import requests


NFL_WEEKLY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{year}.csv"
)
NBA_PLAYER_BOX_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_nba_player_boxscores/player_box_2026.csv"
)
WNBA_PLAYER_BOX_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "wnba_stats_player_boxscores/player_boxscores_2026.csv"
)
MLB_STATS_URL = "https://statsapi.mlb.com/api/v1/stats"


def normalize_person(value: str | None) -> str:
    raw = unicodedata.normalize("NFKD", str(value or ""))
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    chars = [ch.lower() if ch.isalnum() else " " for ch in raw]
    tokens = "".join(chars).split()
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    if tokens and tokens[-1] in suffixes:
        tokens = tokens[:-1]
    return " ".join(tokens)


def person_aliases(value: str | None) -> tuple[str, ...]:
    name = normalize_person(value)
    if not name:
        return ()
    tokens = name.split()
    out = {name}
    if len(tokens) >= 3:
        out.add(f"{tokens[0]} {tokens[-1]}")
    if len(tokens) == 2:
        out.add(f"{tokens[1]} {tokens[0]}")
    return tuple(sorted(out))


def _request_csv(url: str, *, timeout: int = 25) -> tuple[dict, ...]:
    r = requests.get(
        url,
        headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "text/csv,*/*"},
        timeout=timeout,
    )
    r.raise_for_status()
    return tuple(csv.DictReader(io.StringIO(r.text)))


@lru_cache(maxsize=4)
def nfl_weekly_rows(year: int) -> tuple[dict, ...]:
    return _request_csv(NFL_WEEKLY_URL.format(year=year))


@lru_cache(maxsize=2)
def nba_player_rows() -> tuple[dict, ...]:
    return _request_csv(NBA_PLAYER_BOX_URL, timeout=35)


@lru_cache(maxsize=2)
def wnba_player_rows() -> tuple[dict, ...]:
    return _request_csv(WNBA_PLAYER_BOX_URL, timeout=35)


@lru_cache(maxsize=4)
def mlb_season_rows(group: str, season: int) -> tuple[dict, ...]:
    r = requests.get(
        MLB_STATS_URL,
        params={
            "stats": "season",
            "group": group,
            "season": season,
            "sportIds": 1,
            "playerPool": "ALL",
            "limit": 2000,
        },
        headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "application/json"},
        timeout=25,
    )
    r.raise_for_status()
    payload = r.json()
    stats = payload.get("stats") or []
    if not stats:
        return ()
    out: list[dict] = []
    for split in stats[0].get("splits", []) or []:
        player = split.get("player") or {}
        stat = split.get("stat") or {}
        row = {
            "player_name": player.get("fullName"),
            "player_id": player.get("id"),
            "team_name": (split.get("team") or {}).get("name"),
            "season": split.get("season"),
        }
        row.update(stat)
        out.append(row)
    return tuple(out)


def unique_person_index(rows: Iterable[dict], name_getter) -> dict[str, dict]:
    """Index exact and conservative aliases; ambiguous aliases are removed."""
    exact: dict[str, dict] = {}
    aliases: dict[str, list[dict]] = {}
    for row in rows:
        name = str(name_getter(row) or "").strip()
        key = normalize_person(name)
        if not key:
            continue
        exact[key] = row
        for alias in person_aliases(name):
            aliases.setdefault(alias, []).append(row)

    out = dict(exact)
    for alias, hits in aliases.items():
        unique_ids = {id(row) for row in hits}
        if len(unique_ids) == 1 and alias not in out:
            out[alias] = hits[0]
    return out


def resolve_person(index: dict[str, dict], name: str | None) -> dict | None:
    exact = normalize_person(name)
    if exact in index:
        return index[exact]
    hits: list[dict] = []
    seen: set[int] = set()
    for alias in person_aliases(name):
        row = index.get(alias)
        if row is not None and id(row) not in seen:
            hits.append(row)
            seen.add(id(row))
    return hits[0] if len(hits) == 1 else None
