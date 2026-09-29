from __future__ import annotations

import csv
from datetime import date
from functools import lru_cache
import io

import requests

from sports_edge.models.game_scope import normalize


_RELEASE_TAG = {
    "NBA": "espn_nba_player_boxscores",
    "WNBA": "espn_wnba_player_boxscores",
}


def _url(sport: str, season: int) -> str:
    tag = _RELEASE_TAG[str(sport).upper()]
    league = "nba" if str(sport).upper() == "NBA" else "wnba"
    return (
        "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
        f"{tag}/player_box_{int(season)}.csv"
    )


@lru_cache(maxsize=8)
def player_box_rows(sport: str, season: int) -> tuple[dict, ...]:
    sport = str(sport).upper()
    if sport not in _RELEASE_TAG:
        return ()
    r = requests.get(
        _url(sport, int(season)),
        headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "text/csv,*/*"},
        timeout=45,
    )
    r.raise_for_status()
    return tuple(csv.DictReader(io.StringIO(r.text)))


def _row_date(row: dict) -> date | None:
    raw = str(row.get("game_date") or "")[:10]
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def _f(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def usable_player_row(row: dict) -> bool:
    if str(row.get("season_type") or "") not in {"2", "3", "2.0", "3.0"}:
        return False
    if _truthy(row.get("did_not_play")):
        return False
    minutes = _f(row.get("minutes"))
    return minutes is not None and minutes > 0


def resolve_player_name(
    sport: str,
    player_name: str,
    *,
    seasons: tuple[int, ...],
) -> str | None:
    q = normalize(player_name)
    if not q:
        return None

    exact: set[str] = set()
    loose: set[str] = set()
    for season in seasons:
        for row in player_box_rows(sport, season):
            display = str(row.get("athlete_display_name") or "").strip()
            if not display:
                continue
            n = normalize(display)
            if n == q:
                exact.add(display)
            elif q and (
                n.endswith(" " + q)
                or q.endswith(" " + n)
                or normalize(row.get("athlete_short_name")) == q
            ):
                loose.add(display)

    if len(exact) == 1:
        return next(iter(exact))
    if len(exact) > 1:
        return sorted(exact)[0]
    return next(iter(loose)) if len(loose) == 1 else None


@lru_cache(maxsize=512)
def player_history(
    sport: str,
    player_name: str,
    season: int,
    before_iso: str,
) -> tuple[dict, ...]:
    before = date.fromisoformat(before_iso)
    resolved = resolve_player_name(
        sport,
        player_name,
        seasons=(int(season), int(season) - 1),
    )
    if not resolved:
        return ()

    wanted = normalize(resolved)
    out: list[dict] = []
    for row in player_box_rows(sport, int(season)):
        if normalize(row.get("athlete_display_name")) != wanted:
            continue
        gd = _row_date(row)
        if gd is None or gd >= before:
            continue
        if not usable_player_row(row):
            continue
        out.append(dict(row))
    out.sort(key=lambda r: (str(r.get("game_date") or ""), str(r.get("game_id") or "")))
    return tuple(out)


@lru_cache(maxsize=512)
def prior_player_history(
    sport: str,
    player_name: str,
    season: int,
) -> tuple[dict, ...]:
    resolved = resolve_player_name(
        sport,
        player_name,
        seasons=(int(season), int(season) - 1),
    )
    if not resolved:
        return ()
    wanted = normalize(resolved)
    out = [
        dict(row)
        for row in player_box_rows(sport, int(season) - 1)
        if normalize(row.get("athlete_display_name")) == wanted
        and usable_player_row(row)
    ]
    out.sort(key=lambda r: (str(r.get("game_date") or ""), str(r.get("game_id") or "")))
    return tuple(out)


def latest_team_abbreviation(rows: tuple[dict, ...] | list[dict]) -> str:
    if not rows:
        return ""
    return str(rows[-1].get("team_abbreviation") or "").strip().upper()


def latest_team_name(rows: tuple[dict, ...] | list[dict]) -> str:
    if not rows:
        return ""
    return str(rows[-1].get("team_display_name") or "").strip()


def stat_value(row: dict, metric: str) -> float | None:
    if metric == "points":
        return _f(row.get("points"))
    if metric == "rebounds":
        return _f(row.get("rebounds"))
    if metric == "assists":
        return _f(row.get("assists"))
    if metric == "threes":
        return _f(row.get("three_point_field_goals_made"))
    if metric == "pra":
        pts = _f(row.get("points"))
        reb = _f(row.get("rebounds"))
        ast = _f(row.get("assists"))
        if None in (pts, reb, ast):
            return None
        return float(pts + reb + ast)
    raise ValueError(f"Unsupported basketball metric: {metric}")


def minutes_value(row: dict) -> float | None:
    return _f(row.get("minutes"))
