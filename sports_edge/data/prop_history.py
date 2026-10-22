"""Causal, descriptive player-prop game logs for the research workspace.

This module reports observed historical box-score outcomes only. It does not
turn hit rates into next-game probabilities or declare betting value.
Unsupported markets and incomplete source joins fail closed.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import re
from typing import Any


_TICKER_DATE_RE = re.compile(r"(?:^|-)(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})", re.I)
_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def event_date_from_ticker(value: str | None) -> date | None:
    match = _TICKER_DATE_RE.search(str(value or "").upper())
    if not match:
        return None
    yy, month, dd = match.groups()
    try:
        return date(2000 + int(yy), _MONTHS[month], int(dd))
    except ValueError:
        return None


def market_player_and_threshold(market: dict) -> tuple[str | None, float | None]:
    """Extract only explicit player-prefix and milestone information."""
    title = str((market or {}).get("title") or "").strip()
    if ":" not in title:
        return None, None
    player = title.split(":", 1)[0].strip()
    if not player:
        return None, None
    floor = _number((market or {}).get("floor_strike"))
    if floor is not None:
        threshold = float(int(floor + 0.5))
    else:
        match = re.search(r":\s*(?:Over\s+)?(\d+(?:\.\d+)?)\s*\+", title, re.I)
        threshold = float(match.group(1)) if match else None
    return player, threshold


def _number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _hit_summary(rows: list[dict], threshold: float | None) -> dict:
    eligible = [row for row in rows if row.get("value") is not None]
    if threshold is None or not eligible:
        return {"hits": None, "sample_size": len(eligible), "hit_rate": None}
    hits = sum(float(row["value"]) >= threshold for row in eligible)
    return {"hits": hits, "sample_size": len(eligible), "hit_rate": hits / len(eligible)}


def _report(*, sport: str, player: str, market: str, threshold: float | None,
            source: str, rows: list[dict], season_used: str, note: str) -> dict:
    all_rows = sorted(rows, key=lambda row: str(row.get("date") or ""))
    display_rows = all_rows[-20:]
    recent = all_rows[-5:]
    home_rows = [row for row in all_rows if row.get("home_away") == "Home"]
    away_rows = [row for row in all_rows if row.get("home_away") == "Away"]
    return {
        "status": "AVAILABLE" if all_rows else "NO_HISTORY",
        "sport": sport,
        "player": player,
        "market": market,
        "threshold": threshold,
        "source": source,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "season_used": season_used,
        "sample_size": len(all_rows),
        "date_from": all_rows[0].get("date") if all_rows else None,
        "date_to": all_rows[-1].get("date") if all_rows else None,
        "season_hit_rate": _hit_summary(all_rows, threshold),
        "last5_hit_rate": _hit_summary(recent, threshold),
        "home_hit_rate": _hit_summary(home_rows, threshold),
        "away_hit_rate": _hit_summary(away_rows, threshold),
        "rows": display_rows,
        "note": note,
    }


def prop_history_report(*, sport: str, player_name: str, market: str,
                        event_date: date, threshold: float | None,
                        event_ticker: str | None = None) -> dict:
    """Return last-20 causal game logs for supported, official/free datasets."""
    sport = str(sport).upper()
    market = str(market).strip()
    if not player_name.strip():
        return {"status": "MISSING_PLAYER", "note": "Player name could not be resolved from the selected contract.", "rows": []}
    if not isinstance(event_date, date):
        return {"status": "MISSING_EVENT_DATE", "note": "A trustworthy event date is required; close time is not substituted.", "rows": []}

    if sport in {"NBA", "WNBA"}:
        from sports_edge.data.basketball_prop_data import player_history, prior_player_history, stat_value, latest_team_abbreviation
        season = event_date.year if event_date.month >= 8 else event_date.year - 1
        metric_map = {
            "Points": "points", "Rebounds": "rebounds", "Assists": "assists",
            "Three-Pointers": "threes", "3-Pointers": "threes",
            "Points + Rebounds + Assists": "pra", "PRA": "pra",
        }
        metric = metric_map.get(market)
        if metric is None:
            return {"status": "UNSUPPORTED_MARKET", "note": f"No audited box-score mapping exists for {sport} {market}.", "rows": []}
        try:
            history = player_history(sport, player_name, season, event_date.isoformat())
            season_used = str(season)
            if not history:
                history = prior_player_history(sport, player_name, season)
                season_used = f"{season - 1} prior season (current-season log empty)"
        except Exception as exc:
            return {"status": "SOURCE_UNAVAILABLE", "note": f"Official/free basketball history request failed ({type(exc).__name__}).", "rows": []}
        rows = []
        for raw in history:
            value = stat_value(raw, metric)
            if value is None:
                continue
            date_value = str(raw.get("game_date") or "")[:10]
            if not date_value or date_value >= event_date.isoformat():
                continue
            home_away_raw = str(raw.get("home_away") or raw.get("homeAway") or "").lower()
            is_home = raw.get("is_home")
            if is_home is None:
                is_home = raw.get("isHome")
            home_away = "Home" if str(is_home).lower() in {"true", "1"} or home_away_raw in {"home", "h"} else (
                "Away" if str(is_home).lower() in {"false", "0"} or home_away_raw in {"away", "a"} else "Unknown"
            )
            rows.append({
                "date": date_value,
                "opponent": raw.get("opponent_display_name") or raw.get("opponent_abbreviation") or raw.get("opponent") or "Unknown",
                "home_away": home_away,
                "value": float(value),
                "game_id": raw.get("game_id") or raw.get("gameId"),
                "team": raw.get("team_abbreviation") or latest_team_abbreviation([raw]) or "Unknown",
            })
        return _report(sport=sport, player=player_name, market=market, threshold=threshold,
                       source="SportsDataverse player box-score releases", rows=rows,
                       season_used=season_used, note="Descriptive historical hit rates only; no injury, projected-minutes, lineup or opponent adjustment is implied. A prior-season fallback is labeled explicitly and is not a current-season hit rate.")

    if sport == "NFL":
        from sports_edge.data.nfl_prop_data import player_context, schedule_rows
        metric_map = {
            "Passing Yards": ("passing_yards",), "Passing TDs": ("passing_tds",),
            "Pass Attempts": ("attempts",), "Pass Completions": ("completions",),
            "Pass Interceptions": ("interceptions",), "Rushing Yards": ("rushing_yards",),
            "Rush Attempts": ("carries",), "Rushing + Receiving Yards": ("rushing_yards", "receiving_yards"),
            "Receiving Yards": ("receiving_yards",), "Receptions": ("receptions",),
            "Player Touchdowns": ("rushing_tds", "receiving_tds"),
        }
        fields = metric_map.get(market)
        if fields is None:
            return {"status": "UNSUPPORTED_MARKET", "note": f"No audited box-score mapping exists for NFL {market}.", "rows": []}
        try:
            context = player_context(player_name, event_date, event_ticker)
            schedule = schedule_rows()
        except Exception as exc:
            return {"status": "SOURCE_UNAVAILABLE", "note": f"nflverse history request failed ({type(exc).__name__}).", "rows": []}
        if context is None:
            return {"status": "PLAYER_UNRESOLVED", "note": "Player identity or causal history could not be resolved uniquely.", "rows": []}
        schedule_by_id = {str(game.get("game_id") or ""): game for game in schedule}
        history_source_rows = context.current_rows
        season_used = str(context.event_date.year)
        if not history_source_rows:
            history_source_rows = context.prior_rows
            season_used = f"{context.event_date.year - 1} prior season (current-season log empty)"
        rows = []
        for raw in history_source_rows:
            day = str(raw.get("date") or "")[:10]
            if not day or day >= event_date.isoformat():
                continue
            values = [_number(raw.get(field)) for field in fields]
            if any(value is None for value in values):
                continue
            game = schedule_by_id.get(str(raw.get("game_id") or ""), {})
            team = str(raw.get("team") or "")
            home = str(game.get("home_team") or "")
            away = str(game.get("away_team") or "")
            opponent = away if team == home else home if team == away else "Unknown"
            rows.append({"date": day, "opponent": opponent, "home_away": "Home" if team == home else "Away" if team == away else "Unknown",
                         "value": sum(values), "game_id": raw.get("game_id"), "team": team})
        return _report(sport=sport, player=context.player_name, market=market, threshold=threshold,
                       source="nflverse player-week stats + official schedule", rows=rows,
                       season_used=season_used, note="Observed causal games only. A prior-season fallback is labeled explicitly and is not a current-season hit rate. Passing TDs are not counted as player touchdown-scorer outcomes; player touchdowns sum rushing and receiving scores only.")

    if sport == "MLB":
        from sports_edge.data.mlb_prop_data import resolve_player, game_log_stats
        season = event_date.year if event_date.month >= 3 else event_date.year - 1
        pitching = market in {"Strikeouts", "Pitcher Strikeouts"}
        group = "pitching" if pitching else "hitting"
        metric_map = {
            "Hits": ("hits",), "Home Runs": ("homeRuns",), "Total Bases": ("totalBases",),
            "RBIs": ("rbi",), "Runs": ("runs",), "Hits + Runs + RBIs": ("hits", "runs", "rbi"),
            "Strikeouts": ("strikeOuts",), "Pitcher Strikeouts": ("strikeOuts",),
        }
        fields = metric_map.get(market)
        if fields is None:
            return {"status": "UNSUPPORTED_MARKET", "note": f"No audited box-score mapping exists for MLB {market}.", "rows": []}
        try:
            person = resolve_player(player_name, season)
            if person is None:
                return {"status": "PLAYER_UNRESOLVED", "note": "MLB player identity could not be resolved uniquely.", "rows": []}
            logs = game_log_stats(int(person["id"]), group, season)
            season_used = str(season)
            if not logs:
                logs = game_log_stats(int(person["id"]), group, season - 1)
                season_used = f"{season - 1} prior season (current-season log empty)"
        except Exception as exc:
            return {"status": "SOURCE_UNAVAILABLE", "note": f"MLB Stats API history request failed ({type(exc).__name__}).", "rows": []}
        rows = []
        for raw in logs:
            day = str(raw.get("_date") or "")[:10]
            if not day or day >= event_date.isoformat():
                continue
            values = [_number(raw.get(field)) for field in fields]
            if any(value is None for value in values):
                continue
            rows.append({"date": day, "opponent": raw.get("_opponent") or "Unknown",
                         "home_away": "Home" if str(raw.get("_is_home")).lower() in {"true", "1"} else "Away" if str(raw.get("_is_home")).lower() in {"false", "0"} else "Unknown",
                         "value": sum(values), "game_id": raw.get("_game_pk"), "team": person.get("currentTeam", {}).get("name", "Unknown")})
        return _report(sport=sport, player=person.get("fullName") or player_name, market=market, threshold=threshold,
                       source="Official MLB Stats API", rows=rows, season_used=season_used,
                       note="Observed game logs only. A prior-season fallback is labeled explicitly; it is not a current-season hit rate.")

    return {"status": "UNSUPPORTED_SPORT", "note": f"No audited game-log adapter is available for {sport} {market}.", "rows": []}
