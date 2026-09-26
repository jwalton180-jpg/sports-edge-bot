from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from statistics import median
from typing import Any

from sports_edge.core.math import american_to_implied, clamp


PROP_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "MLB": {
        "Hits": ("batter_hits", "batter_hits_alternate"),
        "Home Runs": ("batter_home_runs", "batter_home_runs_alternate"),
        "Pitcher Strikeouts": ("pitcher_strikeouts", "pitcher_strikeouts_alternate"),
        "Total Bases": ("batter_total_bases", "batter_total_bases_alternate"),
        "RBIs": ("batter_rbis", "batter_rbis_alternate"),
    },
    "NFL": {
        "Passing": ("player_pass_yds", "player_pass_tds", "player_pass_attempts", "player_pass_completions", "player_pass_interceptions"),
        "Rushing": ("player_rush_yds", "player_rush_attempts", "player_rush_tds", "player_rush_reception_yds"),
        "Receiving": ("player_receptions", "player_reception_yds", "player_reception_tds"),
        "Touchdowns": ("player_anytime_td", "player_tds", "player_1st_td"),
    },
    "NBA": {
        "Points": ("player_points", "player_points_alternate"),
        "Rebounds": ("player_rebounds", "player_rebounds_alternate"),
        "Assists": ("player_assists", "player_assists_alternate"),
        "Threes": ("player_threes", "player_threes_alternate"),
        "PRA": ("player_points_rebounds_assists", "player_points_rebounds_assists_alternate"),
        "Steals": ("player_steals", "player_steals_alternate"),
        "Blocks": ("player_blocks", "player_blocks_alternate"),
        "Double Double": ("player_double_double",),
        "Triple Double": ("player_triple_double",),
    },
    "WNBA": {
        "Points": ("player_points", "player_points_alternate"),
        "Rebounds": ("player_rebounds", "player_rebounds_alternate"),
        "Assists": ("player_assists", "player_assists_alternate"),
        "Threes": ("player_threes", "player_threes_alternate"),
        "PRA": ("player_points_rebounds_assists", "player_points_rebounds_assists_alternate"),
        "Steals": ("player_steals", "player_steals_alternate"),
        "Blocks": ("player_blocks", "player_blocks_alternate"),
        "Double Double": ("player_double_double",),
        "Triple Double": ("player_triple_double",),
    },
}


MARKET_LABELS = {
    "batter_hits": "Hits",
    "batter_hits_alternate": "Hits",
    "batter_home_runs": "Home Runs",
    "batter_home_runs_alternate": "Home Runs",
    "pitcher_strikeouts": "Pitcher Strikeouts",
    "pitcher_strikeouts_alternate": "Pitcher Strikeouts",
    "batter_total_bases": "Total Bases",
    "batter_total_bases_alternate": "Total Bases",
    "batter_rbis": "RBIs",
    "batter_rbis_alternate": "RBIs",
    "player_pass_yds": "Passing Yards",
    "player_pass_tds": "Passing TDs",
    "player_pass_attempts": "Pass Attempts",
    "player_pass_completions": "Pass Completions",
    "player_pass_interceptions": "Pass Interceptions",
    "player_rush_yds": "Rushing Yards",
    "player_rush_attempts": "Rush Attempts",
    "player_rush_reception_yds": "Rushing + Receiving Yards",
    "player_rush_tds": "Rushing TDs",
    "player_receptions": "Receptions",
    "player_reception_yds": "Receiving Yards",
    "player_reception_tds": "Receiving TDs",
    "player_anytime_td": "Anytime TD",
    "player_tds": "Touchdowns",
    "player_1st_td": "First TD",
    "player_points": "Points",
    "player_points_alternate": "Points",
    "player_rebounds": "Rebounds",
    "player_rebounds_alternate": "Rebounds",
    "player_assists": "Assists",
    "player_assists_alternate": "Assists",
    "player_threes": "Threes",
    "player_threes_alternate": "Threes",
    "player_points_rebounds_assists": "Points + Rebounds + Assists",
    "player_points_rebounds_assists_alternate": "Points + Rebounds + Assists",
    "player_steals": "Steals",
    "player_steals_alternate": "Steals",
    "player_blocks": "Blocks",
    "player_blocks_alternate": "Blocks",
    "player_double_double": "Double Double",
    "player_triple_double": "Triple Double",
}


@dataclass(frozen=True)
class PropConsensus:
    market_key: str
    market_label: str
    player: str
    side: str
    line: float | None
    fair_probability: float
    book_count: int
    median_age_s: float
    median_price: float
    warnings: tuple[str, ...]


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _age(value: Any, now: datetime) -> float:
    dt = _parse_iso(value)
    if dt is None:
        return float("inf")
    return max(0.0, (now - dt).total_seconds())


def _player_name(outcome: dict) -> str:
    description = str(outcome.get("description") or "").strip()
    name = str(outcome.get("name") or "").strip()
    if description:
        return description
    if name.lower() not in {"over", "under", "yes", "no"}:
        return name
    return ""


def _pair_key(outcome: dict) -> tuple[str, float | None]:
    return _player_name(outcome), outcome.get("point")


def prop_consensus(
    payload: dict,
    *,
    market_keys: tuple[str, ...] | list[str],
    now: datetime | None = None,
    max_age_s: float = 120.0,
    min_books: int = 2,
) -> list[PropConsensus]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    wanted = set(market_keys)

    samples: dict[tuple[str, str, float | None, str], list[tuple[float, float, float]]] = {}

    for bookmaker in payload.get("bookmakers", []) or []:
        for market in bookmaker.get("markets", []) or []:
            mkey = str(market.get("key") or "")
            if mkey not in wanted:
                continue
            age_s = _age(market.get("last_update") or bookmaker.get("last_update"), now)
            if not math.isfinite(age_s) or age_s > max_age_s:
                continue

            outcomes = market.get("outcomes", []) or []
            grouped: dict[tuple[str, float | None], list[dict]] = {}
            for outcome in outcomes:
                grouped.setdefault(_pair_key(outcome), []).append(outcome)

            for (player, point), pair in grouped.items():
                if not player:
                    continue

                # No-vig when a two-sided pair is available.
                parsed: list[tuple[dict, float]] = []
                for outcome in pair:
                    try:
                        price = float(outcome.get("price"))
                        implied = american_to_implied(price)
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue
                    parsed.append((outcome, implied))

                if not parsed:
                    continue
                total = sum(p for _, p in parsed)
                if len(parsed) >= 2 and total > 0:
                    probabilities = [(o, p / total) for o, p in parsed]
                else:
                    probabilities = parsed

                for outcome, fair in probabilities:
                    side = str(outcome.get("name") or "").strip()
                    try:
                        price = float(outcome.get("price"))
                    except (TypeError, ValueError):
                        continue
                    key = (mkey, player, point, side)
                    samples.setdefault(key, []).append((clamp(fair), age_s, price))

    rows: list[PropConsensus] = []
    for (mkey, player, point, side), values in samples.items():
        probs = [v[0] for v in values]
        ages = [v[1] for v in values]
        prices = [v[2] for v in values]
        warnings: list[str] = []
        if len(values) < min_books:
            warnings.append(f"Only {len(values)} fresh book(s)")
        rows.append(
            PropConsensus(
                market_key=mkey,
                market_label=MARKET_LABELS.get(mkey, mkey),
                player=player,
                side=side,
                line=float(point) if point is not None else None,
                fair_probability=float(median(probs)),
                book_count=len(values),
                median_age_s=float(median(ages)),
                median_price=float(median(prices)),
                warnings=tuple(warnings),
            )
        )

    return sorted(
        rows,
        key=lambda r: (not r.warnings, r.book_count, r.fair_probability),
        reverse=True,
    )


def group_markets_for_sport(sport: str) -> tuple[str, ...]:
    groups = PROP_GROUPS.get(sport, {})
    out: list[str] = []
    for keys in groups.values():
        out.extend(keys)
    return tuple(dict.fromkeys(out))
