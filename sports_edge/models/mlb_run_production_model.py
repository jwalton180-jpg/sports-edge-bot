from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import exp, lgamma, log
from statistics import mean, pvariance

import requests

from sports_edge.core.math import clamp
from sports_edge.data.mlb_prop_data import (
    date_range_stat,
    find_player_game,
    game_context_for_hitter,
    game_log_stats,
    resolve_player,
    season_stat,
)
from sports_edge.models.model_evidence import ModelEvidence


LEAGUE_RBI_PER_PA = 0.105
LEAGUE_HRR_PER_PA = 0.445
LEAGUE_ERA = 4.20
LEAGUE_WHIP = 1.30


@dataclass(frozen=True)
class MLBRunProductionProjection:
    evidence: ModelEvidence
    player_name: str
    market_key: str
    market_label: str
    milestone: int
    line: float
    game_title: str
    probable_pitcher_name: str | None
    expected_plate_appearances: float
    projected_mean: float
    projected_variance: float


def _f(stat: dict | None, key: str) -> float | None:
    if not stat:
        return None
    value = stat.get(key)
    if value in (None, "", "-.--"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(stat: dict, kind: str) -> float:
    if kind == "rbi":
        return max(0.0, _f(stat, "rbi") or 0.0)
    return max(
        0.0,
        (_f(stat, "hits") or 0.0)
        + (_f(stat, "runs") or 0.0)
        + (_f(stat, "rbi") or 0.0),
    )


def _poisson_tail(lam: float, k: int) -> float:
    lam = clamp(lam, 0.01, 12.0)
    if k <= 0:
        return 1.0
    term = exp(-lam)
    cdf = term
    for i in range(1, k):
        term *= lam / i
        cdf += term
    return clamp(1.0 - cdf, 0.005, 0.995)


def _negative_binomial_tail(mu: float, variance: float, k: int) -> float:
    if k <= 0:
        return 1.0
    mu = max(0.01, mu)
    variance = max(mu + 1e-6, variance)
    if variance <= mu * 1.03:
        return _poisson_tail(mu, k)

    r = mu * mu / (variance - mu)
    p = r / (r + mu)
    cdf = 0.0
    for x in range(k):
        log_pmf = (
            lgamma(x + r)
            - lgamma(r)
            - lgamma(x + 1)
            + r * log(p)
            + x * log(1.0 - p)
        )
        cdf += exp(log_pmf)
    return clamp(1.0 - cdf, 0.005, 0.995)


def _pregame_logs(player_id: int, season: int, event_date: date) -> list[dict]:
    rows = []
    for row in game_log_stats(player_id, "hitting", season):
        raw_date = str(row.get("_date") or "")
        if raw_date and raw_date >= event_date.isoformat():
            continue
        # Exclude appearances without a batting opportunity.
        pa = _f(row, "plateAppearances")
        ab = _f(row, "atBats")
        if (pa or 0.0) <= 0 and (ab or 0.0) <= 0:
            continue
        rows.append(row)
    return rows


def _pitcher_environment(
    pitcher_id: int | None,
    pitcher_name: str | None,
    season: int,
) -> tuple[float, float, float, list[str]]:
    if not pitcher_id:
        return 1.0, 0.0, 0.0, ["probable starter unavailable; run environment held at league prior"]

    stat = season_stat(int(pitcher_id), "pitching", season)
    era = _f(stat, "era")
    whip = _f(stat, "whip")
    innings = _f(stat, "inningsPitched") or 0.0
    if era is None and whip is None:
        return 1.0, innings, 0.0, ["probable starter ERA/WHIP unavailable"]

    components = []
    if era is not None:
        components.append(clamp(era / LEAGUE_ERA, 0.55, 1.65))
    if whip is not None:
        components.append(clamp(whip / LEAGUE_WHIP, 0.60, 1.55))
    raw = mean(components)
    reliability = clamp(innings / 100.0, 0.0, 1.0)
    # Starter context affects only part of the hitter's game, so shrink hard.
    factor = 1.0 + 0.28 * reliability * (raw - 1.0)
    factor = clamp(factor, 0.86, 1.16)

    factors = [
        f"Probable starter {pitcher_name or pitcher_id}: "
        + (f"ERA {era:.2f}" if era is not None else "ERA n/a")
        + (f", WHIP {whip:.2f}" if whip is not None else ", WHIP n/a")
        + f" over {innings:.1f} IP",
        f"Starter run-environment multiplier {factor:.3f}",
    ]
    return factor, innings, raw, factors


def _project(
    *,
    player_name: str,
    milestone: int,
    event_date: date,
    event_ticker: str | None,
    kind: str,
) -> MLBRunProductionProjection | None:
    if milestone < 1 or milestone > (5 if kind == "rbi" else 8):
        return None

    try:
        player = resolve_player(player_name, event_date.year)
        if player is None:
            return None
        player_id = int(player.get("id"))
        team_id = int((player.get("currentTeam") or {}).get("id"))

        game = find_player_game(
            team_id=team_id,
            event_date=event_date,
            event_ticker=event_ticker,
        )
        if game is None:
            return None
        context = game_context_for_hitter(player, game)
        if context is None or context["game_status"].lower() != "preview":
            return None

        season = season_stat(player_id, "hitting", event_date.year)
        if not season:
            return None
        season_pa = _f(season, "plateAppearances")
        season_games = _f(season, "gamesPlayed")
        if season_pa is None or not season_games or season_games < 20 or season_pa < 60:
            return None

        season_total = _metric(season, kind)
        season_rate = season_total / season_pa
        prior = LEAGUE_RBI_PER_PA if kind == "rbi" else LEAGUE_HRR_PER_PA
        prior_weight = 75.0 if kind == "rbi" else 60.0
        weight = season_pa / (season_pa + prior_weight)
        rate = weight * season_rate + (1.0 - weight) * prior

        prior_end = event_date - timedelta(days=1)
        recent_start = prior_end - timedelta(days=27)
        recent = date_range_stat(
            player_id,
            "hitting",
            recent_start.isoformat(),
            prior_end.isoformat(),
        )
        recent_pa = _f(recent, "plateAppearances")
        recent_games = _f(recent, "gamesPlayed")
        recent_total = _metric(recent or {}, kind)
        recent_rate = (
            recent_total / recent_pa
            if recent_pa is not None and recent_pa > 0
            else None
        )
        if recent_rate is not None and recent_pa is not None:
            recent_weight = 0.27 * clamp(recent_pa / 75.0, 0.0, 1.0)
            rate = (1.0 - recent_weight) * rate + recent_weight * recent_rate

        season_pa_game = season_pa / season_games
        expected_pa = season_pa_game
        if recent_pa is not None and recent_games and recent_games >= 4:
            recent_pa_game = recent_pa / recent_games
            recent_pa_weight = 0.30 * clamp(recent_games / 14.0, 0.0, 1.0)
            expected_pa = (
                (1.0 - recent_pa_weight) * season_pa_game
                + recent_pa_weight * recent_pa_game
            )
        expected_pa = clamp(expected_pa, 2.5, 5.5)

        pitcher_id = context.get("probable_pitcher_id")
        pitcher_name = context.get("probable_pitcher_name") or None
        pitcher_factor, pitcher_ip, _, pitcher_factors = _pitcher_environment(
            int(pitcher_id) if pitcher_id else None,
            pitcher_name,
            event_date.year,
        )

        projected_mean = clamp(expected_pa * rate * pitcher_factor, 0.03, 8.0)

        logs = _pregame_logs(player_id, event_date.year, event_date)
        values = [_metric(row, kind) for row in logs]
        empirical_mean = mean(values) if values else projected_mean
        empirical_var = pvariance(values) if len(values) >= 8 else projected_mean
        scale = projected_mean / max(empirical_mean, 0.10)
        projected_var = max(
            projected_mean * 1.05,
            empirical_var * scale * scale,
        )
        # H+R+RBI intentionally counts correlated components; allow additional
        # overdispersion rather than pretending the components are independent.
        if kind == "hrr":
            projected_var = max(projected_var, projected_mean * 1.25)

        probability = _negative_binomial_tail(
            projected_mean,
            projected_var,
            milestone,
        )

        label = "RBIs" if kind == "rbi" else "Hits + Runs + RBIs"
        market_key = "batter_rbis" if kind == "rbi" else "batter_hrr"
        factors = [
            f"Season {label} rate {season_total:.0f}/{season_pa:.0f} PA ({season_rate:.3f}/PA)",
            f"Expected plate appearances {expected_pa:.2f}",
            f"Projected mean {projected_mean:.2f} {label}",
            f"Projected count variance {projected_var:.2f} from {len(values)} game-log observation(s)",
            *pitcher_factors,
        ]
        warnings: list[str] = []
        if recent_rate is not None and recent_pa is not None:
            factors.append(
                f"Recent 28-day {label} rate {recent_total:.0f}/{recent_pa:.0f} PA ({recent_rate:.3f}/PA)"
            )
        else:
            warnings.append("recent hitter run-production sample unavailable")
        if len(values) < 20:
            warnings.append("limited per-game distribution sample")
        if kind == "hrr":
            warnings.append("H+R+RBI components are correlated; overdispersion is modeled explicitly")
        warnings.append(
            "pregame model; confirmed batting-order slot, park, weather, and bullpen quality are not yet separately modeled"
        )

        confidence = 0.43
        confidence += 0.18 * clamp(season_pa / 500.0, 0.0, 1.0)
        confidence += 0.08 * clamp((recent_pa or 0.0) / 75.0, 0.0, 1.0)
        confidence += 0.08 * clamp(len(values) / 80.0, 0.0, 1.0)
        confidence += 0.08 * clamp(pitcher_ip / 100.0, 0.0, 1.0)
        if milestone >= 3:
            confidence -= 0.04
        if kind == "hrr":
            confidence -= 0.02
        confidence = clamp(confidence, 0.35, 0.76)

        own = context.get("own_team_name") or "Hitter team"
        opp = context.get("opponent_team_name") or "Opponent"
        evidence = ModelEvidence(
            sport="MLB",
            model_name=(
                "MLB RBIs: per-PA production + game-log dispersion + starter context"
                if kind == "rbi"
                else "MLB H+R+RBI: per-PA production + game-log dispersion + starter context"
            ),
            fair_probability=probability,
            confidence=confidence,
            sample_size=int(season_games),
            factors=tuple(factors),
            warnings=tuple(warnings),
        )
        return MLBRunProductionProjection(
            evidence=evidence,
            player_name=str(player.get("fullName") or player_name),
            market_key=market_key,
            market_label=label,
            milestone=milestone,
            line=float(milestone) - 0.5,
            game_title=f"{own} vs {opp}",
            probable_pitcher_name=pitcher_name,
            expected_plate_appearances=expected_pa,
            projected_mean=projected_mean,
            projected_variance=projected_var,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError, OverflowError):
        return None


def project_mlb_rbis(
    *,
    player_name: str,
    milestone_rbis: int,
    event_date: date,
    event_ticker: str | None,
) -> MLBRunProductionProjection | None:
    return _project(
        player_name=player_name,
        milestone=milestone_rbis,
        event_date=event_date,
        event_ticker=event_ticker,
        kind="rbi",
    )


def project_mlb_hrr(
    *,
    player_name: str,
    milestone_hrr: int,
    event_date: date,
    event_ticker: str | None,
) -> MLBRunProductionProjection | None:
    return _project(
        player_name=player_name,
        milestone=milestone_hrr,
        event_date=event_date,
        event_ticker=event_ticker,
        kind="hrr",
    )
