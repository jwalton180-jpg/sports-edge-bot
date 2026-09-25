from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import exp, factorial

import requests

from sports_edge.core.math import clamp
from sports_edge.data.mlb_prop_data import (
    date_range_stat,
    find_player_game,
    game_context_for_pitcher,
    resolve_player,
    season_stat,
    team_season_stat,
)
from sports_edge.models.model_evidence import ModelEvidence


LEAGUE_K_PER_BF = 0.225


@dataclass(frozen=True)
class MLBStrikeoutProjection:
    evidence: ModelEvidence
    player_name: str
    milestone_strikeouts: int
    line: float
    game_title: str
    opponent_team_name: str
    expected_batters_faced: float
    strikeout_rate_per_bf: float
    expected_strikeouts: float


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


def _rate(strikeouts: float | None, batters_faced: float | None) -> float | None:
    if strikeouts is None or batters_faced is None or batters_faced <= 0:
        return None
    return clamp(strikeouts / batters_faced, 0.05, 0.45)


def _shrunk_rate(rate: float, sample: float, prior: float, prior_weight: float) -> float:
    weight = sample / (sample + prior_weight)
    return clamp(weight * rate + (1.0 - weight) * prior, 0.05, 0.45)


def _blend_recent(
    season_rate: float,
    recent_rate: float | None,
    recent_bf: float | None,
    max_weight: float,
) -> float:
    if recent_rate is None or recent_bf is None or recent_bf <= 0:
        return season_rate
    weight = max_weight * clamp(recent_bf / 120.0, 0.0, 1.0)
    return clamp((1.0 - weight) * season_rate + weight * recent_rate, 0.05, 0.45)


def poisson_at_least_probability(mean: float, threshold: int) -> float:
    lam = max(0.01, float(mean))
    if threshold <= 0:
        return 1.0
    cdf = 0.0
    for k in range(threshold):
        cdf += exp(-lam) * (lam ** k) / factorial(k)
    return clamp(1.0 - cdf, 0.001, 0.999)


def project_mlb_pitcher_strikeouts(
    *,
    player_name: str,
    milestone_strikeouts: int,
    event_date: date,
    event_ticker: str | None,
) -> MLBStrikeoutProjection | None:
    if milestone_strikeouts < 1 or milestone_strikeouts > 15:
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

        context = game_context_for_pitcher(player, game)
        if context is None:
            return None
        if context["game_status"].lower() != "preview":
            return None
        if not context["is_probable_starter"]:
            # Starter props must resolve to the actual probable starter.
            return None

        season = season_stat(player_id, "pitching", event_date.year)
        if not season:
            return None

        season_so = _f(season, "strikeOuts")
        season_bf = _f(season, "battersFaced")
        season_gs = _f(season, "gamesStarted")
        if season_bf is None or season_bf <= 0:
            season_bf = _f(season, "atBats")
        season_rate = _rate(season_so, season_bf)
        if (
            season_rate is None
            or season_bf is None
            or season_bf < 100
            or not season_gs
            or season_gs < 4
        ):
            return None

        prior_end = event_date - timedelta(days=1)
        recent_start = prior_end - timedelta(days=34)
        recent = date_range_stat(
            player_id,
            "pitching",
            recent_start.isoformat(),
            prior_end.isoformat(),
        )
        recent_so = _f(recent, "strikeOuts")
        recent_bf = _f(recent, "battersFaced")
        recent_gs = _f(recent, "gamesStarted")
        if recent_bf is None or recent_bf <= 0:
            recent_bf = _f(recent, "atBats")
        recent_rate = _rate(recent_so, recent_bf)

        pitcher_rate = _shrunk_rate(
            season_rate,
            season_bf,
            LEAGUE_K_PER_BF,
            180.0,
        )
        pitcher_rate = _blend_recent(
            pitcher_rate,
            recent_rate,
            recent_bf,
            0.24,
        )

        opponent_id = int(context.get("opponent_team_id") or 0)
        opponent = team_season_stat(opponent_id, "hitting", event_date.year) if opponent_id else None
        opponent_so = _f(opponent, "strikeOuts")
        opponent_pa = _f(opponent, "plateAppearances")
        if opponent_pa is None or opponent_pa <= 0:
            opponent_pa = _f(opponent, "atBats")
        opponent_rate = _rate(opponent_so, opponent_pa)
        if opponent_rate is not None and opponent_pa and opponent_pa >= 250:
            opponent_rate = _shrunk_rate(
                opponent_rate,
                opponent_pa,
                LEAGUE_K_PER_BF,
                700.0,
            )
        else:
            opponent_rate = LEAGUE_K_PER_BF

        expected_bf = season_bf / season_gs
        if recent_bf is not None and recent_gs and recent_gs >= 2:
            recent_bf_per_start = recent_bf / recent_gs
            recent_weight = 0.28 * clamp(recent_gs / 6.0, 0.0, 1.0)
            expected_bf = (
                (1.0 - recent_weight) * expected_bf
                + recent_weight * recent_bf_per_start
            )
        expected_bf = clamp(expected_bf, 14.0, 31.0)

        opponent_multiplier = (opponent_rate / LEAGUE_K_PER_BF) ** 0.28
        adjusted_rate = clamp(pitcher_rate * opponent_multiplier, 0.08, 0.40)
        expected_ks = clamp(expected_bf * adjusted_rate, 1.0, 14.0)
        probability = poisson_at_least_probability(
            expected_ks,
            milestone_strikeouts,
        )

        factors = [
            f"Season strikeout rate {season_so:.0f}/{season_bf:.0f} BF ({season_rate:.3f})",
            f"Expected batters faced {expected_bf:.1f}",
            f"Opponent team strikeout rate {opponent_rate:.3f}",
            f"Expected strikeouts {expected_ks:.2f}",
            f"Verified probable starter: {context.get('probable_pitcher_name') or player_name}",
        ]
        warnings: list[str] = []

        if recent_rate is not None and recent_bf is not None:
            factors.append(
                f"Recent 35-day strikeout rate {recent_so:.0f}/{recent_bf:.0f} BF ({recent_rate:.3f})"
            )
        else:
            warnings.append("recent pitcher strikeout sample unavailable")

        if opponent is None or opponent_pa is None:
            warnings.append("opponent team strikeout sample unavailable; league prior used")

        confidence = 0.42
        confidence += 0.18 * clamp(season_bf / 700.0, 0.0, 1.0)
        confidence += 0.09 * clamp((recent_bf or 0.0) / 120.0, 0.0, 1.0)
        confidence += 0.08 * clamp((opponent_pa or 0.0) / 3000.0, 0.0, 1.0)
        confidence += 0.06
        if milestone_strikeouts >= 9:
            confidence -= 0.04
        confidence = clamp(confidence, 0.40, 0.78)

        if season_gs < 10:
            warnings.append("limited season start sample")
        warnings.append("pregame model; umpire, weather, bullpen hook, and confirmed opposing lineup are not yet included")

        own = context.get("own_team_name") or "Pitcher team"
        opp = context.get("opponent_team_name") or "Opponent"
        game_title = f"{own} vs {opp}"

        evidence = ModelEvidence(
            sport="MLB",
            model_name="MLB Pitcher Ks: K/BF + recent form + opponent strikeout tendency",
            fair_probability=probability,
            confidence=confidence,
            sample_size=int(season_bf),
            factors=tuple(factors),
            warnings=tuple(warnings),
        )
        return MLBStrikeoutProjection(
            evidence=evidence,
            player_name=str(player.get("fullName") or player_name),
            milestone_strikeouts=milestone_strikeouts,
            line=float(milestone_strikeouts) - 0.5,
            game_title=game_title,
            opponent_team_name=opp,
            expected_batters_faced=expected_bf,
            strikeout_rate_per_bf=adjusted_rate,
            expected_strikeouts=expected_ks,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None
