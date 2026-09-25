from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import comb

import requests

from sports_edge.core.math import clamp
from sports_edge.data.mlb_prop_data import (
    date_range_stat,
    find_player_game,
    game_context_for_hitter,
    resolve_player,
    season_stat,
)
from sports_edge.models.model_evidence import ModelEvidence


LEAGUE_HIT_RATE = 0.245


@dataclass(frozen=True)
class MLBHitProjection:
    evidence: ModelEvidence
    player_name: str
    milestone_hits: int
    line: float
    game_title: str
    probable_pitcher_name: str | None
    expected_at_bats: float
    per_ab_hit_probability: float


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


def _rate(hits: float | None, at_bats: float | None) -> float | None:
    if hits is None or at_bats is None or at_bats <= 0:
        return None
    return clamp(hits / at_bats, 0.05, 0.60)


def _shrunk_rate(rate: float, sample: float, prior: float, prior_weight: float) -> float:
    weight = sample / (sample + prior_weight)
    return clamp(weight * rate + (1.0 - weight) * prior, 0.05, 0.60)


def _blend_recent(
    season_rate: float,
    recent_rate: float | None,
    recent_ab: float | None,
    max_weight: float,
) -> float:
    if recent_rate is None or recent_ab is None or recent_ab <= 0:
        return season_rate
    weight = max_weight * clamp(recent_ab / 60.0, 0.0, 1.0)
    return clamp((1.0 - weight) * season_rate + weight * recent_rate, 0.05, 0.60)


def at_least_k_hits_probability(per_ab_hit_prob: float, expected_ab: float, k: int) -> float:
    """Fractional-at-bat mixture of two binomial tails."""
    p = clamp(per_ab_hit_prob, 0.001, 0.999)
    n0 = max(0, int(expected_ab))
    frac = max(0.0, min(1.0, expected_ab - n0))

    def tail(n: int) -> float:
        if k <= 0:
            return 1.0
        if n < k:
            return 0.0
        cdf = 0.0
        for hits in range(0, k):
            cdf += comb(n, hits) * (p ** hits) * ((1.0 - p) ** (n - hits))
        return clamp(1.0 - cdf, 0.0, 1.0)

    return clamp((1.0 - frac) * tail(n0) + frac * tail(n0 + 1), 0.01, 0.99)


def project_mlb_hits(
    *,
    player_name: str,
    milestone_hits: int,
    event_date: date,
    event_ticker: str | None,
) -> MLBHitProjection | None:
    if milestone_hits not in {1, 2, 3}:
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
        if context is None:
            return None
        if context["game_status"].lower() != "preview":
            # This first production model is pregame-only. Live hit props need a
            # separate state model using completed PA and remaining lineup turns.
            return None

        season = season_stat(player_id, "hitting", event_date.year)
        if not season:
            return None

        season_ab = _f(season, "atBats")
        season_hits = _f(season, "hits")
        season_games = _f(season, "gamesPlayed")
        season_rate = _rate(season_hits, season_ab)
        if season_rate is None or season_ab is None or season_ab < 40 or not season_games:
            return None

        prior_end = event_date - timedelta(days=1)
        recent_start = prior_end - timedelta(days=27)
        recent = date_range_stat(
            player_id,
            "hitting",
            recent_start.isoformat(),
            prior_end.isoformat(),
        )
        recent_ab = _f(recent, "atBats")
        recent_hits = _f(recent, "hits")
        recent_games = _f(recent, "gamesPlayed")
        recent_rate = _rate(recent_hits, recent_ab)

        hitter_rate = _shrunk_rate(
            season_rate,
            season_ab,
            LEAGUE_HIT_RATE,
            45.0,
        )
        hitter_rate = _blend_recent(
            hitter_rate,
            recent_rate,
            recent_ab,
            0.25,
        )

        pitcher_name = context.get("probable_pitcher_name") or None
        pitcher_rate = LEAGUE_HIT_RATE
        pitcher_ab = 0.0
        pitcher_recent_ab = 0.0
        pitcher_recent_rate = None
        pitcher_season_rate = None
        pitcher_id = context.get("probable_pitcher_id")

        if pitcher_id:
            pseason = season_stat(int(pitcher_id), "pitching", event_date.year)
            pitcher_ab = _f(pseason, "atBats") or 0.0
            pitcher_hits = _f(pseason, "hits")
            pitcher_season_rate = _rate(pitcher_hits, pitcher_ab)
            if pitcher_season_rate is not None and pitcher_ab >= 30:
                pitcher_rate = _shrunk_rate(
                    pitcher_season_rate,
                    pitcher_ab,
                    LEAGUE_HIT_RATE,
                    70.0,
                )

            precent = date_range_stat(
                int(pitcher_id),
                "pitching",
                recent_start.isoformat(),
                prior_end.isoformat(),
            )
            pitcher_recent_ab = _f(precent, "atBats") or 0.0
            pitcher_recent_hits = _f(precent, "hits")
            pitcher_recent_rate = _rate(pitcher_recent_hits, pitcher_recent_ab)
            pitcher_rate = _blend_recent(
                pitcher_rate,
                pitcher_recent_rate,
                pitcher_recent_ab,
                0.18,
            )

        # The starter only accounts for part of a hitter's game, so the pitcher
        # effect is deliberately attenuated instead of treating starter AVG
        # allowed as the whole-game environment.
        pitcher_multiplier = (pitcher_rate / LEAGUE_HIT_RATE) ** 0.35
        per_ab = clamp(hitter_rate * pitcher_multiplier, 0.10, 0.45)

        season_ab_per_game = season_ab / season_games
        expected_ab = season_ab_per_game
        if recent_ab is not None and recent_games and recent_games >= 3:
            recent_ab_per_game = recent_ab / recent_games
            recent_weight = 0.30 * clamp(recent_games / 12.0, 0.0, 1.0)
            expected_ab = (
                (1.0 - recent_weight) * season_ab_per_game
                + recent_weight * recent_ab_per_game
            )
        expected_ab = clamp(expected_ab, 2.5, 5.0)

        probability = at_least_k_hits_probability(
            per_ab,
            expected_ab,
            milestone_hits,
        )

        factors = [
            f"Season hit rate {season_hits:.0f}/{season_ab:.0f} AB ({season_rate:.3f})",
            f"Expected at-bats {expected_ab:.2f}",
            f"Modeled hit probability per AB {per_ab:.3f}",
        ]
        warnings: list[str] = []

        if recent_rate is not None and recent_ab is not None:
            factors.append(
                f"Recent 28-day hit rate {recent_hits:.0f}/{recent_ab:.0f} AB ({recent_rate:.3f})"
            )
        else:
            warnings.append("recent hitter sample unavailable")

        if pitcher_id and pitcher_season_rate is not None:
            factors.append(
                f"Probable starter {pitcher_name}: opponents {pitcher_season_rate:.3f} AVG over {pitcher_ab:.0f} AB"
            )
            if pitcher_recent_rate is not None and pitcher_recent_ab >= 20:
                factors.append(
                    f"Starter recent opponent AVG {pitcher_recent_rate:.3f} over {pitcher_recent_ab:.0f} AB"
                )
        else:
            warnings.append("probable starter unavailable; pitcher adjustment held at league prior")

        confidence = 0.42
        confidence += 0.18 * clamp(season_ab / 450.0, 0.0, 1.0)
        confidence += 0.08 * clamp((recent_ab or 0.0) / 60.0, 0.0, 1.0)
        confidence += 0.10 * clamp(pitcher_ab / 350.0, 0.0, 1.0)
        confidence += 0.05 if pitcher_id else 0.0
        if milestone_hits >= 2:
            confidence -= 0.05
        confidence = clamp(confidence, 0.35, 0.78)

        if season_ab < 150:
            warnings.append("limited season hitter sample")
        if pitcher_id and pitcher_ab < 80:
            warnings.append("limited probable-pitcher opponent sample")
        warnings.append("pregame model; confirmed batting-order position is not yet included")

        own = context.get("own_team_name") or "Hitter team"
        opp = context.get("opponent_team_name") or "Opponent"
        game_title = f"{own} vs {opp}"

        evidence = ModelEvidence(
            sport="MLB",
            model_name="MLB Hits: hitter rate + recent form + starter context",
            fair_probability=probability,
            confidence=confidence,
            sample_size=int(season_ab),
            factors=tuple(factors),
            warnings=tuple(warnings),
        )
        return MLBHitProjection(
            evidence=evidence,
            player_name=str(player.get("fullName") or player_name),
            milestone_hits=milestone_hits,
            line=float(milestone_hits) - 0.5,
            game_title=game_title,
            probable_pitcher_name=pitcher_name,
            expected_at_bats=expected_ab,
            per_ab_hit_probability=per_ab,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None
