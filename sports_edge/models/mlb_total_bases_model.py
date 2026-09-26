from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

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


# Approximate MLB per-AB outcome priors. These are deliberately broad priors;
# player and probable-pitcher samples drive the projection when data is deep.
LEAGUE_1B_PER_AB = 0.160
LEAGUE_2B_PER_AB = 0.048
LEAGUE_3B_PER_AB = 0.004
LEAGUE_HR_PER_AB = 0.032
LEAGUE_TB_PER_AB = (
    LEAGUE_1B_PER_AB
    + 2.0 * LEAGUE_2B_PER_AB
    + 3.0 * LEAGUE_3B_PER_AB
    + 4.0 * LEAGUE_HR_PER_AB
)


@dataclass(frozen=True)
class MLBTotalBasesProjection:
    evidence: ModelEvidence
    player_name: str
    milestone_total_bases: int
    line: float
    game_title: str
    probable_pitcher_name: str | None
    expected_at_bats: float
    expected_total_bases: float


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


def _outcome_rates(stat: dict | None) -> tuple[float, float, float, float, float] | None:
    if not stat:
        return None
    ab = _f(stat, "atBats")
    hits = _f(stat, "hits")
    doubles = _f(stat, "doubles") or 0.0
    triples = _f(stat, "triples") or 0.0
    homers = _f(stat, "homeRuns") or 0.0
    if ab is None or hits is None or ab <= 0:
        return None
    singles = max(0.0, hits - doubles - triples - homers)
    rates = (
        clamp(singles / ab, 0.0, 0.45),
        clamp(doubles / ab, 0.0, 0.16),
        clamp(triples / ab, 0.0, 0.04),
        clamp(homers / ab, 0.0, 0.15),
    )
    total_hit_rate = sum(rates)
    if total_hit_rate >= 0.70:
        scale = 0.69 / total_hit_rate
        rates = tuple(x * scale for x in rates)
    return (*rates, ab)


def _shrink_component(rate: float, sample: float, prior: float, prior_weight: float) -> float:
    weight = sample / (sample + prior_weight)
    return max(0.0, weight * rate + (1.0 - weight) * prior)


def _blend_components(
    season: tuple[float, float, float, float],
    recent: tuple[float, float, float, float] | None,
    recent_ab: float | None,
) -> tuple[float, float, float, float]:
    if recent is None or recent_ab is None or recent_ab <= 0:
        return season
    weight = 0.24 * clamp(recent_ab / 70.0, 0.0, 1.0)
    return tuple((1.0 - weight) * a + weight * b for a, b in zip(season, recent))


def _tb_per_ab(rates: tuple[float, float, float, float]) -> float:
    p1, p2, p3, p4 = rates
    return p1 + 2.0 * p2 + 3.0 * p3 + 4.0 * p4


def _normalize_outcomes(
    rates: tuple[float, float, float, float],
    *,
    max_hit_probability: float = 0.62,
) -> tuple[float, float, float, float]:
    total = sum(rates)
    if total <= max_hit_probability:
        return tuple(max(0.0, x) for x in rates)
    scale = max_hit_probability / total
    return tuple(max(0.0, x * scale) for x in rates)


def at_least_k_total_bases_probability(
    per_ab_rates: tuple[float, float, float, float],
    expected_ab: float,
    k: int,
) -> float:
    """Probability of at least k total bases using a fractional-AB DP mixture."""
    if k <= 0:
        return 1.0

    rates = _normalize_outcomes(per_ab_rates)
    p1, p2, p3, p4 = rates
    p0 = max(0.0, 1.0 - (p1 + p2 + p3 + p4))
    outcome_probs = (p0, p1, p2, p3, p4)

    n0 = max(0, int(expected_ab))
    frac = clamp(expected_ab - n0, 0.0, 1.0)

    def tail(n: int) -> float:
        dist = [1.0]
        for _ in range(n):
            nxt = [0.0] * (len(dist) + 4)
            for bases_so_far, mass in enumerate(dist):
                if mass <= 0:
                    continue
                for bases, prob in enumerate(outcome_probs):
                    nxt[bases_so_far + bases] += mass * prob
            dist = nxt
        return clamp(sum(dist[k:]) if k < len(dist) else 0.0, 0.0, 1.0)

    return clamp((1.0 - frac) * tail(n0) + frac * tail(n0 + 1), 0.01, 0.99)


def project_mlb_total_bases(
    *,
    player_name: str,
    milestone_total_bases: int,
    event_date: date,
    event_ticker: str | None,
) -> MLBTotalBasesProjection | None:
    if milestone_total_bases not in {1, 2, 3, 4}:
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
        parsed = _outcome_rates(season)
        season_games = _f(season, "gamesPlayed")
        if parsed is None or not season_games:
            return None
        s1, s2, s3, s4, season_ab = parsed
        if season_ab < 60:
            return None

        priors = (LEAGUE_1B_PER_AB, LEAGUE_2B_PER_AB, LEAGUE_3B_PER_AB, LEAGUE_HR_PER_AB)
        season_rates = tuple(
            _shrink_component(rate, season_ab, prior, 80.0)
            for rate, prior in zip((s1, s2, s3, s4), priors)
        )

        prior_end = event_date - timedelta(days=1)
        recent_start = prior_end - timedelta(days=34)
        recent = date_range_stat(
            player_id,
            "hitting",
            recent_start.isoformat(),
            prior_end.isoformat(),
        )
        recent_parsed = _outcome_rates(recent)
        recent_rates = recent_parsed[:4] if recent_parsed is not None else None
        recent_ab = recent_parsed[4] if recent_parsed is not None else None
        hitter_rates = _blend_components(season_rates, recent_rates, recent_ab)

        pitcher_name = context.get("probable_pitcher_name") or None
        pitcher_id = context.get("probable_pitcher_id")
        pitcher_ab = 0.0
        pitcher_recent_ab = 0.0
        pitcher_tb_rate = LEAGUE_TB_PER_AB
        pitcher_season_tb_rate = None
        pitcher_recent_tb_rate = None

        if pitcher_id:
            pseason = season_stat(int(pitcher_id), "pitching", event_date.year)
            pparsed = _outcome_rates(pseason)
            if pparsed is not None:
                p1, p2, p3, p4, pitcher_ab = pparsed
                pitcher_season_tb_rate = _tb_per_ab((p1, p2, p3, p4))
                weight = pitcher_ab / (pitcher_ab + 120.0)
                pitcher_tb_rate = (
                    weight * pitcher_season_tb_rate
                    + (1.0 - weight) * LEAGUE_TB_PER_AB
                )

            precent = date_range_stat(
                int(pitcher_id),
                "pitching",
                recent_start.isoformat(),
                prior_end.isoformat(),
            )
            precent_parsed = _outcome_rates(precent)
            if precent_parsed is not None:
                pr1, pr2, pr3, pr4, pitcher_recent_ab = precent_parsed
                pitcher_recent_tb_rate = _tb_per_ab((pr1, pr2, pr3, pr4))
                recent_weight = 0.15 * clamp(pitcher_recent_ab / 90.0, 0.0, 1.0)
                pitcher_tb_rate = (
                    (1.0 - recent_weight) * pitcher_tb_rate
                    + recent_weight * pitcher_recent_tb_rate
                )

        # Starter context affects only part of the game. Apply a conservative
        # attenuated multiplier to the hitter's full outcome distribution.
        pitcher_multiplier = clamp(
            (pitcher_tb_rate / LEAGUE_TB_PER_AB) ** 0.24,
            0.82,
            1.20,
        )
        modeled_rates = _normalize_outcomes(
            tuple(rate * pitcher_multiplier for rate in hitter_rates)
        )

        season_ab_per_game = season_ab / season_games
        expected_ab = season_ab_per_game
        recent_games = _f(recent, "gamesPlayed")
        if recent_ab is not None and recent_games and recent_games >= 4:
            recent_ab_per_game = recent_ab / recent_games
            recent_weight = 0.28 * clamp(recent_games / 14.0, 0.0, 1.0)
            expected_ab = (
                (1.0 - recent_weight) * season_ab_per_game
                + recent_weight * recent_ab_per_game
            )
        expected_ab = clamp(expected_ab, 2.5, 5.0)

        probability = at_least_k_total_bases_probability(
            modeled_rates,
            expected_ab,
            milestone_total_bases,
        )
        expected_tb = expected_ab * _tb_per_ab(modeled_rates)

        factors = [
            f"Season total-base rate {_tb_per_ab((s1, s2, s3, s4)):.3f} TB/AB over {season_ab:.0f} AB",
            f"Expected at-bats {expected_ab:.2f}",
            f"Modeled expected total bases {expected_tb:.2f}",
            (
                "Modeled per-AB outcomes "
                f"1B {modeled_rates[0]:.3f}, 2B {modeled_rates[1]:.3f}, "
                f"3B {modeled_rates[2]:.3f}, HR {modeled_rates[3]:.3f}"
            ),
        ]
        warnings: list[str] = []

        if recent_rates is not None and recent_ab is not None:
            factors.append(
                f"Recent 35-day total-base rate {_tb_per_ab(recent_rates):.3f} TB/AB over {recent_ab:.0f} AB"
            )
        else:
            warnings.append("recent hitter total-base sample unavailable")

        if pitcher_id and pitcher_season_tb_rate is not None:
            factors.append(
                f"Probable starter {pitcher_name}: {pitcher_season_tb_rate:.3f} TB/AB allowed over {pitcher_ab:.0f} AB"
            )
            if pitcher_recent_tb_rate is not None and pitcher_recent_ab >= 30:
                factors.append(
                    f"Starter recent total-base rate allowed {pitcher_recent_tb_rate:.3f} over {pitcher_recent_ab:.0f} AB"
                )
        else:
            warnings.append("probable starter total-base profile unavailable; matchup held at league prior")

        confidence = 0.39
        confidence += 0.17 * clamp(season_ab / 500.0, 0.0, 1.0)
        confidence += 0.08 * clamp((recent_ab or 0.0) / 70.0, 0.0, 1.0)
        confidence += 0.09 * clamp(pitcher_ab / 400.0, 0.0, 1.0)
        confidence += 0.04 if pitcher_id else 0.0
        if milestone_total_bases >= 3:
            confidence -= 0.03
        if milestone_total_bases >= 4:
            confidence -= 0.03
        confidence = clamp(confidence, 0.35, 0.74)

        if season_ab < 160:
            warnings.append("limited season total-base sample")
        if pitcher_id and pitcher_ab < 100:
            warnings.append("limited probable-pitcher batted-ball outcome sample")
        warnings.append(
            "pregame model; park, weather, bullpen quality, and confirmed batting-order position are not yet included"
        )

        own = context.get("own_team_name") or "Hitter team"
        opp = context.get("opponent_team_name") or "Opponent"
        game_title = f"{own} vs {opp}"

        evidence = ModelEvidence(
            sport="MLB",
            model_name="MLB Total Bases: hit-type distribution + recent form + starter context",
            fair_probability=probability,
            confidence=confidence,
            sample_size=int(season_ab),
            factors=tuple(factors),
            warnings=tuple(warnings),
        )
        return MLBTotalBasesProjection(
            evidence=evidence,
            player_name=str(player.get("fullName") or player_name),
            milestone_total_bases=milestone_total_bases,
            line=float(milestone_total_bases) - 0.5,
            game_title=game_title,
            probable_pitcher_name=pitcher_name,
            expected_at_bats=expected_ab,
            expected_total_bases=expected_tb,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None
