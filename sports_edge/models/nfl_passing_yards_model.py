from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt
from statistics import mean, pstdev

from sports_edge.core.math import clamp
from sports_edge.data.nfl_prop_data import NFLPassingContext, resolve_passing_context
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.nfl import normal_over_probability


LEAGUE_YPA_PRIOR = 7.0
LEAGUE_ATTEMPTS_PRIOR = 32.5
YARDS_SD_PRIOR = 70.0


@dataclass(frozen=True)
class NFLPassingProjection:
    evidence: ModelEvidence
    player_name: str
    milestone_yards: int
    line: float
    game_title: str
    team: str
    opponent: str
    target_week: int
    expected_attempts: float
    adjusted_ypa: float
    expected_passing_yards: float
    passing_yards_sd: float


def _f(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "NA", "NaN", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum(rows: tuple[dict, ...], key: str) -> float:
    return sum((_f(row, key) or 0.0) for row in rows)


def _weighted_recent_average(
    rows: tuple[dict, ...],
    key: str,
    *,
    last_n: int = 2,
) -> float | None:
    values = [(_f(row, key), row) for row in rows[-last_n:]]
    clean = [value for value, _ in values if value is not None]
    return mean(clean) if clean else None


def _shrunk_rate(
    observed: float,
    sample: float,
    prior: float,
    prior_weight: float,
) -> float:
    weight = sample / (sample + prior_weight)
    return (weight * observed) + ((1.0 - weight) * prior)


def _passing_environment(context: NFLPassingContext) -> tuple[float, float, float, float]:
    """Return league YPA, league attempts/game, opponent YPA allowed, opponent attempts allowed."""
    league_attempts = _sum(context.league_rows, "attempts")
    league_yards = _sum(context.league_rows, "passing_yards")
    league_games = len(context.league_rows)

    league_ypa = (
        league_yards / league_attempts
        if league_attempts > 0
        else LEAGUE_YPA_PRIOR
    )
    league_att_pg = (
        league_attempts / league_games
        if league_games > 0
        else LEAGUE_ATTEMPTS_PRIOR
    )

    opp_attempts = _sum(context.opponent_allowed_rows, "attempts")
    opp_yards = _sum(context.opponent_allowed_rows, "passing_yards")
    opp_games = len(context.opponent_allowed_rows)
    opp_ypa = (
        opp_yards / opp_attempts
        if opp_attempts > 0
        else league_ypa
    )
    opp_att_pg = (
        opp_attempts / opp_games
        if opp_games > 0
        else league_att_pg
    )

    # Defense samples are tiny early in the season; aggressively shrink them.
    opp_ypa = _shrunk_rate(opp_ypa, opp_attempts, league_ypa, 120.0)
    opp_att_pg = _shrunk_rate(
        opp_att_pg,
        float(opp_games),
        league_att_pg,
        4.0,
    )
    return league_ypa, league_att_pg, opp_ypa, opp_att_pg


def project_nfl_passing_yards(
    *,
    player_name: str,
    milestone_yards: int,
    event_date: date,
) -> NFLPassingProjection | None:
    if milestone_yards < 25 or milestone_yards > 600:
        return None

    context = resolve_passing_context(
        player_name=player_name,
        event_date=event_date,
    )
    if context is None:
        return None

    rows = context.player_rows
    if len(rows) < 2:
        return None

    total_attempts = _sum(rows, "attempts")
    total_yards = _sum(rows, "passing_yards")
    if total_attempts < 40 or total_yards <= 0:
        return None

    season_attempts_pg = total_attempts / len(rows)
    season_ypa = total_yards / total_attempts

    recent_attempts_pg = _weighted_recent_average(rows, "attempts", last_n=2)
    recent_yards = _weighted_recent_average(rows, "passing_yards", last_n=2)
    recent_ypa = None
    recent_attempts_total = sum((_f(row, "attempts") or 0.0) for row in rows[-2:])
    recent_yards_total = sum((_f(row, "passing_yards") or 0.0) for row in rows[-2:])
    if recent_attempts_total > 0:
        recent_ypa = recent_yards_total / recent_attempts_total

    league_ypa, league_att_pg, opp_ypa, opp_att_pg = _passing_environment(context)

    base_ypa = _shrunk_rate(
        season_ypa,
        total_attempts,
        league_ypa,
        80.0,
    )
    if recent_ypa is not None:
        recent_weight = 0.22 * clamp(recent_attempts_total / 70.0, 0.0, 1.0)
        base_ypa = (
            (1.0 - recent_weight) * base_ypa
            + recent_weight * recent_ypa
        )

    # Matchup matters, but it must not dominate a quarterback's own production.
    adjusted_ypa = clamp(
        0.72 * base_ypa + 0.28 * opp_ypa,
        4.5,
        10.5,
    )

    expected_attempts = season_attempts_pg
    if recent_attempts_pg is not None:
        expected_attempts = 0.74 * expected_attempts + 0.26 * recent_attempts_pg
    expected_attempts = 0.82 * expected_attempts + 0.18 * opp_att_pg
    expected_attempts = clamp(expected_attempts, 20.0, 48.0)

    # CPOE is useful as a modest efficiency stabilizer, not as a replacement
    # for actual yards/attempt.
    recent_cpoe_values = [
        _f(row, "passing_cpoe")
        for row in rows[-2:]
        if _f(row, "passing_cpoe") is not None
    ]
    recent_cpoe = mean(recent_cpoe_values) if recent_cpoe_values else None
    if recent_cpoe is not None:
        adjusted_ypa = clamp(
            adjusted_ypa + 0.012 * clamp(recent_cpoe, -12.0, 12.0),
            4.5,
            10.5,
        )

    expected_yards = clamp(expected_attempts * adjusted_ypa, 80.0, 450.0)

    weekly_yards = [
        value
        for row in rows
        if (value := _f(row, "passing_yards")) is not None
    ]
    empirical_sd = (
        pstdev(weekly_yards)
        if len(weekly_yards) >= 2
        else YARDS_SD_PRIOR
    )
    sample_weight = len(weekly_yards) / (len(weekly_yards) + 4.0)
    sd = sqrt(
        sample_weight * (empirical_sd ** 2)
        + (1.0 - sample_weight) * (YARDS_SD_PRIOR ** 2)
    )
    sd = clamp(sd, 48.0, 105.0)

    line = float(milestone_yards) - 0.5
    probability = normal_over_probability(
        expected_yards,
        sd,
        line,
    )

    opponent_sample = len(context.opponent_allowed_rows)
    confidence = 0.37
    confidence += 0.14 * clamp(len(rows) / 6.0, 0.0, 1.0)
    confidence += 0.12 * clamp(total_attempts / 180.0, 0.0, 1.0)
    confidence += 0.08 * clamp(opponent_sample / 4.0, 0.0, 1.0)
    confidence += 0.05 if context.scheduled_qb_name else 0.0
    confidence += 0.04 if context.roster_week >= context.week - 1 else 0.0
    confidence = clamp(confidence, 0.40, 0.76)

    factors = [
        f"Prior games {len(rows)}; {total_yards:.0f} passing yards on {total_attempts:.0f} attempts",
        f"Season YPA {season_ypa:.2f}; adjusted matchup YPA {adjusted_ypa:.2f}",
        f"Expected attempts {expected_attempts:.1f}",
        f"Opponent pass defense: {opp_ypa:.2f} yards/attempt allowed over {opponent_sample} prior game(s)",
        f"Projected passing yards {expected_yards:.1f} with modeled SD {sd:.1f}",
        f"Roster status {context.roster_status} (week {context.roster_week})",
    ]
    warnings: list[str] = []

    if recent_ypa is not None and recent_yards is not None:
        factors.append(
            f"Recent two-game passing form: {recent_yards:.1f} yards/game, {recent_ypa:.2f} YPA"
        )
    if recent_cpoe is not None:
        factors.append(f"Recent CPOE {recent_cpoe:+.1f}")

    if context.scheduled_qb_name:
        factors.append(f"Scheduled QB identity matches {context.scheduled_qb_name}")
    else:
        warnings.append("schedule has no named QB; active roster and prior role used")

    if len(rows) < 3:
        warnings.append("only two prior regular-season QB samples")
    if opponent_sample < 2:
        warnings.append("thin opponent pass-defense sample")
    warnings.append(
        "pregame model; weather, OL injuries, inactives, and game-script changes are not yet fully modeled"
    )

    evidence = ModelEvidence(
        sport="NFL",
        model_name="NFL Passing Yards: attempts + YPA + recent CPOE + opponent pass defense",
        fair_probability=probability,
        confidence=confidence,
        sample_size=int(total_attempts),
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
    return NFLPassingProjection(
        evidence=evidence,
        player_name=context.player_name,
        milestone_yards=milestone_yards,
        line=line,
        game_title=f"{context.team} vs {context.opponent}",
        team=context.team,
        opponent=context.opponent,
        target_week=context.week,
        expected_attempts=expected_attempts,
        adjusted_ypa=adjusted_ypa,
        expected_passing_yards=expected_yards,
        passing_yards_sd=sd,
    )
