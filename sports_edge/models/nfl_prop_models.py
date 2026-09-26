from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from statistics import mean, pstdev

from sports_edge.core.math import clamp
from sports_edge.data.nfl_prop_data import (
    NFLPlayerContext,
    defensive_week_values,
    league_defensive_week_values,
    nfl_season,
    player_context,
)
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.nfl import normal_over_probability


@dataclass(frozen=True)
class NFLPropProjection:
    evidence: ModelEvidence
    player_name: str
    position: str
    market_key: str
    market_label: str
    milestone: int
    line: float
    game_title: str
    projected_mean: float
    projected_sd: float | None


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _values(rows, key: str) -> list[float]:
    return [_f(row, key) for row in rows]


def _recent(values: list[float], n: int = 3) -> list[float]:
    return values[-n:] if values else []


def _blend_mean(current: list[float], prior: list[float]) -> tuple[float, float]:
    if not current and not prior:
        return 0.0, 0.0
    if not prior:
        return mean(current), 1.0
    if not current:
        return mean(prior), 0.0

    # Early-season current form matters, but three games should not erase a
    # full prior season. Current reaches ~59% weight after three games and
    # asymptotes below 70% until the sample grows.
    current_weight = min(0.68, 0.35 + 0.08 * len(current))
    blended = current_weight * mean(current) + (1.0 - current_weight) * mean(prior)
    return blended, current_weight


def _weighted_sd(current: list[float], prior: list[float], floor: float) -> float:
    samples: list[float] = []
    weights: list[float] = []
    for value in prior[-17:]:
        samples.append(value)
        weights.append(0.45)
    for value in current:
        samples.append(value)
        weights.append(1.60)
    if len(samples) < 2:
        return floor
    total = sum(weights)
    mu = sum(v * w for v, w in zip(samples, weights)) / total
    var = sum(w * (v - mu) ** 2 for v, w in zip(samples, weights)) / total
    return max(floor, var ** 0.5)


def _matchup_factor(ctx: NFLPlayerContext, stat_key: str) -> tuple[float, int, float | None]:
    season = nfl_season(ctx.event_date)
    opp = defensive_week_values(
        season=season,
        opponent_team=ctx.opponent,
        stat_key=stat_key,
    )
    league = league_defensive_week_values(season=season, stat_key=stat_key)
    if not opp or not league:
        return 1.0, 0, None

    opp_mean = mean(opp)
    league_mean = mean(league)
    if league_mean <= 0:
        return 1.0, len(opp), None

    raw = clamp(opp_mean / league_mean, 0.70, 1.30)
    # Only a few current-season games exist in September. Shrink matchup
    # strength hard toward neutral until the defense sample deepens.
    reliability = len(opp) / (len(opp) + 5.0)
    factor = clamp(1.0 + reliability * (raw - 1.0), 0.88, 1.12)
    return factor, len(opp), opp_mean


def _sample_and_confidence(
    ctx: NFLPlayerContext,
    *,
    matchup_games: int,
    role_signal: float,
    volatile: bool = False,
) -> tuple[int, float]:
    current_n = len(ctx.current_rows)
    prior_n = min(len(ctx.prior_rows), 17)
    sample = current_n + prior_n

    confidence = 0.36
    confidence += 0.12 * clamp(current_n / 4.0, 0.0, 1.0)
    confidence += 0.10 * clamp(prior_n / 12.0, 0.0, 1.0)
    confidence += 0.07 * clamp(matchup_games / 6.0, 0.0, 1.0)
    confidence += 0.06 * clamp(role_signal, 0.0, 1.0)
    if not ctx.prior_rows:
        confidence -= 0.04
    if volatile:
        confidence -= 0.04
    return max(1, sample), clamp(confidence, 0.35, 0.70)


def _poisson_at_least(lam: float, k: int) -> float:
    lam = clamp(lam, 0.01, 8.0)
    if k <= 0:
        return 1.0
    cdf = 0.0
    for i in range(k):
        cdf += exp(-lam) * (lam ** i) / factorial(i)
    return clamp(1.0 - cdf, 0.005, 0.995)


def _context(player_name: str, event_date, event_ticker: str | None = None) -> NFLPlayerContext | None:
    ctx = player_context(player_name, event_date, event_ticker)
    if ctx is None or len(ctx.current_rows) < 2:
        return None
    return ctx


def project_nfl_passing_yards(
    *,
    player_name: str,
    milestone_yards: int,
    event_date,
    event_ticker: str | None = None,
) -> NFLPropProjection | None:
    if milestone_yards < 50 or milestone_yards > 500:
        return None
    ctx = _context(player_name, event_date, event_ticker)
    if ctx is None or ctx.position.upper() != "QB":
        return None

    cur_yards = _values(ctx.current_rows, "passing_yards")
    prior_yards = _values(ctx.prior_rows, "passing_yards")
    cur_att = _values(ctx.current_rows, "attempts")
    prior_att = _values(ctx.prior_rows, "attempts")

    base_yards, current_weight = _blend_mean(cur_yards, prior_yards)
    expected_att, _ = _blend_mean(cur_att, prior_att)

    cur_total_att = sum(cur_att)
    prior_total_att = sum(prior_att)
    cur_ypa = sum(cur_yards) / cur_total_att if cur_total_att > 0 else 0.0
    prior_ypa = sum(prior_yards) / prior_total_att if prior_total_att > 0 else 0.0
    if prior_ypa > 0 and cur_ypa > 0:
        ypa = current_weight * cur_ypa + (1.0 - current_weight) * prior_ypa
    else:
        ypa = cur_ypa or prior_ypa
    efficiency_projection = expected_att * ypa if ypa > 0 else base_yards
    mean_yards = 0.55 * base_yards + 0.45 * efficiency_projection

    matchup, matchup_games, opp_allowed = _matchup_factor(ctx, "passing_yards")
    projected = clamp(mean_yards * matchup, 80.0, 420.0)
    sd = _weighted_sd(cur_yards, prior_yards, 42.0)
    probability = normal_over_probability(projected, sd, float(milestone_yards) - 0.5)

    recent_yards = _recent(cur_yards)
    recent_att = _recent(cur_att)
    factors = [
        f"Current-season passing yards {mean(cur_yards):.1f}/game over {len(cur_yards)} game(s)",
        f"Expected pass attempts {expected_att:.1f}; blended yards/attempt {ypa:.2f}",
        f"Recent passing yards {mean(recent_yards):.1f}/game on {mean(recent_att):.1f} attempts" if recent_yards and recent_att else "Recent passing sample unavailable",
        f"Opponent pass-defense factor {matchup:.3f} from {matchup_games} current game(s)",
        f"Projected mean {projected:.1f} yards with {sd:.1f} yard weekly volatility",
    ]
    if ctx.prior_rows:
        factors.append(f"Prior-season passing baseline {mean(prior_yards):.1f}/game over {len(prior_yards)} game(s)")

    sample, confidence = _sample_and_confidence(
        ctx,
        matchup_games=matchup_games,
        role_signal=clamp(expected_att / 36.0, 0.0, 1.0),
    )
    warnings = []
    if len(ctx.current_rows) < 4:
        warnings.append("early-season NFL sample; prior-year baseline receives material weight")
    if matchup_games < 3:
        warnings.append("thin current opponent-defense sample; matchup adjustment heavily shrunk")

    evidence = ModelEvidence(
        sport="NFL",
        model_name="NFL Passing Yards: volume + efficiency + prior + opponent defense",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
    return NFLPropProjection(
        evidence=evidence,
        player_name=ctx.player_name,
        position=ctx.position,
        market_key="player_pass_yds",
        market_label="Passing Yards",
        milestone=milestone_yards,
        line=float(milestone_yards) - 0.5,
        game_title=ctx.game_title,
        projected_mean=projected,
        projected_sd=sd,
    )


def project_nfl_passing_tds(
    *,
    player_name: str,
    milestone_tds: int,
    event_date,
    event_ticker: str | None = None,
) -> NFLPropProjection | None:
    if milestone_tds not in {1, 2, 3, 4, 5}:
        return None
    ctx = _context(player_name, event_date, event_ticker)
    if ctx is None or ctx.position.upper() != "QB":
        return None

    cur_td = _values(ctx.current_rows, "passing_tds")
    prior_td = _values(ctx.prior_rows, "passing_tds")
    cur_att = _values(ctx.current_rows, "attempts")
    prior_att = _values(ctx.prior_rows, "attempts")

    td_rate, current_weight = _blend_mean(cur_td, prior_td)
    expected_att, _ = _blend_mean(cur_att, prior_att)
    cur_att_mean = mean(cur_att) if cur_att else expected_att
    volume_factor = clamp(expected_att / max(cur_att_mean, 1.0), 0.85, 1.15)

    matchup, matchup_games, opp_allowed = _matchup_factor(ctx, "passing_tds")
    lam = clamp(td_rate * volume_factor * matchup, 0.20, 4.5)
    probability = _poisson_at_least(lam, milestone_tds)

    factors = [
        f"Current passing TD rate {mean(cur_td):.2f}/game over {len(cur_td)} game(s)",
        f"Expected pass attempts {expected_att:.1f}",
        f"Opponent passing-TD factor {matchup:.3f} from {matchup_games} current game(s)",
        f"Poisson scoring mean {lam:.2f} passing TDs",
    ]
    if ctx.prior_rows:
        factors.append(f"Prior passing TD baseline {mean(prior_td):.2f}/game over {len(prior_td)} game(s)")

    sample, confidence = _sample_and_confidence(
        ctx,
        matchup_games=matchup_games,
        role_signal=clamp(expected_att / 36.0, 0.0, 1.0),
        volatile=True,
    )
    warnings = ["touchdown counts are intrinsically high-variance"]
    if len(ctx.current_rows) < 4:
        warnings.append("early-season NFL sample; prior-year baseline receives material weight")

    evidence = ModelEvidence(
        sport="NFL",
        model_name="NFL Passing TDs: scoring rate + volume + prior + opponent defense",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
    return NFLPropProjection(
        evidence=evidence,
        player_name=ctx.player_name,
        position=ctx.position,
        market_key="player_pass_tds",
        market_label="Passing TDs",
        milestone=milestone_tds,
        line=float(milestone_tds) - 0.5,
        game_title=ctx.game_title,
        projected_mean=lam,
        projected_sd=None,
    )


def project_nfl_pass_attempts(
    *,
    player_name: str,
    milestone_attempts: int,
    event_date,
    event_ticker: str | None = None,
) -> NFLPropProjection | None:
    if milestone_attempts < 10 or milestone_attempts > 65:
        return None
    ctx = _context(player_name, event_date, event_ticker)
    if ctx is None or ctx.position.upper() != "QB":
        return None

    cur = _values(ctx.current_rows, "attempts")
    prior = _values(ctx.prior_rows, "attempts")
    projected, _ = _blend_mean(cur, prior)
    sd = _weighted_sd(cur, prior, 4.0)
    probability = normal_over_probability(projected, sd, float(milestone_attempts) - 0.5)

    factors = [
        f"Current pass attempts {mean(cur):.1f}/game over {len(cur)} game(s)",
        f"Projected pass attempts {projected:.1f} with {sd:.1f} weekly volatility",
    ]
    if ctx.prior_rows:
        factors.append(f"Prior pass-attempt baseline {mean(prior):.1f}/game over {len(prior)} game(s)")
    sample, confidence = _sample_and_confidence(
        ctx,
        matchup_games=0,
        role_signal=clamp(projected / 36.0, 0.0, 1.0),
    )
    warnings = ["game script and pace are not yet separately modeled"]
    if len(ctx.current_rows) < 4:
        warnings.append("early-season NFL sample; prior-year baseline receives material weight")

    evidence = ModelEvidence(
        sport="NFL",
        model_name="NFL Pass Attempts: current volume + prior-season role",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
    return NFLPropProjection(
        evidence=evidence,
        player_name=ctx.player_name,
        position=ctx.position,
        market_key="player_pass_attempts",
        market_label="Pass Attempts",
        milestone=milestone_attempts,
        line=float(milestone_attempts) - 0.5,
        game_title=ctx.game_title,
        projected_mean=projected,
        projected_sd=sd,
    )


def project_nfl_pass_completions(
    *,
    player_name: str,
    milestone_completions: int,
    event_date,
    event_ticker: str | None = None,
) -> NFLPropProjection | None:
    if milestone_completions < 5 or milestone_completions > 50:
        return None
    ctx = _context(player_name, event_date, event_ticker)
    if ctx is None or ctx.position.upper() != "QB":
        return None

    cur_comp = _values(ctx.current_rows, "completions")
    prior_comp = _values(ctx.prior_rows, "completions")
    cur_att = _values(ctx.current_rows, "attempts")
    prior_att = _values(ctx.prior_rows, "attempts")

    base_comp, current_weight = _blend_mean(cur_comp, prior_comp)
    expected_att, _ = _blend_mean(cur_att, prior_att)
    cur_att_total = sum(cur_att)
    prior_att_total = sum(prior_att)
    cur_rate = sum(cur_comp) / cur_att_total if cur_att_total > 0 else 0.0
    prior_rate = sum(prior_comp) / prior_att_total if prior_att_total > 0 else 0.0
    if cur_rate > 0 and prior_rate > 0:
        completion_rate = current_weight * cur_rate + (1.0 - current_weight) * prior_rate
    else:
        completion_rate = cur_rate or prior_rate
    completion_rate = clamp(completion_rate, 0.45, 0.80)
    projected = clamp(0.55 * base_comp + 0.45 * expected_att * completion_rate, 5.0, 45.0)
    sd = _weighted_sd(cur_comp, prior_comp, 2.8)
    probability = normal_over_probability(projected, sd, float(milestone_completions) - 0.5)

    factors = [
        f"Current completions {mean(cur_comp):.1f}/game over {len(cur_comp)} game(s)",
        f"Expected pass attempts {expected_att:.1f}; blended completion rate {completion_rate:.1%}",
        f"Projected completions {projected:.1f} with {sd:.1f} weekly volatility",
    ]
    if ctx.prior_rows:
        factors.append(f"Prior completions baseline {mean(prior_comp):.1f}/game over {len(prior_comp)} game(s)")
    sample, confidence = _sample_and_confidence(
        ctx,
        matchup_games=0,
        role_signal=clamp(expected_att / 36.0, 0.0, 1.0),
    )
    warnings = ["game script and pass-rush pressure are not yet separately modeled"]
    if len(ctx.current_rows) < 4:
        warnings.append("early-season NFL sample; prior-year baseline receives material weight")

    evidence = ModelEvidence(
        sport="NFL",
        model_name="NFL Pass Completions: attempts + completion rate + prior",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
    return NFLPropProjection(
        evidence=evidence,
        player_name=ctx.player_name,
        position=ctx.position,
        market_key="player_pass_completions",
        market_label="Pass Completions",
        milestone=milestone_completions,
        line=float(milestone_completions) - 0.5,
        game_title=ctx.game_title,
        projected_mean=projected,
        projected_sd=sd,
    )


def project_nfl_pass_interceptions(
    *,
    player_name: str,
    milestone_interceptions: int,
    event_date,
    event_ticker: str | None = None,
) -> NFLPropProjection | None:
    if milestone_interceptions not in {1, 2, 3, 4}:
        return None
    ctx = _context(player_name, event_date, event_ticker)
    if ctx is None or ctx.position.upper() != "QB":
        return None

    cur_int = _values(ctx.current_rows, "passing_interceptions")
    prior_int = _values(ctx.prior_rows, "passing_interceptions")
    cur_att = _values(ctx.current_rows, "attempts")
    prior_att = _values(ctx.prior_rows, "attempts")

    game_rate, current_weight = _blend_mean(cur_int, prior_int)
    expected_att, _ = _blend_mean(cur_att, prior_att)
    cur_att_total = sum(cur_att)
    prior_att_total = sum(prior_att)
    cur_rate = sum(cur_int) / cur_att_total if cur_att_total > 0 else 0.0
    prior_rate = sum(prior_int) / prior_att_total if prior_att_total > 0 else 0.0
    if prior_att_total > 0:
        int_rate = current_weight * cur_rate + (1.0 - current_weight) * prior_rate
    else:
        int_rate = cur_rate
    attempt_projection = expected_att * max(0.005, int_rate)
    lam = clamp(0.55 * game_rate + 0.45 * attempt_projection, 0.05, 2.5)
    probability = _poisson_at_least(lam, milestone_interceptions)

    factors = [
        f"Current interceptions {mean(cur_int):.2f}/game over {len(cur_int)} game(s)",
        f"Expected pass attempts {expected_att:.1f}; blended interception rate {int_rate:.2%}",
        f"Poisson interception mean {lam:.2f}",
    ]
    if ctx.prior_rows:
        factors.append(f"Prior interceptions baseline {mean(prior_int):.2f}/game over {len(prior_int)} game(s)")
    sample, confidence = _sample_and_confidence(
        ctx,
        matchup_games=0,
        role_signal=clamp(expected_att / 36.0, 0.0, 1.0),
        volatile=True,
    )
    confidence = min(confidence, 0.58)
    warnings = [
        "interceptions are high-variance discrete events",
        "opponent takeaway skill and pass-rush pressure are not yet separately modeled",
    ]
    if len(ctx.current_rows) < 4:
        warnings.append("early-season NFL sample; prior-year baseline receives material weight")

    evidence = ModelEvidence(
        sport="NFL",
        model_name="NFL Pass Interceptions: attempt volume + interception rate + prior",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
    return NFLPropProjection(
        evidence=evidence,
        player_name=ctx.player_name,
        position=ctx.position,
        market_key="player_pass_interceptions",
        market_label="Pass Interceptions",
        milestone=milestone_interceptions,
        line=float(milestone_interceptions) - 0.5,
        game_title=ctx.game_title,
        projected_mean=lam,
        projected_sd=None,
    )


def project_nfl_receiving_yards(
    *,
    player_name: str,
    milestone_yards: int,
    event_date,
    event_ticker: str | None = None,
) -> NFLPropProjection | None:
    if milestone_yards < 10 or milestone_yards > 250:
        return None
    ctx = _context(player_name, event_date, event_ticker)
    if ctx is None or ctx.position.upper() not in {"WR", "TE", "RB", "FB"}:
        return None

    cur_yards = _values(ctx.current_rows, "receiving_yards")
    prior_yards = _values(ctx.prior_rows, "receiving_yards")
    cur_targets = _values(ctx.current_rows, "targets")
    prior_targets = _values(ctx.prior_rows, "targets")

    base_yards, current_weight = _blend_mean(cur_yards, prior_yards)
    expected_targets, _ = _blend_mean(cur_targets, prior_targets)

    cur_targets_total = sum(cur_targets)
    prior_targets_total = sum(prior_targets)
    cur_ypt = sum(cur_yards) / cur_targets_total if cur_targets_total > 0 else 0.0
    prior_ypt = sum(prior_yards) / prior_targets_total if prior_targets_total > 0 else 0.0
    if prior_ypt > 0 and cur_ypt > 0:
        ypt = current_weight * cur_ypt + (1.0 - current_weight) * prior_ypt
    else:
        ypt = cur_ypt or prior_ypt
    opportunity_projection = expected_targets * ypt if ypt > 0 else base_yards
    mean_yards = 0.55 * base_yards + 0.45 * opportunity_projection

    matchup, matchup_games, opp_allowed = _matchup_factor(ctx, "receiving_yards")
    projected = clamp(mean_yards * matchup, 3.0, 180.0)
    sd = _weighted_sd(cur_yards, prior_yards, 20.0)
    probability = normal_over_probability(projected, sd, float(milestone_yards) - 0.5)

    target_shares = [_f(r, "target_share") for r in ctx.current_rows if _f(r, "target_share") > 0]
    air_shares = [_f(r, "air_yards_share") for r in ctx.current_rows if _f(r, "air_yards_share") > 0]
    factors = [
        f"Current receiving yards {mean(cur_yards):.1f}/game over {len(cur_yards)} game(s)",
        f"Expected targets {expected_targets:.1f}; blended yards/target {ypt:.2f}",
        f"Opponent receiving-yard factor {matchup:.3f} from {matchup_games} current game(s)",
        f"Projected mean {projected:.1f} yards with {sd:.1f} yard weekly volatility",
    ]
    if target_shares:
        factors.append(f"Current target share {mean(target_shares):.1%}")
    if air_shares:
        factors.append(f"Current air-yards share {mean(air_shares):.1%}")
    if ctx.prior_rows:
        factors.append(f"Prior receiving baseline {mean(prior_yards):.1f}/game over {len(prior_yards)} game(s)")

    sample, confidence = _sample_and_confidence(
        ctx,
        matchup_games=matchup_games,
        role_signal=clamp(expected_targets / 8.0, 0.0, 1.0),
    )
    warnings = []
    if len(ctx.current_rows) < 4:
        warnings.append("early-season NFL sample; prior-year baseline receives material weight")
    if expected_targets < 3.0:
        warnings.append("low projected target volume")

    evidence = ModelEvidence(
        sport="NFL",
        model_name="NFL Receiving Yards: targets + efficiency + prior + opponent defense",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
    return NFLPropProjection(
        evidence=evidence,
        player_name=ctx.player_name,
        position=ctx.position,
        market_key="player_reception_yds",
        market_label="Receiving Yards",
        milestone=milestone_yards,
        line=float(milestone_yards) - 0.5,
        game_title=ctx.game_title,
        projected_mean=projected,
        projected_sd=sd,
    )


def project_nfl_rushing_yards(
    *,
    player_name: str,
    milestone_yards: int,
    event_date,
    event_ticker: str | None = None,
) -> NFLPropProjection | None:
    if milestone_yards < 5 or milestone_yards > 250:
        return None
    ctx = _context(player_name, event_date, event_ticker)
    if ctx is None or ctx.position.upper() not in {"QB", "RB", "WR", "FB"}:
        return None

    cur_yards = _values(ctx.current_rows, "rushing_yards")
    prior_yards = _values(ctx.prior_rows, "rushing_yards")
    cur_carries = _values(ctx.current_rows, "carries")
    prior_carries = _values(ctx.prior_rows, "carries")

    base_yards, current_weight = _blend_mean(cur_yards, prior_yards)
    expected_carries, _ = _blend_mean(cur_carries, prior_carries)

    cur_carries_total = sum(cur_carries)
    prior_carries_total = sum(prior_carries)
    cur_ypc = sum(cur_yards) / cur_carries_total if cur_carries_total > 0 else 0.0
    prior_ypc = sum(prior_yards) / prior_carries_total if prior_carries_total > 0 else 0.0
    if prior_ypc > 0 and cur_ypc > 0:
        ypc = current_weight * cur_ypc + (1.0 - current_weight) * prior_ypc
    else:
        ypc = cur_ypc or prior_ypc

    opportunity_projection = expected_carries * ypc if ypc > 0 else base_yards
    mean_yards = 0.55 * base_yards + 0.45 * opportunity_projection
    matchup, matchup_games, opp_allowed = _matchup_factor(ctx, "rushing_yards")
    projected = clamp(mean_yards * matchup, -5.0, 190.0)
    sd = _weighted_sd(cur_yards, prior_yards, 16.0)
    probability = normal_over_probability(projected, sd, float(milestone_yards) - 0.5)

    recent_yards = _recent(cur_yards)
    recent_carries = _recent(cur_carries)
    factors = [
        f"Current rushing yards {mean(cur_yards):.1f}/game over {len(cur_yards)} game(s)",
        f"Expected carries {expected_carries:.1f}; blended yards/carry {ypc:.2f}",
        f"Recent rushing yards {mean(recent_yards):.1f}/game on {mean(recent_carries):.1f} carries" if recent_yards and recent_carries else "Recent rushing sample unavailable",
        f"Opponent rushing-yard factor {matchup:.3f} from {matchup_games} current game(s)",
        f"Projected mean {projected:.1f} yards with {sd:.1f} yard weekly volatility",
    ]
    if ctx.prior_rows:
        factors.append(f"Prior rushing baseline {mean(prior_yards):.1f}/game over {len(prior_yards)} game(s)")

    sample, confidence = _sample_and_confidence(
        ctx,
        matchup_games=matchup_games,
        role_signal=clamp(expected_carries / 15.0, 0.0, 1.0),
    )
    warnings = []
    if len(ctx.current_rows) < 4:
        warnings.append("early-season NFL sample; prior-year baseline receives material weight")
    if expected_carries < 4.0:
        warnings.append("low projected rushing volume")

    evidence = ModelEvidence(
        sport="NFL",
        model_name="NFL Rushing Yards: carries + efficiency + prior + opponent defense",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
    return NFLPropProjection(
        evidence=evidence,
        player_name=ctx.player_name,
        position=ctx.position,
        market_key="player_rush_yds",
        market_label="Rushing Yards",
        milestone=milestone_yards,
        line=float(milestone_yards) - 0.5,
        game_title=ctx.game_title,
        projected_mean=projected,
        projected_sd=sd,
    )


def project_nfl_rush_attempts(
    *,
    player_name: str,
    milestone_attempts: int,
    event_date,
    event_ticker: str | None = None,
) -> NFLPropProjection | None:
    if milestone_attempts < 1 or milestone_attempts > 40:
        return None
    ctx = _context(player_name, event_date, event_ticker)
    if ctx is None or ctx.position.upper() not in {"QB", "RB", "WR", "FB"}:
        return None

    cur = _values(ctx.current_rows, "carries")
    prior = _values(ctx.prior_rows, "carries")
    projected, _ = _blend_mean(cur, prior)
    sd = _weighted_sd(cur, prior, 2.8)
    probability = normal_over_probability(projected, sd, float(milestone_attempts) - 0.5)

    factors = [
        f"Current carries {mean(cur):.1f}/game over {len(cur)} game(s)",
        f"Projected carries {projected:.1f} with {sd:.1f} weekly volatility",
    ]
    if ctx.prior_rows:
        factors.append(f"Prior carry baseline {mean(prior):.1f}/game over {len(prior)} game(s)")
    sample, confidence = _sample_and_confidence(
        ctx,
        matchup_games=0,
        role_signal=clamp(projected / 15.0, 0.0, 1.0),
    )
    confidence = min(confidence, 0.64)
    warnings = ["game script can materially change rushing volume"]
    if len(ctx.current_rows) < 4:
        warnings.append("early-season NFL sample; prior-year baseline receives material weight")

    evidence = ModelEvidence(
        sport="NFL",
        model_name="NFL Rush Attempts: current workload + prior-season role",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
    return NFLPropProjection(
        evidence=evidence,
        player_name=ctx.player_name,
        position=ctx.position,
        market_key="player_rush_attempts",
        market_label="Rush Attempts",
        milestone=milestone_attempts,
        line=float(milestone_attempts) - 0.5,
        game_title=ctx.game_title,
        projected_mean=projected,
        projected_sd=sd,
    )


def project_nfl_rush_receiving_yards(
    *,
    player_name: str,
    milestone_yards: int,
    event_date,
    event_ticker: str | None = None,
) -> NFLPropProjection | None:
    if milestone_yards < 10 or milestone_yards > 300:
        return None
    ctx = _context(player_name, event_date, event_ticker)
    if ctx is None or ctx.position.upper() not in {"QB", "RB", "WR", "TE", "FB"}:
        return None

    cur_rush = _values(ctx.current_rows, "rushing_yards")
    prior_rush = _values(ctx.prior_rows, "rushing_yards")
    cur_rec = _values(ctx.current_rows, "receiving_yards")
    prior_rec = _values(ctx.prior_rows, "receiving_yards")
    cur_total = [a + b for a, b in zip(cur_rush, cur_rec)]
    prior_total = [a + b for a, b in zip(prior_rush, prior_rec)]

    rush_base, _ = _blend_mean(cur_rush, prior_rush)
    rec_base, _ = _blend_mean(cur_rec, prior_rec)
    rush_factor, rush_games, _ = _matchup_factor(ctx, "rushing_yards")
    rec_factor, rec_games, _ = _matchup_factor(ctx, "receiving_yards")
    projected = clamp(rush_base * rush_factor + rec_base * rec_factor, 2.0, 240.0)
    sd = _weighted_sd(cur_total, prior_total, 20.0)
    probability = normal_over_probability(projected, sd, float(milestone_yards) - 0.5)

    current_ops = [
        _f(r, "carries") + _f(r, "targets")
        for r in ctx.current_rows
    ]
    prior_ops = [
        _f(r, "carries") + _f(r, "targets")
        for r in ctx.prior_rows
    ]
    expected_ops, _ = _blend_mean(current_ops, prior_ops)
    factors = [
        f"Current rushing + receiving yards {mean(cur_total):.1f}/game over {len(cur_total)} game(s)",
        f"Blended rushing component {rush_base:.1f} yards; receiving component {rec_base:.1f} yards",
        f"Opponent rushing factor {rush_factor:.3f}; receiving factor {rec_factor:.3f}",
        f"Expected carries + targets {expected_ops:.1f}",
        f"Projected combined mean {projected:.1f} yards with {sd:.1f} weekly volatility",
    ]
    if ctx.prior_rows:
        factors.append(f"Prior combined-yards baseline {mean(prior_total):.1f}/game over {len(prior_total)} game(s)")
    matchup_games = min(rush_games, rec_games)
    sample, confidence = _sample_and_confidence(
        ctx,
        matchup_games=matchup_games,
        role_signal=clamp(expected_ops / 16.0, 0.0, 1.0),
    )
    warnings = []
    if len(ctx.current_rows) < 4:
        warnings.append("early-season NFL sample; prior-year baseline receives material weight")

    evidence = ModelEvidence(
        sport="NFL",
        model_name="NFL Rush + Receiving Yards: dual-role volume + prior + matchup",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
    return NFLPropProjection(
        evidence=evidence,
        player_name=ctx.player_name,
        position=ctx.position,
        market_key="player_rush_reception_yds",
        market_label="Rushing + Receiving Yards",
        milestone=milestone_yards,
        line=float(milestone_yards) - 0.5,
        game_title=ctx.game_title,
        projected_mean=projected,
        projected_sd=sd,
    )


def project_nfl_receptions(
    *,
    player_name: str,
    milestone_receptions: int,
    event_date,
    event_ticker: str | None = None,
) -> NFLPropProjection | None:
    if milestone_receptions < 1 or milestone_receptions > 15:
        return None
    ctx = _context(player_name, event_date, event_ticker)
    if ctx is None or ctx.position.upper() not in {"WR", "TE", "RB", "FB"}:
        return None

    cur_rec = _values(ctx.current_rows, "receptions")
    prior_rec = _values(ctx.prior_rows, "receptions")
    cur_targets = _values(ctx.current_rows, "targets")
    prior_targets = _values(ctx.prior_rows, "targets")

    base_rec, current_weight = _blend_mean(cur_rec, prior_rec)
    expected_targets, _ = _blend_mean(cur_targets, prior_targets)

    cur_targets_total = sum(cur_targets)
    prior_targets_total = sum(prior_targets)
    cur_catch = sum(cur_rec) / cur_targets_total if cur_targets_total > 0 else 0.0
    prior_catch = sum(prior_rec) / prior_targets_total if prior_targets_total > 0 else 0.0
    if prior_catch > 0 and cur_catch > 0:
        catch_rate = current_weight * cur_catch + (1.0 - current_weight) * prior_catch
    else:
        catch_rate = cur_catch or prior_catch
    catch_rate = clamp(catch_rate, 0.35, 0.90)

    opportunity_projection = expected_targets * catch_rate
    projected = clamp(0.55 * base_rec + 0.45 * opportunity_projection, 0.2, 12.0)
    sd = _weighted_sd(cur_rec, prior_rec, 1.35)
    probability = normal_over_probability(
        projected,
        sd,
        float(milestone_receptions) - 0.5,
    )

    target_shares = [_f(r, "target_share") for r in ctx.current_rows if _f(r, "target_share") > 0]
    factors = [
        f"Current receptions {mean(cur_rec):.2f}/game over {len(cur_rec)} game(s)",
        f"Expected targets {expected_targets:.1f}; blended catch rate {catch_rate:.1%}",
        f"Projected mean {projected:.2f} receptions with {sd:.2f} weekly volatility",
    ]
    if target_shares:
        factors.append(f"Current target share {mean(target_shares):.1%}")
    if ctx.prior_rows:
        factors.append(f"Prior receptions baseline {mean(prior_rec):.2f}/game over {len(prior_rec)} game(s)")

    sample, confidence = _sample_and_confidence(
        ctx,
        matchup_games=0,
        role_signal=clamp(expected_targets / 8.0, 0.0, 1.0),
    )
    warnings = []
    if len(ctx.current_rows) < 4:
        warnings.append("early-season NFL sample; prior-year baseline receives material weight")
    if expected_targets < 3.0:
        warnings.append("low projected target volume")

    evidence = ModelEvidence(
        sport="NFL",
        model_name="NFL Receptions: targets + catch rate + prior-season role",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
    return NFLPropProjection(
        evidence=evidence,
        player_name=ctx.player_name,
        position=ctx.position,
        market_key="player_receptions",
        market_label="Receptions",
        milestone=milestone_receptions,
        line=float(milestone_receptions) - 0.5,
        game_title=ctx.game_title,
        projected_mean=projected,
        projected_sd=sd,
    )


def project_nfl_touchdowns(
    *,
    player_name: str,
    milestone_tds: int,
    event_date,
    event_ticker: str | None = None,
) -> NFLPropProjection | None:
    if milestone_tds not in {1, 2, 3}:
        return None
    ctx = _context(player_name, event_date, event_ticker)
    if ctx is None or ctx.position.upper() not in {"QB", "RB", "WR", "TE", "FB"}:
        return None

    def scored(rows) -> list[float]:
        return [_f(r, "rushing_tds") + _f(r, "receiving_tds") for r in rows]

    cur_td = scored(ctx.current_rows)
    prior_td = scored(ctx.prior_rows)
    cur_opp = [
        _f(r, "carries") + _f(r, "targets")
        for r in ctx.current_rows
    ]
    prior_opp = [
        _f(r, "carries") + _f(r, "targets")
        for r in ctx.prior_rows
    ]

    td_rate, current_weight = _blend_mean(cur_td, prior_td)
    expected_opp, _ = _blend_mean(cur_opp, prior_opp)
    cur_opp_mean = mean(cur_opp) if cur_opp else expected_opp
    role_factor = clamp(expected_opp / max(cur_opp_mean, 1.0), 0.85, 1.15)
    lam = clamp(td_rate * role_factor, 0.03, 2.2)
    probability = _poisson_at_least(lam, milestone_tds)

    factors = [
        f"Current scored-TD rate {mean(cur_td):.2f}/game over {len(cur_td)} game(s)",
        f"Expected carries + targets {expected_opp:.1f}",
        f"Poisson scoring mean {lam:.2f} TDs",
        "Passing TDs are excluded from this scorer market model",
    ]
    if ctx.prior_rows:
        factors.append(f"Prior scored-TD baseline {mean(prior_td):.2f}/game over {len(prior_td)} game(s)")

    sample, confidence = _sample_and_confidence(
        ctx,
        matchup_games=0,
        role_signal=clamp(expected_opp / 14.0, 0.0, 1.0),
        volatile=True,
    )
    confidence = min(confidence, 0.62)
    warnings = [
        "touchdown scorer outcomes are high-variance",
        "red-zone role and injury/active status are not yet separately modeled",
    ]
    if len(ctx.current_rows) < 4:
        warnings.append("early-season NFL sample; prior-year baseline receives material weight")

    evidence = ModelEvidence(
        sport="NFL",
        model_name="NFL Touchdowns: rushing/receiving scoring rate + opportunity + prior",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
    return NFLPropProjection(
        evidence=evidence,
        player_name=ctx.player_name,
        position=ctx.position,
        market_key="player_anytime_td",
        market_label="Player Touchdowns",
        milestone=milestone_tds,
        line=float(milestone_tds) - 0.5,
        game_title=ctx.game_title,
        projected_mean=lam,
        projected_sd=None,
    )
