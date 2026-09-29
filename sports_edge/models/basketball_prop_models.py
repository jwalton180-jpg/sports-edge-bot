from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from statistics import mean, pstdev

import requests

from sports_edge.core.math import clamp
from sports_edge.data.basketball_prop_data import (
    latest_team_abbreviation,
    latest_team_name,
    minutes_value,
    player_history,
    prior_player_history,
    stat_value,
)
from sports_edge.data.public_team_data import _truthy, _wnba_schedule_rows
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.wnba_lines_model import (
    _match_event_row,
    _resolve_matchup,
    _score_projection,
)


@dataclass(frozen=True)
class BasketballPropProjection:
    evidence: ModelEvidence
    player_name: str
    market_key: str
    market_label: str
    milestone: int
    line: float
    game_title: str
    projected_mean: float
    projected_variance: float
    expected_minutes: float


_METRICS = {
    "Points": ("points", "player_points", "Points"),
    "Rebounds": ("rebounds", "player_rebounds", "Rebounds"),
    "Assists": ("assists", "player_assists", "Assists"),
    "Three-Pointers": ("threes", "player_threes", "Threes"),
    "Points + Rebounds + Assists": (
        "pra",
        "player_points_rebounds_assists",
        "Points + Rebounds + Assists",
    ),
}


def _nb_tail(mean_value: float, variance: float, milestone: int) -> float:
    """P(X >= milestone) using Poisson or NB2 when empirical variance is larger."""
    mu = max(0.01, float(mean_value))
    k = max(0, int(milestone))
    if k <= 0:
        return 1.0

    if variance <= mu * 1.03:
        term = math.exp(-mu)
        cdf = term
        for x in range(1, k):
            term *= mu / x
            cdf += term
        return clamp(1.0 - cdf, 0.01, 0.99)

    r = max(0.25, mu * mu / max(variance - mu, 1e-6))
    p = r / (r + mu)
    cdf = 0.0
    for x in range(k):
        log_pmf = (
            math.lgamma(x + r)
            - math.lgamma(r)
            - math.lgamma(x + 1)
            + r * math.log(p)
            + x * math.log(1.0 - p)
        )
        cdf += math.exp(log_pmf)
    return clamp(1.0 - cdf, 0.01, 0.99)


def _values(rows: tuple[dict, ...], metric: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = stat_value(row, metric)
        if value is not None:
            out.append(float(value))
    return out


def _minutes(rows: tuple[dict, ...]) -> list[float]:
    return [float(v) for row in rows if (v := minutes_value(row)) is not None and v > 0]


def _rate(rows: tuple[dict, ...], metric: str) -> float | None:
    total_stat = 0.0
    total_minutes = 0.0
    for row in rows:
        stat = stat_value(row, metric)
        mins = minutes_value(row)
        if stat is None or mins is None or mins <= 0:
            continue
        total_stat += float(stat)
        total_minutes += float(mins)
    return total_stat / total_minutes if total_minutes > 0 else None


def _expected_minutes(current: tuple[dict, ...], prior: tuple[dict, ...]) -> tuple[float, str | None]:
    cur = _minutes(current)
    old = _minutes(prior[-20:])
    if not cur:
        return 0.0, None

    season_mean = mean(cur)
    prior_mean = mean(old) if old else season_mean
    current_weight = clamp(0.50 + 0.04 * len(cur), 0.50, 0.88)
    baseline = current_weight * season_mean + (1.0 - current_weight) * prior_mean

    recent = mean(cur[-5:])
    role_ratio = clamp(recent / max(season_mean, 1.0), 0.75, 1.25)
    role_factor = clamp(1.0 + 0.45 * (role_ratio - 1.0), 0.90, 1.10)
    expected = clamp(baseline * role_factor, 8.0, 42.0)

    reason = None
    if abs(role_factor - 1.0) >= 0.025:
        direction = "up" if role_factor > 1.0 else "down"
        reason = (
            f"Minutes role trend {direction}: last-5 {recent:.1f} vs "
            f"season {season_mean:.1f} ({role_factor:.3f} bounded multiplier)"
        )
    return expected, reason


def _blended_rate(
    current: tuple[dict, ...],
    prior: tuple[dict, ...],
    metric: str,
) -> tuple[float | None, str | None]:
    cur_rate = _rate(current, metric)
    prior_rate = _rate(prior[-24:], metric)
    if cur_rate is None and prior_rate is None:
        return None, None
    if cur_rate is None:
        return prior_rate, "No current-season rate; prior-season baseline only"
    if prior_rate is None:
        base = cur_rate
    else:
        cur_minutes = sum(_minutes(current))
        w = clamp(0.45 + cur_minutes / 900.0, 0.45, 0.88)
        base = w * cur_rate + (1.0 - w) * prior_rate

    recent_rows = current[-5:]
    recent_rate = _rate(recent_rows, metric)
    if recent_rate is None:
        return base, None
    # Recent efficiency gets modest weight. Minutes carry the stronger role signal.
    blended = 0.86 * base + 0.14 * recent_rate
    delta = recent_rate / max(base, 1e-6) - 1.0
    reason = None
    if abs(delta) >= 0.08:
        reason = (
            f"Recent per-minute {metric} trend {delta:+.1%} vs blended baseline "
            f"(14% recent-efficiency weight)"
        )
    return blended, reason


def _opponent_factor(
    current: tuple[dict, ...],
    prior: tuple[dict, ...],
    *,
    metric: str,
    opponent_abbr: str,
    baseline_rate: float,
) -> tuple[float, str | None, int]:
    rows = list(prior[-30:]) + list(current)
    opp = str(opponent_abbr or "").upper()
    h2h = tuple(
        row for row in rows
        if str(row.get("opponent_team_abbreviation") or "").upper() == opp
    )
    if len(h2h) < 2:
        return 1.0, None, len(h2h)
    rate = _rate(h2h, metric)
    if rate is None or baseline_rate <= 0:
        return 1.0, None, len(h2h)
    raw = clamp(rate / baseline_rate, 0.65, 1.35)
    strength = len(h2h) / (len(h2h) + 5.0)
    factor = clamp(1.0 + 0.22 * strength * (raw - 1.0), 0.95, 1.05)
    if abs(factor - 1.0) < 0.01:
        return 1.0, None, len(h2h)
    return (
        factor,
        f"Prior matchup rate vs {opp}: {len(h2h)} game(s), {factor:.3f} shrunk matchup multiplier",
        len(h2h),
    )


def _pregame_only(event_date: date, event_ticker: str) -> bool:
    row = _match_event_row(_wnba_schedule_rows(), event_date, event_ticker)
    # Fail closed when the exact scheduled event cannot be verified. Kalshi
    # contracts can remain active after tip, so "market active" is not proof
    # that a player prop is still pregame.
    if row is None:
        return False
    if _truthy(row.get("status_type_completed")):
        return False
    state = str(row.get("status_type_state") or "").strip().lower()
    return state in {"pre", "scheduled", ""}


def project_wnba_player_prop(
    *,
    player_name: str,
    family: str,
    milestone: int,
    event_date: date,
    event_ticker: str,
) -> BasketballPropProjection | None:
    if family not in _METRICS:
        return None
    metric, market_key, market_label = _METRICS[family]

    try:
        if not _pregame_only(event_date, event_ticker):
            return None

        matchup = _resolve_matchup(event_date, event_ticker)
        if matchup is None:
            return None

        current = player_history(
            "WNBA",
            player_name,
            event_date.year,
            event_date.isoformat(),
        )
        prior = prior_player_history("WNBA", player_name, event_date.year)
        if len(current) < 6 or len(current) + len(prior) < 10:
            return None

        player_team = latest_team_abbreviation(current)
        if player_team == matchup.away_abbr.upper():
            opponent = matchup.home_abbr.upper()
            player_team_name = matchup.away_name
            team_games = matchup.away_games
            projected_team_score = _score_projection(matchup)[0]
        elif player_team == matchup.home_abbr.upper():
            opponent = matchup.away_abbr.upper()
            player_team_name = matchup.home_name
            team_games = matchup.home_games
            projected_team_score = _score_projection(matchup)[1]
        else:
            return None

        expected_minutes, minutes_reason = _expected_minutes(current, prior)
        rate, rate_reason = _blended_rate(current, prior, metric)
        if rate is None or expected_minutes <= 0:
            return None

        opponent_mult, opponent_reason, h2h_games = _opponent_factor(
            current,
            prior,
            metric=metric,
            opponent_abbr=opponent,
            baseline_rate=rate,
        )

        historical_team_pf = mean(g.pf for g in team_games) if team_games else projected_team_score
        environment = clamp(
            projected_team_score / max(historical_team_pf, 1.0),
            0.92,
            1.08,
        )
        # Rebounds are less directly tied to scoring environment; damp that input.
        environment_effect = (
            1.0 + 0.45 * (environment - 1.0)
            if metric == "rebounds"
            else environment
        )

        projected = clamp(
            rate * expected_minutes * opponent_mult * environment_effect,
            0.05,
            70.0,
        )

        cur_vals = _values(current, metric)
        prior_vals = _values(prior[-20:], metric)
        empirical = cur_vals + prior_vals
        variance = (
            pstdev(empirical) ** 2
            if len(empirical) >= 5
            else max(projected * 1.25, 2.0)
        )
        # Preserve realistic count overdispersion after mean adjustments.
        if metric == "threes":
            variance = max(variance, projected * 1.20)
        elif metric in {"assists", "rebounds"}:
            variance = max(variance, projected * 1.08)
        else:
            variance = max(variance, projected * 1.12)

        probability = _nb_tail(projected, variance, int(milestone))

        minute_sd = pstdev(_minutes(current[-10:])) if len(_minutes(current[-10:])) >= 3 else 6.0
        confidence = 0.46
        confidence += 0.11 * clamp(len(current) / 24.0, 0.0, 1.0)
        confidence += 0.05 * clamp(len(prior) / 20.0, 0.0, 1.0)
        confidence += 0.05 * clamp(1.0 - minute_sd / 12.0, 0.0, 1.0)
        confidence += 0.025 if h2h_games >= 2 else 0.0
        if metric == "pra":
            confidence -= 0.02
        confidence = clamp(confidence, 0.46, 0.70)

        factors = [
            f"Expected minutes {expected_minutes:.1f}; current history {len(current)} game(s), prior {len(prior)}",
            f"Projected {market_label.lower()} {projected:.2f}; empirical variance {variance:.2f}",
            f"Projected team score {projected_team_score:.1f} vs historical {historical_team_pf:.1f} ({environment_effect:.3f} environment multiplier)",
            f"Current team {latest_team_name(current) or player_team_name}; opponent {opponent}",
        ]
        for reason in (minutes_reason, rate_reason, opponent_reason):
            if reason:
                factors.append(reason)

        warnings = [
            "pregame player model; injuries, confirmed availability, and starting lineup changes are not separately verified",
        ]
        if len(current) < 10:
            warnings.append("thin current-season player sample; prior-season shrinkage carries material weight")
        if minute_sd >= 8:
            warnings.append(f"recent minutes are volatile (SD {minute_sd:.1f}); role uncertainty is elevated")

        evidence = ModelEvidence(
            sport="WNBA",
            model_name=(
                "WNBA Player Prop: minutes + per-minute production + "
                "recent role + opponent/team environment"
            ),
            fair_probability=probability,
            confidence=confidence,
            sample_size=len(current) + min(len(prior), 20),
            factors=tuple(factors),
            warnings=tuple(warnings),
        )
        return BasketballPropProjection(
            evidence=evidence,
            player_name=str(current[-1].get("athlete_display_name") or player_name),
            market_key=market_key,
            market_label=market_label,
            milestone=int(milestone),
            line=float(milestone) - 0.5,
            game_title=matchup.title,
            projected_mean=projected,
            projected_variance=variance,
            expected_minutes=expected_minutes,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError, OverflowError):
        return None
