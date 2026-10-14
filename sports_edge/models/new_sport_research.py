"""Independent, pre-event research models. Inputs MUST be sourced from prior events,
not Kalshi quotes. These estimates are unvalidated until prospective holdout gates pass."""
from __future__ import annotations
from dataclasses import dataclass
from math import exp, factorial, isfinite, sqrt

@dataclass(frozen=True)
class Projection:
    sport: str
    outcome: str
    probability: float | None
    model: str
    status: str
    reasons: tuple[str, ...]
    sample_size: int

def _valid_rate(value: float | None) -> bool:
    try:
        return value is not None and isfinite(float(value)) and 0 <= float(value) <= 15
    except (TypeError, ValueError):
        return False

def _poisson_at_least(rate: float, threshold: int) -> float:
    return max(0., min(1., 1. - exp(-rate) * sum(rate**k / factorial(k) for k in range(threshold))))

def _unavailable(sport: str, outcome: str, reason: str) -> Projection:
    return Projection(sport, outcome, None, "none", "RESEARCH_ONLY", (reason,), 0)

def hockey_goals(*, goals_per_game: float | None, games: int,
                 threshold: int = 2, source_verified: bool = False,
                 as_of_before_event: bool = False) -> Projection:
    """Player goal count via empirical Poisson; no market odds used."""
    if not source_verified or not as_of_before_event:
        return _unavailable("NHL", f"{threshold}+ player goals", "unverified_or_noncausal_source")
    if not _valid_rate(goals_per_game) or games < 20 or not 1 <= threshold <= 8:
        return _unavailable("NHL", f"{threshold}+ player goals", "insufficient_player_goal_history")
    p = _poisson_at_least(float(goals_per_game), threshold)
    return Projection("NHL", f"{threshold}+ player goals", p, "empirical_poisson_v1",
                      "UNVALIDATED_RESEARCH", ("poisson_assumption", "opponent_goalie_and_ice_time_unadjusted"), games)

def soccer_goals(*, home_scored: float | None, away_conceded: float | None,
                 away_scored: float | None, home_conceded: float | None,
                 home_games: int, away_games: int, threshold: int = 2,
                 source_verified: bool = False, as_of_before_event: bool = False) -> Projection:
    """Independent team-rate Poisson for total goals, with no market-derived inputs."""
    if not source_verified or not as_of_before_event:
        return _unavailable("Soccer", f"{threshold}+ match goals", "unverified_or_noncausal_source")
    rates = (home_scored, away_conceded, away_scored, home_conceded)
    if any(not _valid_rate(x) for x in rates) or min(home_games, away_games) < 10 or not 1 <= threshold <= 8:
        return _unavailable("Soccer", f"{threshold}+ match goals", "insufficient_team_goal_history")
    rate = (float(home_scored) + float(away_conceded) + float(away_scored) + float(home_conceded)) / 2
    return Projection("Soccer", f"{threshold}+ match goals", _poisson_at_least(rate, threshold),
                      "independent_goal_poisson_v1", "UNVALIDATED_RESEARCH",
                      ("independent_goals_assumption", "lineup_and_weather_unadjusted"), min(home_games, away_games))

def college_football_win(*, favorite_rating: float | None, underdog_rating: float | None,
                         favorite_games: int, underdog_games: int,
                         home_adjustment_points: float | None,
                         upset_red_flags: tuple[str, ...] | None,
                         source_verified: bool = False,
                         as_of_before_event: bool = False) -> Projection:
    """Rating-gap logistic prior. Mandatory vulnerability review before any promotion."""
    if not source_verified or not as_of_before_event:
        return _unavailable("NCAAF", "favorite wins", "unverified_or_noncausal_source")
    vals = (favorite_rating, underdog_rating, home_adjustment_points)
    if any(v is None or not isfinite(float(v)) for v in vals) or min(favorite_games, underdog_games) < 5:
        return _unavailable("NCAAF", "favorite wins", "insufficient_rating_history")
    if upset_red_flags is None:
        return _unavailable("NCAAF", "favorite wins", "upset_vulnerability_review_missing")
    gap = float(favorite_rating) - float(underdog_rating) + float(home_adjustment_points)
    p = 1. / (1. + exp(-max(-40., min(40., gap)) / 7.5))
    flags = tuple(upset_red_flags)
    return Projection("NCAAF", "favorite wins", p, "rating_gap_logit_v1",
                      "UNVALIDATED_RESEARCH",
                      ("rating_scale_requires_calibration", *flags), min(favorite_games, underdog_games))
