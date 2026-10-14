"""NBA full-game alternate spread: team wins by >18.5 (at least 19).
Research-only. Prices do not enter the predictor; no confidence is certified.
"""
from __future__ import annotations

from datetime import timedelta
from math import erf, isfinite, sqrt
from statistics import mean, pvariance
from sports_edge.models.sleeper_counts import CountEstimate, utc

MODEL = "nba_full_game_margin_19plus_v1"
MIN_TEAM_GAMES = 24


def _normal_tail(mu: float, sigma: float, line: float) -> float:
    return max(0., min(1., .5 * (1 - erf((line - mu) / (sqrt(2) * sigma)))))


def nba_margin_tail(games, *, home_id: str, away_id: str, selected_id: str,
                    before_time: str, line: float = 18.5,
                    season_type: str = "regular") -> CountEstimate:
    """Deterministic causal model of final margin including OT.
    
    Each completed matchup counts ONCE, whatever team schedules provided it.
    Historical postseason and exhibition games never train a regular-season lane.
    Missing, duplicate or conflicting scores fail closed, not imputed as zeros.
    """
    def rejected(reason, n=0):
        return CountEstimate(None, None, None, None, None, n, MODEL,
                             (reason,), "RESEARCH_ONLY")
    try:
        cutoff = utc(before_time)
        if season_type != "regular" or home_id == away_id or selected_id not in (home_id, away_id):
            return rejected("unsupported_season_or_team_identity")
        if not isinstance(line, (float, int)) or not isfinite(line) or line < 0 or line % 1 != .5:
            return rejected("not_exact_half_point_full_game_spread")
    except (TypeError, ValueError):
        return rejected("invalid_start_or_spread")
    unique = {}
    for g in games:
        try:
            stamp = utc(g["event_time"])
            # ESPN schedule timestamps mark game start, not finish. A full-day
            # gap excludes unfinished and target-day results from input.
            if stamp + timedelta(hours=24) >= cutoff:
                continue
            if g.get("source") != "ESPN" or g.get("sport") != "NBA" or g.get("season_type") != "regular":
                continue
            game_id = str(g["event_id"])
            h, a = str(g["home_id"]), str(g["away_id"])
            hs, aws = g["home_score"], g["away_score"]
            if (not game_id or not h or not a or h == a or
                type(hs) is not int or type(aws) is not int or
                not 0 <= hs <= 250 or not 0 <= aws <= 250):
                return rejected("invalid_official_game_data")
            record = (stamp.isoformat(), h, a, hs, aws)
            if game_id in unique and unique[game_id] != record:
                return rejected("conflicting_duplicate_official_game")
            unique[game_id] = record
        except (KeyError, ValueError, TypeError, OverflowError):
            return rejected("malformed_official_game")
    past = sorted(unique.values())
    def signed(rows, team):
        return [(ts, (hs-aws) if h == team else (aws-hs), h == team)
                for ts,h,a,hs,aws in rows if team in (h,a)][-82:]
    hg, ag = signed(past, home_id), signed(past, away_id)
    n = min(len(hg), len(ag))
    if n < MIN_TEAM_GAMES:
        return rejected("insufficient_completed_regular_season_history", n)
    def rating(g):
        recent = [m for _,m,_ in g[-10:]]
        long = [m for _,m,_ in g]
        # Shrink toward neutral; recent form is subordinate to full sample.
        return .75 * sum(long)/(len(long)+16) + .25*sum(recent)/(len(recent)+16)
    home_rating, away_rating = rating(hg), rating(ag)
    # Ratings contain both venues; apply a fixed prespecified home advantage.
    mu_home = max(-28., min(28., home_rating-away_rating+2.0))
    mu = mu_home if selected_id == home_id else -mu_home
    # Team-specific variance shrunk to broad league spread; prevents apparent
    # 90% confidence when samples happen to show small game-score variance.
    all_margins = [m for _,m,_ in hg+ag]
    variance = max(13.5**2, min(21.0**2, .5*pvariance(all_margins)+.5*14.5**2))
    sd = sqrt(variance)
    probability = _normal_tail(mu, sd, line)
    low = _normal_tail(mu-4., min(23., sd+2.), line)
    high = _normal_tail(mu+4., max(12., sd-2.), line)
    if low > high:
        low, high = high, low
    # Source timestamps are game STARTS; buffer 24h for past-game finality.
    last_time = max(utc(g[0]) for g in hg+ag)
    return CountEstimate(probability, low, high, mu, variance, n,
                         MODEL,
                         ("pregame_final_margin_only","regular_season_history_only",
                          "fixed_home_adjustment_not_injury_adjusted",
                          "score_margin_gaussian_tail_unvalidated",
                          "sensitivity_interval_not_calibrated_confidence",
                          "prospective_model_vs_executable_ask_validation_required"),
                         "UNVALIDATED_RESEARCH",
                         source_as_of=(last_time+timedelta(hours=24)).isoformat())
