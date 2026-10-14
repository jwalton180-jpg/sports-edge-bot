"""NBA 19+ point victory-margin tails from independent completed game margins.

A pregame-only research challenger. It never reads prices or enables promotion.
Half-point spreads avoid pushes: 'over 18.5' means margin >= 19 points.
"""
from __future__ import annotations
from datetime import timedelta
from math import erfc, isfinite, sqrt
from statistics import mean, pvariance
from sports_edge.models.new_sport_research import Projection
from sports_edge.models.sleeper_counts import utc

VERSION = "nba_full_game_blowout_margin_v1"


def nba_blowout_tail(games, *, home_id, away_id, selected_id, threshold,
                     before_time, season_type):
    outcome = f"win by {threshold}+ points"
    def reject(reason, n=0):
        return Projection("NBA", outcome, None, VERSION, "RESEARCH_ONLY", (reason,), n)

    if season_type not in {"regular", "playoffs"}:
        return reject("preseason_or_unknown_game_type_requires_separate_validation")
    if type(threshold) is not int or not 19 <= threshold <= 60:
        return reject("unsupported_blowout_threshold")
    if not home_id or not away_id or home_id == away_id or selected_id not in {home_id, away_id}:
        return reject("selected_team_or_matchup_unverified")
    try:
        cutoff = utc(before_time) - timedelta(hours=24)
    except (ValueError, TypeError):
        return reject("verified_start_missing")

    history = {}
    for game in games:
        try:
            if game.get("source") != "ESPN" or game.get("sport") != "NBA":
                return reject("unverified_source_or_sport")
            start = utc(game["event_time"])
            if start > cutoff:  # Cannot use the event or the immediately preceding day.
                continue
            if game.get("season_type") not in {"regular", "playoffs"}:
                return reject("unverified_historical_season_type")
            gid = str(game["event_id"])
            h, a = str(game["home_id"]), str(game["away_id"])
            hs, aws = game["home_score"], game["away_score"]
            if not gid or not h or not a or h == a or type(hs) is not int or type(aws) is not int:
                return reject("unverified_completed_game_or_duplicate")
            if not (0 <= hs <= 250 and 0 <= aws <= 250):
                return reject("implausible_final_score")
            if gid in history:
                if history[gid] != game:
                    return reject("conflicting_historical_game")
                continue
            history[gid] = game
        except (ValueError, TypeError, KeyError):
            return reject("historical_identity_or_timestamp_missing")

    rows = sorted(history.values(), key=lambda g:g["event_time"])
    margins = {home_id: [], away_id: []}
    residuals = []
    for game in rows:
        h, a = str(game["home_id"]), str(game["away_id"])
        margin = game["home_score"] - game["away_score"]
        if h in margins:
            margins[h].append((utc(game["event_time"]), margin - 2.0))
            residuals.append(margin - 2.0)
        if a in margins:
            margins[a].append((utc(game["event_time"]), -margin + 2.0))
            residuals.append(-margin + 2.0)
    nh, na = len(margins[home_id]), len(margins[away_id])
    n = min(nh, na)
    if n < 12:
        return reject("insufficient_completed_NBA_team_game_history", n)

    def rating(samples):
        xs = [v for _, v in samples[-45:]]
        recent = [v for _, v in samples[-10:]]
        # Shrink early-season and anomalous margins toward zero. Never use market spreads.
        return .75 * sum(xs)/(len(xs)+16) + .25 * sum(recent)/(len(recent)+12)

    expected_home = max(-35., min(35., rating(margins[home_id]) - rating(margins[away_id]) + 2.0))
    expected_selected = expected_home if selected_id == home_id else -expected_home
    # NBA margin variance is material. Do not assign a 90%+ blowout chance
    # based only on a large favorite's mean projected margin.
    centered = [v - mean(residuals) for v in residuals]
    empirical_var = pvariance(centered) if len(centered) >= 20 else 14.0**2
    sigma = sqrt((len(centered) * empirical_var + 40 * 14.0**2) / (len(centered)+40))
    sigma = max(11.0, min(22.0, sigma))
    # Continuity correction for integer-valued final margin.
    p = .5 * erfc(((threshold - .5) - expected_selected)/(sigma*sqrt(2)))
    if not isfinite(p):
        return reject("invalid_margin_probability", n)
    return Projection("NBA", outcome, max(0., min(1., p)), VERSION,
                      "UNVALIDATED_RESEARCH",
                      ("full_game_including_overtime", "independent_ESPN_final_margins",
                       "shrunk_home_away_strength", "regular_season_and_playoff_history",
                       "rotations_injury_rest_and_garbage_time_not_confirmed",
                       "unvalidated_distribution_and_calibration"), n)
