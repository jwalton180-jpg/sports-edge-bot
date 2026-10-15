"""Independent NBA regular-season high game-total YES/NO research.

Targets listed Over 239.5 through Over 259.5, including the screenshot-style
NO on Over 244.5. Exact Kalshi prices and results never enter the predictor.
"""
from __future__ import annotations
from datetime import timedelta
from math import erf,isfinite,sqrt
from statistics import pvariance
from sports_edge.models.sleeper_counts import CountEstimate,utc

MODEL="nba_high_total_yes_no_v1"
MIN_GAMES=24


def _tail(mean,standard_deviation,line):
    return max(0.,min(1.,.5*(1-erf((line-mean)/(sqrt(2)*standard_deviation)))))


def nba_high_total(games,*,home_id,away_id,before_time,line,season_type="regular"):
    def unavailable(reason,n=0):
        return CountEstimate(None,None,None,None,None,n,MODEL,(reason,),"RESEARCH_ONLY")
    try:
        cutoff=utc(before_time)
        if (season_type!="regular" or not home_id or not away_id or home_id==away_id
            or type(line) not in (int,float) or not isfinite(line)
            or line<239.5 or line>259.5 or line%1!=.5):
            return unavailable("unsupported_high_total_or_season_scope")
    except (TypeError,ValueError):
        return unavailable("invalid_event_time")
    unique={}
    for g in games:
        try:
            stamp=utc(g["event_time"])
            # ESPN game timestamp is start, not final. Preserve a 24h gap.
            if stamp+timedelta(hours=24)>=cutoff:
                continue
            if g.get("source")!="ESPN" or g.get("sport")!="NBA" or g.get("season_type")!="regular":
                continue
            eid=str(g["event_id"]); home=str(g["home_id"]);away=str(g["away_id"])
            hs=g["home_score"];aws=g["away_score"]
            if (not eid or home==away or type(hs) is not int or type(aws) is not int
                or min(hs,aws)<0 or max(hs,aws)>250):
                return unavailable("invalid_official_scores")
            record=(stamp.isoformat(),home,away,hs,aws)
            if eid in unique and unique[eid]!=record:
                return unavailable("conflicting_official_completed_game")
            unique[eid]=record
        except (TypeError,ValueError,KeyError,OverflowError):
            return unavailable("malformed_history_record")
    past=sorted(unique.values())
    def team_games(tid):
        return [(ts,hs if h==tid else aws,aws if h==tid else hs,hs+aws)
                for ts,h,a,hs,aws in past if tid in (h,a)][-82:]
    h,a=team_games(home_id),team_games(away_id)
    n=min(len(h),len(a))
    if n<MIN_GAMES:
        return unavailable("insufficient_completed_team_history",n)
    # Prespecified shrinkage avoids spurious extremes at 24-game coverage.
    def attack(rows):
        return (sum(row[1] for row in rows)+16*114)/(len(rows)+16)
    def allowed(rows):
        return (sum(row[2] for row in rows)+16*114)/(len(rows)+16)
    mu=0.5*(attack(h)+allowed(a)+attack(a)+allowed(h))
    mu=max(175.,min(290.,mu))
    totals=[r[3] for r in h+a]
    variance=max(19.**2,min(30.**2,0.5*pvariance(totals)+0.5*23.**2))
    sigma=sqrt(variance)
    p=_tail(mu,sigma,line)
    low=_tail(mu-7.,min(34.,sigma+3.),line)
    high=_tail(mu+7.,max(17.,sigma-3.),line)
    return CountEstimate(p,min(low,high),max(low,high),mu,variance,n,MODEL,
        ("regular_season_history_only","full_game_including_overtime",
         "normal_tail_and_fixed_shrinkage_assumptions",
         "pace_injuries_rest_and_rotations_unverified",
         "sensitivity_not_calibrated_confidence",
         "NO_side_equals_complement_of_over_probability",
         "prospective_price_vs_settlement_review_required"),
        "UNVALIDATED_RESEARCH",
        source_as_of=(max(utc(row[0]) for row in h+a)+timedelta(hours=24)).isoformat())
