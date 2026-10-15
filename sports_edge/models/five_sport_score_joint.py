"""Causal paired-score Monte Carlo for NFL/NCAAF/NBA/NHL/MLB research.

Whole-game score pairs preserve observed shared-score-game effects.
No player stats, bookmaker odds or Kalshi quotes are consumed. Outputs are
hypotheses, not validated or settlement-rule equivalent Kalshi prices.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from math import isfinite
from random import Random
import json

from sports_edge.data.five_sport_score_archive import PATHS, utc, validate_archive

MODEL="paired_completed_score_residual_v1"
LINES={"NFL":44.5,"NCAAF":51.5,"NBA":224.5,"NHL":5.5,"MLB":8.5}
MARKETS=("home_wins","away_wins","total_over","home_total_over","away_total_over")
MIN_LEAGUE=50
MIN_TEAM=4
MAX_GAME_HISTORY=250
MIN_DRAWS=500
MAX_DRAWS=5000


def _value(v,kind):
    if isinstance(v,bool):raise ValueError(kind+"_boolean_not_allowed")
    try:x=float(v)
    except (ValueError,TypeError):raise ValueError("invalid_"+kind) from None
    if not isfinite(x):raise ValueError("nonfinite_"+kind)
    return x


def _line(v):
    x=_value(v,"line")
    if not 0<x<=300 or abs((x*2)%2-1)>1e-8:
        raise ValueError("only_positive_half_point_binary_lines_supported")
    return x


def _target(target):
    if not isinstance(target,dict) or target.get("sport") not in PATHS:
        raise ValueError("unsupported_sport_or_game")
    required=("event_id","home_id","away_id","event_time")
    if any(not isinstance(target.get(x),str) or not target[x] for x in required):
        raise ValueError("event_identity_incomplete")
    if target["home_id"]==target["away_id"]:raise ValueError("identical_teams")
    start=utc(target["event_time"])
    return start


def score_outcome(h, a, market, line=None):
    if type(h) is not int or type(a) is not int or min(h,a)<0:
        raise ValueError("invalid_final_score")
    if market=="home_wins":return h>a
    if market=="away_wins":return a>h
    if market not in MARKETS or line is None:
        raise ValueError("unsupported_market_or_missing_line")
    threshold=_line(line)
    value=h+a if market=="total_over" else h if market=="home_total_over" else a
    return value>threshold


def complete_history(games, *, target, before):
    """Strict game/date cutoff prevents using completed same-day later scores."""
    t=_target(target)
    clock=utc(before)
    if clock>=t:raise ValueError("simulation_clock_after_kickoff")
    selected=[]
    for row in games:
        if row["sport"]!=target["sport"]:continue
        event=utc(row["event_time"])
        if event>=clock or event.date()>=clock.date() or event>=t:
            continue
        selected.append(row)
    return selected


def simulate_paired_scores(games, *, target, as_of, draws=1000,
                           max_history_age_days=None):
    """Sample past home/away *residual pairs* after causal team-level adjustment.

    The method is intentionally an uncalibrated research baseline. Paired
    residual bootstrap captures shared pace/game-script without assuming
    independence, not a validated player-prop correlation model.
    """
    if type(draws) is not int or not MIN_DRAWS<=draws<=MAX_DRAWS:
        raise ValueError("bounded_scenario_count_required")
    target_time=_target(target)
    asof=utc(as_of)
    if asof>=target_time:raise ValueError("prediction_not_pregame")
    history=complete_history(games,target=target,before=as_of)
    if max_history_age_days is not None:
        if (type(max_history_age_days) is not int or
            not 1<=max_history_age_days<=365):
            raise ValueError("bounded_age_days_required")
        if (not history or
            (asof-max(utc(row["event_time"]) for row in history)).total_seconds() >
            max_history_age_days*86400):
            return {"status":"STALE_SPORT_HISTORY_FOR_FUTURE_PREDICTION",
                    "qualified_for_live":False,"scores":[]}
    if len(history)<MIN_LEAGUE:
        return {"status":"INSUFFICIENT_LEAGUE_HISTORY","count":len(history),
                "qualified_for_live":False,"scores":[]}
    home_id,away_id=target["home_id"],target["away_id"]
    def recent(team):
        return [g for g in history if team in (g["home_id"],g["away_id"])][-12:]
    home_hist,away_hist=recent(home_id),recent(away_id)
    if min(len(home_hist),len(away_hist))<MIN_TEAM:
        return {"status":"INSUFFICIENT_TEAM_HISTORY",
                "home_n":len(home_hist),"away_n":len(away_hist),
                "qualified_for_live":False,"scores":[]}
    def scored(g,team):
        return (g["home_score"],g["away_score"]) if g["home_id"]==team else (g["away_score"],g["home_score"])
    h_off=sum(scored(g,home_id)[0] for g in home_hist)/len(home_hist)
    a_def=sum(scored(g,away_id)[1] for g in away_hist)/len(away_hist)
    a_off=sum(scored(g,away_id)[0] for g in away_hist)/len(away_hist)
    h_def=sum(scored(g,home_id)[1] for g in home_hist)/len(home_hist)
    mu_h=(h_off+a_def)/2
    mu_a=(a_off+h_def)/2
    paired=history[-MAX_GAME_HISTORY:]
    base_h=sum(g["home_score"] for g in paired)/len(paired)
    base_a=sum(g["away_score"] for g in paired)/len(paired)
    residuals=[(g["home_score"]-base_h,g["away_score"]-base_a) for g in paired]
    digest=sha256(json.dumps({"model":MODEL,"target":target,"as_of":asof.isoformat(),
                              "history_ids":[(r["sport"],r["event_id"]) for r in history],
                              "draws":draws},sort_keys=True).encode()).hexdigest()
    rng=Random(int(digest[:16],16))
    scores=[]
    for _ in range(draws):
        rh,ra=residuals[rng.randrange(len(residuals))]
        scores.append((max(0,round(mu_h+rh)),max(0,round(mu_a+ra))))
    return {"status":"UNVALIDATED_HISTORICAL_SCORE_SCENARIOS",
            "sport":target["sport"],"target_event_id":target["event_id"],
            "as_of_utc":asof.isoformat(),"model_version":MODEL,
            "league_n":len(history),"home_team_n":len(home_hist),
            "away_team_n":len(away_hist),"scores":scores,
            "scenario_sha256":sha256(json.dumps(scores,separators=(",",":")).encode()).hexdigest(),
            "qualified_for_live":False,"player_props_modeled":False,
            "kalshi_market_rules_verified":False}


def joint_score_market_research(scenarios, *, selections):
    if scenarios.get("status")!="UNVALIDATED_HISTORICAL_SCORE_SCENARIOS":
        return {"status":"BLOCKED_INSUFFICIENT_SCORE_EVIDENCE","joint_probability":None,
                "qualified_for_live":False}
    specs=list(selections)
    if not 2<=len(specs)<=10:raise ValueError("two_to_ten_score_markets_required")
    selections_seen=set()
    for spec in specs:
        if not isinstance(spec,dict) or spec.get("market") not in MARKETS:
            raise ValueError("unsupported_exact_score_outcome")
        market=spec["market"]
        line=spec.get("line")
        if market not in ("home_wins","away_wins"):_line(line)
        if (market,line) in selections_seen:
            raise ValueError("duplicate_leg")
        selections_seen.add((market,line))
    outcomes=[[score_outcome(int(h),int(a),spec["market"],spec.get("line"))
               for spec in specs] for h,a in scenarios["scores"]]
    marginal=[sum(r[i] for r in outcomes)/len(outcomes) for i in range(len(specs))]
    joint=sum(all(r) for r in outcomes)/len(outcomes)
    product=1.
    for p in marginal:product*=p
    return {"status":"RESEARCH_ONLY_UNCALIBRATED",
            "joint_probability":joint,"marginal_probabilities":marginal,
            "independence_benchmark_not_a_forecast":product,
            "selection_count":len(specs),
            "scenario_sha256":scenarios["scenario_sha256"],
            "scenario_count":len(outcomes),
            "qualified_for_live":False,"kalshi_combo_edge":None,
            "cashout_probability":None}


def historical_score_holdout(games, *, sport, holdout_start, line=None,
                             min_train=60, min_test=30):
    """Chronological, fully out-of-sample score-combo Brier versus priors.

    One score pair yields the conjunction home_wins AND total_over.
    Retrospective, not prospective; never permits promotion.
    """
    if sport not in PATHS:raise ValueError("unsupported_sport")
    threshold=_line(LINES[sport] if line is None else line)
    cutoff=utc(holdout_start)
    ordered=sorted((r for r in games if r["sport"]==sport),
                   key=lambda r:(r["event_time"],r["event_id"]))
    if len({r["event_id"] for r in ordered})!=len(ordered):
        raise ValueError("duplicate_holdout_game")
    score_rows=[]
    for game in ordered:
        t=utc(game["event_time"])
        if t<cutoff:continue
        # Freeze at PREVIOUS UTC midnight. Some real sports kickoffs are
        # exactly at 00:00 UTC, so same-day midnight is *not* pregame.
        # The extra full-day embargo also excludes same-day late final scores.
        observed=t.replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(days=1)
        prior=[r for r in ordered if utc(r["event_time"]).date()<t.date()]
        if len(prior)<max(MIN_LEAGUE,min_train):continue
        scen=simulate_paired_scores(ordered,target=game,as_of=observed.isoformat(),
                                    draws=1000)
        if scen["status"]!="UNVALIDATED_HISTORICAL_SCORE_SCENARIOS":
            continue
        p=joint_score_market_research(scen,selections=[
            {"market":"home_wins"},
            {"market":"total_over","line":threshold}])
        y=int(score_outcome(game["home_score"],game["away_score"],"home_wins") and
              score_outcome(game["home_score"],game["away_score"],"total_over",threshold))
        # Honest non-model prior from earlier FINAL games only.
        home_base=sum(g["home_score"]>g["away_score"] for g in prior)/len(prior)
        total_base=sum(g["home_score"]+g["away_score"]>threshold for g in prior)/len(prior)
        score_rows.append({"sport":sport,"event_id":game["event_id"],
                           "event_time":t.isoformat(),"predicted_at":observed.isoformat(),
                           "p":p["joint_probability"],"independence_model":p["independence_benchmark_not_a_forecast"],
                           "prior_baseline":home_base*total_base,
                           "won":y,"scenario_hash":p["scenario_sha256"]})
    if len(score_rows)<min_test:
        return {"status":"INSUFFICIENT_RETROSPECTIVE_HOLDOUT","sport":sport,
                "n":len(score_rows),"required_test":min_test,
                "qualified_for_live":False},score_rows
    def brier(key):
        return sum((r[key]-r["won"])**2 for r in score_rows)/len(score_rows)
    def calib():
        bins=[]
        for lo,hi in zip((0,.2,.4,.6,.8),( .2,.4,.6,.8,1.000001)):
            sub=[r for r in score_rows if lo<=r["p"]<hi]
            bins.append({"from":lo,"to":hi,"n":len(sub),
                         "mean_p":sum(x["p"] for x in sub)/len(sub) if sub else None,
                         "win_rate":sum(x["won"] for x in sub)/len(sub) if sub else None})
        return bins
    return {"status":"HISTORICAL_HOLDOUT_RESEARCH_ONLY",
            "sport":sport,"line":threshold,
            "n":len(score_rows),"unique_dates":len({r["event_time"][:10] for r in score_rows}),
            "joint_brier":brier("p"),
            "independence_model_brier":brier("independence_model"),
            "league_prior_brier":brier("prior_baseline"),
            "joint_minus_independence_brier":brier("p")-brier("independence_model"),
            "joint_minus_league_prior_brier":brier("p")-brier("prior_baseline"),
            "calibration_bins":calib(),
            "qualified_for_live":False,"holdout_start":cutoff.isoformat(),
            "limitations":["Score outcomes only; no player stats",
                           "ESPN full-time scores not Kalshi contract resolution",
                           "Not a prospectively immutable prediction sample",
                           "No historical executable combo asks/fees or exits"]},score_rows
