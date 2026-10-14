"""NHL player prop probability research for distinct verified market families.

A projected win probability is NOT a price edge. All outputs are research-only
until official holdout calibration, prospectively frozen calls and executable
Kalshi asks/fees are verified. Missing stat history always fails closed.
"""
from __future__ import annotations
from math import exp,factorial,isfinite
from sports_edge.models.new_sport_research import Projection

MARKETS={
    "goals":("goals",0.12,30.0,1,5),
    "shots":("shots",2.5,12.0,1,12),
    "assists":("assists",0.35,25.0,1,6),
    "points":("points",0.55,25.0,1,7),
    "power_play_goals":("power_play_goals",0.025,60.0,1,3),
}
def _poisson_at_least(rate,threshold):
    return max(0.,min(1.,1.-exp(-rate)*sum(rate**k/factorial(k) for k in range(threshold))))

def nhl_player_market(prior:list[dict],*,market:str,threshold:int=1)->Projection:
    if market not in MARKETS: raise ValueError("unsupported NHL market")
    field,prior_rate,shrink,low,high=MARKETS[market]
    label=f"{threshold}+ player {market}"
    model=f"nhl_{market}_volume_v1"
    if not low<=threshold<=high or len(prior)<20:
        return Projection("NHL",label,None,model,"RESEARCH_ONLY",("insufficient_history",),len(prior))
    values=[]
    for r in prior:
        v=r.get(field)
        if r.get("source")!="NHL" or type(v) is not int or not 0<=v<=25:
            return Projection("NHL",label,None,model,"RESEARCH_ONLY",
                              ("unverified_or_missing_stat",),len(prior))
        if r.get("goals") is not None and r.get("assists") is not None and r.get("points") is not None:
            if r["points"]!=r["goals"]+r["assists"]:
                return Projection("NHL",label,None,model,"RESEARCH_ONLY",
                                  ("inconsistent_official_stats",),len(prior))
        values.append(v)
    # Fixed predeclared empirical-Bayes priors, not tuned on test outcomes.
    # 10-game form gets 60% weight; full season gets 40%.
    full=(sum(values)+shrink*prior_rate)/(len(values)+shrink)
    recent=(sum(values[-10:])+shrink*prior_rate)/(min(10,len(values))+shrink)
    rate=max(0.,min(20.,.6*recent+.4*full))
    return Projection("NHL",label,_poisson_at_least(rate,threshold),
                      model,"UNVALIDATED_RESEARCH",
                      ("poisson_dispersion_unverified","ice_time_and_opponent_unadjusted"),len(prior))

def nhl_market_menu(prior:list[dict])->tuple[Projection,...]:
    """Multiple independently estimated outcomes; no artificial sleeper designation."""
    return tuple(nhl_player_market(prior,market=market,threshold=threshold)
                 for market,thresholds in {
                     "goals":(1,2),"shots":(2,3,4,5),"assists":(1,2),
                     "points":(1,2,3),"power_play_goals":(1,)
                 }.items() for threshold in thresholds)
