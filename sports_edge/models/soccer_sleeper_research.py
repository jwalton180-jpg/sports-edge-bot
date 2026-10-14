"""Independent EPL research probabilities for team goals, totals and BTTS.

Historical-only inputs. These are not live recommendations or verified value.
"""
from __future__ import annotations
from datetime import datetime,timedelta
from math import exp,factorial
from sports_edge.models.new_sport_research import Projection

def _utc(value):
    dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if dt.tzinfo is None: raise ValueError("timezone required")
    return dt

def _poisson_at_least(rate,threshold):
    return max(0.,min(1.,1-exp(-rate)*sum(rate**k/factorial(k) for k in range(threshold))))

def soccer_market_projection(games,*,home_id,away_id,before_time,market,threshold=None):
    """No result from target day or later can affect a prediction."""
    allowed={"total_goals","home_goals","away_goals","both_teams_score"}
    if market not in allowed: raise ValueError("unsupported market")
    if market=="both_teams_score":
        if threshold is not None: raise ValueError("BTTS has no threshold")
    elif type(threshold) is not int or not 1<=threshold<=7:
        raise ValueError("invalid goal threshold")
    model="epl_team_goal_components_v1"
    outcome="both teams score" if market=="both_teams_score" else f"{threshold}+ {market}"
    cutoff=_utc(before_time)-timedelta(hours=24)
    prior=[]
    for g in games:
        if g.get("sport")!="Soccer" or g.get("source")!="ESPN":
            continue
        when=_utc(g["event_time"])
        if when>cutoff: continue
        if not all(type(g.get(k)) is int and g[k]>=0 for k in ("home_score","away_score")):
            continue
        if not g.get("home_id") or not g.get("away_id") or g["home_id"]==g["away_id"]:
            continue
        prior.append(g)
    prior.sort(key=lambda g:(g["event_time"],g.get("event_id","")))
    h=[g for g in prior if home_id in (g["home_id"],g["away_id"])][-20:]
    a=[g for g in prior if away_id in (g["home_id"],g["away_id"])][-20:]
    sample=min(len(h),len(a))
    if home_id==away_id or sample<10:
        return Projection("Soccer",outcome,None,model,"RESEARCH_ONLY",
                          ("insufficient_verified_team_history",),sample)
    def goals(rows,team,scored):
        total=0
        for g in rows:
            home=g["home_id"]==team
            total+=g["home_score" if home==scored else "away_score"]
        return total
    # Fixed shrinkage avoids tuning to held-out events.
    h_att=(goals(h,home_id,True)+8*1.5)/(len(h)+8)
    h_def=(goals(h,home_id,False)+8*1.25)/(len(h)+8)
    a_att=(goals(a,away_id,True)+8*1.25)/(len(a)+8)
    a_def=(goals(a,away_id,False)+8*1.5)/(len(a)+8)
    lam_home=max(.05,min(5.,(h_att+a_def)/2))
    lam_away=max(.05,min(5.,(a_att+h_def)/2))
    if market=="both_teams_score":
        p=(1-exp(-lam_home))*(1-exp(-lam_away))
    else:
        rate={"total_goals":lam_home+lam_away,
              "home_goals":lam_home,"away_goals":lam_away}[market]
        p=_poisson_at_least(rate,threshold)
    return Projection("Soccer",outcome,p,model,"UNVALIDATED_RESEARCH",
                      ("EPL_scope_only","independent_poisson_assumption",
                       "lineups_and_weather_unverified"),sample)
