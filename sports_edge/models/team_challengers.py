"""Independent pregame NCAA football and EPL challengers, research-only."""
from __future__ import annotations
from datetime import datetime,timedelta
from math import exp,factorial
from sports_edge.models.new_sport_research import Projection

def _time(value):
    x=datetime.fromisoformat(value.replace("Z","+00:00"))
    if x.tzinfo is None: raise ValueError("timezone required")
    return x

def ncaaf_opponent_challenger(games, *, home_id, away_id, before_time):
    cutoff=_time(before_time)-timedelta(hours=24)
    past=[g for g in games if g.get("source")=="ESPN" and g.get("sport")=="NCAAF"
          and _time(g["event_time"])<=cutoff]
    teams={t for g in past for t in (g["home_id"],g["away_id"])}
    played={t:[] for t in teams}
    for g in past:
        margin=g["home_score"]-g["away_score"]
        played[g["home_id"]].append((g["away_id"],margin-2.5))
        played[g["away_id"]].append((g["home_id"],-margin+2.5))
    sample=min(len(played.get(home_id,())),len(played.get(away_id,())))
    if sample<5:
        return Projection("NCAAF","home win",None,"ncaaf_opponent_v2","RESEARCH_ONLY",
                          ("insufficient_opponent_history",),sample)
    ratings={t:0. for t in teams}
    for _ in range(12):
        ratings={t:sum(max(-45.,min(45.,m+ratings[opp])) for opp,m in played[t])/(len(played[t])+5.)
                 for t in teams}
    gap=max(-40.,min(40.,ratings[home_id]-ratings[away_id]+2.5))
    return Projection("NCAAF","home win",1/(1+exp(-gap/9.)),
                      "ncaaf_opponent_v2","UNVALIDATED_RESEARCH",
                      ("injury_weather_and_upset_review_missing",),sample)

def soccer_shrink_challenger(games, *, home_id, away_id, before_time, threshold=2):
    if not 1<=threshold<=6: raise ValueError("threshold")
    cutoff=_time(before_time)-timedelta(hours=24)
    past=[g for g in games if g.get("source")=="ESPN" and g.get("sport")=="Soccer"
          and _time(g["event_time"])<=cutoff]
    home=[g for g in past if home_id in (g["home_id"],g["away_id"])][-20:]
    away=[g for g in past if away_id in (g["home_id"],g["away_id"])][-20:]
    sample=min(len(home),len(away))
    if sample<10:
        return Projection("Soccer",f"{threshold}+ match goals",None,"soccer_shrink_v2",
                          "RESEARCH_ONLY",("insufficient_team_history",),sample)
    def rate(rows,team,for_goals,prior):
        vals=[g["home_score" if (g["home_id"]==team)==for_goals else "away_score"] for g in rows]
        return (sum(vals)+8*prior)/(len(vals)+8)
    attack_home=rate(home,home_id,True,1.5)
    defense_home=rate(home,home_id,False,1.25)
    attack_away=rate(away,away_id,True,1.25)
    defense_away=rate(away,away_id,False,1.5)
    total=max(.1,min(7.,(attack_home+defense_away+attack_away+defense_home)/2))
    p=1-exp(-total)*sum(total**k/factorial(k) for k in range(threshold))
    return Projection("Soccer",f"{threshold}+ match goals",max(0.,min(1.,p)),
                      "soccer_shrink_v2","UNVALIDATED_RESEARCH",
                      ("EPL_only","lineups_unadjusted"),sample)
