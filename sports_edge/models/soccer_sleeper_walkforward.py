"""Retrospective strictly-prior soccer market evaluation, research-only."""
from __future__ import annotations
from sports_edge.models.soccer_sleeper_research import soccer_market_projection

def soccer_sleeper_walkforward(games):
    ordered=sorted(games,key=lambda g:(g["event_time"],g["event_id"]))
    out=[]
    for game in ordered:
        past=[g for g in ordered if g["event_time"]<game["event_time"]]
        for market,threshold in (("both_teams_score",None),("total_goals",2),
                                 ("total_goals",3),("total_goals",4),
                                 ("home_goals",1),("home_goals",2),
                                 ("away_goals",1),("away_goals",2)):
            proj=soccer_market_projection(past,home_id=game["home_id"],
                    away_id=game["away_id"],before_time=game["event_time"],
                    market=market,threshold=threshold)
            if proj.probability is None: continue
            hg,ag=game["home_score"],game["away_score"]
            won=((hg>0 and ag>0) if market=="both_teams_score" else
                 (hg+ag>=threshold if market=="total_goals" else
                  hg>=threshold if market=="home_goals" else ag>=threshold))
            name=market if threshold is None else f"{market}_{threshold}_plus"
            out.append({"sport":"Soccer","event_id":game["event_id"],
                        "event_time":game["event_time"],
                        "predicted_at":past[-1]["event_time"],
                        "source_id":"ESPN:"+game["event_id"],
                        "model_version":proj.model,"outcome":name,
                        "probability":proj.probability,"won":won,
                        "evaluation_kind":"historical_walkforward",
                        "status":"RESEARCH_ONLY"})
    return out
