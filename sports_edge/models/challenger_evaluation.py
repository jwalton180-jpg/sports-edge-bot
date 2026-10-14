"""Evaluate challengers against matching legacy model rows without promoting them."""
from __future__ import annotations
from collections import defaultdict
from sports_edge.models.sport_walkforward import calibration_report,_date
from sports_edge.models.sport_challengers import nhl_shot_challenger
from sports_edge.models.team_challengers import ncaaf_opponent_challenger,soccer_shrink_challenger

def challenger_team_rows(games,sport):
    ordered=sorted(games,key=lambda x:(x["event_time"],x["event_id"]))
    out=[]
    for game in ordered:
        past=[g for g in ordered if _date(g["event_time"])<_date(game["event_time"])]
        if sport=="NCAAF":
            proj=ncaaf_opponent_challenger(past,home_id=game["home_id"],
                                           away_id=game["away_id"],before_time=game["event_time"])
            won=game["home_score"]>game["away_score"]
            if game["home_score"]==game["away_score"]: continue
            outcome="home_win"
        else:
            proj=soccer_shrink_challenger(past,home_id=game["home_id"],
                                          away_id=game["away_id"],before_time=game["event_time"])
            won=game["home_score"]+game["away_score"]>=2
            outcome="total_goals_2_plus"
        if proj.probability is None: continue
        out.append({"sport":sport,"event_id":game["event_id"],"event_time":game["event_time"],
                    "predicted_at":past[-1]["event_time"],"source_id":"ESPN:"+game["event_id"],
                    "model_version":proj.model,"outcome":outcome,"probability":proj.probability,
                    "won":won,"evaluation_kind":"historical_walkforward","status":"RESEARCH_ONLY"})
    return out

def challenger_nhl_rows(log):
    ordered=sorted(log,key=lambda r:(r["date"],r["game_id"]))
    out=[]
    for game in ordered:
        past=[g for g in ordered if g["date"]<game["date"]]
        proj=nhl_shot_challenger(past)
        if proj.probability is None: continue
        out.append({"sport":"NHL","event_id":str(game["game_id"])+":"+str(game["player_id"]),
                    "event_time":game["date"]+"T12:00:00Z",
                    "predicted_at":past[-1]["date"]+"T23:59:59Z",
                    "source_id":"NHL:"+str(game["game_id"]),"model_version":proj.model,
                    "outcome":"player_goals_2_plus","probability":proj.probability,
                    "won":game["goals"]>=2,"evaluation_kind":"historical_walkforward",
                    "status":"RESEARCH_ONLY"})
    return out

def matched_holdout(legacy,challenger,*,fraction=.25):
    """Evaluate on same events, with temporal split and no after-the-fact tuning."""
    a={(r["sport"],r["event_id"],r["outcome"]):r for r in legacy}
    b={(r["sport"],r["event_id"],r["outcome"]):r for r in challenger}
    common=sorted(set(a)&set(b),key=lambda k:(a[k]["event_time"],k[1]))
    if len(common)<200:
        return {"status":"INSUFFICIENT_MATCHED_EVENTS","n":len(common),"qualified_for_live":False}
    cut=int(len(common)*(1-fraction))
    while cut>0 and a[common[cut-1]]["event_time"]==a[common[cut]]["event_time"]:
        cut-=1
    keys=common[cut:]
    old=[a[k] for k in keys]
    new=[b[k] for k in keys]
    sport=old[0]["sport"]
    old_metrics=calibration_report(old)[sport]
    new_metrics=calibration_report(new)[sport]
    return {"status":"HISTORICAL_MATCHED_HOLDOUT","qualified_for_live":False,
            "matched_total":len(common),"holdout_n":len(keys),
            "legacy":old_metrics,"challenger":new_metrics,
            "brier_delta_challenger_minus_legacy":new_metrics["brier"]-old_metrics["brier"],
            "evidence":"retrospective_only_no_prospective_promotion"}
