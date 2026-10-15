"""Chronological NBA 19+, 20+, 31+ final-margin historical population study.

Collect official-final ESPN games for 2024-25 and 2025-26, freeze a
2025-26 holdout, never use target-day outcomes/features or market prices.
Historical holdout is NOT prospective Kalshi edge evidence.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
import requests

from sports_edge.models.nba_margin_sleeper import nba_margin_tail

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
MONTHS = [(2024,m) for m in (10,11,12)] + [(2025,m) for m in (1,2,3,4,10,11,12)] + [(2026,m) for m in (1,2,3,4)]
HOLDOUT = "2025-10-01T00:00:00Z"

def _season_type(e):
    s=e.get("season") or e.get("seasonType") or {}
    return str(s.get("type")) if isinstance(s,dict) else str(s)

def parse_game(e):
    if _season_type(e)!="2":
        return None
    if (e.get("status",{}).get("type",{}).get("completed") is not True and
        e.get("competitions",[{}])[0].get("status",{}).get("type",{}).get("completed") is not True):
        return None
    c=e["competitions"][0]["competitors"]
    if len(c)!=2:return None
    home=next(x for x in c if x.get("homeAway")=="home")
    away=next(x for x in c if x.get("homeAway")=="away")
    def score(x):
        v=x["score"]
        raw=v.get("value",v.get("displayValue")) if isinstance(v,dict) else v
        return int(raw)
    home_score,away_score=score(home),score(away)
    if min(home_score,away_score)<0:return None
    return {"source":"ESPN","sport":"NBA","season_type":"regular",
            "event_id":str(e["id"]),"event_time":str(e["date"]),
            "home_id":str(home["id"]),"away_id":str(away["id"]),
            "home_score":home_score,"away_score":away_score}

def fetch_population():
    rows={};snapshots=[]
    session=requests.Session()
    for year,month in MONTHS:
        label=f"{year}{month:02d}"
        response=session.get(BASE,params={"dates":label,"limit":500},
                             timeout=35)
        response.raise_for_status()
        data=response.json()
        events=data.get("events")
        if not isinstance(events,list):
            raise ValueError(f"NBA ESPN monthly events absent: {label}")
        if not events and month in (1,2,3,11,12):
            raise ValueError(f"NBA full-season month inexplicably empty: {label}")
        snapshots.append({"month":label,"url":response.url,
                          "sha256":sha256(response.content).hexdigest(),
                          "source_events":len(events)})
        for e in events:
            try:g=parse_game(e)
            except (TypeError,ValueError,KeyError,StopIteration):
                raise ValueError(f"malformed completed NBA event in {label}: {e.get('id')}")
            if g is None:continue
            key=g["event_id"]
            if key in rows and rows[key]!=g:
                raise ValueError("conflicting official scores for same game")
            rows[key]=g
    past=sorted(rows.values(),key=lambda x:(x["event_time"],x["event_id"]))
    return past,snapshots

def brier(p,y):return (p-y)**2

def replay(games, *, line=18.5):
    if line not in (18.5,19.5,30.5):
        raise ValueError("line outside pre-registered NBA tail set")
    threshold=int(line)+1
    if not games:return {"status":"NO_DATA"}
    records=[];all_prior=[]
    for game in games:
        start=game["event_time"]
        if start<HOLDOUT:
            all_prior.append(game)
            continue
        # Allow only past games at least 24h old via model's own cutoff.
        prev=[x for x in all_prior if x["event_time"] < start]
        for selected,side in ((game["home_id"],"HOME"),(game["away_id"],"AWAY")):
            model=nba_margin_tail(prev,home_id=game["home_id"],away_id=game["away_id"],
                                  selected_id=selected,before_time=start,line=line)
            if model.probability is None:
                continue
            y=int((game["home_score"]-game["away_score"] if side=="HOME"
                   else game["away_score"]-game["home_score"])>=threshold)
            ref=[x for x in prev if x["event_time"] < start and
                 x["event_time"][:10]<start[:10]]
            p_ref=(2+sum((x["home_score"]-x["away_score"]>=threshold) if side=="HOME"
                         else (x["away_score"]-x["home_score"]>=threshold) for x in ref))/(len(ref)+16)
            records.append({"event_id":game["event_id"],"date":start[:10],"side":side,"threshold":threshold,
                            "p":model.probability,"baseline":p_ref,"won":y,
                            "model_mean_margin":model.mean,
                            "sensitivity_range":[model.lower,model.upper]})
        all_prior.append(game)
    def stats(rows):
        if not rows:return {"n":0}
        bs=sum(brier(r["p"],r["won"]) for r in rows)/len(rows)
        base=sum(brier(r["baseline"],r["won"]) for r in rows)/len(rows)
        return {"n":len(rows),"unique_games":len({r["event_id"] for r in rows}),
                "unique_dates":len({r["date"] for r in rows}),
                "realized_margin_rate":sum(r["won"] for r in rows)/len(rows),
                "model_brier":bs,"baseline_brier":base,
                "model_minus_baseline_brier":bs-base,
                "model_classification_accuracy":sum((r["p"]>=.5)==bool(r["won"]) for r in rows)/len(rows),
                "always_no_accuracy":1-sum(r["won"] for r in rows)/len(rows),
                "mean_predicted_margin":sum(r["p"] for r in rows)/len(rows)}
    # Preregistered bins prevent the misleading "87% accuracy" argument:
    # this rare tail requires reliability diagrams and positive-case support.
    bands=(0.,.05,.10,.20,.30,.40,.60,1.000001)
    calibration=[]
    for lo,hi in zip(bands,bands[1:]):
        subset=[r for r in records if lo <= r["p"] < hi]
        if not subset:
            calibration.append({"p_range":[lo,hi],"n":0})
            continue
        n=len(subset)
        wins=sum(r["won"] for r in subset)
        phat=wins/n
        z=1.96
        denominator=1+z*z/n
        center=(phat+z*z/(2*n))/denominator
        half=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/denominator
        calibration.append({"p_range":[lo,hi],"n":n,"wins":wins,
                            "observed_hit_rate":phat,
                            "mean_predicted":sum(r["p"] for r in subset)/n,
                            "wilson_95_interval":[max(0,center-half),min(1,center+half)]})
    ece=sum(r["n"]*abs(r["mean_predicted"]-r["observed_hit_rate"])
            for r in calibration if r["n"])/max(1,len(records))
    # Date-clustered deterministic bootstrap; one game/day cluster per unit.
    per_date=defaultdict(list)
    for r in records:per_date[r["date"]].append(r)
    dates=sorted(per_date)
    rng=Random(20261014)
    uplift=[]
    if dates:
        for _ in range(400):
            chosen=[r for d in (dates[rng.randrange(len(dates))] for _ in dates) for r in per_date[d]]
            uplift.append(sum(brier(r["baseline"],r["won"])-brier(r["p"],r["won"]) for r in chosen)/len(chosen))
        uplift.sort()
    report={"evaluation_kind":"retrospective_chronological_holdout",
            "model":"nba_full_game_margin_tail_v2",
            "margin_line":line,"required_margin":threshold,
            "holdout_start":HOLDOUT,
            "outcome":f"full_game_including_overtime_margin_at_least_{threshold}",
            "observations":len(records),
            "official_completed_games_fetched":len(games),
            "overall":stats(records),
            "home":stats([r for r in records if r["side"]=="HOME"]),
            "away":stats([r for r in records if r["side"]=="AWAY"]),
            "probability_band_reliability":calibration,
            "expected_calibration_error_by_fixed_bands":ece,
            "brier_advantage_date_bootstrap95":([uplift[10],uplift[389]] if len(uplift)==400 else None),
            "no_promotions":True,"qualified_for_live":False,
            "limitations":["ESPN scoreboard historical, not immutable future forecasts",
                           "Preseason and playoffs excluded",
                           "No Kalshi ask/fee archive or executed P&L",
                           "No player availability adjustments",
                           "Model vs market advantage not established"]}
    if len(dates)<30 or len(records)<200:
        report["status"]="INSUFFICIENT_RETROSPECTIVE_HOLDOUT"
    elif uplift[10]>0:
        report["status"]="HISTORICAL_BRIER_IMPROVEMENT_ONLY"
    else:
        report["status"]="NO_CONFIRMED_HISTORICAL_ADVANTAGE"
    return report,records

def main():
    rows,sources=fetch_population()
    out=Path("nba-margin-historical-evidence")
    out.mkdir(exist_ok=True)
    reports={}
    observations=[]
    for line in (18.5,19.5,30.5):
        report,records=replay(rows,line=line)
        reports[str(line)]=report
        observations.extend(records)
        print(json.dumps({"line":line,"status":report["status"],
                          "overall":report["overall"],
                          "bootstrap_95":report["brier_advantage_date_bootstrap95"],
                          "ece":report["expected_calibration_error_by_fixed_bands"]},sort_keys=True))
    (out/"metrics.json").write_text(json.dumps({"evaluation_kind":"historical_multi_threshold_holdout",
        "pre_registered_lines":[18.5,19.5,30.5],"reports":reports,
        "sources":sources,"qualified_for_live":False},indent=2))
    (out/"predictions.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in observations))
    (out/"completed_games.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows))
    if any(r["status"]=="INSUFFICIENT_RETROSPECTIVE_HOLDOUT" for r in reports.values()):
        raise SystemExit("Not enough honest independent holdout observations")

if __name__=="__main__":
    main()
