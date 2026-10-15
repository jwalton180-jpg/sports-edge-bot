"""Official completed NBA regular-season high-total chronological study.

The same 2024-25 training and 2025-26 holdout dates as the 19/20/31+
margin study are reused. Never use sportsbook odds or final prices as labels.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from random import Random

from scripts.run_nba_margin_replay import fetch_population,HOLDOUT
from sports_edge.models.nba_high_total import nba_high_total

LINES=(239.5,244.5,249.5)


def replay(games,*,line=244.5):
    if line not in LINES:raise ValueError("not preregistered high-total line")
    cutoff=int(line)+1
    ordered=sorted(games,key=lambda g:(g["event_time"],g["event_id"]))
    before=[];rows=[]
    for game in ordered:
        start=game["event_time"]
        if start<HOLDOUT:
            before.append(game)
            continue
        model=nba_high_total(before,home_id=game["home_id"],away_id=game["away_id"],
                             before_time=start,line=line)
        if model.probability is not None:
            target=int(game["home_score"]+game["away_score"]>=cutoff)
            # Only prior finished games, no label leakage or arbitrary markets.
            prior=[r for r in before if r["event_time"][:10]<start[:10]]
            raw=(3+sum(r["home_score"]+r["away_score"]>=cutoff for r in prior))/(len(prior)+12)
            rows.append({"event_id":game["event_id"],"event_time":start,
                         "line":line,"outcome":target,"p":model.probability,
                         "base":raw,"source_as_of":model.source_as_of})
        before.append(game)
    if not rows:
        return {"status":"NO_HISTORICAL_EVIDENCE","qualified_for_live":False},[]
    def brier(row,field):
        return (row[field]-row["outcome"])**2
    n=len(rows)
    diffs=[brier(r,"base")-brier(r,"p") for r in rows]
    dates=defaultdict(list)
    for row in rows:dates[row["event_time"][:10]].append(row)
    keys=sorted(dates)
    boot=[]
    rng=Random(20261014)
    if keys:
        for _ in range(300):
            sample=[r for key in (keys[rng.randrange(len(keys))] for _ in keys)
                    for r in dates[key]]
            boot.append(sum(brier(r,"base")-brier(r,"p") for r in sample)/len(sample))
        boot.sort()
    report={
      "evaluation_kind":"historical_chronological_holdout",
      "outcome":f"NBA complete-game total over {line} points",
      "market":"KXNBATOTAL exact full-game greater half strike",
      "model":"nba_high_total_yes_no_v1",
      "n":n,"unique_dates":len(keys),
      "realized_over_frequency":sum(r["outcome"] for r in rows)/n,
      "realized_no_frequency":1-sum(r["outcome"] for r in rows)/n,
      "predicted_mean_over":sum(r["p"] for r in rows)/n,
      "model_brier":sum(brier(r,"p") for r in rows)/n,
      "baseline_brier":sum(brier(r,"base") for r in rows)/n,
      "bootstrap_model_advantage_95":[boot[7],boot[292]] if len(boot)==300 else None,
      "no_promotion":True,"qualified_for_live":False,
      "limitations":["No executable Kalshi ask/fee historical archive",
                     "No training on money prices or screenshots",
                     "Normal tail uncalibrated without immutable future model forecasts",
                     "Regular season only; preseason games refused"] }
    if n<200 or len(keys)<30:
        report["status"]="INSUFFICIENT_HISTORY"
    elif report["bootstrap_model_advantage_95"][0]>0:
        report["status"]="HISTORICAL_BRIER_IMPROVEMENT_ONLY"
    else:
        report["status"]="NO_CONFIRMED_HISTORICAL_ADVANTAGE"
    return report,rows


def main():
    games,sources=fetch_population()
    reports={};predictions=[]
    for line in LINES:
        result,rows=replay(games,line=line)
        reports[str(line)]=result;predictions.extend(rows)
        print(json.dumps({"line":line,"status":result["status"],"n":result.get("n"),
            "over_rate":result.get("realized_over_frequency"),
            "model_brier":result.get("model_brier"),
            "baseline_brier":result.get("baseline_brier"),
            "bootstrap_95":result.get("bootstrap_model_advantage_95")},sort_keys=True))
    output=Path("nba-high-total-historical-evidence");output.mkdir(exist_ok=True)
    (output/"metrics.json").write_text(json.dumps({
        "preregistered_lines":LINES,"reports":reports,"sources":sources,
        "qualified_for_live":False},indent=2))
    (output/"predictions.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in predictions))
    if any(r["status"]=="INSUFFICIENT_HISTORY" for r in reports.values()):
        raise SystemExit("insufficient high-total historical holdout")

if __name__=="__main__":main()
