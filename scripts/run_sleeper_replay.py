"""Source-backed smoke or complete NHL/NFL challenger replay; never promotion.

Run on demand. --sample-size is a nonrepresentative integration smoke, and the
report explicitly excludes live betting performance or calibrated edge claims.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from sports_edge.data.sleeper_sources import SleeperSources, NHL
from sports_edge.models.sleeper_counts import nhl_count, nfl_scored_touchdowns
from sports_edge.models.sport_calibration import chronological_calibration


def replay(log, *, sport, field, threshold):
    ordered=sorted(log,key=lambda r:r["date"])
    out=[]
    for i,r in enumerate(ordered):
        before=r["date"]+"T00:00:00Z"
        projection=(nhl_count(ordered[:i],market=field,threshold=threshold,before_time=before)
                    if sport=="NHL" else nfl_scored_touchdowns(ordered[:i],threshold=threshold,before_time=before))
        value=r.get(field) if sport=="NHL" else r.get("scored_tds")
        if projection.probability is None or type(value) is not int:
            continue
        pid=r.get("player_id") or r.get("team_id")
        out.append({"sport":sport,"event_id":str(r["game_id"])+":"+str(pid),"event_time":r["date"]+"T12:00:00Z",
                    "predicted_at":(datetime.fromisoformat(r["date"])-timedelta(days=1)).isoformat()+"Z",
                    "source_id":r["source"]+":"+str(r["game_id"]),"model_version":projection.model,
                    "outcome":field+"_"+str(threshold)+"_plus", "probability":projection.probability,
                    "won":value>=threshold,"evaluation_kind":"historical_walkforward","status":"RESEARCH_ONLY"})
    return out


def run(output_dir, sample_size=4):
    dest=Path(output_dir); dest.mkdir(parents=True,exist_ok=True)
    sources=SleeperSources()
    records=[]; coverage={}; failures=[]
    for kind in ("skater","goalie"):
        url=f"https://api.nhle.com/stats/rest/en/{kind}/summary"
        data=sources.get(url,{"isAggregate":"false","isGame":"false","cayenneExp":"seasonId=20242025 and gameTypeId=2","limit":-1,"start":0})
        rows=data.get("data",[])
        if len(rows)!=data.get("total") or not rows:
            raise ValueError("official population incomplete")
        ids=sorted({r["playerId"] for r in rows})
        selected=ids[:sample_size] if sample_size else ids
        coverage[kind]={"official_population":len(ids),"sampled":len(selected),"representative":not sample_size}
        def collect(pid):
            s=SleeperSources()
            data=s.get(f"{NHL}/player/{pid}/game-log/20242025/2")
            log=[]
            for r in data.get("gameLog",[]):
                if kind=="goalie" and r.get("gamesStarted")!=1:
                    continue
                item={"source":"NHL","player_id":pid,"game_id":str(r["gameId"]),"date":r["gameDate"]}
                for field in ("shots","goals","assists","points"):
                    item[field]=r.get(field)
                mm,ss=r.get("toi","0:00").split(":");item["toi_minutes"]=int(mm)+int(ss)/60
                if type(r.get("shotsAgainst")) is int and type(r.get("goalsAgainst")) is int:
                    item["saves"]=r["shotsAgainst"]-r["goalsAgainst"]
                log.append(item)
            return log,s.provenance
        with ThreadPoolExecutor(max_workers=4) as pool:
            for log,provenance in pool.map(collect,selected):
                sources.provenance.extend(provenance)
                if kind=="goalie":
                    records.extend(replay(log,sport="NHL",field="saves",threshold=25))
                else:
                    for field,threshold in (("shots",3),("goals",2),("assists",1),("points",2)):
                        records.extend(replay(log,sport="NHL",field=field,threshold=threshold))
    # All NHL franchises enumerated from official standings; no scoring-leader sample.
    teams=sources.get(f"{NHL}/standings/2025-04-17")["standings"]
    abbreviations=sorted({r["teamAbbrev"]["default"] for r in teams})
    if len(abbreviations) != 32:
        raise ValueError("historical NHL team population incomplete")
    selected=abbreviations[:sample_size] if sample_size else abbreviations
    coverage["team"]={"official_population":len(abbreviations),"sampled":len(selected),"representative":not sample_size}
    for abbr in selected:
        data=sources.get(f"{NHL}/club-schedule-season/{abbr}/20242025")
        log=[]
        for g in data.get("games",[]):
            if g.get("gameType")!=2 or g.get("gameState") not in {"OFF","FINAL"}:
                continue
            side="homeTeam" if g["homeTeam"]["abbrev"]==abbr else "awayTeam"
            score=g[side]["score"]
            # Official final score includes the shootout-winning goal, matching
            # the verified KXNHLTEAMTOTAL contract scope.
            log.append({"source":"NHL","team_id":abbr,"game_id":str(g["id"]),"date":g["gameDate"],"team_goals":score})
        records.extend(replay(log,sport="NHL",field="team_goals",threshold=3))
    try:
        from sports_edge.data.nfl_prop_data import player_week_rows, causal_rows
        from datetime import date
        from math import isfinite
        raw=causal_rows(player_week_rows(2025),season=2025,before_date=date(2026,3,1))
        groups=defaultdict(list)
        for r in raw:
            if not r.get("player_id") or r.get("season_type") not in {"REG",""}:
                continue
            values=[float(r[k]) for k in ("rushing_tds","receiving_tds")]
            if any(not isfinite(v) or v<0 or not v.is_integer() for v in values):
                raise ValueError("invalid official touchdown count")
            groups[r["player_id"]].append(dict(r,source="nflverse",scored_tds=int(sum(values))))
        eligible={k:v for k,v in groups.items() if len(v)>=12}
        selected=sorted(eligible)[:sample_size] if sample_size else sorted(eligible)
        coverage["NFL"]={"eligible_players":len(eligible),"sampled":len(selected),"representative":not sample_size,
                         "source":"nflverse stats_player_week_2025; schedule-joined causal dates"}
        for pid in selected:
            records.extend(replay(eligible[pid],sport="NFL",field="scored_tds",threshold=2))
    except Exception as exc:
        failures.append("NFL source collection failed:"+type(exc).__name__)
    metrics={}
    for sport,outcome in sorted({(r["sport"],r["outcome"]) for r in records}):
        rows=[r for r in records if r["sport"]==sport and r["outcome"]==outcome]
        metrics[sport+":"+outcome]=chronological_calibration(rows)
    report={"evaluation_kind":"nonrepresentative_source_smoke" if sample_size else "historical_population_replay",
            "prospective":False,"promoted":False,"coverage":coverage,"failures":failures,
            "predictions":len(records),"per_market":metrics,"sources":sources.provenance}
    (dest/"sleeper_replay_report.json").write_text(json.dumps(report,indent=2)+"\n")
    (dest/"sleeper_predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output-dir",required=True);p.add_argument("--sample-size",type=int,default=4)
    a=p.parse_args()
    if a.sample_size<0: p.error("sample-size must be zero (all) or positive")
    report=run(a.output_dir,a.sample_size)
    print(json.dumps(report,indent=2))
    if report["failures"]:
        raise SystemExit(1)
