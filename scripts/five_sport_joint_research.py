"""On-demand five-sport historical scores, holdouts and immutable paper calls.

No orders, accounts, persistent workers or background jobs. All output
is local and refuses to overwrite files. Individual picks are TEAM/GAME
SCORE research only. Never claims historical Kalshi market identity.
"""
from __future__ import annotations
import argparse
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from sports_edge.data.five_sport_score_archive import (
    PATHS, collect_range, merge_archives, validate_archive,
    scheduled_day, utc,
)
from sports_edge.models.five_sport_score_joint import (
    LINES, simulate_paired_scores, joint_score_market_research,
    historical_score_holdout, score_outcome,
)


def read_json(path):
    f=Path(path)
    if f.stat().st_size>15_000_000:raise ValueError("research_file_too_large")
    return json.loads(f.read_text(encoding="utf-8"))


def write_once(path,doc):
    dest=Path(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    content=(json.dumps(doc,indent=2,sort_keys=True)+"\n").encode()
    fd=os.open(dest,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        with os.fdopen(fd,"wb") as writer:
            writer.write(content)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return dest


def digest(data):
    return sha256(json.dumps(data,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def make_frozen_calls(archive, fixture_doc, *, as_of):
    games=validate_archive(archive)
    if fixture_doc.get("schema")!="SPORTEDGE-FIVE-SPORT-FIXTURES-V1" or not fixture_doc.get("research_only"):
        raise ValueError("validated_future_fixtures_required")
    if not fixture_doc.get("provenance") or fixture_doc.get("no_kalshi_prices") is not True:
        raise ValueError("fixture_provenance_or_market_separation_missing")
    fixtures=fixture_doc.get("fixtures")
    if not isinstance(fixtures,list) or len(fixtures)>50:
        raise ValueError("unbounded_fixtures")
    now=utc(as_of)
    if utc(fixture_doc.get("captured_at_utc"))>now:
        raise ValueError("fixture_capture_after_prediction")
    rows=[];seen=set()
    for event in fixtures:
        if event.get("source")!="ESPN_SCHEDULED_SCOREBOARD" or event.get("status")!="FUTURE_UNPLAYED":
            raise ValueError("fixture_not_from_schedule")
        if event.get("sport") not in PATHS or utc(event["event_time"])<=now:
            raise ValueError("already_started_or_unsupported_fixture")
        key=(event["sport"],event["event_id"])
        if key in seen:raise ValueError("duplicate_event_in_slate")
        seen.add(key)
        scenarios=simulate_paired_scores(games,target=event,
                                         as_of=now.isoformat(),draws=1000)
        if scenarios["status"]!="UNVALIDATED_HISTORICAL_SCORE_SCENARIOS":
            rows.append({"sport":event["sport"],"event_id":event["event_id"],
                         "event_time":event["event_time"],"status":scenarios["status"],
                         "joint_p":None})
            continue
        line=LINES[event["sport"]]
        combo=joint_score_market_research(scenarios,selections=[
            {"market":"home_wins"},{"market":"total_over","line":line}])
        rows.append({"sport":event["sport"],"event_id":event["event_id"],
                     "event_time":event["event_time"],"home_id":event["home_id"],
                     "away_id":event["away_id"],"status":combo["status"],
                     "market_hypothesis":"home_wins_AND_full_game_total_over",
                     "total_line":line,"joint_p":combo["joint_probability"],
                     "independence_benchmark":combo["independence_benchmark_not_a_forecast"],
                     "draws":combo["scenario_count"],"scenario_sha256":combo["scenario_sha256"]})
    result={"schema":"SPORTEDGE-FIVE-SPORT-FROZEN-PAPER-V1",
            "as_of_utc":now.isoformat(),"fixture_source_sha256":digest(fixture_doc),
            "source_archive_sha256":digest(archive),
            "calls":rows,"research_only":True,
            "qualified_for_live":False,"cashout_probability":None,
            "scope":"SCORE_ONLY_NOT_KALSHI_CONTRACT_SETTLEMENT"}
    result["snapshot_sha256"]=digest(result)
    return result


def grade_frozen_calls(frozen,archive):
    if frozen.get("schema")!="SPORTEDGE-FIVE-SPORT-FROZEN-PAPER-V1" or not frozen.get("research_only"):
        raise ValueError("invalid_frozen_snapshot")
    supplied=frozen.get("snapshot_sha256")
    if not isinstance(supplied,str) or digest({k:v for k,v in frozen.items() if k!="snapshot_sha256"})!=supplied:
        raise ValueError("frozen_snapshot_mutated")
    now=utc(frozen["as_of_utc"])
    finals={(x["sport"],x["event_id"]):x for x in validate_archive(archive)}
    rows=[];seen=set()
    for row in frozen.get("calls",[]):
        sport=row.get("sport");id_=row.get("event_id")
        key=(sport,id_)
        if key in seen:raise ValueError("duplicate_frozen_call")
        seen.add(key)
        if utc(row["event_time"])<=now:raise ValueError("after_kickoff_prediction")
        outcome=finals.get(key)
        if row.get("joint_p") is None or outcome is None:
            rows.append({"sport":sport,"event_id":id_,"status":"PENDING_OR_UNSCORABLE","won":None})
            continue
        if (outcome["home_id"]!=row["home_id"] or outcome["away_id"]!=row["away_id"] or
            utc(outcome["event_time"])!=utc(row["event_time"])):
            raise ValueError("fixture_and_final_identity_mismatch")
        if not 0<=float(row["joint_p"])<=1:raise ValueError("invalid_frozen_probability")
        y=score_outcome(outcome["home_score"],outcome["away_score"],"home_wins") and \
          score_outcome(outcome["home_score"],outcome["away_score"],"total_over",row["total_line"])
        rows.append({"sport":sport,"event_id":id_,"status":"GRADED_PAPER_ONLY",
                     "won":bool(y),"p":row["joint_p"],"brier":(row["joint_p"]-int(y))**2})
    graded=[r for r in rows if r["status"]=="GRADED_PAPER_ONLY"]
    return {"status":"IMMUTABLE_PAPER_GRADE_NOT_LIVE_QUALIFICATION",
            "frozen_snapshot_sha256":supplied,"total":len(rows),
            "graded":len(graded),"pending":len(rows)-len(graded),
            "brier":sum(x["brier"] for x in graded)/len(graded) if graded else None,
            "records":rows,"qualified_for_live":False,
            "no_combo_bids_or_fills":True}


def main():
    p=argparse.ArgumentParser(description="One-shot public scores / paper evidence; NEVER trades")
    cmd=p.add_subparsers(dest="cmd",required=True)
    collect=cmd.add_parser("collect")
    collect.add_argument("--sport",choices=sorted(PATHS),required=True)
    collect.add_argument("--start",required=True)
    collect.add_argument("--end",required=True)
    collect.add_argument("--output",required=True)
    merge=cmd.add_parser("merge")
    merge.add_argument("--archive",action="append",required=True)
    merge.add_argument("--output",required=True)
    bt=cmd.add_parser("backtest")
    bt.add_argument("--archive",required=True)
    bt.add_argument("--holdout-start",required=True)
    bt.add_argument("--output",required=True)
    sl=cmd.add_parser("fixtures")
    sl.add_argument("--sport",choices=sorted(PATHS),required=True)
    sl.add_argument("--day",required=True)
    sl.add_argument("--output",required=True)
    freeze=cmd.add_parser("freeze")
    freeze.add_argument("--archive",required=True)
    freeze.add_argument("--fixtures",required=True)
    freeze.add_argument("--output",required=True)
    grade=cmd.add_parser("grade")
    grade.add_argument("--frozen",required=True)
    grade.add_argument("--final-archive",required=True)
    grade.add_argument("--output",required=True)
    args=p.parse_args()
    if args.cmd=="collect":
        doc=collect_range(sport=args.sport,start=date.fromisoformat(args.start),
                          end=date.fromisoformat(args.end))
    elif args.cmd=="merge":
        doc=merge_archives([read_json(s) for s in args.archive])
    elif args.cmd=="backtest":
        doc=read_json(args.archive)
        games=validate_archive(doc)
        reports={}
        for sport in PATHS:
            report,records=historical_score_holdout(games,sport=sport,
                      holdout_start=args.holdout_start)
            reports[sport]={"report":report,"prediction_rows":records}
        doc={"schema":"SPORTEDGE-FIVE-SPORT-SCORE-BACKTEST-V1","results":reports,
             "archive_sha256":digest(doc),"qualified_for_live":False,
             "historical_only":True}
    elif args.cmd=="fixtures":
        doc=scheduled_day(args.sport,date.fromisoformat(args.day),
             as_of=datetime.now(timezone.utc).isoformat())
    elif args.cmd=="freeze":
        doc=make_frozen_calls(read_json(args.archive),read_json(args.fixtures),
                as_of=datetime.now(timezone.utc).isoformat())
    elif args.cmd=="grade":
        doc=grade_frozen_calls(read_json(args.frozen),read_json(args.final_archive))
    path=write_once(args.output,doc)
    print("RESEARCH_OUTPUT_WRITTEN",path)
    if "games" in doc:print("COMPLETED_GAMES",len(doc["games"]))
    if "calls" in doc:print("FROZEN_CALLS",len(doc["calls"]))
    if "results" in doc:
        for sp,item in doc["results"].items():
            print("BACKTEST",sp,item["report"]["status"],item["report"]["n"])
    print("LIVE_TRADING",False,"NO_BACKGROUND_SERVICE",True)


if __name__=="__main__":
    main()
