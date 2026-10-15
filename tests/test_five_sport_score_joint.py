from datetime import date, datetime, timedelta, timezone
import json
from dataclasses import replace
import pytest

from sports_edge.data.five_sport_score_archive import (
    PATHS, completed_scores, collect_range, merge_archives,
    validate_archive, upcoming_fixtures, scheduled_day,
)
from sports_edge.models.five_sport_score_joint import (
    LINES, simulate_paired_scores, joint_score_market_research,
    score_outcome, historical_score_holdout,
)
from scripts.five_sport_joint_research import (
    make_frozen_calls, grade_frozen_calls, digest, write_once
)

BASE=datetime(2025,1,1,12,tzinfo=timezone.utc)


def history(sport="NFL", n=135):
    rows=[]
    for i in range(n):
        t=BASE+timedelta(days=i)
        h=str(10+i%4)
        a=str(10+(i+1)%4)
        rows.append({"source":"ESPN_COMPLETED_SCOREBOARD","source_scope":"FULL_GAME_FINAL_SCORE_NOT_KALSHI_SETTLEMENT",
                     "sport":sport,"event_id":f"{sport}-{i}","event_time":t.isoformat(),
                     "home_id":h,"away_id":a,
                     "home_score":25+i%19+int(sport=="NBA")*80,
                     "away_score":16+i%17+int(sport=="NBA")*79})
    return rows


def archive(sports=("NFL",)):
    rows=[g for sport in sports for g in history(sport)]
    return {"schema":"SPORTEDGE-FIVE-SPORT-SCORE-ARCHIVE-V1",
            "source":"ESPN_PUBLIC_COMPLETED_SCOREBOARDS",
            "games":rows,"sports":list(sports),"provenance":[{"date":"2025-01-01","sha256":"synthetic"}],
            "research_only":True,"no_kalshi_market_settlement_claim":True}


def fixture(sport="NFL",index="future"):
    return {"sport":sport,"event_id":sport+"-"+index,"event_time":"2027-01-10T21:00:00+00:00",
            "home_id":"10","away_id":"11","source":"ESPN_SCHEDULED_SCOREBOARD",
            "status":"FUTURE_UNPLAYED"}


def future_document(sports=("NFL",)):
    return {"schema":"SPORTEDGE-FIVE-SPORT-FIXTURES-V1","sports":list(sports),
            "captured_at_utc":"2026-10-15T10:00:00Z",
            "source":"ESPN_PUBLIC_SCHEDULED_SCOREBOARD",
            "provenance":{"url":"https://example.test/","digest_of_json":"fixture-test-only"},
            "fixtures":[fixture(sport) for sport in sports],
            "no_kalshi_prices":True,"research_only":True}


@pytest.mark.parametrize("sport",tuple(PATHS))
def test_source_adapters_exact_final_game_and_not_fake_live_score(sport):
    e={"id":"888","date":"2025-06-10T20:00:00Z",
       "status":{"type":{"completed":True}},
       "competitions":[{"competitors":[
           {"homeAway":"home","score":"11","team":{"id":"A"}},
           {"homeAway":"away","score":"6","team":{"id":"B"}}]}]}
    data={"events":[e]}
    g=completed_scores(sport,data)
    assert g[0]["home_score"]==11 and g[0]["sport"]==sport
    e["status"]["type"]["completed"]=False
    assert completed_scores(sport,data)==[]
    e["status"]["type"]["completed"]=True
    e["competitions"][0]["competitors"][0]["score"]="NA"
    with pytest.raises(ValueError):
        completed_scores(sport,data)


def test_strict_archive_validation_missing_source_and_duplicate_rejected():
    doc=archive()
    assert len(validate_archive(doc))==135
    with pytest.raises(ValueError):validate_archive(dict(doc,games=doc["games"]+[doc["games"][0]]))
    with pytest.raises(ValueError):validate_archive(dict(doc,games=[dict(doc["games"][0],source="UNKNOWN")]))
    with pytest.raises(ValueError):validate_archive(doc,as_of="2025-02-01T00:00:00Z")
    with pytest.raises(ValueError):merge_archives([dict(doc,provenance=[])])


def test_bounded_fetch_and_source_digests_and_no_future():
    event={"id":"888","date":"2025-06-10T20:00:00Z","status":{"type":{"completed":True}},
       "competitions":[{"competitors":[{"homeAway":"home","score":"11","team":{"id":"A"}},
            {"homeAway":"away","score":"6","team":{"id":"B"}}]}]}
    one=collect_range(sport="NCAAF",start=date(2025,6,10),end=date(2025,6,10),
                      get=lambda url:{"events":[event]})
    assert len(one["games"])==1 and len(one["provenance"])==1
    assert one["provenance"][0]["sha256"]
    assert merge_archives([one,one])["games"]==one["games"]
    with pytest.raises(ValueError):
        collect_range(sport="NFL",start=date(2025,1,1),end=date(2025,3,1),
                      get=lambda url:{"events":[]})
    with pytest.raises(ValueError):
        collect_range(sport="MLB",start=date(2027,1,1),end=date(2027,1,1),
                      get=lambda url:{"events":[]})


@pytest.mark.parametrize("sport",tuple(PATHS))
def test_real_joint_home_win_and_total_score_scenarios_five_sports(sport):
    games=history(sport)
    target=fixture(sport)
    result=simulate_paired_scores(games,target=target,as_of="2026-10-15T10:00:00Z",draws=1000)
    assert result["status"]=="UNVALIDATED_HISTORICAL_SCORE_SCENARIOS"
    assert len(result["scores"])==1000
    assert result["qualified_for_live"] is False
    assert result["player_props_modeled"] is False
    again=simulate_paired_scores(games,target=target,as_of="2026-10-15T10:00:00Z",draws=1000)
    assert result["scenario_sha256"]==again["scenario_sha256"]
    report=joint_score_market_research(result,selections=[
        {"market":"home_wins"},{"market":"total_over","line":LINES[sport]}])
    assert 0<=report["joint_probability"]<=1
    assert report["kalshi_combo_edge"] is None
    assert report["cashout_probability"] is None
    with pytest.raises(ValueError):
        joint_score_market_research(result,selections=[{"market":"player_anytime_td"},{"market":"home_wins"}])


def test_joint_probability_differs_from_product_from_shared_game_scripts():
    scenarios={"status":"UNVALIDATED_HISTORICAL_SCORE_SCENARIOS",
               "scenario_sha256":"synthetic-correlated",
               "scores":[(35,10)]*600+[(7,12)]*400}
    r=joint_score_market_research(scenarios,selections=[
        {"market":"home_wins"},{"market":"total_over","line":35.5}])
    assert r["joint_probability"]==.6
    assert r["marginal_probabilities"]==[.6,.6]
    assert r["independence_benchmark_not_a_forecast"]==.36


def test_out_of_sample_history_does_not_peek_at_filled_outcome():
    games=history("NFL")
    report,records=historical_score_holdout(games,sport="NFL",
         holdout_start="2025-04-15T00:00:00Z",min_test=20)
    assert report["status"]=="HISTORICAL_HOLDOUT_RESEARCH_ONLY"
    assert len(records)>=20
    assert len({r["event_id"] for r in records})==len(records)
    assert all(r["predicted_at"]<r["event_time"] for r in records)
    assert report["qualified_for_live"] is False
    assert 0<=report["joint_brier"]<=1
    fewer,rows=historical_score_holdout(games[:75],sport="NFL",
          holdout_start="2025-03-15T00:00:00Z",min_test=50)
    assert fewer["status"]=="INSUFFICIENT_RETROSPECTIVE_HOLDOUT"


def test_duplicate_games_and_post_kickoff_rejected_without_predictions():
    games=history("NBA")
    with pytest.raises(ValueError):historical_score_holdout(
        games+games[:1],sport="NBA",holdout_start="2025-04-01T00:00:00Z")
    with pytest.raises(ValueError):simulate_paired_scores(
        games,target=fixture("NBA"),as_of="2027-02-01T00:00:00Z")
    with pytest.raises(ValueError):simulate_paired_scores(
        games,target=fixture("NBA"),as_of="2026-10-15T10:00:00Z",draws=20)


def test_future_fixture_scrub_and_capture_grade_identity_sealed(tmp_path):
    sports=("NFL","NCAAF","NBA","NHL","MLB")
    raw=archive(sports)
    fixture_doc=future_document(sports)
    # A prior-season-only archive is too stale for prospective predictions.
    stale=make_frozen_calls(raw,fixture_doc,as_of="2026-10-15T10:01:00Z")
    assert all(r["joint_p"] is None and
               r["status"]=="STALE_SPORT_HISTORY_FOR_FUTURE_PREDICTION"
               for r in stale["calls"])
    # Add a causally earlier recent season with the same verified score fields.
    # Only these recent records can support an October 2026 future paper call.
    for sp in sports:
        for i,r in enumerate(history(sp)):
            raw["games"].append(dict(r,event_id=r["event_id"]+"-2026",
                event_time=(datetime(2026,5,20,12,tzinfo=timezone.utc)+
                            timedelta(days=i)).isoformat()))
    frozen=make_frozen_calls(raw,fixture_doc,as_of="2026-10-15T10:01:00Z")
    assert frozen["qualified_for_live"] is False
    assert len(frozen["calls"])==5
    assert all(x["joint_p"] is not None for x in frozen["calls"])
    location=tmp_path/"immutable.json"
    write_once(location,frozen)
    with pytest.raises(FileExistsError):write_once(location,frozen)
    assert (location.stat().st_mode & 0o077)==0
    pending=grade_frozen_calls(frozen,raw)
    assert pending["graded"]==0 and pending["pending"]==5
    finals=[]
    for sp in sports:
        x=fixture(sp)
        finals.append({
            "sport":sp,"event_id":x["event_id"],"event_time":x["event_time"],
            "home_id":"10","away_id":"11","home_score":30,"away_score":10,
            "source":"ESPN_COMPLETED_SCOREBOARD",
            "source_scope":"FULL_GAME_FINAL_SCORE_NOT_KALSHI_SETTLEMENT"})
    graded=grade_frozen_calls(frozen,dict(raw,games=raw["games"]+finals))
    assert graded["graded"]==5
    assert all(r["status"]=="GRADED_PAPER_ONLY" for r in graded["records"])
    damaged=dict(frozen,calls=[dict(frozen["calls"][0],joint_p=.999)]+frozen["calls"][1:])
    with pytest.raises(ValueError,match="mutated"):
        grade_frozen_calls(damaged,raw)
    mismatched=dict(raw,games=raw["games"]+[dict(finals[0],home_id="OTHER")])
    with pytest.raises(ValueError,match="mismatch"):
        grade_frozen_calls(frozen,mismatched)


def test_fixture_parser_ignores_finished_and_started_but_includes_future():
    future={"id":"g1","date":"2027-01-10T21:00:00Z",
        "status":{"type":{"completed":False}},
        "competitions":[{"competitors":[
            {"homeAway":"home","team":{"id":"10"}},
            {"homeAway":"away","team":{"id":"11"}}]}]}
    import copy
    finished=copy.deepcopy(future);finished["id"]="g2";finished["status"]["type"]["completed"]=True
    started=copy.deepcopy(future);started["id"]="g3";started["date"]="2025-02-10T21:00:00Z"
    from sports_edge.data.five_sport_score_archive import upcoming_fixtures
    selected=upcoming_fixtures("NFL",{"events":[future,finished,started]},as_of="2026-10-15T10:00:00Z")
    assert len(selected)==1 and selected[0]["event_id"]=="g1"
    with pytest.raises(ValueError):
        make_frozen_calls(archive(),dict(future_document(),fixtures=[dict(fixture(),status="FINAL")]),
                          as_of="2026-10-15T10:00:00Z")


@pytest.mark.parametrize("sport",("NCAAF","NBA","NHL"))
def test_exact_midnight_utc_kickoffs_require_previous_day_freeze(sport):
    rows=[dict(r,event_time=r["event_time"].replace("T12:00:00","T00:00:00"))
          for r in history(sport)]
    report,predictions=historical_score_holdout(
        rows,sport=sport,holdout_start="2025-04-15T00:00:00Z",min_test=20)
    assert report["status"]=="HISTORICAL_HOLDOUT_RESEARCH_ONLY"
    assert all(datetime.fromisoformat(r["predicted_at"])<
               datetime.fromisoformat(r["event_time"]) for r in predictions)
    assert all((datetime.fromisoformat(r["event_time"])-
               datetime.fromisoformat(r["predicted_at"])).days>=1
               for r in predictions)
