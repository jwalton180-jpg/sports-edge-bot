from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import pytest

from sports_edge.models.sleeper_counts import nhl_count, nfl_scored_touchdowns
from sports_edge.models.sleeper_pipeline import contract_spec, entry_fee, execution_quote, scan_sleeper_markets, audit_app_candidates, export_evidence
from sports_edge.models.kalshi_sports import KalshiSportMarket, classify_kalshi_market
from sports_edge.models.ncaaf_vulnerability import review_upset, REQUIRED
from sports_edge.models.sleeper_ledger import SleeperLedger
from sports_edge.models.ticket_dependencies import incompatible, probability_bounds
from sports_edge.models.basketball_sleeper import basketball_sleeper
from sports_edge.models.parlay_intelligence import assess_leg
from tests.test_parlay_intelligence import leg

NOW = datetime(2026,10,14,18,tzinfo=timezone.utc)


def market():
    return {"ticker":"KXNHLPTS-26OCT14MTLWSH-A-2", "series_ticker":"KXNHLPTS", "event_ticker":"KXNHLPTS-26OCT14MTLWSH",
            "status":"active", "title":"Alpha Player: 2+ points", "market_type":"binary",
            "floor_strike":1.5, "strike_type":"greater", "occurrence_datetime":"2026-10-14T23:30:00Z",
            "rules_primary":"If Alpha Player records 2+ points in Montreal vs Washington then the market resolves to Yes.",
            "rules_secondary":"If active but never enters the game, settle to fair market price.",
            "yes_ask_dollars":".10", "no_ask_dollars":".93", "yes_bid_dollars":".08", "no_bid_dollars":".90",
            "yes_ask_size_fp":"10.00", "no_ask_size_fp":"15.00"}


def history():
    return [{"source":"NHL", "player_id":1, "game_id":str(i), "date":(NOW.date()-timedelta(days=40-i)).isoformat(),
             "shots":i%5, "goals":int(i%5==2), "assists":i%3, "points":int(i%5==2)+i%3,
             "saves":20+i%15,"team_goals":i%6, "toi_minutes":18+i%4} for i in range(30)]


@pytest.mark.parametrize("field,threshold",[("shots",3),("goals",2),("assists",2),("points",2),("saves",25),("team_goals",3)])
def test_nhl_menu_has_monotone_tails_and_explicit_uncertainty(field,threshold):
    a=nhl_count(history(),market=field,threshold=threshold,before_time=NOW.isoformat())
    b=nhl_count(history(),market=field,threshold=threshold+1,before_time=NOW.isoformat())
    assert 0 <= b.probability <= a.probability <= 1
    assert a.lower <= a.probability <= a.upper
    assert a.status == "UNVALIDATED_RESEARCH"


def test_future_outcome_and_duplicate_or_missing_history():
    h=history()
    baseline=nhl_count(h,market="points",threshold=2,before_time=NOW.isoformat())
    future=dict(h[-1],date="2026-10-14",game_id="future",points=80)
    assert nhl_count(h+[future],market="points",threshold=2,before_time=NOW.isoformat()) == baseline
    assert nhl_count(h+[h[-1]],market="points",threshold=2,before_time=NOW.isoformat()).probability is None
    assert nhl_count([dict(r,shots=None) for r in h],market="shots",threshold=3,before_time=NOW.isoformat()).probability is None
    assert nhl_count([dict(r,player_id=i) for i,r in enumerate(h)],market="points",threshold=2,before_time=NOW.isoformat()).probability is None


def test_nfl_two_plus_excludes_passing_touchdowns_and_checks_tail():
    h=[dict(r,source="nflverse",rushing_tds=i%2,receiving_tds=int(i%4==0),passing_tds=100) for i,r in enumerate(history())]
    a=nfl_scored_touchdowns(h,threshold=2,before_time=NOW.isoformat())
    b=nfl_scored_touchdowns([dict(r,passing_tds=0) for r in h],threshold=2,before_time=NOW.isoformat())
    assert a == b and 0 < a.probability < 1
    assert nfl_scored_touchdowns([dict(r,receiving_tds=None) for r in h],before_time=NOW.isoformat()).probability is None


def test_real_nhl_aliases_and_strike_rule_disagreement():
    for series,family in [("KXNHLGOAL","Player Goals"),("KXNHLPTS","Player Points"),("KXNHLAST","Player Assists"),("KXNHLSAVES","Goalie Saves")]:
        assert classify_kalshi_market(dict(market(),ticker=series+"-X",series_ticker=series)) == ("NHL",family)
    row=KalshiSportMarket("NHL","Player Points",market())
    assert contract_spec(row).rules_supported
    assert not contract_spec(replace(row,market=dict(market(),floor_strike=2.5))).rules_supported
    assert not contract_spec(replace(row,market=dict(market(),rules_secondary="Escalator payout increases"))).rules_supported


def test_team_goals_recognizes_verified_shootout_bonus_scope():
    m=dict(market(),title="Vegas over 2.5 goals scored",yes_sub_title="Vegas over 2.5",floor_strike=2.5,
           rules_primary="If Vegas scores over 2.5 goals, then resolves to Yes.",
           rules_secondary="Team totals include goals scored in overtime. If the game goes to a shootout, the shootout winner is credited with one additional goal, and that goal counts toward this market.")
    spec=contract_spec(KalshiSportMarket("NHL","Team Total",m))
    assert spec.rules_supported and spec.participant=="Vegas"


def test_executable_costs_round_by_quantity_and_fail_closed():
    series={"fee_type":"quadratic", "fee_multiplier":1}
    assert entry_fee(.1,series,contracts=1) == .01
    assert entry_fee(.1,series,contracts=100) == .0063
    q=execution_quote(market(),"YES",series,received_at=NOW.isoformat(),now=NOW.isoformat())
    assert q["break_even"] == pytest.approx(.11) and not q["reasons"]
    assert entry_fee(.1,{"fee_type":"flat","fee_multiplier":1}) is None
    assert entry_fee(.1,{"fee_type":"quadratic","fee_multiplier":"NaN"}) is None
    q=execution_quote(dict(market(),yes_ask_size_fp="0"),"YES",series,received_at=NOW.isoformat(),now=(NOW+timedelta(minutes=3)).isoformat())
    assert "quote_stale_or_timestamp_missing" in q["reasons"]
    assert "ask_size_unverified_or_insufficient" in q["reasons"]


def test_ncaaf_missing_evidence_and_multiple_material_flags_block_best():
    assert not review_upset({},event_id="G",now=NOW.isoformat()).best_available_eligible
    evidence={k:{"event_id":"G","source_url":"https://official.example/injury","as_of":NOW.isoformat(),
                 "verified":True,"finding":"reviewed","severity":"clear"} for k in REQUIRED}
    assert review_upset(evidence,event_id="G",now=NOW.isoformat()).best_available_eligible
    evidence["qb_availability"]["severity"]="material"
    evidence["underdog_attack_vs_defense"]["severity"]="material"
    assert "multiple_material_upset_red_flags" in review_upset(evidence,event_id="G",now=NOW.isoformat()).reasons


def test_preseason_never_borrows_regular_season_rotations():
    p=basketball_sleeper(sport="NBA",player="Player",family="Points",threshold=20,before_time=NOW.isoformat(),season_type="preseason")
    assert p.probability is None
    assert "preseason" in p.reasons[0]


class FakeClient:
    def market(self,ticker):
        from types import SimpleNamespace
        return SimpleNamespace(data={"market":market()},received_at=NOW)
    def series(self,ticker):
        from types import SimpleNamespace
        return SimpleNamespace(data={"series":{"ticker":ticker,"fee_type":"quadratic","fee_multiplier":1}})


class FakeSources:
    provenance=[]
    def nhl_player(self,m,p):
        return {"rows":history(),"missing":["confirmed_line_role"]}


def example_report():
    report=scan_sleeper_markets([market()],coverage_complete=True,client=FakeClient(),sources=FakeSources(),now=NOW)
    return report


def test_both_sides_research_positive_and_negative_edges_never_auto_promote():
    report=example_report()
    assert len(report["records"]) == 2
    assert {r["side"] for r in report["records"]} == {"YES","NO"}
    assert report["records"][0]["net_edge"] > 0
    assert report["records"][1]["net_edge"] < 0
    assert all(r["status"] != "QUALIFIED_VALUE" for r in report["records"])
    assert "prospective_validation_required" in report["records"][0]["reasons"]
    assert json.loads(export_evidence(report))["safety"]["order_submission"] is False


def test_scan_capacity_keeps_unprocessed_inventory_and_marks_incomplete():
    report=scan_sleeper_markets([market(),dict(market(),ticker="KXNHLPTS-X")],coverage_complete=False,limit=1,client=FakeClient(),sources=FakeSources(),now=NOW)
    assert len(report["records"]) == 4
    assert not report["coverage"]["scan_complete"]
    assert "scan_limit_not_analyzed" in report["records"][2]["reasons"]


def test_audited_candidates_cannot_bypass_missing_validation():
    raw=leg(fair=.7,price=.1)
    m=dict(market(),sports_edge_received_at=NOW.isoformat(),sports_edge_fee_type="quadratic",sports_edge_fee_multiplier=1)
    raw=replace(raw,kalshi_ticker=m["ticker"],sport="NHL")
    audited=audit_app_candidates([raw],[m],coverage_complete=True,now=NOW)[0]
    assert not assess_leg(audited,"best").qualified
    assert audited.entry_fee_per_contract == .01


def test_ticket_rejects_opposite_sides_redundant_ladders_and_winners():
    a=replace(leg(selection="Player Over 1.5 TD"),event_id="G",market_key="player_anytime_td")
    b=replace(a,kalshi_side="NO")
    assert incompatible(a,b)
    b=replace(a,kalshi_ticker="OTHER",selection="Player Over 2.5 TD")
    assert incompatible(a,b)
    a=replace(a,market_key="model_h2h",selection="Team A")
    b=replace(b,market_key="model_h2h",selection="Team B")
    assert incompatible(a,b)
    assert probability_bounds([.7,.8]) == pytest.approx((.5,.7))


def test_ledger_preserves_rejections_and_prevents_edits(tmp_path):
    report=example_report()
    for r in report["records"]:
        r["observed_at"]="2026-10-13T18:00:00Z"
        r["event_time"]="2026-10-13T23:30:00Z"
    ledger=SleeperLedger(tmp_path/"evidence.sqlite")
    key=ledger.append_scan(report)
    assert ledger.append_scan(report) == key
    saved=json.loads(ledger.export_jsonl())
    assert len(saved["payload"]["records"]) == 2
    with pytest.raises(sqlite3.IntegrityError):
        ledger.db.execute("UPDATE records SET payload='{}'")
    ledger.settle(scan_id=key,ticker=market()["ticker"],result="yes",source_url="https://kalshi.com/market",settled_at="2026-10-14T18:01:00Z")
    with pytest.raises(sqlite3.IntegrityError):
        ledger.settle(scan_id=key,ticker=market()["ticker"],result="no",source_url="https://kalshi.com/market",settled_at="2026-10-14T18:02:00Z")
    ledger.close()


def test_unsigned_or_modified_promotion_cannot_qualify(tmp_path):
    from sports_edge.models.sleeper_promotion import load_reviewed_promotions
    path=tmp_path/"reviews.json"
    path.write_text(json.dumps([{"payload":{},"signature":"bad"}]))
    assert load_reviewed_promotions(path,verification_key="local-fixture-only",now=NOW.isoformat()) == {}
    assert load_reviewed_promotions(path,verification_key=None,now=NOW.isoformat()) == {}


def test_settlements_before_event_are_rejected(tmp_path):
    ledger=SleeperLedger(tmp_path/"records.sqlite")
    report=example_report();key=ledger.append_scan(report)
    with pytest.raises(ValueError):
        ledger.settle(scan_id=key,ticker=market()["ticker"],result="yes",source_url="https://kalshi.com/market",settled_at="2026-10-14T18:01:00Z")
    assert ledger.metrics()["groups"] == []
    ledger.close()


def test_official_finalized_status_grades_and_fair_price_does_not(tmp_path):
    from types import SimpleNamespace
    ledger=SleeperLedger(tmp_path/"official.sqlite")
    report=example_report()
    for r in report["records"]:
        r["observed_at"]="2026-10-13T18:00:00Z"
        r["event_time"]="2026-10-13T23:30:00Z"
    ledger.append_scan(report)
    class Official:
        base="https://external-api.kalshi.com/trade-api/v2"
        value="0.50"
        def market(self,ticker):
            return SimpleNamespace(data={"market":{"ticker":ticker,"status":"finalized","result":"yes",
                                    "settlement_value_dollars":self.value,"settlement_ts":"2026-10-14T01:00:00Z"}})
    client=Official()
    assert ledger.grade_official(client)["graded"] == 0
    client.value="1.00"
    assert ledger.grade_official(client)["graded"] == 1
    assert len(ledger.metrics()["groups"]) == 2
    assert ledger.metrics()["realized_returns"] is None
    ledger.close()


def test_bounded_grading_rotates_past_open_markets(tmp_path):
    from types import SimpleNamespace
    ledger=SleeperLedger(tmp_path/"rotate.sqlite")
    for ticker in ("A","B"):
        report=example_report()
        for r in report["records"]:
            r["ticker"]=ticker
        ledger.append_scan(report)
    class Open:
        calls=[]
        def market(self,ticker):
            self.calls.append(ticker)
            return SimpleNamespace(data={"market":{"ticker":ticker,"status":"active"}})
    client=Open()
    ledger.grade_official(client,limit=1)
    ledger.grade_official(client,limit=1)
    assert client.calls==["A","B"]
    ledger.close()
