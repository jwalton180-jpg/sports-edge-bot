from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from tests.test_parlay_intelligence import leg
from sports_edge.models.all_game_joint_research import (
    SPORTS, all_game_research_board, evaluate_joint_scenarios, same_event_risk,
)

NOW=datetime(2026,10,15,18,tzinfo=timezone.utc)
START="2026-10-18T20:00:00Z"


def candidate(sport="NFL", event="GAME-A", selection="Player A: Over 5.5 receptions",
              fair=.70,price=.49,key="player_receptions",side="YES"):
    return replace(leg(sport=sport,event_id=event,selection=selection,
                       fair=fair,price=price,model_conf=.83),
                   market_key=key,kalshi_side=side,entry_fee_per_contract=.01,
                   execution_audit_status="QUALIFIED_VALUE",
                   verified_start=START,validation_reviewed=True)


def bundle(rows, joint=True):
    keys=[f"{r.kalshi_ticker}|{r.kalshi_side}" for r in rows]
    out=[]
    for n in range(1000):
        # First marginal .7 and second .6, same-event positive dependence.
        mask=[n<700,n<600] if joint else [n<700,n>=400]
        out.append({"scenario_id":n,"outcomes":{k:mask[i%2] for i,k in enumerate(keys)}})
    return {
        "source_kind":"INDEPENDENT_PREGAME_JOINT_SCENARIOS",
        "generated_at_utc":"2026-10-15T17:55:00Z",
        "review_status":"RESEARCH_ONLY",
        "model_version":"synthetic-fixture-only-v1",
        "draws":out
    }


@pytest.mark.parametrize("sport,key",[
    ("NFL","player_receptions"),("NCAAF","ncaaf_game_total"),
    ("NBA","player_points"),("NHL","player_shots"),
    ("MLB","batter_hits"),
])
def test_all_five_sports_game_coverage_any_day(sport,key):
    x=candidate(sport=sport,key=key)
    board=all_game_research_board([x],now=NOW)
    assert board["total"]==board["qualified_pregame"]==1
    assert board["blocked"]==0
    assert board["sports"][sport]["games"]==1
    assert board["rows"][0]["Type"]
    assert board["cashout_probability_status"].startswith("UNVALIDATED")
    assert board["joint_probability_status"].startswith("UNKNOWN")


def test_all_games_and_no_thursday_or_one_per_game_collapsing():
    x=candidate(event="SUNDAY-1")
    y=candidate(event="SUNDAY-2",selection="Player B: Over 30.5 rush yards",key="player_rush_yds")
    z=candidate(event="SUNDAY-1",selection="Player C: Over 55.5 receiving yards",key="player_reception_yds")
    a=all_game_research_board([x,y,z,z],now=NOW)
    assert a["total"]==3 and a["event_count"]==2
    assert a["qualified_pregame"]==3
    assert a["same_game_risks"][0]["Dependent pairs"]==1
    assert a["same_game_risks"][0]["Joint P"].startswith("UNKNOWN")


def test_no_unverified_data_or_game_after_kickoff_promoted():
    x=candidate()
    rows=[
        replace(x,verified_start=None),
        replace(x,verified_start="2026-10-15T16:00:00Z",selection="started"),
        replace(x,model_probability=None,selection="no model"),
        replace(x,kalshi_price=None,selection="missing ask"),
        replace(x,entry_fee_per_contract=None,selection="missing fee"),
        replace(x,execution_audit_status="RESEARCH_ONLY",selection="bad audit"),
        replace(x,kalshi_price=.88,selection="negative EV"),
    ]
    b=all_game_research_board(rows,now=NOW)
    assert b["blocked"]==len(rows) and b["qualified_pregame"]==0
    assert all(r["Status"]=="RESEARCH ONLY" for r in b["rows"])
    assert b["common_blockers"]


def test_same_game_players_and_game_lines_flag_even_different_player():
    a=candidate(key="player_pass_yds",selection="QB Alpha: Over 250 passing yards")
    b=candidate(key="player_reception_yds",selection="WR Beta: Over 80 receiving yards")
    assert same_event_risk(a,b)=="SHARED_GAME_SCRIPT"
    b=replace(b,selection="QB Alpha: Over 100 receiving yards")
    assert same_event_risk(a,b)=="SHARED_PLAYER_USAGE"
    c=candidate(event="SUNDAY-OTHER")
    assert same_event_risk(a,c) is None


def test_joint_scenarios_positive_dependence_not_independence_product():
    a=candidate(selection="QB Alpha",fair=.7,price=.48)
    b=candidate(selection="WR Bravo",fair=.6,price=.42)
    report=evaluate_joint_scenarios([a,b],bundle=bundle([a,b]),now=NOW)
    assert report["joint_model_probability"]==.6
    assert report["independence_benchmark_not_a_joint_forecast"]==.42
    assert report["qualified_for_live"] is False
    assert report["joint_market_cost"] is None
    assert report["cashout_probability"] is None


def test_negative_dependence_reduces_joint_probability():
    a=candidate(selection="QB Alpha",fair=.7)
    b=candidate(selection="WR Bravo",fair=.6)
    res=evaluate_joint_scenarios([a,b],bundle=bundle([a,b],joint=False),now=NOW)
    assert res["joint_model_probability"]==.3


def test_exact_combo_quote_fees_and_staleness():
    a=candidate(selection="QB Alpha",fair=.7)
    b=candidate(selection="WR Bravo",fair=.6)
    quote={
        "source":"KALSHI_EXACT_COMBO_EXECUTABLE_ASK",
        "exact_combo_ticker":"KXMVECROSSCATEGORY-TEST",
        "constituent_contract_sides":[
            f"{a.kalshi_ticker}|{a.kalshi_side}",
            f"{b.kalshi_ticker}|{b.kalshi_side}",
        ],
        "executable_ask_dollars":".52",
        "entry_fee_per_contract":".02",
        "observed_at_utc":NOW.isoformat(),
    }
    res=evaluate_joint_scenarios([a,b],bundle=bundle([a,b]),now=NOW,combo_quote=quote)
    assert res["joint_market_cost"]==.54
    assert res["model_minus_exact_combo_cost"]==.06
    assert res["qualified_for_live"] is False
    for value in [{**quote,"source":"LEG_PRICE_PRODUCT"},
                  {**quote,"constituent_contract_sides":["WRONG|YES",f"{b.kalshi_ticker}|YES"]},
                  {**quote,"observed_at_utc":(NOW-timedelta(minutes=2)).isoformat()},
                  {**quote,"entry_fee_per_contract":None}]:
        with pytest.raises(ValueError):
            evaluate_joint_scenarios([a,b],bundle=bundle([a,b]),now=NOW,combo_quote=value if value is not None else {"source":"bad"})


def test_fail_closed_for_missing_draws_forward_time_or_market_based_scenario():
    a=candidate(selection="QB Alpha",fair=.7)
    b=candidate(selection="WR Bravo",fair=.6)
    base=bundle([a,b])
    invalid=[
        dict(base,source_kind="KALSHI_MARKET_REVERSE_FIT"),
        dict(base,generated_at_utc="2026-10-20T00:00:00Z"),
        dict(base,draws=base["draws"][:15]),
        dict(base,draws=base["draws"][:-1]+[dict(base["draws"][-1],outcomes={f"{a.kalshi_ticker}|YES":True})]),
        dict(base,draws=base["draws"][:-1]+[dict(base["draws"][-1],scenario_id=0)]),
    ]
    for invalid_bundle in invalid:
        with pytest.raises(ValueError):
            evaluate_joint_scenarios([a,b],bundle=invalid_bundle,now=NOW)


def test_marginal_mismatch_and_unreviewed_leg_fail_closed():
    a=candidate(selection="QB Alpha",fair=.7)
    b=candidate(selection="WR Bravo",fair=.6)
    with pytest.raises(ValueError,match="marginals"):
        evaluate_joint_scenarios([replace(a,model_probability=.98),b],
                                 bundle=bundle([a,b]),now=NOW)
    with pytest.raises(ValueError,match="not_execution_qualified"):
        evaluate_joint_scenarios([a,replace(b,execution_audit_status=None)],
                                 bundle=bundle([a,b]),now=NOW)


def test_joint_simulations_span_all_five_sports_without_pairwise_multiplies():
    sports=list(SPORTS)
    legs=[candidate(sport=s,event="GAME-"+s,
                    selection="ATHLETE-"+s,
                    fair=(.7 if i%2==0 else .6),
                    price=.48) for i,s in enumerate(sports)]
    report=evaluate_joint_scenarios(legs,bundle=bundle(legs),now=NOW)
    assert report["sports"]==sorted(sports)
    assert report["distinct_game_count"]==5
    assert report["joint_model_probability"]==.6
    assert report["qualified_for_live"] is False


def test_input_guards_and_not_supported_sports():
    with pytest.raises(ValueError):
        all_game_research_board([],now=datetime(2026,10,15))
    with pytest.raises(ValueError):
        all_game_research_board([],now=NOW,sports=["Tennis"])
    with pytest.raises(ValueError):
        all_game_research_board([],now=NOW,max_preview=101)
    with pytest.raises(ValueError):
        all_game_research_board([candidate()]*501,now=NOW)
    with pytest.raises(ValueError):
        evaluate_joint_scenarios([candidate()],bundle={},now=NOW)
