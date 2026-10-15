from dataclasses import replace
from datetime import date,datetime,timezone,timedelta
import pytest

from sports_edge.models.tnf_research import (
    kickoff,market_group,thursday_research_board
)
from tests.test_parlay_intelligence import leg

KICK="2026-10-16T00:15:00Z"
NOW=datetime(2026,10,15,18,0,tzinfo=timezone.utc)
THU=date(2026,10,15)


def nfl(selection="Jaxon Smith-Njigba: Over 4.5 receptions",
        key="player_receptions",fair=.68,price=.53,**kwargs):
    l=replace(leg(sport="NFL",event_id="SEADEN",selection=selection,
        fair=fair,price=price,books=0,model_conf=.77),
        event_title="Seattle Seahawks @ Denver Broncos",
        market_key=key,verified_start=KICK,
        execution_audit_status="QUALIFIED_VALUE",entry_fee_per_contract=.01,
        validation_reviewed=True,kalshi_ticker="KXNFLPROP-26OCT15-"+key)
    return replace(l,**kwargs)


def test_verified_thursday_kickoff_and_audited_legs_only():
    good=nfl()
    board=thursday_research_board([good],local_date=THU,now=NOW)
    assert board["scope"]=="thursday_pregame_model_only"
    assert board["qualified"]==1 and board["blocked"]==0
    assert board["research"][0]["Status"]=="PREGAME QUALIFIED"
    assert board["research"][0]["Model-ask-fee (pp)"]==14
    assert board["not_a_parlay_probability"] and board["cashout_probability_not_calibrated"]


def test_qualifying_does_not_follow_popular_player_or_missing_start_or_market_evidence():
    base=nfl()
    rows=[
        replace(base,selection="Public social tip",verified_start=None),
        replace(base,selection="No independent model",model_probability=None),
        replace(base,selection="No Kalshi ask",kalshi_price=None),
        replace(base,selection="No verified fee",entry_fee_per_contract=None),
        replace(base,selection="Blocked injury status",
                execution_audit_status="RESEARCH_ONLY",
                execution_audit_reasons=("confirmed_starters_and_injuries",)),
        replace(base,selection="High price no positive value",kalshi_price=.75),
    ]
    board=thursday_research_board(rows,local_date=THU,now=NOW)
    assert board["qualified"]==0
    assert board["blocked"]==len(rows)
    assert board["kickoff_unverified"]==1
    assert board["role_uncertain"]>=1
    assert all(r["Status"]=="RESEARCH ONLY" for r in board["research"])


def test_wrong_day_wrong_time_and_started_game_never_show_as_pregame_qualified():
    thursday=nfl()
    sunday=replace(thursday,verified_start="2026-10-18T20:25:00Z")
    wednesday=replace(thursday,verified_start="2026-10-15T00:15:00Z")
    board=thursday_research_board([sunday,wednesday,thursday],
                                  local_date=THU,now=NOW)
    assert board["total"]==1 and board["qualified"]==1
    started=thursday_research_board([thursday],local_date=THU,
                                   now=datetime(2026,10,16,1,tzinfo=timezone.utc))
    assert started["qualified"]==0 and started["already_started"]==1
    assert thursday_research_board([thursday],local_date=date(2026,10,16),
                                    now=NOW)["scope"]=="not_thursday"
    assert kickoff(replace(thursday,verified_start="2026-10-15T20:15:00")) is None


def test_same_player_receptions_receiving_yds_warn_high_correlation():
    a=nfl()
    b=nfl("Jaxon Smith-Njigba: Over 55.5 receiving yards",
          key="player_reception_yds",fair=.63,price=.50)
    board=thursday_research_board([a,b],local_date=THU,now=NOW)
    assert board["qualified"]==2
    assert board["dependence_pairs"]==1
    assert market_group(a)=="Receiving / receptions"


def test_two_touchdown_player_market_is_included_but_not_invented():
    a=nfl("Jaxon Smith-Njigba: 2+ touchdowns",key="player_anytime_td",
          fair=.13,price=.12)
    board=thursday_research_board([a],local_date=THU,now=NOW)
    assert board["research"][0]["Category"]=="Touchdown scorer / 2+ TD"
    assert board["qualified"]==0
    assert board["research"][0]["Status"]=="RESEARCH ONLY"


def test_duplicate_tickers_bounded_preview_and_timezone_guards():
    a=nfl()
    x=thursday_research_board([a,a],local_date=THU,now=NOW,max_rows=2)
    assert x["total"]==1
    with pytest.raises(ValueError):
        thursday_research_board([],local_date=THU,now=NOW,max_rows=0)
    with pytest.raises(ValueError):
        thursday_research_board([],local_date=THU,now=datetime(2026,10,15))
