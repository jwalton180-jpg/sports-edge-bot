from dataclasses import replace
from datetime import datetime,timezone,timedelta

from sports_edge.models.tennis_early_research import (
    parse_point_score, point_aware_live_probability, build_early_reversal_watches,
    build_extreme_cheap_observations,
)
from sports_edge.models.tennis_reversal_ledger import TennisSignalLedger,SignalObservation
from sports_edge.data.tennis_live import TennisLiveScoreState
from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.event_identity import canonical_event_id,canonical_participant


NOW=datetime(2026,10,8,18,0,tzinfo=timezone.utc)


def _state(points="30-40"):
    a,b="Guido Ivan Justo","Francisco Comesana"
    return TennisLiveScoreState(
        event_id=canonical_event_id("Tennis",a,b,"2026-10-08"),
        selection_key=canonical_participant("Tennis",a),
        player=a,opponent=b,tour="CHALLENGER",period=3,
        player_sets=1,opponent_sets=1,player_games=1,opponent_games=1,
        lost_first_set=True,won_latest_completed_set=True,
        turnaround=True,deciding_set=True,current_set_lead=0,
        score_label="Justo vs Comesana · 6-7 7-6 1-1",fetched_at=NOW,
        serving=True,net_break_advantage=0,point_score=points,
        score_sources=("Tennis365",),
    )


def _leg():
    s=_state()
    return ParlayCandidateLeg(
        sport="Tennis",event_id=s.event_id,event_title="Justo vs Comesana",
        market_key="model_h2h",market_label="Match Winner",
        selection=s.player,consensus_probability=.38,book_count=0,
        source_age_s=0.,median_odds=None,kalshi_ticker="KX-TEST",
        kalshi_side="YES",kalshi_price=.15,kalshi_edge_points=23.,
        kalshi_status="MODEL",evidence_class="MODEL",
        model_probability=.38,model_confidence=.65,model_name="Tennis adaptive",
        model_sample_size=25,model_reasons=(),model_warnings=(),
    )


def _candles():
    values=[.34,.27,.15,.095,.11,.13]
    out=[]
    for i,p in enumerate(values):
        end=int((NOW-timedelta(minutes=5-i)).timestamp())
        out.append({"end_period_ts":end,"volume_fp":"10",
            "price":{"close_dollars":str(p),"high_dollars":str(p),
                     "low_dollars":str(p)},
            "yes_ask":{"close_dollars":str(p),"high_dollars":str(p),
                       "low_dollars":str(p)},
            "yes_bid":{"close_dollars":str(max(.01,p-.02)),
                       "high_dollars":str(max(.01,p-.02)),
                       "low_dollars":str(max(.01,p-.02))}})
    return out


def test_point_state_parser_guards_invalid_scores():
    assert parse_point_score("30-40")== (2,3)
    assert parse_point_score("40-AD")== (3,4)
    assert parse_point_score("garbage") is None
    assert parse_point_score("6-5") is None


def test_point_aware_research_moves_in_expected_direction():
    down=point_aware_live_probability(.38,_state("0-40"))
    up=point_aware_live_probability(.38,_state("40-0"))
    assert down is not None and up is not None
    assert up>down


def test_no_point_awareness_when_server_unknown_or_conflict():
    s=_state()
    assert point_aware_live_probability(.38,replace(s,serving=None)) is None
    assert point_aware_live_probability(.38,replace(s,score_conflict=True)) is None
    assert point_aware_live_probability(.38,replace(s,at_tiebreak=True)) is None


def test_early_watch_stays_research_and_requires_quoted_recent_market():
    s=_state("30-40")
    rows=build_early_reversal_watches([_leg()],{"KX-TEST":_candles()},
        {(s.event_id,s.selection_key):s},now=NOW)
    assert len(rows)==1
    assert rows[0].status=="EARLY WATCH — RESEARCH"
    assert rows[0].price<=.20
    assert rows[0].point_aware
    assert build_early_reversal_watches([_leg()],{"KX-TEST":_candles()},
        {(s.event_id,s.selection_key):replace(s,score_conflict=True)},now=NOW)==()


def test_ledger_freezes_first_trigger_and_grades_once(tmp_path):
    ledger=TennisSignalLedger(tmp_path/"ledger.sqlite")
    obs=SignalObservation("event","ticker","Justo","YES",
        "EARLY WATCH — RESEARCH","version1",NOW,.13,.39,
        "6-7 7-6 1-1",{"point_aware":True})
    key,created=ledger.record_first(obs)
    assert created
    _,created2=ledger.record_first(replace(obs,entry_ask=.40,model_fair=.90))
    assert not created2
    assert ledger.snapshot()[0]["entry_ask"]==.13
    assert ledger.settle_verified(key,"WIN",source="Kalshi official",settled_at=NOW+timedelta(hours=1))
    assert not ledger.settle_verified(key,"LOSS",source="Kalshi official",settled_at=NOW+timedelta(hours=2))
    assert ledger.metrics()["wins"]==1
    assert ledger.metrics()["settled"]==1



def _shi_scenario(price: float = .03):
    """Synthetic stress case; screenshot's selected 3c chart time and live 2–6, 3–5 score are not proven simultaneous."""
    state = _state()
    state = replace(
        state,
        event_id=canonical_event_id("Tennis", "Han Shi", "Peyton Stearns", "2026-10-09"),
        selection_key=canonical_participant("Tennis", "Han Shi"),
        player="Han Shi", opponent="Peyton Stearns", tour="WTA",
        period=2, player_sets=0, opponent_sets=1,
        player_games=3, opponent_games=5,
        lost_first_set=True, won_latest_completed_set=False,
        turnaround=False, deciding_set=False,
        current_set_lead=-2, serving=False,
        net_break_advantage=-1, point_score=None,
        score_sources=("ESPN",),
        score_label="Han Shi vs Peyton Stearns · 2-6 · 3-5",
    )
    leg = replace(
        _leg(), event_id=state.event_id, selection="Han Shi",
        model_probability=.416, model_confidence=.88,
        kalshi_ticker="KXWTAMATCH-26OCT09STESHI-SHI",
    )
    quoted=[.32,.19,.10,.05,price]
    candles=[]
    for i,p in enumerate(quoted):
        timestamp=int((NOW-timedelta(minutes=4-i)).timestamp())
        candles.append({
            "end_period_ts": timestamp, "volume_fp":"12",
            "price":{"close_dollars":str(p),"low_dollars":str(p),"high_dollars":str(p)},
            "yes_ask":{"close_dollars":str(p),"low_dollars":str(p),"high_dollars":str(p)},
            "yes_bid":{"close_dollars":str(max(.01,p-.01)),
                       "low_dollars":str(max(.01,p-.01)),
                       "high_dollars":str(max(.01,p-.01))},
        })
    return leg,state,candles


def test_stearns_shi_three_cent_snapshot_is_tracked_not_misrepresented_as_pick():
    leg,state,history=_shi_scenario(.03)
    index={(state.event_id,state.selection_key):state}
    rows=build_extreme_cheap_observations(
        [leg], {leg.kalshi_ticker:history}, index, now=NOW
    )
    assert len(rows)==1
    assert rows[0].current_ask==.03
    assert rows[0].observed_trough==.03
    assert rows[0].lane=="EXTREME DIP — TRACKING ONLY"
    assert rows[0].score_sources==("ESPN",)
    assert build_early_reversal_watches(
        [leg], {leg.kalshi_ticker:history}, index, now=NOW
    )==()


def test_extreme_dip_requires_fresh_two_sided_quote_and_structural_score():
    leg,state,history=_shi_scenario(.03)
    key={(state.event_id,state.selection_key):state}
    stale=replace(state,fetched_at=NOW-timedelta(minutes=10))
    assert not build_extreme_cheap_observations(
        [leg], {leg.kalshi_ticker:history},
        {(state.event_id,state.selection_key):stale}, now=NOW
    )
    assert not build_extreme_cheap_observations(
        [leg], {leg.kalshi_ticker:history},
        {(state.event_id,state.selection_key):replace(state,score_conflict=True)}, now=NOW
    )
    no_bid=[{**x,"yes_bid":{}} for x in history]
    assert not build_extreme_cheap_observations(
        [leg], {leg.kalshi_ticker:no_bid}, key, now=NOW
    )


def test_extreme_dip_can_transition_to_recovery_research_without_promoting_deep():
    leg,state,history=_shi_scenario(.03)
    # Independent synthetic stronger-score scenario, not the user's actual match.
    state=replace(state,player_sets=1,opponent_sets=1,
                  period=3,player_games=3,opponent_games=3,
                  current_set_lead=0,serving=True,net_break_advantage=0,
                  deciding_set=True,turnaround=True)
    leg=replace(leg,model_probability=.55)
    history[-1]["end_period_ts"]=int((NOW-timedelta(minutes=1)).timestamp())
    end=int(NOW.timestamp())
    history.append({
        "end_period_ts":end,"volume_fp":"12",
        "price":{"close_dollars":".08","low_dollars":".08","high_dollars":".08"},
        "yes_ask":{"close_dollars":".08","low_dollars":".08","high_dollars":".08"},
        "yes_bid":{"close_dollars":".07","low_dollars":".07","high_dollars":".07"},
    })
    rows=build_extreme_cheap_observations(
        [leg], {leg.kalshi_ticker:history},
        {(state.event_id,state.selection_key):state},now=NOW,
    )
    assert len(rows)==1
    assert rows[0].lane=="RECOVERY BUILDING — RESEARCH"
    assert rows[0].rebound_pp>=4.
