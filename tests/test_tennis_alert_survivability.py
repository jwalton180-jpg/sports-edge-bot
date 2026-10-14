from dataclasses import replace
from datetime import datetime, timedelta, timezone

from sports_edge.data.tennis_live import TennisLiveScoreState
from sports_edge.models.tennis_alert_survivability import (
    survivability_reason,qualified_tickers,
)
from sports_edge.models.event_identity import (
    canonical_event_id, canonical_participant,
)

NOW=datetime(2026,10,14,5,0,tzinfo=timezone.utc)


def state(*, player_sets=1,opponent_sets=1,player_games=3,opponent_games=0,
          break_adv=1, score_sources=("ESPN",)):
    a,b="Maja Chwalinska","Marie Bouzkova"
    return TennisLiveScoreState(
        event_id=canonical_event_id("Tennis",a,b,"2026-10-14"),
        selection_key=canonical_participant("Tennis",a),
        player=a,opponent=b,tour="WTA",period=3,
        player_sets=player_sets,opponent_sets=opponent_sets,
        player_games=player_games,opponent_games=opponent_games,
        lost_first_set=True,won_latest_completed_set=True,
        turnaround=True,deciding_set=True,current_set_lead=player_games-opponent_games,
        score_label="Chwalinska–Bouzkova 1-6 6-1 3-0",fetched_at=NOW,
        serving=True,net_break_advantage=break_adv,point_score="15-0",
        score_sources=score_sources,
    )


def markets():
    return [
        {"ticker":"KX-WTA-CHW","event_ticker":"KX-WTA-MATCH",
         "yes_sub_title":"Maja Chwalinska","status":"active"},
        {"ticker":"KX-WTA-BOU","event_ticker":"KX-WTA-MATCH",
         "yes_sub_title":"Marie Bouzkova","status":"active"},
    ]


def test_chwalinska_deciding_set_lead_remains_eligible():
    s=state()
    assert survivability_reason(s,now=NOW)=="eligible"
    allowed, reasons=qualified_tickers(markets(),[s],now=NOW)
    assert "KX-WTA-CHW" in allowed
    assert "KX-WTA-BOU" not in allowed
    assert reasons["unique_live_score_not_found"]==1


def test_straight_sets_nearly_over_no_push():
    losing=state(player_sets=0,opponent_sets=1,player_games=1,opponent_games=5)
    assert survivability_reason(losing,now=NOW)=="near_game_or_match_loss"
    assert "KX-WTA-CHW" not in qualified_tickers(markets(),[losing],now=NOW)[0]
    trailing=state(player_sets=0,opponent_sets=1,player_games=3,opponent_games=3,break_adv=0)
    assert survivability_reason(trailing,now=NOW)=="elimination_set_without_control"
    recovered=state(player_sets=0,opponent_sets=1,player_games=4,opponent_games=1,break_adv=1)
    assert survivability_reason(recovered,now=NOW)=="eligible"


def test_final_set_behind_even_after_previous_recovery_is_rejected():
    behind=state(player_sets=1,opponent_sets=1,player_games=2,opponent_games=5,break_adv=-1)
    assert survivability_reason(behind,now=NOW)=="near_game_or_match_loss"
    tied=state(player_sets=1,opponent_sets=1,player_games=1,opponent_games=1,break_adv=0)
    assert survivability_reason(tied,now=NOW)=="no_deciding_set_control"


def test_missing_stale_conflicting_ambiguous_and_completed_scores_fail_closed():
    good=state()
    for bad in (
        None,
        replace(good,score_sources=()),
        replace(good,score_conflict=True),
        replace(good,fetched_at=NOW-timedelta(minutes=4)),
        replace(good,player_games=None),
        replace(good,at_tiebreak=True),
        replace(good,player_sets=2),
    ):
        assert survivability_reason(bad,now=NOW)!="eligible"
    allowed,reasons=qualified_tickers(markets(),[],now=NOW)
    assert allowed=={}
    assert reasons["unique_live_score_not_found"]==2
    duplicate=replace(good,event_id=canonical_event_id(
        "Tennis",good.player,good.opponent,"2026-10-13"))
    assert qualified_tickers(markets(),[good,duplicate],now=NOW)[0]=={}


def test_market_pair_must_have_both_players_and_match_score_names():
    good=state()
    one_side=markets()[:1]
    assert qualified_tickers(one_side,[good],now=NOW)[0]=={}
    wrong_pair=[{**m,"yes_sub_title":("Other Player" if i else "Maja Chwalinska")}
                for i,m in enumerate(markets())]
    assert qualified_tickers(wrong_pair,[good],now=NOW)[0]=={}
