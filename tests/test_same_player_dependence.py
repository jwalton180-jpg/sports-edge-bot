from dataclasses import replace
from sports_edge.models.parlay_intelligence import build_intelligent_parlay
from sports_edge.models.ticket_dependencies import same_player_cross_market_dependence
from tests.test_parlay_intelligence import leg


def same_event(name, stat, market_key, event="ATL-NO"):
    return replace(leg(event_id=event,selection=name+":"+stat,fair=.78,price=.58),
                   sport="NFL",market_key=market_key,
                   kalshi_ticker="KXNFL-"+event+"-"+market_key,
                   execution_audit_status="QUALIFIED_VALUE",
                   entry_fee_per_contract=.01)


def test_cross_stat_same_player_is_a_dependence_not_two_independent_wins():
    receptions=same_event("Juwan Johnson"," 3+ receptions","player_receptions")
    yards=same_event("Juwan Johnson"," 40+ receiving yards","player_reception_yds")
    other=same_event("Drake London"," 5+ receptions","player_receptions")
    assert same_player_cross_market_dependence(receptions,yards)
    assert not same_player_cross_market_dependence(receptions,other)
    assert not same_player_cross_market_dependence(receptions,replace(yards,event_id="NEXT"))
    assert not same_player_cross_market_dependence(receptions,replace(yards,market_key="player_receptions"))
    result=build_intelligent_parlay([receptions,yards],mode="value",target_legs=2,max_per_event=4)
    assert len(result.legs)==2
    assert result.correlation_risk=="HIGH"
    assert any("same-player, cross-market" in w for w in result.warnings)
    assert any("independence benchmark" in w for w in result.warnings)


def test_player_names_from_other_games_are_never_conflated():
    a=same_event("Brock Purdy"," 150+ passing yards","player_passing_yards")
    b=same_event("Brock Purdy"," 10+ rushing yards","player_rushing_yards",event="OTHER")
    assert not same_player_cross_market_dependence(a,b)
    # Never infer a player identity from 'over 1.5' without a named athlete.
    c=replace(a,selection="Over 150.5 yards")
    assert not same_player_cross_market_dependence(c,replace(b,event_id="ATL-NO"))
