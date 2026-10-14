from sports_edge.models.nhl_multimarket import nhl_player_market,nhl_market_menu
from sports_edge.models.nhl_prop_walkforward import nhl_prop_walkforward

def _logs():
    return [{"source":"NHL","player_id":123,"game_id":str(i),
             "date":f"2025-01-{i+1:02d}","goals":i%5==0,
             "assists":i%3==0,"points":int(i%5==0)+int(i%3==0),
             "shots":i%5+1,"power_play_goals":0} for i in range(25)]

def test_multi_market_menu_and_missing_stat_fail_closed():
    prior=_logs()[:20]
    menu=nhl_market_menu(prior)
    assert len(menu)==12
    assert all(p.probability is not None and p.status=="UNVALIDATED_RESEARCH" for p in menu)
    assert nhl_player_market([dict(r,shots=None) for r in prior],market="shots").probability is None
    assert nhl_player_market(prior,market="shots",threshold=3).probability!=nhl_player_market(prior,market="shots",threshold=4).probability

def test_no_same_day_leakage_and_no_implied_edge():
    prior=_logs()
    a=nhl_prop_walkforward(prior)
    changed=[dict(r,shots=20) if r["date"]=="2025-01-25" else r for r in prior]
    b=nhl_prop_walkforward(changed)
    event="24:123"
    assert [r["probability"] for r in a if r["event_id"]==event]==[r["probability"] for r in b if r["event_id"]==event]
    assert all(r["status"]=="RESEARCH_ONLY" for r in a)
    assert any(r["outcome"]=="player_shots_3_plus" for r in a)
