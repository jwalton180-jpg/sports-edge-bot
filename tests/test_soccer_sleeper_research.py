from sports_edge.models.soccer_sleeper_research import soccer_market_projection

def games():
    return [{"source":"ESPN","sport":"Soccer","event_id":str(i),
             "home_id":"A","away_id":"B","home_score":2,"away_score":1,
             "event_time":f"2025-01-{i+1:02d}T12:00:00Z"} for i in range(12)]

def test_market_families_have_probabilities():
    for market,threshold in (("both_teams_score",None),("total_goals",2),
                             ("home_goals",1),("away_goals",1)):
        p=soccer_market_projection(games(),home_id="A",away_id="B",
                                    before_time="2025-02-01T12:00:00Z",
                                    market=market,threshold=threshold)
        assert p.probability is not None and 0<p.probability<1
        assert p.status=="UNVALIDATED_RESEARCH"

def test_target_day_does_not_leak():
    base=games()
    a=soccer_market_projection(base,home_id="A",away_id="B",
                               before_time="2025-02-01T12:00:00Z",
                               market="both_teams_score")
    future=[dict(g,event_time="2025-02-01T12:00:00Z",home_score=50) for g in base]
    b=soccer_market_projection(base+future,home_id="A",away_id="B",
                               before_time="2025-02-01T12:00:00Z",
                               market="both_teams_score")
    assert a.probability==b.probability

def test_missing_history_fails_closed():
    p=soccer_market_projection([],home_id="A",away_id="B",
                               before_time="2025-02-01T12:00:00Z",
                               market="total_goals",threshold=2)
    assert p.probability is None
