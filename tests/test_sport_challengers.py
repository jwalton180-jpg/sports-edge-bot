from sports_edge.models.sport_challengers import nhl_shot_challenger
from sports_edge.models.team_challengers import ncaaf_opponent_challenger,soccer_shrink_challenger
from sports_edge.models.challenger_evaluation import matched_holdout

def test_shot_challenger_rejects_missing_shots():
    games=[{"source":"NHL","goals":0,"shots":3} for _ in range(30)]
    p=nhl_shot_challenger(games)
    assert p.probability is not None and 0<p.probability<1
    assert nhl_shot_challenger([{"source":"NHL","goals":0}]*30).probability is None

def test_opponent_model_requires_strict_prior_games():
    games=[{"source":"ESPN","sport":"NCAAF","home_id":"A","away_id":"B",
            "home_score":35,"away_score":17,"event_time":"2025-09-01T12:00:00Z"}]*6
    assert ncaaf_opponent_challenger(games,home_id="A",away_id="B",
                                     before_time="2025-09-03T12:00:00Z").probability is not None
    assert ncaaf_opponent_challenger(games,home_id="A",away_id="B",
                                     before_time="2025-09-01T12:00:00Z").probability is None

def test_soccer_no_target_match_leakage():
    games=[{"source":"ESPN","sport":"Soccer","home_id":"A","away_id":"B",
            "home_score":2,"away_score":1,"event_time":"2025-01-01T12:00:00Z"}]*12
    before="2025-01-04T12:00:00Z"
    p=soccer_shrink_challenger(games,home_id="A",away_id="B",before_time=before)
    assert p.probability is not None
    future=[dict(g,event_time=before,home_score=99) for g in games]
    q=soccer_shrink_challenger(games+future,home_id="A",away_id="B",before_time=before)
    assert p.probability==q.probability

def test_matched_holdout_does_not_claim_live_promotion():
    assert matched_holdout([],[])["qualified_for_live"] is False
