from sports_edge.data.historical_sport_evidence import (
    completed_team_games, historical_nhl_projection,
    historical_soccer_projection, historical_ncaaf_projection)
from sports_edge.models.sport_holdout_validation import assess_holdout
import pytest

def test_nhl_history_is_strictly_prior_and_no_leakage():
    logs=tuple({"source":"NHL","game_id":str(i),"date":f"2026-09-{i+1:02d}","goals":1} for i in range(20))
    logs+=({"source":"NHL","game_id":"99","date":"2026-10-15","goals":10},)
    p=historical_nhl_projection(logs,before_date="2026-10-01")
    assert p.sample_size==20 and p.status=="UNVALIDATED_RESEARCH"
    assert historical_nhl_projection(logs,before_date="2026-09-10").probability is None

def test_soccer_model_from_prior_results_only():
    games=[]
    for i in range(12):
        games.append({"source":"ESPN","sport":"Soccer","event_time":f"2026-09-{i+1:02d}T10:00:00+00:00",
                      "home_id":"A","away_id":"B","home_score":2,"away_score":1})
    games.append({**games[0],"event_time":"2026-11-01T10:00:00+00:00","home_score":20})
    p=historical_soccer_projection(tuple(games),home_id="A",away_id="B",
                                   before_time="2026-10-01T00:00:00Z")
    assert p.sample_size==12 and p.status=="UNVALIDATED_RESEARCH"

def test_ncaaf_missing_upset_review_fails_closed():
    games=tuple({"source":"ESPN","sport":"NCAAF",
                 "event_time":f"2026-09-{i+1:02d}T10:00:00Z",
                 "home_id":"A","away_id":"B","home_score":21,"away_score":14}
                for i in range(7))
    p=historical_ncaaf_projection(games,favorite_id="A",underdog_id="B",
         before_time="2026-10-01T00:00:00Z",favorite_home=True,upset_red_flags=None)
    assert p.probability is None

def test_holdout_requires_causal_prospective_evidence():
    r={"sport":"NHL","event_id":"1","outcome":"2+","source_id":"nhl",
       "model_version":"v1","event_time":"2026-10-15T10:00:00Z",
       "predicted_at":"2026-10-14T10:00:00Z","probability":.7,"won":True}
    assert assess_holdout([r])["graded"]==1
    assert not assess_holdout([r])["qualified"]
    assert assess_holdout([{**r,"predicted_at":"2026-10-16T10:00:00Z"}])["graded"]==0
    assert assess_holdout([r,r])["graded"]==1

def test_espn_source_rejects_incomplete(monkeypatch):
    from sports_edge.data import historical_sport_evidence as h
    monkeypatch.setattr(h,"_get_json",lambda *args,**kwargs:{"events":[
      {"id":"a","date":"2026-09-10T10:00:00Z",
       "status":{"type":{"completed":False}},"competitions":[]},
      {"id":"b","date":"2026-09-10T10:00:00Z",
       "status":{"type":{"completed":True}},"competitions":[{"competitors":[
          {"homeAway":"home","score":"2","team":{"id":"1"}},
          {"homeAway":"away","score":"1","team":{"id":"2"}}]}]}]})
    assert len(h.completed_team_games("Soccer","20260910"))==1
