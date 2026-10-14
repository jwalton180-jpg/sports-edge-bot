import pytest
from sports_edge.models.sport_walkforward import team_walkforward, nhl_player_walkforward, calibration_report

def test_soccer_strict_pre_event_only_and_research_status():
    games=[]
    for i in range(35):
        games.append({"sport":"Soccer","source":"ESPN","event_id":str(i),
          "event_time":f"2025-01-{i+1:02d}T12:00:00Z" if i<28 else f"2025-02-{i-27:02d}T12:00:00Z",
          "home_id":"A","away_id":"B","home_score":2 if i%2 else 0,"away_score":1})
    rows=team_walkforward(games,sport="Soccer")
    assert rows and all(r["status"]=="RESEARCH_ONLY" for r in rows)
    assert all(r["predicted_at"] < r["event_time"] for r in rows)
    report=calibration_report(rows)
    assert report["Soccer"]["qualified_for_live"] is False

def test_ncaaf_missing_early_history_fails_closed():
    games=[{"sport":"NCAAF","source":"ESPN","event_id":str(i),
      "event_time":f"2025-10-{i+1:02d}T12:00:00Z","home_id":"A","away_id":"B",
      "home_score":20+i,"away_score":14} for i in range(10)]
    rows=team_walkforward(games,sport="NCAAF")
    assert rows
    assert all(r["predicted_at"] < r["event_time"] for r in rows)

def test_nhl_prior_games_only():
    log=[{"source":"NHL","player_id":10,"game_id":str(i),
          "date":f"2025-01-{i+1:02d}","goals":int(i%4==0)} for i in range(25)]
    rows=nhl_player_walkforward(log)
    assert len(rows)==5
    assert all(r["status"]=="RESEARCH_ONLY" for r in rows)

def test_no_retroactive_promotion_or_fake_data():
    with pytest.raises(ValueError):
        calibration_report([{"sport":"NHL","event_id":"1","outcome":"goals",
          "probability":.9,"won":True,"event_time":"2025-01-02T00:00:00Z",
          "predicted_at":"2025-01-01T00:00:00Z","evaluation_kind":"prospective",
          "status":"QUALIFIED_VALUE"}])
    with pytest.raises(ValueError):
        team_walkforward([{"sport":"Soccer","source":"ESPN","event_id":"1",
          "event_time":"2025-01-01T00:00:00Z","home_id":"A","away_id":"B",
          "home_score":2,"away_score":1}]*2,sport="Soccer")
