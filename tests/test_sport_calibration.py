import pytest
from datetime import date,timedelta
from sports_edge.models.sport_calibration import fit_platt, chronological_calibration

def make_rows(n=240):
    start=date(2024,1,1)
    return [{"sport":"Soccer","event_id":str(i),"event_time":(start+timedelta(days=i)).isoformat()+"T15:00:00Z",
             "predicted_at":(start+timedelta(days=i-1)).isoformat()+"T15:00:00Z",
             "outcome":"total_goals_2_plus","source_id":"ESPN:"+str(i),
             "model_version":"poisson","probability":.8 if i%2 else .3,
             "won":bool(i%3), "evaluation_kind":"historical_walkforward",
             "status":"RESEARCH_ONLY"} for i in range(n)]

def test_frozen_temporal_split_and_no_live_promotion():
    result=chronological_calibration(make_rows())
    assert result["status"]=="HISTORICAL_HOLDOUT_ONLY"
    assert result["train_cutoff"] < result["test_start"]
    assert result["test_n"] >= 50
    assert result["qualified_for_live"] is False
    assert 0 <= result["calibrated"]["brier"] <= 1

def test_not_enough_history_is_not_validated():
    assert chronological_calibration(make_rows(60))["qualified_for_live"] is False

def test_invalid_train_rows_fail():
    rows=make_rows()
    rows[0]["probability"]=float("nan")
    with pytest.raises(ValueError):
        fit_platt(rows[:150])
