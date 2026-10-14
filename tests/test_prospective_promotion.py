from datetime import date,timedelta
from sports_edge.models.prospective_promotion import assess_prospective_promotion

def rows(n=320):
    d=date(2024,1,1)
    out=[]
    for i in range(n):
        day=d+timedelta(days=i)
        out.append({"evaluation_kind":"prospective","status":"IMMUTABLE_LOCK",
                    "prediction_id":str(i),"source_id":"official:"+str(i),
                    "settlement_source_id":"official-final:"+str(i),
                    "model_version":"v1","event_id":str(i),"sport":"Soccer",
                    "outcome":"2+ goals","locked_at":day.isoformat()+"T10:00:00Z",
                    "source_as_of":day.isoformat()+"T09:00:00Z",
                    "event_time":day.isoformat()+"T12:00:00Z",
                    "settled_at":day.isoformat()+"T16:00:00Z",
                    "model_probability":.5,"baseline_probability":.5,
                    "won":bool(i%2),"settlement_verified":True,"source_verified":True})
    return out

def test_rejects_historical_or_noncausal_records():
    r=rows()
    r[0]["evaluation_kind"]="historical_walkforward"
    assert not assess_prospective_promotion(r)["qualified"]
    r=rows()
    r[0]["source_as_of"]=r[0]["settled_at"]
    assert not assess_prospective_promotion(r)["qualified"]

def test_does_not_promote_without_significant_independent_improvement():
    r=assess_prospective_promotion(rows())
    assert r["graded"]==320
    assert not r["qualified"]
    assert r["paired_brier_improvement_lower95"]==0

def test_rejects_missing_verified_settlement_or_duplicate():
    r=rows()
    r[0]["settlement_verified"]=False
    assert not assess_prospective_promotion(r)["qualified"]
    r=rows()
    r[0]["prediction_id"]=r[1]["prediction_id"]
    # Duplicate IDs are invalid even when event IDs differ.
    assert not assess_prospective_promotion(r)["qualified"]
