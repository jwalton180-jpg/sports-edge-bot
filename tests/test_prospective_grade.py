import hashlib
from datetime import datetime,timezone
from sports_edge.models.prospective_capture import PROTOCOL,canonical_bytes
from sports_edge.models.prospective_grade import first_archived_predictions,grade_official,summarize,verify_snapshot

CAPTURE="2026-10-15T12:00:00Z"
def pair(*,side="YES",artifact_time="2026-10-15T12:01:00Z"):
    row={"sport":"NHL","ticker":"KXNHL-G","side":side,"model":"v1",
         "family":"Player Goals","threshold":2,"event_time":"2026-10-15T19:00:00Z",
         "observed_at":CAPTURE,"source_as_of":"2026-10-15T11:59:00Z",
         "probability":.24,"quote":{"ask":.22,"entry_fee_per_contract":.01,
                                    "reasons":[]},"context":{"source_urls":[]}}
    snapshot={"study":{"protocol":PROTOCOL},"sampling":{"protocol":PROTOCOL},"records":[row]}
    manifest={"schema_version":PROTOCOL,"no_promotions":True,"github_run_id":"100",
              "git_sha":"a"*40,"captured_at":CAPTURE,
              "capture_sha256":hashlib.sha256(canonical_bytes(snapshot)).hexdigest()}
    artifact={"id":24,"created_at":artifact_time,
              "workflow_run":{"id":100,"head_sha":"a"*40}}
    return snapshot,manifest,artifact

def test_first_outcome_no_side_or_repeat_inflation():
    rows=first_archived_predictions([pair(artifact_time="2026-10-15T12:02:00Z"),
                                     pair(side="NO"),pair()])
    assert len(rows)==1 and rows[0]["reasons"]==[]
    assert not rows[0]["qualified_for_live"]
def test_prior_protocol_artifact_remains_verifiable():
    p=pair()
    old="SPORTSEDGE-PROSPECTIVE-V2-20261014"
    p[0]["study"]["protocol"]=old
    p[0]["sampling"]["protocol"]=old
    p[1]["schema_version"]=old
    p[1]["capture_sha256"]=hashlib.sha256(canonical_bytes(p[0])).hexdigest()
    assert verify_snapshot(*p).isoformat().startswith("2026-10-15T12:01:00")

def test_artifact_after_game_or_forgery_rejected():
    late=pair(artifact_time="2026-10-15T20:00:00Z")
    try:first_archived_predictions([late])
    except ValueError:pass
    else: assert False
    p=pair();p[1]["capture_sha256"]="z"*64
    try:verify_snapshot(*p)
    except ValueError:pass
    else:assert False
def test_official_final_only_and_no_promotion():
    class Client:
        def __init__(self,m):self.m=m
        def market(self,ticker):
            class Answer:pass
            out=Answer();out.data={"market":self.m};return out
    market={"ticker":"KXNHL-G","status":"settled","result":"yes",
            "settlement_value_dollars":"1",
            "settlement_time":"2026-10-15T22:00:00Z"}
    row=first_archived_predictions([pair()])[0]
    graded=grade_official([row],Client(market),
         now=datetime(2026,10,16,tzinfo=timezone.utc))[0]
    assert graded["won"] is True
    result=summarize([graded])
    assert result["graded_unblocked"]==1 and not result["qualified_for_live"]
    bad=first_archived_predictions([pair()])[0]
    market["settlement_value_dollars"]="0.55"
    graded=grade_official([bad],Client(market),
         now=datetime(2026,10,16,tzinfo=timezone.utc))[0]
    assert graded["won"] is None
