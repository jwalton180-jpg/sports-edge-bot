import json
from sports_edge.models.prospective_review import prospective_review_export

def make_record(side="YES", observed="2026-10-01T10:00:00Z"):
    return {
        "sport":"NHL","family":"Player Points","threshold":2,"side":side,
        "ticker":"KXNHLPTS-G-2","model":"nhl_points_dispersion_v2",
        "observed_at":observed,"event_time":"2026-10-01T18:00:00Z",
        "source_as_of":"2026-09-29T00:00:00Z","probability":0.3,
        "quote":{"ask":0.2,"entry_fee_per_contract":0.01,"reasons":[]},
        "context":{"event_id":"G","source_urls":["https://api-web.nhle.com/v1"]},
    }
def ledger(lines):
    return "".join(json.dumps(x)+"\n" for x in lines)
def scan(scan_id,rows):
    return {"id":scan_id,"kind":"scan","payload":{"records":rows,
        "coverage":{"scan_complete":True,"catalog_complete":True}}}
def settlement(scan_id, result="yes", verified=True):
    return {"id":"S-"+scan_id,"kind":"settlement",
        "payload":{"scan_id":scan_id,"ticker":"KXNHLPTS-G-2",
                   "result":result,"settlement_verified":verified,
                   "settled_at":"2026-10-01T22:00:00Z",
                   "source_url":"https://external-api.kalshi.com/trade-api/v2/markets/KXNHLPTS-G-2"}}

def test_first_prediction_only_no_side_or_rescan_inflation():
    records=[scan("new",[make_record(observed="2026-10-01T11:00:00Z")]),
             scan("old",[make_record(),make_record(side="NO")]),
             settlement("old")]
    result=prospective_review_export(ledger(records))
    assert result["coverage"]["canonical_predictions"]==1
    assert result["coverage"]["repeat_scans_excluded"]==1
    assert result["coverage"]["officially_graded"]==1
    assert result["records"][0]["scan_id"]=="old"
    assert result["records"][0]["won"] is True
    assert result["records"][0]["baseline_probability"] is None
    assert result["qualified_for_live"] is False
    assert result["records"][0]["source_verified"] is False

def test_void_and_unverified_settlements_do_not_become_wins():
    for result,verified in [("void",True),("yes",False),("fair_price",True)]:
        r=prospective_review_export(ledger([scan("old",[make_record()]),settlement("old",result,verified)]))
        assert r["coverage"]["officially_graded"]==0
        assert r["records"][0]["won"] is None

def test_post_event_and_missing_source_are_flagged():
    r=prospective_review_export(ledger([scan("old",[dict(make_record(),
        source_as_of=None,observed_at="2026-10-02T10:00:00Z")])]))
    reasons=r["records"][0]["reasons"]
    assert "not_recorded_before_event" in reasons
    assert "source_timestamp_unverified" in reasons
    assert r["records"][0]["promotion_eligible"] is False

def test_conflicting_settlements_and_unknown_records_are_rejected():
    try:
        prospective_review_export(ledger([scan("old",[make_record()]),settlement("old"),
            dict(settlement("old"),id="other")]))
    except ValueError:
        pass
    else:
        assert False
    try:
        prospective_review_export(ledger([{"id":"x","kind":"delete","payload":{}}]))
    except ValueError:
        pass
    else:
        assert False
