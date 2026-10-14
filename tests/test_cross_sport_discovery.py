from sports_edge.models.kalshi_sports import classify_kalshi_market, group_kalshi_sports
from sports_edge.models.value_audit import audit_value

def test_new_sports_discovered_without_inventing_models():
    cases = [("KXNHLGAME", "NHL", "Moneyline"), ("KXNCAAFSPREAD", "NCAAF", "Spread"),
             ("KXEPLTOTAL", "Soccer", "Game Total")]
    rows = []
    for series, sport, family in cases:
        row = {"ticker":series+"-TEST", "series_ticker":series, "status":"open"}
        assert classify_kalshi_market(row) == (sport, family)
        rows.append(row)
    grouped = group_kalshi_sports(rows)
    assert all(len(grouped[sport]) == 1 for _, sport, _ in cases)
    for row in rows:
        decision = audit_value(row, side="YES", independent_probability=None,
                               model_validated=False, evidence_fresh=False,
                               coverage_complete=False, rules_verified=False)
        assert decision.status == "RESEARCH_ONLY"

def test_unrecognized_series_not_misclassified():
    assert classify_kalshi_market({"ticker":"KXUNKNOWN-ABC", "title":"Unrelated event"}) is None
