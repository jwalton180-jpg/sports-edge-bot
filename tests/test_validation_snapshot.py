from sports_edge.data.validation_snapshot import HISTORICAL_RESEARCH_SNAPSHOT

def test_verified_research_snapshot_never_claims_live_qualification():
    assert set(HISTORICAL_RESEARCH_SNAPSHOT)=={"NHL","NCAAF","Soccer"}
    for sport, row in HISTORICAL_RESEARCH_SNAPSHOT.items():
        assert row["status"]=="RESEARCH_ONLY"
        assert row["holdout_predictions"] > 0
        assert 0 <= row["holdout_brier"] <= 1
        assert 0 <= row["holdout_ece"] <= 1
    assert HISTORICAL_RESEARCH_SNAPSHOT["NHL"]["historical_yes_calls"]==0
    assert "EPL" in HISTORICAL_RESEARCH_SNAPSHOT["Soccer"]["market"]
