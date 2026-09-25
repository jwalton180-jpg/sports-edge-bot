import pytest
from sports_edge.data.kalshi_resolution import KalshiResolutionError, resolve_event_ticker, resolve_unique_milestone_id, resolve_market_context

def test_resolves_nested_market_and_unique_milestone():
    market={"ticker":"KXGAME-A"}
    events={"events":[{"event_ticker":"KXGAME","markets":[{"ticker":"KXGAME-A"}]}]}
    live={"data":{"milestones":[{"milestone_id":"ms-123","status":"active"}]}}
    c=resolve_market_context(market,events,live)
    assert (c.event_ticker,c.milestone_id)==("KXGAME","ms-123")

def test_explicit_event_metadata_is_supported_and_conflict_fails_closed():
    events={"events":[{"event_ticker":"EV1","markets":[{"ticker":"M1"}]}]}
    assert resolve_event_ticker({"ticker":"M1","event_ticker":"EV1"},events)=="EV1"
    with pytest.raises(KalshiResolutionError, match="conflicts"):
        resolve_event_ticker({"ticker":"M1","event_ticker":"EV2"},events)

def test_duplicate_event_membership_fails_closed():
    events={"events":[{"event_ticker":"EV1","markets":[{"ticker":"M1"}]},{"event_ticker":"EV2","markets":[{"ticker":"M1"}]}]}
    with pytest.raises(KalshiResolutionError, match="multiple"):
        resolve_event_ticker({"ticker":"M1"},events)

def test_milestone_missing_or_ambiguous_fails_closed():
    with pytest.raises(KalshiResolutionError, match="missing"):
        resolve_unique_milestone_id({"data":{"status":"scheduled"}})
    with pytest.raises(KalshiResolutionError, match="ambiguous"):
        resolve_unique_milestone_id({"milestones":[{"milestone_id":"a"},{"milestone_id":"b"}]})
