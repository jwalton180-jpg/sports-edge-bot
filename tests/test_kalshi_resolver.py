from sports_edge.models.kalshi_resolver import resolve_sports_live_binding


def milestone(mid="m1", event="KXNFL-GAME"):
    return {
        "id": mid,
        "type": "football_game",
        "title": "Chiefs at Bills",
        "primary_event_tickers": [event],
        "related_event_tickers": [],
    }


def test_resolver_accepts_one_explicit_link():
    market = {"ticker": "KXNFL-GAME-YES", "event_ticker": "KXNFL-GAME"}
    event = {"event_ticker": "KXNFL-GAME", "milestones": [milestone()]}
    binding = resolve_sports_live_binding(market, event=event)
    assert binding is not None
    assert binding.event_ticker == "KXNFL-GAME"
    assert binding.milestone_id == "m1"
    assert binding.source == "event"


def test_resolver_merges_same_milestone_from_event_and_lookup():
    market = {"event_ticker": "KXNFL-GAME"}
    event = {"event_ticker": "KXNFL-GAME", "milestones": [milestone()]}
    binding = resolve_sports_live_binding(market, event=event, milestones=[milestone()])
    assert binding is not None
    assert binding.source == "event+lookup"


def test_resolver_fails_closed_on_conflicting_event_identity():
    market = {"event_ticker": "KXNFL-A"}
    event = {"event_ticker": "KXNFL-B", "milestones": [milestone(event="KXNFL-B")]}
    assert resolve_sports_live_binding(market, event=event) is None


def test_resolver_fails_closed_on_ambiguous_milestones():
    market = {"event_ticker": "KXNFL-GAME"}
    rows = [milestone("m1"), milestone("m2")]
    assert resolve_sports_live_binding(market, milestones=rows) is None


def test_resolver_does_not_guess_from_title_or_ticker():
    market = {"ticker": "KXNFL-GAME-YES", "title": "Chiefs at Bills"}
    assert resolve_sports_live_binding(market, milestones=[milestone()]) is None


def test_resolver_rejects_unlinked_milestone():
    market = {"event_ticker": "KXNFL-GAME"}
    assert resolve_sports_live_binding(market, milestones=[milestone(event="OTHER")]) is None
