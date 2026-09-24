from datetime import datetime, timedelta, timezone

from sports_edge.models.consensus import consensus_from_event, match_market_to_event
from sports_edge.models.live_board import build_live_signals, build_underdog_signals, market_yes_probability
from sports_edge.models.parlay import build_parlay_research
from sports_edge.models.live_board import LiveSignal


NOW = datetime(2026, 9, 24, 22, 30, tzinfo=timezone.utc)


def book(key, a, a_price, b, b_price, age=10):
    ts = (NOW - timedelta(seconds=age)).isoformat().replace("+00:00", "Z")
    return {
        "key": key,
        "last_update": ts,
        "markets": [
            {
                "key": "h2h",
                "last_update": ts,
                "outcomes": [
                    {"name": a, "price": a_price},
                    {"name": b, "price": b_price},
                ],
            }
        ],
    }


def event(a="Kansas City Chiefs", b="Buffalo Bills", prices=(-150, 130), count=4, age=10):
    return {
        "id": "evt-1",
        "away_team": b,
        "home_team": a,
        "bookmakers": [
            book(f"book-{i}", a, prices[0], b, prices[1], age=age + i)
            for i in range(count)
        ],
    }


def market(price=0.35, title="Will the Kansas City Chiefs win?", yes="Kansas City Chiefs"):
    return {
        "ticker": "KXNFL-TEST-YES",
        "title": title,
        "yes_sub_title": yes,
        "yes_ask_dollars": price,
        "volume": 1000,
    }


def test_consensus_removes_vig_and_uses_multiple_books():
    quotes = consensus_from_event(event(), now=NOW)
    chiefs = quotes["Kansas City Chiefs"]
    bills = quotes["Buffalo Bills"]
    assert chiefs.book_count == 4
    assert abs((chiefs.fair_probability + bills.fair_probability) - 1.0) < 1e-9
    assert chiefs.fair_probability > 0.5
    assert chiefs.data_quality >= 0.70
    assert not chiefs.warnings


def test_stale_books_fail_closed():
    quotes = consensus_from_event(event(count=4, age=500), now=NOW, max_age_s=180)
    assert quotes == {}


def test_single_book_is_warning_not_qualified_source():
    quotes = consensus_from_event(event(count=1), now=NOW)
    q = quotes["Kansas City Chiefs"]
    assert q.book_count == 1
    assert any("Only 1 fresh" in w for w in q.warnings)


def test_market_match_requires_unambiguous_participant():
    matched = match_market_to_event(market(), [event()], now=NOW)
    assert matched is not None
    assert matched.selection == "Kansas City Chiefs"

    ambiguous = market(title="Who wins the game?", yes="")
    assert match_market_to_event(ambiguous, [event()], now=NOW) is None


def test_market_probability_accepts_dollars_and_cents():
    assert market_yes_probability({"yes_ask_dollars": 0.17}) == 0.17
    assert market_yes_probability({"yes_ask": 17}) == 0.17
    assert market_yes_probability({"yes_ask": None}) is None


def test_live_edge_qualifies_large_fresh_consensus_gap():
    rows = build_live_signals([market(price=0.35)], [event()], sport="NFL", now=NOW)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "QUALIFIED"
    assert row.edge_points > 3
    assert row.book_count == 4
    assert row.data_quality >= 0.70


def test_underdog_band_and_gate():
    dog_event = event(a="Player Alpha", b="Player Beta", prices=(-250, 210), count=4)
    dog_market = market(price=0.10, title="Will Player Alpha win?", yes="Player Alpha")
    rows = build_underdog_signals([dog_market], [dog_event], sport="Tennis", now=NOW)
    assert len(rows) == 1
    assert rows[0].market_probability == 0.10
    assert rows[0].status == "QUALIFIED"

    outside = dict(dog_market)
    outside["yes_ask_dollars"] = 0.40
    assert build_underdog_signals([outside], [dog_event], sport="Tennis", now=NOW) == []


def signal(event_id, ticker, fair=0.65, market_p=0.50, sport="NFL"):
    return LiveSignal(
        ticker=ticker,
        sport=sport,
        event_id=event_id,
        event_title=event_id,
        market=ticker,
        selection=ticker,
        market_probability=market_p,
        fair_probability=fair,
        edge_points=(fair - market_p) * 100,
        ev_per_contract=fair - market_p,
        confidence=0.80,
        data_quality=0.90,
        book_count=4,
        source_age_s=10,
        status="QUALIFIED",
        tier="EDGE",
        reasons=(),
        warnings=(),
    )


def test_parlay_builder_deduplicates_events_and_fails_closed_on_short_pool():
    rows = [
        signal("event-a", "a1"),
        signal("event-a", "a2", fair=0.64),
        signal("event-b", "b1"),
        signal("event-c", "c1"),
    ]
    p = build_parlay_research(rows, mode="high_confidence")
    assert len(p.legs) == 3
    assert len({leg.event_id for leg in p.legs}) == 3
    assert 0 < p.independent_fair_probability < 1

    longshot = build_parlay_research(rows, mode="longshot")
    assert any("Need at least 5" in warning for warning in longshot.warnings)
