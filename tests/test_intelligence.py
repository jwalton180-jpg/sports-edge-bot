from datetime import datetime, timedelta, timezone

from sports_edge.models.intelligence import h2h_intelligence, prop_book_offer_edges

NOW = datetime(2026, 9, 24, 20, 0, tzinfo=timezone.utc)


def _book(key, a, a_price, b, b_price, age=10):
    ts = (NOW - timedelta(seconds=age)).isoformat().replace("+00:00", "Z")
    return {
        "key": key,
        "title": key,
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


def test_h2h_intelligence_weights_fresh_multibook_consensus_and_edge():
    event = {
        "bookmakers": [
            _book("pinnacle", "A", -150, "B", 130, 5),
            _book("draftkings", "A", -145, "B", 125, 8),
            _book("fanduel", "A", -148, "B", 128, 12),
            _book("betmgm", "A", -142, "B", 122, 15),
        ]
    }
    result = h2h_intelligence(event, "A", market_probability=0.48, now=NOW)
    assert result is not None
    assert result.book_count == 4
    assert result.reference_book_count == 1
    assert result.edge_points is not None and result.edge_points > 5
    assert result.intelligence_score > 60
    assert result.tier in {"A", "B", "WATCH"}


def test_h2h_intelligence_rejects_stale_books():
    event = {"bookmakers": [_book("pinnacle", "A", -150, "B", 130, 500)]}
    assert h2h_intelligence(event, "A", market_probability=0.5, now=NOW) is None


def test_prop_leave_one_out_detects_underpriced_book_without_using_itself():
    ts = (NOW - timedelta(seconds=10)).isoformat().replace("+00:00", "Z")
    payload = {
        "id": "evt1",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "last_update": ts,
                "markets": [{
                    "key": "batter_hits",
                    "last_update": ts,
                    "outcomes": [
                        {"name": "Over", "description": "Player A", "price": -190, "point": 0.5},
                        {"name": "Under", "description": "Player A", "price": 150, "point": 0.5},
                    ],
                }],
            },
            {
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": ts,
                "markets": [{
                    "key": "batter_hits",
                    "last_update": ts,
                    "outcomes": [
                        {"name": "Over", "description": "Player A", "price": -185, "point": 0.5},
                        {"name": "Under", "description": "Player A", "price": 145, "point": 0.5},
                    ],
                }],
            },
            {
                "key": "fanduel",
                "title": "FanDuel",
                "last_update": ts,
                "markets": [{
                    "key": "batter_hits",
                    "last_update": ts,
                    "outcomes": [
                        {"name": "Over", "description": "Player A", "price": -105, "point": 0.5},
                        {"name": "Under", "description": "Player A", "price": -115, "point": 0.5},
                    ],
                }],
            },
        ],
    }
    rows = prop_book_offer_edges(payload, ("batter_hits",), now=NOW)
    fd = next(r for r in rows if r.bookmaker_key == "fanduel" and "Player A Over" in r.selection)
    assert fd.comparison_books == 2
    assert fd.edge_points > 5
    assert fd.leave_one_out_fair_probability > fd.break_even_probability



def test_default_freshness_accepts_120_seconds_but_rejects_121():
    fresh_event = {
        "bookmakers": [
            _book("pinnacle", "A", -150, "B", 130, 120),
            _book("draftkings", "A", -145, "B", 125, 120),
        ]
    }
    stale_event = {
        "bookmakers": [
            _book("pinnacle", "A", -150, "B", 130, 121),
            _book("draftkings", "A", -145, "B", 125, 121),
        ]
    }

    assert h2h_intelligence(fresh_event, "A", market_probability=0.50, now=NOW) is not None
    assert h2h_intelligence(stale_event, "A", market_probability=0.50, now=NOW) is None


def test_prop_default_freshness_rejects_121_second_quotes():
    def payload(age):
        ts = (NOW - timedelta(seconds=age)).isoformat().replace("+00:00", "Z")
        return {
            "id": "evt-age",
            "bookmakers": [
                {
                    "key": key,
                    "title": key,
                    "last_update": ts,
                    "markets": [{
                        "key": "player_points",
                        "last_update": ts,
                        "outcomes": [
                            {"name": "Over", "description": "Player A", "price": -110, "point": 20.5},
                            {"name": "Under", "description": "Player A", "price": -110, "point": 20.5},
                        ],
                    }],
                }
                for key in ("pinnacle", "draftkings", "fanduel")
            ],
        }

    assert prop_book_offer_edges(payload(120), ("player_points",), now=NOW)
    assert prop_book_offer_edges(payload(121), ("player_points",), now=NOW) == []
