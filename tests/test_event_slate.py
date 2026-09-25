from datetime import datetime, timezone

from sports_edge.models.event_slate import (
    PROP_PRESETS,
    event_identity_matches,
    game_line_summary,
    is_futures_or_non_game_market,
    kalshi_markets_for_event,
    prop_consensus_rows,
)


def event():
    return {
        "id": "evt",
        "sport_key": "americanfootball_nfl",
        "home_team": "Carolina Panthers",
        "away_team": "Cleveland Browns",
        "commence_time": "2026-09-24T23:00:00Z",
        "bookmakers": [
            {
                "key": "book-a",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Carolina Panthers", "price": 120},
                            {"name": "Cleveland Browns", "price": -140},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Carolina Panthers", "point": 2.5, "price": -110},
                            {"name": "Cleveland Browns", "point": -2.5, "price": -110},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 44.5, "price": -105},
                            {"name": "Under", "point": 44.5, "price": -115},
                        ],
                    },
                ],
            }
        ],
    }


def game_market():
    return {
        "ticker": "KXNFL-CAR-CLE",
        "event_title": "Cleveland Browns at Carolina Panthers",
        "title": "Will the Carolina Panthers win this game?",
        "yes_sub_title": "Carolina Panthers",
        "occurrence_datetime": "2026-09-24T23:00:00Z",
        "yes_ask_dollars": 0.48,
    }


def test_actual_game_market_matches_exact_event():
    assert event_identity_matches(game_market(), event())
    rows = kalshi_markets_for_event(event(), [game_market()])
    assert [m["ticker"] for m in rows] == ["KXNFL-CAR-CLE"]


def test_futures_market_never_enters_game_slate():
    future = {
        "ticker": "KXCITYCHAMPS",
        "event_title": "Carolina Panthers and other teams",
        "title": "Will the Carolina Panthers win a championship before 2030?",
        "yes_sub_title": "Carolina Panthers",
        "yes_ask_dollars": 0.04,
    }
    assert is_futures_or_non_game_market(future)
    assert not event_identity_matches(future, event())
    assert kalshi_markets_for_event(event(), [future]) == []


def test_single_participant_does_not_count_as_event_identity():
    wrong = {
        "ticker": "WRONG",
        "event_title": "Miami sports teams",
        "title": "Will the Carolina Panthers do something?",
        "yes_sub_title": "Carolina Panthers",
    }
    assert not event_identity_matches(wrong, event())


def test_game_line_summary_has_ml_spread_total():
    summary = game_line_summary(event())
    assert summary["home_fair"] is not None
    assert summary["away_fair"] is not None
    assert summary["home_spread"][0] == 2.5
    assert summary["away_spread"][0] == -2.5
    assert summary["total_over"][0] == 44.5
    assert summary["total_under"][0] == 44.5


def test_prop_consensus_is_event_scoped_and_aggregates_books():
    payload = {
        "bookmakers": [
            {
                "key": "a",
                "markets": [
                    {
                        "key": "player_pass_yds",
                        "outcomes": [
                            {"description": "QB One", "name": "Over", "point": 249.5, "price": -110},
                            {"description": "QB One", "name": "Under", "point": 249.5, "price": -110},
                        ],
                    }
                ],
            },
            {
                "key": "b",
                "markets": [
                    {
                        "key": "player_pass_yds",
                        "outcomes": [
                            {"description": "QB One", "name": "Over", "point": 249.5, "price": -105},
                            {"description": "QB One", "name": "Under", "point": 249.5, "price": -115},
                        ],
                    }
                ],
            },
        ]
    }
    rows = prop_consensus_rows(payload, PROP_PRESETS["NFL Passing"])
    assert len(rows) == 2
    assert {r["Side"] for r in rows} == {"Over", "Under"}
    assert all(r["Books"] == 2 for r in rows)
