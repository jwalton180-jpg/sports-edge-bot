from datetime import datetime, timedelta, timezone

from sports_edge.models.game_scope import (
    build_game_events,
    game_scoped_markets,
    looks_like_future,
    market_matches_game,
)
from sports_edge.models.prop_edges import build_prop_signals, prop_quote_matches_market
from sports_edge.models.props import PropConsensus, prop_consensus


NOW = datetime(2026, 9, 24, 20, 0, tzinfo=timezone.utc)


def raw_event(event_id="g1", home="Kansas City Chiefs", away="Buffalo Bills", hours=2):
    return {
        "id": event_id,
        "home_team": home,
        "away_team": away,
        "commence_time": (NOW + timedelta(hours=hours)).isoformat().replace("+00:00", "Z"),
    }


def game_market(title, event_title, price=0.40):
    return {
        "ticker": "KXGAME-TEST",
        "title": title,
        "event_title": event_title,
        "yes_ask_dollars": price,
        "volume": 10,
    }


def test_game_universe_only_contains_current_real_events():
    rows = [
        raw_event("live-ish", hours=-2),
        raw_event("soon", hours=3),
        raw_event("future", hours=100),
    ]
    games = build_game_events(rows, sport_key="americanfootball_nfl", sport="NFL", now=NOW)
    assert [g.event_id for g in games] == ["live-ish", "soon"]


def test_market_must_contain_both_game_participants():
    game = build_game_events([raw_event()], sport_key="americanfootball_nfl", sport="NFL", now=NOW)[0]
    good = game_market(
        "Will Kansas City Chiefs beat Buffalo Bills?",
        "Buffalo Bills @ Kansas City Chiefs",
    )
    bad_future = game_market(
        "Will Kansas City Chiefs win the Super Bowl before 2030?",
        "Kansas City Chiefs futures",
    )
    assert market_matches_game(good, game)
    assert not market_matches_game(bad_future, game)

    scoped = game_scoped_markets([good, bad_future], [game])
    assert scoped[game.event_id] == [good]


def test_future_terms_are_detected_for_diagnostics():
    market = game_market(
        "Will the Pittsburgh Pirates win the championship before 2030?",
        "Pittsburgh Pirates futures",
    )
    assert looks_like_future(market)


def prop_payload(now=NOW):
    ts = (now - timedelta(seconds=15)).isoformat().replace("+00:00", "Z")
    return {
        "id": "mlb1",
        "sport_key": "baseball_mlb",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "commence_time": (now + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "bookmakers": [
            {
                "key": f"book{i}",
                "title": f"Book {i}",
                "last_update": ts,
                "markets": [
                    {
                        "key": "batter_hits",
                        "last_update": ts,
                        "outcomes": [
                            {"name": "Over", "description": "Aaron Judge", "price": -150, "point": 0.5},
                            {"name": "Under", "description": "Aaron Judge", "price": 120, "point": 0.5},
                        ],
                    }
                ],
            }
            for i in range(3)
        ],
    }


def test_prop_consensus_builds_fresh_no_vig_player_rows():
    rows = prop_consensus(prop_payload(), market_keys=("batter_hits",), now=NOW)
    over = next(r for r in rows if r.player == "Aaron Judge" and r.side == "Over")
    assert over.book_count == 3
    assert over.line == 0.5
    assert over.fair_probability > 0.5
    assert not over.warnings


def test_prop_match_requires_exact_game_player_family_and_milestone():
    game = build_game_events(
        [raw_event("mlb1", home="New York Yankees", away="Boston Red Sox")],
        sport_key="baseball_mlb",
        sport="MLB",
        now=NOW,
    )[0]
    quote = PropConsensus(
        market_key="batter_hits",
        market_label="Hits",
        player="Aaron Judge",
        side="Over",
        line=0.5,
        fair_probability=0.62,
        book_count=3,
        median_age_s=10,
        median_price=-150,
        warnings=(),
    )

    exact = game_market(
        "Will Aaron Judge record a hit?",
        "Boston Red Sox @ New York Yankees",
        price=0.40,
    )
    wrong_player = game_market(
        "Will Juan Soto record a hit?",
        "Boston Red Sox @ New York Yankees",
        price=0.40,
    )
    wrong_game = game_market(
        "Will Aaron Judge record a hit?",
        "Baltimore Orioles @ New York Yankees",
        price=0.40,
    )
    wrong_prop = game_market(
        "Will Aaron Judge hit a home run?",
        "Boston Red Sox @ New York Yankees",
        price=0.40,
    )

    assert prop_quote_matches_market(exact, game, quote)
    assert not prop_quote_matches_market(wrong_player, game, quote)
    assert not prop_quote_matches_market(wrong_game, game, quote)
    assert not prop_quote_matches_market(wrong_prop, game, quote)

    signals = build_prop_signals([exact, wrong_player, wrong_game, wrong_prop], game, [quote])
    assert len(signals) == 1
    assert signals[0].event_id == "mlb1"
    assert "Aaron Judge" in signals[0].selection
