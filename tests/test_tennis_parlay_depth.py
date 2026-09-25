from datetime import datetime, timedelta, timezone

from sports_edge.models.game_scope import GameEvent, game_scoped_markets
from sports_edge.models.live_board import LiveSignal
from sports_edge.models.parlay_candidates import candidate_legs_from_h2h


NOW = datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)


def _book(key, a, a_price, b, b_price, age=15):
    ts = (NOW - timedelta(seconds=age)).isoformat().replace("+00:00", "Z")
    return {
        "key": key,
        "last_update": ts,
        "markets": [{
            "key": "h2h",
            "last_update": ts,
            "outcomes": [
                {"name": a, "price": a_price},
                {"name": b, "price": b_price},
            ],
        }],
    }


def _event(a, b):
    return {
        "id": "evt-tennis-1",
        "home_team": a,
        "away_team": b,
        "bookmakers": [
            _book(f"book-{i}", a, -220, b, 180, age=10 + i)
            for i in range(4)
        ],
    }


def _signal(selection, ticker, fair, price):
    return LiveSignal(
        ticker=ticker,
        sport="Tennis",
        event_id="evt-tennis-1",
        event_title="Joao Fonseca @ Sebastian Baez",
        market=ticker,
        side="YES",
        selection=selection,
        market_probability=price,
        fair_probability=fair,
        edge_points=(fair - price) * 100,
        ev_per_contract=fair - price,
        confidence=0.9,
        data_quality=0.9,
        book_count=4,
        source_age_s=15,
        status="QUALIFIED",
        tier="EDGE",
        reasons=(),
        warnings=(),
    )


def test_tennis_event_scope_combines_sibling_markets_and_strips_diacritics():
    game = GameEvent(
        event_id="evt-tennis-1",
        sport_key="tennis_atp_test",
        sport="Tennis",
        home_team="Sebastián Báez",
        away_team="João Fonseca",
        commence_time=NOW + timedelta(hours=1),
        state="UPCOMING",
    )
    markets = [
        {
            "ticker": "KXATPMATCH-E1-BAEZ",
            "event_ticker": "KXATPMATCH-E1",
            "title": "Will Sebastian Baez win?",
            "yes_sub_title": "Sebastian Baez",
            "yes_ask_dollars": 0.68,
        },
        {
            "ticker": "KXATPMATCH-E1-FONSECA",
            "event_ticker": "KXATPMATCH-E1",
            "title": "Will Joao Fonseca win?",
            "yes_sub_title": "Joao Fonseca",
            "yes_ask_dollars": 0.34,
        },
    ]

    scoped = game_scoped_markets(markets, [game])
    assert {m["ticker"] for m in scoped[game.event_id]} == {
        "KXATPMATCH-E1-BAEZ",
        "KXATPMATCH-E1-FONSECA",
    }


def test_longshot_h2h_preserves_underdog_for_ev_scoring():
    game = GameEvent(
        event_id="evt-tennis-1",
        sport_key="tennis_atp_test",
        sport="Tennis",
        home_team="Player Favorite",
        away_team="Player Underdog",
        commence_time=NOW + timedelta(hours=1),
        state="UPCOMING",
    )
    payload = _event("Player Favorite", "Player Underdog")
    exact = [
        _signal("Player Favorite", "KX-FAV", 0.69, 0.62),
        _signal("Player Underdog", "KX-DOG", 0.31, 0.22),
    ]

    longshot = candidate_legs_from_h2h(game, payload, exact, mode="longshot", now=NOW)
    best = candidate_legs_from_h2h(game, payload, exact, mode="high_confidence", now=NOW)

    assert {row.selection for row in longshot} == {"Player Favorite", "Player Underdog"}
    assert len(best) == 1
    assert best[0].selection == "Player Favorite"
