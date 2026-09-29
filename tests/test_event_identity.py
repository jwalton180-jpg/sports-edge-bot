from datetime import datetime, timezone

from sports_edge.models.event_identity import (
    canonical_event_id,
    canonical_event_id_from_game,
    canonical_event_id_from_title,
)
from sports_edge.models.game_scope import GameEvent
from sports_edge.models.kalshi_model_candidates import attach_sportsbook_context
from sports_edge.models.parlay_candidates import ParlayCandidateLeg


DATE = "2026-09-27"


def _leg(*, sport, event_id, selection, books=0):
    return ParlayCandidateLeg(
        sport=sport,
        event_id=event_id,
        event_title="Game",
        market_key="model_h2h",
        market_label="Moneyline",
        selection=selection,
        consensus_probability=0.60,
        book_count=books,
        source_age_s=15.0,
        median_odds=None,
        kalshi_ticker="KX",
        kalshi_side="YES",
        kalshi_price=0.52,
        kalshi_edge_points=8.0,
        kalshi_status="MODEL",
        evidence_class="MODEL" if books == 0 else "CONSENSUS",
        model_probability=0.60 if books == 0 else None,
        model_confidence=0.70 if books == 0 else 0.0,
        model_name="test" if books == 0 else None,
        model_sample_size=20 if books == 0 else 0,
    )


def test_nfl_kalshi_short_names_match_sportsbook_full_names():
    kalshi = canonical_event_id("NFL", "LAC Chargers", "BUF Bills", DATE)
    book_game = GameEvent(
        event_id="book-123",
        sport_key="americanfootball_nfl",
        sport="NFL",
        home_team="Buffalo Bills",
        away_team="Los Angeles Chargers",
        commence_time=datetime(2026, 9, 27, 17, 0, tzinfo=timezone.utc),
        state="UPCOMING",
    )
    assert kalshi == canonical_event_id_from_game(book_game)


def test_mlb_shortened_kalshi_names_match_full_model_title():
    short = canonical_event_id("MLB", "Los Angeles D", "San Francisco G", DATE)
    full = canonical_event_id_from_title(
        "MLB",
        "Los Angeles Dodgers @ San Francisco Giants",
        DATE,
    )
    assert short == full


def test_tennis_diacritics_and_order_share_one_match_identity():
    a = canonical_event_id("Tennis", "João Fonseca", "Sebastián Báez", DATE)
    b = canonical_event_id("Tennis", "Sebastian Baez", "Joao Fonseca", DATE)
    assert a == b


def test_sportsbook_enrichment_requires_same_physical_event():
    event_a = canonical_event_id("Tennis", "Player One", "Player Two", DATE)
    event_b = canonical_event_id("Tennis", "Player One", "Player Three", DATE)
    model = _leg(sport="Tennis", event_id=event_a, selection="Player One")
    wrong_book = _leg(
        sport="Tennis",
        event_id=event_b,
        selection="Player One",
        books=4,
    )
    enriched = attach_sportsbook_context([model], [wrong_book])
    assert enriched[0].book_count == 0
    assert enriched[0].evidence_class == "MODEL"


def test_sportsbook_enrichment_attaches_same_event_selection():
    event_id = canonical_event_id("Tennis", "Player One", "Player Two", DATE)
    model = _leg(sport="Tennis", event_id=event_id, selection="Player One")
    book = _leg(sport="Tennis", event_id=event_id, selection="Player One", books=4)
    enriched = attach_sportsbook_context([model], [book])
    assert enriched[0].book_count == 4
    assert enriched[0].evidence_class == "MODEL + BOOKS"
