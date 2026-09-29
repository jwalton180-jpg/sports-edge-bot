from datetime import datetime, timezone

from sports_edge.models.consensus import line_consensus_from_event
from sports_edge.models.event_identity import canonical_event_id_from_game
from sports_edge.models.game_scope import GameEvent
from sports_edge.models.parlay_candidates import (
    ParlayCandidateLeg,
    candidate_book_context_for_model_lines,
)


NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


def _event():
    books = []
    for key, spread_a, spread_b, over, under in (
        ("draftkings", -108, -112, -110, -110),
        ("fanduel", -105, -115, -108, -112),
        ("betmgm", -110, -110, -105, -115),
    ):
        books.append(
            {
                "key": key,
                "last_update": "2026-09-29T11:59:30Z",
                "markets": [
                    {
                        "key": "spreads",
                        "last_update": "2026-09-29T11:59:30Z",
                        "outcomes": [
                            {"name": "Buffalo Bills", "price": spread_a, "point": -3.5},
                            {"name": "Los Angeles Chargers", "price": spread_b, "point": 3.5},
                        ],
                    },
                    {
                        "key": "totals",
                        "last_update": "2026-09-29T11:59:30Z",
                        "outcomes": [
                            {"name": "Over", "price": over, "point": 47.5},
                            {"name": "Under", "price": under, "point": 47.5},
                        ],
                    },
                    {
                        "key": "alternate_team_totals",
                        "last_update": "2026-09-29T11:59:30Z",
                        "outcomes": [
                            {"name": "Over", "description": "Buffalo Bills", "price": -110, "point": 24.5},
                            {"name": "Under", "description": "Buffalo Bills", "price": -110, "point": 24.5},
                            {"name": "Over", "description": "Los Angeles Chargers", "price": -105, "point": 21.5},
                            {"name": "Under", "description": "Los Angeles Chargers", "price": -115, "point": 21.5},
                        ],
                    },
                ],
            }
        )
    return {
        "id": "book-event",
        "home_team": "Buffalo Bills",
        "away_team": "Los Angeles Chargers",
        "bookmakers": books,
    }


def _game():
    return GameEvent(
        event_id="book-event",
        sport_key="americanfootball_nfl",
        sport="NFL",
        home_team="Buffalo Bills",
        away_team="Los Angeles Chargers",
        commence_time=NOW,
        state="UPCOMING",
    )


def _model(selection, key, fair=0.61):
    game = _game()
    return ParlayCandidateLeg(
        sport="NFL",
        event_id=canonical_event_id_from_game(game),
        event_title="LAC @ BUF",
        market_key=key,
        market_label=(
            "Spread" if key == "nfl_spread"
            else "Team Total" if key == "nfl_team_total"
            else "Game Total"
        ),
        selection=selection,
        consensus_probability=fair,
        book_count=0,
        source_age_s=0.0,
        median_odds=None,
        kalshi_ticker="KXTEST",
        kalshi_side="YES",
        kalshi_price=0.50,
        kalshi_edge_points=11.0,
        kalshi_status="MODEL",
        evidence_class="MODEL",
        model_probability=fair,
        model_confidence=0.65,
        model_name="test",
        model_sample_size=20,
    )


def test_line_consensus_requires_exact_point_and_has_multiple_books():
    quote = line_consensus_from_event(
        _event(),
        market_key="spreads",
        target_name="Buffalo Bills",
        target_point=-3.5,
        selection_label="BUF margin > 3.5",
        now=NOW,
    )
    assert quote is not None
    assert quote.book_count == 3
    assert not quote.warnings
    assert 0.45 < quote.fair_probability < 0.55

    missing = line_consensus_from_event(
        _event(),
        market_key="spreads",
        target_name="Buffalo Bills",
        target_point=-4.5,
        selection_label="BUF margin > 4.5",
        now=NOW,
    )
    assert missing is None


def test_spread_yes_and_no_map_to_exact_complementary_book_sides():
    game = _game()
    targets = [
        _model("BUF margin > 3.5", "nfl_spread"),
        _model("BUF margin ≤ 3.5", "nfl_spread", fair=0.39),
    ]
    rows = candidate_book_context_for_model_lines(
        game,
        _event(),
        targets,
        now=NOW,
    )
    assert len(rows) == 2
    by_selection = {r.selection: r for r in rows}
    assert by_selection["BUF margin > 3.5"].book_count == 3
    assert by_selection["BUF margin ≤ 3.5"].book_count == 3
    total = (
        by_selection["BUF margin > 3.5"].consensus_probability
        + by_selection["BUF margin ≤ 3.5"].consensus_probability
    )
    assert abs(total - 1.0) < 1e-9


def test_game_total_over_under_book_context_matches_model_selection():
    game = _game()
    targets = [
        _model("Over 47.5 Game Total", "nfl_game_total"),
        _model("Under 47.5 Game Total", "nfl_game_total", fair=0.39),
    ]
    rows = candidate_book_context_for_model_lines(
        game,
        _event(),
        targets,
        now=NOW,
    )
    assert len(rows) == 2
    assert {r.selection for r in rows} == {
        "Over 47.5 Game Total",
        "Under 47.5 Game Total",
    }
    assert all(r.event_id == canonical_event_id_from_game(game) for r in rows)


def test_integer_line_fails_closed_due_to_push_mismatch():
    rows = candidate_book_context_for_model_lines(
        _game(),
        _event(),
        [_model("BUF margin > 3", "nfl_spread")],
        now=NOW,
    )
    assert rows == []


def test_team_total_over_under_book_context_is_exact_team_and_line():
    game = _game()
    targets = [
        _model("BUF Over 24.5 Team Total", "nfl_team_total"),
        _model("BUF Under 24.5 Team Total", "nfl_team_total", fair=0.39),
    ]
    rows = candidate_book_context_for_model_lines(
        game,
        _event(),
        targets,
        now=NOW,
    )
    assert len(rows) == 2
    by_selection = {row.selection: row for row in rows}
    assert by_selection["BUF Over 24.5 Team Total"].book_count == 3
    assert by_selection["BUF Under 24.5 Team Total"].book_count == 3
    assert abs(
        by_selection["BUF Over 24.5 Team Total"].consensus_probability
        + by_selection["BUF Under 24.5 Team Total"].consensus_probability
        - 1.0
    ) < 1e-9


def test_team_total_wrong_line_and_integer_push_fail_closed():
    game = _game()
    wrong = candidate_book_context_for_model_lines(
        game,
        _event(),
        [_model("BUF Over 25.5 Team Total", "nfl_team_total")],
        now=NOW,
    )
    integer = candidate_book_context_for_model_lines(
        game,
        _event(),
        [_model("BUF Over 24 Team Total", "nfl_team_total")],
        now=NOW,
    )
    assert wrong == []
    assert integer == []
