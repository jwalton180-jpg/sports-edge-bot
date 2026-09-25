from datetime import date
from unittest.mock import patch

from sports_edge.models.tennis_research import TennisResearchModel

ROWS = (
    {"tourney_date":"20260101","match_num":"1","winner_name":"Alpha","loser_name":"Beta","tourney_level":"A","surface":"Hard"},
    {"tourney_date":"20260108","match_num":"2","winner_name":"Alpha","loser_name":"Beta","tourney_level":"A","surface":"Hard"},
    {"tourney_date":"20260115","match_num":"3","winner_name":"Alpha","loser_name":"Beta","tourney_level":"A","surface":"Hard"},
    {"tourney_date":"20260201","match_num":"4","winner_name":"Beta","loser_name":"Alpha","tourney_level":"C","surface":"Clay"},
)

def _model():
    with patch("sports_edge.models.tennis_research.load_recent_tennis_rows", return_value=ROWS):
        return TennisResearchModel("men", current_year=2026)

def test_fit_tracks_chronological_surface_and_level_ratings():
    model = _model()
    alpha, beta = model.players["alpha"], model.players["beta"]
    assert {"hard", "clay"} <= set(alpha.surface_elo)
    assert {"A", "C"} <= set(alpha.level_elo)
    assert alpha.surface_elo["hard"] > beta.surface_elo["hard"]

def test_probability_uses_surface_context_when_both_players_have_it():
    model = _model()
    hard = model.probability("Alpha", "Beta", level="ATP Tour", surface="Hard")
    clay = model.probability("Alpha", "Beta", level="ATP Tour", surface="Clay")
    assert hard is not None and clay is not None
    assert any("Hard Elo" in factor for factor in hard.factors)
    assert any("Clay Elo" in factor for factor in clay.factors)
    assert hard.fair_probability > clay.fair_probability

def test_unknown_surface_fails_soft_to_other_model_evidence():
    model = _model()
    row = model.probability("Alpha", "Beta", level="ATP Tour", surface="Grass")
    assert row is not None
    assert not any("Grass Elo" in factor for factor in row.factors)


def test_as_of_excludes_prediction_day_and_future_rows():
    rows = ROWS + (
        {"tourney_date":"20260210","match_num":"5","winner_name":"Beta","loser_name":"Alpha","tourney_level":"A","surface":"Hard"},
        {"tourney_date":"20260211","match_num":"6","winner_name":"Beta","loser_name":"Alpha","tourney_level":"A","surface":"Hard"},
    )
    with patch("sports_edge.models.tennis_research.load_recent_tennis_rows", return_value=rows):
        snapshot = TennisResearchModel("men", current_year=2026, as_of=date(2026, 2, 10))
    alpha, beta = snapshot.players["alpha"], snapshot.players["beta"]
    # Only the four rows strictly before Feb 10 are allowed into the snapshot.
    assert alpha.matches == 4
    assert beta.matches == 4


def test_event_candidate_model_cache_is_keyed_by_full_event_date():
    from sports_edge.models.kalshi_model_candidates import _tennis_model
    _tennis_model.cache_clear()
    with patch("sports_edge.models.tennis_research.load_recent_tennis_rows", return_value=ROWS):
        first = _tennis_model("men", date(2026, 2, 1))
        second = _tennis_model("men", date(2026, 2, 2))
    assert first is not second
    assert first.as_of == date(2026, 2, 1)
    assert second.as_of == date(2026, 2, 2)
