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
