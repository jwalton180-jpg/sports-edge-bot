from sports_edge.models.new_sport_research import hockey_goals, soccer_goals, college_football_win
from sports_edge.models.prospective_evidence import record, settle, evaluate, export_csv, export_jsonl
import pytest

def test_models_do_not_use_market_prices_or_claim_validation():
    h = hockey_goals(goals_per_game=.4, games=30, source_verified=True, as_of_before_event=True)
    s = soccer_goals(home_scored=1.5, away_conceded=1.4, away_scored=1.2, home_conceded=1.1,
                     home_games=15, away_games=15, source_verified=True, as_of_before_event=True)
    c = college_football_win(favorite_rating=18, underdog_rating=10, favorite_games=6, underdog_games=6,
                             home_adjustment_points=2, upset_red_flags=("qb_health_uncertain",),
                             source_verified=True, as_of_before_event=True)
    for row in (h, s, c):
        assert row.probability is not None and 0 < row.probability < 1
        assert row.status == "UNVALIDATED_RESEARCH"
    assert hockey_goals(goals_per_game=.5, games=5).probability is None
    assert college_football_win(favorite_rating=20, underdog_rating=10, favorite_games=8,
        underdog_games=8, home_adjustment_points=2, upset_red_flags=None,
        source_verified=True, as_of_before_event=True).probability is None

def test_prospective_evidence_export_and_grading():
    r = record(sport="NHL", ticker="KXNHL-1", side="YES", outcome="2+ goals",
               model="poisson", probability=.2, ask=.1, source_id="official-1",
               source_as_of="2026-10-14T10:00:00Z", observed_at="2026-10-14T10:01:00Z")
    assert export_csv([r]).startswith("id,observed_at")
    assert r["id"] in export_jsonl([r])
    assert evaluate([r])["graded"] == 0
    graded = settle(r, settled_at="2026-10-14T12:00:00Z", won=False)
    assert evaluate([graded])["brier"] == pytest.approx(.04)
    assert not evaluate([graded])["qualified"]
    with pytest.raises(ValueError):
        settle(graded, settled_at="2026-10-14T13:00:00Z", won=True)
    with pytest.raises(ValueError):
        record(sport="NHL", ticker="KXNHL-1", side="YES", outcome="x", model="x",
               probability=.5, ask=.3, source_id="x",
               source_as_of="2026-10-14T10:02:00Z", observed_at="2026-10-14T10:01:00Z")
