from datetime import date

from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.mlb_run_production_model import (
    MLBRunProductionProjection,
    _metric,
    _negative_binomial_tail,
    project_mlb_hrr,
    project_mlb_rbis,
)
from sports_edge.models.model_evidence import ModelEvidence
import sports_edge.models.kalshi_model_candidates as kmc
import sports_edge.models.mlb_run_production_model as mrp


EVENT_DATE = date(2026, 9, 26)


def test_run_production_tail_is_monotone():
    p1 = _negative_binomial_tail(1.8, 3.5, 1)
    p2 = _negative_binomial_tail(1.8, 3.5, 2)
    p3 = _negative_binomial_tail(1.8, 3.5, 3)
    assert 0 < p3 < p2 < p1 < 1


def test_hrr_metric_counts_hits_runs_and_rbis():
    row = {"hits": 2, "runs": 1, "rbi": 3}
    assert _metric(row, "rbi") == 3
    assert _metric(row, "hrr") == 6


def _patch_live_inputs(monkeypatch):
    player = {
        "id": 10,
        "fullName": "Test Hitter",
        "currentTeam": {"id": 20},
    }
    game = {"gamePk": 99}
    context = {
        "game_status": "Preview",
        "probable_pitcher_id": 30,
        "probable_pitcher_name": "Test Starter",
        "own_team_name": "Home",
        "opponent_team_name": "Away",
    }
    season_hitting = {
        "plateAppearances": 600,
        "gamesPlayed": 150,
        "hits": 165,
        "runs": 92,
        "rbi": 84,
    }
    recent_hitting = {
        "plateAppearances": 105,
        "gamesPlayed": 25,
        "hits": 31,
        "runs": 19,
        "rbi": 17,
    }
    pitcher = {
        "era": "4.60",
        "whip": "1.36",
        "inningsPitched": "155.0",
    }
    logs = tuple(
        {
            "_date": f"2026-09-{(i % 20) + 1:02d}",
            "plateAppearances": 4,
            "hits": i % 3,
            "runs": 1 if i % 4 == 0 else 0,
            "rbi": 2 if i % 9 == 0 else (1 if i % 4 == 0 else 0),
        }
        for i in range(50)
    )

    monkeypatch.setattr(mrp, "resolve_player", lambda *args, **kwargs: player)
    monkeypatch.setattr(mrp, "find_player_game", lambda **kwargs: game)
    monkeypatch.setattr(mrp, "game_context_for_hitter", lambda *args, **kwargs: context)
    monkeypatch.setattr(
        mrp,
        "season_stat",
        lambda player_id, group, season: pitcher if group == "pitching" else season_hitting,
    )
    monkeypatch.setattr(mrp, "date_range_stat", lambda *args, **kwargs: recent_hitting)
    monkeypatch.setattr(mrp, "game_log_stats", lambda *args, **kwargs: logs)


def test_rbi_and_hrr_models_use_game_log_dispersion(monkeypatch):
    _patch_live_inputs(monkeypatch)
    rbi = project_mlb_rbis(
        player_name="Test Hitter",
        milestone_rbis=1,
        event_date=EVENT_DATE,
        event_ticker="KXMLBRBI-26SEP26HOMEAWAY",
    )
    hrr = project_mlb_hrr(
        player_name="Test Hitter",
        milestone_hrr=2,
        event_date=EVENT_DATE,
        event_ticker="KXMLBHRR-26SEP26HOMEAWAY",
    )
    assert rbi is not None and hrr is not None
    assert rbi.market_key == "batter_rbis"
    assert hrr.market_key == "batter_hrr"
    assert rbi.projected_variance >= rbi.projected_mean
    assert hrr.projected_variance >= hrr.projected_mean * 1.25
    assert hrr.evidence.sample_size == 150
    assert any("game-log" in hrr.evidence.model_name.lower() for _ in [0])


def test_mlb_hrr_market_builds_yes_and_no_candidates(monkeypatch):
    evidence = ModelEvidence(
        sport="MLB",
        model_name="test MLB H+R+RBI",
        fair_probability=0.61,
        confidence=0.70,
        sample_size=150,
        factors=("game-log dispersion",),
    )
    projection = MLBRunProductionProjection(
        evidence=evidence,
        player_name="Test Hitter",
        market_key="batter_hrr",
        market_label="Hits + Runs + RBIs",
        milestone=2,
        line=1.5,
        game_title="Home vs Away",
        probable_pitcher_name="Starter",
        expected_plate_appearances=4.3,
        projected_mean=2.2,
        projected_variance=3.4,
    )
    monkeypatch.setattr(kmc, "project_mlb_hrr", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="MLB",
        family="Hits + Runs + RBIs",
        market={
            "ticker": "KXMLBHRR-26SEP26HOMEAWAY-TEST-2",
            "event_ticker": "KXMLBHRR-26SEP26HOMEAWAY",
            "series_ticker": "KXMLBHRR",
            "title": "Test Hitter: 2+ hits + runs + RBIs?",
            "floor_strike": 1.5,
            "yes_ask_dollars": "0.54",
            "no_ask_dollars": "0.47",
        },
    )
    out = kmc._mlb_run_candidates([row])
    assert len(out) == 2
    yes = next(x for x in out if x.kalshi_side == "YES")
    no = next(x for x in out if x.kalshi_side == "NO")
    assert yes.selection == "Test Hitter Over 1.5 Hits + Runs + RBIs"
    assert yes.market_key == "batter_hrr"
    assert abs(yes.model_probability - 0.61) < 1e-9
    assert no.selection == "Test Hitter Under 1.5 Hits + Runs + RBIs"
    assert abs(no.model_probability - 0.39) < 1e-9


def test_mlb_rbi_market_builds_yes_and_no_candidates(monkeypatch):
    evidence = ModelEvidence(
        sport="MLB",
        model_name="test MLB RBI",
        fair_probability=0.57,
        confidence=0.68,
        sample_size=145,
        factors=("game-log dispersion",),
    )
    projection = MLBRunProductionProjection(
        evidence=evidence,
        player_name="Test Hitter",
        market_key="batter_rbis",
        market_label="RBIs",
        milestone=1,
        line=0.5,
        game_title="Home vs Away",
        probable_pitcher_name="Starter",
        expected_plate_appearances=4.2,
        projected_mean=0.82,
        projected_variance=1.20,
    )
    monkeypatch.setattr(kmc, "project_mlb_rbis", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="MLB",
        family="RBIs",
        market={
            "ticker": "KXMLBRBI-26SEP26HOMEAWAY-TEST-1",
            "event_ticker": "KXMLBRBI-26SEP26HOMEAWAY",
            "series_ticker": "KXMLBRBI",
            "title": "Test Hitter: 1+ RBIs",
            "floor_strike": 0.5,
            "yes_ask_dollars": "0.49",
            "no_ask_dollars": "0.52",
        },
    )
    out = kmc._mlb_run_candidates([row])
    assert len(out) == 2
    yes = next(x for x in out if x.kalshi_side == "YES")
    no = next(x for x in out if x.kalshi_side == "NO")
    assert yes.selection == "Test Hitter Over 0.5 RBIs"
    assert yes.market_key == "batter_rbis"
    assert abs(yes.model_probability - 0.57) < 1e-9
    assert no.selection == "Test Hitter Under 0.5 RBIs"
    assert abs(no.model_probability - 0.43) < 1e-9
