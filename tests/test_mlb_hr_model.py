from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.mlb_hr_model import (
    MLBHomeRunProjection,
    at_least_k_home_runs_probability,
)
import sports_edge.models.kalshi_model_candidates as kmc


def test_home_run_milestone_probability_is_monotone():
    p1 = at_least_k_home_runs_probability(0.055, 4.2, 1)
    p2 = at_least_k_home_runs_probability(0.055, 4.2, 2)
    assert 0 < p2 < p1 < 1


def test_more_expected_at_bats_raise_home_run_probability():
    low = at_least_k_home_runs_probability(0.05, 3.0, 1)
    high = at_least_k_home_runs_probability(0.05, 4.8, 1)
    assert high > low


def test_mlb_hr_market_builds_yes_and_no_model_candidates(monkeypatch):
    evidence = ModelEvidence(
        sport="MLB",
        model_name="test MLB HR",
        fair_probability=0.18,
        confidence=0.62,
        sample_size=420,
        factors=("season/recent/starter HR tendency",),
    )
    projection = MLBHomeRunProjection(
        evidence=evidence,
        player_name="Test Slugger",
        milestone_home_runs=1,
        line=0.5,
        game_title="Away vs Home",
        probable_pitcher_name="Starter",
        expected_at_bats=4.2,
        per_ab_home_run_probability=0.046,
    )

    monkeypatch.setattr(kmc, "project_mlb_home_runs", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="MLB",
        family="Home Runs",
        market={
            "ticker": "KXMLBHR-26SEP251900AAABBB-TEST1-1",
            "event_ticker": "KXMLBHR-26SEP251900AAABBB",
            "series_ticker": "KXMLBHR",
            "title": "Test Slugger: 1+ home runs?",
            "floor_strike": 0.5,
            "yes_bid_dollars": "0.13",
            "yes_ask_dollars": "0.15",
            "no_bid_dollars": "0.85",
            "no_ask_dollars": "0.87",
        },
    )

    out = kmc._mlb_hr_candidates([row])
    assert len(out) == 2

    yes = next(x for x in out if x.kalshi_side == "YES")
    no = next(x for x in out if x.kalshi_side == "NO")

    assert yes.market_key == "batter_home_runs"
    assert yes.selection == "Test Slugger Over 0.5 Home Runs"
    assert abs(yes.model_probability - 0.18) < 1e-9
    assert yes.kalshi_ticker == row.market["ticker"]

    assert no.selection == "Test Slugger Under 0.5 Home Runs"
    assert abs(no.model_probability - 0.82) < 1e-9


def test_two_home_run_contract_uses_one_point_five_line(monkeypatch):
    evidence = ModelEvidence(
        sport="MLB",
        model_name="test MLB HR",
        fair_probability=0.035,
        confidence=0.55,
        sample_size=390,
    )
    projection = MLBHomeRunProjection(
        evidence=evidence,
        player_name="Test Slugger",
        milestone_home_runs=2,
        line=1.5,
        game_title="Away vs Home",
        probable_pitcher_name=None,
        expected_at_bats=4.0,
        per_ab_home_run_probability=0.05,
    )
    monkeypatch.setattr(kmc, "project_mlb_home_runs", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="MLB",
        family="Home Runs",
        market={
            "ticker": "KXMLBHR-26SEP251900AAABBB-TEST1-2",
            "event_ticker": "KXMLBHR-26SEP251900AAABBB",
            "series_ticker": "KXMLBHR",
            "title": "Test Slugger: 2+ home runs?",
            "floor_strike": 1.5,
            "yes_bid_dollars": "0.02",
            "yes_ask_dollars": "0.03",
            "no_bid_dollars": "0.97",
            "no_ask_dollars": "0.98",
        },
    )
    out = kmc._mlb_hr_candidates([row])
    assert any(x.selection == "Test Slugger Over 1.5 Home Runs" for x in out)
    assert any(x.selection == "Test Slugger Under 1.5 Home Runs" for x in out)
