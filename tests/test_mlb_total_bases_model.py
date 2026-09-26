from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.mlb_total_bases_model import (
    MLBTotalBasesProjection,
    at_least_k_total_bases_probability,
)
import sports_edge.models.kalshi_model_candidates as kmc


def test_total_bases_milestone_probability_is_monotone():
    rates = (0.16, 0.05, 0.004, 0.035)
    p1 = at_least_k_total_bases_probability(rates, 4.2, 1)
    p2 = at_least_k_total_bases_probability(rates, 4.2, 2)
    p3 = at_least_k_total_bases_probability(rates, 4.2, 3)
    p4 = at_least_k_total_bases_probability(rates, 4.2, 4)
    assert 0 < p4 < p3 < p2 < p1 < 1


def test_more_extra_base_power_raises_multi_base_probability():
    contact_heavy = (0.20, 0.025, 0.002, 0.015)
    power_heavy = (0.14, 0.06, 0.006, 0.05)
    assert sum(contact_heavy) == sum(power_heavy)
    low = at_least_k_total_bases_probability(contact_heavy, 4.0, 2)
    high = at_least_k_total_bases_probability(power_heavy, 4.0, 2)
    assert high > low


def test_more_expected_at_bats_raise_total_bases_probability():
    rates = (0.16, 0.05, 0.004, 0.035)
    low = at_least_k_total_bases_probability(rates, 3.0, 2)
    high = at_least_k_total_bases_probability(rates, 4.8, 2)
    assert high > low


def test_mlb_total_bases_market_builds_yes_and_no_model_candidates(monkeypatch):
    evidence = ModelEvidence(
        sport="MLB",
        model_name="test MLB Total Bases",
        fair_probability=0.58,
        confidence=0.68,
        sample_size=410,
        factors=("hit-type/recent/starter",),
    )
    projection = MLBTotalBasesProjection(
        evidence=evidence,
        player_name="Test Hitter",
        milestone_total_bases=2,
        line=1.5,
        game_title="Away vs Home",
        probable_pitcher_name="Starter",
        expected_at_bats=4.2,
        expected_total_bases=1.75,
    )

    monkeypatch.setattr(kmc, "project_mlb_total_bases", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="MLB",
        family="Total Bases",
        market={
            "ticker": "KXMLBTB-26SEP251900AAABBB-TEST1-2",
            "event_ticker": "KXMLBTB-26SEP251900AAABBB",
            "series_ticker": "KXMLBTB",
            "title": "Test Hitter: 2+ total bases?",
            "floor_strike": 1.5,
            "yes_bid_dollars": "0.48",
            "yes_ask_dollars": "0.50",
            "no_bid_dollars": "0.50",
            "no_ask_dollars": "0.52",
        },
    )

    out = kmc._mlb_tb_candidates([row])
    assert len(out) == 2

    yes = next(x for x in out if x.kalshi_side == "YES")
    no = next(x for x in out if x.kalshi_side == "NO")

    assert yes.market_key == "batter_total_bases"
    assert yes.selection == "Test Hitter Over 1.5 Total Bases"
    assert abs(yes.model_probability - 0.58) < 1e-9
    assert yes.kalshi_ticker == row.market["ticker"]

    assert no.selection == "Test Hitter Under 1.5 Total Bases"
    assert abs(no.model_probability - 0.42) < 1e-9


def test_four_total_base_contract_uses_three_point_five_line(monkeypatch):
    evidence = ModelEvidence(
        sport="MLB",
        model_name="test MLB Total Bases",
        fair_probability=0.21,
        confidence=0.61,
        sample_size=360,
    )
    projection = MLBTotalBasesProjection(
        evidence=evidence,
        player_name="Test Hitter",
        milestone_total_bases=4,
        line=3.5,
        game_title="Away vs Home",
        probable_pitcher_name=None,
        expected_at_bats=4.0,
        expected_total_bases=1.55,
    )
    monkeypatch.setattr(kmc, "project_mlb_total_bases", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="MLB",
        family="Total Bases",
        market={
            "ticker": "KXMLBTB-26SEP251900AAABBB-TEST1-4",
            "event_ticker": "KXMLBTB-26SEP251900AAABBB",
            "series_ticker": "KXMLBTB",
            "title": "Test Hitter: 4+ total bases?",
            "floor_strike": 3.5,
            "yes_bid_dollars": "0.17",
            "yes_ask_dollars": "0.19",
            "no_bid_dollars": "0.81",
            "no_ask_dollars": "0.83",
        },
    )
    out = kmc._mlb_tb_candidates([row])
    assert any(x.selection == "Test Hitter Over 3.5 Total Bases" for x in out)
    assert any(x.selection == "Test Hitter Under 3.5 Total Bases" for x in out)
