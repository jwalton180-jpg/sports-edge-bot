from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.mlb_hits_model import MLBHitProjection, at_least_k_hits_probability
import sports_edge.models.kalshi_model_candidates as kmc


def test_hit_milestone_probability_is_monotone():
    p1 = at_least_k_hits_probability(0.28, 4.1, 1)
    p2 = at_least_k_hits_probability(0.28, 4.1, 2)
    p3 = at_least_k_hits_probability(0.28, 4.1, 3)
    assert 0 < p3 < p2 < p1 < 1


def test_more_expected_at_bats_raise_hit_probability():
    low = at_least_k_hits_probability(0.27, 3.2, 1)
    high = at_least_k_hits_probability(0.27, 4.5, 1)
    assert high > low


def test_mlb_hit_market_builds_yes_and_no_model_candidates(monkeypatch):
    evidence = ModelEvidence(
        sport="MLB",
        model_name="test MLB Hits",
        fair_probability=0.64,
        confidence=0.70,
        sample_size=400,
        factors=("season/recent/pitcher",),
    )
    projection = MLBHitProjection(
        evidence=evidence,
        player_name="Test Hitter",
        milestone_hits=1,
        line=0.5,
        game_title="Away vs Home",
        probable_pitcher_name="Starter",
        expected_at_bats=4.1,
        per_ab_hit_probability=0.28,
    )

    monkeypatch.setattr(kmc, "project_mlb_hits", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="MLB",
        family="Hits",
        market={
            "ticker": "KXMLBHIT-26SEP251900AAABBB-BBBTEST1-1",
            "event_ticker": "KXMLBHIT-26SEP251900AAABBB",
            "series_ticker": "KXMLBHIT",
            "title": "Test Hitter: 1+ hits?",
            "floor_strike": 0.5,
            "yes_bid_dollars": "0.54",
            "yes_ask_dollars": "0.56",
            "no_bid_dollars": "0.44",
            "no_ask_dollars": "0.46",
        },
    )

    out = kmc._mlb_hit_candidates([row])
    assert len(out) == 2
    yes = next(x for x in out if x.kalshi_side == "YES")
    no = next(x for x in out if x.kalshi_side == "NO")

    assert yes.market_key == "batter_hits"
    assert yes.selection == "Test Hitter Over 0.5 Hits"
    assert yes.model_probability == 0.64
    assert yes.kalshi_ticker == row.market["ticker"]

    assert no.selection == "Test Hitter Under 0.5 Hits"
    assert abs(no.model_probability - 0.36) < 1e-9


def test_two_hit_contract_uses_one_point_five_line(monkeypatch):
    evidence = ModelEvidence(
        sport="MLB",
        model_name="test MLB Hits",
        fair_probability=0.25,
        confidence=0.65,
        sample_size=350,
    )
    projection = MLBHitProjection(
        evidence=evidence,
        player_name="Test Hitter",
        milestone_hits=2,
        line=1.5,
        game_title="Away vs Home",
        probable_pitcher_name=None,
        expected_at_bats=4.0,
        per_ab_hit_probability=0.25,
    )
    monkeypatch.setattr(kmc, "project_mlb_hits", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="MLB",
        family="Hits",
        market={
            "ticker": "KXMLBHIT-26SEP251900AAABBB-BBBTEST1-2",
            "event_ticker": "KXMLBHIT-26SEP251900AAABBB",
            "series_ticker": "KXMLBHIT",
            "title": "Test Hitter: 2+ hits?",
            "floor_strike": 1.5,
            "yes_bid_dollars": "0.20",
            "yes_ask_dollars": "0.22",
            "no_bid_dollars": "0.78",
            "no_ask_dollars": "0.80",
        },
    )
    out = kmc._mlb_hit_candidates([row])
    assert any(x.selection == "Test Hitter Over 1.5 Hits" for x in out)
    assert any(x.selection == "Test Hitter Under 1.5 Hits" for x in out)
