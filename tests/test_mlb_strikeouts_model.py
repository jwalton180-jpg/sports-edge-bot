from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.mlb_strikeouts_model import (
    MLBStrikeoutProjection,
    poisson_at_least_probability,
    project_mlb_pitcher_strikeouts,
)
import sports_edge.models.kalshi_model_candidates as kmc
import sports_edge.models.mlb_strikeouts_model as kmodel


def test_poisson_strikeout_probability_is_monotone():
    p5 = poisson_at_least_probability(6.4, 5)
    p7 = poisson_at_least_probability(6.4, 7)
    p9 = poisson_at_least_probability(6.4, 9)
    assert 0 < p9 < p7 < p5 < 1


def test_higher_expected_strikeouts_raise_threshold_probability():
    low = poisson_at_least_probability(4.5, 6)
    high = poisson_at_least_probability(7.0, 6)
    assert high > low


def test_non_probable_starter_fails_closed(monkeypatch):
    monkeypatch.setattr(
        kmodel,
        "resolve_player",
        lambda *args, **kwargs: {
            "id": 10,
            "fullName": "Test Pitcher",
            "currentTeam": {"id": 100},
        },
    )
    monkeypatch.setattr(
        kmodel,
        "find_player_game",
        lambda **kwargs: {"gamePk": 1},
    )
    monkeypatch.setattr(
        kmodel,
        "game_context_for_pitcher",
        lambda *args, **kwargs: {
            "game_status": "Preview",
            "is_probable_starter": False,
        },
    )

    assert (
        project_mlb_pitcher_strikeouts(
            player_name="Test Pitcher",
            milestone_strikeouts=6,
            event_date=__import__("datetime").date(2026, 9, 25),
            event_ticker="KXMLBKS-26SEP251900AAABBB",
        )
        is None
    )


def test_mlb_strikeout_market_builds_yes_and_no_candidates(monkeypatch):
    evidence = ModelEvidence(
        sport="MLB",
        model_name="test MLB Ks",
        fair_probability=0.61,
        confidence=0.72,
        sample_size=620,
        factors=("K/BF + opponent tendency",),
    )
    projection = MLBStrikeoutProjection(
        evidence=evidence,
        player_name="Test Pitcher",
        milestone_strikeouts=6,
        line=5.5,
        game_title="Away vs Home",
        opponent_team_name="Opponent",
        expected_batters_faced=23.5,
        strikeout_rate_per_bf=0.27,
        expected_strikeouts=6.35,
    )

    monkeypatch.setattr(kmc, "project_mlb_pitcher_strikeouts", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="MLB",
        family="Strikeouts",
        market={
            "ticker": "KXMLBKS-26SEP251900AAABBB-TESTP-6",
            "event_ticker": "KXMLBKS-26SEP251900AAABBB",
            "series_ticker": "KXMLBKS",
            "title": "Test Pitcher: 6+ strikeouts?",
            "floor_strike": 5.5,
            "yes_bid_dollars": "0.51",
            "yes_ask_dollars": "0.53",
            "no_bid_dollars": "0.47",
            "no_ask_dollars": "0.49",
        },
    )

    out = kmc._mlb_k_candidates([row])
    assert len(out) == 2

    yes = next(x for x in out if x.kalshi_side == "YES")
    no = next(x for x in out if x.kalshi_side == "NO")

    assert yes.market_key == "pitcher_strikeouts"
    assert yes.selection == "Test Pitcher Over 5.5 Strikeouts"
    assert abs(yes.model_probability - 0.61) < 1e-9
    assert yes.kalshi_ticker == row.market["ticker"]

    assert no.selection == "Test Pitcher Under 5.5 Strikeouts"
    assert abs(no.model_probability - 0.39) < 1e-9


def test_strikeout_title_fallback_parses_milestone(monkeypatch):
    evidence = ModelEvidence(
        sport="MLB",
        model_name="test MLB Ks",
        fair_probability=0.48,
        confidence=0.66,
        sample_size=500,
    )
    projection = MLBStrikeoutProjection(
        evidence=evidence,
        player_name="Test Pitcher",
        milestone_strikeouts=8,
        line=7.5,
        game_title="Away vs Home",
        opponent_team_name="Opponent",
        expected_batters_faced=22.0,
        strikeout_rate_per_bf=0.30,
        expected_strikeouts=6.6,
    )
    monkeypatch.setattr(kmc, "project_mlb_pitcher_strikeouts", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="MLB",
        family="Strikeouts",
        market={
            "ticker": "KXMLBKS-26SEP251900AAABBB-TESTP-8",
            "event_ticker": "KXMLBKS-26SEP251900AAABBB",
            "series_ticker": "KXMLBKS",
            "title": "Test Pitcher: 8+ strikeouts?",
            "yes_ask_dollars": "0.40",
            "no_ask_dollars": "0.60",
        },
    )
    out = kmc._mlb_k_candidates([row])
    assert any(x.selection == "Test Pitcher Over 7.5 Strikeouts" for x in out)
