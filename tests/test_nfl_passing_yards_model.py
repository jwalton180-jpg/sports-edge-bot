from datetime import date

from sports_edge.data.nfl_prop_data import NFLPassingContext
from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.nfl_passing_yards_model import (
    NFLPassingProjection,
    project_nfl_passing_yards,
)
import sports_edge.models.kalshi_model_candidates as kmc
import sports_edge.models.nfl_passing_yards_model as pmodel


def _context(rows=3, prior_rows_count=8):
    player_rows = []
    values = [
        (31, 248, 5.7),
        (37, 277, 8.2),
        (34, 262, 3.1),
    ][:rows]
    for week, (attempts, yards, cpoe) in enumerate(values, start=1):
        player_rows.append(
            {
                "week": str(week),
                "attempts": str(attempts),
                "passing_yards": str(yards),
                "passing_cpoe": str(cpoe),
            }
        )

    prior_player_rows = []
    prior_values = [
        (32, 241, 1.8),
        (35, 266, 4.1),
        (30, 218, -0.5),
        (36, 281, 5.0),
        (34, 254, 2.3),
        (29, 207, -1.1),
        (38, 296, 6.2),
        (33, 245, 1.0),
    ][:prior_rows_count]
    for week, (attempts, yards, cpoe) in enumerate(prior_values, start=1):
        prior_player_rows.append(
            {
                "week": str(week),
                "attempts": str(attempts),
                "passing_yards": str(yards),
                "passing_cpoe": str(cpoe),
            }
        )

    opponent_allowed = (
        {"week": "1", "attempts": "35", "passing_yards": "255"},
        {"week": "2", "attempts": "31", "passing_yards": "218"},
        {"week": "3", "attempts": "39", "passing_yards": "289"},
    )
    league = (
        {"week": "1", "attempts": "34", "passing_yards": "238"},
        {"week": "1", "attempts": "31", "passing_yards": "211"},
        {"week": "2", "attempts": "36", "passing_yards": "260"},
        {"week": "2", "attempts": "29", "passing_yards": "196"},
        {"week": "3", "attempts": "33", "passing_yards": "229"},
        {"week": "3", "attempts": "35", "passing_yards": "251"},
    )

    return NFLPassingContext(
        player_id="00-test",
        player_name="Test Quarterback",
        team="KC",
        opponent="DEN",
        week=4,
        game_date=date(2026, 9, 27),
        game_id="2026_04_DEN_KC",
        home=True,
        roster_status="ACT",
        roster_week=4,
        scheduled_qb_name="Test Quarterback",
        player_rows=tuple(player_rows),
        prior_player_rows=tuple(prior_player_rows),
        opponent_allowed_rows=opponent_allowed,
        league_rows=league,
    )


def test_passing_model_produces_reasonable_projection(monkeypatch):
    monkeypatch.setattr(pmodel, "resolve_passing_context", lambda **kwargs: _context())

    projection = project_nfl_passing_yards(
        player_name="Test Quarterback",
        milestone_yards=250,
        event_date=date(2026, 9, 27),
    )
    assert projection is not None
    assert 20 <= projection.expected_attempts <= 48
    assert 4.5 <= projection.adjusted_ypa <= 10.5
    assert 150 <= projection.expected_passing_yards <= 350
    assert 48 <= projection.passing_yards_sd <= 105
    assert 0 < projection.evidence.fair_probability < 1
    assert projection.evidence.confidence >= 0.45
    assert any("Opponent pass defense" in x for x in projection.evidence.factors)


def test_passing_model_fails_closed_on_thin_sample_without_prior(monkeypatch):
    monkeypatch.setattr(
        pmodel,
        "resolve_passing_context",
        lambda **kwargs: _context(rows=1, prior_rows_count=0),
    )
    assert (
        project_nfl_passing_yards(
            player_name="Test Quarterback",
            milestone_yards=250,
            event_date=date(2026, 9, 27),
        )
        is None
    )


def test_higher_passing_line_has_lower_probability(monkeypatch):
    monkeypatch.setattr(pmodel, "resolve_passing_context", lambda **kwargs: _context())
    low = project_nfl_passing_yards(
        player_name="Test Quarterback",
        milestone_yards=225,
        event_date=date(2026, 9, 27),
    )
    high = project_nfl_passing_yards(
        player_name="Test Quarterback",
        milestone_yards=300,
        event_date=date(2026, 9, 27),
    )
    assert low is not None and high is not None
    assert low.evidence.fair_probability > high.evidence.fair_probability


def test_kalshi_passing_market_builds_yes_and_no_candidates(monkeypatch):
    evidence = ModelEvidence(
        sport="NFL",
        model_name="test NFL passing",
        fair_probability=0.64,
        confidence=0.70,
        sample_size=102,
        factors=("attempts + YPA + opponent",),
    )
    projection = NFLPassingProjection(
        evidence=evidence,
        player_name="Test Quarterback",
        milestone_yards=250,
        line=249.5,
        game_title="KC vs DEN",
        team="KC",
        opponent="DEN",
        target_week=4,
        expected_attempts=34.0,
        adjusted_ypa=7.55,
        expected_passing_yards=256.7,
        passing_yards_sd=67.0,
    )
    monkeypatch.setattr(kmc, "project_nfl_passing_yards", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="NFL",
        family="Passing Yards",
        market={
            "ticker": "KXNFLPASSYDS-26SEP27KCDEN-TESTQB-250",
            "event_ticker": "KXNFLPASSYDS-26SEP27KCDEN",
            "series_ticker": "KXNFLPASSYDS",
            "title": "Test Quarterback: 250+ passing yards?",
            "floor_strike": 249.5,
            "yes_bid_dollars": "0.55",
            "yes_ask_dollars": "0.57",
            "no_bid_dollars": "0.43",
            "no_ask_dollars": "0.45",
        },
    )

    out = kmc._nfl_passing_candidates([row])
    assert len(out) == 2
    yes = next(x for x in out if x.kalshi_side == "YES")
    no = next(x for x in out if x.kalshi_side == "NO")

    assert yes.market_key == "player_pass_yds"
    assert yes.selection == "Test Quarterback Over 249.5 Passing Yards"
    assert abs(yes.model_probability - 0.64) < 1e-9
    assert no.selection == "Test Quarterback Under 249.5 Passing Yards"
    assert abs(no.model_probability - 0.36) < 1e-9


def test_passing_title_fallback_parses_threshold(monkeypatch):
    evidence = ModelEvidence(
        sport="NFL",
        model_name="test NFL passing",
        fair_probability=0.52,
        confidence=0.66,
        sample_size=90,
    )
    projection = NFLPassingProjection(
        evidence=evidence,
        player_name="Test Quarterback",
        milestone_yards=275,
        line=274.5,
        game_title="KC vs DEN",
        team="KC",
        opponent="DEN",
        target_week=4,
        expected_attempts=35.0,
        adjusted_ypa=7.4,
        expected_passing_yards=259.0,
        passing_yards_sd=70.0,
    )
    monkeypatch.setattr(kmc, "project_nfl_passing_yards", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="NFL",
        family="Passing Yards",
        market={
            "ticker": "KXNFLPASSYDS-26SEP27KCDEN-TESTQB-275",
            "event_ticker": "KXNFLPASSYDS-26SEP27KCDEN",
            "series_ticker": "KXNFLPASSYDS",
            "title": "Test Quarterback: 275+ passing yards?",
            "yes_ask_dollars": "0.45",
            "no_ask_dollars": "0.55",
        },
    )
    out = kmc._nfl_passing_candidates([row])
    assert any(x.selection == "Test Quarterback Over 274.5 Passing Yards" for x in out)



def test_two_game_sample_is_stabilized_by_prior_season(monkeypatch):
    monkeypatch.setattr(
        pmodel,
        "resolve_passing_context",
        lambda **kwargs: _context(rows=2, prior_rows_count=8),
    )
    projection = project_nfl_passing_yards(
        player_name="Test Quarterback",
        milestone_yards=250,
        event_date=date(2026, 9, 27),
    )
    assert projection is not None
    assert projection.evidence.confidence <= 0.58
    assert any("Decayed prior-season baseline" in x for x in projection.evidence.factors)
    assert any("stabilized with decayed prior-season" in x for x in projection.evidence.warnings)
