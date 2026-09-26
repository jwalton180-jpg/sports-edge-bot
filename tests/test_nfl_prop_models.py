from datetime import date

from sports_edge.data.nfl_prop_data import NFLPlayerContext, _resolve_player
from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.nfl_prop_models import (
    NFLPropProjection,
    project_nfl_passing_tds,
    project_nfl_passing_yards,
    project_nfl_receiving_yards,
    project_nfl_touchdowns,
)
import sports_edge.models.kalshi_model_candidates as kmc
import sports_edge.models.nfl_prop_models as npm


EVENT_DATE = date(2026, 9, 27)


def _qb_context():
    current = (
        {"passing_yards": "240", "attempts": "34", "passing_tds": "2", "rushing_tds": "0", "receiving_tds": "0", "carries": "3", "targets": "0"},
        {"passing_yards": "275", "attempts": "38", "passing_tds": "1", "rushing_tds": "1", "receiving_tds": "0", "carries": "5", "targets": "0"},
        {"passing_yards": "255", "attempts": "36", "passing_tds": "3", "rushing_tds": "0", "receiving_tds": "0", "carries": "4", "targets": "0"},
    )
    prior = tuple(
        {
            "passing_yards": str(230 + (i % 5) * 8),
            "attempts": str(32 + (i % 4)),
            "passing_tds": str(1 + (i % 3 == 0)),
            "rushing_tds": str(i % 6 == 0 and 1 or 0),
            "receiving_tds": "0",
            "carries": "4",
            "targets": "0",
        }
        for i in range(12)
    )
    return NFLPlayerContext(
        player_id="qb1",
        player_name="Test Quarterback",
        position="QB",
        team="KC",
        opponent="MIA",
        event_date=EVENT_DATE,
        game_title="KC @ MIA",
        current_rows=current,
        prior_rows=prior,
    )


def _wr_context():
    current = (
        {"receiving_yards": "72", "targets": "9", "target_share": "0.26", "air_yards_share": "0.31", "receiving_tds": "1", "rushing_tds": "0", "carries": "0"},
        {"receiving_yards": "88", "targets": "10", "target_share": "0.29", "air_yards_share": "0.34", "receiving_tds": "0", "rushing_tds": "0", "carries": "1"},
        {"receiving_yards": "61", "targets": "8", "target_share": "0.24", "air_yards_share": "0.28", "receiving_tds": "1", "rushing_tds": "0", "carries": "0"},
    )
    prior = tuple(
        {
            "receiving_yards": str(58 + (i % 5) * 7),
            "targets": str(7 + (i % 4)),
            "target_share": "0.23",
            "air_yards_share": "0.27",
            "receiving_tds": str(i % 4 == 0 and 1 or 0),
            "rushing_tds": "0",
            "carries": "0",
        }
        for i in range(12)
    )
    return NFLPlayerContext(
        player_id="wr1",
        player_name="Test Receiver",
        position="WR",
        team="BUF",
        opponent="LAC",
        event_date=EVENT_DATE,
        game_title="LAC @ BUF",
        current_rows=current,
        prior_rows=prior,
    )


def test_name_resolver_fails_closed_on_ambiguous_alias():
    current = (
        {"player_id": "1", "player_display_name": "John Smith Jr.", "position": "WR"},
        {"player_id": "2", "player_display_name": "John Smith", "position": "RB"},
    )
    assert _resolve_player("John Smith", current, ()) is None


def test_passing_yards_probability_falls_as_milestone_rises(monkeypatch):
    monkeypatch.setattr(npm, "player_context", lambda *args, **kwargs: _qb_context())
    monkeypatch.setattr(npm, "_matchup_factor", lambda *args, **kwargs: (1.0, 3, 240.0))
    low = project_nfl_passing_yards(player_name="Test Quarterback", milestone_yards=225, event_date=EVENT_DATE)
    high = project_nfl_passing_yards(player_name="Test Quarterback", milestone_yards=300, event_date=EVENT_DATE)
    assert low is not None and high is not None
    assert low.evidence.fair_probability > high.evidence.fair_probability
    assert low.evidence.sample_size >= 8
    assert low.line == 224.5


def test_passing_td_probability_is_monotone(monkeypatch):
    monkeypatch.setattr(npm, "player_context", lambda *args, **kwargs: _qb_context())
    monkeypatch.setattr(npm, "_matchup_factor", lambda *args, **kwargs: (1.0, 3, 1.5))
    one = project_nfl_passing_tds(player_name="Test Quarterback", milestone_tds=1, event_date=EVENT_DATE)
    three = project_nfl_passing_tds(player_name="Test Quarterback", milestone_tds=3, event_date=EVENT_DATE)
    assert one is not None and three is not None
    assert one.evidence.fair_probability > three.evidence.fair_probability


def test_receiving_model_uses_target_role_and_prior(monkeypatch):
    monkeypatch.setattr(npm, "player_context", lambda *args, **kwargs: _wr_context())
    monkeypatch.setattr(npm, "_matchup_factor", lambda *args, **kwargs: (1.02, 3, 240.0))
    projection = project_nfl_receiving_yards(
        player_name="Test Receiver",
        milestone_yards=60,
        event_date=EVENT_DATE,
    )
    assert projection is not None
    assert projection.market_key == "player_reception_yds"
    assert projection.projected_mean > 0
    assert any("target share" in factor.lower() for factor in projection.evidence.factors)


def test_player_touchdown_model_excludes_passing_tds(monkeypatch):
    ctx = _qb_context()
    monkeypatch.setattr(npm, "player_context", lambda *args, **kwargs: ctx)
    one = project_nfl_touchdowns(player_name="Test Quarterback", milestone_tds=1, event_date=EVENT_DATE)
    two = project_nfl_touchdowns(player_name="Test Quarterback", milestone_tds=2, event_date=EVENT_DATE)
    assert one is not None and two is not None
    assert one.evidence.fair_probability > two.evidence.fair_probability
    assert any("Passing TDs are excluded" in factor for factor in one.evidence.factors)


def test_receiving_yards_kalshi_market_builds_yes_and_no_candidates(monkeypatch):
    evidence = ModelEvidence(
        sport="NFL",
        model_name="test NFL receiving",
        fair_probability=0.62,
        confidence=0.60,
        sample_size=15,
        factors=("usage/efficiency/prior",),
    )
    projection = NFLPropProjection(
        evidence=evidence,
        player_name="Test Receiver",
        position="WR",
        market_key="player_reception_yds",
        market_label="Receiving Yards",
        milestone=60,
        line=59.5,
        game_title="LAC @ BUF",
        projected_mean=68.0,
        projected_sd=28.0,
    )
    monkeypatch.setattr(kmc, "project_nfl_receiving_yards", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="NFL",
        family="Receiving Yards",
        market={
            "ticker": "KXNFLRECYDS-26SEP27LACBUF-TEST-60",
            "event_ticker": "KXNFLRECYDS-26SEP27LACBUF",
            "series_ticker": "KXNFLRECYDS",
            "title": "Test Receiver: 60+ receiving yards",
            "floor_strike": 59.5,
            "yes_ask_dollars": "0.54",
            "no_ask_dollars": "0.47",
        },
    )
    out = kmc._nfl_prop_candidates([row])
    assert len(out) == 2
    yes = next(x for x in out if x.kalshi_side == "YES")
    no = next(x for x in out if x.kalshi_side == "NO")
    assert yes.selection == "Test Receiver Over 59.5 Receiving Yards"
    assert yes.market_key == "player_reception_yds"
    assert abs(yes.model_probability - 0.62) < 1e-9
    assert no.selection == "Test Receiver Under 59.5 Receiving Yards"
    assert abs(no.model_probability - 0.38) < 1e-9


def test_touchdown_candidate_rejects_defense_and_no_touchdown_pseudo_players(monkeypatch):
    called = []
    monkeypatch.setattr(kmc, "project_nfl_touchdowns", lambda **kwargs: called.append(kwargs))
    rows = [
        KalshiSportMarket(
            sport="NFL",
            family="Player Touchdowns",
            market={
                "ticker": "KXNFLTD-X-DST-1",
                "event_ticker": "KXNFLTD-X",
                "title": "SEA Seahawks D/ST: 1+ touchdowns",
                "floor_strike": 0.5,
                "yes_ask_dollars": "0.05",
                "no_ask_dollars": "0.96",
            },
        ),
        KalshiSportMarket(
            sport="NFL",
            family="Player Touchdowns",
            market={
                "ticker": "KXNFLTD-X-NONE-1",
                "event_ticker": "KXNFLTD-X",
                "title": "No Touchdown: 1+ touchdowns",
                "floor_strike": 0.5,
                "yes_ask_dollars": "0.05",
                "no_ask_dollars": "0.96",
            },
        ),
    ]
    assert kmc._nfl_prop_candidates(rows) == []
    assert called == []
