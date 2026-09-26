from datetime import date

from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.nfl_lines_model import (
    NFLLinesProjection,
    _resolve_matchup,
    project_nfl_game_total,
    project_nfl_spread,
    project_nfl_team_total,
)
import sports_edge.models.kalshi_model_candidates as kmc
import sports_edge.models.nfl_lines_model as nlm


EVENT_DATE = date(2026, 9, 27)
EVENT_TICKER = "KXNFLSPREAD-26SEP27LACBUF"


def _row(season, gameday, away, home, away_score, home_score):
    return {
        "season": str(season),
        "game_type": "REG",
        "gameday": gameday,
        "away_team": away,
        "home_team": home,
        "away_score": str(away_score),
        "home_score": str(home_score),
    }


def _schedule():
    rows = [
        # Exact upcoming matchup.
        _row(2026, "2026-09-27", "LAC", "BUF", "", ""),
        # Current LAC.
        _row(2026, "2026-09-06", "LAC", "KC", 27, 24),
        _row(2026, "2026-09-13", "LV", "LAC", 17, 24),
        _row(2026, "2026-09-20", "LAC", "DEN", 20, 16),
        # Current BUF.
        _row(2026, "2026-09-06", "BAL", "BUF", 24, 31),
        _row(2026, "2026-09-13", "BUF", "NYJ", 28, 14),
        _row(2026, "2026-09-20", "MIA", "BUF", 20, 27),
    ]
    # Prior-season baseline for both clubs.
    for idx in range(12):
        day = 7 + idx
        rows.append(_row(2025, f"2025-10-{day:02d}", "LAC", "DEN", 21 + idx % 7, 17 + idx % 5))
        rows.append(_row(2025, f"2025-11-{day:02d}", "MIA", "BUF", 18 + idx % 5, 25 + idx % 8))
    return tuple(rows)


def _patch_schedule(monkeypatch):
    monkeypatch.setattr(nlm, "schedule_rows", _schedule)
    nlm._resolve_matchup.cache_clear()


def test_nfl_game_total_probability_falls_as_line_rises(monkeypatch):
    _patch_schedule(monkeypatch)
    low = project_nfl_game_total(
        line=41.5,
        event_date=EVENT_DATE,
        event_ticker="KXNFLTOTAL-26SEP27LACBUF",
    )
    high = project_nfl_game_total(
        line=55.5,
        event_date=EVENT_DATE,
        event_ticker="KXNFLTOTAL-26SEP27LACBUF",
    )
    assert low is not None and high is not None
    assert low.evidence.fair_probability > high.evidence.fair_probability
    assert 35.0 < low.projected_value < 60.0
    assert low.evidence.sample_size >= 8


def test_nfl_spread_resolves_kalshi_code_plus_mascot(monkeypatch):
    _patch_schedule(monkeypatch)
    projection = project_nfl_spread(
        team_name="LAC Chargers",
        line=3.5,
        event_date=EVENT_DATE,
        event_ticker=EVENT_TICKER,
    )
    assert projection is not None
    assert projection.market_key == "nfl_spread"
    assert projection.game_title == "LAC @ BUF"
    assert projection.selection_label.startswith("LAC margin >")


def test_nfl_team_total_resolves_abbreviated_title(monkeypatch):
    _patch_schedule(monkeypatch)
    projection = project_nfl_team_total(
        team_name="BUF Bills",
        line=24.5,
        event_date=EVENT_DATE,
        event_ticker="KXNFLTEAMTOTAL-26SEP27LACBUF",
    )
    assert projection is not None
    assert projection.market_key == "nfl_team_total"
    assert projection.selection_label.startswith("BUF over 24.5")
    assert 7 <= projection.projected_sd <= 17


def test_nfl_line_model_fails_closed_on_wrong_kalshi_game(monkeypatch):
    _patch_schedule(monkeypatch)
    projection = project_nfl_game_total(
        line=45.5,
        event_date=EVENT_DATE,
        event_ticker="KXNFLTOTAL-26SEP27KCMIA",
    )
    assert projection is None


def _projection(key, label, fair=0.61):
    evidence = ModelEvidence(
        sport="NFL",
        model_name=f"test {label}",
        fair_probability=fair,
        confidence=0.62,
        sample_size=15,
        factors=("current/prior scoring",),
    )
    return NFLLinesProjection(
        evidence=evidence,
        market_key=key,
        market_label=label,
        game_title="LAC @ BUF",
        selection_label=(
            "LAC margin > 3.5" if key == "nfl_spread"
            else "Over 45.5 Game Total" if key == "nfl_game_total"
            else "BUF over 24.5 points"
        ),
        line=3.5 if key == "nfl_spread" else 45.5 if key == "nfl_game_total" else 24.5,
        projected_value=5.0 if key == "nfl_spread" else 47.0 if key == "nfl_game_total" else 26.0,
        projected_sd=12.0,
    )


def test_nfl_line_candidates_build_yes_no_and_share_canonical_game(monkeypatch):
    monkeypatch.setattr(kmc, "project_nfl_spread", lambda **kwargs: _projection("nfl_spread", "Spread"))
    monkeypatch.setattr(kmc, "project_nfl_game_total", lambda **kwargs: _projection("nfl_game_total", "Game Total"))

    spread = KalshiSportMarket(
        sport="NFL",
        family="Spread",
        market={
            "ticker": "KXNFLSPREAD-26SEP27LACBUF-LAC4",
            "event_ticker": "KXNFLSPREAD-26SEP27LACBUF",
            "series_ticker": "KXNFLSPREAD",
            "title": "LAC Chargers wins by over 3.5 points?",
            "floor_strike": 3.5,
            "yes_ask_dollars": "0.52",
            "no_ask_dollars": "0.49",
        },
    )
    total = KalshiSportMarket(
        sport="NFL",
        family="Game Total",
        market={
            "ticker": "KXNFLTOTAL-26SEP27LACBUF-46",
            "event_ticker": "KXNFLTOTAL-26SEP27LACBUF",
            "series_ticker": "KXNFLTOTAL",
            "title": "Full Game: over 45.5 points scored?",
            "floor_strike": 45.5,
            "yes_ask_dollars": "0.53",
            "no_ask_dollars": "0.48",
        },
    )

    spread_out = kmc._nfl_line_candidate_for_market(spread)
    total_out = kmc._nfl_line_candidate_for_market(total)
    assert len(spread_out) == 2
    assert len(total_out) == 2
    assert spread_out[0].event_id == total_out[0].event_id
    assert {x.kalshi_side for x in spread_out} == {"YES", "NO"}
    assert any(x.selection == "LAC margin > 3.5" for x in spread_out)
    assert any(x.selection == "Under 45.5 Game Total" for x in total_out)
