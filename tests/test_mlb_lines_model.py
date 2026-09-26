from datetime import date

from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.mlb_lines_model import (
    MLBLineProjection,
    _poisson_over,
    _margin_over,
)
from sports_edge.models.model_evidence import ModelEvidence
import sports_edge.models.kalshi_model_candidates as kmc


def test_mlb_poisson_over_probability_falls_as_line_rises():
    low = _poisson_over(7.5, 9.1)
    high = _poisson_over(10.5, 9.1)
    assert 0 < high < low < 1


def test_mlb_margin_probability_is_directional():
    favored = _margin_over(1.5, 5.3, 3.7)
    dog = _margin_over(1.5, 3.7, 5.3)
    assert favored > dog


def _projection(key, label, fair=0.61):
    evidence = ModelEvidence(
        sport="MLB",
        model_name=f"test {label}",
        fair_probability=fair,
        confidence=0.63,
        sample_size=120,
        factors=("team scoring/allowance",),
    )
    return MLBLineProjection(
        evidence=evidence,
        market_key=key,
        market_label=label,
        game_title="Los Angeles Dodgers @ San Francisco Giants",
        selection_label=(
            "Los Angeles Dodgers margin > 1.5"
            if key == "mlb_spread"
            else "Over 8.5 Game Total"
            if key == "mlb_game_total"
            else "Los Angeles Dodgers over 4.5 runs"
        ),
        line=1.5 if key == "mlb_spread" else 8.5 if key == "mlb_game_total" else 4.5,
        projected_value=1.8 if key == "mlb_spread" else 8.9 if key == "mlb_game_total" else 4.9,
    )


def test_mlb_line_candidates_build_yes_no_and_canonical_game(monkeypatch):
    monkeypatch.setattr(kmc, "project_mlb_spread", lambda **kwargs: _projection("mlb_spread", "Spread"))
    monkeypatch.setattr(kmc, "project_mlb_game_total", lambda **kwargs: _projection("mlb_game_total", "Game Total"))

    spread = KalshiSportMarket(
        sport="MLB",
        family="Spread",
        market={
            "ticker": "KXMLBSPREAD-26SEP25LADSF-LAD2",
            "event_ticker": "KXMLBSPREAD-26SEP25LADSF",
            "event_title": "Los Angeles Dodgers at San Francisco Giants",
            "series_ticker": "KXMLBSPREAD",
            "title": "Los Angeles D wins by over 1.5 runs?",
            "floor_strike": 1.5,
            "yes_ask_dollars": "0.52",
            "no_ask_dollars": "0.49",
        },
    )
    total = KalshiSportMarket(
        sport="MLB",
        family="Game Total",
        market={
            "ticker": "KXMLBTOTAL-26SEP25LADSF-9",
            "event_ticker": "KXMLBTOTAL-26SEP25LADSF",
            "event_title": "Los Angeles Dodgers at San Francisco Giants",
            "series_ticker": "KXMLBTOTAL",
            "title": "Over 8.5 runs scored",
            "floor_strike": 8.5,
            "yes_ask_dollars": "0.53",
            "no_ask_dollars": "0.48",
        },
    )

    spread_out = kmc._mlb_line_candidate_for_market(spread)
    total_out = kmc._mlb_line_candidate_for_market(total)

    assert len(spread_out) == 2
    assert len(total_out) == 2
    assert spread_out[0].event_id == total_out[0].event_id
    assert {x.kalshi_side for x in spread_out} == {"YES", "NO"}
    assert any("Under 8.5 Game Total" == x.selection for x in total_out)
