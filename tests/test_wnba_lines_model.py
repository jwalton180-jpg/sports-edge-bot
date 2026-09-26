from datetime import date, timedelta

from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.wnba_lines_model import (
    WNBALineProjection,
    project_wnba_game_total,
    project_wnba_spread,
    project_wnba_team_total,
)
import sports_edge.models.kalshi_model_candidates as kmc
import sports_edge.models.wnba_lines_model as wlm


EVENT_DATE = date(2026, 9, 27)


def _row(
    gd,
    home_id,
    away_id,
    home_name,
    away_name,
    home_abbr,
    away_abbr,
    home_score="",
    away_score="",
    completed=True,
):
    return {
        "game_date": gd.isoformat(),
        "season_type": "2",
        "status_type_completed": "true" if completed else "false",
        "home_id": str(home_id),
        "away_id": str(away_id),
        "home_score": str(home_score),
        "away_score": str(away_score),
        "home_display_name": home_name,
        "away_display_name": away_name,
        "home_name": home_name.split()[-1],
        "away_name": away_name.split()[-1],
        "home_location": " ".join(home_name.split()[:-1]),
        "away_location": " ".join(away_name.split()[:-1]),
        "home_abbreviation": home_abbr,
        "away_abbreviation": away_abbr,
    }


def _schedule():
    rows = []
    for i in range(14):
        gd = EVENT_DATE - timedelta(days=40 - i * 2)
        rows.append(
            _row(
                gd,
                1,
                3,
                "Atlanta Dream",
                "Connecticut Sun",
                "ATL",
                "CON",
                88 + (i % 5),
                79 + (i % 4),
            )
        )
        rows.append(
            _row(
                gd,
                4,
                2,
                "Chicago Sky",
                "Washington Mystics",
                "CHI",
                "WSH",
                80 + (i % 4),
                84 + (i % 5),
            )
        )
    rows.append(
        _row(
            EVENT_DATE,
            1,
            2,
            "Atlanta Dream",
            "Washington Mystics",
            "ATL",
            "WSH",
            completed=False,
        )
    )
    return tuple(rows)


def _patch_schedule(monkeypatch):
    monkeypatch.setattr(wlm, "_wnba_schedule_rows", lambda: _schedule())
    wlm._resolve_matchup.cache_clear()


def test_wnba_spread_probability_falls_as_margin_line_rises(monkeypatch):
    _patch_schedule(monkeypatch)
    low = project_wnba_spread(
        team_name="Atlanta",
        line=2.5,
        event_date=EVENT_DATE,
        event_ticker="KXWNBASPREAD-26SEP27WSHATL",
    )
    high = project_wnba_spread(
        team_name="Atlanta",
        line=10.5,
        event_date=EVENT_DATE,
        event_ticker="KXWNBASPREAD-26SEP27WSHATL",
    )
    assert low is not None and high is not None
    assert low.evidence.fair_probability > high.evidence.fair_probability
    assert low.game_title == "Washington Mystics @ Atlanta Dream"
    assert low.evidence.sample_size >= 8


def test_wnba_total_probability_is_monotone(monkeypatch):
    _patch_schedule(monkeypatch)
    low = project_wnba_game_total(
        line=160.5,
        event_date=EVENT_DATE,
        event_ticker="KXWNBATOTAL-26SEP27WSHATL",
    )
    high = project_wnba_game_total(
        line=185.5,
        event_date=EVENT_DATE,
        event_ticker="KXWNBATOTAL-26SEP27WSHATL",
    )
    assert low is not None and high is not None
    assert low.evidence.fair_probability > high.evidence.fair_probability


def test_wnba_team_total_resolves_city_name(monkeypatch):
    _patch_schedule(monkeypatch)
    projection = project_wnba_team_total(
        team_name="Washington",
        line=80.5,
        event_date=EVENT_DATE,
        event_ticker="KXWNBATEAMTOTAL-26SEP27WSHATL",
    )
    assert projection is not None
    assert projection.market_key == "wnba_team_total"
    assert "Washington Mystics" in projection.selection_label
    assert projection.projected_value > 0


def test_wnba_game_total_kalshi_candidate_builds_yes_and_no(monkeypatch):
    evidence = ModelEvidence(
        sport="WNBA",
        model_name="test WNBA total",
        fair_probability=0.61,
        confidence=0.63,
        sample_size=36,
        factors=("scoring/defense history",),
    )
    projection = WNBALineProjection(
        evidence=evidence,
        market_key="wnba_game_total",
        market_label="Game Total",
        game_title="Washington Mystics @ Atlanta Dream",
        selection_label="Over 169.5 total points",
        line=169.5,
        projected_value=174.0,
        projected_sd=14.0,
    )
    monkeypatch.setattr(kmc, "project_wnba_game_total", lambda **kwargs: projection)

    row = KalshiSportMarket(
        sport="WNBA",
        family="Game Total",
        market={
            "ticker": "KXWNBATOTAL-26SEP27WSHATL-170",
            "event_ticker": "KXWNBATOTAL-26SEP27WSHATL",
            "series_ticker": "KXWNBATOTAL",
            "title": "Over 169.5 points scored",
            "floor_strike": 169.5,
            "yes_ask_dollars": "0.52",
            "no_ask_dollars": "0.49",
        },
    )
    out = kmc._wnba_line_candidates([row])
    assert len(out) == 2
    yes = next(x for x in out if x.kalshi_side == "YES")
    no = next(x for x in out if x.kalshi_side == "NO")
    assert yes.market_key == "wnba_game_total"
    assert yes.selection == "Over 169.5 Game Total"
    assert abs(yes.model_probability - 0.61) < 1e-9
    assert no.selection == "Under 169.5 Game Total"
    assert abs(no.model_probability - 0.39) < 1e-9
    assert yes.event_id == no.event_id
    assert yes.event_id.startswith("WNBA:2026-09-27:")


def test_wnba_spread_candidate_uses_exact_team_title(monkeypatch):
    evidence = ModelEvidence(
        sport="WNBA",
        model_name="test WNBA spread",
        fair_probability=0.57,
        confidence=0.62,
        sample_size=40,
    )
    projection = WNBALineProjection(
        evidence=evidence,
        market_key="wnba_spread",
        market_label="Spread",
        game_title="Washington Mystics @ Atlanta Dream",
        selection_label="Atlanta Dream margin > 6.5",
        line=6.5,
        projected_value=8.2,
        projected_sd=12.0,
    )
    monkeypatch.setattr(kmc, "project_wnba_spread", lambda **kwargs: projection)
    row = KalshiSportMarket(
        sport="WNBA",
        family="Spread",
        market={
            "ticker": "KXWNBASPREAD-26SEP27WSHATL-ATL7",
            "event_ticker": "KXWNBASPREAD-26SEP27WSHATL",
            "series_ticker": "KXWNBASPREAD",
            "title": "Atlanta wins the game by over 6.5 points",
            "floor_strike": 6.5,
            "yes_ask_dollars": "0.51",
            "no_ask_dollars": "0.50",
        },
    )
    out = kmc._wnba_line_candidates([row])
    assert any(x.selection == "Atlanta Dream margin > 6.5" for x in out)
    assert any("≤ 6.5" in x.selection for x in out)
