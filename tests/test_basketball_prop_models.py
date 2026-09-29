from datetime import date

import sports_edge.models.basketball_prop_models as bpm
import sports_edge.models.kalshi_model_candidates as kmc
from sports_edge.models.basketball_prop_models import BasketballPropProjection, _nb_tail
from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.wnba_lines_model import _Matchup, _TeamGame


EVENT_DATE = date(2026, 9, 29)


def _player_row(day, *, pts, reb, ast, threes, minutes=34, team="IND", opp="LV"):
    return {
        "game_date": day,
        "athlete_display_name": "Caitlin Clark",
        "team_abbreviation": team,
        "team_display_name": "Indiana Fever",
        "opponent_team_abbreviation": opp,
        "minutes": str(minutes),
        "points": str(pts),
        "rebounds": str(reb),
        "assists": str(ast),
        "three_point_field_goals_made": str(threes),
    }


def _matchup():
    away_games = tuple(
        _TeamGame(EVENT_DATE, 88 + i % 4, 84 + i % 5, False)
        for i in range(12)
    )
    home_games = tuple(
        _TeamGame(EVENT_DATE, 89 + i % 5, 86 + i % 4, True)
        for i in range(12)
    )
    return _Matchup(
        away_id=1,
        home_id=2,
        away_name="Las Vegas Aces",
        home_name="Indiana Fever",
        away_abbr="LV",
        home_abbr="IND",
        away_games=away_games,
        home_games=home_games,
        venue_confirmed=True,
    )


def test_nb_tail_falls_as_milestone_rises():
    assert _nb_tail(20.0, 35.0, 15) > _nb_tail(20.0, 35.0, 20)
    assert _nb_tail(20.0, 35.0, 20) > _nb_tail(20.0, 35.0, 25)


def test_wnba_player_projection_uses_minutes_role_and_environment(monkeypatch):
    current = tuple(
        _player_row(
            f"2026-09-{10+i:02d}",
            pts=18 + i % 7,
            reb=5 + i % 4,
            ast=7 + i % 5,
            threes=2 + i % 3,
            minutes=31 + (i % 5),
        )
        for i in range(12)
    )
    prior = tuple(
        _player_row(
            f"2025-08-{1+i:02d}",
            pts=17 + i % 6,
            reb=5 + i % 3,
            ast=6 + i % 4,
            threes=2 + i % 2,
            minutes=32 + (i % 4),
        )
        for i in range(12)
    )

    monkeypatch.setattr(bpm, "_pregame_only", lambda *args, **kwargs: True)
    monkeypatch.setattr(bpm, "_resolve_matchup", lambda *args, **kwargs: _matchup())
    monkeypatch.setattr(bpm, "player_history", lambda *args, **kwargs: current)
    monkeypatch.setattr(bpm, "prior_player_history", lambda *args, **kwargs: prior)
    monkeypatch.setattr(
        bpm,
        "_score_projection",
        lambda matchup: (88.0, 94.0, 6.0, 182.0, 13.0),
    )

    projection = bpm.project_wnba_player_prop(
        player_name="Caitlin Clark",
        family="Points",
        milestone=20,
        event_date=EVENT_DATE,
        event_ticker="KXWNBAPTS-26SEP29LVIND",
    )
    assert projection is not None
    assert projection.market_key == "player_points"
    assert projection.game_title == "Las Vegas Aces @ Indiana Fever"
    assert projection.expected_minutes > 30
    assert projection.projected_mean > 10
    assert 0 < projection.evidence.fair_probability < 1
    assert projection.evidence.sample_size >= 20
    assert any("Projected team score" in x for x in projection.evidence.factors)


def test_wnba_player_projection_fails_closed_if_player_team_not_in_game(monkeypatch):
    current = tuple(
        _player_row(
            f"2026-09-{10+i:02d}",
            pts=20,
            reb=5,
            ast=8,
            threes=3,
            team="NY",
            opp="MIN",
        )
        for i in range(10)
    )
    monkeypatch.setattr(bpm, "_pregame_only", lambda *args, **kwargs: True)
    monkeypatch.setattr(bpm, "_resolve_matchup", lambda *args, **kwargs: _matchup())
    monkeypatch.setattr(bpm, "player_history", lambda *args, **kwargs: current)
    monkeypatch.setattr(bpm, "prior_player_history", lambda *args, **kwargs: current)

    assert bpm.project_wnba_player_prop(
        player_name="Caitlin Clark",
        family="Points",
        milestone=20,
        event_date=EVENT_DATE,
        event_ticker="KXWNBAPTS-26SEP29LVIND",
    ) is None


def _projection():
    evidence = ModelEvidence(
        sport="WNBA",
        model_name="test WNBA prop",
        fair_probability=0.62,
        confidence=0.64,
        sample_size=30,
        factors=("minutes role",),
    )
    return BasketballPropProjection(
        evidence=evidence,
        player_name="Caitlin Clark",
        market_key="player_assists",
        market_label="Assists",
        milestone=10,
        line=9.5,
        game_title="Las Vegas Aces @ Indiana Fever",
        projected_mean=10.8,
        projected_variance=12.0,
        expected_minutes=36.0,
    )


def test_wnba_candidate_builds_yes_no_model_sides(monkeypatch):
    monkeypatch.setattr(kmc, "project_wnba_player_prop", lambda **kwargs: _projection())
    row = KalshiSportMarket(
        sport="WNBA",
        family="Assists",
        market={
            "ticker": "KXWNBAAST-26SEP29LVIND-CC10",
            "event_ticker": "KXWNBAAST-26SEP29LVIND",
            "series_ticker": "KXWNBAAST",
            "title": "Caitlin Clark: 10+ assists",
            "floor_strike": 9.5,
            "yes_ask_dollars": "0.52",
            "no_ask_dollars": "0.49",
        },
    )
    out = kmc._wnba_player_candidate_for_market(row)
    assert len(out) == 2
    assert {x.kalshi_side for x in out} == {"YES", "NO"}
    assert {x.selection for x in out} == {
        "Caitlin Clark Over 9.5 Assists",
        "Caitlin Clark Under 9.5 Assists",
    }
    assert all(x.event_id == "WNBA:2026-09-29:indiana fever|las vegas aces" for x in out)
