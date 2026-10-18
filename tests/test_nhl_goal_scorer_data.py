from unittest.mock import Mock

import pytest

from sports_edge.data import nhl_goal_scorers as nhl


def test_player_game_log_normalizes_and_rejects_inconsistent_rows(monkeypatch):
    monkeypatch.setattr(nhl, "_get_json", lambda *a, **k: {
        "gameLog": [
            {"gameId": 2, "gameDate": "2026-10-15", "goals": 1,
             "assists": 2, "points": 3, "shots": 4, "powerPlayGoals": 1},
            {"gameId": 1, "gameDate": "2026-10-14", "goals": 1,
             "assists": 0, "points": 9, "shots": 2, "powerPlayGoals": 0},
            {"gameId": 3, "gameDate": "2026-10-16", "goals": None,
             "assists": 0, "points": 0, "shots": 1},
        ]
    })
    rows = nhl.player_game_log(8478402, "20252026")
    assert len(rows) == 1
    assert rows[0]["source"] == "NHL"
    assert rows[0]["points"] == rows[0]["goals"] + rows[0]["assists"]


def test_player_game_log_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        nhl.player_game_log(0, "20252026")
    with pytest.raises(ValueError):
        nhl.player_game_log(8478402, "2025")
    with pytest.raises(ValueError):
        nhl.player_game_log(8478402, "20252026", 1)


def test_schedule_filters_requested_date(monkeypatch):
    monkeypatch.setattr(nhl, "_get_json", lambda *a, **k: {
        "gameWeek": [
            {"date": "2026-10-17", "games": [{"id": 1}]},
            {"date": "2026-10-18", "games": [{"id": 2}]},
        ]
    })
    assert nhl.schedule_for_day("2026-10-17") == ({"id": 1},)


def test_candidate_player_ids_only_returns_explicit_ids():
    games = ({
        "homeTeam": {"players": [{"playerId": 12}, {"id": 13}, {"name": "unknown"}]},
        "awayTeam": {"players": [{"playerId": "12"}, {"playerId": -1}]},
    },)
    assert nhl.candidate_player_ids(games) == (12, 13)


def test_goal_probability_fails_closed_on_short_history():
    assert nhl.probability_goal_1plus(tuple()) == {
        "probability": None, "status": "INSUFFICIENT_HISTORY", "games": 0
    }


def test_goal_probability_is_bounded_and_explicitly_unvalidated():
    logs = tuple({"source": "NHL", "goals": 1 if i < 5 else 0} for i in range(20))
    result = nhl.probability_goal_1plus(logs)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["status"] == "UNVALIDATED_RESEARCH"
    assert result["games"] == 20


def test_roster_parser_uses_explicit_ids_and_names(monkeypatch):
    monkeypatch.setattr(nhl, "_get_json", lambda *a, **k: {
        "forwards": [{"id": 12, "fullName": "Example Forward", "positionCode": "C"},
                     {"id": "bad", "fullName": "Unknown"}],
        "defensemen": [{"id": 13, "fullName": "Example Defense", "positionCode": "D"}],
    })
    assert nhl.roster_for_team("MIN", "20262027") == (
        {"player_id": 12, "name": "Example Forward", "team": "MIN", "position": "C"},
        {"player_id": 13, "name": "Example Defense", "team": "MIN", "position": "D"},
    )
