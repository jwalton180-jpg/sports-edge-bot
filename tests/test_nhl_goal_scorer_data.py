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


def test_roster_parser_uses_legacy_names_and_explicit_ids(monkeypatch):
    monkeypatch.setattr(nhl, "_get_json", lambda *a, **k: {
        "forwards": [{"id": 12, "fullName": "Example Forward", "positionCode": "C"},
                     {"id": "bad", "fullName": "Unknown"}],
        "defensemen": [{"id": 13, "fullName": "Example Defense", "positionCode": "D"}],
    })
    assert nhl.roster_for_team("MIN", "20262027") == (
        {"player_id": 12, "name": "Example Forward", "team": "MIN", "position": "C"},
        {"player_id": 13, "name": "Example Defense", "team": "MIN", "position": "D"},
    )


def test_roster_parser_supports_current_nhl_localized_name_schema(monkeypatch):
    monkeypatch.setattr(nhl, "_get_json", lambda *a, **k: {
        "forwards": [
            {"id": 8484145, "firstName": {"default": "Zach"}, "lastName": {"default": "Benson"},
             "positionCode": "L"},
            {"id": 8482659, "firstName": {"default": "Josh"}, "lastName": {"default": "Doan"},
             "positionCode": "R"},
        ],
        "defensemen": [
            {"id": 8471234, "firstName": {"default": "Example"}, "lastName": {"default": "Defender"},
             "positionCode": "D"},
        ],
    })
    assert nhl.roster_for_team("BUF", "20262027") == (
        {"player_id": 8484145, "name": "Zach Benson", "team": "BUF", "position": "L"},
        {"player_id": 8482659, "name": "Josh Doan", "team": "BUF", "position": "R"},
        {"player_id": 8471234, "name": "Example Defender", "team": "BUF", "position": "D"},
    )


def test_goal_scorer_candidates_emits_rows_when_official_schema_data_is_present(monkeypatch):
    monkeypatch.setattr(nhl, "schedule_for_day", lambda day: ({
        "id": 2026020152,
        "homeTeam": {"abbrev": "BUF"},
        "awayTeam": {"abbrev": "PHI"},
    },))
    monkeypatch.setattr(nhl, "roster_for_team", lambda team, season: (
        {"player_id": 8484145, "name": "Zach Benson", "team": team, "position": "L"},
    ))
    current_logs = tuple(
        {"source": "NHL", "goals": 1 if i < 5 else 0, "date": f"2026-10-{i + 1:02d}", "game_id": str(i)}
        for i in range(20)
    )
    monkeypatch.setattr(nhl, "player_game_log", lambda player_id, season: current_logs)
    rows = nhl.goal_scorer_candidates(
        "2026-10-21", current_season="20262027", prior_season="20252026"
    )
    assert len(rows) == 2
    assert {row["team"] for row in rows} == {"BUF", "PHI"}
    assert all(row["status"] == "UNVALIDATED_RESEARCH" for row in rows)


def test_scorer_candidate_limit_is_two_per_game_not_two_per_team(monkeypatch):
    monkeypatch.setattr(nhl, "schedule_for_day", lambda day: ({
        "id": 2026020152,
        "startTimeUTC": "2026-10-22T23:00:00Z",
        "homeTeam": {"abbrev": "BUF"},
        "awayTeam": {"abbrev": "PHI"},
    },))
    monkeypatch.setattr(nhl, "roster_for_team", lambda team, season: tuple(
        {"player_id": player_id, "name": f"{team} Player {player_id}", "team": team, "position": "L"}
        for player_id in (1, 2, 3)
    ))
    logs = tuple(
        {"source": "NHL", "goals": 1 if i < 5 else 0, "date": f"2026-10-{i + 1:02d}", "game_id": str(i)}
        for i in range(20)
    )
    monkeypatch.setattr(nhl, "player_game_log", lambda player_id, season: logs)
    report = nhl.goal_scorer_report(
        "2026-10-21", current_season="20262027", prior_season="20252026",
        max_per_game=2, minimum_games=10, max_skaters_per_team=24,
    )
    assert report["history_qualified"] == 6
    assert len(report["candidates"]) == 2


def test_goal_scorer_candidates_preserve_verified_event_start(monkeypatch):
    start = "2026-10-22T23:00:00Z"
    monkeypatch.setattr(nhl, "schedule_for_day", lambda day: ({
        "id": 2026020152,
        "startTimeUTC": start,
        "homeTeam": {"abbrev": "BUF"},
        "awayTeam": {"abbrev": "PHI"},
    },))
    monkeypatch.setattr(nhl, "roster_for_team", lambda team, season: (
        {"player_id": 8484145, "name": f"Player {team}", "team": team, "position": "L"},
    ))
    logs = tuple(
        {"source": "NHL", "goals": 1 if i < 5 else 0, "date": f"2026-10-{i + 1:02d}", "game_id": str(i)}
        for i in range(20)
    )
    monkeypatch.setattr(nhl, "player_game_log", lambda player_id, season: logs)
    rows = nhl.goal_scorer_candidates(
        "2026-10-21", current_season="20262027", prior_season="20252026"
    )
    assert len(rows) == 2
    assert all(row["event_start"] == start for row in rows)
    assert all(row["event_title"] == "PHI at BUF" for row in rows)


def test_report_uses_older_season_when_current_season_is_empty(monkeypatch):
    monkeypatch.setattr(nhl, "schedule_for_day", lambda day: ({
        "id": 2026020152, "homeTeam": {"abbrev": "BUF"}, "awayTeam": {"abbrev": "PHI"},
    },))
    monkeypatch.setattr(nhl, "roster_for_team", lambda team, season: (
        {"player_id": 8484145, "name": f"Player {team}", "team": team, "position": "L"},
    ))

    def logs(player_id, season):
        count = {"20262027": 0, "20252026": 5, "20242025": 24}.get(season, 0)
        return tuple({"source": "NHL", "goals": 1 if i < 4 else 0, "date": f"2025-01-{i + 1:02d}",
                      "game_id": str(i)} for i in range(count))

    monkeypatch.setattr(nhl, "player_game_log", logs)
    report = nhl.goal_scorer_report(
        "2026-10-21", current_season="20262027", prior_season="20252026",
        max_per_game=1, minimum_games=10, max_skaters_per_team=24,
    )
    assert report["schedule_games"] == 1
    assert report["teams_seen"] == 2
    assert report["history_qualified"] == 2
    assert len(report["candidates"]) == 2
    assert all(row["season_used"] == "20242025" for row in report["candidates"])
    assert "research candidates" in report["diagnostic"]


def test_report_explains_empty_history_coverage(monkeypatch):
    monkeypatch.setattr(nhl, "schedule_for_day", lambda day: ({
        "id": 2026020152, "homeTeam": {"abbrev": "BUF"}, "awayTeam": {"abbrev": "PHI"},
    },))
    monkeypatch.setattr(nhl, "roster_for_team", lambda team, season: (
        {"player_id": 8484145, "name": f"Player {team}", "team": team, "position": "L"},
    ))
    monkeypatch.setattr(nhl, "player_game_log", lambda player_id, season: ())
    report = nhl.goal_scorer_report(
        "2026-10-21", current_season="20262027", prior_season="20252026",
        max_per_game=1, minimum_games=10, max_skaters_per_team=24,
    )
    assert report["candidates"] == []
    assert report["history_qualified"] == 0
    assert "history floor" in report["diagnostic"]
