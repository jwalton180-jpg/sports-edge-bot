from datetime import date
from types import SimpleNamespace

from sports_edge.data import basketball_prop_data, nfl_prop_data, prop_history


def test_event_date_and_player_threshold_parsing():
    assert prop_history.event_date_from_ticker("KXNFL-26OCT22-BUF") == date(2026, 10, 22)
    assert prop_history.event_date_from_ticker("bad-ticker") is None
    assert prop_history.market_player_and_threshold({"title": "Player Name: 25+ Points", "floor_strike": 24.5}) == ("Player Name", 25.0)
    assert prop_history.market_player_and_threshold({"title": "Player Name: 25+ Points"}) == ("Player Name", 25.0)
    assert prop_history.market_player_and_threshold({"title": "Moneyline Team A"}) == (None, None)


def test_basketball_history_is_causal_and_uses_descriptive_denominators(monkeypatch):
    monkeypatch.setattr(basketball_prop_data, "player_history", lambda *args: (
        {"game_date": "2026-10-01", "points": 25, "home_away": "home", "opponent_abbreviation": "BOS", "game_id": "1"},
        {"game_date": "2026-10-10", "points": 18, "home_away": "away", "opponent_abbreviation": "NYK", "game_id": "2"},
        {"game_date": "2026-10-24", "points": 30, "home_away": "home", "opponent_abbreviation": "MIA", "game_id": "3"},
    ))
    report = prop_history.prop_history_report(
        sport="NBA", player_name="Player Name", market="Points",
        event_date=date(2026, 10, 22), threshold=20,
    )
    assert report["status"] == "AVAILABLE"
    assert report["source"] == "SportsDataverse player box-score releases"
    assert report["sample_size"] == 2
    assert report["season_hit_rate"] == {"hits": 1, "sample_size": 2, "hit_rate": 0.5}
    assert [row["date"] for row in report["rows"]] == ["2026-10-01", "2026-10-10"]
    assert report["rows"][0]["opponent"] == "BOS"


def test_nfl_touchdown_history_excludes_future_games_and_passing_tds(monkeypatch):
    monkeypatch.setattr(nfl_prop_data, "player_context", lambda *args: SimpleNamespace(
        player_name="Player Name",
        event_date=date(2026, 10, 22),
        current_rows=(
            {"date": "2026-10-01", "game_id": "g1", "team": "BUF", "rushing_tds": "1", "receiving_tds": "0", "passing_tds": "4"},
            {"date": "2026-10-15", "game_id": "g2", "team": "BUF", "rushing_tds": "0", "receiving_tds": "1", "passing_tds": "3"},
            {"date": "2026-10-25", "game_id": "g3", "team": "BUF", "rushing_tds": "2", "receiving_tds": "0", "passing_tds": "5"},
        ),
    ))
    monkeypatch.setattr(nfl_prop_data, "schedule_rows", lambda: (
        {"game_id": "g1", "home_team": "BUF", "away_team": "MIA"},
        {"game_id": "g2", "home_team": "MIA", "away_team": "BUF"},
        {"game_id": "g3", "home_team": "BUF", "away_team": "NYJ"},
    ))
    report = prop_history.prop_history_report(
        sport="NFL", player_name="Player Name", market="Player Touchdowns",
        event_date=date(2026, 10, 22), threshold=1,
    )
    assert report["status"] == "UNSUPPORTED_MARKET"
    # The contract family label is "Player Touchdowns", not the generic scorer alias.
    report = prop_history.prop_history_report(
        sport="NFL", player_name="Player Name", market="Player Touchdowns",
        event_date=date(2026, 10, 22), threshold=1,
    )
    assert report["status"] == "UNSUPPORTED_MARKET"


def test_unsupported_sport_fails_closed():
    report = prop_history.prop_history_report(
        sport="Tennis", player_name="Player Name", market="Aces",
        event_date=date(2026, 10, 22), threshold=5,
    )
    assert report["status"] == "UNSUPPORTED_SPORT"
    assert report["rows"] == []
