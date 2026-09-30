from datetime import date
from types import SimpleNamespace

from sports_edge.models.kalshi_model_candidates import _market_local_date, _rows_for_local_date


def _row(**market):
    return SimpleNamespace(market=market)


def test_occurrence_datetime_is_converted_to_hawaii_calendar_date():
    # 05:30 UTC on Sep 30 is still Sep 29 in Hawaiʻi.
    market = {"occurrence_datetime": "2026-09-30T05:30:00Z", "ticker": "KXATP-26SEP30-TEST"}
    assert _market_local_date(market, "Pacific/Honolulu") == date(2026, 9, 29)


def test_today_scope_rejects_tennis_matches_on_later_local_dates():
    rows = [
        _row(occurrence_datetime="2026-09-30T05:30:00Z", ticker="KXATP-26SEP30-A"),
        _row(occurrence_datetime="2026-10-01T05:30:00Z", ticker="KXATP-26OCT01-B"),
    ]
    kept = _rows_for_local_date(
        rows,
        target_date=date(2026, 9, 29),
        timezone_name="Pacific/Honolulu",
    )
    assert [row.market["ticker"] for row in kept] == ["KXATP-26SEP30-A"]


def test_today_scope_applies_across_sports_and_uses_ticker_date_fallback():
    rows = [
        _row(ticker="KXMLB-26SEP29-NYYBOS"),
        _row(ticker="KXNFL-26SEP30-SEASF"),
    ]
    kept = _rows_for_local_date(
        rows,
        target_date=date(2026, 9, 29),
        timezone_name="Pacific/Honolulu",
    )
    assert [row.market["ticker"] for row in kept] == ["KXMLB-26SEP29-NYYBOS"]


def test_market_local_date_prefers_ticker_date_over_settlement_time():
    market = {
        "ticker": "KXMLB-26SEP30-NYYBOS",
        "close_time": "2026-10-01T08:00:00Z",
        "expected_expiration_time": "2026-10-01T09:00:00Z",
    }
    assert _market_local_date(market, "Pacific/Honolulu") == date(2026, 9, 30)


def test_future_ticket_scope_keeps_only_requested_tomorrow_date():
    rows = [
        _row(occurrence_datetime="2026-09-30T20:00:00Z", ticker="KXMLB-26SEP30-A"),
        _row(occurrence_datetime="2026-10-01T20:00:00Z", ticker="KXMLB-26OCT01-B"),
        _row(occurrence_datetime="2026-10-02T20:00:00Z", ticker="KXMLB-26OCT02-C"),
    ]
    kept = _rows_for_local_date(
        rows,
        target_date=date(2026, 10, 1),
        timezone_name="Pacific/Honolulu",
    )
    assert [row.market["ticker"] for row in kept] == ["KXMLB-26OCT01-B"]


def test_child_market_title_is_not_a_physical_game_name():
    market = {
        "event_ticker": "KXMLB-26SEP30-SDCHC",
        "event_title": "1st inning: Over 0.5 runs",
        "title": "Will there be over 0.5 runs in the 1st inning?",
    }
    event = {
        "event_ticker": "KXMLB-26SEP30-SDCHC",
        "title": "San Diego Padres at Chicago Cubs",
    }
    # Regression contract: selectors must resolve display identity from the
    # parent event, never from a child inning/prop market title.
    assert event["title"] == "San Diego Padres at Chicago Cubs"
    assert market["event_title"] != event["title"]


def test_mlb_schedule_matchup_is_physical_game_not_child_market():
    game = {
        "gamePk": 999001,
        "teams": {
            "away": {"team": {"name": "Boston Red Sox"}},
            "home": {"team": {"name": "New York Yankees"}},
        },
    }
    away = game["teams"]["away"]["team"]["name"]
    home = game["teams"]["home"]["team"]["name"]
    assert f"{away} at {home}" == "Boston Red Sox at New York Yankees"
    assert "inning" not in f"{away} at {home}".lower()
    assert "hits" not in f"{away} at {home}".lower()
