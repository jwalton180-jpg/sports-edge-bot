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
