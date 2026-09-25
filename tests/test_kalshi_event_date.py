from datetime import date

from sports_edge.models.kalshi_model_candidates import _parse_date


def test_event_ticker_date_beats_late_settlement_timestamp():
    market = {
        "event_ticker": "KXNFLGAME-26SEP27KCMIA",
        "ticker": "KXNFLGAME-26SEP27KCMIA-KC",
        "close_time": "2026-09-29T03:00:00Z",
    }
    assert _parse_date(market) == date(2026, 9, 27)


def test_ticker_date_handles_time_suffix_after_date():
    market = {
        "event_ticker": "KXMLBGAME-26SEP251305CHCBOSG1",
        "close_time": "2026-09-28T00:00:00Z",
    }
    assert _parse_date(market) == date(2026, 9, 25)


def test_ticker_date_handles_october():
    market = {
        "event_ticker": "KXNBAGAME-26OCT20PHINYK",
        "close_time": "2026-10-22T00:00:00Z",
    }
    assert _parse_date(market) == date(2026, 10, 20)


def test_timestamp_fallback_remains_for_unstructured_tickers():
    market = {
        "event_ticker": "UNKNOWN",
        "close_time": "2026-09-30T12:00:00Z",
    }
    assert _parse_date(market) == date(2026, 9, 30)
