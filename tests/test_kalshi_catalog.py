from dataclasses import dataclass

from sports_edge.data.kalshi_catalog import fetch_current_sport_catalog, fetch_supported_sport_catalog


@dataclass
class FakeResponse:
    data: dict
    latency_ms: float = 5.0


class FakeClient:
    def __init__(self, series, pages):
        self._series = series
        self.pages = pages
        self.calls = []

    def series_list(self, **kwargs):
        return FakeResponse({"series": self._series})

    def markets(self, **kwargs):
        self.calls.append(kwargs)
        key = (kwargs.get("series_ticker"), kwargs.get("cursor"))
        return FakeResponse(self.pages[key])


def test_catalog_discovers_relevant_series_and_exhausts_each_cursor():
    series = [
        {"ticker": "KXMLBGAME", "title": "MLB games"},
        {"ticker": "KXATPCHALLENGERMATCH", "title": "ATP Challenger"},
        {"ticker": "KXWEATHER", "title": "Weather"},
    ]
    pages = {
        ("KXMLBGAME", None): {"markets": [{"ticker": "MLB-A"}], "cursor": "m2"},
        ("KXMLBGAME", "m2"): {"markets": [{"ticker": "MLB-B"}], "cursor": ""},
        ("KXATPCHALLENGERMATCH", None): {"markets": [{"ticker": "TEN-A"}], "cursor": ""},
    }
    result = fetch_supported_sport_catalog(FakeClient(series, pages), request_pause_s=0)

    assert result.cursor_exhausted is True
    assert result.error is None
    assert result.discovered_series == 3
    assert result.relevant_series == 2
    assert result.pages == 3
    assert {m["ticker"] for m in result.markets} == {"MLB-A", "MLB-B", "TEN-A"}
    by_ticker = {m["ticker"]: m for m in result.markets}
    assert by_ticker["MLB-A"]["sports_edge_sport"] == "MLB"
    assert by_ticker["TEN-A"]["sports_edge_sport"] == "Tennis"


def test_catalog_new_metadata_series_can_be_discovered_without_known_prefix():
    series = [
        {
            "ticker": "KXNEWPROTENNIS",
            "title": "ATP Challenger tennis matches",
            "category": "Sports",
            "tags": ["ATP", "Tennis"],
        }
    ]
    pages = {
        ("KXNEWPROTENNIS", None): {
            "markets": [{"ticker": "KXNEWPROTENNIS-TEST"}],
            "cursor": "",
        }
    }
    result = fetch_supported_sport_catalog(FakeClient(series, pages), request_pause_s=0)
    assert result.relevant_series == 1
    assert result.markets[0]["sports_edge_sport"] == "Tennis"


def test_catalog_repeated_cursor_fails_incomplete_instead_of_claiming_complete():
    series = [{"ticker": "KXNBAGAME", "title": "NBA games"}]
    pages = {
        ("KXNBAGAME", None): {"markets": [{"ticker": "A"}], "cursor": "same"},
        ("KXNBAGAME", "same"): {"markets": [{"ticker": "B"}], "cursor": "same"},
    }
    result = fetch_supported_sport_catalog(FakeClient(series, pages), request_pause_s=0)

    assert result.cursor_exhausted is False
    assert "KXNBAGAME" in result.incomplete_series
    assert "repeated pagination cursor" in (result.error or "").lower()


def test_catalog_series_page_limit_is_not_reported_as_complete():
    series = [{"ticker": "KXWNBAGAME", "title": "WNBA games"}]
    pages = {
        ("KXWNBAGAME", None): {"markets": [{"ticker": "A"}], "cursor": "c1"},
    }
    result = fetch_supported_sport_catalog(
        FakeClient(series, pages),
        page_limit_per_series=1,
        request_pause_s=0,
    )

    assert result.cursor_exhausted is False
    assert "KXWNBAGAME" in result.incomplete_series



class WindowFakeClient:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def markets(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.pages[kwargs.get("cursor")])


def test_current_catalog_uses_close_window_and_keeps_only_open_supported_sports():
    pages = {
        None: {
            "markets": [
                {"ticker": "KXATPMATCH-TEST-A", "status": "open", "title": "A vs B"},
                {"ticker": "KXNBAPTS-TEST-B", "status": "open", "title": "Player points"},
                {"ticker": "KXPOLITICS-TEST-C", "status": "open", "title": "Politics"},
                {"ticker": "KXMLBGAME-OLD", "status": "settled", "title": "Old game"},
            ],
            "cursor": "next",
        },
        "next": {
            "markets": [
                {"ticker": "KXNFLGAME-TEST-D", "status": "open", "title": "Team D vs Team E"},
                {"ticker": "KXWNBAPTS-TEST-E", "status": "open", "title": "Player points"},
            ],
            "cursor": "",
        },
    }
    client = WindowFakeClient(pages)
    result = fetch_current_sport_catalog(
        client,
        now_ts=1_800_000_000,
        past_hours=12,
        future_hours=168,
        page_size=1000,
    )
    assert result.cursor_exhausted is True
    assert result.pages == 2
    assert {m["ticker"] for m in result.markets} == {
        "KXATPMATCH-TEST-A",
        "KXNBAPTS-TEST-B",
        "KXNFLGAME-TEST-D",
        "KXWNBAPTS-TEST-E",
    }
    first = client.calls[0]
    assert first["status"] is None
    assert first["mve_filter"] == "exclude"
    assert first["min_close_ts"] < 1_800_000_000 < first["max_close_ts"]
    assert first["limit"] == 1000


def test_current_catalog_repeated_cursor_reports_incomplete():
    pages = {
        None: {"markets": [{"ticker": "KXATPMATCH-A", "status": "open"}], "cursor": "same"},
        "same": {"markets": [{"ticker": "KXATPMATCH-B", "status": "open"}], "cursor": "same"},
    }
    result = fetch_current_sport_catalog(WindowFakeClient(pages), now_ts=1_800_000_000)
    assert result.cursor_exhausted is False
    assert "cursor repeated" in (result.error or "").lower()
