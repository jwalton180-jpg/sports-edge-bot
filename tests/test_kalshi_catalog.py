from dataclasses import dataclass

from sports_edge.data.kalshi_catalog import fetch_open_market_catalog


@dataclass
class FakeResponse:
    data: dict
    latency_ms: float = 5.0


class FakeClient:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def markets(self, **kwargs):
        self.calls.append(kwargs)
        cursor = kwargs.get("cursor")
        return FakeResponse(self.pages[cursor])


def test_catalog_exhausts_every_cursor_page_and_dedupes():
    pages = {
        None: {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": "c1"},
        "c1": {"markets": [{"ticker": "B"}, {"ticker": "C"}], "cursor": "c2"},
        "c2": {"markets": [{"ticker": "D"}], "cursor": ""},
    }
    client = FakeClient(pages)
    result = fetch_open_market_catalog(client, page_limit=10, page_size=200)

    assert result.cursor_exhausted is True
    assert result.error is None
    assert result.pages == 3
    assert {m["ticker"] for m in result.markets} == {"A", "B", "C", "D"}
    assert [call.get("cursor") for call in client.calls] == [None, "c1", "c2"]


def test_catalog_repeated_cursor_fails_incomplete_instead_of_claiming_complete():
    pages = {
        None: {"markets": [{"ticker": "A"}], "cursor": "same"},
        "same": {"markets": [{"ticker": "B"}], "cursor": "same"},
    }
    result = fetch_open_market_catalog(FakeClient(pages), page_limit=10)

    assert result.cursor_exhausted is False
    assert "cursor repeated" in (result.error or "").lower()


def test_catalog_safety_limit_is_not_reported_as_complete():
    pages = {
        None: {"markets": [{"ticker": "A"}], "cursor": "c1"},
        "c1": {"markets": [{"ticker": "B"}], "cursor": "c2"},
    }
    result = fetch_open_market_catalog(FakeClient(pages), page_limit=2)

    assert result.cursor_exhausted is False
    assert "safety limit" in (result.error or "").lower()
