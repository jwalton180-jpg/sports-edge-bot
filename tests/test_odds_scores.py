from sports_edge.data.odds import OddsClient


class FakeHttp:
    def __init__(self):
        self.calls = []

    def get_json(self, url, params=None, headers=None):
        self.calls.append((url, params, headers))
        return {"ok": True}


def test_scores_uses_bounded_days_from_and_api_key():
    http = FakeHttp()
    client = OddsClient(api_key="test-key", http=http)
    result = client.scores("tennis_atp_test", days_from=9)

    assert result == {"ok": True}
    url, params, _ = http.calls[-1]
    assert url.endswith("/sports/tennis_atp_test/scores")
    assert params == {"apiKey": "test-key", "daysFrom": 3}


def test_scores_requires_api_key():
    client = OddsClient(api_key="", http=FakeHttp())
    try:
        client.scores("tennis_atp_test")
    except RuntimeError as exc:
        assert "THE_ODDS_API_KEY" in str(exc)
    else:
        raise AssertionError("scores should fail closed without API key")
