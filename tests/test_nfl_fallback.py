from datetime import datetime, timezone

from sports_edge.data.http import HttpResult
from sports_edge.data.nfl import NFLClient


def result(data):
    return HttpResult(
        data=data,
        received_at=datetime(2026, 9, 24, tzinfo=timezone.utc),
        latency_ms=12.0,
        status_code=200,
        headers={},
    )


class FakeHttp:
    def __init__(self):
        self.calls = []

    def get_json(self, url, params=None, headers=None):
        self.calls.append((url, params))
        if "site.api.espn.com" in url:
            raise RuntimeError("403")
        return result(
            {
                "content": {
                    "scoreboard": {
                        "events": [
                            {
                                "id": "401",
                                "competitions": [{"id": "401", "competitors": []}],
                            }
                        ]
                    }
                }
            }
        )


def test_nfl_scoreboard_uses_cdn_fallback_and_normalizes_events():
    http = FakeHttp()
    response = NFLClient(http=http).scoreboard()
    assert response.status_code == 200
    assert response.data["_source"] == "espn_cdn"
    assert response.data["events"][0]["id"] == "401"
    assert len(http.calls) == 2
