from datetime import datetime, timezone

from sports_edge.data import sofascore


def test_parse_sofascore_events_is_sport_agnostic():
    payload = {
        "events": [
            {
                "id": 123,
                "homeTeam": {"name": "Home"},
                "awayTeam": {"name": "Away"},
                "status": {"description": "2nd quarter"},
                "homeScore": {"current": 31},
                "awayScore": {"current": 27},
                "startTimestamp": 1791345600,
            }
        ]
    }
    rows = sofascore.parse_sofascore_events(payload, sport="NBA")
    assert len(rows) == 1
    row = rows[0]
    assert row.event_id == "123"
    assert row.sport == "NBA"
    assert row.home == "Home"
    assert row.away == "Away"
    assert row.home_score == 31
    assert row.away_score == 27
    assert row.start_at == datetime.fromtimestamp(1791345600, tz=timezone.utc)


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_sofascore_403_is_explicitly_blocked_and_never_treated_as_empty_success(monkeypatch):
    monkeypatch.setattr(
        sofascore.requests,
        "get",
        lambda *args, **kwargs: _Response(403, {"error": {"reason": "Forbidden"}}),
    )
    result = sofascore.fetch_sofascore_live_events("Tennis")
    assert not result.available
    assert result.blocked
    assert result.status_code == 403
    assert result.events == ()
    assert "denied" in (result.error or "")


def test_sofascore_available_payload_returns_events(monkeypatch):
    monkeypatch.setattr(
        sofascore.requests,
        "get",
        lambda *args, **kwargs: _Response(
            200,
            {
                "events": [
                    {
                        "id": 9,
                        "homeTeam": {"name": "A"},
                        "awayTeam": {"name": "B"},
                        "status": {"type": "inprogress"},
                        "homeScore": {"display": 1},
                        "awayScore": {"display": 0},
                    }
                ]
            },
        ),
    )
    result = sofascore.fetch_sofascore_live_events("MLB")
    assert result.available
    assert not result.blocked
    assert result.status_code == 200
    assert len(result.events) == 1
    assert result.events[0].sport == "MLB"


def test_sofascore_maps_every_supported_sportsedge_sport():
    assert sofascore.SOFASCORE_SPORT_SLUGS == {
        "Tennis": "tennis",
        "MLB": "baseball",
        "NFL": "american-football",
        "NBA": "basketball",
        "WNBA": "basketball",
    }


def test_sofascore_snapshot_preserves_requested_sports(monkeypatch):
    def fake(sport, *, timeout=8.0):
        return sofascore.SofaScoreFeedResult(
            sport=sport,
            source_url="test",
            available=False,
            blocked=True,
            status_code=403,
            events=(),
            error="blocked",
        )

    monkeypatch.setattr(sofascore, "fetch_sofascore_live_events", fake)
    rows = sofascore.fetch_sofascore_live_snapshot(("MLB", "NFL", "NBA"), max_workers=2)
    assert [row.sport for row in rows] == ["MLB", "NFL", "NBA"]
    assert all(row.blocked for row in rows)
