from datetime import datetime, timezone

from sports_edge.data.tennis_live import parse_espn_live_tennis_states
from sports_edge.models.event_identity import canonical_event_id, canonical_participant


NOW = datetime(2026, 10, 5, 8, 30, tzinfo=timezone.utc)


def _competition(*, state="in", completed=False, period=3):
    return {
        "id": "match-1",
        "date": "2026-10-05T07:00:00Z",
        "status": {
            "period": period,
            "type": {
                "state": state,
                "completed": completed,
            },
        },
        "competitors": [
            {
                "athlete": {"displayName": "Adam Walton"},
                "linescores": [
                    {"value": 7, "winner": True},
                    {"value": 3, "winner": False},
                    {"value": 1},
                ],
            },
            {
                "athlete": {"displayName": "Te Rigele"},
                "linescores": [
                    {"value": 5, "winner": False},
                    {"value": 6, "winner": True},
                    {"value": 3},
                ],
            },
        ],
    }


def _payload(comp):
    return {
        "events": [
            {
                "name": "Rolex Shanghai Masters",
                "status": {
                    "type": {"state": "post", "completed": True},
                },
                "groupings": [
                    {
                        "grouping": {"slug": "mens-singles"},
                        "competitions": [comp],
                    }
                ],
            }
        ]
    }


def test_parser_uses_nested_match_status_not_tournament_status():
    rows = parse_espn_live_tennis_states(
        _payload(_competition()),
        tour="ATP",
        fetched_at=NOW,
    )
    assert len(rows) == 2
    rig = next(row for row in rows if row.player == "Te Rigele")
    assert rig.event_id == canonical_event_id(
        "Tennis",
        "Adam Walton",
        "Rigele TE",
        "2026-10-05",
    )
    assert rig.selection_key == canonical_participant("Tennis", "Rigele TE")
    assert rig.player_sets == 1
    assert rig.opponent_sets == 1
    assert rig.turnaround
    assert rig.deciding_set
    assert rig.current_set_lead == 2
    assert "7-5" in rig.score_label
    assert "3-6" in rig.score_label


def test_parser_excludes_completed_match():
    rows = parse_espn_live_tennis_states(
        _payload(_competition(state="post", completed=True)),
        tour="ATP",
        fetched_at=NOW,
    )
    assert rows == ()


def test_parser_excludes_scheduled_match():
    rows = parse_espn_live_tennis_states(
        _payload(_competition(state="pre", completed=False, period=1)),
        tour="ATP",
        fetched_at=NOW,
    )
    assert rows == ()
