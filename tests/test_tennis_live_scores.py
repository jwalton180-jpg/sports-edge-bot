from dataclasses import replace
from datetime import datetime, timezone

from sports_edge.data.tennis_live import _merge_live_score_state, parse_espn_live_tennis_states
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
                "possession": False,
                "linescores": [
                    {"value": 7, "winner": True},
                    {"value": 3, "winner": False},
                    {"value": 1},
                ],
            },
            {
                "athlete": {"displayName": "Te Rigele"},
                "possession": True,
                "linescores": [
                    {"value": 5, "winner": False},
                    {"value": 6, "winner": True},
                    {"value": 3},
                ],
            },
        ],
    }


def _payload(comp, *, major=False):
    return {
        "events": [
            {
                "name": "Rolex Shanghai Masters",
                "major": major,
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
    assert rig.best_of == 3
    assert rig.sets_to_win == 2
    assert rig.serving is True
    assert rig.net_break_advantage == 1
    assert rig.current_set_lead == 2
    assert "7-5" in rig.score_label
    assert "3-6" in rig.score_label

def test_parser_does_not_call_set_three_deciding_in_mens_major_best_of_five():
    rows = parse_espn_live_tennis_states(
        _payload(_competition(period=3), major=True),
        tour="ATP",
        fetched_at=NOW,
    )
    rig = next(row for row in rows if row.player == "Te Rigele")
    assert rig.best_of == 5
    assert rig.sets_to_win == 3
    assert rig.player_sets == 1
    assert rig.opponent_sets == 1
    assert not rig.deciding_set


def test_parser_recognizes_true_fifth_set_decider_in_mens_major():
    comp = _competition(period=5)
    comp["competitors"][0]["linescores"] = [
        {"value": 7, "winner": True},
        {"value": 3, "winner": False},
        {"value": 6, "winner": True},
        {"value": 4, "winner": False},
        {"value": 1},
    ]
    comp["competitors"][1]["linescores"] = [
        {"value": 5, "winner": False},
        {"value": 6, "winner": True},
        {"value": 2, "winner": False},
        {"value": 6, "winner": True},
        {"value": 3},
    ]
    rows = parse_espn_live_tennis_states(
        _payload(comp, major=True),
        tour="ATP",
        fetched_at=NOW,
    )
    rig = next(row for row in rows if row.player == "Te Rigele")
    assert rig.best_of == 5
    assert rig.player_sets == 2
    assert rig.opponent_sets == 2
    assert rig.deciding_set
    assert rig.current_set_lead == 2

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


def test_cross_feed_merge_keeps_espn_score_but_adds_tennis365_context_url():
    espn = next(
        row for row in parse_espn_live_tennis_states(
            _payload(_competition()),
            tour="ATP",
            fetched_at=NOW,
        )
        if row.player == "Te Rigele"
    )
    tennis365 = replace(
        espn,
        score_sources=("Tennis365",),
        source_url="https://livescore.tennis365.com/match/te-rigele-adam-walton",
        serving=None,
        net_break_advantage=None,
    )
    merged = _merge_live_score_state(espn, tennis365)
    assert merged.score_sources == ("ESPN", "Tennis365")
    assert merged.source_url == tennis365.source_url
    assert merged.serving is True
    assert merged.net_break_advantage == 1
    assert not merged.score_conflict
