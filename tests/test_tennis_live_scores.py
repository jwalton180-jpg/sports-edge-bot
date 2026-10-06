from dataclasses import replace
from datetime import datetime, timezone

from sports_edge.data.tennis_live import (
    _merge_live_score_state,
    parse_espn_live_tennis_states,
    parse_sofascore_live_tennis_states,
)
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

def _sofa_quito_payload(*, first_to_serve=None):
    event = {
        "status": {"code": 10, "description": "3rd set", "type": "inprogress"},
        "tournament": {
            "name": "ITF M15 Quito 3 Men",
            "category": {"name": "ITF Men", "slug": "itf-men", "flag": "itf-men"},
        },
        "eventFilters": {"category": ["singles"], "level": ["pro"], "tournament": ["lower"], "gender": ["M"]},
        "homeTeam": {"name": "Felipe de Dios", "gender": "M", "type": 1},
        "awayTeam": {"name": "Darwin Andres Macias Elizalde", "gender": "M", "type": 1},
        "homeScore": {
            "current": 1, "display": 1, "period1": 7, "period2": 5, "period3": 1, "point": "0"
        },
        "awayScore": {
            "current": 1, "display": 1, "period1": 6, "period2": 7, "period3": 3, "point": "0"
        },
        "lastPeriod": "period3",
        "startTimestamp": 1791308632,
        "id": 17264658,
    }
    if first_to_serve is not None:
        event["firstToServe"] = first_to_serve
    return {"events": [event]}


def test_sofascore_parser_covers_exact_quito_itf_reversal_state():
    rows = parse_sofascore_live_tennis_states(
        _sofa_quito_payload(),
        fetched_at=NOW,
    )
    assert len(rows) == 2
    mac = next(row for row in rows if row.player == "Darwin Andres Macias Elizalde")
    assert mac.event_id == canonical_event_id(
        "Tennis",
        "Felipe de Dios",
        "Darwin Andres Macias Elizalde",
        "2026-10-06",
    )
    assert mac.tour == "ITF"
    assert mac.player_sets == 1
    assert mac.opponent_sets == 1
    assert mac.lost_first_set
    assert mac.won_latest_completed_set
    assert mac.turnaround
    assert mac.deciding_set
    assert mac.current_set_lead == 2
    assert mac.player_games == 3
    assert mac.opponent_games == 1
    assert mac.serving is None
    assert mac.net_break_advantage is None
    assert mac.point_score == "0-0"
    assert mac.score_sources == ("SofaScore",)
    assert not mac.score_conflict


def test_sofascore_first_server_plus_game_parity_infers_current_server_and_break():
    rows = parse_sofascore_live_tennis_states(
        _sofa_quito_payload(first_to_serve=1),
        fetched_at=NOW,
    )
    mac = next(row for row in rows if row.player == "Darwin Andres Macias Elizalde")
    assert mac.serving is True
    assert mac.net_break_advantage == 1


def test_cross_feed_agreement_merges_sources_and_missing_server_context():
    sofa = next(
        row
        for row in parse_sofascore_live_tennis_states(
            _sofa_quito_payload(first_to_serve=1), fetched_at=NOW
        )
        if row.player == "Darwin Andres Macias Elizalde"
    )
    espn_like = replace(
        sofa,
        serving=None,
        net_break_advantage=None,
        point_score=None,
        score_sources=("ESPN",),
    )
    merged = _merge_live_score_state(espn_like, sofa)
    assert merged.score_sources == ("ESPN", "SofaScore")
    assert merged.serving is True
    assert merged.net_break_advantage == 1
    assert merged.point_score == "0-0"
    assert not merged.score_conflict


def test_cross_feed_structural_disagreement_is_flagged_fail_closed():
    sofa = next(
        row
        for row in parse_sofascore_live_tennis_states(
            _sofa_quito_payload(first_to_serve=1), fetched_at=NOW
        )
        if row.player == "Darwin Andres Macias Elizalde"
    )
    other = replace(
        sofa,
        player_games=2,
        current_set_lead=1,
        score_sources=("ESPN",),
    )
    merged = _merge_live_score_state(other, sofa)
    assert merged.score_conflict
    assert merged.score_sources == ("ESPN", "SofaScore")


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
