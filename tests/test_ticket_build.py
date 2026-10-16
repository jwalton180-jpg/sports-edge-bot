from sports_edge.models.ticket_build import build_ticket_for_scope, ticket_state_matches


def capturing_builder(
    candidates,
    *,
    mode,
    target_legs,
    max_per_event,
    diversify_sports,
    prioritize_payout_multiplier=False,
    preferred_event_ids=None,
):
    return {
        "candidates": list(candidates),
        "mode": mode,
        "target_legs": target_legs,
        "max_per_event": max_per_event,
        "diversify_sports": diversify_sports,
        "prioritize_payout_multiplier": prioritize_payout_multiplier,
        "preferred_event_ids": preferred_event_ids,
    }


def assessor(candidate, mode):
    raise AssertionError("compatibility assessor should not run for the current builder")


def build(*, scope, sport="MLB", target=6, preferred=()):
    return build_ticket_for_scope(
        ["candidate"],
        builder=capturing_builder,
        assessor=assessor,
        mode="best",
        target_legs=target,
        sport_filter=sport,
        game_scope=scope,
        preferred_event_ids=preferred,
    )


def test_all_games_uses_multi_game_policy_without_selected_coverage():
    result = build(scope="All games", preferred=("E1", "E2"))
    assert result["target_legs"] == 6
    assert result["max_per_event"] == 3
    assert result["prioritize_payout_multiplier"] is False
    assert result["preferred_event_ids"] is None


def test_selected_games_preserves_coverage_on_every_build_pass():
    result = build(scope="Selected games", preferred=("E1", "E2", "E3", "E4"))
    assert result["target_legs"] == 6
    assert result["max_per_event"] == 3
    assert result["preferred_event_ids"] == ("E1", "E2", "E3", "E4")


def test_single_game_preserves_four_leg_cap_and_payout_priority():
    result = build(scope="Single game", target=8, preferred=("E1",))
    assert result["target_legs"] == 4
    assert result["max_per_event"] == 4
    assert result["diversify_sports"] is False
    assert result["prioritize_payout_multiplier"] is True
    assert result["preferred_event_ids"] is None


def test_all_sports_multi_game_keeps_cross_sport_diversification():
    result = build(scope="All games", sport="All")
    assert result["max_per_event"] == 1
    assert result["diversify_sports"] is True


def base_state():
    return {
        "mode": "best",
        "preset": "Best Available",
        "sport": "MLB",
        "target": 5,
        "ticket_date": "2026-10-03",
        "game_scope": "Selected games",
        "selected_games": ("A at B", "C at D"),
    }


def matches(state, **overrides):
    controls = {
        "mode": "best",
        "preset": "Best Available",
        "sport": "MLB",
        "target": 5,
        "ticket_date": "2026-10-03",
        "game_scope": "Selected games",
        "selected_games": ("A at B", "C at D"),
    }
    controls.update(overrides)
    return ticket_state_matches(state, **controls)


def test_ticket_state_matches_the_exact_current_controls():
    assert matches(base_state()) is True


def test_ticket_state_rejects_stale_date_scope_and_selected_slate():
    assert matches(base_state(), ticket_date="2026-10-04") is False
    assert matches(base_state(), game_scope="All games") is False
    assert matches(base_state(), selected_games=("A at B",)) is False


def test_ticket_state_rejects_stale_mode_preset_sport_or_target():
    assert matches(base_state(), mode="longshot") is False
    assert matches(base_state(), preset="MLB Hits") is False
    assert matches(base_state(), sport="All") is False
    assert matches(base_state(), target=6) is False


def test_ticket_date_blocks_missing_or_mismatched_verified_start():
    from types import SimpleNamespace

    same_day = SimpleNamespace(verified_start="2026-10-15T19:00:00-04:00", event_id="same")
    wrong_day = SimpleNamespace(verified_start="2026-10-16T19:00:00-04:00", event_id="wrong")
    missing = SimpleNamespace(verified_start=None, event_id="missing")
    naive = SimpleNamespace(verified_start="2026-10-15T19:00:00", event_id="naive")
    malformed = SimpleNamespace(verified_start="not-a-time", event_id="bad")

    result = build_ticket_for_scope(
        [same_day, wrong_day, missing, naive, malformed],
        builder=capturing_builder,
        assessor=assessor,
        mode="best",
        target_legs=5,
        sport_filter="All",
        game_scope="All games",
        ticket_date="2026-10-15",
        ticket_timezone="Pacific/Honolulu",
    )
    assert [row.event_id for row in result["candidates"]] == ["same"]
