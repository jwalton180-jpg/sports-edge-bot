from collections import defaultdict, deque
from datetime import date, timedelta

from sports_edge.models.tennis_research import (
    HeadToHeadMeeting,
    PlayerState,
    TennisResearchModel,
)


TODAY = date(2026, 9, 29)


def _base_model() -> TennisResearchModel:
    model = TennisResearchModel.__new__(TennisResearchModel)
    model.gender = "men"
    model.start_year = 2025
    model.end_year = 2026
    model.players = defaultdict(PlayerState)
    model.alias_index = {}
    model.head_to_head = defaultdict(list)

    for name, elo, results, serve, ret in [
        ("Alpha Player", 1600.0, [1, 0, 1, 1, 0, 1, 1, 1], [0.64, 0.65, 0.66, 0.65, 0.67, 0.68], [0.34, 0.35, 0.34, 0.36, 0.37, 0.36]),
        ("Beta Player", 1575.0, [1, 1, 0, 1, 0, 0, 1, 0], [0.64, 0.63, 0.64, 0.62, 0.61, 0.62], [0.34, 0.33, 0.34, 0.32, 0.31, 0.32]),
    ]:
        state = model.players[model._name(name)]
        state.elo = elo
        state.matches = 30
        state.recent_results = deque(results, maxlen=20)
        state.serve_points_won = deque(serve, maxlen=20)
        state.return_points_won = deque(ret, maxlen=20)
        state.recent_dates = deque(maxlen=20)
        state.minutes = deque(maxlen=20)

    model._rebuild_alias_index()
    return model


def _meeting(model, days_ago, winner, loser, surface="hard", level="A"):
    pair = tuple(sorted((model._name(winner), model._name(loser))))
    model.head_to_head[pair].append(
        HeadToHeadMeeting(
            match_date=TODAY - timedelta(days=days_ago),
            winner=model._name(winner),
            loser=model._name(loser),
            surface=surface,
            level=level,
        )
    )


def test_h2h_moves_probability_but_is_shrunk():
    base = _base_model()
    no_h2h = base.probability("Alpha Player", "Beta Player", level="ATP Tour", as_of=TODAY, surface="hard")
    assert no_h2h is not None

    _meeting(base, 60, "Alpha Player", "Beta Player")
    with_h2h = base.probability("Alpha Player", "Beta Player", level="ATP Tour", as_of=TODAY, surface="hard")
    assert with_h2h is not None
    assert with_h2h.fair_probability > no_h2h.fair_probability
    assert with_h2h.fair_probability - no_h2h.fair_probability < 0.04
    assert any("Head-to-head" in f for f in with_h2h.factors)
    assert any("single prior H2H" in w for w in with_h2h.warnings)


def test_multiple_recent_h2h_meetings_have_more_weight_than_one_old_meeting():
    old = _base_model()
    _meeting(old, 900, "Alpha Player", "Beta Player")
    old_ev = old.probability("Alpha Player", "Beta Player", level="ATP Tour", as_of=TODAY, surface="hard")

    recent = _base_model()
    for days in (40, 120, 260):
        _meeting(recent, days, "Alpha Player", "Beta Player")
    recent_ev = recent.probability("Alpha Player", "Beta Player", level="ATP Tour", as_of=TODAY, surface="hard")

    assert old_ev is not None and recent_ev is not None
    assert recent_ev.fair_probability > old_ev.fair_probability


def test_future_h2h_meeting_is_not_used():
    model = _base_model()
    pair = tuple(sorted((model._name("Alpha Player"), model._name("Beta Player"))))
    model.head_to_head[pair].append(
        HeadToHeadMeeting(
            match_date=TODAY + timedelta(days=1),
            winner=model._name("Alpha Player"),
            loser=model._name("Beta Player"),
            surface="hard",
            level="A",
        )
    )
    ev = model.probability("Alpha Player", "Beta Player", level="ATP Tour", as_of=TODAY, surface="hard")
    assert ev is not None
    assert not any("Head-to-head" in f for f in ev.factors)


def test_same_surface_h2h_is_explicitly_reported():
    model = _base_model()
    _meeting(model, 100, "Alpha Player", "Beta Player", surface="clay")
    _meeting(model, 200, "Beta Player", "Alpha Player", surface="hard")
    ev = model.probability("Alpha Player", "Beta Player", level="ATP Tour", as_of=TODAY, surface="clay")
    assert ev is not None
    assert any("same surface" in f for f in ev.factors)


def test_trend_factors_are_exposed_for_matchup_analysis():
    model = _base_model()
    ev = model.probability("Alpha Player", "Beta Player", level="ATP Tour", as_of=TODAY)
    assert ev is not None
    assert any("Form trajectory" in f for f in ev.factors)
    assert any("serve/return trend" in f for f in ev.factors)
