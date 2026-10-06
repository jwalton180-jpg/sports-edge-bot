from datetime import datetime, timezone

from sports_edge.data.tennis_live import TennisLiveScoreState
from sports_edge.models.tennis_live_probability import estimate_live_match_probability


NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def _state(
    *,
    best_of=3,
    sets_a=0,
    sets_b=0,
    games_a=0,
    games_b=0,
    serving=None,
    net_break=0,
):
    sets_to_win = best_of // 2 + 1
    period = sets_a + sets_b + 1
    return TennisLiveScoreState(
        event_id="TENNIS:2026-10-05:alpha|beta",
        selection_key="alpha",
        player="Alpha",
        opponent="Beta",
        tour="ATP",
        period=period,
        player_sets=sets_a,
        opponent_sets=sets_b,
        player_games=games_a,
        opponent_games=games_b,
        lost_first_set=sets_b > 0,
        won_latest_completed_set=sets_a > 0,
        turnaround=sets_a > 0 and sets_b > 0,
        deciding_set=(sets_a == sets_b == sets_to_win - 1),
        current_set_lead=games_a - games_b,
        score_label="Alpha vs Beta",
        fetched_at=NOW,
        best_of=best_of,
        sets_to_win=sets_to_win,
        serving=serving,
        net_break_advantage=net_break,
    )


def test_pregame_state_round_trips_model_probability():
    estimate = estimate_live_match_probability(0.31, _state())
    assert estimate is not None
    assert abs(estimate.probability - 0.31) < 0.003


def test_best_of_three_one_set_all_materially_updates_cheap_underdog():
    estimate = estimate_live_match_probability(
        0.10,
        _state(best_of=3, sets_a=1, sets_b=1),
    )
    assert estimate is not None
    assert 0.18 < estimate.probability < 0.23


def test_best_of_five_one_set_all_is_not_mistaken_for_decider():
    estimate = estimate_live_match_probability(
        0.10,
        _state(best_of=5, sets_a=1, sets_b=1),
    )
    assert estimate is not None
    assert 0.13 < estimate.probability < 0.18


def test_true_fifth_set_two_all_updates_more_than_one_all():
    one_all = estimate_live_match_probability(
        0.10, _state(best_of=5, sets_a=1, sets_b=1)
    )
    two_all = estimate_live_match_probability(
        0.10, _state(best_of=5, sets_a=2, sets_b=2)
    )
    assert one_all is not None and two_all is not None
    assert two_all.probability > one_all.probability
    assert two_all.probability > 0.22


def test_current_game_state_moves_probability_in_correct_direction():
    ahead = estimate_live_match_probability(
        0.30,
        _state(games_a=4, games_b=2, serving=True, net_break=1),
    )
    behind = estimate_live_match_probability(
        0.30,
        _state(games_a=2, games_b=4, serving=False, net_break=-1),
    )
    assert ahead is not None and behind is not None
    assert ahead.probability > 0.30
    assert behind.probability < 0.30
    assert ahead.probability > behind.probability
