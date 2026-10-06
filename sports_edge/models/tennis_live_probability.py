from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from sports_edge.core.math import clamp
from sports_edge.data.tennis_live import TennisLiveScoreState


BASE_HOLD_PROBABILITY = 0.74


@dataclass(frozen=True)
class LiveTennisProbability:
    probability: float
    pregame_probability: float
    best_of: int
    server_known: bool
    net_break_advantage: int | None


def _logit(p: float) -> float:
    p = clamp(float(p), 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _hold_probabilities(skill: float) -> tuple[float, float]:
    base = _logit(BASE_HOLD_PROBABILITY)
    return (
        clamp(_sigmoid(base + skill), 0.05, 0.99),
        clamp(_sigmoid(base - skill), 0.05, 0.99),
    )


def _tiebreak_probability(skill: float) -> float:
    # Tiebreaks mute the normal serve advantage but preserve player-strength
    # direction. Keep the response conservative so a single tiebreak does not
    # overwhelm the pre-match prior.
    return clamp(_sigmoid(0.85 * skill), 0.08, 0.92)


def _match_probability_for_skill(
    skill: float,
    *,
    best_of: int,
    sets_a: int = 0,
    sets_b: int = 0,
    games_a: int = 0,
    games_b: int = 0,
    server_a: bool = True,
) -> float:
    best_of = 5 if int(best_of) == 5 else 3
    sets_to_win = best_of // 2 + 1
    hold_a, hold_b = _hold_probabilities(skill)
    p_tb = _tiebreak_probability(skill)

    @lru_cache(maxsize=None)
    def rec(sa: int, sb: int, ga: int, gb: int, a_serves: bool) -> float:
        if sa >= sets_to_win:
            return 1.0
        if sb >= sets_to_win:
            return 0.0

        # Handle a completed set if the caller supplied a terminal game score
        # or the previous recursive game just completed the set. The server
        # flag already points to the next game server.
        if (ga >= 6 and ga - gb >= 2) or (ga == 7 and gb == 6):
            return rec(sa + 1, sb, 0, 0, a_serves)
        if (gb >= 6 and gb - ga >= 2) or (gb == 7 and ga == 6):
            return rec(sa, sb + 1, 0, 0, a_serves)

        # Modern tour/major formats use a tiebreak at 6-6. ESPN does not expose
        # point-by-point tiebreak score in this feed, so use the conservative
        # strength-derived tiebreak probability and flip first server afterward.
        if ga == 6 and gb == 6:
            next_server = not a_serves
            return (
                p_tb * rec(sa + 1, sb, 0, 0, next_server)
                + (1.0 - p_tb) * rec(sa, sb + 1, 0, 0, next_server)
            )

        p_a_game = hold_a if a_serves else (1.0 - hold_b)
        next_server = not a_serves
        return (
            p_a_game * rec(sa, sb, ga + 1, gb, next_server)
            + (1.0 - p_a_game) * rec(sa, sb, ga, gb + 1, next_server)
        )

    return clamp(
        rec(
            max(0, int(sets_a)),
            max(0, int(sets_b)),
            max(0, int(games_a)),
            max(0, int(games_b)),
            bool(server_a),
        ),
        0.001,
        0.999,
    )


@lru_cache(maxsize=512)
def _skill_from_pregame_probability(target_rounded: float, best_of: int) -> float:
    target = clamp(float(target_rounded), 0.01, 0.99)
    best_of = 5 if int(best_of) == 5 else 3

    def prematch(skill: float) -> float:
        # First server is unknown pre-match. Average both possibilities.
        return 0.5 * (
            _match_probability_for_skill(skill, best_of=best_of, server_a=True)
            + _match_probability_for_skill(skill, best_of=best_of, server_a=False)
        )

    lo, hi = -5.0, 5.0
    for _ in range(44):
        mid = (lo + hi) / 2.0
        if prematch(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def estimate_live_match_probability(
    pregame_probability: float,
    state: TennisLiveScoreState | None,
) -> LiveTennisProbability | None:
    """Condition a pre-match model prior on the current sets/games/server state.

    This is deliberately structural rather than a market-price model: it maps
    the independent Sports Edge pre-match probability into generic service-game
    strength, then recomputes match-win probability from the observed score.
    It therefore supplies an independent live sanity check for reversal signals.
    """
    if state is None:
        return None
    try:
        prior = clamp(float(pregame_probability), 0.01, 0.99)
        best_of = 5 if int(state.best_of) == 5 else 3
        sets_a = int(state.player_sets)
        sets_b = int(state.opponent_sets)
    except (TypeError, ValueError):
        return None

    sets_to_win = best_of // 2 + 1
    if sets_a < 0 or sets_b < 0 or sets_a >= sets_to_win or sets_b >= sets_to_win:
        return None
    if state.player_games is None or state.opponent_games is None:
        return None
    try:
        games_a = int(state.player_games)
        games_b = int(state.opponent_games)
    except (TypeError, ValueError):
        return None
    if min(games_a, games_b) < 0 or max(games_a, games_b) > 7:
        return None

    skill = _skill_from_pregame_probability(round(prior, 4), best_of)
    if isinstance(state.serving, bool):
        p_live = _match_probability_for_skill(
            skill,
            best_of=best_of,
            sets_a=sets_a,
            sets_b=sets_b,
            games_a=games_a,
            games_b=games_b,
            server_a=state.serving,
        )
        server_known = True
    else:
        p_live = 0.5 * (
            _match_probability_for_skill(
                skill, best_of=best_of, sets_a=sets_a, sets_b=sets_b,
                games_a=games_a, games_b=games_b, server_a=True,
            )
            + _match_probability_for_skill(
                skill, best_of=best_of, sets_a=sets_a, sets_b=sets_b,
                games_a=games_a, games_b=games_b, server_a=False,
            )
        )
        server_known = False

    return LiveTennisProbability(
        probability=clamp(p_live, 0.001, 0.999),
        pregame_probability=prior,
        best_of=best_of,
        server_known=server_known,
        net_break_advantage=state.net_break_advantage,
    )
