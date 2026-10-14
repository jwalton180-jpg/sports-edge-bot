"""Conservative score-backed eligibility for outward Tennis reversal PUSH alerts.

This deliberately runs AFTER price discovery. Unverified market-only watches
remain research observations, never actionable push notifications. A player
losing in straight sets or facing an imminent match defeat is not promoted
merely because their quoted price rebounded.
"""
from __future__ import annotations

from datetime import datetime, timezone
from collections import defaultdict

from sports_edge.models.event_identity import canonical_participant
from sports_edge.data.tennis_live import TennisLiveScoreState


def survivability_reason(state: TennisLiveScoreState | None, *, now: datetime) -> str:
    """Return 'eligible' only with a fresh, unambiguous, structurally viable score.

    The rule is intentionally stricter than the research radar and may reject
    legitimate turnarounds. This is alert-quality gating, not a win guarantee.
    """
    if state is None:
        return "unverified_live_score"
    if state.score_conflict or not state.score_sources:
        return "conflicted_or_unverified_score"
    if state.fetched_at.tzinfo is None:
        return "unverifiable_score_time"
    age = (now.astimezone(timezone.utc) - state.fetched_at.astimezone(timezone.utc)).total_seconds()
    if age < 0 or age > 120:
        return "stale_live_score"
    if state.player_games is None or state.opponent_games is None:
        return "missing_game_score"
    if not (0 <= state.player_games <= 7 and 0 <= state.opponent_games <= 7):
        return "invalid_game_score"
    if state.best_of not in (3, 5) or state.sets_to_win != (state.best_of // 2 + 1):
        return "untrusted_match_format"
    if (min(state.player_sets, state.opponent_sets) < 0
            or max(state.player_sets, state.opponent_sets) >= state.sets_to_win):
        return "completed_or_invalid_sets"
    if state.player_sets + state.opponent_sets >= state.best_of:
        return "impossible_score"
    if state.at_tiebreak:
        return "tiebreak_high_variance"
    if state.opponent_games >= 5 and state.player_games < state.opponent_games:
        return "near_game_or_match_loss"
    if state.opponent_sets == state.sets_to_win - 1:
        # One lost set from elimination: only a concrete scoreboard comeback
        # can justify waking someone for a reversal. Price action alone fails.
        if state.player_games - state.opponent_games < 2:
            return "elimination_set_without_control"
        if state.net_break_advantage is None or state.net_break_advantage < 1:
            return "elimination_set_break_unverified"
    elif state.player_sets == state.opponent_sets:
        # Deciding set / early match: not enough to infer a turnaround from
        # a 0-0 or 1-1 score and a market price alone.
        if state.player_games - state.opponent_games < 2:
            return "no_deciding_set_control"
    elif state.player_sets > state.opponent_sets:
        # Even when up a set, do not alert on someone getting crushed in the
        # current one merely because they remain cheap.
        if state.player_games - state.opponent_games < -1:
            return "current_set_unfavorable"
    else:
        return "unverified_match_progress"
    return "eligible"


def qualified_tickers(
    markets: list[dict] | tuple[dict, ...],
    states: list[TennisLiveScoreState] | tuple[TennisLiveScoreState, ...],
    *,
    now: datetime,
) -> tuple[dict[str, TennisLiveScoreState], dict[str, int]]:
    """Exact two-player event pairing; never infer match state from odds.

    Kalshi has one YES contract per player under a common event_ticker. Match
    both players' full canonical names against the score feed. If multiple
    score matches exist, decline the alert instead of picking one.
    """
    players_by_event: dict[str, set[str]] = defaultdict(set)
    for market in markets:
        if market.get("event_ticker") and market.get("yes_sub_title"):
            players_by_event[str(market["event_ticker"])].add(
                canonical_participant("Tennis", str(market["yes_sub_title"]))
            )
    state_pairs: dict[frozenset[str], list[TennisLiveScoreState]] = defaultdict(list)
    for state in states:
        key=frozenset({
            canonical_participant("Tennis",state.player),
            canonical_participant("Tennis",state.opponent)
        })
        if len(key)==2:
            state_pairs[key].append(state)

    allowed: dict[str, TennisLiveScoreState] = {}
    rejected: dict[str, int] = defaultdict(int)
    for market in markets:
        ticker=str(market.get("ticker") or "")
        event=str(market.get("event_ticker") or "")
        selection=canonical_participant("Tennis",str(market.get("yes_sub_title") or ""))
        participants=players_by_event.get(event,set())
        if not ticker or len(participants)!=2 or selection not in participants:
            rejected["market_pair_unverified"]+=1
            continue
        matching=[
            s for s in state_pairs.get(frozenset(participants),())
            if canonical_participant("Tennis",s.player)==selection
        ]
        # More than one score row for an identical named matchup = ambiguous,
        # potentially different day/tournament. Never guess.
        if len(matching)!=1:
            rejected["unique_live_score_not_found"]+=1
            continue
        reason=survivability_reason(matching[0],now=now)
        if reason=="eligible":
            allowed[ticker]=matching[0]
        else:
            rejected[reason]+=1
    return allowed,dict(rejected)
