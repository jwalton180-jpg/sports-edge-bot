from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.ticket_policy import (
    construction_rank_key,
)


def selected_games_compat_candidates(
    candidates: Iterable[ParlayCandidateLeg],
    *,
    preferred_event_ids: Iterable[str],
    mode: str,
    target_legs: int,
    max_per_event: int,
    assess: Callable[[ParlayCandidateLeg, str], Any],
) -> list[ParlayCandidateLeg]:
    """Freeze a coverage-safe candidate core for an older parlay builder.

    Streamlit can briefly retain a builder that predates ``preferred_event_ids``.
    Passing that builder a preselected core preserves one qualified leg per
    requested game before extras. Ranking and contract/event caps deliberately
    mirror the normal builder so the compatibility path cannot change policy.
    """
    original = list(candidates)
    preferred = tuple(
        dict.fromkeys(
            str(event_id).strip()
            for event_id in preferred_event_ids
            if str(event_id).strip()
        )
    )
    if not preferred or target_legs <= 0:
        return original

    assessed: list[tuple[ParlayCandidateLeg, Any]] = []
    for candidate in original:
        try:
            row = assess(candidate, mode)
        except Exception:
            continue
        if getattr(row, "qualified", False):
            assessed.append((candidate, row))

    assessed.sort(
        key=lambda pair: construction_rank_key(pair[1]),
        reverse=True,
    )

    selected: list[ParlayCandidateLeg] = []
    selected_ids: set[int] = set()
    seen_contracts: set[tuple[str, str | None]] = set()
    per_event: dict[str, int] = {}

    def try_add(candidate: ParlayCandidateLeg) -> bool:
        event_id = str(getattr(candidate, "event_id", "") or "")
        contract = (
            str(getattr(candidate, "kalshi_ticker", "") or getattr(candidate, "selection", "")),
            getattr(candidate, "kalshi_side", None),
        )
        if id(candidate) in selected_ids or contract in seen_contracts:
            return False
        if per_event.get(event_id, 0) >= max_per_event:
            return False
        selected.append(candidate)
        selected_ids.add(id(candidate))
        seen_contracts.add(contract)
        per_event[event_id] = per_event.get(event_id, 0) + 1
        return True

    best_by_event: dict[str, ParlayCandidateLeg] = {}
    for candidate, _ in assessed:
        event_id = str(getattr(candidate, "event_id", "") or "")
        if event_id in preferred and event_id not in best_by_event:
            best_by_event[event_id] = candidate

    coverage = [best_by_event[event_id] for event_id in preferred if event_id in best_by_event]
    if len(coverage) > target_legs:
        rank = {id(candidate): index for index, (candidate, _) in enumerate(assessed)}
        coverage.sort(key=lambda candidate: rank.get(id(candidate), 10**9))
        coverage = coverage[:target_legs]

    for candidate in coverage:
        if len(selected) >= target_legs:
            break
        try_add(candidate)

    for candidate, _ in assessed:
        if len(selected) >= target_legs:
            break
        try_add(candidate)

    # Keeping the original rejected pool lets the older builder produce its
    # normal fail-closed diagnostics when nothing qualifies.
    return selected or original
