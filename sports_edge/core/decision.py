from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionGateResult:
    allowed: bool
    reasons: tuple[str, ...]


def evidence_decision_gate(
    *,
    book_count: int,
    source_age_s: float,
    disagreement_pp: float,
    edge_points: float | None,
    model_ready: bool,
    dependency_valid: bool = True,
    min_books: int = 3,
    max_age_s: float = 180.0,
    max_disagreement_pp: float = 12.0,
    min_edge_points: float = 3.0,
) -> DecisionGateResult:
    reasons: list[str] = []
    if not dependency_valid:
        reasons.append("player/lineup dependency invalidated")
    if book_count < min_books:
        reasons.append(f"only {book_count} fresh book(s); require {min_books}")
    if source_age_s > max_age_s:
        reasons.append("source evidence stale")
    if disagreement_pp > max_disagreement_pp:
        reasons.append("cross-book disagreement above threshold")
    if edge_points is None or edge_points < min_edge_points:
        reasons.append("price edge below threshold")
    if not model_ready:
        reasons.append("sport model overlay unavailable")
    return DecisionGateResult(not reasons, tuple(reasons))
