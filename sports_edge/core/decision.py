from __future__ import annotations
from dataclasses import dataclass
from sports_edge.core.health import SourceHealth, Freshness

@dataclass(frozen=True)
class DecisionGateResult:
    allowed: bool
    reasons: tuple[str, ...]


def live_decision_gate(source_health: list[SourceHealth], dependency_valid: bool,
                       min_sources: int = 2, stale_s: float = 20, hard_stale_s: float = 60,
                       max_disagreement: bool = False) -> DecisionGateResult:
    reasons=[]
    if not dependency_valid:
        reasons.append("player/lineup dependency invalidated")
    healthy=[s for s in source_health if s.status(stale_s,hard_stale_s) in {Freshness.FRESH,Freshness.DEGRADED}]
    fresh=[s for s in source_health if s.status(stale_s,hard_stale_s)==Freshness.FRESH]
    if len(healthy)<min_sources:
        reasons.append(f"only {len(healthy)} healthy source(s); require {min_sources}")
    if not fresh:
        reasons.append("no fresh source")
    if max_disagreement:
        reasons.append("cross-source live-state disagreement")
    return DecisionGateResult(not reasons, tuple(reasons))
