from __future__ import annotations

from dataclasses import dataclass
from math import prod

from sports_edge.models.live_board import LiveSignal


@dataclass(frozen=True)
class ParlayResearch:
    mode: str
    legs: tuple[LiveSignal, ...]
    independent_fair_probability: float
    independent_market_probability: float
    value_multiple: float
    correlation_risk: str
    warnings: tuple[str, ...]


def _unique_event_signals(signals: list[LiveSignal]) -> list[LiveSignal]:
    qualified = [s for s in signals if s.status == "QUALIFIED"]
    qualified.sort(key=lambda s: (s.edge_points * s.confidence, s.data_quality), reverse=True)
    seen: set[str] = set()
    out: list[LiveSignal] = []
    for signal in qualified:
        key = signal.event_id or signal.ticker
        if key in seen:
            continue
        seen.add(key)
        out.append(signal)
    return out


def build_parlay_research(signals: list[LiveSignal], mode: str = "high_confidence") -> ParlayResearch:
    pool = _unique_event_signals(signals)
    warnings: list[str] = []

    if mode == "longshot":
        wanted = 6
        minimum = 5
        label = "Longshot Lab"
    else:
        wanted = 4
        minimum = 2
        label = "High-Confidence Builder"

    legs = tuple(pool[:wanted])
    if len(legs) < minimum:
        warnings.append(f"Need at least {minimum} independently qualified event legs; found {len(legs)}")

    fair = prod(s.fair_probability for s in legs) if legs else 0.0
    market = prod(s.market_probability for s in legs) if legs else 0.0
    multiple = fair / market if market > 0 else 0.0

    sports = [s.sport for s in legs]
    correlation_risk = "LOW"
    if len(set(sports)) == 1 and len(legs) >= 3:
        correlation_risk = "MEDIUM"
        warnings.append("Same-sport multi-leg combination: independence product is not a calibrated correlated-parlay probability")
    elif len(legs) >= 5:
        correlation_risk = "MEDIUM"
        warnings.append("Long combinations magnify model error; joint probability is an independence benchmark only")

    return ParlayResearch(
        mode=label,
        legs=legs,
        independent_fair_probability=fair,
        independent_market_probability=market,
        value_multiple=multiple,
        correlation_risk=correlation_risk,
        warnings=tuple(warnings),
    )
