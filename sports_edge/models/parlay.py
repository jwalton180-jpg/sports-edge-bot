from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
from scipy.stats import norm

from sports_edge.models.live_board import LiveSignal


@dataclass(frozen=True)
class ParlayResearch:
    mode: str
    preset: str
    legs: tuple[LiveSignal, ...]
    independent_fair_probability: float
    independent_market_probability: float
    value_multiple: float
    correlation_risk: str
    warnings: tuple[str, ...]


PRESETS: tuple[str, ...] = (
    "Best Available",
    "MLB Hits",
    "MLB Home Runs",
    "MLB Strikeouts",
    "NFL Passing",
    "NFL Rushing",
    "NFL Receiving",
    "NFL Touchdowns",
    "NFL Game Markets",
    "Mixed Sports",
)


def _text(signal: LiveSignal) -> str:
    return f"{signal.market} {signal.selection} {signal.ticker}".lower()


def _matches_preset(signal: LiveSignal, preset: str) -> bool:
    text = _text(signal)

    if preset in ("Best Available", "Mixed Sports"):
        return True
    if preset == "MLB Hits":
        return signal.sport == "MLB" and any(k in text for k in (" hit", "hits", "record a hit"))
    if preset == "MLB Home Runs":
        return signal.sport == "MLB" and any(k in text for k in ("home run", " homer", " hr"))
    if preset == "MLB Strikeouts":
        return signal.sport == "MLB" and any(k in text for k in ("strikeout", "strikeouts", " pitcher k", " k+"))
    if preset == "NFL Passing":
        return signal.sport == "NFL" and any(k in text for k in ("passing", "pass yards", "pass touchdown"))
    if preset == "NFL Rushing":
        return signal.sport == "NFL" and any(k in text for k in ("rushing", "rush yards", "rushing yards"))
    if preset == "NFL Receiving":
        return signal.sport == "NFL" and any(k in text for k in ("receiving", "receptions", "receiving yards"))
    if preset == "NFL Touchdowns":
        return signal.sport == "NFL" and any(k in text for k in ("touchdown", "score a td", " td "))
    if preset == "NFL Game Markets":
        return signal.sport == "NFL" and any(k in text for k in (" win", "winner", "spread", "total", "over", "under"))
    return False


def _unique_event_signals(signals: list[LiveSignal], preset: str) -> list[LiveSignal]:
    qualified = [s for s in signals if s.status == "QUALIFIED" and _matches_preset(s, preset)]
    qualified.sort(key=lambda s: (s.edge_points * s.confidence, s.data_quality), reverse=True)

    seen_events: set[str] = set()
    seen_contracts: set[tuple[str, str]] = set()
    out: list[LiveSignal] = []
    for signal in qualified:
        contract_key = (signal.ticker, signal.side)
        if contract_key in seen_contracts:
            continue
        event_key = signal.event_id or signal.ticker
        if event_key in seen_events:
            # Same-game multi-leg correlation is intentionally blocked until
            # sport-specific joint distributions are validated.
            continue
        seen_contracts.add(contract_key)
        seen_events.add(event_key)
        out.append(signal)
    return out



def independent_joint(probabilities: list[float] | tuple[float, ...]) -> float:
    """Independence benchmark for a multi-leg ticket."""
    if not probabilities:
        return 0.0
    vals = [float(p) for p in probabilities]
    if any(p < 0.0 or p > 1.0 for p in vals):
        raise ValueError("probabilities must be between 0 and 1")
    return float(prod(vals))


def correlated_joint_monte_carlo(
    probabilities: list[float] | tuple[float, ...],
    correlation_matrix: list[list[float]] | np.ndarray,
    simulations: int = 50000,
    seed: int = 7,
) -> float:
    """Gaussian-copula estimate of all-leg success probability.

    This is a research primitive only. Callers must provide a validated
    correlation matrix; the app must not invent correlation just to make a
    parlay look better.
    """
    ps = np.asarray(probabilities, dtype=float)
    corr = np.asarray(correlation_matrix, dtype=float)
    n = len(ps)
    if n == 0:
        return 0.0
    if corr.shape != (n, n):
        raise ValueError("correlation matrix shape must match probabilities")
    if np.any(ps <= 0.0) or np.any(ps >= 1.0):
        raise ValueError("correlated simulation requires probabilities strictly between 0 and 1")
    if not np.allclose(corr, corr.T, atol=1e-9):
        raise ValueError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-9):
        raise ValueError("correlation matrix diagonal must equal 1")
    eig = np.linalg.eigvalsh(corr)
    if np.min(eig) < -1e-8:
        raise ValueError("correlation matrix must be positive semidefinite")

    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(n), corr, size=max(1, int(simulations)))
    thresholds = norm.ppf(ps)
    successes = z <= thresholds
    return float(np.mean(np.all(successes, axis=1)))


def build_parlay_research(
    signals: list[LiveSignal],
    mode: str = "high_confidence",
    *,
    preset: str = "Best Available",
    leg_count: int | None = None,
) -> ParlayResearch:
    if preset not in PRESETS:
        preset = "Best Available"

    pool = _unique_event_signals(signals, preset)
    warnings: list[str] = []

    if mode == "longshot":
        wanted = 6 if leg_count is None else max(5, min(int(leg_count), 10))
        minimum = 5
        label = "Longshot Lab"
    else:
        wanted = 4 if leg_count is None else max(2, min(int(leg_count), 8))
        minimum = 2
        label = "High-Confidence Builder"

    legs = tuple(pool[:wanted])
    if len(legs) < minimum:
        warnings.append(f"Need at least {minimum} independently qualified event legs; found {len(legs)}")
    if len(legs) < wanted:
        warnings.append(f"Only {len(legs)} qualified leg(s) matched the {preset} preset")

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

    if preset == "Mixed Sports" and len(set(sports)) < 2 and len(legs) >= 2:
        warnings.append("Mixed Sports requested, but the current qualified pool does not contain at least two sports")

    return ParlayResearch(
        mode=label,
        preset=preset,
        legs=legs,
        independent_fair_probability=fair,
        independent_market_probability=market,
        value_multiple=multiple,
        correlation_risk=correlation_risk,
        warnings=tuple(warnings),
    )


def kalshi_copy_ticket(legs: tuple[LiveSignal, ...] | list[LiveSignal]) -> str:
    """Plain-text manual-entry ticket.

    This intentionally does not submit orders. The ticker + side can be pasted
    into Kalshi search/manual entry while the live price is rechecked there.
    """

    lines = ["SPORTS EDGE — MANUAL KALSHI TICKET", "Recheck live price before entry.", ""]
    for idx, signal in enumerate(legs, 1):
        cents = round(signal.market_probability * 100)
        lines.append(
            f"{idx}. {signal.ticker} | {signal.side} | {signal.selection} | reference {cents}¢ | edge {signal.edge_points:+.1f}pp"
        )
    if len(lines) == 3:
        lines.append("No qualified legs.")
    return "\n".join(lines)
