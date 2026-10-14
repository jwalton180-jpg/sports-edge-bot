"""Read-only, model-independent value qualification. Never infer fair odds from Kalshi."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite

@dataclass(frozen=True)
class ValueAudit:
    ticker: str
    side: str
    status: str
    executable_price: float | None
    break_even: float | None
    independent_probability: float | None
    edge: float | None
    reasons: tuple[str, ...]
    observed_at: str

def _price(market: dict, side: str) -> float | None:
    prefix = "yes" if side == "YES" else "no"
    raw = market.get(prefix + "_ask_dollars")
    try:
        p = float(raw) if raw is not None else float(market[prefix + "_ask"]) / 100
    except (ValueError, TypeError, KeyError):
        return None
    return p if isfinite(p) and 0 < p < 1 else None

def audit_value(market: dict, *, side: str, independent_probability: float | None,
                model_validated: bool, evidence_fresh: bool, coverage_complete: bool,
                rules_verified: bool, fee_fraction: float | None = None,
                observed_at: datetime | None = None) -> ValueAudit:
    """Strict qualification: missing fees are not silently assumed zero."""
    side = side.upper()
    if side not in {"YES", "NO"}:
        raise ValueError("side must be YES or NO")
    price = _price(market, side)
    reasons = []
    if str(market.get("status") or "").lower() not in {"open", "active"}:
        reasons.append("market_not_tradable")
    if not coverage_complete:
        reasons.append("incomplete_market_coverage")
    if not evidence_fresh:
        reasons.append("stale_or_conflicting_evidence")
    if not rules_verified:
        reasons.append("contract_rules_unverified")
    if not model_validated:
        reasons.append("independent_model_unvalidated")
    if price is None:
        reasons.append("executable_ask_missing")
    probability = None
    try:
        if independent_probability is not None:
            probability = float(independent_probability)
            if not isfinite(probability) or not 0 < probability < 1:
                probability = None
    except (ValueError, TypeError):
        pass
    if probability is None:
        reasons.append("independent_probability_missing")
    fee = None
    try:
        if fee_fraction is not None:
            fee = float(fee_fraction)
            if not isfinite(fee) or not 0 <= fee < 1:
                fee = None
    except (TypeError, ValueError):
        pass
    if fee is None:
        reasons.append("fees_unverified")
    break_even = price / (1 - fee) if price is not None and fee is not None else None
    edge = probability - break_even if probability is not None and break_even is not None else None
    if edge is not None and edge <= 0:
        reasons.append("no_positive_net_edge")
    if not reasons:
        status = "QUALIFIED_VALUE"
    elif "no_positive_net_edge" in reasons:
        status = "REJECTED"
    else:
        status = "RESEARCH_ONLY"
    return ValueAudit(
        ticker=str(market.get("ticker") or ""), side=side, status=status,
        executable_price=price, break_even=break_even,
        independent_probability=probability, edge=edge,
        reasons=tuple(reasons),
        observed_at=(observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
    )
