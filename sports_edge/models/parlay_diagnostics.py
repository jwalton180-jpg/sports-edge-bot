"""Transparent all-sport research candidate and rejection diagnostics.

Never turns a blocked candidate into a qualified recommendation. Data appears
as a labeled research watchlist with explicit executable quote/fee status.
"""
from __future__ import annotations

from collections import Counter
from math import isfinite


def parlay_evidence_diagnostics(candidates, *, max_preview: int = 16):
    if not 1 <= max_preview <= 50:
        raise ValueError("bounded preview required")
    reasons = Counter()
    sport_counts = Counter()
    candidates = list(candidates)
    preview = []
    modeled = 0
    priced = 0
    eligible_quote = 0
    for c in candidates:
        sport = str(c.sport or "Unknown")
        sport_counts[sport] += 1
        p = c.model_probability
        try:
            model_available = (p is not None and isfinite(float(p))
                               and 0 <= float(p) <= 1 and bool(c.model_name))
        except (TypeError,ValueError):
            model_available = False
        if model_available:
            modeled += 1
        try:
            ask = float(c.kalshi_price)
            price_available = bool(c.kalshi_ticker) and isfinite(ask) and 0 < ask < 1
        except (TypeError,ValueError):
            price_available = False
            ask = None
        if price_available:
            priced += 1
        fee = c.entry_fee_per_contract
        try:
            fee_ok = fee is not None and isfinite(float(fee)) and 0 <= float(fee) < 1
        except (TypeError,ValueError):
            fee_ok = False
        blocked = tuple(dict.fromkeys(c.execution_audit_reasons or ()))
        for reason in blocked:
            reasons[str(reason)] += 1
        if c.execution_audit_status != "QUALIFIED_VALUE" and not blocked:
            reasons["scoped_validation_or_quote_not_reviewed"] += 1
        if model_available and price_available and fee_ok:
            eligible_quote += 1
            net_edge = round(float(p) - ask - float(fee), 4)
        else:
            net_edge = None
        if model_available and price_available:
            preview.append({
                "Sport":sport, "Match / contract":str(c.event_title or c.kalshi_ticker),
                "Market":str(c.market_label or c.market_key), "Selection":str(c.selection),
                "Ticker":str(c.kalshi_ticker),"Model P":round(float(p),4),
                "Kalshi ask":round(ask,4), "Entry fee":round(float(fee),4) if fee_ok else None,
                "Indicative net edge":net_edge, "Review":"QUALIFIED_VALUE" if c.execution_audit_status=="QUALIFIED_VALUE" else "RESEARCH ONLY",
                "Blocking evidence":"; ".join(blocked[:4]) or ("none" if c.execution_audit_status=="QUALIFIED_VALUE" else "review required")
            })
    preview.sort(key=lambda x:(
        x["Indicative net edge"] is not None,
        x["Indicative net edge"] if x["Indicative net edge"] is not None else -10.,
        x["Model P"]),reverse=True)
    return {"candidates":len(candidates),"modeled":modeled,"priced":priced,
            "modeled_and_fee_priced":eligible_quote,
            "qualified":sum(c.execution_audit_status=="QUALIFIED_VALUE" for c in candidates),
            "sports":dict(sport_counts),"common_blockers":reasons.most_common(12),
            "research_preview":preview[:max_preview],
            "scope":"on_demand_research_only_not_validated_recommendations"}
