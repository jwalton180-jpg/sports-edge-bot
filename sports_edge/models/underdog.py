from __future__ import annotations
from dataclasses import dataclass
from sports_edge.core.math import clamp, edge_points, kalshi_contract_ev

@dataclass(frozen=True)
class UnderdogSignal:
    market_probability: float
    fair_probability: float
    edge_points: float
    ev_per_contract: float
    value_multiple: float
    tier: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]

def score_underdog(market_probability: float, fair_probability: float, *, data_quality: float, source_age_s: float, max_age_s: float = 12.0, min_edge_points: float = 4.0, reasons: list[str] | None = None, warnings: list[str] | None = None) -> UnderdogSignal:
    m=clamp(float(market_probability),.001,.999); f=clamp(float(fair_probability),.001,.999); edge=edge_points(f,m); ev=kalshi_contract_ev(f,m); w=list(warnings or [])
    if not .03<=m<=.30: w.append("Outside underdog-radar market band (3%–30%)")
    if source_age_s>max_age_s: w.append("Live state stale")
    if data_quality<.75: w.append("Data quality below underdog threshold")
    if edge<min_edge_points: w.append("Fair-value gap below threshold")
    if w: tier="PASS"
    elif m<=.12 and edge>=8: tier="ASYMMETRIC"
    elif edge>=6: tier="STRONG_VALUE"
    else: tier="VALUE"
    return UnderdogSignal(m,f,edge,ev,f/m if m else float("inf"),tier,tuple(reasons or []),tuple(w))
