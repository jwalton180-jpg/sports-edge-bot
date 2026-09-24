from __future__ import annotations
import math
from typing import Iterable

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def american_to_decimal(odds: float) -> float:
    if odds == 0: raise ValueError("American odds cannot be zero")
    return 1 + (100 / abs(odds) if odds < 0 else odds / 100)

def american_to_implied(odds: float) -> float:
    return 1.0 / american_to_decimal(odds)

def decimal_to_american(decimal_odds: float) -> float:
    if decimal_odds <= 1: raise ValueError("Decimal odds must be > 1")
    if decimal_odds >= 2: return 100 * (decimal_odds - 1)
    return -100 / (decimal_odds - 1)

def no_vig_two_way(p1_raw: float, p2_raw: float) -> tuple[float, float]:
    s = p1_raw + p2_raw
    if s <= 0: raise ValueError("Probabilities must have positive sum")
    return p1_raw / s, p2_raw / s

def no_vig_from_american(odds_a: float, odds_b: float) -> tuple[float, float]:
    return no_vig_two_way(american_to_implied(odds_a), american_to_implied(odds_b))

def consensus_probability(probabilities: Iterable[float], weights: Iterable[float] | None = None) -> float:
    ps=[clamp(float(p)) for p in probabilities]
    if not ps: raise ValueError("At least one probability is required")
    if weights is None: return sum(ps)/len(ps)
    ws=[max(0.0,float(w)) for w in weights]
    if len(ws)!=len(ps) or sum(ws)<=0: raise ValueError("Weights must match probabilities and have positive sum")
    return sum(p*w for p,w in zip(ps,ws))/sum(ws)

def kalshi_contract_ev(fair_probability: float, price_dollars: float, fee_dollars: float = 0.0) -> float:
    p=clamp(fair_probability); price=clamp(price_dollars)
    return p-price-max(0.0,fee_dollars)

def edge_points(fair_probability: float, market_probability: float) -> float:
    return 100.0*(clamp(fair_probability)-clamp(market_probability))

def brier_score(y_true: Iterable[int], p_pred: Iterable[float]) -> float:
    ys=list(y_true); ps=list(p_pred)
    if len(ys)!=len(ps) or not ys: raise ValueError("Inputs must be same non-zero length")
    return sum((int(y)-clamp(float(p)))**2 for y,p in zip(ys,ps))/len(ys)

def log_loss(y_true: Iterable[int], p_pred: Iterable[float], eps: float = 1e-12) -> float:
    ys=list(y_true); ps=list(p_pred)
    if len(ys)!=len(ps) or not ys: raise ValueError("Inputs must be same non-zero length")
    total=0.0
    for y,p in zip(ys,ps):
        p=min(1-eps,max(eps,float(p)))
        total+=-(int(y)*math.log(p)+(1-int(y))*math.log(1-p))
    return total/len(ys)
