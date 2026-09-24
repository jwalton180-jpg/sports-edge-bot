from __future__ import annotations
from sports_edge.core.math import clamp, edge_points, kalshi_contract_ev
from sports_edge.core.types import EdgeCard

def grade_edge(key: str, sport: str, market: str, selection: str, fair_p: float, market_p: float, model_confidence: float, data_quality: float, reasons: list[str] | None = None, warnings: list[str] | None = None, min_edge_points: float = 2.0) -> EdgeCard:
    fair_p=clamp(fair_p); market_p=clamp(market_p); edge=edge_points(fair_p,market_p); dq=clamp(data_quality); mc=clamp(model_confidence)
    conf=clamp((0.58*mc+0.42*dq)*min(1.0,max(0.25,abs(edge)/8.0)))
    warns=list(warnings or [])
    if dq<.7: warns.append("Data quality below premium threshold")
    if abs(edge)<min_edge_points: warns.append("Edge below action threshold")
    return EdgeCard(key,sport,market,selection,market_p,fair_p,edge,kalshi_contract_ev(fair_p,market_p),conf,dq,list(reasons or []),warns)

def should_surface(card: EdgeCard, min_edge_points: float = 2.0, min_confidence: float = 0.55, min_data_quality: float = 0.70) -> bool:
    return card.edge_points>=min_edge_points and card.confidence>=min_confidence and card.data_quality>=min_data_quality
