from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class MarketQuote:
    ticker: str
    sport: str
    title: str
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    last_price: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    observed_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeCard:
    key: str
    sport: str
    market: str
    selection: str
    market_probability: float
    fair_probability: float
    edge_points: float
    ev_per_dollar: float
    confidence: float
    data_quality: float
    reasons: list[str]
    warnings: list[str]
    observed_at: datetime | None = None
