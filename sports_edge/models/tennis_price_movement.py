"""Read-only, score-unverified Tennis price movement diagnostics.

A price change is not a match prediction. No orders, fills, or recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceMovement:
    ticker: str
    selection: str
    quoted_ask: float
    trough: float
    recovery_points: float
    label: str = "PRICE MOVEMENT — SCORE NOT VERIFIED"
