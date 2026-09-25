from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

class Freshness(str, Enum):
    FRESH = "fresh"
    DEGRADED = "degraded"
    STALE = "stale"
    DOWN = "down"

@dataclass(frozen=True)
class SourceHealth:
    source: str
    observed_at: datetime | None
    received_at: datetime
    latency_ms: float | None
    error: str | None = None

    def age_seconds(self, now: datetime | None = None) -> float | None:
        if self.observed_at is None:
            return None
        now = now or datetime.now(timezone.utc)
        obs = self.observed_at if self.observed_at.tzinfo else self.observed_at.replace(tzinfo=timezone.utc)
        return max(0.0, (now - obs).total_seconds())

    def status(self, stale_s: float = 20, hard_stale_s: float = 60) -> Freshness:
        if self.error:
            return Freshness.DOWN
        age = self.age_seconds()
        if age is None:
            return Freshness.DEGRADED
        if age <= stale_s:
            return Freshness.FRESH
        if age <= hard_stale_s:
            return Freshness.DEGRADED
        return Freshness.STALE
