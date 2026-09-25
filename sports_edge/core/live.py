from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

@dataclass
class LiveObservation:
    source: str
    key: str
    value: Any
    observed_at: datetime
    received_at: datetime
    confidence: float = 1.0

    @property
    def age_seconds(self) -> float:
        return max(0.0, (datetime.now(timezone.utc) - self.observed_at).total_seconds())

class LiveFusion:
    """Freshness-aware source fusion. Never silently selects stale data over fresh data."""
    def __init__(self, hard_stale_seconds: float = 60.0):
        self.hard_stale_seconds = hard_stale_seconds
        self._obs: dict[str, list[LiveObservation]] = {}

    def add(self, obs: LiveObservation):
        self._obs.setdefault(obs.key, []).append(obs)
        self._obs[obs.key] = sorted(self._obs[obs.key], key=lambda x: x.received_at, reverse=True)[:12]

    def best(self, key: str) -> LiveObservation | None:
        choices = [o for o in self._obs.get(key, []) if o.age_seconds <= self.hard_stale_seconds]
        if not choices:
            return None
        return sorted(choices, key=lambda o: (o.age_seconds, -o.confidence))[0]

    def disagreement(self, key: str) -> bool:
        choices = [o for o in self._obs.get(key, []) if o.age_seconds <= self.hard_stale_seconds]
        if len(choices) < 2:
            return False
        vals = {str(o.value) for o in choices[:3]}
        return len(vals) > 1
