from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

class PlayerStatus(str, Enum):
    ACTIVE="active"; PROBABLE="probable"; QUESTIONABLE="questionable"; DOUBTFUL="doubtful"; OUT="out"; IL="il"; UNKNOWN="unknown"

@dataclass(frozen=True)
class AvailabilityUpdate:
    sport: str
    player_id: str
    player_name: str
    status: PlayerStatus
    source: str
    observed_at: datetime
    detail: str = ""

@dataclass
class RecommendationDependency:
    recommendation_id: str
    players: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    valid: bool = True
    invalid_reason: str | None = None
    invalidated_at: datetime | None = None

class AvailabilityLedger:
    MATERIAL = {PlayerStatus.OUT, PlayerStatus.IL, PlayerStatus.DOUBTFUL}
    def __init__(self):
        self.latest: dict[str, AvailabilityUpdate] = {}
        self.dependencies: dict[str, RecommendationDependency] = {}

    def register(self, dep: RecommendationDependency):
        self.dependencies[dep.recommendation_id] = dep

    def ingest(self, update: AvailabilityUpdate) -> list[str]:
        prev = self.latest.get(update.player_id)
        if prev and update.observed_at <= prev.observed_at:
            return []
        self.latest[update.player_id] = update
        invalidated=[]
        became_material = update.status in self.MATERIAL and (prev is None or prev.status not in self.MATERIAL or prev.status != update.status)
        if became_material:
            for dep in self.dependencies.values():
                if dep.valid and update.player_id in dep.players:
                    dep.valid=False
                    dep.invalid_reason=f"{update.player_name} status changed to {update.status.value} ({update.source})"
                    dep.invalidated_at=datetime.now(timezone.utc)
                    invalidated.append(dep.recommendation_id)
        return invalidated

    def current(self, player_id: str) -> AvailabilityUpdate | None:
        return self.latest.get(player_id)
