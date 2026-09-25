from __future__ import annotations
from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class PublicTrackRecord:
    handle: str
    wins: int
    losses: int
    pushes: int = 0
    avg_clv_points: float | None = None
    verified_posts: int | None = None
    roi: float | None = None
    source: str | None = None

    @property
    def decisions(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.decisions if self.decisions else 0.0

    def wilson_lower_bound(self, z: float = 1.96) -> float:
        n = self.decisions
        if n == 0:
            return 0.0
        p = self.win_rate
        den = 1 + z * z / n
        centre = p + z * z / (2 * n)
        adj = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        return max(0.0, (centre - adj) / den)

    def evidence_quality(self) -> float:
        n = self.decisions
        sample = min(1.0, n / 300.0)
        verified = min(
            1.0,
            (self.verified_posts if self.verified_posts is not None else 0) / max(1, n),
        )
        clv = 1.0 if self.avg_clv_points is not None else 0.35
        return max(0.0, min(1.0, 0.55 * sample + 0.30 * verified + 0.15 * clv))

    def tail_eligible(
        self,
        *,
        min_decisions: int = 100,
        min_verified_fraction: float = 0.90,
        require_positive_clv: bool = True,
    ) -> bool:
        if self.decisions < min_decisions:
            return False
        verified = self.verified_posts if self.verified_posts is not None else 0
        if verified / max(1, self.decisions) < min_verified_fraction:
            return False
        if require_positive_clv and (self.avg_clv_points is None or self.avg_clv_points <= 0):
            return False
        return self.evidence_quality() >= 0.70


def consensus_support(records: list[PublicTrackRecord]) -> dict:
    eligible = [r for r in records if r.tail_eligible()]
    if not eligible:
        return {
            "eligible_count": 0,
            "quality": 0.0,
            "avg_wilson_lower": 0.0,
            "avg_clv_points": None,
        }

    quality = sum(r.evidence_quality() for r in eligible) / len(eligible)
    wilson = sum(r.wilson_lower_bound() for r in eligible) / len(eligible)
    clv_values = [r.avg_clv_points for r in eligible if r.avg_clv_points is not None]
    avg_clv = sum(clv_values) / len(clv_values) if clv_values else None
    return {
        "eligible_count": len(eligible),
        "quality": quality,
        "avg_wilson_lower": wilson,
        "avg_clv_points": avg_clv,
    }
