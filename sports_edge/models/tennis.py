from __future__ import annotations
import math
from dataclasses import dataclass, field

from sports_edge.core.math import clamp


@dataclass
class EloState:
    overall: dict[str, float] = field(default_factory=dict)
    surface: dict[tuple[str, str], float] = field(default_factory=dict)
    base: float = 1500.0
    k: float = 28.0

    def rating(self, player: str, surface: str | None = None) -> float:
        if surface:
            sr = self.surface.get((player, surface.lower()))
            if sr is not None:
                return 0.62 * sr + 0.38 * self.overall.get(player, self.base)
        return self.overall.get(player, self.base)

    @staticmethod
    def win_prob(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

    def update(
        self,
        winner: str,
        loser: str,
        surface: str | None = None,
        importance: float = 1.0,
    ) -> None:
        ra = self.overall.get(winner, self.base)
        rb = self.overall.get(loser, self.base)
        p = self.win_prob(ra, rb)
        delta = self.k * importance * (1 - p)
        self.overall[winner] = ra + delta
        self.overall[loser] = rb - delta
        if surface:
            s = surface.lower()
            rsa = self.surface.get((winner, s), self.base)
            rsb = self.surface.get((loser, s), self.base)
            ps = self.win_prob(rsa, rsb)
            d = self.k * 1.08 * importance * (1 - ps)
            self.surface[(winner, s)] = rsa + d
            self.surface[(loser, s)] = rsb - d


def matchup_probability(
    rating_a: float,
    rating_b: float,
    serve_return_delta: float = 0.0,
    fatigue_delta: float = 0.0,
    injury_penalty_a: float = 0.0,
    injury_penalty_b: float = 0.0,
) -> float:
    elo_logit = (rating_a - rating_b) * math.log(10) / 400.0
    z = (
        elo_logit
        + 1.15 * serve_return_delta
        + 0.65 * fatigue_delta
        - injury_penalty_a
        + injury_penalty_b
    )
    return clamp(1 / (1 + math.exp(-z)), 0.01, 0.99)
