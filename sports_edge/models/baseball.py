from __future__ import annotations
import math
import numpy as np

from sports_edge.core.math import clamp


def poisson_game_win_probability(
    home_runs_mean: float,
    away_runs_mean: float,
    simulations: int = 80000,
    seed: int = 7,
) -> float:
    if home_runs_mean <= 0 or away_runs_mean <= 0:
        raise ValueError("Run means must be positive")
    rng = np.random.default_rng(seed)
    h = rng.poisson(home_runs_mean, simulations)
    a = rng.poisson(away_runs_mean, simulations)
    wins = (h > a).mean()
    ties = (h == a).mean()
    return float(clamp(wins + 0.5 * ties, 0.01, 0.99))


def at_least_one_hit_probability(per_pa_hit_prob: float, expected_pa: float) -> float:
    p = clamp(per_pa_hit_prob, 0.001, 0.999)
    pa = max(0.0, expected_pa)
    return clamp(1 - math.exp(pa * math.log(1 - p)))
