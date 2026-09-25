from __future__ import annotations
import math
from sports_edge.core.math import clamp


def normal_over_probability(mean: float, sd: float, line: float) -> float:
    if sd <= 0: raise ValueError("sd must be positive")
    z = (line - mean) / (sd * math.sqrt(2))
    cdf = 0.5 * (1 + math.erf(z))
    return clamp(1 - cdf)


def game_win_probability(point_margin_mean: float, point_margin_sd: float = 13.6) -> float:
    return normal_over_probability(point_margin_mean, point_margin_sd, 0.0)


def prop_mean(base_rate: float, expected_opportunities: float, matchup_multiplier: float = 1.0,
              role_multiplier: float = 1.0, weather_multiplier: float = 1.0) -> float:
    return max(0.0, base_rate * expected_opportunities * matchup_multiplier * role_multiplier * weather_multiplier)
