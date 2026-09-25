from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Iterable

from sports_edge.core.math import clamp


@dataclass(frozen=True)
class ModelEvidence:
    sport: str
    model_name: str
    fair_probability: float
    confidence: float
    sample_size: int
    factors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return (
            0.01 <= self.fair_probability <= 0.99
            and self.confidence >= 0.35
            and self.sample_size >= 1
        )


def model_dominant_fair(
    model: ModelEvidence,
    *,
    sportsbook_probability: float | None = None,
    sportsbook_book_count: int = 0,
    sportsbook_age_s: float | None = None,
) -> tuple[float, float, tuple[str, ...]]:
    """Return a model-first fair probability and evidence quality.

    Sportsbooks are a secondary calibration input only. The model always keeps
    at least 75% of the blend weight; stale/weak sportsbook data gets no weight.
    """
    model_p = clamp(float(model.fair_probability), 0.01, 0.99)
    model_conf = clamp(float(model.confidence), 0.0, 1.0)
    reasons = [
        f"{model.model_name}: {model_p:.1%} fair at {model_conf:.0%} model confidence",
        *model.factors,
    ]

    book_weight = 0.0
    if (
        sportsbook_probability is not None
        and sportsbook_book_count >= 2
        and sportsbook_age_s is not None
        and sportsbook_age_s <= 120
    ):
        freshness = clamp(1.0 - sportsbook_age_s / 120.0, 0.0, 1.0)
        depth = clamp(sportsbook_book_count / 5.0, 0.0, 1.0)
        # Books can confirm/calibrate but never become the primary signal.
        book_weight = min(0.25, 0.10 + 0.10 * depth + 0.05 * freshness)
        reasons.append(
            f"Sportsbook cross-check: {sportsbook_probability:.1%} from "
            f"{sportsbook_book_count} fresh book(s), weight {book_weight:.0%}"
        )

    if book_weight > 0:
        fair = (1.0 - book_weight) * model_p + book_weight * clamp(float(sportsbook_probability), 0.01, 0.99)
    else:
        fair = model_p
        reasons.append("Sportsbook price not required for model qualification")

    evidence_quality = clamp(
        0.78 * model_conf
        + 0.12 * clamp(model.sample_size / 25.0, 0.0, 1.0)
        + 0.10 * (1.0 if book_weight > 0 else 0.0),
        0.0,
        1.0,
    )
    return clamp(fair, 0.01, 0.99), evidence_quality, tuple(reasons)


def log5_probability(win_pct_a: float, win_pct_b: float) -> float:
    """Bill James style Log5 baseline from independent team win rates."""
    a = clamp(win_pct_a, 0.05, 0.95)
    b = clamp(win_pct_b, 0.05, 0.95)
    numerator = a - a * b
    denominator = a + b - 2 * a * b
    if abs(denominator) < 1e-9:
        return 0.5
    return clamp(numerator / denominator, 0.01, 0.99)


def logistic_adjust(base_probability: float, adjustment_logit: float) -> float:
    p = clamp(base_probability, 0.01, 0.99)
    logit = log(p / (1.0 - p)) + adjustment_logit
    return clamp(1.0 / (1.0 + exp(-logit)), 0.01, 0.99)


def team_record_model(
    *,
    sport: str,
    team_a: str,
    team_b: str,
    win_pct_a: float,
    win_pct_b: float,
    games_a: int,
    games_b: int,
    home_a: bool | None = None,
    recent_pct_a: float | None = None,
    recent_pct_b: float | None = None,
    differential_per_game_a: float | None = None,
    differential_per_game_b: float | None = None,
) -> ModelEvidence:
    """Independent team-strength baseline for game-winner markets.

    This is intentionally transparent and conservative; sport-specific richer
    feature stores can replace/augment it without changing the parlay contract.
    """
    p = log5_probability(win_pct_a, win_pct_b)
    factors = [
        f"Season records: {team_a} {win_pct_a:.3f} vs {team_b} {win_pct_b:.3f}",
    ]

    adjustment = 0.0
    if recent_pct_a is not None and recent_pct_b is not None:
        recent_delta = clamp(recent_pct_a - recent_pct_b, -0.5, 0.5)
        adjustment += 0.45 * recent_delta
        factors.append(f"Recent-form delta {recent_delta:+.3f}")

    if differential_per_game_a is not None and differential_per_game_b is not None:
        diff_delta = differential_per_game_a - differential_per_game_b
        scale = 0.045 if sport == "MLB" else (0.030 if sport == "NFL" else 0.018)
        adjustment += clamp(scale * diff_delta, -0.35, 0.35)
        factors.append(f"Scoring/run differential delta {diff_delta:+.2f} per game")

    if home_a is not None:
        home_bump = 0.10 if sport == "NFL" else (0.07 if sport in {"NBA", "WNBA"} else 0.045)
        adjustment += home_bump if home_a else -home_bump
        factors.append("Home/away context included")

    p = logistic_adjust(p, adjustment)
    sample = max(1, min(games_a, games_b))
    confidence = clamp(
        0.36
        + 0.34 * clamp(sample / (16.0 if sport == "NFL" else 50.0), 0.0, 1.0)
        + (0.08 if differential_per_game_a is not None and differential_per_game_b is not None else 0.0)
        + (0.06 if recent_pct_a is not None and recent_pct_b is not None else 0.0),
        0.0,
        0.86,
    )
    if sport == "NFL" and sample < 5:
        confidence = min(confidence, 0.52)

    warnings: list[str] = []
    if sample < (4 if sport == "NFL" else 10):
        warnings.append("small current-season sample")
    if differential_per_game_a is None or differential_per_game_b is None:
        warnings.append("team differential feature unavailable")

    return ModelEvidence(
        sport=sport,
        model_name=f"{sport} public team-strength model",
        fair_probability=p,
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
        warnings=tuple(warnings),
    )
