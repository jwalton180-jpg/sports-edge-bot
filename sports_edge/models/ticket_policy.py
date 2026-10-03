from __future__ import annotations

from dataclasses import dataclass
import re

from sports_edge.models.parlay_candidates import ParlayCandidateLeg


@dataclass(frozen=True)
class TicketPolicyProfile:
    involvement: str
    variance: str
    role_check: str
    reasons: tuple[str, ...]


_HIGH_INVOLVEMENT_KEYS = {
    "h2h",
    "model_h2h",
    "mlb_spread",
    "mlb_game_total",
    "mlb_team_total",
    "nfl_spread",
    "nfl_game_total",
    "nfl_team_total",
    "wnba_spread",
    "wnba_game_total",
    "wnba_team_total",
    "passing_yards",
    "pass_attempts",
    "pass_completions",
    "rushing_yards",
    "rush_attempts",
    "receiving_yards",
    "receptions",
    "rush_receiving_yards",
    "batter_hits",
    "batter_total_bases",
    "pitcher_strikeouts",
    "tennis_games_total",
    "tennis_game_spread",
    "tennis_set_winner",
}

_HIGH_VARIANCE_KEYS = {
    "batter_home_runs",
    "touchdowns",
    "passing_tds",
    "pass_interceptions",
    "first_touchdown",
    "first_basket",
    "longest_reception",
    "double_faults",
    "exact_score",
}

_ROLE_BLOCK_TOKENS = (
    "inactive",
    "ruled out",
    "not starting",
    "will not start",
    "minutes restriction",
    "snap restriction",
    "pitch-count restriction",
    "platoon risk",
    "withdrawal",
    "retired injured",
)

_ROLE_RECHECK_TOKENS = (
    "lineup",
    "batting-order",
    "starter unavailable",
    "probable starter unavailable",
    "injury",
    "fitness",
    "role uncertainty",
    "workload uncertainty",
    "minutes",
    "snap",
)


def _normalized_key(leg: ParlayCandidateLeg) -> str:
    return str(leg.market_key or "").strip().lower()


def _threshold_hint(leg: ParlayCandidateLeg) -> float | None:
    text = f"{leg.selection} {leg.market_label}"
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*\+?", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def profile_leg(leg: ParlayCandidateLeg) -> TicketPolicyProfile:
    key = _normalized_key(leg)
    warnings = " ".join(str(x).lower() for x in leg.model_warnings)
    reasons: list[str] = []

    role_check = "CLEAR"
    if any(token in warnings for token in _ROLE_BLOCK_TOKENS):
        role_check = "BLOCK"
        reasons.append("Material role/availability uncertainty is present")
    elif any(token in warnings for token in _ROLE_RECHECK_TOKENS):
        role_check = "RECHECK"
        reasons.append("Pre-event role/availability recheck required")

    if key in _HIGH_VARIANCE_KEYS:
        variance = "HIGH"
    elif key in _HIGH_INVOLVEMENT_KEYS:
        variance = "LOW"
    else:
        variance = "MEDIUM"

    threshold = _threshold_hint(leg)
    if key == "batter_hits" and threshold is not None:
        variance = "LOW" if threshold <= 1.0 else "HIGH"
    elif key == "batter_total_bases" and threshold is not None:
        variance = "LOW" if threshold <= 1.0 else "MEDIUM"
    elif key == "receptions" and threshold is not None and threshold >= 6:
        variance = "MEDIUM"
    elif key in {"rushing_yards", "receiving_yards"} and threshold is not None and threshold >= 90:
        variance = "MEDIUM"

    confidence = float(leg.model_confidence or 0.0)
    if role_check == "BLOCK":
        involvement = "LOW"
    elif key in _HIGH_INVOLVEMENT_KEYS and confidence >= 0.58 and variance != "HIGH":
        involvement = "HIGH"
    elif confidence >= 0.45:
        involvement = "MEDIUM"
    else:
        involvement = "LOW"

    if involvement == "HIGH":
        reasons.append("Stable, high-opportunity market family with sufficient model confidence")
    elif involvement == "MEDIUM":
        reasons.append("Usable opportunity profile, but not strong enough for HIGH involvement")
    else:
        reasons.append("Low/uncertain involvement or confidence")

    if variance == "LOW":
        reasons.append("Volume-oriented or survivable market family")
    elif variance == "HIGH":
        reasons.append("High-variance outcome; use only when the edge/payout need justifies it")

    return TicketPolicyProfile(
        involvement=involvement,
        variance=variance,
        role_check=role_check,
        reasons=tuple(reasons),
    )


def involvement_rank(value: str) -> int:
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(value, 0)


def variance_rank(value: str) -> int:
    # Higher is better for construction.
    return {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(value, 0)


def role_rank(value: str) -> int:
    return {"BLOCK": 0, "RECHECK": 1, "CLEAR": 2}.get(value, 0)



def construction_rank_key(
    assessment,
    *,
    prioritize_payout_multiplier: bool = False,
) -> tuple[float, ...]:
    """Return the single canonical stability-first ticket ranking key.

    Public/social methodology is intentionally the final tie-breaker. It may
    distinguish otherwise equivalent qualified legs, but it cannot outrank the
    independent model score, price edge, evidence quality, or payout policy.
    Keeping this key here also prevents the stale-builder compatibility path
    from drifting away from normal ticket construction.
    """

    def number(name: str, default: float) -> float:
        value = getattr(assessment, name, None)
        return default if value is None else float(value)

    base = (
        float(involvement_rank(str(getattr(assessment, "involvement_rating", "LOW")))),
        float(role_rank(str(getattr(assessment, "role_check", "RECHECK")))),
        float(variance_rank(str(getattr(assessment, "variance_rating", "HIGH")))),
        number("score", 0.0),
        number("edge_points", -999.0),
    )

    if prioritize_payout_multiplier:
        price = getattr(assessment, "kalshi_probability", None)
        payout = 1.0 / float(price) if price is not None and float(price) > 0 else 0.0
        core = base + (
            number("expected_roi_on_cost", -999.0),
            payout,
        )
    else:
        core = base + (
            number("evidence_quality", 0.0),
            number("fair_probability", 0.0),
        )

    return core + (number("community_methodology_score", 0.0),)
