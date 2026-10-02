from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sports_edge.core.math import clamp
from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.ticket_policy import profile_leg


@dataclass(frozen=True)
class SocialResearchSource:
    handle: str
    platform: str
    evidence_tier: str
    focus: str
    lesson: str
    caution: str
    url: str
    observed_at: str = "2026-10-02"


# Public-source research snapshot. These are methodology/process sources, not
# automatic tail signals. A source never changes SportEdge fair probability
# merely because it has a large P&L badge, many followers, or a recent winner.
SOCIAL_RESEARCH_SOURCES: tuple[SocialResearchSource, ...] = (
    SocialResearchSource(
        handle="vast.stand / @LT_Picks",
        platform="Kalshi + X",
        evidence_tier="PROCESS+",
        focus="Sports; team context; tennis; roster/coaching changes",
        lesson=(
            "Use current roster, coaching, role and matchup context instead of "
            "treating recent box scores as self-explanatory."
        ),
        caution="Public profit/volume is credibility context, not proof that every posted pick is +EV.",
        url="https://kalshi.com/ideas/profiles/vast.stand",
    ),
    SocialResearchSource(
        handle="Hidden.Edge",
        platform="Kalshi",
        evidence_tier="PROCESS+",
        focus="Sports; tennis; combos",
        lesson=(
            "Separate normal plays from explicitly risky plays and treat price, "
            "matchup and current state as distinct inputs."
        ),
        caution="Public history includes both wins and losses; never blind-tail.",
        url="https://kalshi.com/ideas/profiles/Hidden.Edge",
    ),
    SocialResearchSource(
        handle="RIBBRIT",
        platform="Kalshi + community",
        evidence_tier="LONGSHOT-RESEARCH",
        focus="Underpriced longshots; small-stake experiments; tennis/fights",
        lesson=(
            "Cheap contracts can be useful only when independently underpriced; "
            "small-stake longshots should remain separate from the strong core."
        ),
        caution="Longshot success is high variance and does not qualify a leg by itself.",
        url="https://kalshi.com/ideas/profiles/ribbrit",
    ),
    SocialResearchSource(
        handle="endthebookie",
        platform="Kalshi",
        evidence_tier="WATCHLIST",
        focus="Football sides/spreads/totals; small public sample",
        lesson="Prefer simple game-level markets when the edge is clear rather than adding ornamental legs.",
        caution="Recent public sample is too small to label consistently predictive.",
        url="https://kalshi.com/ideas/profiles/endthebookie",
    ),
    SocialResearchSource(
        handle="metromike / @MetroPassMike",
        platform="Kalshi + X/podcast",
        evidence_tier="PROCESS",
        focus="Price discipline; tools; explaining why an edge should exist; sizing",
        lesson=(
            "Require an explainable reason for the edge, use multiple tools to "
            "cross-check it, and separate bet sizing from pick selection."
        ),
        caution="Sparse public Kalshi posting means this is a process reference, not a tail source.",
        url="https://kalshi.com/ideas/profiles/metromike",
    ),
    SocialResearchSource(
        handle="@TheArbFather / tracked +EV community",
        platform="X",
        evidence_tier="PROCESS",
        focus="+EV/player props; line shopping; tracked results; sharp-money tools",
        lesson=(
            "Track every play, measure price quality/CLV, and use sharp-money or "
            "consensus movement only as secondary confirmation."
        ),
        caution="Social performance claims remain secondary unless independently tracked and auditable.",
        url="https://x.com/TheArbFather",
    ),
)


_CONTEXT_TERMS = (
    "role",
    "usage",
    "lineup",
    "batting",
    "starter",
    "matchup",
    "surface",
    "serve",
    "return",
    "head-to-head",
    "h2h",
    "fatigue",
    "workload",
    "recent",
    "route",
    "target",
    "carry",
    "minutes",
    "touch",
    "pace",
    "weather",
    "bullpen",
)


def community_methodology_fit(
    leg: ParlayCandidateLeg,
    *,
    fair_probability: float,
    kalshi_probability: float | None,
    mode: str,
) -> tuple[float, tuple[str, ...]]:
    """Score how well a qualified leg follows repeatable public sharp principles.

    This is deliberately *not* a crowdsourced fair-probability model. It only
    breaks ties/ranks already model-qualified legs by process quality: stable
    involvement, survivable variance, role clarity, context depth and price
    discipline. Social posts never create a candidate or turn negative EV into
    positive EV.
    """
    policy = profile_leg(leg)
    notes: list[str] = []
    score = 0.08

    if policy.involvement == "HIGH":
        score += 0.24
        notes.append("high-involvement opportunity")
    elif policy.involvement == "MEDIUM":
        score += 0.12

    if policy.variance == "LOW":
        score += 0.20
        notes.append("volume/survivability favored over variance")
    elif policy.variance == "MEDIUM":
        score += 0.08
    else:
        score -= 0.10
        notes.append("high-variance market kept behind the strong core")

    if policy.role_check == "CLEAR":
        score += 0.12
        notes.append("role/availability check clear")
    elif policy.role_check == "RECHECK":
        score -= 0.06
        notes.append("pre-event role recheck still required")
    else:
        score -= 0.20

    reasons = " ".join(str(x).lower() for x in leg.model_reasons)
    context_hits = sum(1 for term in _CONTEXT_TERMS if term in reasons)
    if context_hits:
        context_bonus = min(0.16, 0.04 * context_hits)
        score += context_bonus
        notes.append(f"context-rich model evidence ({context_hits} signal family hit(s))")

    confidence = float(leg.model_confidence or 0.0)
    if confidence >= 0.72:
        score += 0.12
        notes.append("deep model confidence")
    elif confidence >= 0.60:
        score += 0.07

    edge_pp = None
    if kalshi_probability is not None:
        edge_pp = 100.0 * (float(fair_probability) - float(kalshi_probability))
        if 3.0 <= edge_pp <= 20.0:
            score += 0.10
            notes.append("meaningful but non-extreme model/market gap")
        elif 1.0 <= edge_pp < 3.0:
            score += 0.04
        elif edge_pp > 25.0:
            score -= 0.08
            notes.append("extreme gap needs independent confirmation")

    if policy.variance == "HIGH" and mode == "longshot" and (edge_pp or 0.0) >= 8.0:
        score += 0.06
        notes.append("longshot variance justified by a large independent edge")

    return clamp(score), tuple(dict.fromkeys(notes))


def research_source_rows(
    sources: Iterable[SocialResearchSource] = SOCIAL_RESEARCH_SOURCES,
) -> list[dict[str, str]]:
    return [
        {
            "Source": row.handle,
            "Platform": row.platform,
            "Tier": row.evidence_tier,
            "Focus": row.focus,
            "Applied lesson": row.lesson,
            "Caution": row.caution,
        }
        for row in sources
    ]
