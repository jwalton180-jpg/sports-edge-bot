from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import exp, log

from sports_edge.core.math import clamp
from sports_edge.data.tennis365 import (
    Tennis365PlayerContext,
    fetch_tennis365_player_context,
)
from sports_edge.data.tennis_live import TennisLiveScoreState
from sports_edge.models.event_identity import canonical_participant
from sports_edge.models.live_board import market_side_probability
from sports_edge.models.parlay_candidates import ParlayCandidateLeg


LOWER_TOUR_MODEL = "Tennis live ranking + prior form fallback"


@dataclass(frozen=True)
class LowerTourPrior:
    probability_a: float
    confidence: float
    sample_size: int
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def _safe_logit(p: float) -> float:
    p = clamp(p, 1e-5, 1.0 - 1e-5)
    return log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def lower_tour_prior(
    a: Tennis365PlayerContext,
    b: Tennis365PlayerContext,
) -> LowerTourPrior | None:
    """Conservative independent prior for live Tennis model coverage gaps.

    No live score, Kalshi price, sportsbook price, or current-match result is
    accepted here. Ranking/points establish baseline strength; completed prior
    matches provide a small form adjustment. Sparse evidence fails closed.
    """
    min_recent = min(a.recent_matches, b.recent_matches)
    have_points = bool(
        a.ranking_points and a.ranking_points > 0
        and b.ranking_points and b.ranking_points > 0
    )
    have_ranks = bool(a.rank and a.rank > 0 and b.rank and b.rank > 0)
    if not (have_points or have_ranks):
        # Unranked players still need enough independent completed history on
        # both sides. A tiny sample is not a model and must remain WATCH-only.
        if min_recent < 6:
            return None
        base = 0.50
        basis = "No reliable ranking for both players; neutral baseline before prior-form adjustment"
    elif have_points:
        # Ranking points are positive strength. Sub-linear exponent and 45%
        # shrinkage toward 50% deliberately reduce tour/ranking noise.
        sa = float(a.ranking_points) ** 0.72
        sb = float(b.ranking_points) ** 0.72
        raw = sa / max(sa + sb, 1e-9)
        base = 0.50 + 0.55 * (raw - 0.50)
        basis = (
            f"Ranking points {a.player} {a.ranking_points} vs {b.player} {b.ranking_points}; "
            "strength ratio shrunk 45% toward even"
        )
    else:
        sa = 1.0 / float(a.rank)
        sb = 1.0 / float(b.rank)
        raw = sa / max(sa + sb, 1e-9)
        base = 0.50 + 0.45 * (raw - 0.50)
        basis = (
            f"Rank {a.player} #{a.rank} vs {b.player} #{b.rank}; "
            "rank-ratio baseline aggressively shrunk toward even"
        )

    reasons = [basis]
    fa = (a.recent_wins + 2.0) / (a.recent_matches + 4.0) if a.recent_matches else 0.50
    fb = (b.recent_wins + 2.0) / (b.recent_matches + 4.0) if b.recent_matches else 0.50
    if min_recent >= 4:
        form_delta = clamp(fa - fb, -0.40, 0.40)
        p = _sigmoid(_safe_logit(base) + 0.65 * form_delta)
        reasons.append(
            f"Prior completed-form: {a.player} {a.recent_wins}-{a.recent_losses} vs "
            f"{b.player} {b.recent_wins}-{b.recent_losses}; shrunk delta {form_delta:+.2f}"
        )
    else:
        p = base
        reasons.append("Recent-form sample too thin to move the ranking baseline")

    sample = min_recent
    if have_points and min_recent >= 6:
        confidence = 0.51
    elif have_ranks and min_recent >= 6:
        confidence = 0.48
    elif min_recent >= 8:
        confidence = 0.46
    else:
        confidence = 0.43

    warnings = [
        "live Tennis fallback used because the primary historical Tennis model had no usable player pair",
        "fallback prior excludes the current live match and all Kalshi/sportsbook prices",
    ]
    if not have_points:
        warnings.append("ranking-points evidence incomplete")
    if sample < 6:
        warnings.append("fewer than six prior completed matches per player; strong reversal promotion blocked")

    return LowerTourPrior(
        probability_a=clamp(p, 0.12, 0.88),
        confidence=confidence,
        sample_size=sample,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )


def _selection_from_market(market: dict) -> str:
    for key in ("yes_sub_title", "yes_title", "yes_label"):
        value = str(market.get(key) or "").strip()
        if value and value.lower() not in {"yes", "no"}:
            return value
    title = str(market.get("title") or "").strip()
    if title.lower().endswith(" wins"):
        return title[:-5].strip()
    return ""


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((
        canonical_participant("Tennis", a),
        canonical_participant("Tennis", b),
    )))


def build_lower_tour_live_fallback_candidates(
    markets: list[dict] | tuple[dict, ...],
    states: list[TennisLiveScoreState] | tuple[TennisLiveScoreState, ...],
    existing: list[ParlayCandidateLeg] | tuple[ParlayCandidateLeg, ...],
    *,
    max_workers: int = 6,
) -> list[ParlayCandidateLeg]:
    """Fill confirmed-live Tennis model holes without replacing the primary model.

    Tennis365 detail pages provide ranking + completed prior-form context. For
    ATP/WTA the live structural state must still come from ESPN; Tennis365 only
    contributes the independent fallback context URL after cross-feed matching.
    """
    eligible_states: dict[tuple[str, str], TennisLiveScoreState] = {}
    for state in states:
        if state.tour not in {"ATP", "WTA", "CHALLENGER", "ITF", "ITF-W"}:
            continue
        if not state.source_url or state.score_conflict:
            continue
        eligible_states[_pair_key(state.player, state.opponent)] = state

    if not eligible_states:
        return []

    existing_pairs: set[tuple[str, str]] = set()
    for row in existing:
        parts = str(row.event_id).split(":", 2)
        if len(parts) == 3 and "|" in parts[2]:
            a, b = parts[2].split("|", 1)
            existing_pairs.add(tuple(sorted((a, b))))

    grouped: dict[str, list[dict]] = {}
    supported_series = {
        "KXATPMATCH",
        "KXATPCHALLENGERMATCH",
        "KXWTAMATCH",
        "KXWTACHALLENGERMATCH",
        "KXITFMATCH",
        "KXITFWMATCH",
    }
    for market in markets:
        series = str(market.get("series_ticker") or "").upper()
        if series not in supported_series:
            continue
        event = str(market.get("event_ticker") or "").strip()
        selection = _selection_from_market(market)
        if event and selection:
            grouped.setdefault(event, []).append(market)

    jobs: list[tuple[tuple[str, str], TennisLiveScoreState, list[dict], str, str]] = []
    for event_markets in grouped.values():
        names = []
        for market in event_markets:
            selection = _selection_from_market(market)
            key = canonical_participant("Tennis", selection)
            if selection and key and key not in {canonical_participant('Tennis', x) for x in names}:
                names.append(selection)
        if len(names) != 2:
            continue
        pair = _pair_key(names[0], names[1])
        if pair in existing_pairs:
            continue
        state = eligible_states.get(pair)
        if state is None:
            continue
        jobs.append((pair, state, event_markets, names[0], names[1]))

    if not jobs:
        return []

    contexts: dict[tuple[str, str], tuple[Tennis365PlayerContext, Tennis365PlayerContext] | None] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(jobs)))) as pool:
        future_map = {
            pool.submit(
                fetch_tennis365_player_context,
                state.source_url or "",
                player_a=a,
                player_b=b,
            ): pair
            for pair, state, _, a, b in jobs
        }
        for future in as_completed(future_map):
            pair = future_map[future]
            try:
                contexts[pair] = future.result()
            except Exception:
                contexts[pair] = None

    out: list[ParlayCandidateLeg] = []
    for pair, state, event_markets, a, b in jobs:
        context = contexts.get(pair)
        if context is None:
            continue
        ctx_a, ctx_b = context
        prior = lower_tour_prior(ctx_a, ctx_b)
        if prior is None:
            continue
        probability_by_key = {
            canonical_participant("Tennis", a): prior.probability_a,
            canonical_participant("Tennis", b): 1.0 - prior.probability_a,
        }
        for market in event_markets:
            selection = _selection_from_market(market)
            key = canonical_participant("Tennis", selection)
            fair = probability_by_key.get(key)
            price = market_side_probability(market, "YES")
            if fair is None or price is None:
                continue
            out.append(ParlayCandidateLeg(
                sport="Tennis",
                event_id=state.event_id,
                event_title=f"{a} vs {b}",
                market_key="model_h2h",
                market_label="Match Winner",
                selection=selection,
                consensus_probability=fair,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side="YES",
                kalshi_price=price,
                kalshi_edge_points=100.0 * (fair - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=fair,
                model_confidence=prior.confidence,
                model_name=LOWER_TOUR_MODEL,
                model_sample_size=prior.sample_size,
                model_reasons=prior.reasons,
                model_warnings=prior.warnings,
            ))
    return out
