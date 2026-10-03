from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable

from sports_edge.core.math import clamp
from sports_edge.models.model_evidence import ModelEvidence, model_dominant_fair
from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.ticket_policy import (
    construction_rank_key,
    involvement_rank,
    profile_leg,
    role_rank,
    variance_rank,
)
from sports_edge.research.social_intelligence import community_methodology_fit


@dataclass(frozen=True)
class LegAssessment:
    leg: ParlayCandidateLeg
    qualified: bool
    score: float
    fair_probability: float
    kalshi_probability: float | None
    edge_points: float | None
    ev_per_contract: float | None
    expected_roi_on_cost: float | None
    value_multiple: float | None
    evidence_quality: float
    involvement_rating: str
    variance_rating: str
    role_check: str
    community_methodology_score: float
    community_methodology_notes: tuple[str, ...]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class IntelligentParlay:
    mode: str
    legs: tuple[LegAssessment, ...]
    fair_joint_probability: float
    market_joint_probability: float
    ticket_value_multiple: float
    market_payout_multiplier: float
    correlation_risk: str
    risk_label: str
    strongest_leg: str | None
    weakest_leg: str | None
    highest_variance_leg: str | None
    primary_failure_scenario: str | None
    requested_event_count: int
    represented_event_count: int
    missing_event_ids: tuple[str, ...]
    warnings: tuple[str, ...]


def _model_evidence(leg: ParlayCandidateLeg) -> ModelEvidence | None:
    if leg.model_probability is None or not leg.model_name:
        return None
    return ModelEvidence(
        sport=leg.sport,
        model_name=leg.model_name,
        fair_probability=float(leg.model_probability),
        confidence=float(leg.model_confidence),
        sample_size=int(leg.model_sample_size),
        factors=tuple(leg.model_reasons),
        warnings=tuple(leg.model_warnings),
    )


def _best_value_gate(model: ModelEvidence | None) -> tuple[float, float, float, str]:
    """Use deeper model evidence to price smaller, still-positive edges.

    Best Available is not the same product as the longshot builder. A rigid
    +3 pp cutoff was discarding well-supported model edges and collapsing
    otherwise broad slates to one leg. We keep +3 pp as the default, but allow
    tighter positive-value gates only when both model confidence and sample
    depth are materially stronger. Negative/zero model edge never qualifies.
    """
    # Best Available is the strong-core builder, not the longshot builder.
    # Keep fair-probability floors meaningfully above coin-flip territory so
    # a five-leg ticket is built from genuinely strong individual legs. Deeper
    # evidence can relax the floor slightly, but never into longshot-style
    # candidate territory.
    if model is None:
        return 3.0, 1.05, 0.58, "standard"
    if model.confidence >= 0.72 and model.sample_size >= 15:
        return 1.0, 1.015, 0.55, "deep-evidence"
    if model.confidence >= 0.60 and model.sample_size >= 8:
        return 2.0, 1.03, 0.56, "strong-evidence"
    return 3.0, 1.05, 0.58, "standard"


def assess_leg(leg: ParlayCandidateLeg, mode: str) -> LegAssessment:
    model = _model_evidence(leg)
    policy = profile_leg(leg)
    price = None if leg.kalshi_price is None else clamp(float(leg.kalshi_price))
    failures: list[str] = []
    warnings: list[str] = []

    if model is None or not model.usable:
        fair = clamp(float(leg.consensus_probability))
        quality = 0.0
        reasons = ["No usable sport-specific model probability"]
        failures.append("sport-specific model evidence is required")
    else:
        book_probability = leg.consensus_probability if leg.book_count > 0 else None
        book_age = leg.source_age_s if leg.book_count > 0 else None
        fair, quality, reasons_tuple = model_dominant_fair(
            model,
            sportsbook_probability=book_probability,
            sportsbook_book_count=leg.book_count,
            sportsbook_age_s=book_age,
        )
        reasons = list(reasons_tuple)
        warnings.extend(model.warnings)

    if price is None or not leg.kalshi_ticker:
        failures.append("no exact current Kalshi price match")
    if policy.role_check == "BLOCK":
        failures.append("material role/availability uncertainty blocks recommendation")
    reasons.extend(policy.reasons)

    # Sportsbook evidence is secondary. Stale book data is ignored, not allowed
    # to veto a valid model signal.
    if leg.book_count > 0 and leg.source_age_s > 120:
        warnings.append("sportsbook cross-check stale; excluded from fair-value blend")

    edge = None if price is None else 100.0 * (fair - price)
    ev = None if price is None else fair - price
    roi = None if price is None or price <= 0 else (fair - price) / price
    multiple = None if price is None or price <= 0 else fair / price

    if edge is not None:
        reasons.append(f"Model-first fair {fair:.1%} vs Kalshi {price:.1%} = {edge:+.1f} pp")
    if leg.book_count > 0 and leg.source_age_s <= 120:
        reasons.append(
            f"Secondary sportsbook check: {leg.book_count} book(s), median age {leg.source_age_s:.0f}s"
        )
    elif leg.book_count == 0:
        reasons.append("No sportsbook listing required for qualification")

    if model is not None:
        min_model_conf = 0.40 if mode == "longshot" else 0.45
        if model.confidence < min_model_conf:
            failures.append(f"model confidence below {min_model_conf:.0%}")

        # Extreme model-vs-market gaps are exactly where data identity,
        # availability/news, or model misspecification can masquerade as edge.
        # Sportsbooks remain secondary evidence, but these outliers require a
        # fresh independent confirmation instead of auto-promotion.
        raw_model_gap = None if price is None else 100.0 * (model.fair_probability - price)
        fresh_secondary = leg.book_count >= 2 and leg.source_age_s <= 120
        if raw_model_gap is not None and raw_model_gap >= 25.0:
            if not fresh_secondary:
                failures.append(
                    "extreme model-market dislocation requires fresh secondary confirmation"
                )
            else:
                book_probability = clamp(float(leg.consensus_probability))
                if abs(model.fair_probability - book_probability) >= 0.20:
                    failures.append(
                        "model and fresh sportsbook confirmation conflict by at least 20 pp"
                    )

    if mode == "longshot":
        if price is not None and not (0.05 <= price <= 0.35):
            failures.append("Kalshi price is outside the 5–35¢ longshot band")
        if edge is None or edge < 4.0:
            failures.append("longshot model edge below +4.0 pp")
        if roi is None or roi < 0.15:
            failures.append("expected ROI on contract cost below +15%")
        if multiple is None or multiple < 1.15:
            failures.append("fair/price value multiple below 1.15x")

        edge_score = clamp((edge or 0.0) / 12.0)
        roi_score = clamp((roi or 0.0) / 0.60)
        fair_support = clamp(fair / 0.45)
        score = 100.0 * (
            0.36 * edge_score
            + 0.27 * roi_score
            + 0.27 * quality
            + 0.10 * fair_support
        )
    else:
        min_edge, min_multiple, min_fair, gate_tier = _best_value_gate(model)
        if edge is None or edge < min_edge:
            failures.append(f"model price edge below +{min_edge:.1f} pp for {gate_tier} Best Available gate")
        if multiple is None or multiple < min_multiple:
            failures.append(f"fair/price value multiple below {min_multiple:.3g}x for {gate_tier} Best Available gate")
        if fair < min_fair:
            failures.append(f"model-first fair probability below {min_fair:.0%} for {gate_tier} Best Available gate")
        if gate_tier != "standard":
            reasons.append(
                f"{gate_tier.replace('-', ' ').title()} Best Available gate: "
                f"confidence {model.confidence:.0%}, sample {model.sample_size}, "
                f"requires +{min_edge:.1f} pp edge and {min_multiple:.3g}x value"
            )

        edge_score = clamp((edge or 0.0) / 10.0)
        fair_score = clamp((fair - 0.40) / 0.35)
        roi_score = clamp((roi or 0.0) / 0.30)
        score = 100.0 * (
            0.37 * edge_score
            + 0.30 * quality
            + 0.20 * fair_score
            + 0.13 * roi_score
        )

    if roi is not None:
        reasons.append(f"Expected ROI on cost {roi:+.0%}; value multiple {multiple:.2f}x")

    methodology_score, methodology_notes = community_methodology_fit(
        leg,
        fair_probability=fair,
        kalshi_probability=price,
        mode=mode,
    )
    if methodology_notes:
        reasons.append(
            "Public-sharp methodology fit: "
            + "; ".join(methodology_notes)
        )
    warnings.extend(failures)

    return LegAssessment(
        leg=leg,
        qualified=not failures,
        score=round(score, 1),
        fair_probability=fair,
        kalshi_probability=price,
        edge_points=edge,
        ev_per_contract=ev,
        expected_roi_on_cost=roi,
        value_multiple=multiple,
        evidence_quality=quality,
        involvement_rating=policy.involvement,
        variance_rating=policy.variance,
        role_check=policy.role_check,
        community_methodology_score=methodology_score,
        community_methodology_notes=methodology_notes,
        reasons=tuple(reasons),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def build_intelligent_parlay(
    candidates: Iterable[ParlayCandidateLeg],
    *,
    mode: str,
    target_legs: int,
    max_per_event: int = 3,
    diversify_sports: bool = False,
    prioritize_payout_multiplier: bool = False,
    preferred_event_ids: Iterable[str] | None = None,
) -> IntelligentParlay:
    assessments = [assess_leg(row, mode) for row in candidates]
    pool = [row for row in assessments if row.qualified]
    if prioritize_payout_multiplier:
        # Build the strongest survivable core first. Payout is only a tiebreaker
        # after involvement, role stability, variance, model score and value.
        pool.sort(
            key=lambda row: construction_rank_key(
                row,
                prioritize_payout_multiplier=True,
            ),
            reverse=True,
        )
    else:
        pool.sort(
            key=construction_rank_key,
            reverse=True,
        )

    if diversify_sports:
        first: list[LegAssessment] = []
        rest: list[LegAssessment] = []
        seen: set[str] = set()
        for row in pool:
            if row.leg.sport not in seen:
                first.append(row)
                seen.add(row.leg.sport)
            else:
                rest.append(row)
        pool = first + rest

    # MLB can legitimately contribute multiple distinct contracts from one game.
    # Do not collapse the slate to one market per physical event. Contract-level
    # qualification remains model-first; ticket construction below caps event
    # concentration and reports same-game dependence explicitly.

    selected: list[LegAssessment] = []
    per_event: dict[str, int] = {}
    seen_contracts: set[tuple[str, str | None]] = set()

    def _try_add(row: LegAssessment) -> bool:
        contract = (row.leg.kalshi_ticker or row.leg.selection, row.leg.kalshi_side)
        if contract in seen_contracts:
            return False
        if per_event.get(row.leg.event_id, 0) >= max_per_event:
            return False
        selected.append(row)
        seen_contracts.add(contract)
        per_event[row.leg.event_id] = per_event.get(row.leg.event_id, 0) + 1
        return True

    preferred = tuple(dict.fromkeys(
        str(event_id).strip()
        for event_id in (preferred_event_ids or ())
        if str(event_id).strip()
    ))

    # Selected Games means "consider all of these games", not "globally rank
    # every leg and silently omit one selected game." Build a one-leg-per-game
    # qualified core first whenever the target size can support it, then fill
    # remaining slots from the globally ranked pool. Weak/unqualified legs are
    # never forced just to satisfy coverage.
    if preferred:
        best_by_event: dict[str, LegAssessment] = {}
        for row in pool:
            if row.leg.event_id in preferred and row.leg.event_id not in best_by_event:
                best_by_event[row.leg.event_id] = row

        coverage_rows = [
            best_by_event[event_id]
            for event_id in preferred
            if event_id in best_by_event
        ]
        if len(coverage_rows) > target_legs:
            pool_rank = {id(row): idx for idx, row in enumerate(pool)}
            coverage_rows.sort(key=lambda row: pool_rank.get(id(row), 10**9))
            coverage_rows = coverage_rows[:target_legs]

        for row in coverage_rows:
            if len(selected) >= target_legs:
                break
            _try_add(row)

    for row in pool:
        if len(selected) >= target_legs:
            break
        _try_add(row)

    fair_joint = prod(row.fair_probability for row in selected) if selected else 0.0
    market_joint = (
        prod(row.kalshi_probability for row in selected if row.kalshi_probability is not None)
        if selected else 0.0
    )
    value_multiple = fair_joint / market_joint if market_joint > 0 else 0.0
    payout_multiple = (1.0 / market_joint) if market_joint > 0 else 0.0

    warnings: list[str] = []
    represented_preferred = {
        row.leg.event_id for row in selected if row.leg.event_id in preferred
    }
    missing_preferred = tuple(
        event_id for event_id in preferred if event_id not in represented_preferred
    )
    if preferred and missing_preferred:
        warnings.append(
            f"{len(represented_preferred)}/{len(preferred)} selected game(s) are represented; "
            f"{len(missing_preferred)} selected game(s) had no model-qualified positive-EV leg "
            "or could not fit within the requested leg target"
        )
    if len(selected) < target_legs:
        warnings.append(
            f"Only {len(selected)} model-qualified positive-EV leg(s) cleared the {mode} gates "
            f"for a {target_legs}-leg target"
        )

    sports = [row.leg.sport for row in selected]
    correlation = "LOW"
    event_counts: dict[str, int] = {}
    for row in selected:
        event_counts[row.leg.event_id] = event_counts.get(row.leg.event_id, 0) + 1
    same_game_max = max(event_counts.values(), default=0)
    if same_game_max >= 2:
        correlation = "MEDIUM"
        warnings.append(
            "Same-game legs are dependent; displayed joint probability is an independence benchmark, not a calibrated same-game probability"
        )
    if same_game_max >= 3:
        correlation = "HIGH"
        warnings.append(
            "Three or more legs share a physical event; treat ticket-level probability as uncalibrated until a sport-specific joint model is available"
        )
    elif len(selected) >= 5 or (len(set(sports)) == 1 and len(selected) >= 3):
        correlation = "MEDIUM"
        warnings.append(
            "Joint probability is an independence benchmark; same-sport/long-ticket dependence is not fully calibrated"
        )

    if payout_multiple >= 50.0:
        risk_label = "VERY LOW-PROBABILITY LONG SHOT"
    elif payout_multiple >= 20.0:
        risk_label = "LONG SHOT"
    elif payout_multiple >= 8.0 or len(selected) >= 5:
        risk_label = "MODERATE/HIGH RISK"
    else:
        risk_label = "CONSERVATIVE RELATIVE TO THIS SLATE"

    strongest = max(
        selected,
        key=lambda row: (
            involvement_rank(row.involvement_rating),
            role_rank(row.role_check),
            variance_rank(row.variance_rating),
            row.score,
            row.fair_probability,
        ),
        default=None,
    )
    weakest = min(
        selected,
        key=lambda row: (
            involvement_rank(row.involvement_rating),
            role_rank(row.role_check),
            variance_rank(row.variance_rating),
            row.score,
            row.fair_probability,
        ),
        default=None,
    )
    highest_variance = min(
        selected,
        key=lambda row: (
            variance_rank(row.variance_rating),
            row.fair_probability,
            row.score,
        ),
        default=None,
    )

    if same_game_max >= 3:
        failure_scenario = "A narrow same-game script breaks several correlated legs at once."
    elif weakest is not None and weakest.role_check == "RECHECK":
        failure_scenario = f"Late role/availability news invalidates {weakest.leg.selection}."
    elif weakest is not None:
        failure_scenario = f"The lowest-strength leg ({weakest.leg.selection}) misses despite the modeled edge."
    else:
        failure_scenario = None

    if any(row.role_check == "RECHECK" for row in selected):
        warnings.append(
            "At least one selected leg still requires a pre-event role/availability recheck before treating the ticket as ready."
        )

    return IntelligentParlay(
        mode=mode,
        legs=tuple(selected),
        fair_joint_probability=fair_joint,
        market_joint_probability=market_joint,
        ticket_value_multiple=value_multiple,
        market_payout_multiplier=payout_multiple,
        correlation_risk=correlation,
        risk_label=risk_label,
        strongest_leg=strongest.leg.selection if strongest else None,
        weakest_leg=weakest.leg.selection if weakest else None,
        highest_variance_leg=highest_variance.leg.selection if highest_variance else None,
        primary_failure_scenario=failure_scenario,
        requested_event_count=len(preferred),
        represented_event_count=len(represented_preferred),
        missing_event_ids=missing_preferred,
        warnings=tuple(warnings),
    )
