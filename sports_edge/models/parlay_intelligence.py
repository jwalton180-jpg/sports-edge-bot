from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable

from sports_edge.core.math import clamp
from sports_edge.models.model_evidence import ModelEvidence, model_dominant_fair
from sports_edge.models.parlay_candidates import ParlayCandidateLeg


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
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class IntelligentParlay:
    mode: str
    legs: tuple[LegAssessment, ...]
    fair_joint_probability: float
    market_joint_probability: float
    ticket_value_multiple: float
    correlation_risk: str
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


def assess_leg(leg: ParlayCandidateLeg, mode: str) -> LegAssessment:
    model = _model_evidence(leg)
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
        confirmation_gap_pp = 15.0 if model.confidence < 0.60 else 25.0
        if raw_model_gap is not None and raw_model_gap >= confirmation_gap_pp:
            if not fresh_secondary:
                failures.append(
                    f"model-market dislocation of at least {confirmation_gap_pp:.0f} pp requires fresh secondary confirmation at this model-confidence level"
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
        if edge is None or edge < 3.0:
            failures.append("model price edge below +3.0 pp")
        if multiple is None or multiple < 1.05:
            failures.append("fair/price value multiple below 1.05x")
        if fair < 0.45:
            failures.append("model-first fair probability below 45% for Best Available")

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
        reasons=tuple(reasons),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def build_intelligent_parlay(
    candidates: Iterable[ParlayCandidateLeg],
    *,
    mode: str,
    target_legs: int,
    max_per_event: int = 1,
    diversify_sports: bool = False,
) -> IntelligentParlay:
    assessments = [assess_leg(row, mode) for row in candidates]
    pool = [row for row in assessments if row.qualified]
    pool.sort(
        key=lambda row: (
            row.score,
            row.edge_points if row.edge_points is not None else -999.0,
            row.evidence_quality,
            row.fair_probability,
        ),
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

    selected: list[LegAssessment] = []
    per_event: dict[str, int] = {}
    seen_contracts: set[tuple[str, str | None]] = set()

    for row in pool:
        contract = (row.leg.kalshi_ticker or row.leg.selection, row.leg.kalshi_side)
        if contract in seen_contracts:
            continue
        if per_event.get(row.leg.event_id, 0) >= max_per_event:
            continue
        selected.append(row)
        seen_contracts.add(contract)
        per_event[row.leg.event_id] = per_event.get(row.leg.event_id, 0) + 1
        if len(selected) >= target_legs:
            break

    fair_joint = prod(row.fair_probability for row in selected) if selected else 0.0
    market_joint = (
        prod(row.kalshi_probability for row in selected if row.kalshi_probability is not None)
        if selected else 0.0
    )
    value_multiple = fair_joint / market_joint if market_joint > 0 else 0.0

    warnings: list[str] = []
    if len(selected) < target_legs:
        warnings.append(
            f"Only {len(selected)} model-qualified positive-EV leg(s) cleared the {mode} gates "
            f"for a {target_legs}-leg target"
        )

    sports = [row.leg.sport for row in selected]
    correlation = "LOW"
    if len(selected) >= 5 or (len(set(sports)) == 1 and len(selected) >= 3):
        correlation = "MEDIUM"
        warnings.append(
            "Joint probability is an independence benchmark; same-sport/long-ticket dependence is not fully calibrated"
        )

    return IntelligentParlay(
        mode=mode,
        legs=tuple(selected),
        fair_joint_probability=fair_joint,
        market_joint_probability=market_joint,
        ticket_value_multiple=value_multiple,
        correlation_risk=correlation,
        warnings=tuple(warnings),
    )
