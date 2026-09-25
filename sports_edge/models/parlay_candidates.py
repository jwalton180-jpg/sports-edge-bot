from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import prod
from typing import Iterable

from sports_edge.models.consensus import consensus_from_event
from sports_edge.models.game_scope import GameEvent
from sports_edge.models.live_board import LiveSignal
from sports_edge.models.props import PropConsensus


@dataclass(frozen=True)
class ParlayCandidateLeg:
    sport: str
    event_id: str
    event_title: str
    market_key: str
    market_label: str
    selection: str
    consensus_probability: float
    book_count: int
    source_age_s: float
    median_odds: float | None
    kalshi_ticker: str | None
    kalshi_side: str | None
    kalshi_price: float | None
    kalshi_edge_points: float | None
    kalshi_status: str
    evidence_class: str


@dataclass(frozen=True)
class GeneratedParlay:
    legs: tuple[ParlayCandidateLeg, ...]
    estimated_independent_probability: float
    correlation_risk: str
    warnings: tuple[str, ...]


def _selection_text(q: PropConsensus) -> str:
    line = "" if q.line is None else f" {q.line:g}"
    return f"{q.player} {q.side}{line} {q.market_label}".strip()


def _find_kalshi_match(
    game: GameEvent,
    quote: PropConsensus,
    exact_signals: Iterable[LiveSignal],
) -> LiveSignal | None:
    player = quote.player.lower()
    label = quote.market_label.lower()
    candidates = [
        s
        for s in exact_signals
        if s.event_id == game.event_id
        and player in s.selection.lower()
        and label in s.selection.lower()
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda s: (s.status == "QUALIFIED", s.edge_points, s.confidence),
        reverse=True,
    )[0]




def _evidence_class(match: LiveSignal | None) -> str:
    if match is None:
        return "CONSENSUS BASELINE"
    if match.status == "QUALIFIED" and match.edge_points >= 3.0:
        return "EDGE-QUALIFIED"
    return "KALSHI MATCH"


def candidate_legs_from_h2h(
    game: GameEvent,
    event_payload: dict,
    exact_signals: list[LiveSignal],
    *,
    mode: str = "high_confidence",
    now: datetime | None = None,
) -> list[ParlayCandidateLeg]:
    quotes = consensus_from_event(event_payload, now=now)
    if not quotes:
        return []

    min_probability = 0.05 if mode == "edge" else (0.56 if mode == "high_confidence" else 0.20)
    rows: list[ParlayCandidateLeg] = []
    for selection, quote in quotes.items():
        if quote.warnings or quote.fair_probability < min_probability:
            continue

        matches = [
            s for s in exact_signals
            if s.event_id == game.event_id and s.selection.lower() == selection.lower()
        ]
        match = sorted(
            matches,
            key=lambda s: (s.status == "QUALIFIED", s.edge_points, s.confidence),
            reverse=True,
        )[0] if matches else None

        rows.append(
            ParlayCandidateLeg(
                sport=game.sport,
                event_id=game.event_id,
                event_title=f"{game.away_team} @ {game.home_team}",
                market_key="h2h",
                market_label="Moneyline",
                selection=selection,
                consensus_probability=quote.fair_probability,
                book_count=quote.book_count,
                source_age_s=quote.median_age_s,
                median_odds=None,
                kalshi_ticker=match.ticker if match else None,
                kalshi_side=match.side if match else None,
                kalshi_price=match.market_probability if match else None,
                kalshi_edge_points=match.edge_points if match else None,
                kalshi_status=match.status if match else "CONSENSUS ONLY",
                evidence_class=_evidence_class(match),
            )
        )

    # One moneyline side per game: take the stronger consensus side.
    return sorted(
        rows,
        key=lambda r: (
            r.evidence_class == "EDGE-QUALIFIED",
            r.kalshi_ticker is not None,
            r.consensus_probability,
            r.book_count,
        ),
        reverse=True,
    )[:1]

def candidate_legs_from_props(
    game: GameEvent,
    quotes: list[PropConsensus],
    exact_signals: list[LiveSignal],
    *,
    mode: str = "high_confidence",
) -> list[ParlayCandidateLeg]:
    min_probability = 0.05 if mode == "edge" else (0.56 if mode == "high_confidence" else 0.20)
    rows: list[ParlayCandidateLeg] = []

    for quote in quotes:
        # Use one actionable side per offered player/line. Unders/No are valid,
        # but we avoid duplicate opposite sides by taking the stronger side later.
        if quote.warnings:
            continue
        if quote.book_count < 2 or quote.fair_probability < min_probability:
            continue

        match = _find_kalshi_match(game, quote, exact_signals)
        rows.append(
            ParlayCandidateLeg(
                sport=game.sport,
                event_id=game.event_id,
                event_title=f"{game.away_team} @ {game.home_team}",
                market_key=quote.market_key,
                market_label=quote.market_label,
                selection=_selection_text(quote),
                consensus_probability=quote.fair_probability,
                book_count=quote.book_count,
                source_age_s=quote.median_age_s,
                median_odds=quote.median_price,
                kalshi_ticker=match.ticker if match else None,
                kalshi_side=match.side if match else None,
                kalshi_price=match.market_probability if match else None,
                kalshi_edge_points=match.edge_points if match else None,
                kalshi_status=match.status if match else "CONSENSUS ONLY",
                evidence_class=_evidence_class(match),
            )
        )

    # Keep only the stronger side for each event/player/market/line identity.
    best: dict[tuple[str, str, str], ParlayCandidateLeg] = {}
    for row in rows:
        # selection begins with the full player name and includes side/line; use
        # event + market + player-ish prefix to keep opposite sides from doubling.
        player_key = row.selection.split(" Over", 1)[0].split(" Under", 1)[0].split(" Yes", 1)[0].split(" No", 1)[0]
        key = (row.event_id, row.market_key, player_key.lower())
        old = best.get(key)
        if old is None or row.consensus_probability > old.consensus_probability:
            best[key] = row

    return sorted(
        best.values(),
        key=lambda r: (
            r.evidence_class == "EDGE-QUALIFIED",
            r.kalshi_ticker is not None,
            r.consensus_probability,
            r.book_count,
            -r.source_age_s,
        ),
        reverse=True,
    )


def research_fallback_candidates(
    candidates: list[ParlayCandidateLeg],
    *,
    min_books: int = 3,
    max_source_age_s: float = 120.0,
    min_consensus_probability: float = 0.52,
) -> list[ParlayCandidateLeg]:
    """Return fresh multi-book research candidates when strict edge is empty.

    This is intentionally not an edge classifier. It only creates a clearly
    labelled fallback slate from current consensus evidence.
    """
    rows = [
        row for row in candidates
        if row.book_count >= min_books
        and row.source_age_s <= max_source_age_s
        and row.consensus_probability >= min_consensus_probability
    ]
    return sorted(
        rows,
        key=lambda r: (
            r.evidence_class == "EDGE-QUALIFIED",
            r.book_count,
            -r.source_age_s,
            r.consensus_probability,
        ),
        reverse=True,
    )


def generate_candidate_parlay(
    candidates: list[ParlayCandidateLeg],
    *,
    target_legs: int,
    mode: str = "high_confidence",
    max_per_event: int = 2,
    require_edge: bool = False,
    diversify_sports: bool = False,
) -> GeneratedParlay:
    pool = list(candidates)
    if require_edge:
        pool = [
            row for row in pool
            if row.evidence_class == "EDGE-QUALIFIED"
            and row.kalshi_edge_points is not None
            and row.kalshi_edge_points >= 3.0
        ]

    pool.sort(
        key=lambda r: (
            r.evidence_class == "EDGE-QUALIFIED",
            r.kalshi_edge_points if r.kalshi_edge_points is not None else -999.0,
            r.book_count,
            -r.source_age_s,
            r.consensus_probability,
        ),
        reverse=True,
    )

    if diversify_sports:
        diversified: list[ParlayCandidateLeg] = []
        remainder: list[ParlayCandidateLeg] = []
        seen_sports: set[str] = set()
        for row in pool:
            if row.sport not in seen_sports:
                diversified.append(row)
                seen_sports.add(row.sport)
            else:
                remainder.append(row)
        pool = diversified + remainder

    selected: list[ParlayCandidateLeg] = []
    per_event: dict[str, int] = {}
    seen_player_market: set[tuple[str, str, str]] = set()

    for row in pool:
        player = row.selection.split(" Over", 1)[0].split(" Under", 1)[0].split(" Yes", 1)[0].split(" No", 1)[0]
        identity = (row.event_id, player.lower(), row.market_label.lower())
        if identity in seen_player_market:
            continue
        if per_event.get(row.event_id, 0) >= max_per_event:
            continue
        selected.append(row)
        seen_player_market.add(identity)
        per_event[row.event_id] = per_event.get(row.event_id, 0) + 1
        if len(selected) >= target_legs:
            break

    warnings: list[str] = []
    if len(selected) < target_legs:
        qualifier = "edge-qualified " if require_edge else ""
        warnings.append(f"Only {len(selected)} usable current {qualifier}leg(s) found for a {target_legs}-leg target")

    repeated_events = sum(v > 1 for v in per_event.values())
    correlation_risk = "LOW"
    if repeated_events:
        correlation_risk = "MEDIUM"
        warnings.append("Some legs share a game; joint probability assumes independence and may overstate the true probability")
    if len(selected) >= 5:
        correlation_risk = "MEDIUM"
        warnings.append("Long parlays magnify pricing/model error")

    independent = prod(x.consensus_probability for x in selected) if selected else 0.0
    return GeneratedParlay(
        legs=tuple(selected),
        estimated_independent_probability=independent,
        correlation_risk=correlation_risk,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def combo_blueprint(legs: Iterable[ParlayCandidateLeg]) -> str:
    lines = ["SPORTS EDGE — KALSHI COMBO BLUEPRINT", "Recheck eligibility and live RFQ price in Kalshi before entry.", ""]
    any_leg = False
    for i, leg in enumerate(legs, 1):
        any_leg = True
        ready = (
            f"{leg.kalshi_ticker} | {leg.kalshi_side} | reference {round((leg.kalshi_price or 0) * 100)}¢"
            if leg.kalshi_ticker
            else "Find matching component in Kalshi Combo Builder"
        )
        lines.append(f"{i}. {leg.event_title} | {leg.selection} | consensus {leg.consensus_probability:.1%} | {leg.evidence_class} | {ready}")
    if not any_leg:
        lines.append("No current legs.")
    return "\n".join(lines)
