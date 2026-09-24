from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sports_edge.core.math import clamp
from sports_edge.models.consensus import match_market_to_event
from sports_edge.models.edge import grade_edge, should_surface
from sports_edge.models.underdog import score_underdog


@dataclass(frozen=True)
class LiveSignal:
    ticker: str
    sport: str
    event_id: str
    event_title: str
    market: str
    side: str
    selection: str
    market_probability: float
    fair_probability: float
    edge_points: float
    ev_per_contract: float
    confidence: float
    data_quality: float
    book_count: int
    source_age_s: float
    status: str
    tier: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def market_side_probability(market: dict, side: str) -> float | None:
    side = side.upper()
    keys = (
        ("yes_ask_dollars", "yes_ask", "last_price_dollars", "last_price")
        if side == "YES"
        else ("no_ask_dollars", "no_ask")
    )
    for key in keys:
        raw = market.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 1.0:
            value /= 100.0
        if 0.0 < value < 1.0:
            return clamp(value)
    return None


def market_yes_probability(market: dict) -> float | None:
    return market_side_probability(market, "YES")


def build_live_signals(
    markets: list[dict],
    odds_events: list[dict],
    *,
    sport: str,
    now: datetime | None = None,
    max_source_age_s: float = 180.0,
    min_edge_points: float = 3.0,
    min_confidence: float = 0.55,
    min_data_quality: float = 0.70,
) -> list[LiveSignal]:
    now = now or datetime.now(timezone.utc)
    rows: list[LiveSignal] = []

    for market in markets:
        match = match_market_to_event(market, odds_events, now=now, max_age_s=max_source_age_s)
        if match is None:
            continue

        q = match.quote
        model_confidence = clamp(match.match_confidence * min(1.0, q.book_count / 4.0))
        shared_reasons = [
            f"No-vig consensus from {q.book_count} fresh sportsbook source(s)",
            f"Matched to {match.event_title or match.event_id}",
        ]
        shared_warnings = list(q.warnings)
        if q.median_age_s > max_source_age_s:
            shared_warnings.append("Sportsbook source stale")

        for side in ("YES", "NO"):
            market_p = market_side_probability(market, side)
            if market_p is None:
                continue
            fair_p = q.fair_probability if side == "YES" else 1.0 - q.fair_probability
            selection = match.selection if side == "YES" else (match.opposite_selection or f"NO — {match.selection}")
            reasons = [*shared_reasons, f"{side} side evaluated independently"]

            card = grade_edge(
                key=str(market.get("ticker") or ""),
                sport=sport,
                market=str(market.get("title") or market.get("subtitle") or market.get("ticker") or ""),
                selection=selection,
                fair_p=fair_p,
                market_p=market_p,
                model_confidence=model_confidence,
                data_quality=q.data_quality,
                reasons=reasons,
                warnings=shared_warnings,
                min_edge_points=min_edge_points,
            )
            surfaced = should_surface(
                card,
                min_edge_points=min_edge_points,
                min_confidence=min_confidence,
                min_data_quality=min_data_quality,
            ) and not q.warnings and q.median_age_s <= max_source_age_s

            if surfaced:
                status = "QUALIFIED"
            elif card.edge_points > 0:
                status = "WATCH"
            else:
                status = "PASS"

            rows.append(
                LiveSignal(
                    ticker=card.key,
                    sport=sport,
                    event_id=match.event_id,
                    event_title=match.event_title,
                    market=card.market,
                    side=side,
                    selection=card.selection,
                    market_probability=card.market_probability,
                    fair_probability=card.fair_probability,
                    edge_points=card.edge_points,
                    ev_per_contract=card.ev_per_dollar,
                    confidence=card.confidence,
                    data_quality=card.data_quality,
                    book_count=q.book_count,
                    source_age_s=q.median_age_s,
                    status=status,
                    tier="EDGE",
                    reasons=tuple(card.reasons),
                    warnings=tuple(card.warnings),
                )
            )

    return sorted(rows, key=lambda x: (x.status == "QUALIFIED", x.edge_points, x.confidence), reverse=True)

def build_underdog_signals(
    markets: list[dict],
    odds_events: list[dict],
    *,
    sport: str = "Tennis",
    now: datetime | None = None,
    max_source_age_s: float = 180.0,
) -> list[LiveSignal]:
    base = build_live_signals(
        markets,
        odds_events,
        sport=sport,
        now=now,
        max_source_age_s=max_source_age_s,
        min_edge_points=4.0,
        min_confidence=0.55,
        min_data_quality=0.75,
    )
    result: list[LiveSignal] = []
    for row in base:
        if not 0.03 <= row.market_probability <= 0.30:
            continue
        scored = score_underdog(
            row.market_probability,
            row.fair_probability,
            data_quality=row.data_quality,
            source_age_s=row.source_age_s,
            max_age_s=max_source_age_s,
            min_edge_points=4.0,
            reasons=list(row.reasons),
            warnings=list(row.warnings),
        )
        status = "QUALIFIED" if scored.tier != "PASS" and row.confidence >= 0.55 else ("WATCH" if scored.edge_points > 0 else "PASS")
        result.append(
            LiveSignal(
                ticker=row.ticker,
                sport=row.sport,
                event_id=row.event_id,
                event_title=row.event_title,
                market=row.market,
                side=row.side,
                selection=row.selection,
                market_probability=row.market_probability,
                fair_probability=row.fair_probability,
                edge_points=scored.edge_points,
                ev_per_contract=scored.ev_per_contract,
                confidence=row.confidence,
                data_quality=row.data_quality,
                book_count=row.book_count,
                source_age_s=row.source_age_s,
                status=status,
                tier=scored.tier,
                reasons=scored.reasons,
                warnings=scored.warnings,
            )
        )
    return sorted(result, key=lambda x: (x.status == "QUALIFIED", x.edge_points, x.confidence), reverse=True)


def unverified_underdog_watchlist(markets: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market in markets:
        p = market_yes_probability(market)
        if p is None or not 0.03 <= p <= 0.30:
            continue
        rows.append(
            {
                "ticker": str(market.get("ticker") or ""),
                "market": str(market.get("title") or market.get("subtitle") or market.get("ticker") or ""),
                "market_probability": p,
                "volume": float(market.get("volume_fp", market.get("volume", 0)) or 0),
            }
        )
    return sorted(rows, key=lambda x: (x["market_probability"], -x["volume"]))
