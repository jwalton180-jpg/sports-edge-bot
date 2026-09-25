from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from statistics import median
from typing import Any

from sports_edge.core.math import american_to_implied, clamp


# Heuristic market-reference weights, not claims about guaranteed predictive
# superiority. These sources get modest extra weight when present because they
# are commonly used as price-discovery references; all output still requires
# cross-book agreement and freshness.
BOOK_WEIGHTS: dict[str, float] = {
    "pinnacle": 1.35,
    "betfair": 1.25,
    "betfair_ex_uk": 1.25,
    "circa": 1.20,
    "bookmaker": 1.20,
    "draftkings": 1.05,
    "fanduel": 1.05,
    "betmgm": 1.00,
    "betrivers": 1.00,
    "williamhill_us": 1.00,
    "bovada": 0.95,
    "betonlineag": 0.95,
}

REFERENCE_BOOKS = {"pinnacle", "betfair", "betfair_ex_uk", "circa", "bookmaker"}
MAJOR_BOOKS = {"draftkings", "fanduel", "betmgm", "betrivers", "williamhill_us", "fanatics"}

# Keep this under ten bookmakers so The Odds API counts it like one region.
PREMIUM_BOOKMAKER_KEYS: tuple[str, ...] = (
    "pinnacle",
    "draftkings",
    "fanduel",
    "betmgm",
    "betrivers",
    "bovada",
    "betonlineag",
)


@dataclass(frozen=True)
class BookProbability:
    bookmaker_key: str
    bookmaker_title: str
    probability: float
    american_price: float
    age_s: float
    weight: float
    is_reference: bool
    is_major: bool


@dataclass(frozen=True)
class IntelligenceResult:
    selection: str
    fair_probability: float
    market_probability: float | None
    edge_points: float | None
    intelligence_score: float
    tier: str
    book_count: int
    reference_book_count: int
    median_age_s: float
    disagreement_pp: float
    model_probability: float | None
    model_gap_points: float | None
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    books: tuple[BookProbability, ...]


@dataclass(frozen=True)
class BookOfferEdge:
    event_id: str
    market_key: str
    selection: str
    point: float | None
    bookmaker_key: str
    bookmaker_title: str
    american_price: float
    break_even_probability: float
    leave_one_out_fair_probability: float
    edge_points: float
    age_s: float
    comparison_books: int
    evidence_quality: float


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _age_seconds(value: Any, now: datetime) -> float:
    dt = _parse_iso(value)
    if dt is None:
        return float("inf")
    return max(0.0, (now - dt).total_seconds())


def _book_weight(key: str) -> float:
    return BOOK_WEIGHTS.get(key, 0.90)


def _weighted_mean(items: list[tuple[float, float]]) -> float:
    total_weight = sum(w for _, w in items)
    if total_weight <= 0:
        raise ValueError("positive total weight required")
    return sum(v * w for v, w in items) / total_weight


def _tier(score: float, edge_points: float | None, book_count: int) -> str:
    if edge_points is None:
        return "RESEARCH"
    if score >= 82 and edge_points >= 5 and book_count >= 4:
        return "A"
    if score >= 70 and edge_points >= 3 and book_count >= 3:
        return "B"
    if score >= 58 and edge_points > 0:
        return "WATCH"
    return "PASS"


def h2h_intelligence(
    event: dict,
    selection: str,
    *,
    market_probability: float | None = None,
    model_probability: float | None = None,
    now: datetime | None = None,
    max_age_s: float = 180.0,
) -> IntelligenceResult | None:
    """Source-weighted h2h intelligence with explicit evidence accounting."""

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    books: list[BookProbability] = []

    for bookmaker in event.get("bookmakers", []) or []:
        key = str(bookmaker.get("key") or "").strip()
        title = str(bookmaker.get("title") or key).strip()
        market = next((m for m in bookmaker.get("markets", []) or [] if m.get("key") == "h2h"), None)
        if not market:
            continue
        outcomes = market.get("outcomes", []) or []
        if len(outcomes) != 2:
            continue
        update = market.get("last_update") or bookmaker.get("last_update")
        age_s = _age_seconds(update, now)
        if not math.isfinite(age_s) or age_s > max_age_s:
            continue

        parsed: list[tuple[str, float, float]] = []
        try:
            for out in outcomes:
                name = str(out.get("name") or "").strip()
                price = float(out.get("price"))
                parsed.append((name, price, american_to_implied(price)))
        except (TypeError, ValueError, ZeroDivisionError):
            continue

        total = sum(p for _, _, p in parsed)
        if total <= 0:
            continue
        match = next((x for x in parsed if x[0].lower() == selection.lower()), None)
        if match is None:
            continue
        _, price, raw = match
        fair = raw / total
        books.append(
            BookProbability(
                bookmaker_key=key,
                bookmaker_title=title,
                probability=clamp(fair),
                american_price=price,
                age_s=age_s,
                weight=_book_weight(key) / (1.0 + age_s / 60.0),
                is_reference=key in REFERENCE_BOOKS,
                is_major=key in MAJOR_BOOKS,
            )
        )

    if len(books) < 2:
        return None

    fair = clamp(_weighted_mean([(b.probability, b.weight) for b in books]))
    probabilities = [b.probability for b in books]
    ages = [b.age_s for b in books]
    disagreement_pp = 100.0 * (max(probabilities) - min(probabilities))
    med_age = float(median(ages))
    ref_count = sum(b.is_reference for b in books)

    edge_points = None if market_probability is None else 100.0 * (fair - float(market_probability))
    model_gap = None if model_probability is None else 100.0 * (float(model_probability) - fair)

    source_score = clamp(0.65 * min(1.0, len(books) / 5.0) + 0.35 * min(1.0, ref_count / 1.0))
    freshness_score = clamp(1.0 - med_age / max_age_s)
    agreement_score = clamp(1.0 - disagreement_pp / 14.0)
    edge_score = 0.0 if edge_points is None else clamp(max(0.0, edge_points) / 8.0)
    model_score = 0.50
    if model_probability is not None:
        # Model agreement helps confidence; a large model-vs-market split is
        # surfaced as a reason to inspect, not automatically rewarded.
        model_score = clamp(1.0 - abs(float(model_probability) - fair) / 0.12)

    score = 100.0 * (
        0.34 * edge_score
        + 0.24 * source_score
        + 0.16 * freshness_score
        + 0.16 * agreement_score
        + 0.10 * model_score
    )

    reasons = [
        f"{len(books)} fresh books in weighted no-vig consensus",
        f"{ref_count} reference-market source(s)" if ref_count else "Major-book consensus without reference-market source",
        f"Cross-book range {disagreement_pp:.1f} pp",
    ]
    if edge_points is not None:
        reasons.append(f"Kalshi/target price gap {edge_points:+.1f} pp")
    if model_probability is not None:
        reasons.append(f"Model overlay {float(model_probability):.1%} vs market fair {fair:.1%}")

    warnings: list[str] = []
    if len(books) < 3:
        warnings.append("Thin book coverage")
    if med_age > max_age_s * 0.66:
        warnings.append("Consensus aging")
    if disagreement_pp > 10:
        warnings.append("High cross-book disagreement")
    if edge_points is not None and edge_points <= 0:
        warnings.append("No positive price edge")

    return IntelligenceResult(
        selection=selection,
        fair_probability=fair,
        market_probability=market_probability,
        edge_points=edge_points,
        intelligence_score=round(score, 1),
        tier=_tier(score, edge_points, len(books)),
        book_count=len(books),
        reference_book_count=ref_count,
        median_age_s=med_age,
        disagreement_pp=disagreement_pp,
        model_probability=model_probability,
        model_gap_points=model_gap,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        books=tuple(sorted(books, key=lambda b: (b.is_reference, b.weight), reverse=True)),
    )


def prop_book_offer_edges(
    payload: dict,
    market_keys: tuple[str, ...] | list[str],
    *,
    now: datetime | None = None,
    max_age_s: float = 180.0,
    min_other_books: int = 2,
) -> list[BookOfferEdge]:
    """Find underpriced prop offers using leave-one-book-out fair value.

    Each book is compared only with the *other* fresh books at the exact same
    player/outcome/line. This avoids circularly using a sportsbook to justify
    an edge against itself.
    """

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    wanted = set(market_keys)

    # key -> bookmaker -> (fair_probability, raw_break_even, price, age, title)
    samples: dict[tuple[str, str, float | None, str], dict[str, tuple[float, float, float, float, str]]] = {}

    for bookmaker in payload.get("bookmakers", []) or []:
        bkey = str(bookmaker.get("key") or "").strip()
        btitle = str(bookmaker.get("title") or bkey).strip()
        if not bkey:
            continue
        for market in bookmaker.get("markets", []) or []:
            mkey = str(market.get("key") or "")
            if mkey not in wanted:
                continue
            age_s = _age_seconds(market.get("last_update") or bookmaker.get("last_update"), now)
            if not math.isfinite(age_s) or age_s > max_age_s:
                continue

            grouped: dict[tuple[str, float | None], list[dict]] = {}
            for out in market.get("outcomes", []) or []:
                desc = str(out.get("description") or "").strip()
                side = str(out.get("name") or "").strip()
                player = desc if desc else (side if side.lower() not in {"over", "under", "yes", "no"} else "")
                if not player:
                    continue
                point_raw = out.get("point")
                try:
                    point = float(point_raw) if point_raw is not None else None
                except (TypeError, ValueError):
                    point = None
                grouped.setdefault((player, point), []).append(out)

            for (player, point), outcomes in grouped.items():
                parsed: list[tuple[str, float, float]] = []
                try:
                    for out in outcomes:
                        side = str(out.get("name") or "").strip()
                        price = float(out.get("price"))
                        parsed.append((side, price, american_to_implied(price)))
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                if not parsed:
                    continue
                total = sum(p for _, _, p in parsed)
                if total <= 0:
                    continue
                for side, price, raw in parsed:
                    fair = raw / total if len(parsed) >= 2 else raw
                    identity = (mkey, player, point, side)
                    samples.setdefault(identity, {})[bkey] = (
                        clamp(fair),
                        clamp(raw),
                        price,
                        age_s,
                        btitle,
                    )

    edges: list[BookOfferEdge] = []
    event_id = str(payload.get("id") or "")

    for (mkey, player, point, side), by_book in samples.items():
        if len(by_book) < min_other_books + 1:
            continue
        for bkey, (book_fair, raw_break_even, price, age_s, btitle) in by_book.items():
            others = []
            for other_key, (other_fair, _, _, other_age, _) in by_book.items():
                if other_key == bkey:
                    continue
                weight = _book_weight(other_key) / (1.0 + other_age / 60.0)
                others.append((other_fair, weight))
            if len(others) < min_other_books:
                continue
            loo_fair = clamp(_weighted_mean(others))
            edge_pp = 100.0 * (loo_fair - raw_break_even)
            quality = clamp(
                0.50 * min(1.0, len(others) / 4.0)
                + 0.25 * clamp(1.0 - age_s / max_age_s)
                + 0.25 * (1.0 if any(k in REFERENCE_BOOKS for k in by_book if k != bkey) else 0.55)
            )
            edges.append(
                BookOfferEdge(
                    event_id=event_id,
                    market_key=mkey,
                    selection=f"{player} {side}" + (f" {point:g}" if point is not None else ""),
                    point=point,
                    bookmaker_key=bkey,
                    bookmaker_title=btitle,
                    american_price=price,
                    break_even_probability=raw_break_even,
                    leave_one_out_fair_probability=loo_fair,
                    edge_points=edge_pp,
                    age_s=age_s,
                    comparison_books=len(others),
                    evidence_quality=quality,
                )
            )

    return sorted(
        edges,
        key=lambda x: (x.edge_points, x.evidence_quality, x.comparison_books),
        reverse=True,
    )
