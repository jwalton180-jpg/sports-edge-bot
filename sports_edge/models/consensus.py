from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from statistics import median
from typing import Any

from sports_edge.core.math import american_to_implied, clamp


@dataclass(frozen=True)
class ConsensusQuote:
    selection: str
    fair_probability: float
    book_count: int
    median_age_s: float
    max_age_s: float
    disagreement_pp: float
    data_quality: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MarketConsensusMatch:
    event_id: str
    event_title: str
    selection: str
    opposite_selection: str | None
    quote: ConsensusQuote
    match_confidence: float


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


def normalize_name(value: str | None) -> str:
    text = (value or "").lower()
    text = text.replace("&", " and ")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _aliases(name: str) -> tuple[str, ...]:
    normalized = normalize_name(name)
    if not normalized:
        return ()
    parts = normalized.split()
    aliases = {normalized}

    # Safer shortened forms: require at least two tokens. A bare nickname such
    # as "Pirates" or "Panthers" is too collision-prone for cross-source
    # identity resolution.
    if len(parts) >= 3:
        last_two = " ".join(parts[-2:])
        if len(last_two) >= 8:
            aliases.add(last_two)
    if len(parts) >= 2:
        first_last = f"{parts[0]} {parts[-1]}"
        if len(first_last) >= 8:
            aliases.add(first_last)

    return tuple(sorted(aliases, key=len, reverse=True))


def _name_score(name: str, text: str) -> float:
    hay = normalize_name(text)
    if not hay:
        return 0.0
    aliases = _aliases(name)
    if not aliases:
        return 0.0
    full = aliases[0]
    if full in hay:
        return 1.0
    for alias in aliases[1:]:
        if re.search(rf"\b{re.escape(alias)}\b", hay):
            return 0.82
    return 0.0


def _event_context_text(market: dict) -> str:
    return " ".join(
        str(market.get(key) or "")
        for key in (
            "title",
            "subtitle",
            "event_title",
            "yes_sub_title",
            "yes_title",
            "yes_label",
            "no_sub_title",
            "no_title",
            "no_label",
            "ticker",
            "event_ticker",
            "series_ticker",
        )
    )


def _event_identity_score(market: dict, event: dict) -> float:
    """Require both external-event participants in the Kalshi event context.

    This prevents cross-domain nickname collisions such as Pittsburgh Pirates
    matching a Pirates-of-the-Caribbean entertainment market.
    """
    home = str(event.get("home_team") or "").strip()
    away = str(event.get("away_team") or "").strip()
    if not home or not away:
        return 0.0

    context = _event_context_text(market)
    home_score = _name_score(home, context)
    away_score = _name_score(away, context)

    if home_score <= 0 or away_score <= 0:
        return 0.0
    return min(home_score, away_score)


def _fresh_age_seconds(last_update: Any, now: datetime) -> float:
    dt = _parse_iso(last_update)
    if dt is None:
        return float("inf")
    return max(0.0, (now - dt).total_seconds())


def team_total_consensus_from_event(
    event: dict,
    *,
    target_team: str,
    target_side: str,
    target_point: float,
    selection_label: str,
    now: datetime | None = None,
    max_age_s: float = 120.0,
    min_books: int = 2,
    max_disagreement_pp: float = 12.0,
    point_tolerance: float = 0.01,
) -> ConsensusQuote | None:
    """No-vig consensus for one exact team-total side and line.

    Team-total markets are event-level additional markets. Outcomes are grouped
    by team description + point; featured and alternate versions may overlap,
    so each bookmaker contributes at most one exact-line sample.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    wanted_team = normalize_name(target_team)
    wanted_side = normalize_name(target_side)
    by_book: dict[str, tuple[float, float]] = {}

    for idx, bookmaker in enumerate(event.get("bookmakers", []) or []):
        book_key = str(bookmaker.get("key") or bookmaker.get("title") or idx)
        best_sample: tuple[float, float] | None = None

        for market in bookmaker.get("markets", []) or []:
            if str(market.get("key") or "") not in {"team_totals", "alternate_team_totals"}:
                continue
            age_s = _fresh_age_seconds(
                market.get("last_update") or bookmaker.get("last_update"),
                now,
            )
            if not math.isfinite(age_s) or age_s > max_age_s:
                continue

            grouped: dict[tuple[str, float], list[tuple[str, float]]] = {}
            for outcome in market.get("outcomes", []) or []:
                team = normalize_name(
                    str(
                        outcome.get("description")
                        or outcome.get("team")
                        or outcome.get("participant")
                        or ""
                    )
                )
                if team != wanted_team:
                    continue
                try:
                    point = float(outcome.get("point"))
                    price = float(outcome.get("price"))
                    implied = american_to_implied(price)
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                grouped.setdefault((team, point), []).append(
                    (normalize_name(str(outcome.get("name") or "")), implied)
                )

            for (_team, point), pair in grouped.items():
                if abs(point - float(target_point)) > point_tolerance:
                    continue
                over = next((p for side, p in pair if side == "over"), None)
                under = next((p for side, p in pair if side == "under"), None)
                if over is None or under is None:
                    continue
                total = over + under
                if total <= 0:
                    continue
                fair = over / total if wanted_side == "over" else under / total
                sample = (clamp(fair), age_s)
                if best_sample is None or age_s < best_sample[1]:
                    best_sample = sample

        if best_sample is not None:
            by_book[book_key] = best_sample

    if not by_book:
        return None

    values = list(by_book.values())
    weights = [1.0 / (1.0 + age / 45.0) for _, age in values]
    denom = sum(weights)
    fair = sum(p * w for (p, _), w in zip(values, weights)) / denom
    ages = [age for _, age in values]
    probabilities = [p for p, _ in values]
    med_age = median(ages)
    disagreement_pp = (
        100.0 * (max(probabilities) - min(probabilities))
        if len(probabilities) > 1
        else 0.0
    )
    book_count = len(values)

    book_factor = min(1.0, book_count / 4.0)
    freshness_factor = clamp(1.0 - med_age / max(max_age_s, 1.0))
    disagreement_factor = clamp(
        1.0 - disagreement_pp / max(max_disagreement_pp * 1.5, 1.0)
    )
    data_quality = clamp(
        0.55 * book_factor + 0.30 * freshness_factor + 0.15 * disagreement_factor
    )

    warnings: list[str] = []
    if book_count < min_books:
        warnings.append(f"Only {book_count} fresh sportsbook source(s)")
    if disagreement_pp > max_disagreement_pp:
        warnings.append(f"Cross-book disagreement {disagreement_pp:.1f} pp")
    if med_age > max_age_s * 0.75:
        warnings.append(f"Sportsbook consensus aging ({med_age:.0f}s)")

    return ConsensusQuote(
        selection=selection_label,
        fair_probability=clamp(fair),
        book_count=book_count,
        median_age_s=float(med_age),
        max_age_s=float(max(ages)),
        disagreement_pp=float(disagreement_pp),
        data_quality=float(data_quality),
        warnings=tuple(warnings),
    )


def line_consensus_from_event(
    event: dict,
    *,
    market_key: str,
    target_name: str,
    target_point: float,
    selection_label: str,
    now: datetime | None = None,
    max_age_s: float = 120.0,
    min_books: int = 2,
    max_disagreement_pp: float = 12.0,
    point_tolerance: float = 0.01,
) -> ConsensusQuote | None:
    """No-vig consensus for one exact spread/total line.

    The line itself is part of the identity. Books offering a nearby but
    different number are ignored rather than silently treated as equivalent.
    """

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    values: list[tuple[float, float]] = []
    wanted = normalize_name(target_name)

    for bookmaker in event.get("bookmakers", []) or []:
        market = next(
            (m for m in bookmaker.get("markets", []) or [] if m.get("key") == market_key),
            None,
        )
        if not market:
            continue
        outcomes = market.get("outcomes", []) or []
        if len(outcomes) != 2:
            continue

        age_s = _fresh_age_seconds(
            market.get("last_update") or bookmaker.get("last_update"),
            now,
        )
        if not math.isfinite(age_s) or age_s > max_age_s:
            continue

        parsed: list[tuple[dict, float]] = []
        valid = True
        for outcome in outcomes:
            try:
                price = float(outcome.get("price"))
                raw_p = american_to_implied(price)
                point = float(outcome.get("point"))
            except (TypeError, ValueError, ZeroDivisionError):
                valid = False
                break
            parsed.append((outcome, raw_p))
        if not valid:
            continue

        hit: tuple[dict, float] | None = None
        other: tuple[dict, float] | None = None
        for item in parsed:
            outcome, raw_p = item
            if (
                normalize_name(str(outcome.get("name") or "")) == wanted
                and abs(float(outcome.get("point")) - float(target_point)) <= point_tolerance
            ):
                hit = item
            else:
                other = item
        if hit is None or other is None:
            continue

        # Require the other half of the exact two-way line. For spreads the
        # handicap signs should be opposite; for totals both sides share point.
        hit_point = float(hit[0].get("point"))
        other_point = float(other[0].get("point"))
        if market_key == "spreads":
            compatible = abs(hit_point + other_point) <= point_tolerance
        elif market_key == "totals":
            compatible = abs(hit_point - other_point) <= point_tolerance
        else:
            compatible = True
        if not compatible:
            continue

        total = hit[1] + other[1]
        if total <= 0:
            continue
        values.append((hit[1] / total, age_s))

    if not values:
        return None

    weights = [1.0 / (1.0 + age / 45.0) for _, age in values]
    denom = sum(weights)
    fair = sum(p * w for (p, _), w in zip(values, weights)) / denom
    ages = [age for _, age in values]
    probabilities = [p for p, _ in values]
    med_age = median(ages)
    disagreement_pp = (
        100.0 * (max(probabilities) - min(probabilities))
        if len(probabilities) > 1
        else 0.0
    )
    book_count = len(values)

    book_factor = min(1.0, book_count / 4.0)
    freshness_factor = clamp(1.0 - med_age / max(max_age_s, 1.0))
    disagreement_factor = clamp(
        1.0 - disagreement_pp / max(max_disagreement_pp * 1.5, 1.0)
    )
    data_quality = clamp(
        0.55 * book_factor + 0.30 * freshness_factor + 0.15 * disagreement_factor
    )

    warnings: list[str] = []
    if book_count < min_books:
        warnings.append(f"Only {book_count} fresh sportsbook source(s)")
    if disagreement_pp > max_disagreement_pp:
        warnings.append(f"Cross-book disagreement {disagreement_pp:.1f} pp")
    if med_age > max_age_s * 0.75:
        warnings.append(f"Sportsbook consensus aging ({med_age:.0f}s)")

    return ConsensusQuote(
        selection=selection_label,
        fair_probability=clamp(fair),
        book_count=book_count,
        median_age_s=float(med_age),
        max_age_s=float(max(ages)),
        disagreement_pp=float(disagreement_pp),
        data_quality=float(data_quality),
        warnings=tuple(warnings),
    )


def consensus_from_event(
    event: dict,
    *,
    market_key: str = "h2h",
    now: datetime | None = None,
    max_age_s: float = 120.0,
    min_books: int = 2,
    max_disagreement_pp: float = 12.0,
) -> dict[str, ConsensusQuote]:
    """Build a recency-weighted, no-vig consensus from independent books.

    Only fresh two-way markets are accepted. Stale books are discarded rather
    than blended into a signal. The returned warnings are gating information;
    callers should fail closed when warnings are present.
    """

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    samples: dict[str, list[tuple[float, float]]] = {}

    for bookmaker in event.get("bookmakers", []) or []:
        market = next((m for m in bookmaker.get("markets", []) or [] if m.get("key") == market_key), None)
        if not market:
            continue
        outcomes = market.get("outcomes", []) or []
        if len(outcomes) != 2:
            continue

        update = market.get("last_update") or bookmaker.get("last_update")
        age_s = _fresh_age_seconds(update, now)
        if not math.isfinite(age_s) or age_s > max_age_s:
            continue

        parsed: list[tuple[str, float]] = []
        valid = True
        for outcome in outcomes:
            name = str(outcome.get("name") or "").strip()
            try:
                price = float(outcome.get("price"))
                raw_p = american_to_implied(price)
            except (TypeError, ValueError, ZeroDivisionError):
                valid = False
                break
            if not name:
                valid = False
                break
            parsed.append((name, raw_p))
        if not valid:
            continue

        total = sum(p for _, p in parsed)
        if total <= 0:
            continue
        for name, raw_p in parsed:
            samples.setdefault(name, []).append((raw_p / total, age_s))

    result: dict[str, ConsensusQuote] = {}
    for selection, values in samples.items():
        if not values:
            continue
        weights = [1.0 / (1.0 + age / 45.0) for _, age in values]
        denom = sum(weights)
        fair = sum(p * w for (p, _), w in zip(values, weights)) / denom
        ages = [age for _, age in values]
        probabilities = [p for p, _ in values]
        med_age = median(ages)
        disagreement_pp = 100.0 * (max(probabilities) - min(probabilities)) if len(probabilities) > 1 else 0.0
        book_count = len(values)

        book_factor = min(1.0, book_count / 4.0)
        freshness_factor = clamp(1.0 - med_age / max(max_age_s, 1.0))
        disagreement_factor = clamp(1.0 - disagreement_pp / max(max_disagreement_pp * 1.5, 1.0))
        data_quality = clamp(0.55 * book_factor + 0.30 * freshness_factor + 0.15 * disagreement_factor)

        warnings: list[str] = []
        if book_count < min_books:
            warnings.append(f"Only {book_count} fresh sportsbook source(s)")
        if disagreement_pp > max_disagreement_pp:
            warnings.append(f"Cross-book disagreement {disagreement_pp:.1f} pp")
        if med_age > max_age_s * 0.75:
            warnings.append(f"Sportsbook consensus aging ({med_age:.0f}s)")

        result[selection] = ConsensusQuote(
            selection=selection,
            fair_probability=clamp(fair),
            book_count=book_count,
            median_age_s=float(med_age),
            max_age_s=float(max(ages)),
            disagreement_pp=float(disagreement_pp),
            data_quality=float(data_quality),
            warnings=tuple(warnings),
        )
    return result


def match_market_to_event(
    market: dict,
    odds_events: list[dict],
    *,
    now: datetime | None = None,
    max_age_s: float = 120.0,
) -> MarketConsensusMatch | None:
    """Resolve a Kalshi YES contract to an external event + participant.

    We only resolve when the YES side can be identified from a participant
    name. Ambiguous event-level questions fail closed.
    """

    title = str(market.get("title") or market.get("subtitle") or market.get("ticker") or "")
    yes_text = " ".join(
        str(market.get(key) or "")
        for key in ("yes_sub_title", "yes_title", "yes_label", "subtitle")
    ).strip()

    best: MarketConsensusMatch | None = None
    for event in odds_events:
        identity_score = _event_identity_score(market, event)
        if identity_score <= 0:
            continue

        consensus = consensus_from_event(event, now=now, max_age_s=max_age_s)
        if not consensus:
            continue

        selections = list(consensus)
        yes_scores = {name: _name_score(name, yes_text) for name in selections}
        title_scores = {name: _name_score(name, title) for name in selections}

        chosen: str | None = None
        confidence = 0.0

        yes_hits = [name for name, score in yes_scores.items() if score > 0]
        if len(yes_hits) == 1:
            chosen = yes_hits[0]
            confidence = min(1.0, 0.93 * yes_scores[chosen] + 0.07)
        else:
            title_hits = [name for name, score in title_scores.items() if score > 0]
            if len(title_hits) == 1:
                chosen = title_hits[0]
                confidence = 0.88 * title_scores[chosen]

        if not chosen:
            continue

        event_title = f"{event.get('away_team', '')} @ {event.get('home_team', '')}".strip(" @")
        opposite = next((name for name in selections if name != chosen), None)
        candidate = MarketConsensusMatch(
            event_id=str(event.get("id") or ""),
            event_title=event_title,
            selection=chosen,
            opposite_selection=opposite,
            quote=consensus[chosen],
            match_confidence=clamp(confidence * identity_score),
        )
        if best is None or candidate.match_confidence > best.match_confidence:
            best = candidate

    return best
