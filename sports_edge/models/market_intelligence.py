from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from math import sqrt
from statistics import median
from typing import Any, Iterable

from sports_edge.core.math import american_to_implied, clamp


# Default free-tier panel: major US books plus an independent reference book.
# The Odds API lets callers request up to 10 bookmakers for the quota cost of
# one region. Paid-only books may be added by the user through configuration.
DEFAULT_BOOKMAKERS: tuple[str, ...] = (
    "draftkings",
    "fanduel",
    "betmgm",
    "betonlineag",
    "bovada",
    "pinnacle",
)

MAJOR_BOOKS = {
    "draftkings",
    "fanduel",
    "betmgm",
    "williamhill_us",  # Caesars
    "fanatics",
    "espnbet",
    "betrivers",
}
REFERENCE_BOOKS = {
    "pinnacle",
}
SECONDARY_REFERENCE_BOOKS = {
    "betonlineag",
    "bovada",
    "lowvig",
}


def bookmaker_csv(bookmakers: Iterable[str] | None = None) -> str:
    values = [str(x).strip() for x in (bookmakers or DEFAULT_BOOKMAKERS) if str(x).strip()]
    return ",".join(dict.fromkeys(values))


def bookmaker_role(key: str | None, title: str | None = None) -> str:
    k = (key or "").lower().strip()
    t = (title or "").lower().strip()
    if k in REFERENCE_BOOKS or "pinnacle" in t:
        return "REFERENCE"
    if k in MAJOR_BOOKS or any(
        name in t
        for name in (
            "draftkings",
            "fanduel",
            "betmgm",
            "caesars",
            "fanatics",
            "betrivers",
            "thescore bet",
            "espn bet",
        )
    ):
        return "MAJOR"
    if k in SECONDARY_REFERENCE_BOOKS or any(
        name in t for name in ("betonline", "bovada", "lowvig")
    ):
        return "SECONDARY_REFERENCE"
    return "OTHER"


def american_to_decimal(price: float) -> float:
    p = float(price)
    if p == 0:
        raise ValueError("American odds cannot be zero")
    return 1.0 + (p / 100.0 if p > 0 else 100.0 / abs(p))


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


@dataclass(frozen=True)
class BookSample:
    bookmaker_key: str
    bookmaker_title: str
    role: str
    selection: str
    side: str
    line: float | None
    american_price: float
    implied_probability: float
    fair_probability: float
    age_s: float
    vig_removed: bool

    @property
    def decimal_price(self) -> float:
        return american_to_decimal(self.american_price)


@dataclass(frozen=True)
class PriceEdge:
    market_key: str
    market_label: str
    event_id: str
    event_title: str
    selection: str
    side: str
    line: float | None
    consensus_probability: float
    reference_probability: float | None
    major_probability: float | None
    reference_vs_major_pp: float | None
    best_book_key: str
    best_book_title: str
    best_price: float
    best_implied_probability: float
    leave_one_out_probability: float
    edge_points: float
    expected_value: float
    book_count: int
    major_book_count: int
    reference_book_count: int
    dispersion_pp: float
    source_age_s: float
    data_quality: float
    evidence_class: str
    vig_removed: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PublicTrackRecord:
    handle: str
    wins: int
    losses: int
    pushes: int = 0
    avg_clv_points: float | None = None
    verified_posts: int | None = None

    @property
    def decisions(self) -> int:
        return int(self.wins) + int(self.losses)

    @property
    def win_rate(self) -> float:
        return self.wins / self.decisions if self.decisions else 0.0

    def wilson_lower_bound(self, z: float = 1.96) -> float:
        n = self.decisions
        if n <= 0:
            return 0.0
        p = self.win_rate
        den = 1.0 + z * z / n
        centre = p + z * z / (2.0 * n)
        adj = z * sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
        return max(0.0, (centre - adj) / den)

    def evidence_quality(self) -> float:
        # Evidence quality, not expected profitability.
        n = self.decisions
        sample = min(1.0, n / 300.0)
        verified = min(
            1.0,
            (self.verified_posts if self.verified_posts is not None else n) / max(1, n),
        )
        clv = 1.0 if self.avg_clv_points is not None else 0.40
        return clamp(0.55 * sample + 0.30 * verified + 0.15 * clv)


@dataclass(frozen=True)
class EvidenceBlend:
    probability: float
    market_weight: float
    model_weight: float
    public_weight: float
    notes: tuple[str, ...]


def blend_fair_probability(
    market_probability: float,
    *,
    model_probability: float | None = None,
    model_reliability: float = 0.0,
    public_probability: float | None = None,
    public_quality: float = 0.0,
) -> EvidenceBlend:
    """Blend evidence without letting unverified social signals dominate.

    Market consensus always retains a floor weight. A model only earns material
    weight when its reliability has been independently established. Public
    bettor/tipster evidence is capped at 8% and is ignored below quality 0.65.
    """

    market = clamp(float(market_probability), 0.001, 0.999)
    model_w = 0.0
    public_w = 0.0
    notes: list[str] = []

    if model_probability is not None and model_reliability > 0:
        rel = clamp(float(model_reliability))
        model_w = min(0.60, 0.60 * rel)
        notes.append(f"validated model weight {model_w:.0%}")

    if public_probability is not None and public_quality >= 0.65:
        q = clamp(float(public_quality))
        public_w = min(0.08, 0.08 * q)
        notes.append(f"verified public-signal weight {public_w:.0%}")
    elif public_probability is not None:
        notes.append("public signal ignored: evidence quality below threshold")

    market_w = max(0.32, 1.0 - model_w - public_w)
    total = market_w + model_w + public_w
    market_w /= total
    model_w /= total
    public_w /= total

    p = market_w * market
    if model_probability is not None:
        p += model_w * clamp(float(model_probability), 0.001, 0.999)
    if public_probability is not None:
        p += public_w * clamp(float(public_probability), 0.001, 0.999)

    return EvidenceBlend(
        probability=clamp(p, 0.001, 0.999),
        market_weight=market_w,
        model_weight=model_w,
        public_weight=public_w,
        notes=tuple(notes),
    )


def _weighted_consensus(samples: list[BookSample]) -> float:
    if not samples:
        raise ValueError("Need samples")
    role_weight = {
        "REFERENCE": 1.25,
        "MAJOR": 1.00,
        "SECONDARY_REFERENCE": 0.90,
        "OTHER": 0.75,
    }
    num = 0.0
    den = 0.0
    for sample in samples:
        freshness = 1.0 / (1.0 + sample.age_s / 60.0)
        w = role_weight.get(sample.role, 0.75) * freshness
        num += sample.fair_probability * w
        den += w
    return clamp(num / den if den else median(s.fair_probability for s in samples))


def _market_label(key: str) -> str:
    labels = {
        "h2h": "Moneyline",
        "spreads": "Spread",
        "totals": "Total",
        "batter_hits": "Hits",
        "batter_hits_alternate": "Hits",
        "batter_home_runs": "Home Runs",
        "batter_home_runs_alternate": "Home Runs",
        "pitcher_strikeouts": "Pitcher Strikeouts",
        "pitcher_strikeouts_alternate": "Pitcher Strikeouts",
        "batter_total_bases": "Total Bases",
        "batter_total_bases_alternate": "Total Bases",
        "batter_rbis": "RBIs",
        "batter_rbis_alternate": "RBIs",
        "player_pass_yds": "Passing Yards",
        "player_pass_tds": "Passing TDs",
        "player_rush_yds": "Rushing Yards",
        "player_rush_tds": "Rushing TDs",
        "player_receptions": "Receptions",
        "player_reception_yds": "Receiving Yards",
        "player_reception_tds": "Receiving TDs",
        "player_anytime_td": "Anytime TD",
        "player_tds": "Touchdowns",
    }
    return labels.get(key, key)


def _event_title(payload: dict) -> str:
    away = str(payload.get("away_team") or "").strip()
    home = str(payload.get("home_team") or "").strip()
    return f"{away} @ {home}".strip(" @")


def _quality(
    *,
    book_count: int,
    major_count: int,
    reference_count: int,
    dispersion_pp: float,
    source_age_s: float,
    vig_removed: bool,
) -> float:
    count_factor = min(1.0, book_count / 5.0)
    major_factor = min(1.0, major_count / 3.0)
    ref_factor = 1.0 if reference_count else 0.45
    dispersion_factor = clamp(1.0 - dispersion_pp / 12.0)
    freshness_factor = clamp(1.0 - source_age_s / 120.0)
    vig_factor = 1.0 if vig_removed else 0.55
    return clamp(
        0.30 * count_factor
        + 0.18 * major_factor
        + 0.14 * ref_factor
        + 0.16 * dispersion_factor
        + 0.14 * freshness_factor
        + 0.08 * vig_factor
    )


def _finalize_edge(
    *,
    market_key: str,
    event: dict,
    selection: str,
    side: str,
    line: float | None,
    samples: list[BookSample],
    min_books: int,
    max_age_s: float,
) -> PriceEdge | None:
    fresh = [s for s in samples if math.isfinite(s.age_s) and s.age_s <= max_age_s]
    if len(fresh) < 2:
        return None

    # Best bettor-facing price is the largest decimal payout.
    best = max(fresh, key=lambda s: s.decimal_price)
    others = [s for s in fresh if s.bookmaker_key != best.bookmaker_key]
    if not others:
        return None

    consensus = _weighted_consensus(fresh)
    loo = _weighted_consensus(others)
    ref = [s for s in fresh if s.role == "REFERENCE"]
    major = [s for s in fresh if s.role == "MAJOR"]
    reference_p = _weighted_consensus(ref) if ref else None
    major_p = _weighted_consensus(major) if major else None
    divergence = (
        100.0 * (reference_p - major_p)
        if reference_p is not None and major_p is not None
        else None
    )
    probs = [s.fair_probability for s in fresh]
    dispersion_pp = 100.0 * (max(probs) - min(probs)) if len(probs) > 1 else 0.0
    age_s = float(median(s.age_s for s in fresh))
    vig_removed = all(s.vig_removed for s in fresh)
    best_implied = american_to_implied(best.american_price)
    edge_pp = 100.0 * (loo - best_implied)
    ev = loo * best.decimal_price - 1.0
    major_count = sum(s.role == "MAJOR" for s in fresh)
    reference_count = sum(s.role == "REFERENCE" for s in fresh)
    data_quality = _quality(
        book_count=len(fresh),
        major_count=major_count,
        reference_count=reference_count,
        dispersion_pp=dispersion_pp,
        source_age_s=age_s,
        vig_removed=vig_removed,
    )

    warnings: list[str] = []
    reasons: list[str] = [
        f"leave-one-out fair {loo:.1%} versus {best.bookmaker_title} {best.american_price:+.0f}",
        f"{len(fresh)} fresh book(s), including {major_count} major and {reference_count} reference",
    ]
    if divergence is not None:
        reasons.append(f"reference vs major divergence {divergence:+.1f} pp")
    if not vig_removed:
        warnings.append("one-sided market: full no-vig removal unavailable")
    if len(fresh) < min_books:
        warnings.append(f"only {len(fresh)} books; require {min_books}")
    if dispersion_pp > 12.0:
        warnings.append(f"cross-book dispersion high ({dispersion_pp:.1f} pp)")
    if age_s > 120:
        warnings.append(f"market aging ({age_s:.0f}s)")
    if data_quality < 0.68:
        warnings.append(f"evidence quality low ({data_quality:.0%})")

    evidence = "MARKET BASELINE"
    if not warnings and ev >= 0.03 and edge_pp >= 2.0:
        evidence = "PRICE EDGE"
    elif ev >= 0.01 and edge_pp > 0:
        evidence = "WATCH"

    return PriceEdge(
        market_key=market_key,
        market_label=_market_label(market_key),
        event_id=str(event.get("id") or ""),
        event_title=_event_title(event),
        selection=selection,
        side=side,
        line=line,
        consensus_probability=consensus,
        reference_probability=reference_p,
        major_probability=major_p,
        reference_vs_major_pp=divergence,
        best_book_key=best.bookmaker_key,
        best_book_title=best.bookmaker_title,
        best_price=best.american_price,
        best_implied_probability=best_implied,
        leave_one_out_probability=loo,
        edge_points=edge_pp,
        expected_value=ev,
        book_count=len(fresh),
        major_book_count=major_count,
        reference_book_count=reference_count,
        dispersion_pp=dispersion_pp,
        source_age_s=age_s,
        data_quality=data_quality,
        evidence_class=evidence,
        vig_removed=vig_removed,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )


def game_price_edges(
    event: dict,
    *,
    market_key: str = "h2h",
    now: datetime | None = None,
    min_books: int = 3,
    max_age_s: float = 120.0,
) -> list[PriceEdge]:
    """Find cross-book price edges for a two-way game market."""

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    by_selection: dict[tuple[str, float | None], list[BookSample]] = {}

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

        update = market.get("last_update") or bookmaker.get("last_update")
        age_s = _age_seconds(update, now)
        parsed: list[tuple[dict, float]] = []
        for outcome in outcomes:
            try:
                raw = american_to_implied(float(outcome.get("price")))
            except (TypeError, ValueError, ZeroDivisionError):
                parsed = []
                break
            parsed.append((outcome, raw))
        if len(parsed) != 2:
            continue
        total = sum(p for _, p in parsed)
        if total <= 0:
            continue

        bkey = str(bookmaker.get("key") or "")
        btitle = str(bookmaker.get("title") or bkey or "Book")
        role = bookmaker_role(bkey, btitle)
        for outcome, raw in parsed:
            name = str(outcome.get("name") or "").strip()
            if not name:
                continue
            point = outcome.get("point")
            line = float(point) if point is not None else None
            sample = BookSample(
                bookmaker_key=bkey,
                bookmaker_title=btitle,
                role=role,
                selection=name,
                side=name,
                line=line,
                american_price=float(outcome.get("price")),
                implied_probability=raw,
                fair_probability=clamp(raw / total),
                age_s=age_s,
                vig_removed=True,
            )
            by_selection.setdefault((name, line), []).append(sample)

    rows: list[PriceEdge] = []
    for (selection, line), samples in by_selection.items():
        row = _finalize_edge(
            market_key=market_key,
            event=event,
            selection=selection,
            side=selection,
            line=line,
            samples=samples,
            min_books=min_books,
            max_age_s=max_age_s,
        )
        if row:
            rows.append(row)
    return sorted(
        rows,
        key=lambda x: (
            x.evidence_class == "PRICE EDGE",
            x.expected_value,
            x.data_quality,
            x.book_count,
        ),
        reverse=True,
    )


def _prop_player(outcome: dict) -> str:
    description = str(outcome.get("description") or "").strip()
    name = str(outcome.get("name") or "").strip()
    if description:
        return description
    if name.lower() not in {"over", "under", "yes", "no"}:
        return name
    return ""


def prop_price_edges(
    payload: dict,
    *,
    market_keys: Iterable[str],
    now: datetime | None = None,
    min_books: int = 3,
    max_age_s: float = 120.0,
) -> list[PriceEdge]:
    """Find underpriced current player-prop prices across actual sportsbooks.

    Two-sided Over/Under or Yes/No pairs receive per-book no-vig probabilities.
    One-sided outcomes are allowed for price-shopping diagnostics but cannot be
    promoted to PRICE EDGE because the bookmaker hold cannot be removed.
    """

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    wanted = set(str(k) for k in market_keys)
    samples: dict[tuple[str, str, float | None, str], list[BookSample]] = {}

    for bookmaker in payload.get("bookmakers", []) or []:
        bkey = str(bookmaker.get("key") or "")
        btitle = str(bookmaker.get("title") or bkey or "Book")
        role = bookmaker_role(bkey, btitle)

        for market in bookmaker.get("markets", []) or []:
            mkey = str(market.get("key") or "")
            if mkey not in wanted:
                continue
            age_s = _age_seconds(
                market.get("last_update") or bookmaker.get("last_update"),
                now,
            )
            outcomes = market.get("outcomes", []) or []

            grouped: dict[tuple[str, float | None], list[dict]] = {}
            for outcome in outcomes:
                player = _prop_player(outcome)
                if not player:
                    continue
                point = outcome.get("point")
                line = float(point) if point is not None else None
                grouped.setdefault((player, line), []).append(outcome)

            for (player, line), group in grouped.items():
                parsed: list[tuple[dict, float]] = []
                for outcome in group:
                    try:
                        raw = american_to_implied(float(outcome.get("price")))
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue
                    parsed.append((outcome, raw))
                if not parsed:
                    continue

                two_sided = len(parsed) >= 2
                total = sum(p for _, p in parsed)
                for outcome, raw in parsed:
                    side = str(outcome.get("name") or "").strip()
                    fair = raw / total if two_sided and total > 0 else raw
                    sample = BookSample(
                        bookmaker_key=bkey,
                        bookmaker_title=btitle,
                        role=role,
                        selection=player,
                        side=side,
                        line=line,
                        american_price=float(outcome.get("price")),
                        implied_probability=raw,
                        fair_probability=clamp(fair),
                        age_s=age_s,
                        vig_removed=two_sided,
                    )
                    samples.setdefault((mkey, player, line, side), []).append(sample)

    rows: list[PriceEdge] = []
    for (mkey, player, line, side), group in samples.items():
        row = _finalize_edge(
            market_key=mkey,
            event=payload,
            selection=player,
            side=side,
            line=line,
            samples=group,
            min_books=min_books,
            max_age_s=max_age_s,
        )
        if row:
            # Do not promote one-sided prices to a model-like edge.
            if not row.vig_removed and row.evidence_class == "PRICE EDGE":
                row = PriceEdge(
                    **{**row.__dict__, "evidence_class": "WATCH",
                       "warnings": tuple(dict.fromkeys((*row.warnings, "one-sided market cannot be edge-qualified")))}
                )
            rows.append(row)

    return sorted(
        rows,
        key=lambda x: (
            x.evidence_class == "PRICE EDGE",
            x.expected_value,
            x.data_quality,
            x.book_count,
        ),
        reverse=True,
    )
