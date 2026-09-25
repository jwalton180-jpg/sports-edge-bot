from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from functools import lru_cache
import re
from typing import Iterable

from sports_edge.data.public_team_data import team_game_model
from sports_edge.models.game_scope import normalize
from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.live_board import market_side_probability
from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.tennis_research import TennisResearchModel, tennis_level_from_series


_TICKER_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_TICKER_DATE_RE = re.compile(r"(?:^|-)(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})", re.I)


def _parse_date(market: dict) -> date:
    """Prefer Kalshi's event/ticker date over settlement timestamps.

    Sports contracts often remain open/settle 1–3 days after the game. Using
    close_time as the event date silently breaks public schedule joins.
    """
    for key in ("event_ticker", "ticker"):
        raw = str(market.get(key) or "").upper()
        match = _TICKER_DATE_RE.search(raw)
        if not match:
            continue
        yy, mon, dd = match.groups()
        try:
            return date(2000 + int(yy), _TICKER_MONTHS[mon.upper()], int(dd))
        except ValueError:
            continue

    for key in ("close_time", "expected_expiration_time", "open_time"):
        raw = market.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
        except ValueError:
            continue
    return datetime.now(timezone.utc).date()


def _event_key(market: dict) -> str:
    return str(market.get("event_ticker") or market.get("ticker") or "").strip()


def _event_title(market: dict) -> str:
    return str(market.get("event_title") or market.get("title") or _event_key(market)).strip()


def _clean_selection(value: str | None) -> str:
    raw = str(value or "").strip()
    if normalize(raw) in {"", "yes", "no"}:
        return ""
    return raw


def _event_choices(rows: Iterable[KalshiSportMarket]) -> list[tuple[str, str, float, dict]]:
    """Return (selection, side, price, market) choices for one event."""
    choices: list[tuple[str, str, float, dict]] = []
    seen: set[str] = set()

    for row in rows:
        market = row.market
        yes = _clean_selection(
            market.get("yes_sub_title")
            or market.get("yes_title")
            or market.get("yes_label")
        )
        if yes:
            p = market_side_probability(market, "YES")
            key = normalize(yes)
            if p is not None and key and key not in seen:
                choices.append((yes, "YES", p, market))
                seen.add(key)

        no = _clean_selection(
            market.get("no_sub_title")
            or market.get("no_title")
            or market.get("no_label")
        )
        if no and not no.lower().startswith("no —"):
            p = market_side_probability(market, "NO")
            key = normalize(no)
            if p is not None and key and key not in seen:
                choices.append((no, "NO", p, market))
                seen.add(key)

    return choices


@lru_cache(maxsize=4)
def _tennis_model(gender: str, year: int) -> TennisResearchModel:
    return TennisResearchModel(gender, current_year=year)


def _tennis_event_candidates(rows: list[KalshiSportMarket]) -> list[ParlayCandidateLeg]:
    choices = _event_choices(rows)
    if len(choices) != 2:
        return []

    first_market = choices[0][3]
    series = str(first_market.get("series_ticker") or "")
    gender, level = tennis_level_from_series(series)
    event_date = _parse_date(first_market)

    a_name, a_side, a_price, a_market = choices[0]
    b_name, b_side, b_price, b_market = choices[1]
    model = _tennis_model(gender, event_date.year)
    evidence_a = model.probability(a_name, b_name, level=level, as_of=event_date)
    if evidence_a is None or not evidence_a.usable:
        return []

    evidence_b = replace(
        evidence_a,
        fair_probability=1.0 - evidence_a.fair_probability,
        factors=tuple(
            [
                f"Opponent-side complement of {a_name} model probability",
                *evidence_a.factors,
            ]
        ),
    )

    out: list[ParlayCandidateLeg] = []
    for name, side, price, market, evidence in (
        (a_name, a_side, a_price, a_market, evidence_a),
        (b_name, b_side, b_price, b_market, evidence_b),
    ):
        out.append(
            ParlayCandidateLeg(
                sport="Tennis",
                event_id=_event_key(market),
                event_title=_event_title(market),
                market_key="model_h2h",
                market_label="Match Winner",
                selection=name,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _team_event_candidates(sport: str, rows: list[KalshiSportMarket]) -> list[ParlayCandidateLeg]:
    choices = _event_choices(rows)
    if len(choices) != 2:
        return []

    a_name, a_side, a_price, a_market = choices[0]
    b_name, b_side, b_price, b_market = choices[1]
    event_date = _parse_date(a_market)
    evidence_a = team_game_model(sport, a_name, b_name, event_date=event_date)
    if evidence_a is None or not evidence_a.usable:
        return []
    evidence_b = replace(
        evidence_a,
        fair_probability=1.0 - evidence_a.fair_probability,
        factors=tuple(
            [
                f"Opponent-side complement of {a_name} team-strength probability",
                *evidence_a.factors,
            ]
        ),
    )

    out: list[ParlayCandidateLeg] = []
    for name, side, price, market, evidence in (
        (a_name, a_side, a_price, a_market, evidence_a),
        (b_name, b_side, b_price, b_market, evidence_b),
    ):
        out.append(
            ParlayCandidateLeg(
                sport=sport,
                event_id=_event_key(market),
                event_title=_event_title(market),
                market_key="model_h2h",
                market_label="Moneyline",
                selection=name,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def model_candidates_from_kalshi(
    grouped: dict[str, list[KalshiSportMarket]],
    *,
    sport_filter: str,
) -> list[ParlayCandidateLeg]:
    sports = (
        ("MLB", "NBA", "WNBA", "NFL", "Tennis")
        if sport_filter == "All"
        else (sport_filter,)
    )
    all_rows: list[ParlayCandidateLeg] = []

    for sport in sports:
        rows = grouped.get(sport, [])
        target_family = "Match Winner" if sport == "Tennis" else "Moneyline"
        events: dict[str, list[KalshiSportMarket]] = {}
        for row in rows:
            if row.family != target_family:
                continue
            key = _event_key(row.market)
            if key:
                events.setdefault(key, []).append(row)

        for event_rows in events.values():
            if sport == "Tennis":
                all_rows.extend(_tennis_event_candidates(event_rows))
            else:
                all_rows.extend(_team_event_candidates(sport, event_rows))

    # Model candidates are allowed to include both sides; the EV gate chooses.
    return all_rows


def attach_sportsbook_context(
    model_candidates: list[ParlayCandidateLeg],
    book_candidates: list[ParlayCandidateLeg],
) -> list[ParlayCandidateLeg]:
    """Attach books only when a selection match is unique within the sport.

    Ambiguous joins fail closed: the model candidate stays model-only rather
    than inheriting evidence from the wrong game.
    """
    by_key: dict[tuple[str, str], list[ParlayCandidateLeg]] = {}
    for row in book_candidates:
        key = (row.sport, normalize(row.selection))
        by_key.setdefault(key, []).append(row)

    out: list[ParlayCandidateLeg] = []
    for row in model_candidates:
        hits = by_key.get((row.sport, normalize(row.selection)), [])
        if len(hits) != 1:
            out.append(row)
            continue
        book = hits[0]
        out.append(
            replace(
                row,
                consensus_probability=book.consensus_probability,
                book_count=book.book_count,
                source_age_s=book.source_age_s,
                median_odds=book.median_odds,
                evidence_class="MODEL + BOOKS",
            )
        )
    return out
