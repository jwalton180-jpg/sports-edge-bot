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
from sports_edge.models.player_prop_models import player_prop_evidence
from sports_edge.models.tennis_research import TennisResearchModel, tennis_level_from_series


_TICKER_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_TICKER_DATE_RE = re.compile(
    r"(?:^|-)(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})",
    re.I,
)

PRESET_MODEL_FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "MLB Hits": ("MLB", ("Hits",)),
    "MLB Home Runs": ("MLB", ("Home Runs",)),
    "MLB Strikeouts": ("MLB", ("Strikeouts",)),
    "NFL Passing": ("NFL", ("Passing Yards", "Passing TDs")),
    "NFL Rushing": ("NFL", ("Rushing Yards",)),
    "NFL Receiving": ("NFL", ("Receiving Yards", "Receptions")),
    "NFL Touchdowns": ("NFL", ("Player Touchdowns",)),
    "NBA Points": ("NBA", ("Points",)),
    "NBA Rebounds": ("NBA", ("Rebounds",)),
    "NBA Assists": ("NBA", ("Assists",)),
    "NBA Threes": ("NBA", ("Three-Pointers",)),
    "NBA PRA": ("NBA", ("Points + Rebounds + Assists",)),
    "WNBA Points": ("WNBA", ("Points",)),
    "WNBA Rebounds": ("WNBA", ("Rebounds",)),
    "WNBA Assists": ("WNBA", ("Assists",)),
    "WNBA Threes": ("WNBA", ("Three-Pointers",)),
    "WNBA PRA": ("WNBA", ("Points + Rebounds + Assists",)),
}

PLAYER_MODEL_FAMILIES: dict[str, tuple[str, ...]] = {
    "MLB": ("Hits", "Home Runs", "Strikeouts"),
    "NFL": (
        "Passing Yards",
        "Passing TDs",
        "Rushing Yards",
        "Receiving Yards",
        "Receptions",
        "Player Touchdowns",
    ),
    "NBA": (
        "Points",
        "Rebounds",
        "Assists",
        "Three-Pointers",
        "Points + Rebounds + Assists",
    ),
    "WNBA": (
        "Points",
        "Rebounds",
        "Assists",
        "Three-Pointers",
        "Points + Rebounds + Assists",
    ),
}


def _parse_date(market: dict) -> date:
    """Prefer Kalshi's event/ticker date over settlement timestamps."""
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


def canonical_game_key(market: dict) -> str:
    """Collapse series-specific event tickers to the underlying sport event.

    KXMLBHIT-<game>, KXMLBKS-<game>, and KXMLBGAME-<game> therefore share the
    same event id. This prevents accidental same-game stacking in normal parlays.
    """
    raw = _event_key(market)
    if "-" not in raw:
        return raw
    suffix = raw.split("-", 1)[1]
    # Tennis set events append "-1"/"-2"; match-winner parlays use the base match.
    if str(market.get("sports_edge_sport") or "") == "Tennis" and re.search(r"-\d+$", suffix):
        suffix = suffix.rsplit("-", 1)[0]
    return suffix


def _event_title(market: dict) -> str:
    return str(
        market.get("event_title")
        or market.get("title")
        or _event_key(market)
    ).strip()


def _clean_selection(value: str | None) -> str:
    raw = str(value or "").strip()
    if normalize(raw) in {"", "yes", "no"}:
        return ""
    return raw


def _event_choices(rows: Iterable[KalshiSportMarket]) -> list[tuple[str, str, float, dict]]:
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
        factors=(
            f"Opponent-side complement of {a_name} model probability",
            *evidence_a.factors,
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
                event_id=canonical_game_key(market),
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


def _team_event_candidates(
    sport: str,
    rows: list[KalshiSportMarket],
) -> list[ParlayCandidateLeg]:
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
        factors=(
            f"Opponent-side complement of {a_name} team-strength probability",
            *evidence_a.factors,
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
                event_id=canonical_game_key(market),
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


def _player_name(market: dict) -> str:
    title = str(market.get("title") or "").strip()
    if ":" in title:
        return title.split(":", 1)[0].strip()
    subtitle = str(market.get("yes_sub_title") or "").strip()
    if ":" in subtitle:
        return subtitle.split(":", 1)[0].strip()
    return ""


def _market_line(market: dict) -> float | None:
    raw = market.get("floor_strike")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    text = " ".join(
        str(market.get(key) or "")
        for key in ("title", "yes_sub_title")
    )
    match = re.search(r"(\d+(?:\.\d+)?)\+", text)
    if not match:
        return None
    return float(match.group(1)) - 0.5


def _threshold_label(line: float) -> str:
    threshold = line + 0.5
    return str(int(threshold)) if threshold.is_integer() else f"{threshold:g}"


def _player_prop_candidates(row: KalshiSportMarket) -> list[ParlayCandidateLeg]:
    sport, family, market = row.sport, row.family, row.market
    if family not in PLAYER_MODEL_FAMILIES.get(sport, ()):
        return []
    player = _player_name(market)
    line = _market_line(market)
    if not player or line is None:
        return []

    event_date = _parse_date(market)
    evidence = player_prop_evidence(
        sport,
        player,
        family,
        line=line,
        season=event_date.year,
    )
    if evidence is None or not evidence.usable:
        return []

    threshold = _threshold_label(line)
    yes_price = market_side_probability(market, "YES")
    no_price = market_side_probability(market, "NO")
    event_id = canonical_game_key(market)
    ticker = str(market.get("ticker") or "")
    event_title = _event_title(market)

    out: list[ParlayCandidateLeg] = []
    for side, price, fair, selection in (
        ("YES", yes_price, evidence.fair_probability, f"{player} {threshold}+ {family}"),
        ("NO", no_price, 1.0 - evidence.fair_probability, f"{player} under {threshold} {family}"),
    ):
        if price is None:
            continue
        side_evidence = evidence if side == "YES" else replace(
            evidence,
            fair_probability=1.0 - evidence.fair_probability,
            factors=(
                f"NO-side complement of {player} {threshold}+ {family}",
                *evidence.factors,
            ),
        )
        out.append(
            ParlayCandidateLeg(
                sport=sport,
                event_id=event_id,
                event_title=event_title,
                market_key=f"model_prop:{family}",
                market_label=family,
                selection=selection,
                consensus_probability=fair,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=ticker,
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (fair - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=fair,
                model_confidence=side_evidence.confidence,
                model_name=side_evidence.model_name,
                model_sample_size=side_evidence.sample_size,
                model_reasons=side_evidence.factors,
                model_warnings=side_evidence.warnings,
            )
        )
    return out


def _sports_for_filter(sport_filter: str) -> tuple[str, ...]:
    return (
        ("MLB", "NBA", "WNBA", "NFL", "Tennis")
        if sport_filter == "All"
        else (sport_filter,)
    )


def _game_candidates(
    grouped: dict[str, list[KalshiSportMarket]],
    *,
    sports: tuple[str, ...],
) -> list[ParlayCandidateLeg]:
    out: list[ParlayCandidateLeg] = []
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
                out.extend(_tennis_event_candidates(event_rows))
            else:
                out.extend(_team_event_candidates(sport, event_rows))
    return out


def preset_contract_count(
    grouped: dict[str, list[KalshiSportMarket]],
    *,
    sport_filter: str,
    preset: str,
) -> int:
    sports = _sports_for_filter(sport_filter)
    if preset == "Tennis Moneyline":
        return sum(
            1 for row in grouped.get("Tennis", [])
            if row.family == "Match Winner"
        )
    if preset == "NFL Game Markets":
        return sum(
            1 for row in grouped.get("NFL", [])
            if row.family == "Moneyline"
        )
    spec = PRESET_MODEL_FAMILIES.get(preset)
    if spec:
        sport, families = spec
        if sport_filter not in {"All", sport}:
            return 0
        return sum(
            1 for row in grouped.get(sport, [])
            if row.family in families
        )
    if preset in {"Best Available", "Mixed Sports"}:
        count = 0
        for sport in sports:
            target = "Match Winner" if sport == "Tennis" else "Moneyline"
            count += sum(1 for row in grouped.get(sport, []) if row.family == target)
            count += sum(
                1 for row in grouped.get(sport, [])
                if row.family in PLAYER_MODEL_FAMILIES.get(sport, ())
            )
        return count
    return 0


def model_candidates_for_preset(
    grouped: dict[str, list[KalshiSportMarket]],
    *,
    sport_filter: str,
    preset: str,
) -> list[ParlayCandidateLeg]:
    sports = _sports_for_filter(sport_filter)

    if preset == "Tennis Moneyline":
        return _game_candidates(grouped, sports=("Tennis",))
    if preset == "NFL Game Markets":
        return _game_candidates(grouped, sports=("NFL",))

    spec = PRESET_MODEL_FAMILIES.get(preset)
    if spec:
        sport, families = spec
        if sport_filter not in {"All", sport}:
            return []
        out: list[ParlayCandidateLeg] = []
        for row in grouped.get(sport, []):
            if row.family in families:
                out.extend(_player_prop_candidates(row))
        return out

    if preset in {"Best Available", "Mixed Sports"}:
        out = _game_candidates(grouped, sports=sports)
        for sport in sports:
            allowed = set(PLAYER_MODEL_FAMILIES.get(sport, ()))
            if not allowed:
                continue
            for row in grouped.get(sport, []):
                if row.family in allowed:
                    out.extend(_player_prop_candidates(row))
        return out

    return []


def model_candidates_from_kalshi(
    grouped: dict[str, list[KalshiSportMarket]],
    *,
    sport_filter: str,
) -> list[ParlayCandidateLeg]:
    """Backward-compatible default: all model-supported candidates."""
    return model_candidates_for_preset(
        grouped,
        sport_filter=sport_filter,
        preset="Best Available",
    )


def attach_sportsbook_context(
    model_candidates: list[ParlayCandidateLeg],
    book_candidates: list[ParlayCandidateLeg],
) -> list[ParlayCandidateLeg]:
    """Attach books only when a selection match is unique within the sport.

    Sportsbooks remain secondary evidence. Ambiguous joins fail closed to the
    model-only candidate rather than borrowing evidence from the wrong event.
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
