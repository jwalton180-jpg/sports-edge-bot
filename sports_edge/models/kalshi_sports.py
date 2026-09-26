from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sports_edge.models.game_scope import looks_like_future, normalize
from sports_edge.models.live_board import market_side_probability


SUPPORTED_SPORTS: tuple[str, ...] = ("MLB", "NBA", "WNBA", "NFL", "Tennis")


@dataclass(frozen=True)
class KalshiSportMarket:
    sport: str
    family: str
    market: dict


@dataclass(frozen=True)
class CatalogDiagnostics:
    total_open_markets: int
    classified_markets: int
    unclassified_supported_prefixes: tuple[str, ...]
    counts_by_sport: dict[str, int]


def _series(market: dict) -> str:
    value = market.get("series_ticker")
    if value:
        return str(value).upper().strip()
    ticker = str(market.get("ticker") or "").upper().strip()
    return ticker.split("-", 1)[0]


def _text(market: dict) -> str:
    return normalize(
        " ".join(
            str(market.get(key) or "")
            for key in (
                "title",
                "subtitle",
                "event_title",
                "yes_sub_title",
                "no_sub_title",
                "ticker",
                "event_ticker",
                "series_ticker",
                "series_title",
                "series_category",
                "series_tags",
                "sports_edge_sport",
                "category",
            )
        )
    )


def _contains(series: str, *tokens: str) -> bool:
    return any(token in series for token in tokens)


def _baseball_family(series: str) -> str:
    checks = (
        (("GAME",), "Moneyline"),
        (("SPREAD",), "Spread"),
        (("TEAMTOTAL",), "Team Total"),
        (("TB", "TOTALBASE"), "Total Bases"),
        (("TOTAL",), "Game Total"),
        (("F3",), "First 3 Innings"),
        (("F5",), "First 5 Innings"),
        (("F7",), "First 7 Innings"),
        (("HRR",), "Hits + Runs + RBIs"),
        (("HIT",), "Hits"),
        (("HR",), "Home Runs"),
        (("RBI",), "RBIs"),
        (("KS", "STRIKEOUT", "PITCHERK"), "Strikeouts"),
        (("RFI",), "First Inning Run"),
        (("EXTRAS",), "Extra Innings"),
    )
    for tokens, family in checks:
        if _contains(series, *tokens):
            return family
    return "Other MLB"


def _football_family(series: str) -> str:
    checks = (
        (("GAME",), "Moneyline"),
        (("SPREAD",), "Spread"),
        (("TEAMTOTAL",), "Team Total"),
        (("TOTAL",), "Game Total"),
        (("1H",), "First Half"),
        (("2H",), "Second Half"),
        (("1Q",), "First Quarter"),
        (("2Q",), "Second Quarter"),
        (("3Q",), "Third Quarter"),
        (("4Q",), "Fourth Quarter"),
        (("PASSYDS", "PASSYARD"), "Passing Yards"),
        (("PASSTDS", "PASSTD"), "Passing TDs"),
        (("PASSATT",), "Pass Attempts"),
        (("PASSCOMP",), "Pass Completions"),
        (("PASSINT",), "Pass Interceptions"),
        (("RSHYDS", "RUSHYDS", "RUSHYARD"), "Rushing Yards"),
        (("RSHATT",), "Rush Attempts"),
        (("RRYDS",), "Rushing + Receiving Yards"),
        (("RECYDS", "RECEIVINGYDS", "RECYARD"), "Receiving Yards"),
        (("KXNFLREC", "RECEPTIONS", "RECPT"), "Receptions"),
        (("FIRSTTD",), "First Touchdown"),
        (("TD",), "Player Touchdowns"),
        (("FG",), "Field Goals"),
        (("WINMARGIN",), "Win Margin"),
        (("OT",), "Overtime"),
    )
    for tokens, family in checks:
        if _contains(series, *tokens):
            return family
    return "Other NFL"


def _basketball_family(series: str, sport: str) -> str:
    checks = (
        (("GAME",), "Moneyline"),
        (("SPREAD",), "Spread"),
        (("TEAMTOTAL",), "Team Total"),
        (("TOTAL",), "Game Total"),
        (("1H",), "First Half"),
        (("2H",), "Second Half"),
        (("1Q",), "First Quarter"),
        (("2Q",), "Second Quarter"),
        (("3Q",), "Third Quarter"),
        (("4Q",), "Fourth Quarter"),
        (("PRA",), "Points + Rebounds + Assists"),
        (("PTS", "POINTS"), "Points"),
        (("REB",), "Rebounds"),
        (("AST",), "Assists"),
        (("3PT", "THREE"), "Three-Pointers"),
        (("STL", "STEAL"), "Steals"),
        (("BLK", "BLOCK"), "Blocks"),
        (("DOUBLEDOUBLE", "DBLDBL"), "Double Double"),
    )
    for tokens, family in checks:
        if _contains(series, *tokens):
            return family
    return f"Other {sport}"


def _tennis_family(series: str) -> str:
    checks = (
        (("ANYSET",), "Any Set Winner"),
        (("SETWINNER",), "Set Winner"),
        (("GWINNER",), "Game Winner"),
        (("ACES",), "Aces"),
        (("GTOTAL", "GAMESTOTAL"), "Games Total"),
        (("GSPREAD", "GAMESSPREAD"), "Games Spread"),
        (("TOTALSETS",), "Total Sets"),
        (("EXACTMATCH",), "Exact Match"),
        (("DOUBLES",), "Doubles Match Winner"),
        (("MATCH",), "Match Winner"),
    )
    for tokens, family in checks:
        if _contains(series, *tokens):
            return family
    return "Other Tennis"


def classify_kalshi_market(market: dict) -> tuple[str, str] | None:
    """Classify any supported current-event market without requiring a whitelist.

    Prefixes identify the sport; family parsing is best-effort. Unknown/new
    market families remain visible as Other <sport> instead of disappearing.
    """
    if looks_like_future(market):
        return None

    context_text = _text(market)
    if "table tennis" in context_text:
        return None

    series = _series(market)
    discovered_sport = str(market.get("sports_edge_sport") or "").strip()
    if discovered_sport in SUPPORTED_SPORTS:
        if discovered_sport == "MLB":
            return "MLB", _baseball_family(series)
        if discovered_sport == "NBA":
            return "NBA", _basketball_family(series, "NBA")
        if discovered_sport == "WNBA":
            return "WNBA", _basketball_family(series, "WNBA")
        if discovered_sport == "NFL":
            return "NFL", _football_family(series)
        if discovered_sport == "Tennis":
            return "Tennis", _tennis_family(series)

    if series.startswith("KXMLB"):
        return "MLB", _baseball_family(series)
    if series.startswith("KXNBA"):
        return "NBA", _basketball_family(series, "NBA")
    if series.startswith("KXWNBA"):
        return "WNBA", _basketball_family(series, "WNBA")
    if series.startswith("KXNFL"):
        return "NFL", _football_family(series)
    if series.startswith(("KXATP", "KXWTA", "KXITF")):
        return "Tennis", _tennis_family(series)

    # Metadata fallback for future ticker changes. This is intentionally
    # conservative and only applies when sport identity is explicit.
    text = _text(market)
    if any(x in text for x in (" major league baseball ", " mlb ", " pro baseball ")):
        return "MLB", "Other MLB"
    if any(x in text for x in (" national basketball association ", " nba ", " pro basketball ")):
        return "NBA", "Other NBA"
    if any(x in text for x in (" wnba ", " women's national basketball association ")):
        return "WNBA", "Other WNBA"
    if any(x in text for x in (" nfl ", " pro football ", " national football league ")):
        return "NFL", "Other NFL"
    if any(x in text for x in (" tennis ", " atp ", " wta ", " itf ", " challenger ")):
        return "Tennis", "Other Tennis"

    return None


def group_kalshi_sports(markets: Iterable[dict]) -> dict[str, list[KalshiSportMarket]]:
    grouped: dict[str, list[KalshiSportMarket]] = {sport: [] for sport in SUPPORTED_SPORTS}
    for market in markets:
        classification = classify_kalshi_market(market)
        if classification is None:
            continue
        sport, family = classification
        grouped[sport].append(KalshiSportMarket(sport=sport, family=family, market=market))

    for sport in grouped:
        grouped[sport].sort(
            key=lambda row: (
                float(row.market.get("volume_fp", row.market.get("volume", 0)) or 0),
                str(row.market.get("ticker") or ""),
            ),
            reverse=True,
        )
    return grouped


def catalog_diagnostics(markets: Iterable[dict]) -> CatalogDiagnostics:
    rows = list(markets)
    counts = {sport: 0 for sport in SUPPORTED_SPORTS}
    classified = 0
    unknown_prefixes: set[str] = set()

    for market in rows:
        classification = classify_kalshi_market(market)
        if classification is not None:
            classified += 1
            counts[classification[0]] += 1
            continue

        series = _series(market)
        if (
            not looks_like_future(market)
            and series.startswith(("KXMLB", "KXNBA", "KXWNBA", "KXNFL", "KXATP", "KXWTA", "KXITF"))
        ):
            unknown_prefixes.add(series)

    return CatalogDiagnostics(
        total_open_markets=len(rows),
        classified_markets=classified,
        unclassified_supported_prefixes=tuple(sorted(unknown_prefixes)),
        counts_by_sport=counts,
    )


def prop_families(sport: str) -> set[str]:
    if sport == "MLB":
        return {"Hits", "Hits + Runs + RBIs", "Home Runs", "Total Bases", "RBIs", "Strikeouts", "Other MLB"}
    if sport == "NFL":
        return {
            "Passing TDs", "Passing Yards", "Pass Attempts", "Pass Completions",
            "Pass Interceptions", "Rushing Yards", "Rush Attempts",
            "Rushing + Receiving Yards", "Receiving Yards", "Receptions",
            "Player Touchdowns", "First Touchdown", "Field Goals",
            "Other NFL",
        }
    if sport in {"NBA", "WNBA"}:
        return {
            "Points", "Rebounds", "Assists", "Three-Pointers",
            "Points + Rebounds + Assists", "Steals", "Blocks",
            "Double Double", f"Other {sport}",
        }
    if sport == "Tennis":
        return {
            "Match Winner", "Doubles Match Winner", "Set Winner", "Any Set Winner",
            "Game Winner", "Aces", "Games Total", "Games Spread", "Total Sets",
            "Exact Match", "Other Tennis",
        }
    return set()


def parlay_pool(
    grouped: dict[str, list[KalshiSportMarket]],
    sport: str,
    *,
    include_props: bool = True,
) -> list[KalshiSportMarket]:
    rows = list(grouped.get(sport, []))
    if include_props:
        return rows
    allowed = {"Moneyline", "Spread", "Game Total", "Match Winner"}
    return [row for row in rows if row.family in allowed]


@dataclass(frozen=True)
class KalshiSideCandidate:
    sport: str
    family: str
    ticker: str
    event_key: str
    event_title: str
    side: str
    selection: str
    price: float
    volume: float
    market: dict


def _selection(market: dict, side: str) -> str:
    side = side.upper()
    if side == "YES":
        return str(
            market.get("yes_sub_title")
            or market.get("yes_title")
            or market.get("yes_label")
            or market.get("title")
            or "YES"
        ).strip()
    return str(
        market.get("no_sub_title")
        or market.get("no_title")
        or market.get("no_label")
        or ("NO — " + str(market.get("title") or ""))
    ).strip()


def side_candidates(rows: Iterable[KalshiSportMarket]) -> list[KalshiSideCandidate]:
    out: list[KalshiSideCandidate] = []
    for row in rows:
        market = row.market
        ticker = str(market.get("ticker") or "").strip()
        if not ticker:
            continue
        event_key = str(market.get("event_ticker") or ticker.rsplit("-", 1)[0] or ticker)
        event_title = str(market.get("event_title") or market.get("title") or ticker).strip()
        try:
            volume = float(market.get("volume_fp", market.get("volume", 0)) or 0)
        except (TypeError, ValueError):
            volume = 0.0
        for side in ("YES", "NO"):
            price = market_side_probability(market, side)
            if price is None:
                continue
            out.append(
                KalshiSideCandidate(
                    sport=row.sport,
                    family=row.family,
                    ticker=ticker,
                    event_key=event_key,
                    event_title=event_title,
                    side=side,
                    selection=_selection(market, side),
                    price=price,
                    volume=volume,
                    market=market,
                )
            )
    return out


def choose_kalshi_ticket(
    candidates: list[KalshiSideCandidate],
    *,
    mode: str,
    target_legs: int,
    max_per_event: int = 1,
) -> list[KalshiSideCandidate]:
    """Build distinct Kalshi-first ticket profiles.

    Longshot is only a candidate profile until independent fair value exists;
    it must not be presented as positive EV merely because the price is low.
    """
    if mode == "longshot":
        pool = [candidate for candidate in candidates if 0.05 <= candidate.price <= 0.35]
        pool.sort(
            key=lambda candidate: (
                candidate.volume,
                -abs(candidate.price - 0.20),
            ),
            reverse=True,
        )
    else:
        pool = [candidate for candidate in candidates if 0.40 <= candidate.price <= 0.82]
        pool.sort(
            key=lambda candidate: (
                candidate.volume,
                candidate.price,
            ),
            reverse=True,
        )

    selected: list[KalshiSideCandidate] = []
    event_counts: dict[str, int] = {}
    used_tickers: set[str] = set()
    for candidate in pool:
        if candidate.ticker in used_tickers:
            continue
        if event_counts.get(candidate.event_key, 0) >= max_per_event:
            continue
        selected.append(candidate)
        used_tickers.add(candidate.ticker)
        event_counts[candidate.event_key] = event_counts.get(candidate.event_key, 0) + 1
        if len(selected) >= target_legs:
            break
    return selected
