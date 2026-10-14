from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sports_edge.models.game_scope import looks_like_future, normalize
from sports_edge.models.live_board import market_side_probability


SUPPORTED_SPORTS: tuple[str, ...] = ("MLB", "NBA", "WNBA", "NFL", "Tennis", "NHL", "NCAAF", "Soccer")


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
        # Period-specific series must be classified before generic GAME/TOTAL.
        (("F3",), "First 3 Innings"),
        (("F5",), "First 5 Innings"),
        (("F7",), "First 7 Innings"),
        (("GAME",), "Moneyline"),
        (("SPREAD",), "Spread"),
        (("TEAMTOTAL",), "Team Total"),
        (("TB", "TOTALBASE"), "Total Bases"),
        (("TOTAL",), "Game Total"),
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
        # Period-specific series can contain GAME/SPREAD/TOTAL in their names.
        # Resolve the period first so a 1H total cannot become a full-game total.
        (("1H",), "First Half"),
        (("2H",), "Second Half"),
        (("1Q",), "First Quarter"),
        (("2Q",), "Second Quarter"),
        (("3Q",), "Third Quarter"),
        (("4Q",), "Fourth Quarter"),
        (("GAME",), "Moneyline"),
        (("SPREAD",), "Spread"),
        (("TEAMTOTAL",), "Team Total"),
        (("TOTAL",), "Game Total"),
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
        # Period-specific series must win over generic GAME/SPREAD/TOTAL tokens.
        (("1H",), "First Half"),
        (("2H",), "Second Half"),
        (("1Q",), "First Quarter"),
        (("2Q",), "Second Quarter"),
        (("3Q",), "Third Quarter"),
        (("4Q",), "Fourth Quarter"),
        (("GAME",), "Moneyline"),
        (("SPREAD",), "Spread"),
        (("TEAMTOTAL",), "Team Total"),
        (("TOTAL",), "Game Total"),
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


def _hockey_family(series: str) -> str:
    for tokens, family in (
        (("SAVES", "KXNHLSAVE"), "Goalie Saves"),
        (("PLAYERASSISTS", "KXNHLAST"), "Player Assists"),
        (("PLAYERGOALS", "GOALSCORER", "ANYTIMEGOAL", "KXNHLGOAL", "KXNHLANYGOAL"), "Player Goals"),
        (("PLAYERSHOTS", "SHOTSONGOAL", "KXNHLSOG"), "Shots on Goal"),
        (("PLAYERPOINTS", "KXNHLPTS"), "Player Points"),
        (("TEAMTOTAL",), "Team Total"),
        (("SPREAD", "PUCKLINE"), "Puck Line"),
        (("TOTAL",), "Game Total"),
        (("GAME", "WINNER"), "Moneyline"),
    ):
        if _contains(series, *tokens):
            return family
    return "Other NHL"


def _soccer_family(series: str) -> str:
    for tokens, family in (
        (("ANYTIMEGOAL", "GOALSCORER", "PLAYERGOALS"), "Player Goals"),
        (("BTTS", "BOTHSCOR"), "Both Teams Score"),
        (("TEAMTOTAL",), "Team Total"),
        (("SPREAD", "HANDICAP"), "Handicap"),
        (("TOTAL",), "Game Total"),
        (("GAME", "MATCH", "WINNER"), "Match Result"),
    ):
        if _contains(series, *tokens):
            return family
    return "Other Soccer"


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
        if discovered_sport == "NHL":
            return "NHL", _hockey_family(series)
        if discovered_sport == "NCAAF":
            return "NCAAF", _football_family(series)
        if discovered_sport == "Soccer":
            return "Soccer", _soccer_family(series)

    if series.startswith("KXNHL"):
        return "NHL", _hockey_family(series)
    if series.startswith(("KXNCAAF", "KXCFB", "KXCOLLEGEFOOTBALL")):
        return "NCAAF", _football_family(series)
    if series.startswith(("KXSOCCER", "KXEPL", "KXMLS", "KXUCL", "KXLALIGA", "KXSERIEA")):
        return "Soccer", _soccer_family(series)
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
    if any(x in text for x in (" nhl ", " national hockey league ")):
        return "NHL", "Other NHL"
    if any(x in text for x in (" ncaaf ", " college football ", " ncaa football ")):
        return "NCAAF", "Other NCAAF"
    if any(x in text for x in (" soccer ", " premier league ", " champions league ", " mls ")):
        return "Soccer", "Other Soccer"
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
            and series.startswith(("KXMLB", "KXNBA", "KXWNBA", "KXNFL", "KXATP", "KXWTA", "KXITF", "KXNHL", "KXNCAAF", "KXCFB", "KXSOCCER", "KXEPL", "KXMLS", "KXUCL"))
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
    if sport == "NHL":
        return {"Player Goals", "Shots on Goal", "Player Points", "Player Assists", "Goalie Saves", "Other NHL"}
    if sport == "NCAAF":
        return {"Passing Yards", "Rushing Yards", "Receiving Yards", "Player Touchdowns", "Other NFL"}
    if sport == "Soccer":
        return {"Player Goals", "Other Soccer"}
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
    verified_value: dict[tuple[str, str], object] | None = None,
) -> list[KalshiSideCandidate]:
    """Only select independently qualified value, never price/volume proxies.

    Callers without verified_value intentionally receive an empty ticket:
    discovery-only markets must not be presented as recommended legs.
    """
    if target_legs <= 0 or max_per_event <= 0 or not verified_value:
        return []
    eligible: list[tuple[float, KalshiSideCandidate]] = []
    for candidate in candidates:
        audit = verified_value.get((candidate.ticker, candidate.side))
        if audit is None or getattr(audit, "status", None) != "QUALIFIED_VALUE":
            continue
        executable = getattr(audit, "executable_price", None)
        edge = getattr(audit, "edge", None)
        if executable is None or edge is None or edge <= 0:
            continue
        if abs(executable - candidate.price) > 0.000001:
            continue  # reject stale or different quote snapshots
        eligible.append((edge, candidate))
    eligible.sort(key=lambda item: item[0], reverse=True)
    selected: list[KalshiSideCandidate] = []
    event_counts: dict[str, int] = {}
    used_tickers: set[str] = set()
    for _, candidate in eligible:
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
