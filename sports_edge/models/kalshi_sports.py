from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from sports_edge.models.game_scope import looks_like_future, normalize
from sports_edge.models.live_board import market_side_probability


@dataclass(frozen=True)
class KalshiSportMarket:
    sport: str
    family: str
    market: dict


# Known live/game/prop prefixes. Prefix matching intentionally excludes
# championship/futures families via looks_like_future() before classification.
MLB_PREFIXES: tuple[tuple[str, str], ...] = (
    ("KXMLBGAME", "Moneyline"),
    ("KXMLBSPREAD", "Spread"),
    ("KXMLBTOTAL", "Game Total"),
    ("KXMLBF3", "First 3 Innings"),
    ("KXMLBF5", "First 5 Innings"),
    ("KXMLBF7", "First 7 Innings"),
    ("KXMLBHIT", "Hits"),
    ("KXMLBHRR", "Home Runs"),
    ("KXMLBHR", "Home Runs"),
    ("KXMLBTB", "Total Bases"),
    ("KXMLBRBI", "RBIs"),
    ("KXMLBTEAMTOTAL", "Team Total"),
    ("KXMLBRFI", "First Inning Run"),
    ("KXMLBEXTRAS", "Extra Innings"),
)

NFL_PREFIXES: tuple[tuple[str, str], ...] = (
    ("KXNFLGAME", "Moneyline"),
    ("KXNFLSPREAD", "Spread"),
    ("KXNFLTOTAL", "Game Total"),
    ("KXNFL1H", "First Half"),
    ("KXNFL2H", "Second Half"),
    ("KXNFL1Q", "First Quarter"),
    ("KXNFL2Q", "Second Quarter"),
    ("KXNFL3Q", "Third Quarter"),
    ("KXNFL4Q", "Fourth Quarter"),
    ("KXNFLOT", "Overtime"),
    ("KXNFLPASSTDS", "Passing TDs"),
    ("KXNFLPASSYDS", "Passing Yards"),
    ("KXNFLTD", "Player Touchdowns"),
    ("KXNFLTOTALTD", "Total TDs"),
    ("KXNFLFIRSTTDTEAM", "First TD Team"),
    ("KXNFLTEAMFIRSTTD", "Team First TD"),
    ("KXNFLTEAMTD", "Team TDs"),
    ("KXNFLFG", "Field Goals"),
    ("KXNFLWINMARGIN", "Win Margin"),
)

TENNIS_SERIES: tuple[str, ...] = (
    "KXATPMATCH",
    "KXATPCHALLENGERMATCH",
    "KXATPDOUBLES",
    "KXATPSETWINNER",
    "KXATPGTOTAL",
    "KXWTAMATCH",
    "KXWTACHALLENGERMATCH",
    "KXWTADOUBLES",
    "KXWTASETWINNER",
    "KXITFMATCH",
    "KXITFDOUBLES",
    "KXITFWMATCH",
    "KXITFWDOUBLES",
)

TENNIS_PREFIXES: tuple[tuple[str, str], ...] = (
    ("KXATPCHALLENGERMATCH", "Match Winner"),
    ("KXWTACHALLENGERMATCH", "Match Winner"),
    ("KXATPDOUBLES", "Doubles Match Winner"),
    ("KXWTADOUBLES", "Doubles Match Winner"),
    ("KXITFDOUBLES", "Doubles Match Winner"),
    ("KXITFWDOUBLES", "Doubles Match Winner"),
    ("KXITFWMATCH", "Match Winner"),
    ("KXITFMATCH", "Match Winner"),
    ("KXATPMATCH", "Match Winner"),
    ("KXWTAMATCH", "Match Winner"),
    ("KXATPSETWINNER", "Set Winner"),
    ("KXWTASETWINNER", "Set Winner"),
    ("KXITFSETWINNER", "Set Winner"),
    ("KXITFWSETWINNER", "Set Winner"),
    ("KXATPGTOTAL", "Games Total"),
    ("KXWTAGTOTAL", "Games Total"),
    ("KXITFGTOTAL", "Games Total"),
    ("KXITFWGTOTAL", "Games Total"),
    ("KXATPGSPREAD", "Games Spread"),
    ("KXWTAGSPREAD", "Games Spread"),
    ("KXITFGSPREAD", "Games Spread"),
    ("KXITFWGSPREAD", "Games Spread"),
    ("KXATPTOTALSETS", "Total Sets"),
    ("KXWTATOTALSETS", "Total Sets"),
    ("KXITFTOTALSETS", "Total Sets"),
    ("KXITFWTOTALSETS", "Total Sets"),
    ("KXATPEXACTMATCH", "Exact Match"),
    ("KXWTAEXACTMATCH", "Exact Match"),
    ("KXITFEXACTMATCH", "Exact Match"),
    ("KXITFWEXACTMATCH", "Exact Match"),
)


ALL_PREFIXES = {
    "MLB": MLB_PREFIXES,
    "NFL": NFL_PREFIXES,
    "Tennis": TENNIS_PREFIXES,
}


def _series(market: dict) -> str:
    value = market.get("series_ticker")
    if value:
        return str(value).upper().strip()
    ticker = str(market.get("ticker") or "").upper().strip()
    # Most Kalshi market tickers start with their series ticker.
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
            )
        )
    )


def _tennis_family_from_series(series: str) -> str | None:
    """Infer direct tennis market family from any ATP/WTA/ITF series ticker."""
    if not series.startswith(("KXATP", "KXWTA", "KXITF")):
        return None
    if "SETWINNER" in series:
        return "Set Winner"
    if "GTOTAL" in series or "GAMESTOTAL" in series:
        return "Games Total"
    if "GSPREAD" in series or "GAMESSPREAD" in series:
        return "Games Spread"
    if "TOTALSETS" in series:
        return "Total Sets"
    if "EXACTMATCH" in series:
        return "Exact Match"
    if "DOUBLES" in series:
        return "Doubles Match Winner"
    if "MATCH" in series:
        return "Match Winner"
    # Keep newly introduced direct tennis series visible instead of silently
    # dropping them. They remain labelled Other Tennis until explicitly mapped.
    return "Other Tennis"


def classify_kalshi_market(market: dict) -> tuple[str, str] | None:
    if looks_like_future(market):
        return None

    series = _series(market)

    tennis_family = _tennis_family_from_series(series)
    if tennis_family is not None:
        return "Tennis", tennis_family

    for sport, prefixes in ALL_PREFIXES.items():
        for prefix, family in prefixes:
            if series.startswith(prefix):
                return sport, family

    # Conservative fallback for newly introduced series. Require explicit sport
    # text plus game/prop wording; do not classify generic "sports" markets.
    text = _text(market)
    if any(x in text for x in ("professional baseball", "pro baseball", "mlb")):
        if any(x in text for x in (" vs ", " hit", "home run", "strikeout", "total base", "rbi", "spread", "total")):
            return "MLB", "Other"
    if "nfl" in text or "pro football" in text:
        if any(x in text for x in (" vs ", "passing", "rushing", "receiving", "touchdown", "spread", "total")):
            return "NFL", "Other"
    if any(x in text for x in ("tennis", " atp ", " wta ", " itf ")):
        if any(x in text for x in (" vs ", "match", "set", "games total", "games spread")):
            return "Tennis", "Other"

    return None


def group_kalshi_sports(markets: Iterable[dict]) -> dict[str, list[KalshiSportMarket]]:
    grouped: dict[str, list[KalshiSportMarket]] = {"MLB": [], "NFL": [], "Tennis": []}
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


def prop_families(sport: str) -> set[str]:
    if sport == "MLB":
        return {"Hits", "Home Runs", "Total Bases", "RBIs"}
    if sport == "NFL":
        return {"Passing TDs", "Passing Yards", "Player Touchdowns", "Team TDs", "Field Goals"}
    if sport == "Tennis":
        return {
            "Match Winner",
            "Doubles Match Winner",
            "Set Winner",
            "Games Total",
            "Games Spread",
            "Total Sets",
            "Exact Match",
            "Other Tennis",
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
    """Build a distinct Kalshi-first ticket pool.

    Longshot mode deliberately targets 7-35c sides. Best mode avoids that band
    and favors stronger market probabilities. Neither mode claims price edge
    until sportsbook/model enrichment is available.
    """
    if mode == "longshot":
        pool = [candidate for candidate in candidates if 0.07 <= candidate.price <= 0.35]
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
