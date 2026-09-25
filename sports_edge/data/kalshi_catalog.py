from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading
import time
from typing import Any

from sports_edge.data.kalshi import KalshiPublicClient


SUPPORTED_PREFIXES: tuple[tuple[str, str], ...] = (
    ("KXMLB", "MLB"),
    ("KXNBA", "NBA"),
    ("KXWNBA", "WNBA"),
    ("KXNFL", "NFL"),
    ("KXATP", "Tennis"),
    ("KXWTA", "Tennis"),
    ("KXITF", "Tennis"),
)


# Production "ship-now" catalog: current game/period/player-stat series observed
# on Kalshi. All mode uses a compact overview; selecting one sport loads the
# full list below. Futures/awards/season markets are intentionally excluded.
PRODUCTION_SERIES_BY_SPORT: dict[str, tuple[str, ...]] = {
    "MLB": (
        "KXMLBGAME","KXMLBSPREAD","KXMLBTOTAL","KXMLBRFI","KXMLBHR",
        "KXMLBF5TOTAL","KXMLBF5","KXMLBKS","KXMLBTEAMTOTAL","KXMLBHIT",
        "KXMLBSTGAME","KXMLBF5SPREAD","KXMLBHRR","KXMLBTB","KXMLBF3",
        "KXMLBRBI","KXMLBF7","KXMLBEXTRAS","KXMLBINNINGWIN",
        "KXMLBINNINGTOTAL","KXMLBWA","KXMLBHA","KXMLBPITCH","KXMLBNEXTHR",
    ),
    "NBA": (
        "KXNBAGAME","KXNBASPREAD","KXNBATOTAL","KXNBASUMMERGAME","KXNBAPTS",
        "KXNBA1HSPREAD","KXNBA1HWINNER","KXNBA1HTOTAL","KXNBASUMMERTOTAL",
        "KXNBATEAMTOTAL","KXNBASUMMERSPREAD","KXNBA3PT","KXNBAREB","KXNBAAST",
        "KXNBA2HWINNER","KXNBA2D","KXNBA3D","KXNBA2HSPREAD","KXNBA2HTOTAL",
        "KXNBA1QWINNER","KXNBA1QSPREAD","KXNBASTL","KXNBABLK","KXNBA3QSPREAD",
        "KXNBA2QWINNER","KXNBA3QWINNER","KXNBA2QSPREAD","KXNBA4QWINNER",
        "KXNBA1QTOTAL","KXNBA3QTOTAL","KXNBA2QTOTAL","KXNBA4QSPREAD",
        "KXNBA4QTOTAL","KXNBAPLAYOFFPTS","KXNBAH2HBENCHPTS","KXNBAH2HPRA",
        "KXNBAH2HPTS","KXNBAH2H3PT","KXNBAH2HTEAM3PT","KXNBARACE",
        "KXNBAPRA","KXNBABENCHPTS","KXNBARA","KXNBASTOCKS","KXNBASTOCK",
        "KXNBAPR","KXNBAFIRSTBASKET","KXNBAWINMARGIN",
    ),
    "WNBA": (
        "KXWNBAGAME","KXWNBASPREAD","KXWNBATOTAL","KXWNBA1HTOTAL",
        "KXWNBA1HSPREAD","KXWNBAPTS","KXWNBA1HWINNER","KXWNBA1QTOTAL",
        "KXWNBA3QTOTAL","KXWNBA2QTOTAL","KXWNBA1QSPREAD","KXWNBAREB",
        "KXWNBA1QWINNER","KXWNBA4QTOTAL","KXWNBA3QSPREAD","KXWNBA3PT",
        "KXWNBA3QWINNER","KXWNBA2QWINNER","KXWNBAAST","KXWNBA2QSPREAD",
        "KXWNBATEAMTOTAL","KXWNBA4QWINNER","KXWNBA2HTOTAL",
        "KXWNBA2HSPREAD","KXWNBA4QSPREAD","KXWNBA2HWINNER","KXWNBA40PTS",
        "KXWNBAH2HPRA","KXWNBAH2HPTS",
    ),
    "NFL": (
        "KXNFLGAME","KXNFLSPREAD","KXNFLTOTAL","KXNFLTD","KXNFLANYTD",
        "KXNFLFIRSTTD","KXNFLRECYDS","KXNFLRSHYDS","KXNFLPASSYDS","KXNFL2TD",
        "KXNFL1HSPREAD","KXNFLREC","KXNFL1HTOTAL","KXNFL1H","KXNFLTEAMTOTAL",
        "KXNFLPASSTDS","KXNFL1HWINNER","KXNFL1Q","KXNFL2HSPREAD","KXNFL1QTOTAL",
        "KXNFL1QSPREAD","KXNFL2H","KXNFL2HTOTAL","KXNFL3QSPREAD","KXNFL3Q",
        "KXNFL3QTOTAL","KXNFLTEAMFIRSTTD","KXNFL4QSPREAD","KXNFL4Q",
        "KXNFL2QSPREAD","KXNFL2Q","KXNFL2QTOTAL","KXNFL1QWINNER",
        "KXNFL4QTOTAL","KXNFL1HTEAMTOTAL","KXNFLRRYDS","KXNFLRSHATT",
        "KXNFLFIRSTTDTEAM","KXNFLFFPTS","KXNFLFIRSTTDTIME","KXNFLRACE",
        "KXNFLDSTTD","KXNFLNEXTTD","KXNFLPASSINT","KXNFLPASSCOMP",
        "KXNFLLEADCHANGE","KXNFLTEAMTD","KXNFLPASSATT","KXNFL2HWINNER",
        "KXNFLFG","KXNFL2PTCONV","KXNFLTEAMSACK","KXNFLLARGELEAD",
        "KXNFLGAMETD","KXNFLTOTALTD","KXNFLGAMESACK","KXNFLLONGRSH",
        "KXNFLLONGREC","KXNFL1QBTTS","KXNFLBOTH","KXNFLSFTY",
        "KXNFLEQBTTS","KXNFL2QBTTS","KXNFL3QBTTS","KXNFL4QBTTS",
        "KXNFLNEXTINT","KXNFLLONGESTFG","KXNFLRSHYDSH2H","KXNFLPASSYDSH2H",
        "KXNFLRECYDSH2H","KXNFLTEAMTO","KXNFLGAMETO","KXNFLTEAM1STDOWNS",
        "KXNFLINT","KXNFLTEAMFG","KXNFL2HTD","KXNFL1HTD","KXNFLSACK",
    ),
    "Tennis": (
        "KXATPMATCH","KXATPCHALLENGERMATCH","KXWTAMATCH","KXITFMATCH",
        "KXITFWMATCH","KXWTACHALLENGERMATCH","KXATPSETWINNER",
        "KXWTASETWINNER","KXITFDOUBLES","KXATPEXACTMATCH","KXWTADOUBLES",
        "KXITFWDOUBLES","KXATPDOUBLES","KXATPGTOTAL","KXATPGSPREAD",
        "KXWTAEXACTMATCH","KXATPCHALLENGERDOUBLES","KXWTAGTOTAL",
        "KXATPTOTALSETS","KXATPSSPREAD","KXATPGAMETOTAL","KXATPACES",
        "KXATPS1GWINNER","KXATPS2GWINNER","KXATPS3GWINNER","KXATPS4GWINNER",
        "KXATPS5GWINNER","KXATPGAMESPREAD","KXATPTIEBREAK","KXATPGWINNER",
        "KXWTAACES","KXATPANYSET",
    ),
}

OVERVIEW_SERIES_BY_SPORT: dict[str, tuple[str, ...]] = {
    "MLB": ("KXMLBGAME","KXMLBSPREAD","KXMLBTOTAL"),
    "NBA": ("KXNBAGAME","KXNBASPREAD","KXNBATOTAL"),
    "WNBA": ("KXWNBAGAME","KXWNBASPREAD","KXWNBATOTAL"),
    "NFL": ("KXNFLGAME","KXNFLSPREAD","KXNFLTOTAL"),
    "Tennis": (
        "KXATPMATCH","KXATPCHALLENGERMATCH","KXWTAMATCH",
        "KXWTACHALLENGERMATCH","KXITFMATCH","KXITFWMATCH",
    ),
}


@dataclass(frozen=True)
class KalshiCatalogResult:
    markets: tuple[dict, ...]
    pages: int
    cursor_exhausted: bool
    max_latency_ms: float | None
    error: str | None
    discovered_series: int = 0
    relevant_series: int = 0
    incomplete_series: tuple[str, ...] = ()


def _series_text(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("ticker") or ""),
        str(row.get("title") or ""),
        str(row.get("category") or ""),
        " ".join(str(x) for x in row.get("categories", []) or []),
        " ".join(str(x) for x in row.get("tags", []) or []),
    ]
    return " ".join(parts).lower()


def _sport_for_series(row: dict[str, Any]) -> str | None:
    ticker = str(row.get("ticker") or "").upper().strip()
    text = " " + _series_text(row) + " "
    if " pickleball " in text or " table tennis " in text:
        return None
    for prefix, sport in SUPPORTED_PREFIXES:
        if ticker.startswith(prefix):
            return sport

    if " wnba " in text or " women's national basketball association " in text:
        return "WNBA"
    if " major league baseball " in text or " mlb " in text:
        return "MLB"
    if " national football league " in text or " nfl " in text:
        return "NFL"
    if (
        " national basketball association " in text
        or (" nba " in text and " wnba " not in text)
    ):
        return "NBA"
    if any(token in text for token in (" tennis ", " atp ", " wta ", " itf ", " challenger ")):
        return "Tennis"
    return None


def _series_is_actionable(row: dict[str, Any], sport: str) -> bool:
    """Keep actual game/match and player/game-stat templates; reject futures."""
    ticker = str(row.get("ticker") or "").upper().strip()
    title = str(row.get("title") or "").lower().strip()
    text = f"{ticker} {title}"

    reject_terms = (
        "season", "award", "mvp", "rookie", "draft", "division",
        "conference", "playoff", "series", "next team", "next club",
        "retir", "hall of fame", "record", "leader", "qualifier",
        "viewership", "stadium", "manager", "coach", "governor",
        "apology", "all-star", "all star", "derby", "ranked", "ranking",
        "tournament winner", "finals champion", "finals winner",
        "win total", "exact wins", "most wins", "best record",
        "worst record", "stage of", "round of elimination",
        "player return", "return by date", "nationality of",
        "field winner", "future",
    )
    if any(term in text.lower() for term in reject_terms):
        return False

    if sport == "Tennis":
        if "table tennis" in title:
            return False
        tokens = (
            "MATCH", "DOUBLES", "SETWINNER", "ANYSET", "GWINNER",
            "GAME", "GSPREAD", "GAMESSPREAD", "SSPREAD", "GTOTAL",
            "GAMETOTAL", "TOTALSETS", "EXACTMATCH", "EXACTSETS",
            "TIEBREAK", "ACES", "FAULT", "SERVE", "BREAK",
        )
        title_terms = (
            "match", "doubles", "set winner", "game winner", "game spread",
            "set spread", "total games", "total sets", "exact match",
            "tiebreak", "aces", "double fault", "serve", "break point",
        )
        return any(token in ticker for token in tokens) or any(term in title for term in title_terms)

    if sport == "MLB":
        tokens = (
            "GAME", "SPREAD", "TEAMTOTAL", "TOTAL", "F1", "F2", "F3",
            "F4", "F5", "F6", "F7", "F8", "F9", "INNING", "HIT",
            "HRR", "RBI", "TOTALBASE", "OUTS", "RFI", "EXTRA",
            "PITCH", "NEXTHR", "HITSALLOWED", "EARNEDRUN",
        )
        exact = {"KXMLBHR", "KXMLBTB", "KXMLBKS", "KXMLBWA"}
        title_terms = (
            " game", "spread", "total", "inning", "player hits", "home runs",
            "hits runs rbis", "total bases", "rbis", "strikeouts", "walks",
            "outs recorded", "earned runs", "hits allowed", "next homerun",
            "player to pitch",
        )
        return (
            ticker in exact
            or any(token in ticker for token in tokens)
            or any(term in title for term in title_terms)
        )

    if sport in {"NBA", "WNBA"}:
        tokens = (
            "GAME", "SPREAD", "TEAMTOTAL", "TOTAL", "1Q", "2Q", "3Q",
            "4Q", "1H", "2H", "PTS", "REB", "AST", "3PT", "PRA",
            "BLK", "STL", "FIRSTBASKET", "RACE", "WINMARGIN",
            "BENCHPTS", "H2H", "DOUBLEDOUBLE",
        )
        exact_suffixes = ("2D",)
        title_terms = (
            " game", "spread", "total", "quarter", "half", "player points",
            "rebounds", "assists", "threes", "three-pointers", "blocks",
            "steals", "points + rebounds", "first basket", "race to points",
            "bench points", "head-to-head", "double double", "team total",
        )
        return (
            any(token in ticker for token in tokens)
            or any(ticker.endswith(x) for x in exact_suffixes)
            or any(term in title for term in title_terms)
        )

    if sport == "NFL":
        if " most " in f" {title} " or " weekly " in f" {title} ":
            return False
        tokens = (
            "GAME", "SPREAD", "TEAMTOTAL", "TOTAL", "1Q", "2Q", "3Q",
            "4Q", "1H", "2H", "PASS", "RSH", "RUSH", "REC", "TD",
            "FG", "SACK", "INT", "SAFETY", "SFTY", "TURNOVER",
            "2PT", "FIRSTDOWN", "LEAD", "BTTS", "FFPTS", "LADDER",
            "ESCALATOR", "NEXTTD", "NEXTINT", "RRYDS", "LONGREC",
            "LONGFG",
        )
        title_terms = (
            " game", "spread", "total", "quarter", "half", "passing",
            "rushing", "receiving", "receptions", "touchdown", "field goal",
            "sacks", "interception", "safety", "turnovers", "2-point",
            "first downs", "lead changes", "both teams to score",
            "fantasy points", "team total",
        )
        return any(token in ticker for token in tokens) or any(term in title for term in title_terms)

    return False


def _status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    try:
        return int(getattr(response, "status_code", None))
    except (TypeError, ValueError):
        return None


class _RequestPacer:
    """Shared start-rate limiter for concurrent Kalshi reads."""
    def __init__(self, min_interval_s: float):
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.min_interval_s
        if sleep_for > 0:
            time.sleep(sleep_for)


def _with_backoff(
    call,
    *,
    attempts: int = 8,
    base_delay_s: float = 0.45,
    pacer: _RequestPacer | None = None,
):
    delay = max(0.05, float(base_delay_s))
    last: Exception | None = None
    for attempt in range(max(1, attempts)):
        if pacer is not None:
            pacer.wait()
        try:
            return call()
        except Exception as exc:
            last = exc
            status = _status_code(exc)
            retryable = status == 429 or (status is not None and 500 <= status < 600)
            if not retryable or attempt >= attempts - 1:
                raise
            time.sleep(delay)
            delay = min(6.0, delay * 2.0)
    if last is not None:
        raise last
    raise RuntimeError("request failed")


def _fetch_one_series(
    series_row: dict[str, Any],
    sport: str,
    *,
    client: KalshiPublicClient | None,
    page_limit_per_series: int,
    page_size: int,
    request_pause_s: float,
    pacer: _RequestPacer | None,
) -> tuple[list[dict], int, list[float], str | None, bool]:
    worker = client or KalshiPublicClient()
    series_ticker = str(series_row.get("ticker") or "").strip()
    cursor: str | None = None
    last_cursor: str | None = None
    pages = 0
    latencies: list[float] = []
    rows: list[dict] = []

    for _ in range(max(1, int(page_limit_per_series))):
        try:
            response = _with_backoff(
                lambda st=series_ticker, cur=cursor: worker.markets(
                    status="open",
                    series_ticker=st,
                    limit=min(1000, max(1, int(page_size))),
                    cursor=cur,
                ),
                pacer=pacer,
            )
        except Exception as exc:
            if _status_code(exc) == 404:
                return rows, pages, latencies, None, True
            return rows, pages, latencies, f"{series_ticker}: {str(exc)[:180]}", False

        pages += 1
        if response.latency_ms is not None:
            latencies.append(float(response.latency_ms))
        market_payload = response.data if isinstance(response.data, dict) else {}

        for raw in market_payload.get("markets", []) or []:
            if not isinstance(raw, dict):
                continue
            ticker = str(raw.get("ticker") or "").strip()
            if not ticker:
                continue
            enriched = dict(raw)
            enriched.setdefault("series_ticker", series_ticker)
            enriched["series_title"] = series_row.get("title")
            enriched["series_category"] = series_row.get("category")
            enriched["series_tags"] = series_row.get("tags") or []
            enriched["sports_edge_sport"] = sport
            rows.append(enriched)

        next_cursor = market_payload.get("cursor")
        if not next_cursor:
            return rows, pages, latencies, None, True

        next_cursor = str(next_cursor)
        if next_cursor == cursor or next_cursor == last_cursor:
            return rows, pages, latencies, f"{series_ticker}: repeated pagination cursor", False

        last_cursor = cursor
        cursor = next_cursor
        if request_pause_s > 0:
            time.sleep(request_pause_s)

    return rows, pages, latencies, f"{series_ticker}: page safety limit reached", False


def fetch_supported_sport_catalog(
    client: KalshiPublicClient | None = None,
    *,
    page_limit_per_series: int = 50,
    page_size: int = 200,
    request_pause_s: float = 0.0,
    max_workers: int = 6,
    request_interval_s: float = 0.15,
    sports: tuple[str, ...] | list[str] | None = None,
    overview_only: bool = False,
) -> KalshiCatalogResult:
    """Discover supported sports from Kalshi /series, then fetch open markets.

    Production uses a small worker pool with separate HTTP clients. A provided
    client (tests/custom integrations) stays sequential to preserve deterministic
    behavior. Every series still exhausts its own cursor or is marked incomplete.
    """
    discovery_client = client or KalshiPublicClient()
    pacer = None if client is not None else _RequestPacer(request_interval_s)
    found: dict[str, dict] = {}
    latencies: list[float] = []
    errors: list[str] = []
    incomplete: list[str] = []
    pages = 0

    if client is None:
        # Production uses the bounded game/prop catalog below; no giant /series
        # payload is needed on every Streamlit cache miss.
        all_series: list[dict] = []
    else:
        try:
            series_response = _with_backoff(
                lambda: discovery_client.series_list(include_volume=True),
                pacer=pacer,
            )
            if series_response.latency_ms is not None:
                latencies.append(float(series_response.latency_ms))
            payload = series_response.data if isinstance(series_response.data, dict) else {}
            all_series = [row for row in payload.get("series", []) or [] if isinstance(row, dict)]
        except Exception as exc:
            return KalshiCatalogResult(
                markets=(),
                pages=0,
                cursor_exhausted=False,
                max_latency_ms=max(latencies) if latencies else None,
                error=f"Kalshi series discovery failed: {str(exc)[:240]}",
            )

    selected_sports = tuple(sports) if sports else tuple(PRODUCTION_SERIES_BY_SPORT)

    if client is None:
        # Production path: bounded live game/prop families. This removes the
        # hundreds of stale/future series requests that caused Kalshi 429s.
        source = OVERVIEW_SERIES_BY_SPORT if overview_only else PRODUCTION_SERIES_BY_SPORT
        relevant = [
            ({"ticker": ticker, "title": ticker}, sport)
            for sport in selected_sports
            if sport in source
            for ticker in source[sport]
        ]
    else:
        # Deterministic/test path retains metadata discovery behavior.
        relevant: list[tuple[dict, str]] = []
        for row in all_series:
            sport = _sport_for_series(row)
            if (
                sport is not None
                and sport in selected_sports
                and row.get("ticker")
                and _series_is_actionable(row, sport)
            ):
                relevant.append((row, sport))

    # A supplied client may be a deterministic fake or a caller-owned session;
    # keep that path sequential. Production creates one client per worker task.
    workers = 1 if client is not None else max(1, min(int(max_workers), 8, len(relevant) or 1))

    def run(item: tuple[dict, str]):
        series_row, sport = item
        return series_row, _fetch_one_series(
            series_row,
            sport,
            client=client,
            page_limit_per_series=page_limit_per_series,
            page_size=page_size,
            request_pause_s=request_pause_s,
            pacer=pacer,
        )

    if workers == 1:
        completed = [run(item) for item in relevant]
    else:
        completed = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(run, item): item for item in relevant}
            for future in as_completed(future_map):
                completed.append(future.result())

    for series_row, result in completed:
        rows, series_pages, series_latencies, error, complete = result
        pages += series_pages
        latencies.extend(series_latencies)
        for row in rows:
            ticker = str(row.get("ticker") or "").strip()
            if ticker:
                found[ticker] = row
        if error:
            errors.append(error)
        if not complete:
            incomplete.append(str(series_row.get("ticker") or ""))

    error = " | ".join(errors[:4]) if errors else None
    return KalshiCatalogResult(
        markets=tuple(found.values()),
        pages=pages,
        cursor_exhausted=not incomplete and not errors,
        max_latency_ms=max(latencies) if latencies else None,
        error=error,
        discovered_series=len(all_series),
        relevant_series=len(relevant),
        incomplete_series=tuple(sorted(x for x in incomplete if x)),
    )


def fetch_current_sport_catalog(
    client: KalshiPublicClient | None = None,
    *,
    past_hours: float = 12.0,
    future_hours: float = 168.0,
    page_limit: int = 60,
    page_size: int = 1000,
    now_ts: int | None = None,
) -> KalshiCatalogResult:
    """Fetch only current/live/upcoming regular sport contracts.

    Kalshi close-time filters cannot be combined with status=open, so this
    requests a bounded close-time window with no status filter and retains
    open markets locally. Futures that happen to close inside the window are
    removed later by the sport classifier's future gate.
    """
    worker = client or KalshiPublicClient()
    now = int(time.time()) if now_ts is None else int(now_ts)
    min_close = now - int(max(0.0, past_hours) * 3600)
    max_close = now + int(max(1.0, future_hours) * 3600)

    found: dict[str, dict] = {}
    latencies: list[float] = []
    cursor: str | None = None
    last_cursor: str | None = None
    pages = 0
    series_seen: set[str] = set()

    try:
        for _ in range(max(1, int(page_limit))):
            response = _with_backoff(
                lambda cur=cursor: worker.markets(
                    status=None,
                    limit=min(1000, max(1, int(page_size))),
                    cursor=cur,
                    min_close_ts=min_close,
                    max_close_ts=max_close,
                    mve_filter="exclude",
                )
            )
            pages += 1
            if response.latency_ms is not None:
                latencies.append(float(response.latency_ms))
            payload = response.data if isinstance(response.data, dict) else {}

            for raw in payload.get("markets", []) or []:
                if not isinstance(raw, dict):
                    continue
                if str(raw.get("status") or "").lower() != "open":
                    continue
                ticker = str(raw.get("ticker") or "").strip()
                if not ticker:
                    continue
                upper = ticker.upper()
                sport = None
                for prefix, candidate_sport in SUPPORTED_PREFIXES:
                    if upper.startswith(prefix):
                        sport = candidate_sport
                        break
                if sport is None:
                    continue

                enriched = dict(raw)
                series_ticker = upper.split("-", 1)[0]
                enriched.setdefault("series_ticker", series_ticker)
                enriched["sports_edge_sport"] = sport
                found[ticker] = enriched
                series_seen.add(series_ticker)

            next_cursor = payload.get("cursor")
            if not next_cursor:
                return KalshiCatalogResult(
                    markets=tuple(found.values()),
                    pages=pages,
                    cursor_exhausted=True,
                    max_latency_ms=max(latencies) if latencies else None,
                    error=None,
                    discovered_series=0,
                    relevant_series=len(series_seen),
                    incomplete_series=(),
                )

            next_cursor = str(next_cursor)
            if next_cursor == cursor or next_cursor == last_cursor:
                return KalshiCatalogResult(
                    markets=tuple(found.values()),
                    pages=pages,
                    cursor_exhausted=False,
                    max_latency_ms=max(latencies) if latencies else None,
                    error="Kalshi current-market cursor repeated before exhaustion",
                    discovered_series=0,
                    relevant_series=len(series_seen),
                    incomplete_series=("CURRENT_WINDOW",),
                )
            last_cursor = cursor
            cursor = next_cursor

        return KalshiCatalogResult(
            markets=tuple(found.values()),
            pages=pages,
            cursor_exhausted=False,
            max_latency_ms=max(latencies) if latencies else None,
            error=f"Kalshi current-market safety limit reached after {pages} pages",
            discovered_series=0,
            relevant_series=len(series_seen),
            incomplete_series=("CURRENT_WINDOW",),
        )
    except Exception as exc:
        return KalshiCatalogResult(
            markets=tuple(found.values()),
            pages=pages,
            cursor_exhausted=False,
            max_latency_ms=max(latencies) if latencies else None,
            error=f"Kalshi current-market fetch failed: {str(exc)[:240]}",
            discovered_series=0,
            relevant_series=len(series_seen),
            incomplete_series=("CURRENT_WINDOW",),
        )
