from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
    for prefix, sport in SUPPORTED_PREFIXES:
        if ticker.startswith(prefix):
            return sport

    text = " " + _series_text(row) + " "
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


def _status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    try:
        return int(getattr(response, "status_code", None))
    except (TypeError, ValueError):
        return None


def _with_backoff(call, *, attempts: int = 6, base_delay_s: float = 0.35):
    delay = max(0.05, float(base_delay_s))
    last: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return call()
        except Exception as exc:
            last = exc
            status = _status_code(exc)
            retryable = status == 429 or (status is not None and 500 <= status < 600)
            if not retryable or attempt >= attempts - 1:
                raise
            time.sleep(delay)
            delay = min(4.0, delay * 2.0)
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
                    limit=min(200, max(1, int(page_size))),
                    cursor=cur,
                )
            )
        except Exception as exc:
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
    request_pause_s: float = 0.05,
    max_workers: int = 4,
) -> KalshiCatalogResult:
    """Discover supported sports from Kalshi /series, then fetch open markets.

    Production uses a small worker pool with separate HTTP clients. A provided
    client (tests/custom integrations) stays sequential to preserve deterministic
    behavior. Every series still exhausts its own cursor or is marked incomplete.
    """
    discovery_client = client or KalshiPublicClient()
    found: dict[str, dict] = {}
    latencies: list[float] = []
    errors: list[str] = []
    incomplete: list[str] = []
    pages = 0

    try:
        series_response = _with_backoff(lambda: discovery_client.series_list(include_volume=True))
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

    relevant: list[tuple[dict, str]] = []
    for row in all_series:
        sport = _sport_for_series(row)
        if sport is not None and row.get("ticker"):
            relevant.append((row, sport))

    # A supplied client may be a deterministic fake or a caller-owned session;
    # keep that path sequential. Production creates one client per worker task.
    workers = 1 if client is not None else max(1, min(int(max_workers), 6, len(relevant) or 1))

    def run(item: tuple[dict, str]):
        series_row, sport = item
        return series_row, _fetch_one_series(
            series_row,
            sport,
            client=client,
            page_limit_per_series=page_limit_per_series,
            page_size=page_size,
            request_pause_s=request_pause_s,
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
