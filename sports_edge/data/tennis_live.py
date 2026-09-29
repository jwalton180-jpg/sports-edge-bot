from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import time

from sports_edge.data.kalshi import KalshiPublicClient


TENNIS_MATCH_SERIES = (
    "KXATPMATCH",
    "KXATPCHALLENGERMATCH",
    "KXWTAMATCH",
    "KXWTACHALLENGERMATCH",
    "KXITFMATCH",
    "KXITFWMATCH",
)


def _payload(result):
    return result.data if hasattr(result, "data") else result


def fetch_open_tennis_match_markets(
    *,
    max_workers: int = 6,
) -> tuple[dict, ...]:
    """Fetch only current open Tennis match-winner markets.

    This is intentionally much smaller than the full Tennis catalog so the
    live-reversal scanner can refresh frequently without reloading set/game
    derivative markets.
    """
    def one(series: str) -> list[dict]:
        client = KalshiPublicClient()
        payload = _payload(client.markets(
            status="open",
            series_ticker=series,
            limit=1000,
        ))
        return list((payload or {}).get("markets", []) or [])

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), len(TENNIS_MATCH_SERIES)))) as pool:
        futures = {pool.submit(one, series): series for series in TENNIS_MATCH_SERIES}
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception:
                # Live radar is fail-soft across series. The UI reports coverage
                # from the returned rows rather than inventing missing markets.
                continue

    dedup = {str(row.get("ticker") or ""): row for row in rows if row.get("ticker")}
    return tuple(dedup.values())


def fetch_tennis_candle_history(
    tickers: list[str] | tuple[str, ...],
    *,
    lookback_minutes: int = 60,
    period_interval: int = 1,
    now: datetime | None = None,
    max_workers: int = 5,
) -> dict[str, tuple[dict, ...]]:
    """Fetch executable price history in public Kalshi batches.

    Kalshi accepts up to 100 tickers per batch request. Chunks are kept under
    that limit and independently fail closed.
    """
    unique = list(dict.fromkeys(str(x).strip() for x in tickers if str(x).strip()))
    if not unique:
        return {}

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    end_ts = int(now.timestamp())
    start_ts = end_ts - max(5, int(lookback_minutes)) * 60
    chunks = [unique[i:i + 100] for i in range(0, len(unique), 100)]

    def one(chunk: list[str]) -> dict[str, tuple[dict, ...]]:
        client = KalshiPublicClient()
        result = client.batch_market_candlesticks(
            chunk,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=max(1, int(period_interval)),
            include_latest_before_start=True,
        )
        payload = _payload(result)
        out: dict[str, tuple[dict, ...]] = {}
        for item in (payload or {}).get("markets", []) or []:
            ticker = str(item.get("market_ticker") or item.get("ticker") or "").strip()
            if not ticker:
                continue
            candles = tuple(item.get("candlesticks", []) or [])
            out[ticker] = candles
        return out

    out: dict[str, tuple[dict, ...]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), len(chunks)))) as pool:
        futures = [pool.submit(one, chunk) for chunk in chunks]
        for future in as_completed(futures):
            try:
                out.update(future.result())
            except Exception:
                continue
    return out
