from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sports_edge.data.kalshi import KalshiPublicClient


@dataclass(frozen=True)
class KalshiCatalogResult:
    markets: tuple[dict, ...]
    pages: int
    cursor_exhausted: bool
    max_latency_ms: float | None
    error: str | None


def fetch_open_market_catalog(
    client: KalshiPublicClient | None = None,
    *,
    page_limit: int = 250,
    page_size: int = 200,
) -> KalshiCatalogResult:
    """Fetch the complete open-market catalog using Kalshi cursor pagination.

    A page_limit is only an outage/safety guard. Reaching it while a cursor is
    still present marks the result incomplete instead of silently treating it
    as "all markets".
    """
    client = client or KalshiPublicClient()
    found: dict[str, dict] = {}
    latencies: list[float] = []
    cursor: str | None = None
    pages = 0
    last_cursor: str | None = None

    try:
        for _ in range(max(1, int(page_limit))):
            response = client.markets(
                status="open",
                limit=min(200, max(1, int(page_size))),
                cursor=cursor,
            )
            pages += 1
            if response.latency_ms is not None:
                latencies.append(float(response.latency_ms))
            payload = response.data if isinstance(response.data, dict) else {}

            for raw in payload.get("markets", []) or []:
                if not isinstance(raw, dict):
                    continue
                ticker = str(raw.get("ticker") or "").strip()
                if not ticker:
                    continue
                found[ticker] = dict(raw)

            next_cursor = payload.get("cursor")
            if not next_cursor:
                return KalshiCatalogResult(
                    markets=tuple(found.values()),
                    pages=pages,
                    cursor_exhausted=True,
                    max_latency_ms=max(latencies) if latencies else None,
                    error=None,
                )

            next_cursor = str(next_cursor)
            if next_cursor == cursor or next_cursor == last_cursor:
                return KalshiCatalogResult(
                    markets=tuple(found.values()),
                    pages=pages,
                    cursor_exhausted=False,
                    max_latency_ms=max(latencies) if latencies else None,
                    error="Kalshi pagination cursor repeated before exhaustion",
                )
            last_cursor = cursor
            cursor = next_cursor

        return KalshiCatalogResult(
            markets=tuple(found.values()),
            pages=pages,
            cursor_exhausted=False,
            max_latency_ms=max(latencies) if latencies else None,
            error=f"Kalshi catalog safety limit reached after {pages} pages before cursor exhaustion",
        )
    except Exception as exc:
        return KalshiCatalogResult(
            markets=tuple(found.values()),
            pages=pages,
            cursor_exhausted=False,
            max_latency_ms=max(latencies) if latencies else None,
            error=str(exc)[:300],
        )
