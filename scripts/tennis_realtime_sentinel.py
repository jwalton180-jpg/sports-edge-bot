"""Cloud-hosted read-only Tennis price Sentinel.

Standalone process for a dedicated cloud service, NEVER the shared Sixty VM.
Uses Kalshi's authenticated ticker stream and independently fetched scores.
Never submits/cancels trades, and never promotes score-unverified quotes to
phone alerts. The existing scheduled GitHub observer remains a fallback.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
import time

from sports_edge.data.kalshi_ws import KalshiReadOnlyWebSocket
from sports_edge.data.tennis_live import (
    fetch_live_tennis_states,
    fetch_open_tennis_match_markets,
    fetch_tennis_candle_history,
)
from sports_edge.models.tennis_alert_survivability import (
    qualified_tickers,
    survivability_reason,
)
from sports_edge.models.tennis_price_movement import build_price_movements
from sports_edge.models.tennis_prospective_store import ProspectiveResearchStore, utc_iso
from sports_edge.models.tennis_push_dispatch import send_tennis_pushes


LOGGER = logging.getLogger("sportsedge.tennis.sentinel")
MAX_SUBSCRIPTIONS = 600
REFRESH_SECONDS = 90


def _quote_price(msg: dict, stem: str) -> float | None:
    raw = msg.get(stem + "_dollars")
    factor = 1.0
    if raw is None:
        raw = msg.get(stem)
        factor = .01
    try:
        value = float(raw) * factor
    except (ValueError, TypeError):
        return None
    return value if 0 < value < 1 else None


def extract_ticker_quote(frame: dict) -> tuple[str, float, float] | None:
    """Only a Kalshi ticker frame containing BOTH contemporaneous bid and ask.

    Trade-only, orderbook delta and partial updates cannot manufacture a fill.
    """
    if frame.get("type") != "ticker":
        return None
    msg = frame.get("msg")
    if not isinstance(msg, dict):
        return None
    ticker = str(msg.get("market_ticker") or msg.get("ticker") or "")
    bid = _quote_price(msg, "yes_bid")
    ask = _quote_price(msg, "yes_ask")
    if not ticker or bid is None or ask is None or bid > ask:
        return None
    return ticker, bid, ask


def quote_candle(*, bid: float, ask: float, at: datetime) -> dict:
    return {
        "end_period_ts": int(at.timestamp()),
        "yes_bid": {"close_dollars": str(bid), "high_dollars": str(bid),
                    "low_dollars": str(bid)},
        "yes_ask": {"close_dollars": str(ask), "high_dollars": str(ask),
                    "low_dollars": str(ask)},
        "price": {"close_dollars": str(ask)},
        "volume_fp": "0",
    }


class TennisSentinel:
    def __init__(self, root: str | Path, *, period_s: int = REFRESH_SECONDS):
        root = Path(root)
        if not root.is_absolute():
            raise ValueError("Sentinel data directory must be absolute")
        self.store = ProspectiveResearchStore(root)
        self.period_s = min(110, max(30, int(period_s)))
        self.markets: dict[str, dict] = {}
        self.candles: dict[str, tuple[dict, ...]] = {}
        self.allowed = {}
        self.metrics = {
            "connected_cycles": 0, "market_count": 0,
            "subscription_count": 0, "score_count": 0,
            "score_eligible": 0, "events": 0, "valid_quoted_events": 0,
            "tracking_watches": 0, "push_accepted": 0,
            "push_failed": 0, "errors": 0,
        }
        self._last_tick: dict[str, tuple[float, float, int]] = {}

    async def refresh(self) -> tuple[str, ...]:
        """Refresh markets, price paths and scores off the async IO thread."""
        markets = await asyncio.to_thread(fetch_open_tennis_match_markets, strict=True)
        if not markets:
            raise RuntimeError("Market enumeration unavailable; fail closed")
        # Include slightly wider watch universe so a drop crossing 20c is
        # already subscribed. This is tracking ONLY, not an entry recommendation.
        watch = {
            str(m["ticker"]):m for m in markets
            if m.get("ticker") and m.get("yes_sub_title")
            and str(m.get("status") or "").lower() in {"active", "open"}
            and (ask := _quote_price(m, "yes_ask")) is not None
            and ask <= .35
        }
        if len(watch) > MAX_SUBSCRIPTIONS:
            raise RuntimeError("Too many markets for safe subscription; do not silently drop")
        scores = await asyncio.to_thread(fetch_live_tennis_states, timeout=10.0)
        allowed, reasons = qualified_tickers(markets, scores, now=datetime.now(timezone.utc))
        # Do not fetch candle history for matches that cannot survive the score
        # gate. Keep the cheap watch universe subscribed for when scores improve.
        candidate_tickers = sorted(set(watch) & set(allowed))
        history = await asyncio.to_thread(
            fetch_tennis_candle_history, tuple(candidate_tickers),
            lookback_minutes=60, period_interval=1,
        ) if candidate_tickers else {}
        self.markets = watch
        self.allowed = {k:v for k,v in allowed.items() if k in watch}
        self.candles = history
        self._last_tick = {}
        self.metrics.update(
            market_count=len(markets), subscription_count=len(watch),
            score_count=len(scores), score_eligible=len(self.allowed),
        )
        self._heartbeat(extra={"score_rejections": reasons})
        return tuple(sorted(watch))

    def _heartbeat(self, *, extra: dict | None = None) -> None:
        payload = {"observed_at":utc_iso(datetime.now(timezone.utc)),
                   "healthy": True, "source":"websocket",
                   **self.metrics, **(extra or {})}
        self.store.add_run(payload, hourly_only=True)
        self.store.save_report()
        # A service's persistent volume, not a Git commit each market tick.
        (self.store.root / "sentinel_status.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2)+"\n", encoding="utf-8"
        )

    async def on_frame(self, envelope) -> None:
        self.metrics["events"] += 1
        parsed = extract_ticker_quote(envelope.payload)
        if parsed is None:
            return
        ticker, bid, ask = parsed
        if ticker not in self.markets:
            return
        self.metrics["valid_quoted_events"] += 1
        state = self.allowed.get(ticker)
        if state is None or survivability_reason(state, now=envelope.received_at) != "eligible":
            return
        # Never use stale, crossed or loose market quotes, even with a strong
        # score. Trading is disabled regardless of quote quality.
        if ask - bid > min(.06, max(.02, ask*.45)):
            return
        if not (.01 <= ask <= .20):
            return
        current_ts = int(envelope.received_at.timestamp())
        previous = self._last_tick.get(ticker)
        if previous == (bid, ask, current_ts):
            return
        self._last_tick[ticker] = (bid, ask, current_ts)
        recent = self.candles.get(ticker, ())
        if not recent:
            return
        # Authentic WebSocket bid/ask is appended as a quote sample;
        # historical data are Kalshi candlesticks, never extrapolated trades.
        sample = quote_candle(bid=bid, ask=ask, at=envelope.received_at)
        original = self.markets[ticker]
        updated = {**original, "yes_ask_dollars": str(ask)}
        watches = build_price_movements(
            [updated], {ticker: (*recent, sample)}, now=envelope.received_at
        )
        if not watches:
            return
        self.metrics["tracking_watches"] += len(watches)
        safe = tuple(replace(
            w, score=f"{state.score_label} | fresh independent score; "
                     "model edge not established",
        ) for w in watches)
        # ntfy accepts at most one per stage; permanent dedup persists on disk.
        outcome = await asyncio.to_thread(
            send_tennis_pushes, self.store, now=envelope.received_at,
            price_observations=safe, allowed_tickers={ticker},
            max_per_poll=1,
        )
        self.metrics["push_accepted"] += outcome["accepted"]
        self.metrics["push_failed"] += outcome["failed"]
        if outcome["failed"]:
            self.metrics["errors"] += outcome["failed"]
            LOGGER.warning("ntfy delivery failed; no secrets logged")
        if outcome["accepted"]:
            self._heartbeat()

    async def run_forever(self, *, stop_after_cycles: int | None = None) -> None:
        failures = 0
        cycles = 0
        while True:
            try:
                tickers = await self.refresh()
                if not tickers:
                    raise RuntimeError("No available Tennis contract subscriptions")
                # Re-subscribe periodically to include newly listed matches,
                # scores and match-status transitions, with clean WS close.
                ws = KalshiReadOnlyWebSocket()
                await asyncio.wait_for(
                    ws.stream_once(tickers, self.on_frame, channels=("ticker",)),
                    timeout=self.period_s,
                )
                failures = 0
            except asyncio.TimeoutError:
                failures = 0  # expected periodic market / score refresh
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures += 1
                self.metrics["errors"] += 1
                LOGGER.warning("Stream unavailable (%s); retrying", type(exc).__name__)
                # Must not send if market/score/subscription state is unavailable.
                await asyncio.sleep(min(60, 2 ** min(failures, 5)))
            cycles += 1
            self.metrics["connected_cycles"] = cycles
            self._heartbeat()
            if stop_after_cycles is not None and cycles >= stop_after_cycles:
                return


def _private_key_path() -> tuple[str, str | None]:
    path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if path:
        return path, None
    pem = os.getenv("KALSHI_PRIVATE_KEY_PEM")
    if not pem:
        raise RuntimeError("Kalshi read-only stream credentials missing")
    fd, temporary = tempfile.mkstemp(prefix="sportsedge_kalshi_", suffix=".pem")
    with os.fdopen(fd,"w") as file:
        file.write(pem.replace("\\n", "\n"))
    os.chmod(temporary, 0o600)
    return temporary, temporary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.getenv("SPORTSEDGE_SENTINEL_DATA_DIR"))
    parser.add_argument("--cycles", type=int, default=None)
    args = parser.parse_args()
    if os.getenv("LIVE_TRADING","false").lower() not in {"false","0"}:
        raise RuntimeError("Trading must remain disabled")
    if os.getenv("ORDER_SUBMISSION_ENABLED","false").lower() not in {"false","0"}:
        raise RuntimeError("Order submission must remain disabled")
    if not args.data_dir:
        raise RuntimeError("Configure SPORTSEDGE_SENTINEL_DATA_DIR on persistent cloud storage")
    if not os.getenv("SPORTSEDGE_NTFY_TOPIC"):
        raise RuntimeError("ntfy topic unavailable")
    path, temporary = _private_key_path()
    os.environ["KALSHI_PRIVATE_KEY_PATH"] = path
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(TennisSentinel(args.data_dir).run_forever(stop_after_cycles=args.cycles))
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
