from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean

from sports_edge.core.math import clamp
from sports_edge.data.kalshi import KalshiPublicClient
from sports_edge.models.parlay_candidates import ParlayCandidateLeg


@dataclass(frozen=True)
class TennisReversalSignal:
    event_id: str
    event_title: str
    selection: str
    ticker: str
    side: str
    current_price: float
    model_fair: float
    model_confidence: float
    pre_dip_peak: float
    dip_low: float
    drawdown_pp: float
    recovery_pp: float
    model_edge_pp: float
    post_low_volume: float
    minutes_since_low: int
    score: float
    status: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def _f(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(value) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        out = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if out.tzinfo is None:
            out = out.replace(tzinfo=timezone.utc)
        return out.astimezone(timezone.utc)
    except ValueError:
        return None


def _probably_live(market: dict, now: datetime) -> bool:
    occurrence = _iso(market.get("occurrence_datetime"))
    close = _iso(market.get("close_time") or market.get("latest_expiration_time"))
    status = str(market.get("status") or "").strip().lower()
    if occurrence is None or close is None:
        return False
    # Kalshi market objects use active/finalized-style object states while the
    # list endpoint is queried with open/settled filters. Require an active-ish
    # state and a clock that is after scheduled occurrence but before close.
    activeish = status in {"active", "open", "inactive", "paused"}
    return activeish and occurrence <= now <= close


def _close_value(block: dict | None) -> float | None:
    if not isinstance(block, dict):
        return None
    for key in ("close_dollars", "close", "mean_dollars", "mean"):
        value = _f(block.get(key))
        if value is not None:
            return clamp(value / 100.0 if value > 1.0 else value, 0.0, 1.0)
    return None


def _yes_mid(candle: dict) -> float | None:
    bid = _close_value(candle.get("yes_bid"))
    ask = _close_value(candle.get("yes_ask"))
    if bid is not None and ask is not None and ask >= bid:
        return clamp((bid + ask) / 2.0, 0.0, 1.0)
    trade = _close_value(candle.get("price"))
    return trade


def _side_price(yes_price: float, side: str) -> float:
    return yes_price if side.upper() == "YES" else 1.0 - yes_price


def _history_points(candles: list[dict], side: str) -> list[tuple[int, float, float]]:
    out: list[tuple[int, float, float]] = []
    for candle in candles:
        yes_price = _yes_mid(candle)
        ts = candle.get("end_period_ts")
        if yes_price is None or ts is None:
            continue
        volume = _f(candle.get("volume_fp", candle.get("volume"))) or 0.0
        out.append((int(ts), _side_price(yes_price, side), max(0.0, volume)))
    out.sort(key=lambda row: row[0])
    return out


def analyze_tennis_reversal(
    *,
    candidate: ParlayCandidateLeg,
    market: dict,
    candles: list[dict],
    now: datetime | None = None,
    min_drawdown_pp: float = 12.0,
    min_recovery_pp: float = 4.0,
    min_model_edge_pp: float = 6.0,
) -> TennisReversalSignal | None:
    if candidate.sport != "Tennis" or candidate.market_key != "model_h2h":
        return None
    if candidate.kalshi_ticker is None or candidate.kalshi_side not in {"YES", "NO"}:
        return None
    if candidate.kalshi_price is None or candidate.model_probability is None:
        return None

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not _probably_live(market, now):
        return None

    current = float(candidate.kalshi_price)
    fair = float(candidate.model_probability)
    confidence = float(candidate.model_confidence)

    # A strict underdog watch should still be priced below 50% at the current
    # executable side. Once it becomes favorite, the early-reversal entry has
    # largely passed.
    if not (0.06 <= current <= 0.49):
        return None
    if confidence < 0.50:
        return None

    points = _history_points(candles, candidate.kalshi_side)
    if len(points) < 8:
        return None

    # Locate a meaningful peak -> low drawdown. The low must not be the very
    # first observation because that would not demonstrate an in-window dip.
    best_peak = points[0][1]
    best_peak_idx = 0
    dip_idx = -1
    dip_low = 1.0
    pre_dip_peak = 0.0
    best_drawdown = 0.0
    for idx, (_, price, _) in enumerate(points[1:], start=1):
        drawdown = best_peak - price
        if drawdown > best_drawdown:
            best_drawdown = drawdown
            dip_idx = idx
            dip_low = price
            pre_dip_peak = best_peak
        if price > best_peak:
            best_peak = price
            best_peak_idx = idx

    if dip_idx <= best_peak_idx:
        # If the running peak moved after the selected low, recompute the
        # selected peak index rather than accepting an impossible sequence.
        prior = points[:dip_idx]
        if not prior:
            return None
        pre_dip_peak = max(x[1] for x in prior)

    drawdown_pp = 100.0 * (pre_dip_peak - dip_low)
    recovery_pp = 100.0 * (current - dip_low)
    edge_pp = 100.0 * (fair - current)

    if drawdown_pp < min_drawdown_pp or recovery_pp < min_recovery_pp:
        return None
    if edge_pp < min_model_edge_pp:
        return None

    # Require recovery confirmation in the recent path, not a single stale
    # quote. Compare the latest current executable price with the last several
    # historical mids after the low.
    post = points[dip_idx:]
    recent_prices = [p for _, p, _ in post[-5:]]
    if len(recent_prices) < 3:
        return None
    recent_baseline = mean(recent_prices[:-1]) if len(recent_prices) > 1 else recent_prices[0]
    momentum_pp = 100.0 * (current - recent_baseline)
    if momentum_pp < 0.75:
        return None

    post_low_volume = sum(v for _, _, v in post)
    low_ts = points[dip_idx][0]
    minutes_since_low = max(0, int((now.timestamp() - low_ts) // 60))
    freshness_penalty = clamp(minutes_since_low / 90.0, 0.0, 1.0)

    # Rank the strongest reversal/model disagreements first. Score is an
    # explainable heuristic, not a win probability.
    score = (
        0.34 * clamp(drawdown_pp / 25.0, 0.0, 1.0)
        + 0.28 * clamp(recovery_pp / 15.0, 0.0, 1.0)
        + 0.28 * clamp(edge_pp / 20.0, 0.0, 1.0)
        + 0.10 * confidence
        - 0.10 * freshness_penalty
    ) * 100.0

    status = "REVERSAL" if (
        drawdown_pp >= 15.0
        and recovery_pp >= 6.0
        and edge_pp >= 8.0
        and confidence >= 0.55
    ) else "WATCH"

    reasons = (
        f"Major in-match market dip {drawdown_pp:.1f} pp from {pre_dip_peak:.1%} to {dip_low:.1%}",
        f"Recovery {recovery_pp:.1f} pp to current executable {current:.1%}",
        f"Independent Tennis model {fair:.1%} vs Kalshi {current:.1%} ({edge_pp:+.1f} pp)",
        f"Recent recovery confirmation {momentum_pp:+.1f} pp vs trailing post-low midpoint",
        f"Post-low traded volume {post_low_volume:.0f} contracts",
    )
    warnings = (
        "Kalshi Tennis event live-data endpoint is unavailable for these events; score/serve state is not inferred.",
        "This is a market-reversal signal, not a guarantee of match outcome.",
    )

    return TennisReversalSignal(
        event_id=candidate.event_id,
        event_title=candidate.event_title,
        selection=candidate.selection,
        ticker=candidate.kalshi_ticker,
        side=candidate.kalshi_side,
        current_price=current,
        model_fair=fair,
        model_confidence=confidence,
        pre_dip_peak=pre_dip_peak,
        dip_low=dip_low,
        drawdown_pp=drawdown_pp,
        recovery_pp=recovery_pp,
        model_edge_pp=edge_pp,
        post_low_volume=post_low_volume,
        minutes_since_low=minutes_since_low,
        score=score,
        status=status,
        reasons=reasons,
        warnings=warnings,
    )


def scan_live_tennis_reversals(
    *,
    grouped: dict,
    model_candidates: list[ParlayCandidateLeg],
    client: KalshiPublicClient | None = None,
    now: datetime | None = None,
    lookback_minutes: int = 150,
    max_markets: int = 36,
) -> tuple[list[TennisReversalSignal], list[str]]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    client = client or KalshiPublicClient()
    tennis_rows = grouped.get("Tennis", []) if isinstance(grouped, dict) else []

    market_by_ticker: dict[str, dict] = {}
    for row in tennis_rows:
        market = getattr(row, "market", None)
        if not isinstance(market, dict):
            continue
        ticker = str(market.get("ticker") or "")
        if ticker:
            market_by_ticker[ticker] = market

    # Spend API calls only on plausible underdog/model-disagreement candidates.
    eligible = [
        c for c in model_candidates
        if c.sport == "Tennis"
        and c.market_key == "model_h2h"
        and c.kalshi_ticker in market_by_ticker
        and c.kalshi_price is not None
        and 0.06 <= float(c.kalshi_price) <= 0.49
        and c.model_probability is not None
        and float(c.model_probability) - float(c.kalshi_price) >= 0.04
        and c.model_confidence >= 0.48
        and _probably_live(market_by_ticker[c.kalshi_ticker], now)
    ]
    eligible.sort(
        key=lambda c: (
            float(c.model_probability or 0.0) - float(c.kalshi_price or 0.0),
            c.model_confidence,
        ),
        reverse=True,
    )

    out: list[TennisReversalSignal] = []
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    start_ts = int((now - timedelta(minutes=max(45, lookback_minutes))).timestamp())
    end_ts = int(now.timestamp())

    for candidate in eligible:
        key = (str(candidate.kalshi_ticker), str(candidate.kalshi_side))
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > max_markets:
            break

        market = market_by_ticker[str(candidate.kalshi_ticker)]
        series = str(market.get("series_ticker") or "").strip()
        if not series:
            continue
        try:
            payload = client.market_candlesticks(
                series,
                str(candidate.kalshi_ticker),
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=1,
                include_latest_before_start=True,
            )
            candles = payload.get("candlesticks", []) if isinstance(payload, dict) else []
            signal = analyze_tennis_reversal(
                candidate=candidate,
                market=market,
                candles=list(candles or []),
                now=now,
            )
            if signal is not None:
                out.append(signal)
        except Exception as exc:
            errors.append(f"{candidate.kalshi_ticker}: {type(exc).__name__}: {exc}")

    out.sort(key=lambda s: (s.status == "REVERSAL", s.score, s.model_edge_pp), reverse=True)
    return out, errors
