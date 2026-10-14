"""Read-only, score-unverified Tennis price movement monitoring.

Reports two-sided Kalshi quote movement without making a prediction or
claiming a match is live. No orders, fills, or wagering recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sports_edge.models.tennis_live_reversal import executable_path


@dataclass(frozen=True)
class PriceMovement:
    ticker: str
    selection: str
    lane: str
    current_ask: float
    observed_trough: float
    rebound_pp: float
    drawdown_pp: float
    quote_spread_pp: float
    score: str = "Live score UNVERIFIED — market quote movement only"
    live_fair: float | None = None


def _ask(market: dict) -> float | None:
    raw = market.get("yes_ask_dollars")
    if raw is None:
        raw = market.get("yes_ask")
        try:
            price = float(raw) / 100.0
        except (TypeError, ValueError):
            return None
    else:
        try:
            price = float(raw)
        except (TypeError, ValueError):
            return None
    return price if 0 < price < 1 else None


def price_watch_tickers(markets: list[dict] | tuple[dict, ...]) -> tuple[str, ...]:
    """All currently cheap YES match sides, independent of historical-model support."""
    return tuple(dict.fromkeys(
        str(m["ticker"])
        for m in markets
        if m.get("ticker") and m.get("yes_sub_title")
        and str(m.get("status") or "").lower() in {"open", "active"}
        and (price := _ask(m)) is not None and price <= .23
    ))


def build_price_movements(
    markets: list[dict] | tuple[dict, ...],
    candles: dict[str, list[dict] | tuple[dict, ...]],
    *,
    now: datetime | None = None,
) -> tuple[PriceMovement, ...]:
    """Find verified ASK/BID price collapses or rebounds, with no score inference.

    Observed quoted close prices only; never trade lows or a market's displayed
    probability as a guaranteed fill. Markets may be pre-match, delayed or live.
    """
    now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    output=[]
    seen=set()
    for market in markets:
        ticker=str(market.get("ticker") or "")
        if ticker in seen or not ticker or not market.get("yes_sub_title"):
            continue
        seen.add(ticker)
        if str(market.get("status") or "").lower() not in {"open", "active"}:
            continue
        current_market_ask=_ask(market)
        if current_market_ask is None or current_market_ask > .23:
            continue
        path=executable_path(candles.get(ticker,()),"YES")
        if len(path)<4:
            continue
        latest=path[-1]
        age=now.timestamp()-latest.end_ts
        if not (0<=age<=150 and latest.quoted and latest.spread is not None):
            continue
        if not (.01<=latest.close<=.20):
            continue
        if abs(latest.close-current_market_ask)>.04:
            continue
        if latest.spread>min(.06,max(.02,latest.close*.45)):
            continue
        quoted=[(i,p.close) for i,p in enumerate(path) if p.quoted]
        if len(quoted)<4:
            continue
        trough_idx,trough=min(quoted,key=lambda item:item[1])
        peaks=[price for i,price in quoted if i<trough_idx]
        if not peaks:
            continue
        peak=max(peaks)
        drawdown=peak-trough
        if drawdown<.07 or drawdown/max(peak,.01)<.45 or trough>.10:
            continue
        rebound=max(0.,latest.close-trough)
        if latest.close<=.04:
            lane="EXTREME DIP — TRACKING ONLY"
        elif .04<latest.close<=.20:
            movements=[b-a for (ia,a),(ib,b) in zip(quoted,quoted[1:]) if ia>=trough_idx]
            # Usually require two upward one-minute quotes. An exceptional
            # single-minute repricing can be tracked sooner, but only if the
            # final quoted ASK is up >=6pp, the last minute added >=5pp and
            # the quote remains fresh and tight-spread (checked above).
            rapid_jump = (
                rebound >= .06
                and len(quoted) >= 2
                and latest.close - quoted[-2][1] >= .05
            )
            if rebound<.02 or (
                sum(move>=.005 for move in movements)<2 and not rapid_jump
            ):
                continue
            if latest.close<quoted[-2][1]:
                continue
            lane="PRICE RECOVERY — TRACKING ONLY"
        else:
            continue
        output.append(PriceMovement(
            ticker=ticker,selection=str(market["yes_sub_title"]).strip(),
            lane=lane,current_ask=latest.close,observed_trough=trough,
            rebound_pp=round(rebound*100,2),
            drawdown_pp=round(drawdown*100,2),
            quote_spread_pp=round(latest.spread*100,2),
        ))
    output.sort(key=lambda x:(x.lane!="PRICE RECOVERY — TRACKING ONLY",-x.rebound_pp,x.ticker))
    return tuple(output)
