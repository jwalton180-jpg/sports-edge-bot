from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite

from sports_edge.core.math import clamp
from sports_edge.data.tennis_live import TennisLiveScoreState
from sports_edge.models.event_identity import canonical_participant
from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.tennis_live_probability import estimate_live_match_probability


@dataclass(frozen=True)
class TennisPricePoint:
    end_ts: int
    close: float
    high: float
    low: float
    volume: float
    spread: float | None = None
    quoted: bool = False


@dataclass(frozen=True)
class TennisReversalSignal:
    ticker: str
    event_id: str
    event_title: str
    selection: str
    current_price: float
    local_peak: float
    trough_price: float
    dip_points: float
    relative_drop: float
    rebound_points: float
    rebound_fraction: float
    recent_momentum_points: float
    model_prior_probability: float
    model_confidence: float
    prior_gap_points: float
    recent_volume: float
    latest_age_s: float
    confirmed_live: bool
    live_score: str | None
    live_probability: float | None
    live_edge_points: float | None
    score_turnaround: bool
    deciding_set: bool
    current_set_lead: int
    serving: bool | None
    net_break_advantage: int | None
    best_of: int
    recovery_confirmations: int
    current_spread_points: float | None
    quote_quality: bool
    h2h_context: bool
    trend_context: bool
    score: float
    status: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def _p(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x > 1:
        x /= 100.0
    if not isfinite(x) or x <= 0 or x >= 1:
        return None
    return x


def _f(value, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if isfinite(x) else default
    except (TypeError, ValueError):
        return default


def executable_path(candles: tuple[dict, ...] | list[dict], side: str) -> tuple[TennisPricePoint, ...]:
    """Convert Kalshi candles into executable side prices plus quote quality.

    YES uses the YES ask. NO ask is implied as 1 - YES bid. A point is marked
    quoted only when both executable ask and opposite-side-derived bid exist;
    trade-price fallback remains usable for WATCH diagnostics but cannot by
    itself satisfy the stronger reversal promotion gates.
    """
    side = str(side or "YES").upper()
    out: list[TennisPricePoint] = []

    for candle in candles or ():
        try:
            end_ts = int(candle.get("end_period_ts"))
        except (TypeError, ValueError):
            continue
        price = candle.get("price") or {}
        ask = candle.get("yes_ask") or {}
        bid = candle.get("yes_bid") or {}

        ask_close = _p(ask.get("close_dollars"))
        bid_close = _p(bid.get("close_dollars"))
        quoted = ask_close is not None and bid_close is not None
        spread = None

        if side == "YES":
            close = ask_close or _p(price.get("close_dollars"))
            high = _p(ask.get("high_dollars")) or _p(price.get("high_dollars")) or close
            low = _p(ask.get("low_dollars")) or _p(price.get("low_dollars")) or close
            if quoted:
                spread = max(0.0, ask_close - bid_close)
        else:
            bid_high = _p(bid.get("high_dollars"))
            bid_low = _p(bid.get("low_dollars"))
            close = (1.0 - bid_close) if bid_close is not None else None
            high = (1.0 - bid_low) if bid_low is not None else close
            low = (1.0 - bid_high) if bid_high is not None else close
            if close is None:
                trade_close = _p(price.get("close_dollars"))
                close = (1.0 - trade_close) if trade_close is not None else None
                high = low = close
            if quoted:
                no_ask = 1.0 - bid_close
                no_bid = 1.0 - ask_close
                spread = max(0.0, no_ask - no_bid)

        if close is None:
            continue
        high = clamp(high if high is not None else close, 0.001, 0.999)
        low = clamp(low if low is not None else close, 0.001, 0.999)
        if low > high:
            low, high = high, low
        out.append(
            TennisPricePoint(
                end_ts=end_ts,
                close=clamp(close, 0.001, 0.999),
                high=high,
                low=low,
                volume=max(0.0, _f(candle.get("volume_fp"), 0.0)),
                spread=spread,
                quoted=quoted,
            )
        )

    out.sort(key=lambda row: row.end_ts)
    return tuple(out)

def _drawdown(path: tuple[TennisPricePoint, ...]) -> tuple[float, float, int, float]:
    if not path:
        return 0.0, 0.0, 0, 0.0
    running_peak = path[0].high
    best_peak = running_peak
    trough = path[0].low
    trough_i = 0
    best_dd = max(0.0, running_peak - trough)

    for i, row in enumerate(path):
        if row.high > running_peak:
            running_peak = row.high
        dd = running_peak - row.low
        if dd > best_dd:
            best_dd = dd
            best_peak = running_peak
            trough = row.low
            trough_i = i
    return best_peak, trough, trough_i, best_dd


def assess_tennis_reversal(
    leg: ParlayCandidateLeg,
    candles: tuple[dict, ...] | list[dict],
    *,
    confirmed_live: bool = False,
    live_state: TennisLiveScoreState | None = None,
    now: datetime | None = None,
) -> TennisReversalSignal | None:
    if leg.sport != "Tennis" or leg.market_key != "model_h2h":
        return None
    if not leg.kalshi_ticker or not leg.kalshi_side or leg.model_probability is None:
        return None

    path = executable_path(candles, leg.kalshi_side)
    if len(path) < 4:
        return None

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    latest = path[-1]
    latest_age = max(0.0, now.timestamp() - latest.end_ts)
    current = latest.close
    if not 0.03 <= current <= 0.45:
        return None
    if live_state is not None:
        confirmed_live = True

    peak, trough, trough_i, drawdown = _drawdown(path)
    if drawdown <= 0 or trough_i >= len(path) - 1:
        return None

    rebound = max(0.0, current - trough)
    relative_drop = drawdown / max(peak, 0.01)
    rebound_fraction = rebound / max(drawdown, 0.001)
    momentum_ref = path[max(0, len(path) - 4)].close
    momentum = current - momentum_ref
    recent_volume = sum(x.volume for x in path[max(0, trough_i):])

    model_prior = clamp(float(leg.model_probability), 0.01, 0.99)
    prior_gap = model_prior - current
    model_conf = clamp(float(leg.model_confidence), 0.0, 1.0)

    dip_points = drawdown * 100.0
    rebound_points = rebound * 100.0
    momentum_points = momentum * 100.0
    prior_gap_points = prior_gap * 100.0

    major_dip = dip_points >= 8.0 or (dip_points >= 4.0 and relative_drop >= 0.30)
    reversal = (
        rebound_points >= 2.0
        and rebound_fraction >= 0.18
        and momentum_points >= 0.5
    )
    model_support = (
        model_conf >= 0.50
        and leg.model_sample_size >= 6
        and prior_gap_points >= 3.0
    )
    model_sanity = (
        model_conf >= 0.45
        and leg.model_sample_size >= 6
        and model_prior >= 0.05
    )
    score_turnaround = bool(live_state and live_state.turnaround)
    deciding_set = bool(live_state and live_state.deciding_set)
    current_set_lead = int(live_state.current_set_lead) if live_state else 0

    # A completed-set turnaround is useful corroboration only while the current
    # deciding-set score is not materially contradicting it. Without this guard,
    # a player who lost Set 1 and won Set 2 could stay "turnaround=True" even
    # after falling multiple games behind in Set 3, allowing stale score context
    # to promote a cheap-price rebound that the live score no longer supports.
    score_contradiction = deciding_set and current_set_lead <= -2
    if deciding_set:
        score_support = (
            (score_turnaround and current_set_lead >= 0)
            or current_set_lead >= 2
        )
    else:
        score_support = score_turnaround
    deep_price = 0.04 <= current <= 0.25
    deep_drawdown = (
        trough <= 0.12
        and (dip_points >= 10.0 or relative_drop >= 0.40)
    )
    deep_recovery = (
        rebound_points >= 3.0
        and rebound_fraction >= 0.20
        and momentum_points >= 0.5
    )
    fresh = latest_age <= 150.0

    reasons = [
        f"Executable price fell {dip_points:.1f}pp from {peak:.0%} to {trough:.0%}",
        f"Rebounded {rebound_points:.1f}pp ({rebound_fraction:.0%} of the drawdown)",
        f"Recent 3-minute price momentum {momentum_points:+.1f}pp",
        f"Pregame Sports Edge prior {model_prior:.1%} vs current {current:.1%} ({prior_gap_points:+.1f}pp prior gap)",
        f"Model confidence {model_conf:.0%} on sample {leg.model_sample_size}",
    ]

    h2h = any("head-to-head" in reason.lower() or "h2h" in reason.lower() for reason in leg.model_reasons)
    trend = any(
        "trajectory" in reason.lower() or "serve/return trend" in reason.lower()
        for reason in leg.model_reasons
    )
    if h2h:
        reasons.append("Historical H2H context is present in the player model")
    if trend:
        reasons.append("Recent form/serve-return trajectory supports matchup context")
    if recent_volume > 0:
        reasons.append(f"{recent_volume:.0f} contracts traded since the detected trough window")
    if live_state is not None:
        reasons.append(f"Live score: {live_state.score_label}")
        if score_turnaround:
            reasons.append("Live-score turnaround: lost the opening set, then won the latest completed set")
        if deciding_set:
            reasons.append(
                f"Deciding-set state {live_state.player_sets}-{live_state.opponent_sets}; "
                f"current-set game lead {current_set_lead:+d}"
            )

    warnings = list(leg.model_warnings)
    warnings.append("pregame model probability is a prior, not a live-score fair probability")
    if not confirmed_live:
        warnings.append("live match-start/state not independently confirmed; treat as radar WATCH only")
    if latest_age > 150:
        warnings.append("latest Kalshi candle is stale")
    if score_contradiction:
        warnings.append(
            "current deciding-set score materially contradicts the earlier turnaround; deep reversal promotion blocked"
        )

    score = 0.0
    score += min(24.0, dip_points * 1.35)
    score += min(20.0, rebound_points * 2.4)
    score += min(18.0, max(0.0, prior_gap_points) * 0.9)
    score += 14.0 * model_conf
    score += 5.0 if h2h else 0.0
    score += 5.0 if trend else 0.0
    score += 5.0 if confirmed_live else 0.0
    score += 7.0 if score_turnaround else 0.0
    score += 4.0 if deciding_set and current_set_lead >= 2 else 0.0
    score += 3.0 if recent_volume >= 100 else (1.0 if recent_volume > 0 else 0.0)
    score = clamp(score, 0.0, 100.0)

    # Deep Reversal is the high-payout lane: a true collapse into the 4–25¢
    # range followed by price recovery plus independent live-score evidence.
    # Unlike the generic lane, it does not require the current price to remain
    # below the pre-match model prior; the live turnaround can legitimately
    # move fair value above a low pre-match underdog prior.
    if (
        deep_price
        and deep_drawdown
        and deep_recovery
        and model_sanity
        and score_support
        and fresh
        and confirmed_live
        and score >= 68
    ):
        status = "DEEP REVERSAL"
    elif major_dip and reversal and model_support and fresh and confirmed_live and score >= 68:
        status = "REVERSAL SIGNAL"
    elif major_dip and reversal and model_support and fresh and score >= 55:
        status = "WATCH"
    else:
        status = "PASS"

    return TennisReversalSignal(
        ticker=str(leg.kalshi_ticker),
        event_id=leg.event_id,
        event_title=leg.event_title,
        selection=leg.selection,
        current_price=current,
        local_peak=peak,
        trough_price=trough,
        dip_points=dip_points,
        relative_drop=relative_drop,
        rebound_points=rebound_points,
        rebound_fraction=rebound_fraction,
        recent_momentum_points=momentum_points,
        model_prior_probability=model_prior,
        model_confidence=model_conf,
        prior_gap_points=prior_gap_points,
        recent_volume=recent_volume,
        latest_age_s=latest_age,
        confirmed_live=confirmed_live,
        live_score=live_state.score_label if live_state is not None else None,
        score_turnaround=score_turnaround,
        deciding_set=deciding_set,
        current_set_lead=current_set_lead,
        h2h_context=h2h,
        trend_context=trend,
        score=score,
        status=status,
        reasons=tuple(reasons),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def build_tennis_reversal_radar(
    candidates: list[ParlayCandidateLeg],
    candle_history: dict[str, tuple[dict, ...]],
    *,
    confirmed_live_event_ids: set[str] | None = None,
    live_states: dict[tuple[str, str], TennisLiveScoreState] | None = None,
    now: datetime | None = None,
) -> list[TennisReversalSignal]:
    confirmed = confirmed_live_event_ids or set()
    state_index = live_states or {}
    rows: list[TennisReversalSignal] = []
    for leg in candidates:
        candles = candle_history.get(str(leg.kalshi_ticker or ""), ())
        selection_key = canonical_participant("Tennis", leg.selection)
        live_state = state_index.get((leg.event_id, selection_key))
        signal = assess_tennis_reversal(
            leg,
            candles,
            confirmed_live=leg.event_id in confirmed,
            live_state=live_state,
            now=now,
        )
        if signal is not None and signal.status != "PASS":
            rows.append(signal)

    # One actionable side per physical match. Keep the stronger signal.
    best: dict[str, TennisReversalSignal] = {}
    for row in rows:
        old = best.get(row.event_id)
        if old is None or (
            row.status == "DEEP REVERSAL",
            row.status == "REVERSAL SIGNAL",
            row.score,
            row.prior_gap_points,
            row.rebound_points,
        ) > (
            old.status == "DEEP REVERSAL",
            old.status == "REVERSAL SIGNAL",
            old.score,
            old.prior_gap_points,
            old.rebound_points,
        ):
            best[row.event_id] = row

    return sorted(
        best.values(),
        key=lambda row: (
            row.status == "DEEP REVERSAL",
            row.status == "REVERSAL SIGNAL",
            row.score,
            row.prior_gap_points,
            row.rebound_points,
        ),
        reverse=True,
    )
