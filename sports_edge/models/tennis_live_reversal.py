from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite

from sports_edge.core.math import clamp
from sports_edge.data.tennis_live import TennisLiveScoreState
from sports_edge.models.event_identity import canonical_participant
from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.tennis_live_probability import estimate_live_match_probability


CHEAP_TENNIS_REVERSAL_MAX_PRICE = 0.20


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
    at_tiebreak: bool
    point_score: str | None
    score_sources: tuple[str, ...]
    score_conflict: bool
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
    post_trough = path[trough_i:]
    recovery_confirmations = sum(
        1
        for previous, observed in zip(post_trough, post_trough[1:])
        if observed.close - previous.close >= 0.005
    )
    sustained_recovery = (
        len(post_trough) >= 3
        and recovery_confirmations >= 2
        and len(path) >= 2
        and latest.close >= path[-2].close
    )
    current_spread_points = (
        latest.spread * 100.0
        if latest.spread is not None
        else None
    )
    spread_limit = min(0.08, max(0.04, current * 0.45))
    quote_quality = bool(
        latest.quoted
        and latest.spread is not None
        and latest.spread <= spread_limit
    )

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
    fallback_model = "ranking + prior form fallback" in str(leg.model_name or "").lower()
    lower_tour_fallback = fallback_model
    model_sanity = (
        model_conf >= 0.45
        and leg.model_sample_size >= 6
        and model_prior >= 0.05
    )
    score_turnaround = bool(live_state and live_state.turnaround)
    deciding_set = bool(live_state and live_state.deciding_set)
    current_set_lead = int(live_state.current_set_lead) if live_state else 0
    serving = live_state.serving if live_state is not None else None
    net_break_advantage = (
        live_state.net_break_advantage
        if live_state is not None
        else None
    )
    best_of = int(live_state.best_of) if live_state is not None else 3
    at_tiebreak = bool(live_state and live_state.at_tiebreak)
    point_score = live_state.point_score if live_state is not None else None
    score_sources = tuple(live_state.score_sources) if live_state is not None else ()
    score_conflict = bool(live_state and live_state.score_conflict)

    live_estimate = estimate_live_match_probability(model_prior, live_state)
    live_probability = live_estimate.probability if live_estimate is not None else None
    live_edge_points = (
        (live_probability - current) * 100.0
        if live_probability is not None
        else None
    )
    # ESPN does not provide reliable point score inside the current game, so
    # demand a margin large enough to absorb that hidden-state uncertainty.
    deep_live_edge_floor = (
        max(7.0, current * 30.0)
        if lower_tour_fallback
        else max(5.0, current * 25.0)
    )
    generic_live_edge_floor = (
        max(6.0, current * 25.0)
        if lower_tour_fallback
        else max(4.0, current * 20.0)
    )
    live_probability_floor = 0.22 if lower_tour_fallback else 0.18
    live_value_support = bool(
        live_probability is not None
        and live_probability >= live_probability_floor
        and live_edge_points is not None
        and live_edge_points >= deep_live_edge_floor
    )
    generic_live_support = bool(
        live_probability is not None
        and live_edge_points is not None
        and live_edge_points >= generic_live_edge_floor
    )

    # Net-break state is more informative than raw game lead. A 0-1 score can
    # simply mean the opponent held serve; being a full break down is the real
    # contradiction. Fall back to the older deciding-set guard only when ESPN
    # does not expose the current server and break state cannot be inferred.
    if net_break_advantage is not None:
        score_contradiction = net_break_advantage <= -1
        score_support = (
            (score_turnaround and net_break_advantage >= 0)
            or net_break_advantage >= 1
        )
    else:
        score_contradiction = deciding_set and current_set_lead <= -2
        score_support = (
            (score_turnaround and (not deciding_set or current_set_lead >= 0))
            or (deciding_set and current_set_lead >= 2)
        )
    deep_price = 0.04 <= current <= 0.25
    deep_drawdown = (
        trough <= 0.12
        and (dip_points >= 10.0 or relative_drop >= 0.40)
    )
    deep_recovery = (
        rebound_points >= 3.0
        and rebound_fraction >= 0.20
        and momentum_points >= 0.5
        and sustained_recovery
    )
    strong_price_confirmation = sustained_recovery and quote_quality
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
    reasons.append(
        f"Recovery confirmation: {recovery_confirmations} upward post-trough candle step(s); "
        f"current spread {current_spread_points:.1f}pp"
        if current_spread_points is not None
        else f"Recovery confirmation: {recovery_confirmations} upward post-trough candle step(s); no complete executable quote"
    )
    if live_state is not None:
        reasons.append(f"Live score: {live_state.score_label}")
        reasons.append(f"Match format: best-of-{best_of}")
        if score_sources:
            reasons.append("Live score source: " + " + ".join(score_sources))
        if point_score is not None:
            reasons.append(f"Current game points: {point_score}")
        if live_probability is not None and live_edge_points is not None:
            reasons.append(
                f"Score-conditioned live model {live_probability:.1%} vs executable {current:.1%} "
                f"({live_edge_points:+.1f}pp live edge)"
            )
        if serving is not None:
            reasons.append("Player is serving now" if serving else "Opponent is serving now")
        if net_break_advantage is not None:
            reasons.append(f"Net break advantage {net_break_advantage:+d}")
        if score_turnaround:
            reasons.append("Live-score turnaround: lost the opening set, then won the latest completed set")
        if deciding_set:
            reasons.append(
                f"True deciding-set state {live_state.player_sets}-{live_state.opponent_sets}; "
                f"current-set game lead {current_set_lead:+d}"
            )

    warnings = list(leg.model_warnings)
    warnings.append(
        "live score probability is a structural state-conditioned estimate, not a settlement guarantee"
    )
    if lower_tour_fallback:
        warnings.append(
            "ranking/prior-form fallback has a stricter live-edge threshold than the primary Tennis model"
        )
    if live_state is None:
        warnings.append("detailed live score/server state unavailable; strong reversal promotion blocked")
    if not confirmed_live:
        warnings.append("live match-start/state not independently confirmed; treat as radar WATCH only")
    if latest_age > 150:
        warnings.append("latest Kalshi candle is stale")
    if not sustained_recovery:
        warnings.append("price rebound lacks multi-candle persistence")
    if not quote_quality:
        warnings.append("executable quote spread is missing or too wide for strong reversal promotion")
    if live_probability is not None and live_edge_points is not None and live_edge_points <= 0:
        warnings.append("score-conditioned live model does not support value at the current executable price")
    if at_tiebreak:
        warnings.append(
            "tiebreak point score is not modeled deeply enough for strong reversal promotion"
        )
    if score_conflict:
        warnings.append(
            "live score feeds disagree on the current structural score; strong reversal promotion blocked"
        )
    if score_contradiction:
        warnings.append(
            "current net-break/game state materially contradicts the earlier turnaround; strong reversal promotion blocked"
        )

    score = 0.0
    score += min(22.0, dip_points * 1.2)
    score += min(18.0, rebound_points * 2.1)
    score += min(10.0, max(0.0, prior_gap_points) * 0.6)
    score += min(18.0, max(0.0, live_edge_points or 0.0) * 1.25)
    score += 12.0 * model_conf
    score += 4.0 if h2h else 0.0
    score += 4.0 if trend else 0.0
    score += 4.0 if confirmed_live else 0.0
    score += 5.0 if score_turnaround else 0.0
    score += 5.0 if (net_break_advantage or 0) >= 1 else 0.0
    score += 4.0 if sustained_recovery else 0.0
    score += 4.0 if quote_quality else 0.0
    score += 2.0 if recent_volume >= 100 else (1.0 if recent_volume > 0 else 0.0)
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
        and not score_contradiction
        and not score_conflict
        and (not lower_tour_fallback or (net_break_advantage is not None and net_break_advantage >= 1))
        and live_value_support
        and quote_quality
        and not at_tiebreak
        and fresh
        and confirmed_live
        and live_state is not None
        and score >= 68
    ):
        status = "DEEP REVERSAL"
    elif (
        major_dip
        and reversal
        and model_sanity
        and generic_live_support
        and not score_contradiction
        and not score_conflict
        and strong_price_confirmation
        and not at_tiebreak
        and fresh
        and confirmed_live
        and live_state is not None
        and score >= 68
    ):
        status = "REVERSAL SIGNAL"
    elif (
        major_dip
        and reversal
        and model_sanity
        and fresh
        and (model_support or generic_live_support or live_value_support)
        and score >= 55
    ):
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
        live_probability=live_probability,
        live_edge_points=live_edge_points,
        score_turnaround=score_turnaround,
        deciding_set=deciding_set,
        current_set_lead=current_set_lead,
        serving=serving,
        net_break_advantage=net_break_advantage,
        best_of=best_of,
        at_tiebreak=at_tiebreak,
        point_score=point_score,
        score_sources=score_sources,
        score_conflict=score_conflict,
        recovery_confirmations=recovery_confirmations,
        current_spread_points=current_spread_points,
        quote_quality=quote_quality,
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

    # Live feeds can place a match on the adjacent UTC date while Kalshi uses
    # its competition-local date. Exact identity remains preferred, but a
    # unique participant-pair match is a safe fallback for an already-live
    # physical event.
    pair_states: dict[tuple[str, str], list[TennisLiveScoreState]] = {}
    for state in state_index.values():
        parts = str(state.event_id).split(":", 2)
        if len(parts) != 3:
            continue
        pair_states.setdefault((parts[2], state.selection_key), []).append(state)

    rows: list[TennisReversalSignal] = []
    for leg in candidates:
        candles = candle_history.get(str(leg.kalshi_ticker or ""), ())
        selection_key = canonical_participant("Tennis", leg.selection)
        live_state = state_index.get((leg.event_id, selection_key))
        if live_state is None:
            parts = str(leg.event_id).split(":", 2)
            if len(parts) == 3:
                matches = pair_states.get((parts[2], selection_key), [])
                if len(matches) == 1:
                    live_state = matches[0]
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
