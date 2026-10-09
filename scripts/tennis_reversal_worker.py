"""Unattended, read-only Tennis reversal prospective observer.

Invoked by GitHub Actions on main. Records only externally observable public
markets; never sends, cancels or manages orders. A Git branch stores immutable
first observations and official terminal settlement checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from sports_edge.data.kalshi import KalshiPublicClient
from sports_edge.data.tennis_live import (
    fetch_live_tennis_states,
    fetch_open_tennis_match_markets,
    fetch_tennis_candle_history,
)
from sports_edge.models.event_identity import canonical_participant
from sports_edge.models.kalshi_model_candidates import model_candidates_from_kalshi
from sports_edge.models.kalshi_sports import group_kalshi_sports
from sports_edge.models.tennis_early_research import build_early_reversal_watches
from sports_edge.models.tennis_live_probability import estimate_live_match_probability
from sports_edge.models.tennis_live_reversal import (
    build_tennis_reversal_radar,
    executable_path,
)
from sports_edge.models.tennis_lower_tour import build_lower_tour_live_fallback_candidates
from sports_edge.models.tennis_prospective_store import (
    ProspectiveResearchStore,
    SIGNAL_VERSION, first_signal_record, stable_id, utc_iso,
    verified_kalshi_settlement,
)


def _unwrap(payload):
    return payload.data if hasattr(payload, "data") else payload


def _state_lookup(states):
    index = {(s.event_id, s.selection_key): s for s in states}
    pair = {}
    for s in states:
        parts = str(s.event_id).split(":", 2)
        if len(parts) == 3:
            pair.setdefault((parts[2], s.selection_key), []).append(s)
    return index, pair


def _match_state(leg, index, by_pair):
    key = canonical_participant("Tennis", leg.selection)
    found = index.get((leg.event_id, key))
    if found is None:
        bits = str(leg.event_id).split(":", 2)
        matches = by_pair.get((bits[2], key), ()) if len(bits) == 3 else ()
        if len(matches) == 1:
            found = matches[0]
    return found


def _fresh_quote(leg, candles, now):
    path = executable_path(candles.get(str(leg.kalshi_ticker), ()), leg.kalshi_side)
    if not path:
        return None
    price = path[-1]
    spread_limit = min(.08, max(.04, price.close*.45))
    age = now.timestamp() - price.end_ts
    if not (0 <= age <= 150 and price.quoted
            and price.spread is not None and 0 <= price.spread <= spread_limit):
        return None
    return price


def observe_once(store: ProspectiveResearchStore, *, now: datetime | None = None,
                 max_grades: int = 200) -> dict:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    diagnostics = {
        "observed_at": utc_iso(now), "healthy": False, "errors": [],
        "open_contracts": 0, "score_states": 0, "model_sides": 0,
        "quoted_snapshots_added": 0, "early_watches": 0,
        "confirmed_radar_signals": 0, "new_signals": 0, "new_settlements": 0,
        "pending_settlements": 0,
    }
    try:
        # Strict public market enumeration: do not misrepresent a partial
        # 6-series API outage as zero opportunities. No user credentials.
        markets = list(fetch_open_tennis_match_markets(strict=True))
        diagnostics["open_contracts"] = len(markets)
        if not markets:
            raise RuntimeError("All six Tennis match series returned zero open markets")
        states = list(fetch_live_tennis_states())
        diagnostics["score_states"] = len(states)
        grouped = group_kalshi_sports(markets)
        candidates = [
            c for c in model_candidates_from_kalshi(grouped, sport_filter="Tennis")
            if c.sport == "Tennis" and c.market_key == "model_h2h"
            and c.kalshi_ticker and c.kalshi_side and c.model_probability is not None
        ]
        fallbacks = build_lower_tour_live_fallback_candidates(markets, states, candidates)
        candidates.extend(fallbacks)
        diagnostics["model_sides"] = len(candidates)
        candles = fetch_tennis_candle_history(
            tuple(sorted({str(c.kalshi_ticker) for c in candidates})),
            lookback_minutes=60, period_interval=1, now=now,
        )
        state_index, by_pair = _state_lookup(states)
        radar = build_tennis_reversal_radar(
            candidates, candles,
            confirmed_live_event_ids={s.event_id for s in states},
            live_states=state_index, now=now,
        )
        early = build_early_reversal_watches(
            candidates, candles, state_index, now=now,
        )
        diagnostics["early_watches"] = len(early)
        diagnostics["confirmed_radar_signals"] = len(radar)

        by_id = {
            (c.event_id, c.selection, str(c.kalshi_ticker)): c
            for c in candidates
        }
        snapshots = []
        quotes = {}
        for c in candidates:
            if str(c.kalshi_ticker) in quotes:
                continue
            quotes[str(c.kalshi_ticker)] = _fresh_quote(c, candles, now)

        for c in candidates:
            state = _match_state(c, state_index, by_pair)
            quote = quotes.get(str(c.kalshi_ticker))
            if not state or not quote or state.score_conflict or not state.score_sources:
                continue
            if abs((now - state.fetched_at).total_seconds()) > 150:
                continue
            if not (.04 <= quote.close <= .25):
                continue
            live = estimate_live_match_probability(float(c.model_probability), state)
            if live is None:
                continue
            ticker = str(c.kalshi_ticker)
            side = str(c.kalshi_side).upper()
            snapshots.append({
                "snapshot_id": stable_id(ticker, side, quote.end_ts, SIGNAL_VERSION),
                "ticker": ticker, "event_id": c.event_id,
                "selection": c.selection, "side": side,
                "observed_at": utc_iso(now), "candle_end_ts": quote.end_ts,
                "executable_ask": round(quote.close, 5),
                "spread_pp": round(quote.spread*100, 4),
                "model_prior": round(float(c.model_probability), 6),
                "live_fair_structural": round(live.probability, 6),
                "model_name": c.model_name, "model_confidence": c.model_confidence,
                "score": state.score_label, "score_sources": list(state.score_sources),
                "state_fetched_at": utc_iso(state.fetched_at),
                "score_conflict": False, "model_version": SIGNAL_VERSION,
                "is_executable_quote_observation_not_fill": True,
            })
        diagnostics["quoted_snapshots_added"] = store.add_snapshots(snapshots)

        for s in early:
            c = next((c for c in candidates
                      if c.event_id == s.event_id and c.selection == s.selection
                      and str(c.kalshi_ticker) == s.ticker), None)
            if c is None:
                continue
            quote = quotes.get(s.ticker)
            if quote is None:
                continue
            record = first_signal_record(
                event_id=s.event_id, ticker=s.ticker, selection=s.selection,
                side=str(c.kalshi_side).upper(), lane=s.status,
                price=quote.close, fair=s.fair, score=s.score,
                observed_at=now, model_version=s.model_version,
                candle_end_ts=quote.end_ts, spread_pp=s.quote_spread_pp,
                evidence={
                    "trough":s.trough, "rebound_pp":s.rebound_pp,
                    "edge_pp":s.edge_pp, "point_aware":s.point_aware,
                }
            )
            diagnostics["new_signals"] += int(store.add_signal(record))

        for s in radar:
            if s.status not in {"DEEP REVERSAL", "REVERSAL SIGNAL", "WATCH"}:
                continue
            c = next((c for c in candidates if c.event_id == s.event_id
                      and c.selection == s.selection and str(c.kalshi_ticker) == s.ticker), None)
            if c is None:
                continue
            quote = quotes.get(s.ticker)
            state = _match_state(c, state_index, by_pair)
            # A WATCH derived from unquoted trade data is not an executable
            # observation. Reject stale or contradicted structural state.
            if quote is None or state is None or state.score_conflict:
                continue
            if abs((now - state.fetched_at).total_seconds()) > 150:
                continue
            fair = s.live_probability
            if fair is None:
                continue
            record = first_signal_record(
                event_id=s.event_id, ticker=s.ticker, selection=s.selection,
                side=str(c.kalshi_side).upper(), lane=s.status,
                price=quote.close, fair=fair, score=s.live_score or "",
                observed_at=now, model_version=SIGNAL_VERSION,
                candle_end_ts=quote.end_ts, spread_pp=quote.spread*100,
                evidence={
                    "signal_score":s.score, "model_prior":s.model_prior_probability,
                    "edge_pp":s.live_edge_points, "point_score":s.point_score,
                    "recovery_confirmations":s.recovery_confirmations,
                    "source":list(s.score_sources),
                },
            )
            diagnostics["new_signals"] += int(store.add_signal(record))

        diagnostics["healthy"] = True
    except Exception as exc:
        diagnostics["errors"].append(f"{type(exc).__name__}: {exc}")

    # Settlement grading is independent of fresh Tennis feeds; even when
    # markets are degraded, earlier observations can finish grading.
    client = KalshiPublicClient()
    market_cache = {}
    for signal in store.pending()[:max(0, int(max_grades))]:
        ticker = signal["ticker"]
        if ticker not in market_cache:
            try:
                market_cache[ticker] = _unwrap(client.market(ticker))
            except Exception:
                market_cache[ticker] = None
        payload = market_cache[ticker]
        if not isinstance(payload, dict):
            continue
        grade = verified_kalshi_settlement(
            payload, ticker=ticker, side=signal["side"],
            signal_id=signal["signal_id"], checked_at=now,
        )
        if grade is not None:
            diagnostics["new_settlements"] += int(store.add_settlement(grade))
    diagnostics["pending_settlements"] = len(store.pending())
    store.add_run(diagnostics, hourly_only=True)
    store.save_report()
    return diagnostics


def main(argv=None) -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-grades", type=int, default=200)
    args=parser.parse_args(argv)
    store=ProspectiveResearchStore(args.data_dir)
    row=observe_once(store, max_grades=args.max_grades)
    print(json.dumps(row,sort_keys=True))
    return 0 if row["healthy"] else 2


if __name__=="__main__":
    sys.exit(main())
