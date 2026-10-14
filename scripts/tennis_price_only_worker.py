"""Lightweight public Tennis market-movement observer for ntfy.

Independent of structural score providers and player-model loading. All
notifications are explicitly MARKET-ONLY and not wagering recommendations.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from sports_edge.data.tennis_live import (
    fetch_open_tennis_match_markets, fetch_tennis_candle_history,
    fetch_live_tennis_states,
)
from sports_edge.models.tennis_price_movement import (
    price_watch_tickers, build_price_movements,
)
from sports_edge.models.tennis_push_dispatch import send_tennis_pushes
from sports_edge.models.tennis_alert_survivability import qualified_tickers
from sports_edge.models.tennis_market_outcomes import (
    sample_followthrough, save_opportunity_report,
)
from sports_edge.models.tennis_prospective_store import (
    ProspectiveResearchStore, utc_iso,
)


def poll_once(store: ProspectiveResearchStore, *, now: datetime | None = None) -> dict:
    now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    status={
        "observed_at":utc_iso(now), "healthy":False, "errors":[],
        "open_contracts":0, "cheap_contracts":0, "history_contracts":0,
        "tracking_observations":0, "push_configured":False,
        "push_accepted":0, "push_failed":0,
        "score_verified":False, "live_score_states":0,
        "survivability_cleared":0, "suppressed_unverified_or_late":0,
        "suppression_reasons":{},
        "poll_gap_minutes":None,
        "late_poll_gap":False, "post_alert_eligible":0,
        "post_alert_bid_samples":0,
    }
    try:
        prior_path=store.root/"latest_poll.json"
        if prior_path.exists():
            try:
                previous=json.loads(prior_path.read_text(encoding="utf-8"))
                seen=datetime.fromisoformat(str(previous["observed_at"]))
                gap=(now-seen).total_seconds()/60
                if gap>=0:
                    status["poll_gap_minutes"]=round(gap,1)
                    status["late_poll_gap"]=gap>15
            except (ValueError,KeyError,TypeError):
                # A corrupt previous heartbeat must not become a fabricated gap.
                pass
        markets=list(fetch_open_tennis_match_markets(strict=True))
        status["open_contracts"]=len(markets)
        if not markets:
            raise RuntimeError("No open Tennis markets returned across required series")
        tickers=price_watch_tickers(markets)
        status["cheap_contracts"]=len(tickers)
        histories=fetch_tennis_candle_history(
            tickers,lookback_minutes=60,period_interval=1,now=now,
        )
        status["history_contracts"]=sum(bool(histories.get(t)) for t in tickers)
        if tickers and not status["history_contracts"]:
            raise RuntimeError("No Kalshi candle histories returned for cheap contracts")
        movement=build_price_movements(markets,histories,now=now)
        status["tracking_observations"]=len(movement)
        # Market movements are kept as research even if no provider can verify
        # the match state. Phone push, by contrast, requires independent,
        # fresh two-player match confirmation and material comeback evidence.
        score_states=fetch_live_tennis_states()
        status["live_score_states"]=len(score_states)
        allowed,reasons=qualified_tickers(markets,score_states,now=now)
        status["score_verified"]=bool(score_states)
        status["survivability_cleared"]=len(allowed)
        status["suppression_reasons"]=reasons
        status["suppressed_unverified_or_late"]=sum(
            row.ticker not in allowed for row in movement
        )
        safe_movement=tuple(
            replace(row,score=(
                f"{allowed[row.ticker].score_label} | "
                "live score independently observed; model edge unverified"
            ))
            for row in movement if row.ticker in allowed
        )
        delivery=send_tennis_pushes(
            store,now=now,price_observations=safe_movement,
            allowed_tickers=set(allowed),max_per_poll=6,
        )
        status["push_configured"]=delivery["configured"]
        status["push_accepted"]=delivery["accepted"]
        status["push_failed"]=delivery["failed"]
        status["errors"].extend(delivery["errors"])
        follow = sample_followthrough(store,markets,now=now)
        status["post_alert_eligible"]=follow["eligible_alerts"]
        status["post_alert_bid_samples"]=follow["new_quote_samples"]
        status["healthy"]=delivery["failed"]==0 and delivery["configured"]
        if not delivery["configured"]:
            status["errors"].append("ntfy push topic not configured")
    except Exception as exc:
        status["errors"].append(f"{type(exc).__name__}: {exc}")
    store.add_run(status,hourly_only=True)
    store.save_report()
    save_opportunity_report(store)
    (store.root/"latest_poll.json").write_text(
        json.dumps(status,sort_keys=True,indent=2)+"\n",encoding="utf-8",
    )
    return status


def main(argv=None) -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    status=poll_once(ProspectiveResearchStore(args.data_dir))
    print(json.dumps(status,sort_keys=True))
    return 0 if status["healthy"] else 2


if __name__=="__main__":
    sys.exit(main())
