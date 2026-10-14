"""Lightweight public Tennis market-movement observer for ntfy.

Independent of structural score providers and player-model loading. All
notifications are explicitly MARKET-ONLY and not wagering recommendations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from sports_edge.data.tennis_live import (
    fetch_open_tennis_match_markets, fetch_tennis_candle_history,
)
from sports_edge.models.tennis_price_movement import (
    price_watch_tickers, build_price_movements,
)
from sports_edge.models.tennis_push_dispatch import send_tennis_pushes
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
        "score_verified":False,
    }
    try:
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
        delivery=send_tennis_pushes(
            store,now=now,price_observations=movement,max_per_poll=6,
        )
        status["push_configured"]=delivery["configured"]
        status["push_accepted"]=delivery["accepted"]
        status["push_failed"]=delivery["failed"]
        status["errors"].extend(delivery["errors"])
        status["healthy"]=delivery["failed"]==0 and delivery["configured"]
        if not delivery["configured"]:
            status["errors"].append("ntfy push topic not configured")
    except Exception as exc:
        status["errors"].append(f"{type(exc).__name__}: {exc}")
    store.add_run(status,hourly_only=True)
    store.save_report()
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
