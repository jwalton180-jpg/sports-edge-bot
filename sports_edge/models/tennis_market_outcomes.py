"""Prospective post-alert market quote follow-through — not user trading returns.

Read-only Kalshi bid/ask samples after an ntfy acceptance receipt. Keeps
observations append-only and reports only price opportunity, never win rate
or realized P/L. Missing intervals remain explicitly unobserved.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from statistics import median

from sports_edge.models.tennis_prospective_store import (
    append_jsonl, read_jsonl, stable_id, utc_iso,
)


def _price(market: dict, dollars_key: str, cents_key: str) -> float | None:
    raw = market.get(dollars_key)
    multiplier = 1.0
    if raw is None or raw == "":
        raw = market.get(cents_key)
        multiplier = .01
    try:
        value = float(raw)*multiplier
    except (TypeError, ValueError):
        return None
    return value if 0 < value < 1 else None


def sample_followthrough(
    store, markets: list[dict] | tuple[dict, ...], *,
    now: datetime,
    follow_hours: int = 6,
) -> dict:
    """Persist independent post-alert BID, never backfill hypothetical fills.

    The alert's initial quoted ASK is not an executed order. A later BID is
    only a sampled exit opportunity, not proof of executable size or profit.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    path = store.root / "market_followthrough.jsonl"
    history = read_jsonl(path)
    ids = {r["sample_id"] for r in history}
    current = {str(m.get("ticker") or ""):m for m in markets}
    eligible = {}
    for notice in store.notifications:
        ticker = str(notice.get("ticker") or "")
        accepted = notice.get("accepted_at")
        price = notice.get("entry_ask")
        if not ticker or not accepted or price is None:
            continue  # old receipts have no price: never invent an entry.
        try:
            age = (now-datetime.fromisoformat(accepted)).total_seconds()
            entry = float(price)
        except (TypeError, ValueError):
            continue
        if not (60 <= age <= 3600*follow_hours and 0 < entry < 1):
            continue
        eligible[notice["alert_id"]] = (notice,ticker,age)

    observations=[]
    for alert_id,(_,ticker,age) in eligible.items():
        market=current.get(ticker)
        if not market or str(market.get("status") or "").lower() not in {"active","open"}:
            continue
        bid=_price(market,"yes_bid_dollars","yes_bid")
        ask=_price(market,"yes_ask_dollars","yes_ask")
        if bid is None or ask is None or bid > ask:
            continue
        spread=ask-bid
        good_spread = spread <= min(.08,max(.04,ask*.45))
        stamp=utc_iso(now)
        sample_id=stable_id("post-alert-market-bid-v1",alert_id,stamp[:16])
        if sample_id in ids:
            continue
        observations.append({
            "sample_id":sample_id,
            "alert_id":alert_id,
            "ticker":ticker,
            "sampled_at":stamp,
            "time_since_alert_minutes":round(age/60,2),
            "yes_bid":round(bid,5),
            "yes_ask":round(ask,5),
            "spread_pp":round(spread*100,3),
            "spread_qualified":good_spread,
            "observation_only":True,
        })
        ids.add(sample_id)
    append_jsonl(path,observations)
    return {
        "eligible_alerts":len(eligible),
        "new_quote_samples":len(observations),
        "previous_quote_samples":len(history),
    }


def market_opportunity_report(store) -> dict:
    """Describe evidence windows, including unsampled alerts as coverage gaps."""
    samples=read_jsonl(store.root/"market_followthrough.jsonl")
    grouped={}
    for row in samples:
        if row.get("spread_qualified"):
            grouped.setdefault(row["alert_id"],[]).append(row)
    table=[]
    for notice in store.notifications:
        raw_price=notice.get("entry_ask")
        try:
            entry=float(raw_price)
        except (ValueError,TypeError):
            entry=None
        if entry is not None and not 0 < entry < 1:
            entry=None
        observations=sorted(grouped.get(notice["alert_id"],[]),key=lambda x:x["sampled_at"])
        if entry is None:
            state="entry_not_recorded"
        elif not observations:
            state="not_observed_after_alert"
        else:
            state="sampled"
        item={
            "alert_id":notice["alert_id"],
            "ticker":notice["ticker"],
            "lane":notice["lane"],
            "accepted_at":notice["accepted_at"],
            "alert_ask":entry,
            "post_alert_sample_count":len(observations),
            "status":state,
            "first_sample_delay_min":observations[0]["time_since_alert_minutes"] if observations else None,
            "best_observed_bid":max((o["yes_bid"] for o in observations),default=None),
            "worst_observed_bid":min((o["yes_bid"] for o in observations),default=None),
        }
        if entry is not None and observations:
            item["max_observed_gross_return_pct"] = round(
                100*(item["best_observed_bid"]-entry)/entry,2
            )
            item["max_observed_quote_improvement_pp"] = round(
                100*(item["best_observed_bid"]-entry),2
            )
        else:
            item["max_observed_gross_return_pct"]=None
            item["max_observed_quote_improvement_pp"]=None
        table.append(item)
    sampled=[r for r in table if r["status"]=="sampled"]
    tracked=[r for r in table if r["alert_ask"] is not None]
    return {
        "schema_version":1,
        "type":"non_executable_quote_observations",
        "notification_count":len(table),
        "entry_price_recorded":len(tracked),
        "with_post_alert_bid_samples":len(sampled),
        "without_post_alert_samples":sum(r["status"]=="not_observed_after_alert" for r in table),
        "older_unpriced_receipts":sum(r["status"]=="entry_not_recorded" for r in table),
        "sampled_opportunities":sum(
            r["best_observed_bid"]>=r["alert_ask"]+.05 for r in sampled
        ),
        "median_first_sample_delay_min":(
            round(median(r["first_sample_delay_min"] for r in sampled),2)
            if sampled else None
        ),
        "alerts":table,
        "warning":"Observed later quoted BIDs are not fills, fees, liquidation offers, settled wins, or actual user returns. Missing polling windows can miss extrema.",
    }


def save_opportunity_report(store) -> dict:
    report=market_opportunity_report(store)
    (store.root/"market_opportunities.json").write_text(
        json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",
        encoding="utf-8"
    )
    return report
