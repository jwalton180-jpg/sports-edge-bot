"""Read-only SportEdge importer for user-selected exact Kalshi combo snapshots.

The source is a deliberately sanitized, public *market* orderbook export
created on the user's Mac. This module never reads account history, performs
network access, or submits orders. Bid levels are *paper* depth, not fills.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re

SCHEMA="SPORTEDGE-COMBO-QUOTE-SNAPSHOT-V1"
TICKER=re.compile(r"^KXMVE[A-Z0-9]+(?:-[A-Z0-9]+)+$")
MAX_MARKETS=12
MAX_SNAPSHOTS=100
MAX_LEVELS=100
MAX_FILE_BYTES=512_000


def _number(x, label, *, price=False):
    try:
        d=Decimal(str(x))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError(label+"_not_decimal") from None
    if not d.is_finite() or d<0 or (price and d>1):
        raise ValueError(label+"_out_of_range")
    return d


def _utc(value):
    try:
        d=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    except (ValueError,TypeError):
        raise ValueError("invalid_snapshot_timestamp") from None
    if d.tzinfo is None:
        raise ValueError("snapshot_timestamp_requires_timezone")
    return d.astimezone(timezone.utc)


def _levels(source):
    if not isinstance(source,list) or len(source)>MAX_LEVELS:
        raise ValueError("invalid_orderbook_levels")
    output=[]
    for level in source:
        if not isinstance(level,dict) or not {"bid_dollars","size_fp"}<=set(level):
            raise ValueError("invalid_orderbook_level")
        px=_number(level["bid_dollars"],"bid",price=True)
        qty=_number(level["size_fp"],"size")
        if not 0<px<1 or qty<=0:
            raise ValueError("empty_or_invalid_book_level")
        output.append((px,qty))
    if output != sorted(output,key=lambda x:x[0],reverse=True):
        raise ValueError("book_levels_not_best_to_worst")
    if len({p for p,_ in output}) != len(output):
        raise ValueError("duplicate_bid_level")
    return output


def validate_sanitized_combo_snapshot(doc):
    """Fail closed if a private-account/raw export or invalid schema is used."""
    if not isinstance(doc,dict) or doc.get("schema")!=SCHEMA:
        raise ValueError("not_a_sanitized_combo_quote_snapshot")
    if doc.get("no_account_history") is not True or doc.get("no_API_credentials") is not True:
        raise ValueError("snapshot_missing_account_data_separation_guards")
    if doc.get("research_only") is not True:
        raise ValueError("live_qualified_snapshot_not_supported")
    markets=doc.get("markets")
    if (not isinstance(markets,list) or
        not 1<=len(markets)<=MAX_MARKETS or doc.get("snapshot_count")!=len(markets)):
        raise ValueError("invalid_snapshot_market_count")
    now=_utc(doc.get("recorded_at_utc"))
    safe=[]
    for record in markets:
        if not isinstance(record,dict):
            raise ValueError("invalid_snapshot_market_row")
        ticker=record.get("ticker")
        if not isinstance(ticker,str) or not TICKER.fullmatch(ticker):
            raise ValueError("invalid_exact_combo_ticker")
        at=_utc(record.get("observed_at_utc"))
        if abs((now-at).total_seconds())>120:
            raise ValueError("snapshot_observation_time_mismatch")
        mode=record.get("status")
        if mode not in ("UNAVAILABLE","NO_CURRENT_ACTIVE_EXECUTABLE_BID_EVIDENCE",
                        "OBSERVED_PUBLIC_BIDS_NOT_GUARANTEED_FILLED"):
            raise ValueError("invalid_snapshot_quote_status")
        yes=_levels(record.get("exact_combo_yes_bid_levels"))
        no=_levels(record.get("exact_combo_no_bid_levels"))
        if mode=="OBSERVED_PUBLIC_BIDS_NOT_GUARANTEED_FILLED" and not (yes or no):
            raise ValueError("observed_bid_status_missing_prices")
        if record.get("fills_verified") is not False or record.get("full_size_exit_verified") is not False:
            raise ValueError("snapshot_cannot_certify_any_execution")
        # Never propagate arbitrary, possibly sensitive fields to the UI.
        safe.append({
            "ticker":ticker,"observed_at_utc":at,
            "market_status":str(record.get("market_status") or ""),
            "status":mode,
            "yes_levels":yes,"no_levels":no,
            "leg_count":len(record.get("leg_tickers_and_sides") or []),
            "error_code":str(record.get("error_code") or "")[:100],
        })
    if len(set((r["ticker"],r["observed_at_utc"]) for r in safe))!=len(safe):
        raise ValueError("duplicate_snapshot_market_rows")
    return safe


def quote_depth_research(row, *, side, contracts, entry_total_including_fees,
                         exit_fee_per_contract, now=None):
    """Price only offered depth in a *single recorded* snapshot, paper-only.

    Fractional quantities supported. Multiple price levels are walked from
    best to worst. No fill, private RFQ or true executable quote asserted.
    A stale snapshot is retained for archival replay, not current trading.
    """
    requested=_number(contracts,"contracts")
    cost=_number(entry_total_including_fees,"entry_cost")
    fee=_number(exit_fee_per_contract,"exit_fee")
    if requested<=0 or cost<=0 or fee>=1:raise ValueError("invalid_paper_position")
    s=str(side).upper()
    if s not in ("YES","NO"):raise ValueError("side_must_be_yes_or_no")
    levels=row["yes_levels"] if s=="YES" else row["no_levels"]
    left=requested
    gross=Decimal(0)
    sold=Decimal(0)
    for px,size in levels:
        amount=min(size,left)
        gross+=amount*px
        sold+=amount
        left-=amount
        if left==0:break
    nowdt=_utc(now) if now is not None else datetime.now(timezone.utc)
    seconds=(nowdt-row["observed_at_utc"]).total_seconds()
    current=0<=seconds<=60
    available=(row["market_status"]=="active" and
               row["status"]=="OBSERVED_PUBLIC_BIDS_NOT_GUARANTEED_FILLED")
    if not available:
        code="NO_VERIFIED_ACTIVE_BID"
    elif sold<=0:
        code="NO_MATCHING_SIDE_BID"
    elif left>0:
        code="PARTIAL_DEPTH_ONLY"
    elif not current:
        code="ARCHIVAL_SNAPSHOT_ONLY"
    else:
        code="FULL_DEPTH_OBSERVED_NOT_EXECUTABLE_GUARANTEE"
    entry_allocated=cost*sold/requested if sold else Decimal(0)
    net_mark=(gross-fee*sold-entry_allocated) if available and sold else None
    mark_roi=net_mark/entry_allocated if net_mark is not None and entry_allocated>0 else None
    return {
        "ticker":row["ticker"],"status":code,
        "observed_at_utc":row["observed_at_utc"].isoformat(),
        "quote_age_seconds":seconds,"side":s,"contracts_requested":str(requested),
        "contracts_with_book_depth":str(sold),
        "contracts_not_priced":str(left),
        "paper_net_profit_for_quoted_part":str(net_mark) if net_mark is not None else None,
        "paper_roi_for_quoted_part":str(mark_roi) if mark_roi is not None else None,
        "full_requested_depth":left==0 and sold>0 and available,
        "executed":False,"qualified_for_live":False,
        "interpretation":"Public snapshot hypothetical bid-walk; levels can vanish and are not actual fills.",
    }


def first_observed_paper_target(rows, *, ticker, side, contracts, cost, exit_fee,
                                entry_utc, target_roi, cutoff_utc=None):
    """Earliest *post-entry* full-depth archival hit; never best future bid."""
    if not TICKER.fullmatch(ticker):raise ValueError("invalid_combo_ticker")
    entry=_utc(entry_utc)
    cutoff=_utc(cutoff_utc) if cutoff_utc else None
    target=_number(target_roi,"target_roi")
    if target>10:raise ValueError("target_above_bound")
    chosen=sorted([r for r in rows if r["ticker"]==ticker],
                  key=lambda r:r["observed_at_utc"])
    if len(chosen)>MAX_SNAPSHOTS:raise ValueError("too_many_snapshots")
    if any(a["observed_at_utc"]>=b["observed_at_utc"] for a,b in zip(chosen,chosen[1:])):
        raise ValueError("duplicate_or_unsorted_observation")
    for r in chosen:
        stamp=r["observed_at_utc"]
        if stamp<=entry:continue
        if cutoff and stamp>=cutoff:break
        result=quote_depth_research(r,side=side,contracts=contracts,
             entry_total_including_fees=cost,exit_fee_per_contract=exit_fee,now=stamp)
        if (result["full_requested_depth"] and
            result["paper_roi_for_quoted_part"] is not None and
            Decimal(result["paper_roi_for_quoted_part"])>=target):
            return {"status":"FIRST_OBSERVED_FULL_DEPTH_PAPER_TARGET",
                    "at_utc":stamp.isoformat(),"estimated_paper_roi":result["paper_roi_for_quoted_part"],
                    "executed":False,"qualified_for_live":False}
    return {"status":"NO_OBSERVED_FULL_DEPTH_PAPER_TARGET",
            "executed":False,"qualified_for_live":False}
