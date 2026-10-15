"""User-selected sanitized Kalshi combo quote snapshots: research-only.

No account connector, API keys, network calls, orders or background polling.
"""
from __future__ import annotations
import json
from decimal import Decimal, InvalidOperation
from sports_edge.models.combo_quote_archive import (
    MAX_FILE_BYTES, MAX_SNAPSHOTS, first_observed_paper_target,
    quote_depth_research, validate_sanitized_combo_snapshot
)


def render_imported_combo_snapshots():
    import streamlit as st
    with st.expander("On-demand Kalshi combo bid snapshots · paper research", expanded=False):
        st.caption(
            "On your Mac, choose 'Record Combo Quote.command'. Import ONLY "
            "sanitized sportedge_combo_quote_*.json snapshots. NEVER upload "
            "Kalshi raw account history, local_config.json, API IDs, or private "
            "keys. Uploading a file to the hosted app transfers it to that app."
        )
        files=st.file_uploader("Choose sanitized combo quote snapshots (up to 20)",
            type=["json"],accept_multiple_files=True,key="cashout_sanitized_quote_upload")
        if not files:
            st.caption("No snapshots imported. No monitoring is active.")
            return
        if len(files)>20:
            st.error("Maximum 20 snapshots per upload; nothing imported.")
            return
        rows=[]
        try:
            for file in files:
                data=file.getvalue()
                if len(data)>MAX_FILE_BYTES:
                    raise ValueError("snapshot_file_larger_than_512kb")
                doc=json.loads(data.decode("utf-8"))
                rows.extend(validate_sanitized_combo_snapshot(doc))
            if len(rows)>MAX_SNAPSHOTS:
                raise ValueError("too_many_snapshot_market_rows")
            if len({(r["ticker"],r["observed_at_utc"]) for r in rows})!=len(rows):
                raise ValueError("duplicate_snapshot_timestamp_for_same_market")
        except (ValueError,UnicodeDecodeError) as exc:
            st.error(f"Rejected invalid or non-sanitized combo quote export: {exc}")
            return
        rows=sorted(rows,key=lambda x:(x["ticker"],x["observed_at_utc"]))
        tickers=sorted({r["ticker"] for r in rows})
        st.caption(
            f"Read {len(rows)} market snapshot(s) across {len(tickers)} combo(s). "
            "Bid sizes are unfilled public book levels, not a realized return "
            "or a promise of future price availability."
        )
        summary=[]
        for r in rows:
            yes=r["yes_levels"][0] if r["yes_levels"] else None
            no=r["no_levels"][0] if r["no_levels"] else None
            summary.append({
                "Combo":r["ticker"],"Observed UTC":r["observed_at_utc"].isoformat(),
                "Market":r["market_status"],"Status":r["status"],
                "Best YES bid":str(yes[0]) if yes else "Missing",
                "YES bid size":str(yes[1]) if yes else "0",
                "Best NO bid":str(no[0]) if no else "Missing",
                "NO bid size":str(no[1]) if no else "0",
                "Leg count":r["leg_count"]})
        st.dataframe(summary,hide_index=True,use_container_width=True)
        ticker=st.selectbox("Exact combo market to evaluate",tickers,key="cqs_ticker")
        side=st.selectbox("Held combo side",["YES","NO"],key="cqs_side")
        count=st.text_input("Paper contracts held (fractional allowed)",value="10.00",key="cqs_contracts")
        cost=st.text_input("Paper total entry cost USD, including fees",value="2.00",key="cqs_cost")
        fee=st.text_input("Estimated sell fee USD per contract (mandatory)",value="0.01",key="cqs_fee")
        chosen=[r for r in rows if r["ticker"]==ticker]
        if st.button("Evaluate saved combo orderbook (never a fill)",key="cqs_value"):
            try:
                result=quote_depth_research(chosen[-1],side=side,contracts=count,
                    entry_total_including_fees=cost,exit_fee_per_contract=fee)
                st.write(f"**Quote classification:** {result['status']}")
                if result["paper_net_profit_for_quoted_part"] is not None:
                    a,b,c=st.columns(3)
                    a.metric("Hypothetical quoted-portion P&L",
                        f"USD {Decimal(result['paper_net_profit_for_quoted_part']):.2f}")
                    b.metric("Paper ROI for quoted portion",
                        f"{Decimal(result['paper_roi_for_quoted_part']):+.1%}")
                    c.metric("Contracts lacking quoted depth",result["contracts_not_priced"])
                else:
                    st.warning("No verified public bid depth for this position side.")
                st.caption("Quote snapshot values can expire instantly. "
                           "No RFQ or user fill was verified. If the snapshot "
                           "is older than 60 seconds, it is archival, not live.")
            except (ValueError,InvalidOperation) as exc:
                st.warning(f"Paper valuation blocked: {exc}")
        with st.expander("First observed paper +50% exit · chronological replay"):
            entry=st.text_input("Entry time UTC ISO8601 (required)",value="",key="cqs_entry_utc")
            cutoff=st.text_input("Settlement time UTC ISO8601 (optional)",value="",key="cqs_cutoff_utc")
            if st.button("Find first post-entry snapshot at +50% (paper only)",key="cqs_replay"):
                if not entry:
                    st.warning("Entry timestamp must precede the first eligible snapshot.")
                else:
                    try:
                        outcome=first_observed_paper_target(
                            chosen,ticker=ticker,side=side,contracts=count,cost=cost,
                            exit_fee=fee,entry_utc=entry,target_roi="0.50",
                            cutoff_utc=cutoff or None)
                        st.json(outcome)
                        st.caption("No fill is inferred. Unrecorded intervals "
                                   "are missing observations, not proof of no bids.")
                    except (ValueError,InvalidOperation) as exc:
                        st.warning(f"Replay blocked: {exc}")
