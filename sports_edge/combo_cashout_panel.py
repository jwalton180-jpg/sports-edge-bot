"""Read-only Streamlit cash-out calculator; no broker connection or orders."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from sports_edge.models.combo_cashout import (
    assess_combo_cashout_quote, calculate_user_reported_cashout,
)


def render_combo_cashout_lab():
    import streamlit as st
    with st.expander("Combo cash-out lab · actual fills vs observed quotes", expanded=False):
        st.caption(
            "Cash-out strategy differs from all-legs-settle-YES. Kalshi combos "
            "have their OWN RFQ/order book. Do not multiply component prices "
            "to infer a combo cash-out. Manual, read-only, no orders."
        )
        ticker=st.text_input("Exact Kalshi combo market ticker",
                             placeholder="Enter the combo ticker",key="cx_ticker")
        mode=st.radio("Cash-out evidence",["Open combo bid (paper mark)",
                          "Completed sale (user-reported)"],horizontal=True,key="cx_mode")
        held=st.number_input("Original combo contracts",min_value=1,max_value=100000,
                             value=10,step=1,key="cx_held")
        entry=st.number_input("Total entry cost USD, including entry fees",
                              min_value=0.01,value=3.00,step=0.10,format="%.2f",key="cx_entry")
        if mode=="Completed sale (user-reported)":
            sold=st.number_input("Contracts actually sold",min_value=1,max_value=int(held),
                                 value=int(held),step=1,key="cx_sold")
            receipts=st.number_input("NET sale receipts USD after sell fees",
                min_value=0.00,value=3.00,step=0.10,format="%.2f",key="cx_receipts")
            if st.button("Calculate reported realized exit",key="cx_filled_button"):
                if not ticker.strip():
                    st.warning("Enter the exact combo ticker, not a component market.")
                else:
                    result=calculate_user_reported_cashout(
                        combo_ticker=ticker,contracts_held_at_entry=int(held),
                        contracts_sold=int(sold),
                        total_entry_cost_including_fees=entry,
                        net_sale_receipts_after_fees=receipts)
                    x,y,z=st.columns(3)
                    x.metric("Reported exit profit",f"USD {result.reported_profit:.2f}")
                    y.metric("ROI on sold portion",f"{result.reported_roi:+.1%}")
                    z.metric("Contracts still at risk",result.contracts_still_open)
                    st.caption("USER-REPORTED, not exchange-authenticated. Earlier "
                        "realized sale profit need not match eventual outcomes. "
                        "Any unsold contracts retain exposure.")
        else:
            st.caption("Quote, size and freshness are MANUALLY entered, NOT "
                "independently verified. Requires exact combo YES bid.")
            bid=st.number_input("Actual COMBO YES bid per contract USD",
                min_value=0.00,max_value=0.99,value=0.30,
                step=0.01,format="%.2f",key="cx_bid")
            qty=st.number_input("Available contracts at that SAME bid",
                min_value=0,max_value=100000,value=0,step=1,key="cx_depth")
            fee=st.number_input("Estimated sell fee USD per contract",
                min_value=0.00,max_value=0.99,value=0.01,
                step=0.01,format="%.2f",key="cx_fee")
            age=st.number_input("Age of user-observed combo bid in seconds",
                min_value=0,max_value=3600,value=30,step=5,key="cx_age")
            if st.button("Evaluate quote only · never a fill",key="cx_quote_button"):
                if not ticker.strip():
                    st.warning("Exact Kalshi combo ticker required.")
                elif bid<=0:
                    st.info("No positive combo bid. No cash-out mark can be priced.")
                else:
                    now=datetime.now(timezone.utc)
                    result=assess_combo_cashout_quote(
                        combo_ticker=ticker,contracts_held=int(held),
                        total_entry_cost=entry,
                        combo_yes_bid=bid,
                        available_bid_contracts=int(qty),
                        estimated_exit_fee_per_contract=fee,
                        quote_observed_at=now-timedelta(seconds=int(age)),now=now)
                    st.write("**Unverified quote status:** "+result.status)
                    if result.estimated_mark_profit is None:
                        st.warning("Unable to price a fresh available quantity; "
                                   "do not infer realizable gains from component legs.")
                    else:
                        x,y,z=st.columns(3)
                        x.metric("Paper profit (not filled)",
                                 f"USD {result.estimated_mark_profit:.2f}")
                        y.metric("Paper ROI, priced portion",
                                 f"{result.estimated_mark_roi:+.1%}")
                        z.metric("Remaining unpriced contracts",
                                 result.unpriced_remaining_contracts)
                    targets=[
                        ("+25%",result.target_bid_for_25pct_roi),
                        ("+50%",result.target_bid_for_50pct_roi),
                        ("+100%",result.target_bid_for_100pct_roi),
                    ]
                    st.dataframe([
                        {"Paper exit ROI":label,
                         "Required exact COMBO bid":f"USD {val:.3f}" if val is not None else "Unattainable below USD 1"}
                        for label,val in targets],
                        use_container_width=True,hide_index=True)
                    st.caption("These targets are arithmetic, not a forecast "
                        "of price movement or buyer availability. A quote "
                        "is not a fill. No orders are placed.")
                    for warning in result.caveats:
                        st.caption("• "+warning)

    # Opt-in: only validated public market snapshots are imported here.
    from sports_edge.combo_quote_panel import render_imported_combo_snapshots
    render_imported_combo_snapshots()
