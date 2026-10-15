"""Opt-in read-only report viewer; not a pipeline to private accounts.

No network, bookmaker, personal history, RFQ, or live trading dependencies.
This view requires an explicitly uploaded, sanitized research-only report.
"""
from __future__ import annotations
import json
from math import isfinite


def render_five_sport_joint_research_reports():
    import streamlit as st

    with st.expander("Historical five-sport joint score backtests · research only"):
        st.caption(
            "Run scripts/five_sport_joint_research.py on demand to collect "
            "verified final-score archives and produce a dated backtest report. "
            "Upload only its sanitized SCORE-BACKTEST JSON—not credentials, "
            "account exports or raw private Kalshi fills. A hosted upload "
            "transfers this public-score research report to the host."
        )
        uploaded=st.file_uploader(
            "Choose a five-sport SCORE-BACKTEST report JSON",
            type=["json"],key="five_sport_joint_backtest_upload")
        if uploaded is None:
            st.info("No historical score report selected. Nothing is being collected or monitored.")
            return
        try:
            payload=uploaded.getvalue()
            if len(payload)>2_000_000:
                raise ValueError("file_too_large")
            report=json.loads(payload.decode("utf-8"))
            if (not isinstance(report,dict) or
                report.get("schema")!="SPORTEDGE-FIVE-SPORT-SCORE-BACKTEST-V1" or
                report.get("qualified_for_live") is not False or
                report.get("historical_only") is not True or
                not isinstance(report.get("archive_sha256"),str)):
                raise ValueError("not_a_valid_historical_score_report")
            rows=[]
            for sport in ("NFL","NCAAF","NBA","NHL","MLB"):
                item=report.get("results",{}).get(sport)
                if not isinstance(item,dict):
                    raise ValueError("five_sport_report_incomplete")
                result=item.get("report")
                if not isinstance(result,dict) or result.get("qualified_for_live") is not False:
                    raise ValueError("research_only_status_required")
                def numeric(v):
                    if v is None:return None
                    x=float(v)
                    if not isfinite(x) or not 0<=x<=1:
                        raise ValueError("invalid_brier")
                    return round(x,4)
                rows.append({
                    "Sport":sport,"Status":result["status"],
                    "Held-out predictions":int(result["n"]),
                    "Joint Brier":numeric(result.get("joint_brier")),
                    "Independence Brier":numeric(result.get("independence_model_brier")),
                    "League prior Brier":numeric(result.get("league_prior_brier")),
                })
        except (KeyError,TypeError,ValueError,UnicodeDecodeError,OverflowError) as error:
            st.error(f"Rejected invalid or unsafe score research report: {error}")
            return
        st.dataframe(rows,use_container_width=True,hide_index=True)
        st.caption(
            "This historical comparison is not live betting evidence. "
            "Game score outcomes do NOT automatically match Kalshi contract "
            "settlement rules, and the scorer does not model player props "
            "or profitable combo exits. Prospective calibration still needed."
        )
