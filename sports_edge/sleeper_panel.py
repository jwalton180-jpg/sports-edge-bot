"""Shared mobile-friendly evidence surface across the on-demand application."""
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from sports_edge.models.sleeper_pipeline import run_application_sleeper_scan, export_evidence
from sports_edge.models.sleeper_ledger import SleeperLedger


def render_sleeper_panel(markets, *, coverage_complete, scope, runner=None):
    with st.expander("Sleeper value • every sport and both sides", expanded=False):
        st.caption("Independent probabilities, executable asks and costs are separate. Research projections need prospective validation before ticket eligibility.")
        day = st.date_input("Sleeper games date (Hawaiʻi)", value=datetime.now(ZoneInfo("Pacific/Honolulu")).date(), key="sleeper_day_"+scope)
        limit = st.number_input("Markets to analyze", min_value=1, max_value=200, value=40, step=10, key="sleeper_limit_"+scope)
        from sports_edge.models.sleeper_counts import utc
        selected, unknown = [], 0
        for m in markets:
            try:
                if abs(utc(m["occurrence_datetime"]).astimezone(ZoneInfo("Pacific/Honolulu")).date()-day) <= timedelta(days=1):
                    selected.append(m)
            except (KeyError, ValueError, TypeError):
                unknown += 1
        complete = coverage_complete and unknown == 0
        st.caption(f"{len(selected)} listed contracts near this date; actual starts are verified during the scan; {unknown} lack a verified start time. Catalog {'complete' if complete else 'incomplete'}.")
        key = "sleeper_report_"+scope+"_"+day.isoformat()
        if st.button("Run sleeper value scan", key="sleeper_scan_"+scope, use_container_width=True):
            with st.spinner("Checking independent history, both sides, rules and executable costs…"):
                report = (runner or run_application_sleeper_scan)(selected, coverage_complete=complete, limit=int(limit))
                verified=[]
                unknown_starts=0
                for record in report["records"]:
                    if record.get("start_verified"):
                        if utc(record["event_time"]).astimezone(ZoneInfo("Pacific/Honolulu")).date()==day:
                            verified.append(record)
                    else:
                        unknown_starts+=1
                        verified.append(record)
                report["records"]=verified
                report["coverage"]["requested_hst_date"]=day.isoformat()
                report["coverage"]["unverified_start_records"]=unknown_starts
                if unknown_starts:
                    report["coverage"]["catalog_complete"]=False
                ledger = None
                try:
                    ledger = SleeperLedger(Path("research/sleeper_ledger.sqlite"))
                    scan_id = ledger.append_scan(report)
                    report["local_record_id"] = scan_id
                    report["ledger_export"] = ledger.export_jsonl()
                except (OSError, __import__("sqlite3").Error, ValueError):
                    st.warning("Evidence persistence failed. Download this scan to preserve it.")
                finally:
                    if ledger is not None:
                        ledger.close()
                st.session_state[key] = report
        report = st.session_state.get(key)
        if st.button("Grade saved predictions from official Kalshi results", key="sleeper_grade_"+scope):
            from sports_edge.data.kalshi import KalshiPublicClient
            ledger=SleeperLedger(Path("research/sleeper_ledger.sqlite"))
            try:
                st.json(ledger.grade_official(KalshiPublicClient()))
                st.json(ledger.metrics())
                if report is not None:
                    report["ledger_export"]=ledger.export_jsonl()
            finally:
                ledger.close()
        if report:
            st.caption("Local evidence is append-only; hosted local storage may reset. Download the full ledger for a durable record. No live win rate is claimed.")
            records = report["records"]
            st.dataframe(pd.DataFrame([{
                "Sport":r["sport"], "Market":r["family"], "Participant":r["participant"],
                "Threshold":r["threshold"], "Side":r["side"], "Ask":r["quote"]["ask"],
                "Probability":r["probability"], "Net edge":r["net_edge"],
                "Status":r["status"], "Reason":"; ".join(r["reasons"]),
            } for r in records]), use_container_width=True, hide_index=True)
            if not report["coverage"]["scan_complete"]:
                st.warning("Scan coverage is incomplete; unprocessed markets are retained with reasons.")
            for r in records[:12]:
                with st.expander(f"{r['participant']} · {r['side']} · {r['ticker']}"):
                    st.json(r)
            st.download_button("Export sleeper evidence (JSON)", data=export_evidence(report),
                file_name="sportsedge_sleeper_evidence.json", mime="application/json", key="sleeper_export_"+scope)
            if report.get("ledger_export"):
                st.download_button("Export all recorded scans and settlements (JSONL)", data=report["ledger_export"],
                    file_name="sportsedge_sleeper_ledger.jsonl", mime="application/x-ndjson", key="sleeper_ledger_export_"+scope)
                from sports_edge.models.prospective_review import prospective_review_export
                import json
                try:
                    review = prospective_review_export(report["ledger_export"])
                    st.caption(f"Prospective review: {review['coverage']['canonical_predictions']} canonical YES-side observations; {review['coverage']['officially_graded']} official outcomes. No authenticated model promotion.")
                    st.download_button("Export prospective review dataset (JSON)",
                        data=json.dumps(review, indent=2, allow_nan=False),
                        file_name="sportsedge_prospective_review.json", mime="application/json",
                        key="sleeper_prospective_export_"+scope)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    st.warning("Prospective review export cannot be verified; retain the raw ledger.")
