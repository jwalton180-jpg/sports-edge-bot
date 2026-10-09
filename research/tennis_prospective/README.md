# SportsEdge Tennis prospective research data

This branch stores **public-market-only** first-observation evidence and verified Kalshi settlement grades written by the scheduled read-only GitHub Action on main.

Data files (created by first worker run):
- signals.jsonl: append-only first observed executable quote per signal/lane/model version
- settlements.jsonl: append-only first verified Kalshi YES/NO terminal result; not sourced from score apps
- snapshots.jsonl: pre-signal price/model/state snapshots for chronological replay, as sampled
- runs.jsonl: hourly worker health record
- report.json: materialized cumulative and chronological report

This is **research**, not executed trading. Quotes are not fills. Git history is the audit trail.
Never commit account details, credentials, user orders, positions or private data.
