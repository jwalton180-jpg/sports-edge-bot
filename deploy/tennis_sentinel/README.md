# SportsEdge Tennis Sentinel — cloud-only deployment

This is an optional, always-on, read-only worker on a dedicated cloud service. It must NOT run on the shared Sixty/Late Caller/AutoProfit Ubuntu VM.

## Railway setup

1. Connect Railway to ChatGPT. Create a NEW service sourced from the SportsEdge GitHub repository, separate from the Streamlit app.
2. Set the Dockerfile to deploy/tennis_sentinel/Dockerfile. The optional config-as-code file is deploy/tennis_sentinel/railway.toml.
3. Attach a persistent volume at /data and set SPORTSEDGE_SENTINEL_DATA_DIR=/data/tennis. The Docker entrypoint creates that directory with UID 10001 ownership, then drops privileges before starting the Python worker.
4. Add secrets in Railway (never GitHub code): KALSHI_ACCESS_KEY and matching KALSHI_PRIVATE_KEY_PEM; SPORTSEDGE_NTFY_TOPIC matching the iPhone topic; optionally SPORTSEDGE_NTFY_TOKEN.
5. Explicitly set LIVE_TRADING=false, ORDER_SUBMISSION_ENABLED=false, READ_ONLY_KALSHI=true.
6. Run python -m scripts.tennis_realtime_sentinel.

The Kalshi API key should have minimum available privileges. No order methods exist in this worker. The ntfy topic is not an encrypted/private channel; only public market information is sent. Hosting may incur charges.

## Health and safe rollout

- The persistent file sentinel_status.json records connection cycles, contract count, fresh-score coverage, quote events, and accepted/failed ntfy pushes.
- notifications.jsonl is append-only for de-duplication within this worker. The scheduled GitHub fallback uses another ledger: duplicate cross-worker pushes are still possible until coordinated.
- The stream refreshes listings and independent score state approximately every 90 seconds, and reconnects after errors. Price updates are event-driven only when the server supplies reliable two-sided quote changes.
- Missing/stale/conflicting scores and late match-elimination situations block alerts. Incomplete quotes or stream sequence gaps cannot produce inferred prices.
- Keep existing GitHub jobs and confirmed DEEP REVERSAL criteria unchanged until 24h of prospective coverage, latency, and real iPhone alert comparisons are reviewed.
- Continuous Railway hosting is NOT deployed by a GitHub merge; cloud account connection, credentials and persistent volume are essential.

The service must explicitly use `deploy/tennis_sentinel/Dockerfile` as its Dockerfile path; the file is not at the repository root. Configure this before first live deployment. Do not launch the worker until its encrypted Kalshi and ntfy variables are present.
