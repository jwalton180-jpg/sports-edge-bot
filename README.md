# Sports Edge

Free-first, read-only sports analytics for Kalshi sports markets with tennis, MLB and NFL as first-class verticals.

## Safety / integrity
- No wager or Kalshi order submission methods.
- Stale, conflicting, ambiguous or unverified live inputs fail closed.
- Paper-study records are internal research artifacts, not a user-facing paper-trading mode.
- No guaranteed-win claims.

## Run locally
```bash
python -m pip install -r requirements.txt
python launch_on_demand.py
```

Or:
```bash
streamlit run app.py
```

See `ON_DEMAND_DEPLOY.md` for Streamlit Community Cloud deployment.
