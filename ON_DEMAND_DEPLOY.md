# Sports Edge — zero-cost on-demand deployment

Sports Edge supports a session-scoped `on_demand` deployment mode intended for Streamlit Community Cloud or a one-click local launch.

## Streamlit Community Cloud
1. Push this repository to GitHub.
2. Create a Streamlit Community Cloud app with `app.py` as the entry point.
3. Add optional secrets in the Streamlit Secrets UI, never in GitHub.
4. Leave `SPORTS_EDGE_MODE="on_demand"` for the free/session mode.

The on-demand app treats local caches as disposable and should never rely on cloud sleep preserving session files. Research/model artifacts should remain versioned separately.
