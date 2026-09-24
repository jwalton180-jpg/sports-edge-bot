from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from sports_edge.core.startup import deployment_mode
from sports_edge.data.kalshi import KalshiPublicClient
from sports_edge.data.mlb import MLBClient
from sports_edge.data.nfl import NFLClient
from sports_edge.data.odds import OddsClient
from sports_edge.models.underdog import score_underdog

st.set_page_config(page_title="Sports Edge", page_icon="◈", layout="wide")

st.markdown(
    """
<style>
.block-container{padding-top:.75rem;max-width:1180px;padding-left:1rem;padding-right:1rem}
.stApp{background:radial-gradient(circle at 15% 0%,#10251d 0,#08110f 38%,#060b0a 100%)}
[data-testid="stMetric"]{background:linear-gradient(180deg,rgba(24,49,41,.72),rgba(10,22,18,.9));border:1px solid #24463a;border-radius:14px;padding:12px}
.badge{display:inline-block;padding:4px 9px;border-radius:999px;border:1px solid #335c4d;background:#12241e;color:#9decc9;font-size:.72rem;margin:0 5px 5px 0}
.hero{font-size:2.35rem;font-weight:850;letter-spacing:-1.2px;line-height:1.05;margin:.25rem 0 .35rem}
.good{color:#62e6a7}.muted{color:#8ba59b}.section-note{color:#a7bbb3;font-size:.92rem;margin-top:-.35rem;margin-bottom:.8rem}
.nav-hint{padding:.7rem .85rem;border:1px solid #24463a;border-radius:12px;background:rgba(14,31,26,.72);margin:.4rem 0 .8rem}
div[data-testid="stExpander"]{border:1px solid #203c33;border-radius:12px;background:rgba(7,17,14,.55)}
hr{border-color:#173127!important}
@media (max-width: 700px){
  .block-container{padding-top:.45rem;padding-left:.65rem;padding-right:.65rem}
  .hero{font-size:1.85rem;letter-spacing:-.7px}
  [data-testid="stMetric"]{padding:9px}
  .nav-hint{font-size:.9rem}
  button[kind="secondary"],button[kind="primary"]{min-height:2.7rem}
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="badge">READ-ONLY</span><span class="badge">LIVE-FIRST</span>'
    '<span class="badge">STALE-DATA GATED</span>',
    unsafe_allow_html=True,
)
st.markdown('<div class="hero">SPORTS EDGE <span class="good">//</span></div>', unsafe_allow_html=True)
st.caption("Live sports + Kalshi intelligence, simplified for mobile.")


@st.cache_data(ttl=5, show_spinner=False)
def get_kalshi_markets():
    try:
        r = KalshiPublicClient().markets(status="open", limit=200)
        return r.data.get("markets", []), None, r.latency_ms
    except Exception as exc:
        return [], str(exc), None


@st.cache_data(ttl=3, show_spinner=False)
def get_nfl():
    try:
        r = NFLClient().scoreboard()
        return r.data, None, r.latency_ms
    except Exception as exc:
        return {}, str(exc), None


@st.cache_data(ttl=3, show_spinner=False)
def get_mlb():
    try:
        r = MLBClient().schedule(datetime.now(timezone.utc).date())
        return r.data, None, r.latency_ms
    except Exception as exc:
        return {}, str(exc), None


def market_probability(market: dict):
    raw = market.get("yes_ask_dollars") or market.get("yes_ask")
    if raw is None:
        return None
    try:
        value = float(raw)
        return value / 100.0 if value > 1 else value
    except Exception:
        return None


def market_rows(markets: list[dict], sport: str):
    out = []
    for market in markets[:100]:
        title = market.get("title") or market.get("subtitle") or market.get("ticker") or "Untitled market"
        blob = f"{title} {market.get('series_ticker', '')} {market.get('ticker', '')}".lower()
        if sport != "All" and sport.lower() not in blob:
            continue
        mp = market_probability(market)
        if mp is None:
            continue
        out.append(
            {
                "Market": title,
                "Yes": f"{mp:.0%}",
                "Volume": market.get("volume_fp", market.get("volume", 0)),
                "Ticker": market.get("ticker", ""),
            }
        )
    return out


def underdog_rows(markets: list[dict], sport: str):
    out = []
    for market in markets:
        title = market.get("title") or market.get("subtitle") or market.get("ticker") or "Untitled market"
        blob = f"{title} {market.get('series_ticker', '')} {market.get('ticker', '')}".lower()
        if sport not in ("All", "Tennis", "ITF") and sport.lower() not in blob:
            continue
        mp = market_probability(market)
        if mp is not None and 0.03 <= mp <= 0.30:
            out.append(
                {
                    "Market": title,
                    "Market chance": f"{mp:.0%}",
                    "Volume": market.get("volume_fp", market.get("volume", 0)),
                    "Status": "WATCH — model required",
                }
            )
    return out


if "view" not in st.session_state:
    st.session_state.view = "Home"

views = ["Home", "Live Games", "Underdog Radar", "Edge Board", "Parlay Lab", "Market Tape", "Model Trust"]
view = st.selectbox(
    "Where do you want to go?",
    views,
    index=views.index(st.session_state.view),
    key="view_selector",
)
st.session_state.view = view

with st.expander("Filters & settings", expanded=False):
    sport = st.selectbox("Sport", ["All", "NFL", "MLB", "Tennis", "ITF"], key="sport_filter")
    st.caption("Advanced thresholds will matter once validated fair-probability models are connected. The app stays read-only and fails closed on weak data.")

if st.button("↻ Refresh live data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Refreshing live sources…"):
    markets, kerr, klat = get_kalshi_markets()
    nfl, nerr, nlat = get_nfl()
    mlb, merr, mlat = get_mlb()

rows = market_rows(markets, sport)
candidates = underdog_rows(markets, sport)
odds_ready = OddsClient().configured

if view == "Home":
    st.markdown('<div class="nav-hint"><b>Start here.</b> Use the menu above to jump to live games, underdogs, market prices, parlays, or model trust. Nothing here places wagers or Kalshi orders.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Kalshi markets", len(markets) if markets else "—", f"{klat:.0f} ms" if klat else "offline")
    c2.metric("Underdog watches", len(candidates))
    c3, c4 = st.columns(2)
    c3.metric("NFL feed", "LIVE" if nfl else "OFF", f"{nlat:.0f} ms" if nlat else "")
    c4.metric("MLB feed", "LIVE" if mlb else "OFF", f"{mlat:.0f} ms" if mlat else "")
    st.markdown("### Quick status")
    st.write(f"**Sportsbook consensus:** {'Configured' if odds_ready else 'Optional API key not added yet'}")
    st.write(f"**Deployment:** {deployment_mode().replace('_', ' ').title()}")
    if kerr:
        st.warning(f"Kalshi is currently unavailable: {kerr[:180]}")
    if not nfl and nerr:
        st.caption(f"NFL feed: {nerr[:160]}")
    if not mlb and merr:
        st.caption(f"MLB feed: {merr[:160]}")
    st.markdown("### What each screen means")
    st.markdown(
        "**Live Games** = scores/status now  ·  **Underdog Radar** = 3%–30% market-price watchlist  ·  "
        "**Edge Board** = Kalshi market prices  ·  **Parlay Lab** = correlation-aware builder framework  ·  "
        "**Model Trust** = why a signal is or is not trustworthy."
    )

elif view == "Live Games":
    st.header("Live Games")
    st.markdown('<div class="section-note">Current score/status feeds. If a source is stale or unavailable, the app shows that instead of guessing.</div>', unsafe_allow_html=True)
    st.subheader("NFL")
    events = nfl.get("events", []) if isinstance(nfl, dict) else []
    if events:
        for event in events[:12]:
            comp = (event.get("competitions") or [{}])[0]
            teams = comp.get("competitors", [])
            names = [t.get("team", {}).get("displayName", "") for t in teams]
            scores = [t.get("score", "") for t in teams]
            detail = event.get("status", {}).get("type", {}).get("detail", "")
            st.markdown(f"**{' vs '.join(names)}**")
            st.caption(f"{' – '.join(scores)} · {detail}")
            st.divider()
    else:
        st.info(nerr or "No NFL events returned right now.")

    st.subheader("MLB")
    games = []
    for day in mlb.get("dates", []) if isinstance(mlb, dict) else []:
        games.extend(day.get("games", []))
    if games:
        for game in games[:12]:
            away = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "")
            home = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            detail = game.get("status", {}).get("detailedState", "")
            st.markdown(f"**{away} @ {home}**")
            st.caption(detail)
            st.divider()
    else:
        st.info(merr or "No MLB events returned right now.")

elif view == "Underdog Radar":
    st.header("Underdog Radar")
    st.markdown('<div class="section-note">Watches Kalshi outcomes priced at 3%–30%. A low price is not automatically value; model confirmation is required.</div>', unsafe_allow_html=True)
    if candidates:
        st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)
    else:
        st.info("No current candidates in the 3%–30% zone for this filter, or live scanning is unavailable.")

    with st.expander("Manual value check", expanded=False):
        market_p = st.number_input("Market probability", min_value=.01, max_value=.99, value=.12, step=.01)
        fair_p = st.number_input("Validated fair probability", min_value=.01, max_value=.99, value=.24, step=.01)
        dq = st.number_input("Data quality", min_value=0.0, max_value=1.0, value=.90, step=.05)
        age = st.number_input("Source age (seconds)", min_value=0.0, value=1.0, step=1.0)
        sig = score_underdog(market_p, fair_p, data_quality=dq, source_age_s=age)
        m1, m2 = st.columns(2)
        m1.metric("Tier", sig.tier)
        m2.metric("Edge", f"{sig.edge_points:+.1f} pp")
        m3, m4 = st.columns(2)
        m3.metric("EV / 1 contract", f"{sig.ev_per_contract:+.3f}")
        m4.metric("Fair / market", f"{sig.value_multiple:.2f}×")
        if sig.warnings:
            st.warning(" · ".join(sig.warnings))

elif view == "Edge Board":
    st.header("Edge Board")
    st.markdown('<div class="section-note">Current Kalshi market prices. These are not called edges until a validated fair probability is available.</div>', unsafe_allow_html=True)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No matching live markets are available right now.")

elif view == "Parlay Lab":
    st.header("Parlay Lab")
    st.markdown('<div class="section-note">Research framework for high-confidence and longshot combinations. Hidden correlation is penalized and nothing is presented as guaranteed.</div>', unsafe_allow_html=True)
    mode = st.selectbox("Builder", ["High-Confidence Builder", "Longshot Lab (5+ legs)"])
    st.info(f"{mode} is connected to the modeling framework, but live recommended legs stay disabled until the required fair-probability and validation gates are satisfied.")

elif view == "Market Tape":
    st.header("Market Tape")
    st.markdown('<div class="section-note">A simple live snapshot of fetched Kalshi markets, sorted by volume.</div>', unsafe_allow_html=True)
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("Volume", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Market tape is unavailable right now.")

elif view == "Model Trust":
    st.header("Model Trust")
    st.markdown('<div class="section-note">Use this screen to understand whether the data and model evidence are strong enough to trust a signal.</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    t1.metric("Leakage policy", "STRICT")
    t2.metric("Calibration", "REQUIRED")
    t3, t4 = st.columns(2)
    t3.metric("Forced picks", "NEVER")
    t4.metric("Orders", "DISABLED")
    st.markdown("### Promotion gates")
    st.markdown(
        "Walk-forward testing, Brier/log loss, calibration curves, closing-line value, minimum sample size, stale-source exclusion, "
        "lineup/injury revalidation, and sport/market-specific thresholds."
    )
    st.warning("No model is treated as production-grade until it passes those gates prospectively.")

st.divider()
st.caption("Sports Edge is read-only research software. It never submits wagers or Kalshi orders.")
