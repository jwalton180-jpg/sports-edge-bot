from __future__ import annotations
import os
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from sports_edge.data.kalshi import KalshiPublicClient
from sports_edge.data.mlb import MLBClient
from sports_edge.data.nfl import NFLClient
from sports_edge.data.odds import OddsClient
from sports_edge.models.edge import grade_edge, should_surface
from sports_edge.models.underdog import score_underdog
from sports_edge.core.startup import deployment_mode

st.set_page_config(page_title="SPORTS EDGE // Terminal", page_icon="◈", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.1rem;max-width:1550px}.stApp{background:radial-gradient(circle at 15% 0%,#10251d 0,#08110f 38%,#060b0a 100%)}
[data-testid="stMetric"]{background:linear-gradient(180deg,rgba(24,49,41,.72),rgba(10,22,18,.9));border:1px solid #24463a;border-radius:14px;padding:12px}
.edge-card{border:1px solid #27483c;border-radius:16px;padding:16px;background:linear-gradient(145deg,rgba(18,39,32,.92),rgba(8,18,15,.96));box-shadow:0 8px 30px rgba(0,0,0,.22)}
.badge{display:inline-block;padding:4px 9px;border-radius:999px;border:1px solid #335c4d;background:#12241e;color:#9decc9;font-size:.76rem;margin-right:6px}
.huge{font-size:2.6rem;font-weight:800;letter-spacing:-1px}.muted{color:#8ba59b}.good{color:#62e6a7}.warn{color:#ffd477}.bad{color:#ff8e8e}
hr{border-color:#173127!important}
</style>
""", unsafe_allow_html=True)

st.markdown('<span class="badge">READ-ONLY</span><span class="badge">LIVE-FIRST</span><span class="badge">STALE-DATA GATED</span>', unsafe_allow_html=True)
st.markdown('<div class="huge">SPORTS EDGE <span class="good">//</span> Terminal</div>', unsafe_allow_html=True)
st.caption("Kalshi-first pricing intelligence • Tennis / ITF • MLB • NFL • Props • Underdogs • Parlays")
st.caption(f"Deployment: {deployment_mode().replace(chr(95), chr(32)).upper()} · session cache is disposable in on-demand mode")

sport = st.sidebar.selectbox("Sport", ["All", "NFL", "MLB", "Tennis", "ITF"])
min_edge = st.sidebar.slider("Minimum model edge", 0.0, 15.0, 3.0, .5)
min_conf = st.sidebar.slider("Minimum confidence", 0.0, 1.0, .60, .05)
st.sidebar.markdown("---")
st.sidebar.caption("Credentials remain optional for public market scanning. WebSocket/orderbook auth can be added through environment variables without placing orders.")

@st.cache_data(ttl=5, show_spinner=False)
def get_kalshi_markets():
    try:
        r = KalshiPublicClient().markets(status="open", limit=200)
        return r.data.get("markets", []), None, r.latency_ms
    except Exception as e: return [], str(e), None

@st.cache_data(ttl=3, show_spinner=False)
def get_nfl():
    try:
        r=NFLClient().scoreboard(); return r.data, None, r.latency_ms
    except Exception as e:return {},str(e),None

@st.cache_data(ttl=3, show_spinner=False)
def get_mlb():
    try:
        r=MLBClient().schedule(datetime.now(timezone.utc).date()); return r.data,None,r.latency_ms
    except Exception as e:return {},str(e),None

markets, kerr, klat = get_kalshi_markets()
nfl, nerr, nlat = get_nfl()
mlb, merr, mlat = get_mlb()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Kalshi open markets", len(markets) if markets else "—", f"{klat:.0f} ms" if klat else "offline")
c2.metric("NFL feed", "LIVE" if nfl else "OFF", f"{nlat:.0f} ms" if nlat else (nerr or ""))
c3.metric("MLB feed", "LIVE" if mlb else "OFF", f"{mlat:.0f} ms" if mlat else (merr or ""))
c4.metric("Odds consensus", "READY" if OddsClient().configured else "OPTIONAL KEY", "no scraping")

tabs=st.tabs(["Edge Board","Underdog Radar","Live Action","Parlay Lab","Market Tape","Model Trust"])
with tabs[0]:
    st.subheader("Edge Board")
    st.caption("A market only becomes actionable after an external/model fair probability exists. Raw Kalshi price alone is never called an edge.")
    rows=[]
    for m in markets[:80]:
        title=(m.get("title") or m.get("subtitle") or "").lower()
        if sport!="All" and sport.lower() not in title and sport.lower() not in str(m.get("series_ticker","")).lower(): continue
        yp=m.get("yes_ask_dollars") or m.get("yes_ask")
        if yp is None: continue
        try: mp=float(yp); mp=mp/100 if mp>1 else mp
        except: continue
        rows.append({"ticker":m.get("ticker"),"market":m.get("title") or m.get("subtitle"),"kalshi_yes":mp,"volume":m.get("volume_fp",m.get("volume",0)),"status":m.get("status","open")})
    if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else: st.info("Live market scan unavailable in this runtime or no matching markets. The app fails closed rather than showing invented edges.")

with tabs[1]:
    st.subheader("Underdog Radar")
    st.caption("3%–30% Kalshi probability zone. Cheap is not the same as value: candidates stay WATCH-only until a calibrated sport model supplies fair probability.")
    cand=[]
    for m in markets:
        title=(m.get("title") or m.get("subtitle") or "")
        blob=(title+" "+str(m.get("series_ticker","")+" "+m.get("ticker",""))).lower()
        if sport not in ("All","Tennis","ITF") and sport.lower() not in blob: continue
        yp=m.get("yes_ask_dollars") or m.get("yes_ask")
        if yp is None: continue
        try:
            mp=float(yp); mp=mp/100 if mp>1 else mp
        except Exception:
            continue
        if .03 <= mp <= .30:
            cand.append({"ticker":m.get("ticker"),"market":title,"market_p":mp,"volume":m.get("volume_fp",m.get("volume",0)),"status":"WATCH — MODEL REQUIRED"})
    if cand:
        df=pd.DataFrame(cand).sort_values(["market_p","volume"],ascending=[True,False])
        st.dataframe(df,use_container_width=True,hide_index=True)
    else:
        st.info("No current 3%–30% candidates in the fetched market page, or live scan is unavailable.")
    st.markdown("#### Value sanity check")
    u1,u2,u3,u4=st.columns(4)
    market_p=u1.number_input("Market probability",min_value=.01,max_value=.99,value=.12,step=.01)
    fair_p=u2.number_input("Validated fair probability",min_value=.01,max_value=.99,value=.24,step=.01)
    dq=u3.number_input("Data quality",min_value=0.0,max_value=1.0,value=.90,step=.05)
    age=u4.number_input("Source age (s)",min_value=0.0,value=1.0,step=1.0)
    sig=score_underdog(market_p,fair_p,data_quality=dq,source_age_s=age)
    a,b,c,d=st.columns(4)
    a.metric("Tier",sig.tier); b.metric("Edge",f"{sig.edge_points:+.1f} pp"); c.metric("EV / $1 contract",f"${sig.ev_per_contract:+.3f}"); d.metric("Fair / market",f"{sig.value_multiple:.2f}×")
    if sig.warnings: st.warning(" · ".join(sig.warnings))

with tabs[2]:
    st.subheader("Live Action")
    l1,l2=st.columns(2)
    with l1:
        st.markdown("#### NFL")
        events=nfl.get("events",[]) if isinstance(nfl,dict) else []
        if events:
            for e in events[:10]:
                comp=(e.get("competitions") or [{}])[0]; teams=comp.get("competitors",[])
                names=[t.get("team",{}).get("displayName","") for t in teams]
                scores=[t.get("score","") for t in teams]
                st.markdown(f"**{' vs '.join(names)}**  ·  {' – '.join(scores)}  ·  {e.get('status',{}).get('type',{}).get('detail','')}")
        else: st.caption(nerr or "No NFL events returned.")
    with l2:
        st.markdown("#### MLB")
        games=[]
        for d in mlb.get("dates",[]) if isinstance(mlb,dict) else []: games += d.get("games",[])
        if games:
            for g in games[:10]:
                a=g.get("teams",{}).get("away",{}).get("team",{}).get("name","")
                h=g.get("teams",{}).get("home",{}).get("team",{}).get("name","")
                st.markdown(f"**{a} @ {h}** · {g.get('status',{}).get('detailedState','')}")
        else: st.caption(merr or "No MLB events returned.")

with tabs[3]:
    st.subheader("Parlay Lab")
    mode=st.radio("Mode",["High-Confidence Builder","Longshot Lab (5+ legs)"],horizontal=True)
    st.write("The optimizer rejects duplicate outcomes and penalizes hidden correlation. It will not call any parlay guaranteed.")
    demo=pd.DataFrame([
      ["Leg A",.72,.68,"+4.0 pp"],["Leg B",.66,.61,"+5.0 pp"],["Leg C",.63,.60,"+3.0 pp"],
      ["Leg D",.59,.54,"+5.0 pp"],["Leg E",.56,.51,"+5.0 pp"]],columns=["Leg","Fair P","Market P","Edge"])
    st.dataframe(demo,use_container_width=True,hide_index=True)
    st.caption("Illustrative layout only until live fair-probability models are trained and validated. No fake live picks are injected into the interface.")

with tabs[4]:
    st.subheader("Market Tape")
    st.caption("Live Kalshi REST snapshot in v0.1; authenticated WebSocket transport is isolated for the next pass.")
    if rows: st.dataframe(pd.DataFrame(rows).sort_values("volume",ascending=False),use_container_width=True,hide_index=True)

with tabs[5]:
    st.subheader("Model Trust")
    t1,t2,t3=st.columns(3); t1.metric("Leakage policy","STRICT");t2.metric("Calibration","REQUIRED");t3.metric("Forced picks","NEVER")
    st.markdown("**Promotion gates:** walk-forward test only; Brier/log loss; calibration curve; ROI and closing-line value by edge bucket; minimum sample size; stale-source exclusion; lineup/injury revalidation; sport/market-specific thresholds.")
    st.warning("No model is labeled production-grade until it passes those gates prospectively. An 80% target is evaluated honestly, not manufactured by excluding losses after the fact.")

if kerr:
    st.caption(f"Kalshi transport note: {kerr[:180]}")
