from __future__ import annotations

from datetime import datetime, timezone
import os

import pandas as pd
import streamlit as st

from sports_edge.core.startup import deployment_mode
from sports_edge.data.kalshi import KalshiPublicClient
from sports_edge.data.mlb import MLBClient
from sports_edge.data.nfl import NFLClient
from sports_edge.data.odds import OddsClient
from sports_edge.models.live_board import (
    LiveSignal,
    build_live_signals,
    build_underdog_signals,
    market_yes_probability,
    unverified_underdog_watchlist,
)
from sports_edge.models.parlay import PRESETS, build_parlay_research, kalshi_copy_ticket
from sports_edge.models.event_slate import (
    PROP_PRESETS,
    game_line_summary,
    kalshi_markets_for_event,
    prop_consensus_rows,
)

st.set_page_config(page_title="Sports Edge", page_icon="◈", layout="wide")

st.markdown(
    """
<style>
.block-container{padding-top:.65rem;max-width:1180px;padding-left:.8rem;padding-right:.8rem}
.stApp{background:radial-gradient(circle at 15% 0%,#10251d 0,#08110f 38%,#060b0a 100%)}
[data-testid="stMetric"]{background:linear-gradient(180deg,rgba(24,49,41,.72),rgba(10,22,18,.9));border:1px solid #24463a;border-radius:14px;padding:11px}
.badge{display:inline-block;padding:4px 9px;border-radius:999px;border:1px solid #335c4d;background:#12241e;color:#9decc9;font-size:.72rem;margin:0 5px 5px 0}
.hero{font-size:2.2rem;font-weight:850;letter-spacing:-1.1px;line-height:1.05;margin:.2rem 0 .3rem}
.good{color:#62e6a7}.muted{color:#8ba59b}
.section-note{color:#a7bbb3;font-size:.92rem;margin-top:-.3rem;margin-bottom:.8rem}
.nav-hint{padding:.7rem .85rem;border:1px solid #24463a;border-radius:12px;background:rgba(14,31,26,.72);margin:.4rem 0 .8rem}
div[data-testid="stExpander"]{border:1px solid #203c33;border-radius:12px;background:rgba(7,17,14,.55)}
hr{border-color:#173127!important}
@media (max-width:700px){
  .block-container{padding-top:.35rem;padding-left:.55rem;padding-right:.55rem}
  .hero{font-size:1.8rem;letter-spacing:-.6px}
  [data-testid="stMetric"]{padding:8px}
  .nav-hint{font-size:.88rem}
  button[kind="secondary"],button[kind="primary"]{min-height:2.7rem}
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="badge">READ-ONLY</span><span class="badge">LIVE-FIRST</span>'
    '<span class="badge">FAIL-CLOSED</span>',
    unsafe_allow_html=True,
)
st.markdown('<div class="hero">SPORTS EDGE <span class="good">//</span></div>', unsafe_allow_html=True)
st.caption("Live sports + Kalshi intelligence. No orders. No forced picks.")


def _secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
        if value:
            value = str(value).strip()
            if value and "YOUR_REAL_API_KEY_HERE" not in value and "YOUR_" not in value:
                return value
    except Exception:
        pass
    value = (os.getenv(name) or "").strip()
    if value and "YOUR_REAL_API_KEY_HERE" not in value and "YOUR_" not in value:
        return value
    return None


@st.cache_data(ttl=8, show_spinner=False)
def get_kalshi_markets(max_pages: int = 4):
    client = KalshiPublicClient()
    found: dict[str, dict] = {}
    latencies: list[float] = []
    errors: list[str] = []

    cursor = None
    for _ in range(max_pages):
        try:
            r = client.markets(status="open", limit=200, cursor=cursor)
            latencies.append(r.latency_ms)
            payload = r.data if isinstance(r.data, dict) else {}
            for market in payload.get("markets", []) or []:
                ticker = str(market.get("ticker") or "")
                if ticker:
                    found[ticker] = market
            cursor = payload.get("cursor")
            if not cursor:
                break
        except Exception as exc:
            errors.append(str(exc))
            break

    cursor = None
    for _ in range(2):
        try:
            r = client.events(status="open", limit=200, cursor=cursor, with_nested_markets=True)
            latencies.append(r.latency_ms)
            payload = r.data if isinstance(r.data, dict) else {}
            for event in payload.get("events", []) or []:
                for market in event.get("markets", []) or []:
                    ticker = str(market.get("ticker") or "")
                    if ticker:
                        market = dict(market)
                        market.setdefault("event_ticker", event.get("event_ticker") or event.get("ticker"))
                        market.setdefault("event_title", event.get("title"))
                        found[ticker] = market
            cursor = payload.get("cursor")
            if not cursor:
                break
        except Exception as exc:
            errors.append(str(exc))
            break

    latency = sum(latencies) if latencies else None
    return list(found.values()), (" | ".join(errors[:2]) if errors else None), latency


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


@st.cache_data(ttl=45, show_spinner=False)
def get_odds_sports(api_key: str):
    try:
        r = OddsClient(api_key=api_key).sports()
        return (r.data if isinstance(r.data, list) else []), None
    except Exception as exc:
        msg = str(exc)
        if "apiKey=" in msg:
            msg = msg.split("apiKey=", 1)[0] + "apiKey=REDACTED"
        return [], msg


@st.cache_data(ttl=20, show_spinner=False)
def get_odds_for_key(api_key: str, sport_key: str):
    try:
        markets = "h2h" if sport_key.startswith("tennis_") else "h2h,spreads,totals"
        r = OddsClient(api_key=api_key).odds(sport_key=sport_key, markets=markets)
        return (r.data if isinstance(r.data, list) else []), None, r.latency_ms
    except Exception as exc:
        msg = str(exc)
        if "apiKey=" in msg:
            msg = msg.split("apiKey=", 1)[0] + "apiKey=REDACTED"
        return [], msg, None


@st.cache_data(ttl=20, show_spinner=False)
def get_event_props(api_key: str, sport_key: str, event_id: str, market_keys: tuple[str, ...]):
    try:
        r = OddsClient(api_key=api_key).event_odds(
            sport_key=sport_key,
            event_id=event_id,
            markets=",".join(market_keys),
        )
        return (r.data if isinstance(r.data, dict) else {}), None, r.latency_ms
    except Exception as exc:
        msg = str(exc)
        if "apiKey=" in msg:
            msg = msg.split("apiKey=", 1)[0] + "apiKey=REDACTED"
        return {}, msg, None


def infer_market_sport(market: dict) -> str:
    text = " ".join(
        str(market.get(k) or "")
        for k in ("title", "subtitle", "ticker", "series_ticker", "event_ticker", "event_title", "category")
    ).lower()
    if any(k in text for k in ("tennis", " atp", " wta", "itf", "challenger")):
        return "Tennis"
    if any(k in text for k in ("nfl", "football")):
        return "NFL"
    if any(k in text for k in ("mlb", "baseball")):
        return "MLB"
    return "Sports"


def filter_markets(markets: list[dict], sport: str) -> list[dict]:
    if sport == "All":
        return markets
    # Qualification must know the Kalshi market belongs to the same sport.
    # Unknown/generic markets stay visible in Market Tape but never enter the
    # cross-source edge matcher.
    return [m for m in markets if infer_market_sport(m) == sport]


def odds_keys_for_sport(sport: str, active_sports: list[dict]) -> list[tuple[str, str]]:
    if sport == "NFL":
        return [("NFL", "americanfootball_nfl")]
    if sport == "MLB":
        return [("MLB", "baseball_mlb")]

    tennis = []
    for item in active_sports:
        key = str(item.get("key") or "")
        group = str(item.get("group") or "")
        title = str(item.get("title") or key)
        if group.lower() == "tennis" or key.startswith("tennis_"):
            tennis.append((title, key))
    tennis = tennis[:10]
    if sport == "Tennis":
        return tennis

    all_keys = [("NFL", "americanfootball_nfl"), ("MLB", "baseball_mlb")]
    all_keys.extend(tennis[:6])
    return all_keys


def build_signal_bundle(markets: list[dict], sport: str, api_key: str | None):
    if not api_key:
        return [], [], [], ["THE_ODDS_API_KEY is not configured"], []

    active_sports, sports_err = get_odds_sports(api_key)
    errors = [sports_err] if sports_err else []
    all_signals: list[LiveSignal] = []
    underdogs: list[LiveSignal] = []
    odds_events: list[dict] = []
    source_notes: list[str] = []

    for label, key in odds_keys_for_sport(sport, active_sports):
        events, err, latency = get_odds_for_key(api_key, key)
        if err:
            errors.append(f"{label}: {err}")
            continue
        if not events:
            continue
        odds_events.extend(events)
        source_notes.append(f"{label}: {len(events)} event(s){f' · {latency:.0f} ms' if latency else ''}")

        signal_sport = "Tennis" if key.startswith("tennis_") else ("NFL" if "nfl" in key else "MLB")

        # Event-first resolution: only Kalshi contracts attached to the same
        # scheduled/live event can enter pricing. Futures/outrights never reach
        # the signal engine.
        for event in events:
            relevant = kalshi_markets_for_event(event, markets)
            if not relevant:
                continue
            all_signals.extend(build_live_signals(relevant, [event], sport=signal_sport))
            if signal_sport == "Tennis":
                underdogs.extend(build_underdog_signals(relevant, [event], sport="Tennis"))

    dedup: dict[tuple[str, str], LiveSignal] = {}
    for signal in all_signals:
        key = (signal.ticker, signal.selection)
        previous = dedup.get(key)
        if previous is None or signal.confidence > previous.confidence:
            dedup[key] = signal

    underdog_dedup: dict[tuple[str, str], LiveSignal] = {}
    for signal in underdogs:
        key = (signal.ticker, signal.selection)
        previous = underdog_dedup.get(key)
        if previous is None or signal.confidence > previous.confidence:
            underdog_dedup[key] = signal

    return (
        sorted(dedup.values(), key=lambda s: (s.status == "QUALIFIED", s.edge_points, s.confidence), reverse=True),
        sorted(underdog_dedup.values(), key=lambda s: (s.status == "QUALIFIED", s.edge_points, s.confidence), reverse=True),
        odds_events,
        [e for e in errors if e],
        source_notes,
    )


def signal_table(signals: list[LiveSignal]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Status": s.status,
                "Sport": s.sport,
                "Side": s.side,
                "Selection": s.selection,
                "Market": s.market,
                "Kalshi": f"{s.market_probability:.1%}",
                "Fair": f"{s.fair_probability:.1%}",
                "Edge": f"{s.edge_points:+.1f} pp",
                "Books": s.book_count,
                "Age": f"{s.source_age_s:.0f}s",
                "Quality": f"{s.data_quality:.0%}",
            }
            for s in signals
        ]
    )


views = ["Games", "Props", "Edge Board", "Parlay Lab", "Underdog Radar", "Live Feed", "Market Tape", "Model Trust"]
if st.session_state.get("view") not in views:
    st.session_state.view = "Games"
view = st.selectbox("Go to", views, index=views.index(st.session_state.view), key="view_selector")
st.session_state.view = view

with st.expander("Filters & settings", expanded=False):
    sport = st.selectbox("Sport", ["All", "NFL", "MLB", "Tennis"], key="sport_filter")
    st.caption("Signals only qualify when independent pricing, freshness, data quality, and edge gates pass.")

if st.button("↻ Refresh all live data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Refreshing live sources…"):
    markets, kerr, klat = get_kalshi_markets()
    nfl, nerr, nlat = get_nfl()
    mlb, merr, mlat = get_mlb()
    api_key = _secret("THE_ODDS_API_KEY")
    signals, underdogs, odds_events, odds_errors, odds_notes = build_signal_bundle(markets, sport, api_key)

qualified = [s for s in signals if s.status == "QUALIFIED"]
watches = [s for s in signals if s.status == "WATCH"]
qualified_dogs = [s for s in underdogs if s.status == "QUALIFIED"]

if view == "Games":
    st.markdown(
        '<div class="nav-hint"><b>Actual games only.</b> This slate comes from current live/upcoming sportsbook events. '
        'Futures, championships, awards, season-long questions, and unrelated Kalshi markets are excluded.</div>',
        unsafe_allow_html=True,
    )

    if not api_key:
        st.warning("Add THE_ODDS_API_KEY in Streamlit Secrets to load the current game slate.")
    elif not odds_events:
        st.info("No current live/upcoming events were returned for this sport filter.")
    else:
        st.markdown("### Live & upcoming games")
        events_sorted = sorted(odds_events, key=lambda e: str(e.get("commence_time") or ""))
        for event in events_sorted[:40]:
            event_sport = str(event.get("sport_key") or "")
            display_sport = "NFL" if "nfl" in event_sport else ("MLB" if "mlb" in event_sport else "Tennis")
            if sport != "All" and display_sport != sport:
                continue

            away = str(event.get("away_team") or "")
            home = str(event.get("home_team") or "")
            commence = str(event.get("commence_time") or "")
            lines = game_line_summary(event)
            matched_kalshi = kalshi_markets_for_event(event, markets)

            with st.expander(f"{display_sport} · {away} @ {home}", expanded=False):
                st.caption(f"Start: {commence}")
                m1, m2 = st.columns(2)
                m1.metric(f"{away} ML fair", f"{lines['away_fair']:.1%}" if lines["away_fair"] is not None else "—")
                m2.metric(f"{home} ML fair", f"{lines['home_fair']:.1%}" if lines["home_fair"] is not None else "—")

                s1, s2 = st.columns(2)
                asp = lines["away_spread"]
                hsp = lines["home_spread"]
                s1.metric(f"{away} spread", f"{asp[0]:+g} ({asp[1]:+g})" if asp else "—")
                s2.metric(f"{home} spread", f"{hsp[0]:+g} ({hsp[1]:+g})" if hsp else "—")

                over = lines["total_over"]
                under = lines["total_under"]
                if over or under:
                    st.write(
                        f"**Total:** "
                        f"{f'Over {over[0]:g} ({over[1]:+g})' if over else 'Over —'} · "
                        f"{f'Under {under[0]:g} ({under[1]:+g})' if under else 'Under —'}"
                    )

                st.write(f"**Sportsbooks:** {lines['book_count']} · **Matched Kalshi game/prop markets:** {len(matched_kalshi)}")
                if matched_kalshi:
                    rows = []
                    for km in matched_kalshi[:12]:
                        rows.append({
                            "Kalshi market": km.get("title") or km.get("subtitle") or km.get("ticker"),
                            "YES": f"{market_yes_probability(km):.1%}" if market_yes_probability(km) is not None else "—",
                            "Ticker": km.get("ticker", ""),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if odds_notes:
        st.caption("Feed refresh: " + " • ".join(odds_notes))
    if kerr:
        st.warning(f"Kalshi warning: {kerr[:220]}")

elif view == "Props":
    st.header("Game & Player Props")
    st.markdown(
        '<div class="section-note">Choose a real game first, then load only that game’s current prop markets. '
        'This avoids futures and saves free-tier API credits.</div>',
        unsafe_allow_html=True,
    )

    prop_events = [
        e for e in odds_events
        if ("nfl" in str(e.get("sport_key") or "") or "mlb" in str(e.get("sport_key") or ""))
        and (sport == "All" or ("NFL" if "nfl" in str(e.get("sport_key") or "") else "MLB") == sport)
    ]
    if not api_key:
        st.warning("Sportsbook API key required for event props.")
    elif not prop_events:
        st.info("No NFL/MLB live or upcoming games are available for the current filter.")
    else:
        labels = {
            f"{e.get('away_team','')} @ {e.get('home_team','')} · {e.get('commence_time','')}": e
            for e in prop_events
        }
        selected_label = st.selectbox("Game", list(labels))
        selected_event = labels[selected_label]
        selected_sport = "NFL" if "nfl" in str(selected_event.get("sport_key") or "") else "MLB"
        available_presets = [p for p in PROP_PRESETS if p.startswith(selected_sport)]
        preset = st.selectbox("Prop category", available_presets)
        st.caption("Props are fetched only for this selected game/category.")

        payload, prop_err, prop_latency = get_event_props(
            api_key,
            str(selected_event.get("sport_key") or ""),
            str(selected_event.get("id") or ""),
            PROP_PRESETS[preset],
        )
        if prop_err:
            st.warning("Prop feed: " + prop_err)
        else:
            prop_rows = prop_consensus_rows(payload, PROP_PRESETS[preset])
            if prop_rows:
                st.dataframe(pd.DataFrame(prop_rows), use_container_width=True, hide_index=True)
                st.caption(f"{len(prop_rows)} current prop outcomes · {f'{prop_latency:.0f} ms' if prop_latency else 'fresh'}")
            else:
                st.info("No current bookmaker prices were returned for this game/category.")

        matched = kalshi_markets_for_event(selected_event, markets)
        if matched:
            st.markdown("### Kalshi markets for this game")
            kalshi_rows = []
            for km in matched[:30]:
                kalshi_rows.append({
                    "Market": km.get("title") or km.get("subtitle") or km.get("ticker"),
                    "YES": f"{market_yes_probability(km):.1%}" if market_yes_probability(km) is not None else "—",
                    "Ticker": km.get("ticker", ""),
                })
            st.dataframe(pd.DataFrame(kalshi_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No confidently matched Kalshi market was found for this exact game. Nothing unrelated will be substituted.")

elif view == "Edge Board":
    st.header("Edge Board")
    st.markdown(
        '<div class="section-note">Only actual scheduled/live event markets. Kalshi price is compared with fresh no-vig sportsbook consensus after exact game identity is resolved; futures and season-long markets are excluded.</div>',
        unsafe_allow_html=True,
    )
    if not api_key:
        st.warning("Edge qualification needs sportsbook consensus. Add THE_ODDS_API_KEY in Streamlit Secrets. Raw Kalshi prices remain visible in Market Tape.")
    elif not signals:
        st.info("No Kalshi markets could be confidently matched to a fresh two-way sportsbook event right now.")
    else:
        show = qualified + [s for s in watches if s not in qualified]
        if show:
            st.dataframe(signal_table(show[:40]), use_container_width=True, hide_index=True)
        else:
            st.info("No positive consensus edges are available after validation gates.")

        for s in show[:8]:
            with st.expander(f"{s.status} · {s.selection} · {s.edge_points:+.1f} pp"):
                st.write(f"**Market:** {s.market}")
                st.write(f"**Event:** {s.event_title or s.event_id}")
                st.write(f"**Kalshi side:** {s.side} · **price:** {s.market_probability:.1%} · **No-vig fair:** {s.fair_probability:.1%}")
                st.write(f"**Fresh books:** {s.book_count} · **Consensus age:** {s.source_age_s:.0f}s · **Data quality:** {s.data_quality:.0%}")
                if s.reasons:
                    st.caption("Evidence: " + " · ".join(s.reasons))
                if s.warnings:
                    st.warning("Gate warnings: " + " · ".join(s.warnings))
                st.caption("Manual Kalshi entry — tap the copy icon, then search the ticker in Kalshi and recheck the live price.")
                st.code(kalshi_copy_ticket([s]), language=None)

elif view == "Underdog Radar":
    st.header("Tennis Underdog Radar")
    st.markdown(
        '<div class="section-note">Kalshi YES outcomes in the 3%–30% zone. Qualification requires a fresh independent no-vig comparison; cheap price alone is never treated as value.</div>',
        unsafe_allow_html=True,
    )

    if underdogs:
        st.dataframe(signal_table(underdogs[:40]), use_container_width=True, hide_index=True)
    else:
        raw = unverified_underdog_watchlist([m for m in markets if infer_market_sport(m) == "Tennis"])
        if raw:
            df = pd.DataFrame(
                [
                    {
                        "Status": "UNVERIFIED",
                        "Market": r["market"],
                        "Kalshi": f"{r['market_probability']:.1%}",
                        "Volume": r["volume"],
                    }
                    for r in raw[:30]
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.warning("These are price-zone watches only. They are not picks until an independent fair probability and tennis-model validation are available.")
        else:
            st.info("No current tennis contracts were found in the 3%–30% zone.")

    st.caption("Surface Elo, serve/return, fatigue/travel, level transitions, and separate ITF calibration remain required before model-based tennis qualification is promoted.")

elif view == "Parlay Lab":
    st.header("Parlay Lab")
    st.markdown(
        '<div class="section-note">Builds only from independently qualified legs. Same-event duplicates are removed. Joint probability is shown as an independence benchmark until sport-specific correlation models are validated.</div>',
        unsafe_allow_html=True,
    )
    mode_label = st.selectbox("Builder", ["High-Confidence Builder", "Longshot Lab (5+ legs)"])
    preset = st.selectbox(
        "Parlay type",
        PRESETS,
        help="Choose MLB hits/HR/K, NFL passing/rushing/receiving/TD/game markets, or mix qualified sports.",
    )
    mode = "longshot" if mode_label.startswith("Longshot") else "high_confidence"
    min_legs = 5 if mode == "longshot" else 2
    default_legs = 6 if mode == "longshot" else 4
    max_legs = 10 if mode == "longshot" else 8
    leg_count = st.slider("Target legs", min_value=min_legs, max_value=max_legs, value=default_legs)
    parlay = build_parlay_research(signals, mode=mode, preset=preset, leg_count=leg_count)

    p1, p2 = st.columns(2)
    p1.metric("Qualified legs", len(parlay.legs))
    p2.metric("Correlation risk", parlay.correlation_risk)
    p3, p4 = st.columns(2)
    p3.metric("Independence fair", f"{parlay.independent_fair_probability:.2%}" if parlay.legs else "—")
    p4.metric("Value multiple", f"{parlay.value_multiple:.2f}×" if parlay.legs else "—")

    if parlay.legs:
        st.dataframe(signal_table(list(parlay.legs)), use_container_width=True, hide_index=True)
        st.markdown("### Copy to Kalshi")
        st.caption("This is a manual-entry slip, not an order. Tap the code-block copy icon, paste/search each ticker in Kalshi, and recheck the live price before entering anything.")
        st.code(kalshi_copy_ticket(parlay.legs), language=None)
    if parlay.warnings:
        st.warning(" · ".join(parlay.warnings))
    if not parlay.legs:
        st.info("No combination is shown because there are not enough independently qualified legs. The app does not fill space with forced picks.")

elif view == "Live Feed":
    st.header("Live Games")
    st.markdown('<div class="section-note">Current public score/status feeds. Missing or unavailable data is shown explicitly.</div>', unsafe_allow_html=True)

    st.subheader("NFL")
    events = nfl.get("events", []) if isinstance(nfl, dict) else []
    if events:
        for event in events[:14]:
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
        for game in games[:14]:
            away = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "")
            home = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            away_score = game.get("teams", {}).get("away", {}).get("score", "")
            home_score = game.get("teams", {}).get("home", {}).get("score", "")
            detail = game.get("status", {}).get("detailedState", "")
            st.markdown(f"**{away} @ {home}**")
            score_text = f"{away_score} – {home_score}" if away_score != "" or home_score != "" else ""
            st.caption(f"{score_text} · {detail}".strip(" ·"))
            st.divider()
    else:
        st.info(merr or "No MLB events returned right now.")

elif view == "Market Tape":
    st.header("Kalshi Market Tape")
    st.markdown('<div class="section-note">Raw current market prices. This is market data, not a recommendation.</div>', unsafe_allow_html=True)
    raw_rows = []
    for market in filter_markets(markets, sport)[:250]:
        p = market_yes_probability(market)
        if p is None:
            continue
        raw_rows.append(
            {
                "Sport": infer_market_sport(market),
                "Market": market.get("title") or market.get("subtitle") or market.get("ticker"),
                "YES": f"{p:.1%}",
                "Volume": market.get("volume_fp", market.get("volume", 0)),
                "Ticker": market.get("ticker", ""),
            }
        )
    if raw_rows:
        st.dataframe(pd.DataFrame(raw_rows).sort_values("Volume", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No matching priced markets are available right now.")

elif view == "Model Trust":
    st.header("Model Trust")
    st.markdown('<div class="section-note">What the app is willing to call qualified right now — and what it is not.</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    t1.metric("Real orders", "DISABLED")
    t2.metric("Forced picks", "NEVER")
    t3, t4 = st.columns(2)
    t3.metric("Sportsbook consensus", "ON" if api_key and odds_events else "OFF")
    t4.metric("Tennis model", "RESEARCH")
    st.markdown("### Current qualification")
    st.write("**Edge Board:** fresh no-vig multi-book consensus versus live Kalshi price, with match-confidence, disagreement, freshness, and data-quality gates.")
    st.write("**Underdog Radar:** same pricing gate plus the 3%–30% Kalshi band. Full surface/serve-return/ITF model validation is not yet promoted.")
    st.write("**Parlay Lab:** only qualified legs; same-event duplicates removed; independence probability clearly labeled until validated correlation models are connected.")
    st.warning("A sportsbook consensus is an independent market benchmark, not proof of outcome. No signal is guaranteed.")

st.divider()
st.caption("Sports Edge is read-only research software. It never submits wagers or Kalshi orders.")
