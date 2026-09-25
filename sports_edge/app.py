from __future__ import annotations

from datetime import datetime, timezone
import os
from statistics import median

import pandas as pd
import streamlit as st

from sports_edge.core.startup import deployment_mode
from sports_edge.data.kalshi import KalshiPublicClient
from sports_edge.data.mlb import MLBClient
from sports_edge.data.nfl import NFLClient
from sports_edge.data.odds import OddsClient
from sports_edge.models.game_scope import GameEvent, build_game_events, game_scoped_markets
from sports_edge.models.live_board import LiveSignal, build_live_signals, build_underdog_signals, market_yes_probability
from sports_edge.models.parlay import PRESETS, build_parlay_research, kalshi_copy_ticket
from sports_edge.models.prop_edges import build_prop_signals
from sports_edge.models.props import PROP_GROUPS, prop_consensus

st.set_page_config(page_title="Sports Edge", page_icon="◈", layout="wide")

st.markdown(
    """
<style>
.block-container{padding-top:.55rem;max-width:1180px;padding-left:.7rem;padding-right:.7rem}
.stApp{background:radial-gradient(circle at 15% 0%,#10251d 0,#08110f 38%,#060b0a 100%)}
[data-testid="stMetric"]{background:linear-gradient(180deg,rgba(24,49,41,.72),rgba(10,22,18,.9));border:1px solid #24463a;border-radius:14px;padding:10px}
.badge{display:inline-block;padding:4px 9px;border-radius:999px;border:1px solid #335c4d;background:#12241e;color:#9decc9;font-size:.72rem;margin:0 5px 5px 0}
.hero{font-size:2.05rem;font-weight:850;letter-spacing:-1px;line-height:1.05;margin:.15rem 0 .25rem}
.good{color:#62e6a7}.section-note{color:#a7bbb3;font-size:.92rem;margin-top:-.25rem;margin-bottom:.8rem}
.nav-hint{padding:.7rem .85rem;border:1px solid #24463a;border-radius:12px;background:rgba(14,31,26,.72);margin:.35rem 0 .7rem}
.game-card{padding:.72rem .78rem;border:1px solid #24463a;border-radius:13px;background:rgba(12,28,23,.72);margin:.45rem 0}
.game-title{font-weight:760;font-size:1.02rem}.live{color:#62e6a7;font-weight:800}.soon{color:#f1cf6d;font-weight:800}
div[data-testid="stExpander"]{border:1px solid #203c33;border-radius:12px;background:rgba(7,17,14,.55)}
hr{border-color:#173127!important}
@media (max-width:700px){
 .block-container{padding-top:.3rem;padding-left:.5rem;padding-right:.5rem}
 .hero{font-size:1.72rem}
 [data-testid="stMetric"]{padding:8px}
 button[kind="secondary"],button[kind="primary"]{min-height:2.7rem}
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="badge">GAME-FIRST</span><span class="badge">READ-ONLY</span>'
    '<span class="badge">NO FUTURES</span>',
    unsafe_allow_html=True,
)
st.markdown('<div class="hero">SPORTS EDGE <span class="good">//</span></div>', unsafe_allow_html=True)
st.caption("Actual games, game lines, player props, Kalshi contracts, and qualified research signals.")


def _secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
        if value:
            value = str(value).strip()
            if value and "YOUR_" not in value:
                return value
    except Exception:
        pass
    value = (os.getenv(name) or "").strip()
    return value if value and "YOUR_" not in value else None


def _safe_error(exc: Exception) -> str:
    msg = str(exc)
    if "apiKey=" in msg:
        msg = msg.split("apiKey=", 1)[0] + "apiKey=REDACTED"
    return msg[:300]


@st.cache_data(ttl=8, show_spinner=False)
def get_kalshi_markets(max_pages: int = 5):
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
            errors.append(_safe_error(exc))
            break

    cursor = None
    for _ in range(3):
        try:
            r = client.events(status="open", limit=200, cursor=cursor, with_nested_markets=True)
            latencies.append(r.latency_ms)
            payload = r.data if isinstance(r.data, dict) else {}
            for event in payload.get("events", []) or []:
                for market in event.get("markets", []) or []:
                    ticker = str(market.get("ticker") or "")
                    if not ticker:
                        continue
                    row = dict(market)
                    row.setdefault("event_ticker", event.get("event_ticker") or event.get("ticker"))
                    row.setdefault("event_title", event.get("title"))
                    found[ticker] = row
            cursor = payload.get("cursor")
            if not cursor:
                break
        except Exception as exc:
            errors.append(_safe_error(exc))
            break

    return list(found.values()), (" | ".join(errors[:2]) if errors else None), (sum(latencies) if latencies else None)


@st.cache_data(ttl=45, show_spinner=False)
def get_active_sports(api_key: str):
    try:
        r = OddsClient(api_key=api_key).sports()
        return (r.data if isinstance(r.data, list) else []), None
    except Exception as exc:
        return [], _safe_error(exc)


@st.cache_data(ttl=20, show_spinner=False)
def get_events(api_key: str, sport_key: str):
    try:
        r = OddsClient(api_key=api_key).events(sport_key)
        return (r.data if isinstance(r.data, list) else []), None, r.latency_ms
    except Exception as exc:
        return [], _safe_error(exc), None


@st.cache_data(ttl=20, show_spinner=False)
def get_featured_odds(api_key: str, sport_key: str):
    try:
        r = OddsClient(api_key=api_key).odds(sport_key=sport_key, markets="h2h")
        return (r.data if isinstance(r.data, list) else []), None, r.latency_ms
    except Exception as exc:
        return [], _safe_error(exc), None


@st.cache_data(ttl=20, show_spinner=False)
def get_event_odds(api_key: str, sport_key: str, event_id: str, markets: str):
    try:
        r = OddsClient(api_key=api_key).event_odds(sport_key=sport_key, event_id=event_id, markets=markets)
        return (r.data if isinstance(r.data, dict) else {}), None, r.latency_ms
    except Exception as exc:
        return {}, _safe_error(exc), None


@st.cache_data(ttl=3, show_spinner=False)
def get_nfl_live():
    try:
        r = NFLClient().scoreboard()
        return r.data, None
    except Exception as exc:
        return {}, str(exc)


@st.cache_data(ttl=3, show_spinner=False)
def get_mlb_live():
    try:
        r = MLBClient().schedule(datetime.now(timezone.utc).date())
        return r.data, None
    except Exception as exc:
        return {}, str(exc)


def sport_pairs(active_sports: list[dict]) -> list[tuple[str, str]]:
    pairs = [("NFL", "americanfootball_nfl"), ("MLB", "baseball_mlb")]
    tennis = []
    for item in active_sports:
        key = str(item.get("key") or "")
        group = str(item.get("group") or "")
        title = str(item.get("title") or key)
        if group.lower() == "tennis" or key.startswith("tennis_"):
            tennis.append((f"Tennis · {title}", key))
    pairs.extend(tennis[:6])
    return pairs


def build_game_universe(api_key: str | None, active_sports: list[dict]):
    if not api_key:
        return [], {}, ["THE_ODDS_API_KEY is not configured"]

    games: list[GameEvent] = []
    raw_by_id: dict[str, dict] = {}
    errors: list[str] = []
    for label, key in sport_pairs(active_sports):
        raw, err, _ = get_events(api_key, key)
        if err:
            errors.append(f"{label}: {err}")
            continue
        sport = "Tennis" if key.startswith("tennis_") else ("NFL" if "nfl" in key else "MLB")
        built = build_game_events(raw, sport_key=key, sport=sport)
        games.extend(built)
        for event in raw:
            if event.get("id"):
                raw_by_id[str(event["id"])] = event
    return sorted(games, key=lambda g: g.commence_time), raw_by_id, errors


def relative_time(game: GameEvent) -> str:
    now = datetime.now(timezone.utc)
    sec = (game.commence_time - now).total_seconds()
    if sec <= 0:
        mins = int(abs(sec) // 60)
        return f"LIVE / started {mins}m ago"
    mins = int(sec // 60)
    if mins < 60:
        return f"starts in {mins}m"
    return f"starts in {mins // 60}h {mins % 60}m"


def game_label(game: GameEvent) -> str:
    return f"{game.state} · {game.away_team} @ {game.home_team} · {relative_time(game)}"


def game_rows(games: list[GameEvent], scoped: dict[str, list[dict]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "State": g.state,
                "Sport": g.sport,
                "Game": f"{g.away_team} @ {g.home_team}",
                "Start": relative_time(g),
                "Kalshi game contracts": len(scoped.get(g.event_id, [])),
            }
            for g in games
        ]
    )


def kalshi_game_rows(markets: list[dict]) -> pd.DataFrame:
    rows = []
    for market in markets:
        p = market_yes_probability(market)
        rows.append(
            {
                "Market": market.get("title") or market.get("subtitle") or market.get("ticker"),
                "YES": f"{p:.1%}" if p is not None else "—",
                "Volume": market.get("volume_fp", market.get("volume", 0)),
                "Ticker": market.get("ticker", ""),
            }
        )
    return pd.DataFrame(rows)


def featured_line_rows(payload: dict) -> pd.DataFrame:
    samples: dict[tuple[str, str, float | None], list[tuple[float, str]]] = {}
    for bookmaker in payload.get("bookmakers", []) or []:
        bname = str(bookmaker.get("title") or bookmaker.get("key") or "")
        for market in bookmaker.get("markets", []) or []:
            mkey = str(market.get("key") or "")
            for outcome in market.get("outcomes", []) or []:
                try:
                    price = float(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                key = (mkey, str(outcome.get("name") or ""), outcome.get("point"))
                samples.setdefault(key, []).append((price, bname))

    labels = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}
    rows = []
    for (mkey, name, point), values in samples.items():
        rows.append(
            {
                "Market": labels.get(mkey, mkey),
                "Selection": name,
                "Line": point if point is not None else "—",
                "Median odds": int(round(median([v[0] for v in values]))),
                "Books": len(values),
            }
        )
    return pd.DataFrame(rows)


def signal_table(signals: list[LiveSignal]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Status": s.status,
                "Sport": s.sport,
                "Selection": s.selection,
                "Kalshi side": s.side,
                "Kalshi": f"{s.market_probability:.1%}",
                "Fair": f"{s.fair_probability:.1%}",
                "Edge": f"{s.edge_points:+.1f} pp",
                "Books": s.book_count,
                "Age": f"{s.source_age_s:.0f}s",
                "Market": s.market,
            }
            for s in signals
        ]
    )


def build_game_line_signals(markets: list[dict], api_key: str | None, sport_filter: str):
    if not api_key:
        return [], ["THE_ODDS_API_KEY is not configured"]

    active, active_err = get_active_sports(api_key)
    errors = [active_err] if active_err else []
    all_signals: list[LiveSignal] = []

    for label, key in sport_pairs(active):
        sport = "Tennis" if key.startswith("tennis_") else ("NFL" if "nfl" in key else "MLB")
        if sport_filter != "All" and sport != sport_filter:
            continue

        raw, err, _ = get_featured_odds(api_key, key)
        if err:
            errors.append(f"{label}: {err}")
            continue

        games = build_game_events(raw, sport_key=key, sport=sport)
        scoped = game_scoped_markets(markets, games)
        raw_by_id = {str(e.get("id")): e for e in raw if e.get("id")}
        for game in games:
            event = raw_by_id.get(game.event_id)
            if not event:
                continue
            game_markets = scoped.get(game.event_id, [])
            if not game_markets:
                continue
            all_signals.extend(build_live_signals(game_markets, [event], sport=sport))

    dedup: dict[tuple[str, str, str], LiveSignal] = {}
    for signal in all_signals:
        key = (signal.ticker, signal.side, signal.selection)
        old = dedup.get(key)
        if old is None or signal.confidence > old.confidence:
            dedup[key] = signal
    return sorted(dedup.values(), key=lambda s: (s.status == "QUALIFIED", s.edge_points, s.confidence), reverse=True), [e for e in errors if e]


if "view" not in st.session_state:
    st.session_state.view = "Games"
if "prop_signal_cache" not in st.session_state:
    st.session_state.prop_signal_cache = {}

views = ["Games", "Game Lines", "Player Props", "Edge Board", "Parlay Generator", "Live Feed", "Model Trust"]
view = st.selectbox("Go to", views, index=views.index(st.session_state.view), key="main_view")
st.session_state.view = view

if st.button("↻ Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

api_key = _secret("THE_ODDS_API_KEY")
markets, kerr, klat = get_kalshi_markets()
active_sports, active_err = get_active_sports(api_key) if api_key else ([], None)
games, raw_events, game_errors = build_game_universe(api_key, active_sports)
scoped = game_scoped_markets(markets, games)

sport_filter = st.selectbox("Sport", ["All", "MLB", "NFL", "Tennis"], key="sport_filter")
visible_games = [g for g in games if sport_filter == "All" or g.sport == sport_filter]

if view == "Games":
    st.header("Today / Live / Upcoming Games")
    st.markdown(
        '<div class="nav-hint"><b>This is the Sports Edge universe now.</b> Only real scheduled/live matchups appear here. '
        'Championship futures, awards, entertainment, and “before 2030” markets are excluded from the main workflow.</div>',
        unsafe_allow_html=True,
    )

    if not api_key:
        st.warning("Add THE_ODDS_API_KEY in Streamlit Secrets to load the current game universe.")
    elif game_errors and not games:
        st.warning("Game feed unavailable: " + " | ".join(game_errors[:2]))
    elif not visible_games:
        st.info("No current games found for this filter.")
    else:
        live_count = sum(g.state == "LIVE" for g in visible_games)
        soon_count = sum(g.state == "SOON" for g in visible_games)
        game_contract_count = sum(len(scoped.get(g.event_id, [])) for g in visible_games)
        c1, c2 = st.columns(2)
        c1.metric("Games", len(visible_games))
        c2.metric("Live", live_count)
        c3, c4 = st.columns(2)
        c3.metric("Starting ≤6h", soon_count)
        c4.metric("Kalshi game contracts", game_contract_count)
        st.dataframe(game_rows(visible_games, scoped), use_container_width=True, hide_index=True)

    if kerr:
        st.warning(f"Kalshi warning: {kerr}")

elif view == "Game Lines":
    st.header("Game Lines")
    st.markdown('<div class="section-note">Pick an actual game, then load ML / spread / total. Full line loading is on-demand to protect your free API quota.</div>', unsafe_allow_html=True)

    if not visible_games:
        st.info("No games available for this filter.")
    else:
        labels = {game_label(g): g for g in visible_games}
        selected_label = st.selectbox("Game", list(labels))
        game = labels[selected_label]
        game_markets = scoped.get(game.event_id, [])

        st.markdown(f"### {game.away_team} @ {game.home_team}")
        st.caption(f"{game.state} · {relative_time(game)} · Kalshi contracts tied to this game: {len(game_markets)}")

        if game_markets:
            with st.expander("Kalshi contracts for this game", expanded=True):
                st.dataframe(kalshi_game_rows(game_markets), use_container_width=True, hide_index=True)
        else:
            st.info("No Kalshi contracts were found with this exact game identity.")

        market_keys = "h2h,spreads,totals" if game.sport in ("NFL", "MLB") else "h2h"
        if st.button("Load current ML / spread / total", use_container_width=True):
            payload, err, latency = get_event_odds(api_key, game.sport_key, game.event_id, market_keys)
            st.session_state[f"lines_{game.event_id}"] = (payload, err, latency)

        loaded = st.session_state.get(f"lines_{game.event_id}")
        if loaded:
            payload, err, latency = loaded
            if err:
                st.warning(err)
            else:
                df = featured_line_rows(payload)
                if not df.empty:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.caption(f"Current sportsbook snapshot · {latency:.0f} ms" if latency else "Current sportsbook snapshot")
                else:
                    st.info("No featured lines returned for this game.")

elif view == "Player Props":
    st.header("Player Props")
    st.markdown(
        '<div class="section-note">Current event-specific props only. Select one real MLB/NFL game and a prop family; futures are never queried here.</div>',
        unsafe_allow_html=True,
    )

    prop_games = [g for g in visible_games if g.sport in PROP_GROUPS]
    if not prop_games:
        st.info("No MLB/NFL games are available for this filter.")
    else:
        labels = {game_label(g): g for g in prop_games}
        selected_label = st.selectbox("Game", list(labels), key="prop_game")
        game = labels[selected_label]
        groups = PROP_GROUPS[game.sport]
        category = st.selectbox("Prop category", list(groups), key="prop_category")
        keys = groups[category]

        st.caption(f"{game.away_team} @ {game.home_team} · {category} · {len(keys)} API market key(s)")
        if st.button("Load current player props", use_container_width=True):
            payload, err, latency = get_event_odds(api_key, game.sport_key, game.event_id, ",".join(keys))
            quotes = prop_consensus(payload, market_keys=keys) if payload else []
            prop_signals = build_prop_signals(scoped.get(game.event_id, []), game, quotes)
            cache_key = f"{game.event_id}|{category}"
            st.session_state.prop_signal_cache[cache_key] = prop_signals
            st.session_state[f"props_{cache_key}"] = (quotes, err, latency)

        cache_key = f"{game.event_id}|{category}"
        loaded = st.session_state.get(f"props_{cache_key}")
        if loaded:
            quotes, err, latency = loaded
            if err:
                st.warning(err)
            elif quotes:
                prop_rows = [
                    {
                        "Player": q.player,
                        "Prop": q.market_label,
                        "Side": q.side,
                        "Line": q.line if q.line is not None else "—",
                        "Consensus": f"{q.fair_probability:.1%}",
                        "Books": q.book_count,
                        "Age": f"{q.median_age_s:.0f}s",
                        "Median odds": int(round(q.median_price)),
                    }
                    for q in quotes[:120]
                ]
                st.dataframe(pd.DataFrame(prop_rows), use_container_width=True, hide_index=True)
                st.caption(f"Event-specific sportsbook snapshot · {latency:.0f} ms" if latency else "Event-specific sportsbook snapshot")

                prop_signals = st.session_state.prop_signal_cache.get(cache_key, [])
                qualified_props = [s for s in prop_signals if s.status == "QUALIFIED"]
                watches = [s for s in prop_signals if s.status == "WATCH"]
                st.markdown("### Exact Kalshi matches")
                if qualified_props or watches:
                    st.dataframe(signal_table((qualified_props + watches)[:30]), use_container_width=True, hide_index=True)
                else:
                    st.info("No exact player + prop + milestone Kalshi contract matched this sportsbook snapshot. Sports Edge does not force approximate matches.")
            else:
                st.info("No props returned for this game/category.")

elif view == "Edge Board":
    st.header("Game Edge Board")
    st.markdown(
        '<div class="section-note">Only actual current game contracts can qualify here. Futures and unrelated markets are not part of this board.</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Loading current game prices…"):
        signals, signal_errors = build_game_line_signals(markets, api_key, sport_filter)

    qualified = [s for s in signals if s.status == "QUALIFIED"]
    watches = [s for s in signals if s.status == "WATCH"]
    show = qualified + watches
    if signal_errors and not signals:
        st.warning(" | ".join(signal_errors[:2]))
    if show:
        st.dataframe(signal_table(show[:50]), use_container_width=True, hide_index=True)
        for s in show[:8]:
            with st.expander(f"{s.status} · {s.event_title} · {s.selection} · {s.edge_points:+.1f} pp"):
                st.write(f"**Kalshi:** {s.side} {s.market_probability:.1%} · **Consensus fair:** {s.fair_probability:.1%}")
                st.write(f"**Market:** {s.market}")
                st.code(kalshi_copy_ticket([s]), language=None)
    else:
        st.info("No current game contract passes the qualification gates right now.")

elif view == "Parlay Generator":
    st.header("Parlay Generator")
    st.markdown(
        '<div class="section-note">Generate from actual game-scoped qualified legs only. Load player-prop categories first if you want MLB hits/HR/K or NFL passing/rushing/receiving/TD legs included.</div>',
        unsafe_allow_html=True,
    )

    mode_label = st.selectbox("Builder", ["High-Confidence Builder", "Longshot Lab (5+ legs)"])
    preset = st.selectbox("Parlay type", PRESETS)
    mode = "longshot" if mode_label.startswith("Longshot") else "high_confidence"

    game_signals: list[LiveSignal] = []
    if preset in ("Best Available", "Mixed Sports", "NFL Game Markets"):
        with st.spinner("Loading qualified game-line legs…"):
            game_signals, _ = build_game_line_signals(markets, api_key, sport_filter)

    loaded_prop_signals: list[LiveSignal] = []
    for values in st.session_state.prop_signal_cache.values():
        loaded_prop_signals.extend(values)

    pool = game_signals + loaded_prop_signals
    min_legs = 5 if mode == "longshot" else 2
    default_legs = 6 if mode == "longshot" else 4
    max_legs = 10 if mode == "longshot" else 8
    target = st.slider("Target legs", min_value=min_legs, max_value=max_legs, value=default_legs)

    parlay = build_parlay_research(pool, mode=mode, preset=preset, leg_count=target)
    p1, p2 = st.columns(2)
    p1.metric("Qualified legs", len(parlay.legs))
    p2.metric("Correlation risk", parlay.correlation_risk)
    p3, p4 = st.columns(2)
    p3.metric("Independence fair", f"{parlay.independent_fair_probability:.2%}" if parlay.legs else "—")
    p4.metric("Value multiple", f"{parlay.value_multiple:.2f}×" if parlay.legs else "—")

    if parlay.legs:
        st.dataframe(signal_table(list(parlay.legs)), use_container_width=True, hide_index=True)
        st.markdown("### Copy to Kalshi")
        st.code(kalshi_copy_ticket(parlay.legs), language=None)
    else:
        st.info("No parlay generated. Load the prop category you want under Player Props, or wait for enough game-line legs to qualify.")

    if parlay.warnings:
        st.warning(" · ".join(parlay.warnings))

elif view == "Live Feed":
    st.header("Live Feed")
    st.markdown('<div class="section-note">Fast official/public game-state feeds. This page is for what is happening now, not futures.</div>', unsafe_allow_html=True)
    nfl, nerr = get_nfl_live()
    mlb, merr = get_mlb_live()

    st.subheader("NFL")
    events = nfl.get("events", []) if isinstance(nfl, dict) else []
    if events:
        for event in events[:16]:
            comp = (event.get("competitions") or [{}])[0]
            teams = comp.get("competitors", [])
            names = [t.get("team", {}).get("displayName", "") for t in teams]
            scores = [t.get("score", "") for t in teams]
            detail = event.get("status", {}).get("type", {}).get("detail", "")
            st.markdown(f"**{' vs '.join(names)}**")
            st.caption(f"{' – '.join(scores)} · {detail}")
            st.divider()
    else:
        st.info(nerr or "No NFL live events right now.")

    st.subheader("MLB")
    mlb_games = []
    for day in mlb.get("dates", []) if isinstance(mlb, dict) else []:
        mlb_games.extend(day.get("games", []))
    if mlb_games:
        for game in mlb_games[:16]:
            away = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "")
            home = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            away_score = game.get("teams", {}).get("away", {}).get("score", "")
            home_score = game.get("teams", {}).get("home", {}).get("score", "")
            detail = game.get("status", {}).get("detailedState", "")
            st.markdown(f"**{away} @ {home}**")
            st.caption(f"{away_score} – {home_score} · {detail}")
            st.divider()
    else:
        st.info(merr or "No MLB live events right now.")

elif view == "Model Trust":
    st.header("Model Trust")
    t1, t2 = st.columns(2)
    t1.metric("Futures in main workflow", "0")
    t2.metric("Real orders", "DISABLED")
    t3, t4 = st.columns(2)
    t3.metric("Game identity", "BOTH TEAMS")
    t4.metric("Approximate prop matches", "REJECTED")
    st.write("**Game universe:** live/upcoming events from current sports feeds.")
    st.write("**Game-line edges:** exact event identity + current Kalshi price + fresh sportsbook no-vig consensus.")
    st.write("**Player-prop edges:** exact game + full player name + prop family + compatible milestone/line before qualification.")
    st.write("**Parlays:** only qualified game-scoped legs; same-event duplicates are blocked until correlation models are validated.")
    st.warning("No signal or parlay is guaranteed. If identity, freshness, or contract semantics are uncertain, Sports Edge shows nothing.")

st.divider()
st.caption(
    f"Game-first on-demand mode · Kalshi request time {f'{klat:.0f} ms' if klat else '—'} · "
    f"Deployment {deployment_mode().replace('_', ' ').title()}"
)
