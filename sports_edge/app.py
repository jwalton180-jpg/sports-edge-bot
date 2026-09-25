from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib
import inspect
import os
from statistics import median
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from sports_edge.core.startup import deployment_mode
from sports_edge.data.kalshi import KalshiPublicClient
from sports_edge.data.kalshi_catalog import fetch_supported_sport_catalog
from sports_edge.data.mlb import MLBClient
from sports_edge.data.nfl import NFLClient
from sports_edge.data.odds import OddsClient
from sports_edge.models.game_scope import GameEvent, build_game_events, game_scoped_markets
from sports_edge.models.intelligence import (
    PREMIUM_BOOKMAKER_KEYS,
    h2h_intelligence,
    prop_book_offer_edges,
)
# Streamlit Community Cloud can briefly hot-reload app.py while keeping an
# older dependency module in sys.modules. Reload this module explicitly so a
# partial deploy cannot crash startup on newly added exports.
_ks = importlib.import_module("sports_edge.models.kalshi_sports")
try:
    _ks = importlib.reload(_ks)
except Exception:
    # Keep the previous module available; safe fallbacks below prevent a
    # missing newly-added export from taking the whole app down.
    pass

SUPPORTED_SPORTS = getattr(_ks, "SUPPORTED_SPORTS", ("MLB", "NBA", "WNBA", "NFL", "Tennis"))
choose_kalshi_ticket = _ks.choose_kalshi_ticket
group_kalshi_sports = _ks.group_kalshi_sports
kalshi_prop_families = _ks.prop_families
kalshi_side_candidates = _ks.side_candidates

def _fallback_catalog_diagnostics(markets):
    grouped = group_kalshi_sports(markets)
    counts = {sport: len(grouped.get(sport, [])) for sport in SUPPORTED_SPORTS}
    return SimpleNamespace(
        total_open_markets=len(markets),
        classified_markets=sum(counts.values()),
        unclassified_supported_prefixes=(),
        counts_by_sport=counts,
    )

catalog_diagnostics = getattr(_ks, "catalog_diagnostics", _fallback_catalog_diagnostics)
from sports_edge.models.live_board import LiveSignal, build_live_signals, build_underdog_signals, market_yes_probability
from sports_edge.models.parlay import PRESETS, kalshi_copy_ticket
from sports_edge.models.parlay_intelligence import assess_leg, build_intelligent_parlay
from sports_edge.models.kalshi_model_candidates import model_candidates_from_kalshi, attach_sportsbook_context
from sports_edge.models.parlay_candidates import (
    ParlayCandidateLeg,
    candidate_legs_from_h2h,
    candidate_legs_from_props,
    combo_blueprint,
    generate_candidate_parlay,
)

try:
    from sports_edge.models.parlay_candidates import research_fallback_candidates as _research_fallback_candidates
except ImportError:
    def _research_fallback_candidates(
        candidates,
        *,
        min_books: int = 3,
        max_source_age_s: float = 120.0,
        min_consensus_probability: float = 0.52,
    ):
        rows = [
            row for row in candidates
            if int(getattr(row, "book_count", 0)) >= min_books
            and float(getattr(row, "source_age_s", 999999.0)) <= max_source_age_s
            and float(
                getattr(
                    row,
                    "consensus_probability",
                    getattr(row, "fair_probability", 0.0),
                )
            ) >= min_consensus_probability
        ]
        return sorted(
            rows,
            key=lambda r: (
                getattr(r, "evidence_class", "") == "EDGE-QUALIFIED",
                int(getattr(r, "book_count", 0)),
                -float(getattr(r, "source_age_s", 999999.0)),
                float(
                    getattr(
                        r,
                        "consensus_probability",
                        getattr(r, "fair_probability", 0.0),
                    )
                ),
            ),
            reverse=True,
        )
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
.market-card{padding:.85rem .9rem;border:1px solid #284a3e;border-radius:16px;background:linear-gradient(180deg,rgba(18,39,32,.94),rgba(8,21,17,.96));margin:.55rem 0;box-shadow:0 8px 28px rgba(0,0,0,.16)}
.market-card-top{display:flex;justify-content:space-between;gap:.6rem;align-items:flex-start}
.market-title{font-weight:760;font-size:1.03rem;line-height:1.25}.market-price{font-size:1.42rem;font-weight:850;white-space:nowrap;color:#e8fff5}
.market-meta{font-size:.82rem;color:#93aaa0;margin-top:.35rem;line-height:1.45}
.edge-pill{display:inline-block;padding:3px 8px;border-radius:999px;background:#153d2f;border:1px solid #2b6c54;color:#72f0b3;font-size:.72rem;font-weight:750;margin-right:5px}
.watch-pill{display:inline-block;padding:3px 8px;border-radius:999px;background:#3b3417;border:1px solid #75672a;color:#f0d777;font-size:.72rem;font-weight:750;margin-right:5px}
.score{font-weight:850;color:#72f0b3}
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
st.caption("Build: 2026-09-25-mlb-prop-models-2")


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


@st.cache_data(ttl=60, show_spinner=False)
def get_kalshi_markets(sport_filter_value: str):
    selected = None if sport_filter_value == "All" else (sport_filter_value,)
    result = fetch_supported_sport_catalog(
        page_limit_per_series=50,
        page_size=1000,
        request_pause_s=0.0,
        max_workers=6,
        request_interval_s=0.15,
        sports=selected,
        overview_only=(sport_filter_value == "All"),
    )
    return (
        list(result.markets),
        result.error,
        result.max_latency_ms,
        result.pages,
        result.cursor_exhausted,
        result.relevant_series,
        result.incomplete_series,
    )


@st.cache_data(ttl=45, show_spinner=False)
@st.cache_data(ttl=300, show_spinner=False)
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
        r = OddsClient(api_key=api_key).odds(
            sport_key=sport_key,
            markets="h2h",
            bookmakers=",".join(PREMIUM_BOOKMAKER_KEYS),
        )
        return (r.data if isinstance(r.data, list) else []), None, r.latency_ms
    except Exception as exc:
        return [], _safe_error(exc), None


@st.cache_data(ttl=45, show_spinner=False)
def get_event_odds(api_key: str, sport_key: str, event_id: str, markets: str):
    try:
        r = OddsClient(api_key=api_key).event_odds(
            sport_key=sport_key,
            event_id=event_id,
            markets=markets,
            bookmakers=",".join(PREMIUM_BOOKMAKER_KEYS),
        )
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


def _sport_from_odds_key(key: str) -> str | None:
    low = key.lower()
    if key.startswith("tennis_"):
        return "Tennis"
    if "wnba" in low:
        return "WNBA"
    if "nba" in low:
        return "NBA"
    if "nfl" in low:
        return "NFL"
    if "mlb" in low:
        return "MLB"
    return None


def sport_pairs(active_sports: list[dict]) -> list[tuple[str, str]]:
    wanted_keys = {
        "americanfootball_nfl": "NFL",
        "baseball_mlb": "MLB",
        "basketball_nba": "NBA",
        "basketball_wnba": "WNBA",
    }
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    for item in active_sports:
        key = str(item.get("key") or "")
        group = str(item.get("group") or "")
        title = str(item.get("title") or key)
        sport = _sport_from_odds_key(key)

        if key in wanted_keys:
            pairs.append((wanted_keys[key], key))
            seen.add(key)
        elif sport == "Tennis" or group.lower() == "tennis":
            pairs.append((f"Tennis · {title}", key))
            seen.add(key)

    # Keep core leagues available even if /sports ordering changes.
    for key, label in wanted_keys.items():
        if key not in seen:
            pairs.append((label, key))

    return pairs

def build_game_universe(api_key: str | None, active_sports: list[dict], sport_filter_value: str = "All"):
    if not api_key:
        return [], {}, ["THE_ODDS_API_KEY is not configured"]

    games: list[GameEvent] = []
    raw_by_id: dict[str, dict] = {}
    errors: list[str] = []
    for label, key in sport_pairs(active_sports):
        sport = _sport_from_odds_key(key)
        if sport is None:
            continue
        if sport_filter_value != "All" and sport != sport_filter_value:
            continue
        raw, err, _ = get_events(api_key, key)
        if err:
            errors.append(f"{label}: {err}")
            continue
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


def _kalshi_rows_for_sport(grouped, sport_filter_value: str):
    if sport_filter_value == "All":
        rows = []
        for sport_name in SUPPORTED_SPORTS:
            rows.extend(grouped.get(sport_name, []))
        return rows
    return list(grouped.get(sport_filter_value, []))


def kalshi_market_table(rows) -> pd.DataFrame:
    data = []
    for row in rows:
        market = row.market
        yes_p = market_side_probability(market, "YES")
        no_p = market_side_probability(market, "NO")
        try:
            volume = float(market.get("volume_fp", market.get("volume", 0)) or 0)
        except (TypeError, ValueError):
            volume = 0.0
        data.append(
            {
                "Sport": row.sport,
                "Family": row.family,
                "Event": market.get("event_title") or market.get("title") or market.get("event_ticker") or "",
                "YES": f"{yes_p:.0%}" if yes_p is not None else "—",
                "NO": f"{no_p:.0%}" if no_p is not None else "—",
                "Volume": int(volume),
                "Ticker": market.get("ticker") or "",
            }
        )
    return pd.DataFrame(data)


def kalshi_event_table(rows) -> pd.DataFrame:
    events = {}
    for row in rows:
        market = row.market
        key = str(market.get("event_ticker") or market.get("ticker") or "")
        if not key:
            continue
        rec = events.setdefault(
            key,
            {
                "Sport": row.sport,
                "Event": market.get("event_title") or market.get("title") or key,
                "Families": set(),
                "Markets": 0,
                "Volume": 0.0,
            },
        )
        rec["Families"].add(row.family)
        rec["Markets"] += 1
        try:
            rec["Volume"] += float(market.get("volume_fp", market.get("volume", 0)) or 0)
        except (TypeError, ValueError):
            pass
    return pd.DataFrame(
        [
            {
                "Sport": rec["Sport"],
                "Event": rec["Event"],
                "Markets": rec["Markets"],
                "Families": ", ".join(sorted(rec["Families"])),
                "Volume": int(rec["Volume"]),
            }
            for rec in sorted(events.values(), key=lambda x: x["Volume"], reverse=True)
        ]
    )


def kalshi_ticket_text(legs) -> str:
    lines = ["SPORTS EDGE — KALSHI TICKET", "Recheck the live price in Kalshi before entry.", ""]
    for idx, leg in enumerate(legs, 1):
        lines.append(
            f"{idx}. {leg.ticker} | {leg.side} | {leg.selection} | {round(leg.price * 100)}¢ | {leg.family}"
        )
    if not legs:
        lines.append("No current legs fit this ticket profile.")
    return "\n".join(lines)


def build_game_line_signals(markets: list[dict], api_key: str | None, sport_filter: str):
    if not api_key:
        return [], {}, ["THE_ODDS_API_KEY is not configured"]

    active, active_err = get_active_sports(api_key)
    errors = [active_err] if active_err else []
    all_signals: list[LiveSignal] = []
    intelligence: dict[tuple[str, str, str], object] = {}

    for label, key in sport_pairs(active):
        sport = _sport_from_odds_key(key)
        if sport is None:
            continue
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
            game_signals = build_live_signals(game_markets, [event], sport=sport)
            all_signals.extend(game_signals)
            for signal in game_signals:
                result = h2h_intelligence(
                    event,
                    signal.selection,
                    market_probability=signal.market_probability,
                )
                if result is not None:
                    intelligence[(signal.ticker, signal.side, signal.selection)] = result

    dedup: dict[tuple[str, str, str], LiveSignal] = {}
    for signal in all_signals:
        key = (signal.ticker, signal.side, signal.selection)
        old = dedup.get(key)
        if old is None or signal.confidence > old.confidence:
            dedup[key] = signal

    rows = list(dedup.values())
    rows.sort(
        key=lambda s: (
            getattr(intelligence.get((s.ticker, s.side, s.selection)), "intelligence_score", 0.0),
            s.status == "QUALIFIED",
            s.edge_points,
            s.confidence,
        ),
        reverse=True,
    )
    return rows, intelligence, [e for e in errors if e]


PARLAY_PROP_PLAN: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "MLB Hits": [("MLB", ("batter_hits",))],
    "MLB Home Runs": [("MLB", ("batter_home_runs",))],
    "MLB Strikeouts": [("MLB", ("pitcher_strikeouts",))],
    "NFL Passing": [("NFL", ("player_pass_yds", "player_pass_tds"))],
    "NFL Rushing": [("NFL", ("player_rush_yds",))],
    "NFL Receiving": [("NFL", ("player_receptions", "player_reception_yds"))],
    "NFL Touchdowns": [("NFL", ("player_anytime_td",))],
    "NBA Points": [("NBA", ("player_points",))],
    "NBA Rebounds": [("NBA", ("player_rebounds",))],
    "NBA Assists": [("NBA", ("player_assists",))],
    "NBA Threes": [("NBA", ("player_threes",))],
    "NBA PRA": [("NBA", ("player_points_rebounds_assists",))],
    "WNBA Points": [("WNBA", ("player_points",))],
    "WNBA Rebounds": [("WNBA", ("player_rebounds",))],
    "WNBA Assists": [("WNBA", ("player_assists",))],
    "WNBA Threes": [("WNBA", ("player_threes",))],
    "WNBA PRA": [("WNBA", ("player_points_rebounds_assists",))],
    "Best Available": [
        ("MLB", ("batter_hits",)),
        ("NFL", ("player_anytime_td",)),
        ("NBA", ("player_points",)),
        ("WNBA", ("player_points",)),
    ],
    "Mixed Sports": [
        ("MLB", ("batter_hits",)),
        ("NFL", ("player_anytime_td",)),
        ("NBA", ("player_points",)),
        ("WNBA", ("player_points",)),
    ],
}


def parlay_candidate_table(legs: list[ParlayCandidateLeg] | tuple[ParlayCandidateLeg, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Evidence": x.evidence_class,
                "Sport": x.sport,
                "Game": x.event_title,
                "Selection": x.selection,
                "Consensus": f"{x.consensus_probability:.1%}",
                "Books": x.book_count,
                "Age": f"{x.source_age_s:.0f}s",
                "Kalshi": f"{x.kalshi_price:.1%}" if x.kalshi_price is not None else "—",
                "Edge": f"{x.kalshi_edge_points:+.1f} pp" if x.kalshi_edge_points is not None else "—",
            }
            for x in legs
        ]
    )


def _generate_parlay_compat(
    candidates: list[ParlayCandidateLeg],
    *,
    target_legs: int,
    mode: str,
    max_per_event: int,
    require_edge: bool,
    diversify_sports: bool,
):
    """Call the generator safely across Streamlit hot-reload version skew.

    Streamlit can briefly retain an older imported module while app.py has
    already updated. We inspect the live callable and emulate newer options
    before calling older signatures so a deploy cannot crash the UI.
    """
    working = list(candidates)
    params = inspect.signature(generate_candidate_parlay).parameters

    if require_edge and "require_edge" not in params:
        working = [
            row for row in working
            if getattr(row, "evidence_class", "") == "EDGE-QUALIFIED"
            and getattr(row, "kalshi_edge_points", None) is not None
            and float(row.kalshi_edge_points) >= 3.0
        ]

    if diversify_sports and "diversify_sports" not in params:
        diversified: list[ParlayCandidateLeg] = []
        remainder: list[ParlayCandidateLeg] = []
        seen_sports: set[str] = set()
        for row in working:
            if row.sport not in seen_sports:
                diversified.append(row)
                seen_sports.add(row.sport)
            else:
                remainder.append(row)
        working = diversified + remainder

    kwargs = {
        "target_legs": target_legs,
        "mode": mode,
        "max_per_event": max_per_event,
    }
    if "require_edge" in params:
        kwargs["require_edge"] = require_edge
    if "diversify_sports" in params:
        kwargs["diversify_sports"] = diversify_sports

    return generate_candidate_parlay(working, **kwargs)


def parlay_presets_for_sport(sport_filter_value: str) -> list[str]:
    if sport_filter_value == "Tennis":
        return ["Tennis Moneyline", "Best Available"]
    if sport_filter_value == "MLB":
        return ["Best Available", "MLB Hits", "MLB Home Runs", "MLB Strikeouts"]
    if sport_filter_value == "NFL":
        return ["Best Available", "NFL Game Markets", "NFL Passing", "NFL Rushing", "NFL Receiving", "NFL Touchdowns"]
    if sport_filter_value == "NBA":
        return ["Best Available", "NBA Points", "NBA Rebounds", "NBA Assists", "NBA Threes", "NBA PRA"]
    if sport_filter_value == "WNBA":
        return ["Best Available", "WNBA Points", "WNBA Rebounds", "WNBA Assists", "WNBA Threes", "WNBA PRA"]
    return [
        "Best Available",
        "Mixed Sports",
        "Tennis Moneyline",
        "MLB Hits",
        "MLB Home Runs",
        "MLB Strikeouts",
        "NFL Game Markets",
        "NFL Passing",
        "NFL Rushing",
        "NFL Receiving",
        "NFL Touchdowns",
    ]


def _round_robin_games(games_in: list[GameEvent], sports: list[str], limit: int) -> list[GameEvent]:
    buckets = {sport: [g for g in games_in if g.sport == sport] for sport in sports}
    out: list[GameEvent] = []
    while len(out) < limit and any(buckets.values()):
        for sport in sports:
            bucket = buckets.get(sport, [])
            if bucket and len(out) < limit:
                out.append(bucket.pop(0))
    return out


def _linked_scan_games(
    games_in: list[GameEvent],
    scoped_markets: dict[str, list[dict]],
    sports: list[str],
    limit: int,
) -> list[GameEvent]:
    """Spend scan budget only on games with exact Kalshi event linkage."""
    linked = [
        game for game in games_in
        if game.sport in sports and scoped_markets.get(game.event_id)
    ]
    return _round_robin_games(linked, sports, limit)


def scan_parlay_candidates(
    *,
    preset: str,
    mode: str,
    sport_filter_value: str,
    games_in: list[GameEvent],
    scoped_markets: dict[str, list[dict]],
    api_key_value: str,
    max_games: int,
) -> tuple[list[ParlayCandidateLeg], list[str], int]:
    candidates: list[ParlayCandidateLeg] = []
    errors: list[str] = []
    calls = 0

    # Decide which sports' moneylines are allowed. This now obeys the user's
    # Sport filter and supports Tennis instead of silently falling back to MLB.
    if preset == "NFL Game Markets":
        h2h_sports = ["NFL"]
    elif preset == "Tennis Moneyline":
        h2h_sports = ["Tennis"]
    elif preset in ("Best Available", "Mixed Sports"):
        if sport_filter_value == "All":
            h2h_sports = ["MLB", "NBA", "WNBA", "NFL", "Tennis"]
        else:
            h2h_sports = [sport_filter_value]
    else:
        h2h_sports = []

    if h2h_sports:
        # Bound tennis/API cost by looking only at sport keys represented among
        # the selected games, then fetching h2h once per unique sport key.
        scan_games = _linked_scan_games(games_in, scoped_markets, h2h_sports, max_games)
        by_sport_key: dict[str, list[GameEvent]] = {}
        for game in scan_games:
            by_sport_key.setdefault(game.sport_key, []).append(game)

        for sport_key, key_games in by_sport_key.items():
            sport_name = key_games[0].sport
            events, err, _ = get_featured_odds(api_key_value, sport_key)
            calls += 1
            if err:
                errors.append(f"{sport_name}: {err}")
                continue
            event_map = {str(e.get("id")): e for e in events if e.get("id")}
            for game in key_games:
                event_payload = event_map.get(game.event_id)
                if not event_payload:
                    continue
                exact = build_live_signals(
                    scoped_markets.get(game.event_id, []),
                    [event_payload],
                    sport=sport_name,
                )
                candidates.extend(
                    candidate_legs_from_h2h(
                        game,
                        event_payload,
                        exact,
                        mode=mode,
                    )
                )

    plan = PARLAY_PROP_PLAN.get(preset, [])
    if preset in ("Best Available", "Mixed Sports") and sport_filter_value != "All":
        plan = [item for item in plan if item[0] == sport_filter_value]

    if plan:
        sports = list(dict.fromkeys(s for s, _ in plan))
        scan_games = _linked_scan_games(games_in, scoped_markets, sports, max_games)
        keys_by_sport = {sport: keys for sport, keys in plan}
        tasks: list[tuple[GameEvent, tuple[str, ...]]] = []
        for game in scan_games:
            keys = keys_by_sport.get(game.sport)
            if keys:
                tasks.append((game, keys))

        calls += len(tasks)
        if tasks:
            max_workers = min(4, len(tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_map = {
                    pool.submit(
                        get_event_odds,
                        api_key_value,
                        game.sport_key,
                        game.event_id,
                        ",".join(keys),
                    ): (game, keys)
                    for game, keys in tasks
                }
                for future in as_completed(future_map):
                    game, keys = future_map[future]
                    try:
                        payload, err, _ = future.result()
                    except Exception as exc:
                        payload, err = {}, _safe_error(exc)
                    if err:
                        errors.append(f"{game.away_team} @ {game.home_team}: {err}")
                        continue
                    quotes = prop_consensus(payload, market_keys=keys) if payload else []
                    exact = build_prop_signals(scoped_markets.get(game.event_id, []), game, quotes)
                    candidates.extend(candidate_legs_from_props(game, quotes, exact, mode=mode))

    dedup: dict[tuple[str, str], ParlayCandidateLeg] = {}
    for row in candidates:
        key = (row.event_id, row.selection.lower())
        old = dedup.get(key)
        if old is None or (
            row.evidence_class == "EDGE-QUALIFIED",
            row.kalshi_ticker is not None,
            row.kalshi_edge_points if row.kalshi_edge_points is not None else -999.0,
            row.book_count,
            row.consensus_probability,
        ) > (
            old.evidence_class == "EDGE-QUALIFIED",
            old.kalshi_ticker is not None,
            old.kalshi_edge_points if old.kalshi_edge_points is not None else -999.0,
            old.book_count,
            old.consensus_probability,
        ):
            dedup[key] = row

    return list(dedup.values()), errors, calls


if "view" not in st.session_state:
    st.session_state.view = "Edge Board"
if "prop_signal_cache" not in st.session_state:
    st.session_state.prop_signal_cache = {}

nav_map = {
    "For You": "Edge Board",
    "Games": "Games",
    "Markets": "Game Lines",
    "Props": "Player Props",
    "Parlays": "Parlay Generator",
    "Live": "Live Feed",
}
reverse_nav = {value: key for key, value in nav_map.items()}
current_nav = reverse_nav.get(st.session_state.view, "For You")
nav_labels = list(nav_map)

selected_nav = st.segmented_control(
    "Navigation",
    nav_labels,
    default=current_nav,
    key="main_nav_v2",
    label_visibility="collapsed",
)
view = nav_map.get(selected_nav or current_nav, "Edge Board")
st.session_state.view = view

sport_filter = st.segmented_control(
    "Sport",
    ["All", "MLB", "NBA", "WNBA", "NFL", "Tennis"],
    default="All",
    key="sport_filter_v2",
    label_visibility="collapsed",
) or "All"

if st.button("↻ Refresh live data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

api_key = _secret("THE_ODDS_API_KEY")
markets, kerr, klat, kpages, kcursor_exhausted, kseries, kincomplete = get_kalshi_markets(sport_filter)
kalshi_grouped = group_kalshi_sports(markets)
catalog_health = catalog_diagnostics(markets)
kalshi_rows = _kalshi_rows_for_sport(kalshi_grouped, sport_filter)

# Sportsbook schedules are no longer part of the default render path. They are
# loaded only for explicit intelligence scans/enrichment.
games: list[GameEvent] = []
raw_events: dict[str, dict] = {}
game_errors: list[str] = []
scoped: dict[str, list[dict]] = {}
visible_games: list[GameEvent] = []

if view == "Games":
    st.header("Kalshi Sports")
    st.markdown(
        '<div class="nav-hint"><b>Kalshi-first universe.</b> All loads a fast game overview; selecting MLB, NBA, WNBA, NFL or Tennis loads that sport’s full current game/prop catalog. '
        'Futures/championship markets are removed before they reach the app.</div>',
        unsafe_allow_html=True,
    )
    if not kalshi_rows:
        sport_count = catalog_health.counts_by_sport.get(sport_filter, 0) if sport_filter != "All" else catalog_health.classified_markets
        if markets and sport_count == 0:
            st.error(
                f"Kalshi returned {len(markets)} open markets, but 0 were classified for {sport_filter}. "
                "This is a catalog/classifier condition, not a legitimate empty slate."
            )
        elif not kcursor_exhausted:
            st.warning("Kalshi catalog ingestion is incomplete; the cursor did not exhaust.")
        else:
            st.info("No current Kalshi markets exist for this sport in the fully fetched open catalog.")
    else:
        event_df = kalshi_event_table(kalshi_rows)
        c1, c2 = st.columns(2)
        c1.metric("Kalshi events", len(event_df))
        c2.metric("Kalshi markets", len(kalshi_rows))
        st.dataframe(event_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Catalog: {len(markets)} open markets · {kseries} series requested · {kpages} market page(s) · "
        f"{'complete' if kcursor_exhausted else 'INCOMPLETE'} · "
        + " · ".join(f"{sport} {catalog_health.counts_by_sport.get(sport, 0)}" for sport in SUPPORTED_SPORTS)
    )
    if kincomplete:
        st.warning("Incomplete Kalshi series: " + ", ".join(kincomplete[:12]))
    if catalog_health.unclassified_supported_prefixes:
        st.warning(
            "Supported-sport series reached the catalog but were not classified: "
            + ", ".join(catalog_health.unclassified_supported_prefixes[:12])
        )
    if kerr:
        st.warning(f"Kalshi warning: {kerr}")

elif view == "Game Lines":
    st.header("Kalshi Markets")
    st.markdown(
        '<div class="section-note">Direct from current Kalshi sport markets. Sportsbook data is enrichment, not the source of this list.</div>',
        unsafe_allow_html=True,
    )
    # Markets is the complete current-event universe for the selected sport.
    # Props is a focused view, but no current Kalshi family is hidden here.
    line_rows = list(kalshi_rows)
    if not line_rows:
        st.info("No current Kalshi markets found for this sport.")
    else:
        family_options = ["All"] + sorted({row.family for row in line_rows})
        family = st.selectbox("Market type", family_options, key=f"kalshi_lines_{sport_filter}")
        show_rows = line_rows if family == "All" else [row for row in line_rows if row.family == family]
        st.dataframe(kalshi_market_table(show_rows[:250]), use_container_width=True, hide_index=True)

elif view == "Player Props":
    st.header("Kalshi Props")
    st.markdown(
        '<div class="section-note">Direct Kalshi prop markets first. MLB/NBA/WNBA/NFL player props and Tennis match/set/game/stat props appear here even when sportsbook enrichment is unavailable.</div>',
        unsafe_allow_html=True,
    )
    if sport_filter == "All":
        st.info("Select MLB, NBA, WNBA, NFL or Tennis above to load that sport’s full prop catalog.")
    sports_to_show = () if sport_filter == "All" else (sport_filter,)
    prop_rows = []
    for sport_name in sports_to_show:
        allowed = kalshi_prop_families(sport_name)
        prop_rows.extend([row for row in kalshi_grouped.get(sport_name, []) if row.family in allowed])

    if not prop_rows:
        st.info("No current Kalshi prop markets found for this sport.")
    else:
        family_options = ["All"] + sorted({row.family for row in prop_rows})
        family = st.selectbox("Prop type", family_options, key=f"kalshi_props_{sport_filter}")
        show_rows = prop_rows if family == "All" else [row for row in prop_rows if row.family == family]
        st.dataframe(kalshi_market_table(show_rows[:300]), use_container_width=True, hide_index=True)
        st.caption(
            "These are actual Kalshi contracts. Sports Edge will enrich exact markets with sportsbook/model evidence separately; "
            "missing external data no longer hides the Kalshi prop itself."
        )

        model_family_key = None
        if sport_filter == "MLB" and family == "Hits":
            model_family_key = "batter_hits"
        elif sport_filter == "MLB" and family == "Home Runs":
            model_family_key = "batter_home_runs"
        elif sport_filter == "MLB" and family == "Strikeouts":
            model_family_key = "pitcher_strikeouts"

        if model_family_key:
            st.success("Independent Sports Edge model available for this prop family.")
            if st.button(
                "Run Sports Edge prop analysis",
                type="primary",
                use_container_width=True,
                key=f"prop_model_scan_{sport_filter}_{family}",
            ):
                with st.spinner(f"Modeling current {sport_filter} {family} contracts…"):
                    prop_model_candidates = model_candidates_from_kalshi(
                        kalshi_grouped,
                        sport_filter=sport_filter,
                        include_mlb_hits=(model_family_key == "batter_hits"),
                        max_mlb_hit_players=24 if model_family_key == "batter_hits" else None,
                        include_mlb_home_runs=(model_family_key == "batter_home_runs"),
                        max_mlb_hr_players=24 if model_family_key == "batter_home_runs" else None,
                        include_mlb_strikeouts=(model_family_key == "pitcher_strikeouts"),
                        max_mlb_k_pitchers=16 if model_family_key == "pitcher_strikeouts" else None,
                    )
                    prop_model_candidates = [
                        row for row in prop_model_candidates
                        if row.market_key == model_family_key
                    ]
                    assessments = [assess_leg(row, "best") for row in prop_model_candidates]
                    assessments.sort(
                        key=lambda row: (
                            row.qualified,
                            row.score,
                            row.edge_points if row.edge_points is not None else -999.0,
                        ),
                        reverse=True,
                    )
                    st.session_state[f"prop_model_results_{sport_filter}_{family}"] = assessments

            assessments = st.session_state.get(
                f"prop_model_results_{sport_filter}_{family}",
                [],
            )
            if assessments:
                qualified_count = sum(row.qualified for row in assessments)
                c1, c2 = st.columns(2)
                c1.metric("Model candidates", len(assessments))
                c2.metric("Qualified edges", qualified_count)

                analysis_rows = pd.DataFrame(
                    [
                        {
                            "Status": "QUALIFIED" if row.qualified else "WATCH/PASS",
                            "Selection": row.leg.selection,
                            "Model fair": f"{row.fair_probability:.1%}",
                            "Kalshi": f"{row.kalshi_probability:.1%}" if row.kalshi_probability is not None else "—",
                            "Edge": f"{row.edge_points:+.1f} pp" if row.edge_points is not None else "—",
                            "Model confidence": f"{row.leg.model_confidence:.0%}",
                            "Score": f"{row.score:.0f}",
                        }
                        for row in assessments[:30]
                    ]
                )
                st.dataframe(analysis_rows, use_container_width=True, hide_index=True)

                for row in assessments[:12]:
                    with st.expander(f"{row.leg.selection} — Sports Edge analysis"):
                        st.write(f"**Model:** {row.leg.model_name or '—'}")
                        st.write(f"**Model confidence:** {row.leg.model_confidence:.0%}")
                        st.write(f"**Model-first fair:** {row.fair_probability:.1%}")
                        if row.kalshi_probability is not None:
                            st.write(f"**Kalshi:** {row.kalshi_probability:.1%}")
                        if row.edge_points is not None:
                            st.write(f"**Price edge:** {row.edge_points:+.1f} percentage points")
                        for reason in row.reasons:
                            st.caption("• " + reason)
                        if row.warnings:
                            st.warning(" · ".join(row.warnings))
        elif sport_filter != "All":
            st.info(
                "This prop family is currently catalog-visible but does not yet have a production-validated "
                "independent Sports Edge model. It will not be promoted as an intelligent pick from sportsbook consensus alone."
            )

elif view == "Edge Board":
    st.header("Edge Board")
    st.markdown(
        '<div class="section-note">Ranked by Sports Edge Intelligence: executable Kalshi gap + source quality + freshness + cross-book agreement. '
        'Only real current games are eligible; futures are excluded.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Run intelligence scan", type="primary", use_container_width=True):
        with st.spinner("Enriching current Kalshi markets with sportsbook/model evidence…"):
            signals, intelligence_map, signal_errors = build_game_line_signals(markets, api_key, sport_filter)
            st.session_state["edge_scan_v2"] = (signals, intelligence_map, signal_errors)
    signals, intelligence_map, signal_errors = st.session_state.get("edge_scan_v2", ([], {}, []))

    show = [s for s in signals if s.status in ("QUALIFIED", "WATCH")]
    if signal_errors and not signals:
        st.warning(" | ".join(signal_errors[:2]))

    if show:
        ranked = []
        for s in show:
            intel = intelligence_map.get((s.ticker, s.side, s.selection))
            score = getattr(intel, "intelligence_score", 0.0)
            ranked.append((score, s, intel))
        ranked.sort(key=lambda row: (row[0], row[1].edge_points), reverse=True)

        for score, s, intel in ranked[:14]:
            pill = "edge-pill" if s.status == "QUALIFIED" else "watch-pill"
            tier = getattr(intel, "tier", s.status)
            books = getattr(intel, "book_count", s.book_count)
            refs = getattr(intel, "reference_book_count", 0)
            age = getattr(intel, "median_age_s", s.source_age_s)
            disagreement = getattr(intel, "disagreement_pp", None)
            st.markdown(
                f"""
                <div class="market-card">
                  <div class="market-card-top">
                    <div>
                      <span class="{pill}">{s.status}</span>
                      <span class="edge-pill">TIER {tier}</span>
                      <div class="market-title">{s.selection}</div>
                    </div>
                    <div class="market-price">{s.market_probability:.0%}</div>
                  </div>
                  <div class="market-meta">
                    {s.event_title} · {s.sport}<br/>
                    <span class="score">Sports Edge {score:.0f}/100</span> · Fair {s.fair_probability:.1%} · Edge {s.edge_points:+.1f} pp<br/>
                    {books} fresh books · {refs} reference source(s) · age {age:.0f}s
                    {f' · disagreement {disagreement:.1f} pp' if disagreement is not None else ''}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Why this signal / copy ticket"):
                st.write(f"**Kalshi:** {s.side} {s.market_probability:.1%}")
                st.write(f"**Independent fair:** {s.fair_probability:.1%}")
                st.write(f"**Price edge:** {s.edge_points:+.1f} percentage points")
                if intel is not None:
                    for reason in intel.reasons:
                        st.caption("• " + reason)
                    if intel.warnings:
                        st.warning(" · ".join(intel.warnings))
                    source_rows = [
                        {
                            "Book": b.bookmaker_title,
                            "No-vig": f"{b.probability:.1%}",
                            "Age": f"{b.age_s:.0f}s",
                            "Reference": "YES" if b.is_reference else "",
                        }
                        for b in intel.books
                    ]
                    if source_rows:
                        st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)
                st.code(kalshi_copy_ticket([s]), language=None)
    else:
        st.info("No current game contract clears the live edge/watch gates. Sports Edge will not manufacture a pick.")


elif view == "Parlay Generator":
    st.header("Sports Edge Parlay Intelligence")
    st.markdown(
        '<div class="section-note"><b>Model-first.</b> Kalshi defines what is tradable, but Kalshi price and sportsbook consensus do not define the pick. '
        'Each qualifying leg needs a sport-specific model probability first. Sportsbooks are secondary calibration when an exact match exists.</div>',
        unsafe_allow_html=True,
    )

    builder_label = st.selectbox(
        "Builder",
        ["Best Available", "Priced Longshot (5+ legs)"],
        key="intel_builder_v3",
    )
    mode = "longshot" if builder_label.startswith("Priced Longshot") else "best"
    preset_options = parlay_presets_for_sport(sport_filter)
    preset = st.selectbox("Analysis type", preset_options, key=f"intel_preset_v3_{sport_filter}")

    min_legs = 5 if mode == "longshot" else 2
    default_legs = 6 if mode == "longshot" else 4
    max_legs = 10 if mode == "longshot" else 8
    target = st.slider(
        "Target legs",
        min_value=min_legs,
        max_value=max_legs,
        value=default_legs,
        key=f"intel_target_v3_{mode}",
    )

    if sport_filter == "Tennis":
        scan_min, scan_max, scan_default = 8, 40, 24
    else:
        scan_min, scan_max, scan_default = 4, 16, 8

    scan_games_n = st.slider(
        "Sportsbook cross-check games",
        min_value=scan_min,
        max_value=scan_max,
        value=scan_default,
        help="This only controls the optional secondary sportsbook cross-check. The first ticket is built from Kalshi + sport models without waiting on books.",
        key=f"intel_games_v3_{sport_filter}",
    )

    if mode == "longshot":
        st.info(
            "Priced Longshot: Kalshi 5–35¢, model-first edge at least +4pp, expected ROI on cost at least +15%, "
            "value multiple at least 1.15x, exact Kalshi contract, and adequate sport-model confidence. "
            "Sportsbook confirmation is optional—not required."
        )
    else:
        st.info(
            "Best Available: every leg must have positive independent-model value. Standard evidence still requires +3pp / 1.05x; "
            "deep, high-confidence model evidence may qualify smaller +1–2pp positive edges with stricter fair-probability floors. "
            "Exact Kalshi contract and adequate sport-model confidence remain mandatory. It is not a favorite detector."
        )

    if st.button("Analyze models & build ticket", type="primary", use_container_width=True):
        with st.spinner("Running sport models against current Kalshi markets…"):
            model_candidates = model_candidates_from_kalshi(
                kalshi_grouped,
                sport_filter=sport_filter,
                include_mlb_hits=(preset == "MLB Hits"),
                max_mlb_hit_players=(
                    max(12, min(24, target * 3))
                    if preset == "MLB Hits"
                    else None
                ),
                include_mlb_home_runs=(preset == "MLB Home Runs"),
                max_mlb_hr_players=(
                    max(12, min(24, target * 3))
                    if preset == "MLB Home Runs"
                    else None
                ),
                include_mlb_strikeouts=(preset == "MLB Strikeouts"),
                max_mlb_k_pitchers=(
                    max(8, min(16, target * 2))
                    if preset == "MLB Strikeouts"
                    else None
                ),
            )

            supported_model_presets = {
                "Best Available",
                "Mixed Sports",
                "Tennis Moneyline",
                "NFL Game Markets",
                "MLB Hits",
                "MLB Home Runs",
                "MLB Strikeouts",
            }
            if preset == "MLB Hits":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "MLB" and row.market_key == "batter_hits"
                ]
            elif preset == "MLB Home Runs":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "MLB" and row.market_key == "batter_home_runs"
                ]
            elif preset == "MLB Strikeouts":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "MLB" and row.market_key == "pitcher_strikeouts"
                ]
            elif preset == "Tennis Moneyline":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "Tennis" and row.market_key == "model_h2h"
                ]
            elif preset == "NFL Game Markets":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "NFL" and row.market_key == "model_h2h"
                ]
            elif preset not in supported_model_presets:
                model_candidates = []

            # Render from independent sport models first. Do not make the user
            # wait for optional sportsbook discovery/enrichment.
            result = build_intelligent_parlay(
                model_candidates,
                mode=mode,
                target_legs=target,
                max_per_event=1,
                diversify_sports=(sport_filter == "All"),
            )

            st.session_state["intel_parlay_v3"] = {
                "result": result,
                "preset": preset,
                "mode": mode,
                "sport": sport_filter,
                "target": target,
                "model_candidates": len(model_candidates),
                "model_covered": sum(1 for row in model_candidates if row.model_probability is not None),
                "book_confirmed": 0,
                "games_discovered": 0,
                "linked_games": 0,
                "calls": 0,
                "errors": [],
                "model_candidate_rows": model_candidates,
                "secondary_done": False,
            }

    state = st.session_state.get("intel_parlay_v3")
    state_matches = (
        state
        and state.get("mode") == mode
        and state.get("preset") == preset
        and state.get("sport") == sport_filter
        and state.get("target") == target
    )

    if state_matches and api_key and state.get("model_candidate_rows") and not state.get("secondary_done"):
        st.caption("Model-first ticket is ready. Sportsbook confirmation is optional and runs separately.")
        if st.button("Add optional sportsbook cross-check", use_container_width=True, key="secondary_crosscheck_v4"):
            with st.spinner("Cross-checking a bounded set of fresh sportsbook props…"):
                model_candidates = list(state.get("model_candidate_rows") or [])
                active, active_err = get_active_sports(api_key)
                lazy_games, _, universe_errors = build_game_universe(api_key, active, sport_filter)
                lazy_scoped = game_scoped_markets(markets, lazy_games)
                candidate_mode = "longshot" if mode == "longshot" else "high_confidence"
                book_candidates, scan_errors, calls = scan_parlay_candidates(
                    preset=preset,
                    mode=candidate_mode,
                    sport_filter_value=sport_filter,
                    games_in=lazy_games,
                    scoped_markets=lazy_scoped,
                    api_key_value=api_key,
                    max_games=min(scan_games_n, 6 if preset == "MLB Hits" else scan_games_n),
                )
                candidates = attach_sportsbook_context(model_candidates, book_candidates)
                result = build_intelligent_parlay(
                    candidates,
                    mode=mode,
                    target_legs=target,
                    max_per_event=1,
                    diversify_sports=(sport_filter == "All"),
                )
                st.session_state["intel_parlay_v3"] = {
                    "result": result,
                    "preset": preset,
                    "mode": mode,
                    "sport": sport_filter,
                    "target": target,
                    "model_candidates": len(model_candidates),
                    "model_covered": sum(1 for row in candidates if row.model_probability is not None),
                    "book_confirmed": sum(1 for row in candidates if row.book_count > 0),
                    "games_discovered": len(lazy_games),
                    "linked_games": sum(1 for game in lazy_games if lazy_scoped.get(game.event_id)),
                    "calls": calls,
                    "errors": [e for e in ([active_err] + universe_errors + scan_errors) if e],
                    "model_candidate_rows": model_candidates,
                    "secondary_done": True,
                }
                st.rerun()

    state = st.session_state.get("intel_parlay_v3")
    if state and state.get("mode") == mode and state.get("preset") == preset and state.get("sport") == sport_filter and state.get("target") == target:
        result = state["result"]
        if result.legs:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Qualified legs", len(result.legs))
            c2.metric("Model fair", f"{result.fair_joint_probability:.2%}")
            c3.metric("Kalshi price product", f"{result.market_joint_probability:.2%}")
            c4.metric("Ticket value", f"{result.ticket_value_multiple:.2f}x")

            rows = pd.DataFrame(
                [
                    {
                        "Score": f"{row.score:.0f}",
                        "Sport": row.leg.sport,
                        "Game": row.leg.event_title,
                        "Selection": row.leg.selection,
                        "Model": row.leg.model_name or "—",
                        "Model conf": f"{row.leg.model_confidence:.0%}",
                        "Fair": f"{row.fair_probability:.1%}",
                        "Kalshi": f"{row.kalshi_probability:.1%}",
                        "Edge": f"{row.edge_points:+.1f} pp",
                        "EV/contract": f"USD {row.ev_per_contract:+.2f}",
                        "ROI on cost": f"{row.expected_roi_on_cost:+.0%}",
                        "Value": f"{row.value_multiple:.2f}x",
                        "Books": row.leg.book_count,
                    }
                    for row in result.legs
                ]
            )
            st.dataframe(rows, use_container_width=True, hide_index=True)

            for idx, row in enumerate(result.legs, 1):
                with st.expander(f"{idx}. {row.leg.selection} — model analysis"):
                    st.write(f"**Event:** {row.leg.event_title}")
                    st.write(f"**Sport model:** {row.leg.model_name or '—'}")
                    st.write(f"**Model confidence:** {row.leg.model_confidence:.0%} · sample {row.leg.model_sample_size}")
                    st.write(f"**Kalshi:** {row.leg.kalshi_side} · {row.kalshi_probability:.1%} · {row.leg.kalshi_ticker}")
                    st.write(f"**Model-first fair:** {row.fair_probability:.1%}")
                    st.write(f"**Price edge:** {row.edge_points:+.1f} percentage points")
                    st.write(f"**Expected value:** USD {row.ev_per_contract:+.2f} per USD 1 payout contract")
                    st.write(f"**Expected ROI on cost:** {row.expected_roi_on_cost:+.0%}")
                    st.write(f"**Sports Edge leg score:** {row.score:.0f}/100")
                    for reason in row.reasons:
                        st.caption("• " + reason)
                    if row.warnings:
                        st.warning(" · ".join(row.warnings))

            if result.warnings:
                st.warning(" · ".join(result.warnings))
            st.caption(
                f"Kalshi/model candidates: {state.get('model_candidates', 0)} · "
                f"model-covered: {state.get('model_covered', 0)} · "
                f"sportsbook-confirmed: {state.get('book_confirmed', 0)} · "
                f"optional sportsbook calls: {state.get('calls', 0)} · "
                f"correlation risk: {result.correlation_risk}. "
                "Joint probabilities are independence benchmarks, not guarantees."
            )
            st.markdown("### Kalshi combo blueprint")
            st.code(combo_blueprint([row.leg for row in result.legs]), language=None)
        else:
            if preset not in {"Best Available", "Mixed Sports", "Tennis Moneyline", "NFL Game Markets", "MLB Hits", "MLB Home Runs", "MLB Strikeouts"}:
                st.warning(
                    "This player-prop family does not yet have a production sport-specific model. "
                    "Sports Edge is intentionally refusing book-only prop picks."
                )
            else:
                st.warning(
                    "No current leg cleared the model-first probability, confidence, and EV gates. "
                    "Sports Edge will not manufacture a ticket from Kalshi favorites or sportsbook consensus."
                )
            st.caption(
                f"Kalshi/model candidates: {state.get('model_candidates', 0)} · "
                f"model-covered: {state.get('model_covered', 0)} · "
                f"sportsbook-confirmed: {state.get('book_confirmed', 0)}."
            )
        if state.get("errors"):
            with st.expander("Secondary data diagnostics"):
                for error in state["errors"][:12]:
                    st.caption("• " + str(error))

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


with st.expander("System status / Model Trust"):
    st.write("**Futures:** excluded from the main workflow.")
    st.write("**Orders:** disabled; Sports Edge is read-only.")
    st.write("**Game identity:** both participants must match before cross-source pricing is used.")
    st.write("**Player props:** exact game + full player + prop family + compatible line/milestone required.")
    st.write("**Sportsbook intelligence:** source-weighted no-vig consensus plus leave-one-book-out offer checks.")
    st.write("**Catalog:** full open-market cursor exhaustion for MLB, NBA, WNBA, NFL and all Tennis families; unknown supported families stay visible instead of disappearing.")
    st.write("**Sport models:** model evidence is mandatory for parlay qualification. Tennis uses Elo/form/serve-return/workload; MLB Hits, Home Runs, and Pitcher Strikeouts use player/recent/opponent/probable-starter context; MLB/NFL/NBA/WNBA game winners use public team-strength baselines. Sportsbooks are secondary calibration only.")
    st.write("**Public bettors:** records must clear sample, verification, and CLV gates before they can count as supporting evidence.")
    st.warning("No pick or parlay is guaranteed. Missing, stale, conflicting, or unverified evidence fails closed.")

st.divider()
st.caption(
    f"Actionable Kalshi catalog · {len(markets)} open · {kseries} series · {kpages} pages · Kalshi request max {f'{klat:.0f} ms' if klat else '—'} · "
    f"Deployment {deployment_mode().replace('_', ' ').title()}"
)
