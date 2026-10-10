from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import importlib
import inspect
import json
import os
import requests
from statistics import median
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from sports_edge.core.startup import deployment_mode
from sports_edge.data.kalshi import KalshiPublicClient
from sports_edge.data.kalshi_catalog import fetch_supported_sport_catalog
from sports_edge.data.mlb import MLBClient
from sports_edge.data.mlb_prop_data import schedule_for_day
from sports_edge.data.nfl import NFLClient
from sports_edge.data.odds import OddsClient
from sports_edge.core.hot_reload import import_module_fresh

# Streamlit Community Cloud can hot-reload this entrypoint while dependency
# modules from the previous deploy remain in sys.modules. Tennis live helpers
# must therefore be resolved dynamically: a stale export should degrade the
# radar, never crash the entire app at import time.
_tennis_live = import_module_fresh("sports_edge.data.tennis_live")

def _empty_tennis_markets(*args, **kwargs):
    return ()

def _empty_tennis_candles(*args, **kwargs):
    return {}

def _empty_tennis_scores(*args, **kwargs):
    return ()

fetch_open_tennis_match_markets = getattr(
    _tennis_live,
    "fetch_open_tennis_match_markets",
    _empty_tennis_markets,
)
fetch_tennis_candle_history = getattr(
    _tennis_live,
    "fetch_tennis_candle_history",
    _empty_tennis_candles,
)
fetch_live_tennis_states = getattr(
    _tennis_live,
    "fetch_live_tennis_states",
    getattr(
        _tennis_live,
        "fetch_espn_live_tennis_states",
        _empty_tennis_scores,
    ),
)
from sports_edge.models.event_identity import (
    canonical_event_id_from_game,
    canonical_event_id_from_title,
    canonical_participant,
)
from sports_edge.models.game_scope import GameEvent, build_game_events, game_scoped_markets, market_matches_game
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

# Streamlit Community Cloud can hot-reload the app entrypoint before every
# dependency module is refreshed. Keep the parlay engine in lockstep with the
# UI so newly-added builder kwargs cannot crash production with a stale module.
_pi = importlib.import_module("sports_edge.models.parlay_intelligence")
try:
    _pi = importlib.reload(_pi)
except Exception:
    pass
assess_leg = _pi.assess_leg
build_intelligent_parlay = _pi.build_intelligent_parlay

from sports_edge.models.kalshi_model_candidates import model_candidates_from_kalshi, attach_sportsbook_context, _market_local_date

_tennis_early = import_module_fresh("sports_edge.models.tennis_early_research")
build_early_reversal_watches = getattr(
    _tennis_early, "build_early_reversal_watches", lambda *args, **kwargs: ()
)
build_extreme_cheap_observations = getattr(
    _tennis_early, "build_extreme_cheap_observations", lambda *args, **kwargs: ()
)
_tennis_ledger = import_module_fresh("sports_edge.models.tennis_reversal_ledger")

_tennis_reversal = import_module_fresh("sports_edge.models.tennis_live_reversal")

def _empty_tennis_reversal_radar(*args, **kwargs):
    return []

build_tennis_reversal_radar = getattr(
    _tennis_reversal,
    "build_tennis_reversal_radar",
    _empty_tennis_reversal_radar,
)
CHEAP_TENNIS_REVERSAL_MAX_PRICE = float(
    getattr(_tennis_reversal, "CHEAP_TENNIS_REVERSAL_MAX_PRICE", 0.20)
)

_tennis_lower_tour = import_module_fresh("sports_edge.models.tennis_lower_tour")

def _empty_lower_tour_candidates(*args, **kwargs):
    return []

build_lower_tour_live_fallback_candidates = getattr(
    _tennis_lower_tour,
    "build_lower_tour_live_fallback_candidates",
    _empty_lower_tour_candidates,
)

_tennis_coverage = import_module_fresh("sports_edge.models.tennis_live_coverage")

def _empty_tennis_coverage(*args, **kwargs):
    return SimpleNamespace(
        rows=(),
        open_matches=0,
        confirmed_live_matches=0,
        score_tracked_matches=0,
        model_covered_live_matches=0,
        unsupported_live_matches=0,
        start_passed_unverified_matches=0,
        source_counts=(),
    )

build_tennis_live_coverage = getattr(
    _tennis_coverage,
    "build_tennis_live_coverage",
    _empty_tennis_coverage,
)

_sofascore = import_module_fresh("sports_edge.data.sofascore")

def _empty_sofascore_snapshot(*args, **kwargs):
    return ()

fetch_sofascore_live_snapshot = getattr(
    _sofascore,
    "fetch_sofascore_live_snapshot",
    _empty_sofascore_snapshot,
)

from sports_edge.models.parlay_candidates import (
    ParlayCandidateLeg,
    candidate_legs_from_h2h,
    candidate_legs_from_props,
    candidate_book_context_for_model_lines,
    combo_blueprint,
    generate_candidate_parlay,
)
from sports_edge.models.ticket_build import build_ticket_for_scope, ticket_state_matches

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
from sports_edge.research.social_intelligence import research_source_rows

st.set_page_config(page_title="Sports Edge", page_icon="◈", layout="wide")

st.markdown(
    """
<style>
:root{--se-bg:#030706;--se-panel:#09120f;--se-panel2:#0d1c17;--se-line:#1c3b30;--se-text:#f4fbf8;--se-muted:#8da49a;--se-green:#5ff0ad;--se-green2:#25c986;--se-gold:#f0cf74;--se-danger:#ff7777;--se-radius:18px}
html,body,[class*="css"]{font-family:Inter,-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",sans-serif}
.block-container{padding-top:.5rem;max-width:1220px;padding-left:.8rem;padding-right:.8rem;padding-bottom:5.5rem}
.stApp{background:radial-gradient(circle at 18% -10%,#17382c 0,#08130f 30%,#030706 68%);color:var(--se-text)}
.stApp:before{content:"";position:fixed;inset:0;pointer-events:none;background:linear-gradient(115deg,rgba(95,240,173,.025),transparent 34%,rgba(240,207,116,.018));z-index:0}
header[data-testid="stHeader"]{background:rgba(3,7,6,.72);backdrop-filter:blur(18px);border-bottom:1px solid rgba(95,240,173,.08)}
[data-testid="stToolbar"]{background:transparent}
[data-testid="stAppViewContainer"]>.main{scroll-behavior:smooth}
.hero{font-size:2.15rem;font-weight:900;letter-spacing:-1.5px;line-height:1;margin:.15rem 0 .3rem}
.good{color:var(--se-green)} .section-note{color:var(--se-muted);font-size:.9rem;margin:-.2rem 0 .85rem;line-height:1.45}
.badge,.edge-pill,.watch-pill{display:inline-flex;align-items:center;padding:4px 9px;border-radius:999px;font-size:.7rem;font-weight:800;letter-spacing:.02em}
.badge,.edge-pill{border:1px solid #2e7057;background:#12372a;color:#78f2b7}.watch-pill{border:1px solid #716126;background:#383115;color:#f0d777}
.nav-hint,.surface{padding:.9rem 1rem;border:1px solid var(--se-line);border-radius:var(--se-radius);background:linear-gradient(145deg,rgba(15,34,28,.95),rgba(5,14,11,.96));margin:.45rem 0 .9rem;box-shadow:0 14px 42px rgba(0,0,0,.18)}
.section-hero{padding:1.05rem;border:1px solid #295241;border-radius:20px;background:linear-gradient(135deg,rgba(24,58,46,.96),rgba(7,18,14,.98));box-shadow:0 16px 45px rgba(0,0,0,.24);margin:.45rem 0 1rem}
.section-eyebrow{font-size:.68rem;letter-spacing:.13em;text-transform:uppercase;color:var(--se-green);font-weight:900}
.section-title{font-size:1.55rem;font-weight:900;letter-spacing:-.04em;margin:.15rem 0}.section-copy{font-size:.85rem;color:#9db2a8;max-width:760px;line-height:1.45}
[data-testid="stMetric"]{background:linear-gradient(180deg,rgba(19,42,34,.9),rgba(8,19,15,.95));border:1px solid var(--se-line);border-radius:16px;padding:11px;box-shadow:0 8px 24px rgba(0,0,0,.12)}
[data-testid="stMetricLabel"]{color:var(--se-muted)} [data-testid="stMetricValue"]{font-weight:850}
[data-testid="stSegmentedControl"]{background:rgba(6,15,12,.9);border:1px solid #1c382e;border-radius:16px;padding:4px;overflow-x:auto;scrollbar-width:none}
[data-testid="stSegmentedControl"]::-webkit-scrollbar{display:none}[data-testid="stSegmentedControl"] button{border-radius:12px!important;white-space:nowrap;font-weight:800;min-height:2.75rem;transition:transform .12s ease,background .12s ease}
[data-testid="stSegmentedControl"] button:hover{transform:translateY(-1px)}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:.35rem;background:rgba(5,14,11,.88);padding:.3rem;border:1px solid var(--se-line);border-radius:15px;overflow-x:auto}
[data-testid="stTabs"] [data-baseweb="tab"]{border-radius:11px;padding:.55rem .85rem;font-weight:800;white-space:nowrap}
[data-testid="stTabs"] [aria-selected="true"]{background:rgba(95,240,173,.12);color:var(--se-green)}
[data-testid="stSelectbox"]>div>div,[data-testid="stDateInput"]>div>div,[data-testid="stMultiSelect"]>div>div,[data-testid="stSlider"]{border-radius:13px!important}
[data-testid="stButton"] button{border-radius:14px;font-weight:850;min-height:2.85rem;border-color:#28503f;transition:transform .12s ease,box-shadow .12s ease}
[data-testid="stButton"] button:hover{transform:translateY(-1px);box-shadow:0 10px 28px rgba(0,0,0,.22)}
[data-testid="stAlert"]{border-radius:16px;border-width:1px}
[data-testid="stSelectbox"],[data-testid="stMultiSelect"],[data-testid="stDateInput"]{margin-bottom:.2rem}
[data-testid="stButton"] button[kind="primary"]{box-shadow:0 8px 24px rgba(44,184,123,.14)}
.ticket-shell{padding:1rem;border:1px solid #2b5b48;border-radius:20px;background:linear-gradient(145deg,rgba(20,51,40,.97),rgba(7,18,14,.98));box-shadow:0 14px 38px rgba(0,0,0,.22);margin:.5rem 0 .9rem}
.ticket-kicker{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:var(--se-green);font-weight:900}.ticket-title{font-size:1.4rem;font-weight:900;letter-spacing:-.03em;margin:.15rem 0}.ticket-sub{font-size:.84rem;color:#9cb2a8}
.game-card,.market-card{padding:.9rem;border:1px solid #25463a;border-radius:17px;background:linear-gradient(180deg,rgba(16,35,29,.94),rgba(7,18,14,.97));margin:.55rem 0;box-shadow:0 8px 28px rgba(0,0,0,.16)}
.game-title,.market-title{font-weight:800;font-size:1.04rem;line-height:1.25}.market-card-top{display:flex;justify-content:space-between;gap:.7rem;align-items:flex-start}.market-price{font-size:1.45rem;font-weight:900;white-space:nowrap}.market-meta{font-size:.82rem;color:#93aaa0;margin-top:.4rem;line-height:1.5}.live{color:var(--se-green);font-weight:900}.soon{color:var(--se-gold);font-weight:900}.score{font-weight:900;color:var(--se-green)}
[data-testid="stDataFrame"]{border:1px solid var(--se-line);border-radius:16px;overflow:hidden;background:rgba(7,16,13,.8)}
div[data-testid="stExpander"]{border:1px solid var(--se-line);border-radius:14px;background:rgba(7,17,14,.7)} hr{border-color:#173127!important}
@media(max-width:700px){.block-container{padding:.25rem .6rem 7rem}.hero{font-size:1.8rem}.section-hero{padding:.85rem;border-radius:17px}.section-title{font-size:1.3rem}[data-testid="stMetric"]{padding:8px}.ticket-shell{padding:.85rem;border-radius:17px}.market-card{padding:.8rem}.market-price{font-size:1.25rem}button[kind="secondary"],button[kind="primary"]{min-height:3.15rem}.section-copy,.ticket-sub,.market-meta{font-size:.88rem}.game-card,.market-card{border-radius:18px}.stCaption{line-height:1.4}}
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
st.caption("Build: 2026-09-29-adaptive-tennis-radar")


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


@st.cache_data(ttl=60, show_spinner=False)
def get_parlay_kalshi_markets(sport_filter_value: str):
    """Full current Kalshi universe for parlay game/date selection.

    The main All-sports screen intentionally uses an overview catalog for speed;
    parlay scoping cannot, because users must be able to select every eligible
    game on the requested date.
    """
    selected = None if sport_filter_value == "All" else (sport_filter_value,)
    result = fetch_supported_sport_catalog(
        page_limit_per_series=50,
        page_size=1000,
        request_pause_s=0.0,
        max_workers=6,
        request_interval_s=0.15,
        sports=selected,
        overview_only=False,
    )
    return (
        list(result.markets),
        result.error,
        result.cursor_exhausted,
        result.incomplete_series,
    )


@st.cache_data(ttl=60, show_spinner=False)
def get_parlay_kalshi_events():
    """Best-effort open-event metadata for ticket labels; never a fatal dependency."""
    client = KalshiPublicClient()
    events: dict[str, dict] = {}
    cursor = None
    error = None
    for _ in range(50):
        try:
            payload = client.events(status="open", limit=200, cursor=cursor, with_nested_markets=True)
        except Exception as exc:
            error = _safe_error(exc)
            break
        for event in payload.get("events", []) if isinstance(payload, dict) else []:
            key = str(event.get("event_ticker") or event.get("ticker") or "").strip()
            if key:
                events[key] = event
        cursor = str(payload.get("cursor") or "").strip() if isinstance(payload, dict) else ""
        if not cursor:
            break
    return events, error


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


@st.cache_data(ttl=15, show_spinner=False)
def get_scores(api_key: str, sport_key: str):
    try:
        r = OddsClient(api_key=api_key).scores(sport_key, days_from=1)
        return (r.data if isinstance(r.data, list) else []), None
    except Exception as exc:
        return [], _safe_error(exc)


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


@st.cache_data(ttl=20, show_spinner=False)
def get_featured_line_odds(api_key: str, sport_key: str):
    try:
        r = OddsClient(api_key=api_key).odds(
            sport_key=sport_key,
            markets="spreads,totals",
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


@st.cache_data(ttl=15, show_spinner=False)
def get_tennis_match_markets_live():
    try:
        return list(fetch_open_tennis_match_markets()), None
    except Exception as exc:
        return [], _safe_error(exc)


@st.cache_data(ttl=15, show_spinner=False)
def get_tennis_candles_live(tickers: tuple[str, ...]):
    try:
        return fetch_tennis_candle_history(
            tickers,
            lookback_minutes=60,
            period_interval=1,
        ), None
    except Exception as exc:
        return {}, _safe_error(exc)



@st.cache_data(ttl=300, show_spinner=False)
def get_sofascore_live_snapshot_cached():
    try:
        return tuple(fetch_sofascore_live_snapshot()), None
    except Exception as exc:
        return (), _safe_error(exc)


@st.cache_data(ttl=60, show_spinner=False)
def get_tennis_recent_prospective_signals():
    url = (
        "https://raw.githubusercontent.com/jwalton180-jpg/sports-edge-bot/"
        "tennis-prospective-data/research/tennis_prospective/signals.jsonl"
    )
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        rows = [
            json.loads(line)
            for line in response.text.splitlines()
            if line.strip()
        ]
        rows = [row for row in rows if isinstance(row, dict)]
        rows.sort(key=lambda row: row.get("first_observed_at", ""), reverse=True)
        return tuple(rows[:100]), None
    except (requests.RequestException, ValueError, TypeError) as exc:
        return (), _safe_error(exc)


@st.cache_data(ttl=300, show_spinner=False)
def get_tennis_cloud_prospective_report():
    url = (
        "https://raw.githubusercontent.com/jwalton180-jpg/sports-edge-bot/"
        "tennis-prospective-data/research/tennis_prospective/report.json"
    )
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            return None, "Unrecognized research report schema."
        return data, None
    except (requests.RequestException, ValueError, TypeError) as exc:
        return None, _safe_error(exc)


@st.cache_data(ttl=12, show_spinner=False)
def get_tennis_score_states_live():
    try:
        return list(fetch_live_tennis_states()), None
    except Exception as exc:
        return [], _safe_error(exc)


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
    "MLB Total Bases": [("MLB", ("batter_total_bases",))],
    "MLB RBIs": [("MLB", ("batter_rbis",))],
    "MLB H+R+RBI": [("MLB", ("batter_hits_runs_rbis",))],
    "MLB Strikeouts": [("MLB", ("pitcher_strikeouts",))],
    "NFL Passing": [("NFL", (
        "player_pass_yds", "player_pass_yds_alternate",
        "player_pass_tds", "player_pass_tds_alternate",
        "player_pass_attempts", "player_pass_attempts_alternate",
        "player_pass_completions", "player_pass_completions_alternate",
        "player_pass_interceptions", "player_pass_interceptions_alternate",
    ))],
    "NFL Rushing": [("NFL", (
        "player_rush_yds", "player_rush_yds_alternate",
        "player_rush_attempts", "player_rush_attempts_alternate",
        "player_rush_reception_yds", "player_rush_reception_yds_alternate",
    ))],
    "NFL Receiving": [("NFL", (
        "player_receptions", "player_receptions_alternate",
        "player_reception_yds", "player_reception_yds_alternate",
    ))],
    "NFL Touchdowns": [("NFL", ("player_anytime_td",))],
    "NBA Points": [("NBA", ("player_points",))],
    "NBA Rebounds": [("NBA", ("player_rebounds",))],
    "NBA Assists": [("NBA", ("player_assists",))],
    "NBA Threes": [("NBA", ("player_threes",))],
    "NBA PRA": [("NBA", ("player_points_rebounds_assists",))],
    "WNBA Points": [("WNBA", ("player_points", "player_points_alternate"))],
    "WNBA Rebounds": [("WNBA", ("player_rebounds", "player_rebounds_alternate"))],
    "WNBA Assists": [("WNBA", ("player_assists", "player_assists_alternate"))],
    "WNBA Threes": [("WNBA", ("player_threes", "player_threes_alternate"))],
    "WNBA PRA": [("WNBA", ("player_points_rebounds_assists", "player_points_rebounds_assists_alternate"))],
    "Best Available": [
        ("MLB", ("batter_hits", "batter_home_runs", "batter_total_bases", "batter_rbis", "batter_hits_runs_rbis", "pitcher_strikeouts")),
        ("NFL", (
            "player_pass_yds", "player_pass_yds_alternate",
            "player_pass_tds", "player_pass_tds_alternate",
            "player_pass_attempts", "player_pass_attempts_alternate",
            "player_pass_completions", "player_pass_completions_alternate",
            "player_pass_interceptions", "player_pass_interceptions_alternate",
            "player_rush_yds", "player_rush_yds_alternate",
            "player_rush_attempts", "player_rush_attempts_alternate",
            "player_rush_reception_yds", "player_rush_reception_yds_alternate",
            "player_receptions", "player_receptions_alternate",
            "player_reception_yds", "player_reception_yds_alternate",
            "player_anytime_td",
        )),
        ("NBA", ("player_points",)),
        ("WNBA", (
            "player_points", "player_points_alternate",
            "player_rebounds", "player_rebounds_alternate",
            "player_assists", "player_assists_alternate",
            "player_threes", "player_threes_alternate",
            "player_points_rebounds_assists", "player_points_rebounds_assists_alternate",
        )),
    ],
    "Mixed Sports": [
        ("MLB", ("batter_hits", "batter_home_runs", "batter_total_bases", "batter_rbis", "batter_hits_runs_rbis", "pitcher_strikeouts")),
        ("NFL", (
            "player_pass_yds", "player_pass_yds_alternate",
            "player_pass_tds", "player_pass_tds_alternate",
            "player_pass_attempts", "player_pass_attempts_alternate",
            "player_pass_completions", "player_pass_completions_alternate",
            "player_pass_interceptions", "player_pass_interceptions_alternate",
            "player_rush_yds", "player_rush_yds_alternate",
            "player_rush_attempts", "player_rush_attempts_alternate",
            "player_rush_reception_yds", "player_rush_reception_yds_alternate",
            "player_receptions", "player_receptions_alternate",
            "player_reception_yds", "player_reception_yds_alternate",
            "player_anytime_td",
        )),
        ("NBA", ("player_points",)),
        ("WNBA", (
            "player_points", "player_points_alternate",
            "player_rebounds", "player_rebounds_alternate",
            "player_assists", "player_assists_alternate",
            "player_threes", "player_threes_alternate",
            "player_points_rebounds_assists", "player_points_rebounds_assists_alternate",
        )),
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
        return ["Tennis Moneyline", "Tennis Games Total", "Best Available"]
    if sport_filter_value == "MLB":
        return ["Best Available", "MLB Game Markets", "MLB Hits", "MLB Home Runs", "MLB Total Bases", "MLB RBIs", "MLB H+R+RBI", "MLB Strikeouts"]
    if sport_filter_value == "NFL":
        return ["Best Available", "NFL Game Markets", "NFL Passing", "NFL Rushing", "NFL Receiving", "NFL Touchdowns"]
    if sport_filter_value == "NBA":
        return ["Best Available", "NBA Points", "NBA Rebounds", "NBA Assists", "NBA Threes", "NBA PRA"]
    if sport_filter_value == "WNBA":
        return ["Best Available", "WNBA Game Markets", "WNBA Points", "WNBA Rebounds", "WNBA Assists", "WNBA Threes", "WNBA PRA"]
    return [
        "Best Available",
        "Mixed Sports",
        "Tennis Moneyline",
        "Tennis Games Total",
        "MLB Hits",
        "MLB Home Runs",
        "MLB Total Bases",
        "MLB RBIs",
        "MLB H+R+RBI",
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
    model_targets: list[ParlayCandidateLeg] | None = None,
) -> tuple[list[ParlayCandidateLeg], list[str], int]:
    candidates: list[ParlayCandidateLeg] = []
    errors: list[str] = []
    calls = 0

    # Decide which sports' moneylines are allowed. This now obeys the user's
    # Sport filter and supports Tennis instead of silently falling back to MLB.
    if preset == "MLB Game Markets":
        h2h_sports = ["MLB"]
    elif preset == "NFL Game Markets":
        h2h_sports = ["NFL"]
    elif preset == "WNBA Game Markets":
        h2h_sports = ["WNBA"]
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

    line_targets = [
        row for row in (model_targets or [])
        if row.market_key in {
            "mlb_spread", "mlb_game_total", "mlb_team_total",
            "nfl_spread", "nfl_game_total", "nfl_team_total",
            "wnba_spread", "wnba_game_total", "wnba_team_total",
        }
    ]
    if line_targets:
        line_sports = list(dict.fromkeys(row.sport for row in line_targets))
        scan_games = _linked_scan_games(
            games_in,
            scoped_markets,
            line_sports,
            max_games,
        )
        by_sport_key: dict[str, list[GameEvent]] = {}
        for game in scan_games:
            by_sport_key.setdefault(game.sport_key, []).append(game)

        for sport_key, key_games in by_sport_key.items():
            sport_name = key_games[0].sport
            events, err, _ = get_featured_line_odds(api_key_value, sport_key)
            calls += 1
            if err:
                errors.append(f"{sport_name} game lines: {err}")
                continue
            event_map = {str(e.get("id")): e for e in events if e.get("id")}
            for game in key_games:
                payload = event_map.get(game.event_id)
                if not payload:
                    continue
                canonical_id = canonical_event_id_from_game(game)
                targets = [
                    row for row in line_targets
                    if row.event_id == canonical_id and row.sport == game.sport
                ]
                if not targets:
                    continue
                featured_targets = [
                    row for row in targets
                    if row.market_key not in {"mlb_team_total", "nfl_team_total", "wnba_team_total"}
                ]
                if featured_targets:
                    candidates.extend(
                        candidate_book_context_for_model_lines(
                            game,
                            payload,
                            featured_targets,
                        )
                    )

                team_total_targets = [
                    row for row in targets
                    if row.market_key in {"mlb_team_total", "nfl_team_total", "wnba_team_total"}
                ]
                if team_total_targets:
                    team_payload, team_err, _ = get_event_odds(
                        api_key_value,
                        game.sport_key,
                        game.event_id,
                        "team_totals,alternate_team_totals",
                    )
                    calls += 1
                    if team_err:
                        errors.append(
                            f"{game.away_team} @ {game.home_team} team totals: {team_err}"
                        )
                    elif team_payload:
                        candidates.extend(
                            candidate_book_context_for_model_lines(
                                game,
                                team_payload,
                                team_total_targets,
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


def _section_hero(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="section-hero"><div class="section-eyebrow">{kicker}</div>'
        f'<div class="section-title">{title}</div><div class="section-copy">{copy}</div></div>',
        unsafe_allow_html=True,
    )


def _premium_empty(title: str, copy: str, status: str = "WAITING") -> None:
    st.markdown(
        f'<div class="surface"><div class="section-eyebrow">{status}</div>'
        f'<div class="game-title">{title}</div><div class="section-copy">{copy}</div></div>',
        unsafe_allow_html=True,
    )


def _premium_stat_strip(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        f'<div class="intel-stat"><div class="intel-stat-value">{value}</div>'
        f'<div class="intel-stat-label">{label}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="intel-strip">{cards}</div>', unsafe_allow_html=True)


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

st.markdown('<div class="section-eyebrow">EXPLORE</div>', unsafe_allow_html=True)
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

def _mlb_schedule_game_choices(target_date: date) -> list[tuple[str, str, str]]:
    """Physical MLB games from the official MLB schedule, independent of child market titles."""
    choices: list[tuple[str, str, str]] = []
    for game in schedule_for_day(target_date.isoformat()):
        teams = game.get("teams") or {}
        away = str((((teams.get("away") or {}).get("team") or {}).get("name")) or "").strip()
        home = str((((teams.get("home") or {}).get("team") or {}).get("name")) or "").strip()
        game_pk = str(game.get("gamePk") or "").strip()
        if not away or not home or not game_pk:
            continue
        title = f"{away} at {home}"
        choices.append((f"MLB:{game_pk}", f"MLB · {title}", title))
    return sorted(choices, key=lambda item: item[1].lower())


def _event_matchup_title(event: dict) -> str:
    """Prefer the physical event title; never promote a child market title as a game."""
    for key in ("title", "event_title", "subtitle"):
        value = str(event.get(key) or "").strip()
        if value:
            return value
    return ""


def _parlay_game_choices(grouped, sport_filter_value: str, target_date: date, ticket_timezone: str, events_by_ticker: dict[str, dict]) -> list[tuple[str, str, str]]:
    """Return stable physical-game choices, resolved through Kalshi event metadata."""
    choices: dict[str, tuple[str, str, str]] = {}
    sports = SUPPORTED_SPORTS if sport_filter_value == "All" else (sport_filter_value,)
    for sport in sports:
        for row in grouped.get(sport, []):
            market = getattr(row, "market", {}) or {}
            try:
                if _market_local_date(market, ticket_timezone) != target_date:
                    continue
            except (TypeError, ValueError, KeyError):
                continue
            event_key = str(market.get("event_ticker") or "").strip()
            if not event_key:
                continue
            event = events_by_ticker.get(event_key) or {}
            title = _event_matchup_title(event)
            if not title:
                continue
            label = f"{sport} · {title}"
            choices.setdefault(event_key, (event_key, label, title))
    if sport_filter_value in {"All", "MLB"}:
        try:
            for game_key, label, title in _mlb_schedule_game_choices(target_date):
                choices.setdefault(game_key, (game_key, label, title))
        except Exception:
            # Kalshi-resolved games for other sports remain available; MLB
            # schedule failure must not invent child-market game identities.
            pass
    return sorted(choices.values(), key=lambda item: (item[1].lower(), item[0]))


if view == "Games":
    _section_hero("GAME HUB", "Today’s sports universe", "Browse the current Kalshi game slate first. Choose a sport for its complete market catalog; futures stay out of the way.")
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
        _premium_stat_strip([("Game events", str(len(event_df))), ("Open markets", str(len(kalshi_rows)))])
        for _, row in event_df.head(18).iterrows():
            title = str(row.get("Event") or row.get("event") or row.get("Title") or row.get("title") or "Current game")
            st.markdown(f'<div class="game-card"><div class="game-title">{title}</div></div>', unsafe_allow_html=True)
        with st.expander("Full game slate"):
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
    _section_hero("MARKET DESK", "Every current contract", "Scan the Kalshi market universe by family, then let Sports Edge intelligence decide whether price and evidence create an actionable edge.")
    st.markdown(
        '<div class="section-note">Direct from current Kalshi sport markets. Sportsbook data is enrichment, not the source of this list.</div>',
        unsafe_allow_html=True,
    )
    # Markets is the complete current-event universe for the selected sport.
    # Props is a focused view, but no current Kalshi family is hidden here.
    line_rows = list(kalshi_rows)
    if not line_rows:
        _premium_empty("No current markets", "No eligible current Kalshi contracts are available for this sport. Sports Edge will not fill the screen with futures or stale markets.")
    else:
        family_options = ["All"] + sorted({row.family for row in line_rows})
        family = st.selectbox("Market type", family_options, key=f"kalshi_lines_{sport_filter}")
        show_rows = line_rows if family == "All" else [row for row in line_rows if row.family == family]
        st.dataframe(kalshi_market_table(show_rows[:250]), use_container_width=True, hide_index=True)

elif view == "Player Props":
    _section_hero("PLAYER LAB", "Props with model accountability", "See the real Kalshi prop board, then run independent Sports Edge models only where production-grade player evidence exists.")
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
        _premium_empty("No current player props", "No eligible current Kalshi player-prop contracts are available for this sport. Unsupported markets remain hidden from model recommendations.")
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
        elif sport_filter == "MLB" and family == "Total Bases":
            model_family_key = "batter_total_bases"
        elif sport_filter == "MLB" and family == "RBIs":
            model_family_key = "batter_rbis"
        elif sport_filter == "MLB" and family == "Hits + Runs + RBIs":
            model_family_key = "batter_hrr"
        elif sport_filter == "MLB" and family == "Strikeouts":
            model_family_key = "pitcher_strikeouts"
        elif sport_filter == "NFL" and family == "Passing Yards":
            model_family_key = "player_pass_yds"
        elif sport_filter == "NFL" and family == "Passing TDs":
            model_family_key = "player_pass_tds"
        elif sport_filter == "NFL" and family == "Pass Attempts":
            model_family_key = "player_pass_attempts"
        elif sport_filter == "NFL" and family == "Pass Completions":
            model_family_key = "player_pass_completions"
        elif sport_filter == "NFL" and family == "Pass Interceptions":
            model_family_key = "player_pass_interceptions"
        elif sport_filter == "NFL" and family == "Rushing Yards":
            model_family_key = "player_rush_yds"
        elif sport_filter == "NFL" and family == "Rush Attempts":
            model_family_key = "player_rush_attempts"
        elif sport_filter == "NFL" and family == "Rushing + Receiving Yards":
            model_family_key = "player_rush_reception_yds"
        elif sport_filter == "NFL" and family == "Receiving Yards":
            model_family_key = "player_reception_yds"
        elif sport_filter == "NFL" and family == "Receptions":
            model_family_key = "player_receptions"
        elif sport_filter == "NFL" and family == "Player Touchdowns":
            model_family_key = "player_anytime_td"
        elif sport_filter == "WNBA" and family == "Points":
            model_family_key = "player_points"
        elif sport_filter == "WNBA" and family == "Rebounds":
            model_family_key = "player_rebounds"
        elif sport_filter == "WNBA" and family == "Assists":
            model_family_key = "player_assists"
        elif sport_filter == "WNBA" and family == "Three-Pointers":
            model_family_key = "player_threes"
        elif sport_filter == "WNBA" and family == "Points + Rebounds + Assists":
            model_family_key = "player_points_rebounds_assists"
        elif sport_filter == "Tennis" and family == "Games Total":
            model_family_key = "tennis_games_total"

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
                        include_mlb_total_bases=(model_family_key == "batter_total_bases"),
                        max_mlb_tb_players=24 if model_family_key == "batter_total_bases" else None,
                        include_mlb_rbis=(model_family_key == "batter_rbis"),
                        max_mlb_rbi_players=24 if model_family_key == "batter_rbis" else None,
                        include_mlb_hrr=(model_family_key == "batter_hrr"),
                        max_mlb_hrr_players=24 if model_family_key == "batter_hrr" else None,
                        include_mlb_strikeouts=(model_family_key == "pitcher_strikeouts"),
                        max_mlb_k_pitchers=16 if model_family_key == "pitcher_strikeouts" else None,
                        include_nfl_passing_yards=(model_family_key == "player_pass_yds"),
                        max_nfl_passing_players=18 if model_family_key in {"player_pass_yds", "player_pass_tds", "player_pass_attempts", "player_pass_completions", "player_pass_interceptions"} else None,
                        include_nfl_passing_tds=(model_family_key == "player_pass_tds"),
                        include_nfl_pass_attempts=(model_family_key == "player_pass_attempts"),
                        include_nfl_pass_completions=(model_family_key == "player_pass_completions"),
                        include_nfl_pass_interceptions=(model_family_key == "player_pass_interceptions"),
                        include_nfl_rushing_yards=(model_family_key == "player_rush_yds"),
                        include_nfl_rush_attempts=(model_family_key == "player_rush_attempts"),
                        include_nfl_rush_receiving_yards=(model_family_key == "player_rush_reception_yds"),
                        max_nfl_rushing_players=24 if model_family_key in {"player_rush_yds", "player_rush_attempts", "player_rush_reception_yds"} else None,
                        include_nfl_receiving_yards=(model_family_key == "player_reception_yds"),
                        include_nfl_receptions=(model_family_key == "player_receptions"),
                        max_nfl_receiving_players=24 if model_family_key in {"player_reception_yds", "player_receptions"} else None,
                        include_nfl_touchdowns=(model_family_key == "player_anytime_td"),
                        max_nfl_td_players=24 if model_family_key == "player_anytime_td" else None,
                        include_wnba_player_props=(sport_filter == "WNBA"),
                        max_wnba_players=24 if sport_filter == "WNBA" else None,
                        include_tennis_match_winner=(model_family_key != "tennis_games_total"),
                        include_tennis_games_total=(model_family_key == "tennis_games_total"),
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
    _section_hero("FOR YOU", "Sports Edge Intelligence", "The strongest current model-versus-market disagreements, ranked by evidence quality, freshness, and executable Kalshi price.")
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
    _section_hero("PARLAY LAB", "Build an intelligent ticket", "Choose the slate you want. Sports Edge models eligible contracts first, then applies price, EV, confidence, and correlation controls.")
    st.markdown(
        '<div class="section-note"><b>Model-first.</b> Kalshi defines what is tradable, but Kalshi price and sportsbook consensus do not define the pick. '
        'Each qualifying leg needs a sport-specific model probability first. Sportsbooks and public sharp/social research are secondary confirmation only.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Public sharp / social research layer"):
        st.caption(
            "SportEdge studies public Kalshi profiles, tracked betting communities and public X/process sources for repeatable methodology. "
            "It does not copy viral parlays or alter fair probability because a popular account posted a pick. The layer only ranks already-qualified legs by process fit."
        )
        st.dataframe(
            pd.DataFrame(research_source_rows()),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        '<div class="ticket-shell"><div class="ticket-kicker">Ticket Lab</div>'
        '<div class="ticket-title">Build from the games you want</div>'
        '<div class="ticket-sub">Sports Edge models the eligible Kalshi contracts first, then applies price, EV, confidence, and correlation gates.</div></div>',
        unsafe_allow_html=True,
    )
    builder_label = st.segmented_control(
        "Build style",
        ["Best Available", "Priced Longshot (5+ legs)"],
        default="Best Available",
        key="intel_builder_v4",
    ) or "Best Available"
    mode = "longshot" if builder_label.startswith("Priced Longshot") else "best"
    preset_options = parlay_presets_for_sport(sport_filter)
    preset = st.selectbox("Market focus", preset_options, key=f"intel_preset_v4_{sport_filter}")
    ticket_timezone = "Pacific/Honolulu"
    local_today = datetime.now(ZoneInfo(ticket_timezone)).date()
    ticket_local_date = st.date_input(
        "Games date",
        value=local_today,
        min_value=local_today,
        max_value=local_today + timedelta(days=7),
        help="Parlay markets are scoped to this calendar date in Hawaiʻi time.",
        key=f"intel_ticket_date_v4_{sport_filter}",
    )
    parlay_markets, parlay_catalog_error, parlay_catalog_complete, parlay_catalog_incomplete = get_parlay_kalshi_markets(sport_filter)
    parlay_grouped = group_kalshi_sports(parlay_markets)
    parlay_events_by_ticker, parlay_events_error = get_parlay_kalshi_events()
    if parlay_events_error:
        st.caption("Kalshi event-label metadata is temporarily unavailable; using market/schedule identities instead.")
    available_games = _parlay_game_choices(
        parlay_grouped, sport_filter, ticket_local_date, ticket_timezone, parlay_events_by_ticker
    )
    available_game_titles = [title for _, _, title in available_games]
    game_labels = {key: label for key, label, _ in available_games}
    game_titles = {key: title for key, _, title in available_games}
    game_event_keys = {key for key, _, _ in available_games}
    mlb_schedule_titles = {
        key: title for key, _, title in available_games if key.startswith("MLB:")
    }
    if not parlay_catalog_complete:
        st.warning("Game selector catalog is still loading/incomplete; Sports Edge will not pretend the visible list is exhaustive.")
    if parlay_catalog_incomplete:
        st.caption("Incomplete Kalshi series: " + ", ".join(parlay_catalog_incomplete[:8]))
    if parlay_catalog_error:
        st.warning(f"Game selector warning: {parlay_catalog_error}")
    st.caption(f"{len(available_games)} open game event(s) found for {ticket_local_date.isoformat()} · Hawaiʻi time")
    game_scope = st.segmented_control(
        "Game scope",
        ["All games", "Selected games", "Single game"],
        default="All games",
        key=f"intel_game_scope_v4_{sport_filter}",
    )
    selected_game_titles: list[str] = []
    selected_game_keys: list[str] = []
    if game_scope == "Single game":
        if available_games:
            selected_key = st.selectbox(
                "Choose game",
                [key for key, _, _ in available_games],
                format_func=lambda key: game_labels.get(key, key),
                key=f"intel_single_game_v5_{sport_filter}_{ticket_local_date}",
            )
            selected_game_keys = [selected_key]
            selected_game_titles = [game_titles[selected_key]]
        else:
            st.caption("No open Kalshi game events found for this date/sport yet.")
    elif game_scope == "Selected games":
        selected_keys = st.multiselect(
            "Choose games",
            [key for key, _, _ in available_games],
            format_func=lambda key: game_labels.get(key, key),
            key=f"intel_multi_games_v5_{sport_filter}_{ticket_local_date}",
        )
        selected_game_keys = list(selected_keys)
        selected_game_titles = [game_titles[key] for key in selected_keys]
    if selected_game_titles:
        st.markdown(
            '<div class="ticket-shell"><div class="ticket-kicker">Selected slate</div>'
            + "".join(f'<div class="game-title">✓ {title}</div>' for title in selected_game_titles)
            + '</div>',
            unsafe_allow_html=True,
        )
    if game_scope != "All games" and not selected_game_titles:
        st.caption("Choose at least one game before building the ticket.")

    if game_scope == "Single game":
        # Preserve the existing Single Game cap: four strong legs max.
        min_legs = 2
        default_legs = 4
        max_legs = 4
    else:
        # Low-stake strong-core profile: default to five qualified legs and do
        # not shrink Best Available below four merely to produce a ticket.
        min_legs = 5 if mode == "longshot" else 4
        default_legs = 6 if mode == "longshot" else 5
        max_legs = 10 if mode == "longshot" else 8
    target = st.slider(
        "Target legs",
        min_value=min_legs,
        max_value=max_legs,
        value=default_legs,
        key=f"intel_target_v3_{mode}_{game_scope}",
    )
    if game_scope == "Single game":
        st.caption("Single Game remains capped at 4 strong model-qualified positive-value legs.")
    else:
        st.caption("Strong-core default: 5 legs for Best Available (6 for Longshot). If 4–5+ strong legs do not qualify, Sports Edge returns fewer rather than adding weak filler.")

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
            "Best Available is the strong-core builder: every leg must have positive independent-model value and a model-first fair probability "
            "of at least 55–58% depending on evidence depth. Standard evidence still requires +3pp / 1.05x; deep, high-confidence model evidence "
            "may qualify smaller +1–2pp positive edges. Exact Kalshi contract and adequate sport-model confidence remain mandatory. "
            "A 5-leg ticket can still be mathematically high-risk because probabilities multiply, but Best Available never switches into the longshot price band."
        )

    build_disabled = game_scope != "All games" and not selected_game_titles
    if st.button("Build Sports Edge ticket", type="primary", use_container_width=True, disabled=build_disabled):
        with st.spinner("Running sport models against current Kalshi markets…"):
            use_all_mlb_models = preset in {"Best Available", "Mixed Sports"}
            use_mlb_game_lines = preset in {"Best Available", "Mixed Sports", "MLB Game Markets"}
            use_all_nfl_models = preset in {"Best Available", "Mixed Sports"}
            use_nfl_game_lines = preset in {"Best Available", "Mixed Sports", "NFL Game Markets"}
            use_all_wnba_models = preset in {"Best Available", "Mixed Sports"}
            use_wnba_game_lines = preset in {"Best Available", "Mixed Sports", "WNBA Game Markets"}
            use_all_tennis_models = preset in {"Best Available", "Mixed Sports"}
            focused_mlb_cap = max(12, min(24, target * 3))
            broad_mlb_cap = max(10, min(16, target * 3))
            pitcher_cap = max(8, min(16, target * 2))
            focused_nfl_cap = max(12, min(24, target * 3))
            broad_nfl_cap = max(12, min(18, target * 3))
            focused_wnba_cap = max(12, min(24, target * 3))
            broad_wnba_cap = max(10, min(18, target * 3))

            st.caption(f"Ticket scope: {ticket_local_date.isoformat()} · Hawaiʻi time")

            model_candidates = model_candidates_from_kalshi(
                parlay_grouped,
                sport_filter=sport_filter,
                target_local_date=ticket_local_date,
                ticket_timezone=ticket_timezone,
                include_mlb_hits=(preset == "MLB Hits" or use_all_mlb_models),
                max_mlb_hit_players=(
                    focused_mlb_cap if preset == "MLB Hits"
                    else broad_mlb_cap if use_all_mlb_models
                    else None
                ),
                include_mlb_home_runs=(preset == "MLB Home Runs" or use_all_mlb_models),
                max_mlb_hr_players=(
                    focused_mlb_cap if preset == "MLB Home Runs"
                    else broad_mlb_cap if use_all_mlb_models
                    else None
                ),
                include_mlb_total_bases=(preset == "MLB Total Bases" or use_all_mlb_models),
                max_mlb_tb_players=(
                    focused_mlb_cap if preset == "MLB Total Bases"
                    else broad_mlb_cap if use_all_mlb_models
                    else None
                ),
                include_mlb_rbis=(preset == "MLB RBIs" or use_all_mlb_models),
                max_mlb_rbi_players=(
                    focused_mlb_cap if preset == "MLB RBIs"
                    else broad_mlb_cap if use_all_mlb_models
                    else None
                ),
                include_mlb_hrr=(preset == "MLB H+R+RBI" or use_all_mlb_models),
                max_mlb_hrr_players=(
                    focused_mlb_cap if preset == "MLB H+R+RBI"
                    else broad_mlb_cap if use_all_mlb_models
                    else None
                ),
                include_mlb_strikeouts=(preset == "MLB Strikeouts" or use_all_mlb_models),
                max_mlb_k_pitchers=(
                    pitcher_cap if (preset == "MLB Strikeouts" or use_all_mlb_models)
                    else None
                ),
                include_mlb_game_lines=use_mlb_game_lines,
                include_nfl_passing_yards=(preset == "NFL Passing" or use_all_nfl_models),
                max_nfl_passing_players=(
                    focused_nfl_cap if preset == "NFL Passing"
                    else broad_nfl_cap if use_all_nfl_models
                    else None
                ),
                include_nfl_passing_tds=(preset == "NFL Passing" or use_all_nfl_models),
                include_nfl_pass_attempts=(preset == "NFL Passing" or use_all_nfl_models),
                include_nfl_pass_completions=(preset == "NFL Passing" or use_all_nfl_models),
                include_nfl_pass_interceptions=(preset == "NFL Passing" or use_all_nfl_models),
                include_nfl_rushing_yards=(preset == "NFL Rushing" or use_all_nfl_models),
                include_nfl_rush_attempts=(preset == "NFL Rushing" or use_all_nfl_models),
                include_nfl_rush_receiving_yards=(preset == "NFL Rushing" or use_all_nfl_models),
                max_nfl_rushing_players=(
                    focused_nfl_cap if preset == "NFL Rushing"
                    else broad_nfl_cap if use_all_nfl_models
                    else None
                ),
                include_nfl_receiving_yards=(preset == "NFL Receiving" or use_all_nfl_models),
                include_nfl_receptions=(preset == "NFL Receiving" or use_all_nfl_models),
                max_nfl_receiving_players=(
                    focused_nfl_cap if preset == "NFL Receiving"
                    else broad_nfl_cap if use_all_nfl_models
                    else None
                ),
                include_nfl_touchdowns=(preset == "NFL Touchdowns" or use_all_nfl_models),
                max_nfl_td_players=(
                    focused_nfl_cap if preset == "NFL Touchdowns"
                    else broad_nfl_cap if use_all_nfl_models
                    else None
                ),
                include_nfl_game_lines=use_nfl_game_lines,
                include_wnba_game_lines=use_wnba_game_lines,
                include_wnba_player_props=(
                    preset in {"WNBA Points", "WNBA Rebounds", "WNBA Assists", "WNBA Threes", "WNBA PRA"}
                    or use_all_wnba_models
                ),
                max_wnba_players=(
                    focused_wnba_cap
                    if preset in {"WNBA Points", "WNBA Rebounds", "WNBA Assists", "WNBA Threes", "WNBA PRA"}
                    else broad_wnba_cap if use_all_wnba_models
                    else None
                ),
                include_tennis_match_winner=(preset != "Tennis Games Total"),
                include_tennis_games_total=(
                    preset == "Tennis Games Total" or use_all_tennis_models
                ),
            )

            selected_coverage_ids: list[str] = []
            selected_coverage_labels: dict[str, str] = {}
            if selected_game_keys:
                selected_key_set = set(selected_game_keys)
                selected_title_set = set(selected_game_titles)
                selected_canonical_ids: set[str] = set()

                # Resolve exactly one preferred event identity per selected
                # game. Prefer the identity already emitted by the model
                # candidates; otherwise retain a canonical fallback so a
                # zero-qualified game can still be reported explicitly.
                for key, title in zip(selected_game_keys, selected_game_titles):
                    coverage_sport = sport_filter
                    if coverage_sport == "All" and ":" in key:
                        coverage_sport = key.split(":", 1)[0]
                    canonical_id = ""
                    if coverage_sport in SUPPORTED_SPORTS:
                        canonical_id = canonical_event_id_from_title(
                            coverage_sport, title, ticket_local_date
                        )
                        if canonical_id:
                            selected_canonical_ids.add(canonical_id)

                    actual_ids = [
                        str(getattr(row, "event_id", "") or "")
                        for row in model_candidates
                        if (
                            str(getattr(row, "event_id", "") or "") == key
                            or str(getattr(row, "event_title", "") or "") == title
                            or (
                                canonical_id
                                and str(getattr(row, "event_id", "") or "") == canonical_id
                            )
                        )
                    ]
                    coverage_id = next((event_id for event_id in actual_ids if event_id), canonical_id or key)
                    if coverage_id and coverage_id not in selected_coverage_ids:
                        selected_coverage_ids.append(coverage_id)
                        selected_coverage_labels[coverage_id] = title

                model_candidates = [
                    row for row in model_candidates
                    if (
                        str(getattr(row, "event_id", "") or "") in selected_key_set
                        or str(getattr(row, "event_id", "") or "") in selected_canonical_ids
                        or str(getattr(row, "event_title", "") or "") in selected_title_set
                    )
                ]
            elif game_scope != "All games":
                model_candidates = []

            supported_model_presets = {
                "Best Available",
                "Mixed Sports",
                "Tennis Moneyline",
                "Tennis Games Total",
                "MLB Game Markets",
                "NFL Game Markets",
                "NFL Passing",
                "NFL Rushing",
                "NFL Receiving",
                "NFL Touchdowns",
                "WNBA Game Markets",
                "WNBA Points",
                "WNBA Rebounds",
                "WNBA Assists",
                "WNBA Threes",
                "WNBA PRA",
                "MLB Hits",
                "MLB Home Runs",
                "MLB Total Bases",
                "MLB RBIs",
                "MLB H+R+RBI",
                "MLB Strikeouts",
            }
            if preset == "MLB Game Markets":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "MLB" and row.market_key in {"model_h2h", "mlb_spread", "mlb_game_total", "mlb_team_total"}
                ]
            elif preset == "MLB Hits":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "MLB" and row.market_key == "batter_hits"
                ]
            elif preset == "MLB Home Runs":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "MLB" and row.market_key == "batter_home_runs"
                ]
            elif preset == "MLB Total Bases":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "MLB" and row.market_key == "batter_total_bases"
                ]
            elif preset == "MLB RBIs":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "MLB" and row.market_key == "batter_rbis"
                ]
            elif preset == "MLB H+R+RBI":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "MLB" and row.market_key == "batter_hrr"
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
            elif preset == "Tennis Games Total":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "Tennis" and row.market_key == "tennis_games_total"
                ]
            elif preset == "NFL Game Markets":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "NFL" and row.market_key in {"model_h2h", "nfl_spread", "nfl_game_total", "nfl_team_total"}
                ]
            elif preset == "NFL Passing":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "NFL" and row.market_key in {"player_pass_yds", "player_pass_tds", "player_pass_attempts", "player_pass_completions", "player_pass_interceptions"}
                ]
            elif preset == "NFL Rushing":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "NFL" and row.market_key in {"player_rush_yds", "player_rush_attempts", "player_rush_reception_yds"}
                ]
            elif preset == "NFL Receiving":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "NFL" and row.market_key in {"player_reception_yds", "player_receptions"}
                ]
            elif preset == "NFL Touchdowns":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "NFL" and row.market_key == "player_anytime_td"
                ]
            elif preset == "WNBA Game Markets":
                model_candidates = [
                    row for row in model_candidates
                    if row.sport == "WNBA" and row.market_key in {"model_h2h", "wnba_spread", "wnba_game_total", "wnba_team_total"}
                ]
            elif preset == "WNBA Points":
                model_candidates = [row for row in model_candidates if row.sport == "WNBA" and row.market_key == "player_points"]
            elif preset == "WNBA Rebounds":
                model_candidates = [row for row in model_candidates if row.sport == "WNBA" and row.market_key == "player_rebounds"]
            elif preset == "WNBA Assists":
                model_candidates = [row for row in model_candidates if row.sport == "WNBA" and row.market_key == "player_assists"]
            elif preset == "WNBA Threes":
                model_candidates = [row for row in model_candidates if row.sport == "WNBA" and row.market_key == "player_threes"]
            elif preset == "WNBA PRA":
                model_candidates = [row for row in model_candidates if row.sport == "WNBA" and row.market_key == "player_points_rebounds_assists"]
            elif preset not in supported_model_presets:
                model_candidates = []

            # Render from independent sport models first. Initial construction
            # and optional enrichment must share the exact same scope policy.
            result = build_ticket_for_scope(
                model_candidates,
                builder=build_intelligent_parlay,
                assessor=assess_leg,
                mode=mode,
                target_legs=target,
                sport_filter=sport_filter,
                game_scope=game_scope,
                preferred_event_ids=selected_coverage_ids,
            )

            st.session_state["intel_parlay_v3"] = {
                "result": result,
                "preset": preset,
                "mode": mode,
                "sport": sport_filter,
                "target": target,
                "ticket_date": ticket_local_date.isoformat(),
                "game_scope": game_scope,
                "selected_games": tuple(selected_game_titles),
                "selected_coverage_ids": tuple(selected_coverage_ids),
                "selected_coverage_labels": dict(selected_coverage_labels),
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
    state_matches = ticket_state_matches(
        state,
        mode=mode,
        preset=preset,
        sport=sport_filter,
        target=target,
        ticket_date=ticket_local_date.isoformat(),
        game_scope=game_scope,
        selected_games=selected_game_titles,
    )

    if state_matches and api_key and state.get("model_candidate_rows") and not state.get("secondary_done"):
        st.caption("Model-first ticket is ready. Sportsbook confirmation is optional and runs separately.")
        if st.button("Add optional sportsbook cross-check", use_container_width=True, key="secondary_crosscheck_v4"):
            with st.spinner("Cross-checking a bounded set of exact-game sportsbook markets…"):
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
                    model_targets=model_candidates,
                )
                candidates = attach_sportsbook_context(model_candidates, book_candidates)
                result = build_ticket_for_scope(
                    candidates,
                    builder=build_intelligent_parlay,
                    assessor=assess_leg,
                    mode=mode,
                    target_legs=target,
                    sport_filter=sport_filter,
                    game_scope=game_scope,
                    preferred_event_ids=state.get("selected_coverage_ids") or (),
                )
                st.session_state["intel_parlay_v3"] = {
                    **state,
                    "result": result,
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
    if state_matches:
        result = state["result"]
        if result.legs:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Qualified legs", len(result.legs))
            c2.metric("Model fair", f"{result.fair_joint_probability:.2%}")
            c3.metric("Kalshi implied", f"{result.market_joint_probability:.2%}")
            c4.metric("Current payout", f"{result.market_payout_multiplier:.2f}x")
            st.caption(
                f"Build: {builder_label.upper()} · Ticket math risk: {result.risk_label} · "
                f"model/market value ratio {result.ticket_value_multiple:.2f}x"
            )

            rows = pd.DataFrame(
                [
                    {
                        "Score": f"{row.score:.0f}",
                        "Sport": row.leg.sport,
                        "Game": row.leg.event_title,
                        "Selection": row.leg.selection,
                        "Model": row.leg.model_name or "—",
                        "Model conf": f"{row.leg.model_confidence:.0%}",
                        "Involvement": row.involvement_rating,
                        "Variance": row.variance_rating,
                        "Role check": row.role_check,
                        "Sharp-method fit": f"{row.community_methodology_score:.0%}",
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
                    st.write(
                        f"**Involvement:** {row.involvement_rating} · "
                        f"**Variance:** {row.variance_rating} · "
                        f"**Role check:** {row.role_check}"
                    )
                    st.write(
                        f"**Public-sharp methodology fit:** "
                        f"{row.community_methodology_score:.0%}"
                    )
                    for note in row.community_methodology_notes:
                        st.caption("• Social/process: " + note)
                    for reason in row.reasons:
                        st.caption("• " + reason)
                    if row.warnings:
                        st.warning(" · ".join(row.warnings))

            st.markdown("### Failure map")
            f1, f2 = st.columns(2)
            f1.write(f"**Strongest leg:** {result.strongest_leg or '—'}")
            f1.write(f"**Weakest leg:** {result.weakest_leg or '—'}")
            f2.write(f"**Highest-variance leg:** {result.highest_variance_leg or '—'}")
            f2.write(f"**Primary failure scenario:** {result.primary_failure_scenario or '—'}")
            if game_scope == "Selected games":
                coverage_labels = state.get("selected_coverage_labels") or {}
                requested_count = int(
                    getattr(result, "requested_event_count", 0)
                    or len(coverage_labels)
                )
                represented_count = int(
                    getattr(result, "represented_event_count", 0)
                    or len({
                        str(getattr(row.leg, "event_id", "") or "")
                        for row in result.legs
                        if str(getattr(row.leg, "event_id", "") or "") in coverage_labels
                    })
                )
                missing_ids = tuple(
                    getattr(result, "missing_event_ids", ())
                    or tuple(
                        event_id
                        for event_id in coverage_labels
                        if event_id not in {
                            str(getattr(row.leg, "event_id", "") or "")
                            for row in result.legs
                        }
                    )
                )
                if requested_count:
                    st.caption(
                        f"Selected-game coverage: {represented_count}/"
                        f"{requested_count} represented."
                    )
                if missing_ids:
                    missing_labels = [
                        coverage_labels.get(event_id, event_id)
                        for event_id in missing_ids
                    ]
                    st.warning(
                        "No qualifying leg from: " + " · ".join(missing_labels)
                        + ". Sports Edge will not force a weak leg just to represent the game."
                    )
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
            if preset not in {"Best Available", "Mixed Sports", "Tennis Moneyline", "MLB Game Markets", "NFL Game Markets", "NFL Passing", "NFL Rushing", "NFL Receiving", "NFL Touchdowns", "WNBA Game Markets", "MLB Hits", "MLB Home Runs", "MLB Total Bases", "MLB RBIs", "MLB H+R+RBI", "MLB Strikeouts"}:
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
    _section_hero("LIVE COMMAND", "What is changing now", "Current game state and live reversal intelligence, with freshness checks and fail-closed evidence rules.")
    st.markdown('<div class="section-note">Fast official/public game-state feeds plus Tennis price-path reversal intelligence. This page is for what is happening now, not futures.</div>', unsafe_allow_html=True)

    sofascore_snapshot, sofascore_err = get_sofascore_live_snapshot_cached()
    with st.expander("External corroboration sources", expanded=False):
        st.caption(
            "TennisExplorer: active Tennis research source for ranking, yearly/surface record, recent form and prior H2H when the primary Tennis model has a coverage hole."
        )
        if sofascore_err:
            st.caption("SofaScore: optional cross-sport corroboration unavailable — " + sofascore_err)
        elif sofascore_snapshot:
            source_rows = []
            for result in sofascore_snapshot:
                if result.available:
                    state = f"available · {len(result.events)} live event(s)"
                elif result.blocked:
                    state = f"blocked by upstream ({result.status_code}); excluded from evidence"
                else:
                    state = result.error or "unavailable; excluded from evidence"
                source_rows.append({"Sport": result.sport, "SofaScore": state})
            st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("SofaScore: no usable response; excluded from SportsEdge evidence.")

    nfl, nerr = get_nfl_live()
    mlb, merr = get_mlb_live()

    st.subheader("Tennis Live Reversal Radar")
    st.caption(
        "Scans all open ATP/WTA/Challenger/ITF match-winner contracts for a major executable-price dip followed by a real rebound. "
        "The DEEP REVERSAL lane targets 4–25¢ underdogs only when executable-price recovery is corroborated by live sets/games from ESPN or the cloud-reachable Tennis365 lower-tour layer. "
        "Sports Edge's primary Tennis model remains first; TennisExplorer can independently fill rank/form/H2H context only when that model has a coverage hole."
    )

    def _render_tennis_reversal_radar():
        live_tennis_markets, tennis_market_err = get_tennis_match_markets_live()
        if tennis_market_err:
            st.warning(tennis_market_err)
        if not live_tennis_markets:
            st.info("No open Tennis match-winner markets are available for the live radar.")
            return

        live_grouped = group_kalshi_sports(live_tennis_markets)
        tennis_candidates = [
            row for row in model_candidates_from_kalshi(
                live_grouped,
                sport_filter="Tennis",
            )
            if row.sport == "Tennis"
            and row.market_key == "model_h2h"
            and row.kalshi_ticker
        ]

        # ESPN remains the preferred ATP/WTA structural feed while Tennis365
        # supplies cloud-reachable Challenger/ITF live state. Cross-feed
        # disagreements are blocked from strong reversal promotion.
        score_states, score_state_err = get_tennis_score_states_live()
        if score_state_err:
            st.caption("Public Tennis score-state feed unavailable: " + score_state_err)
        live_state_index = {
            (row.event_id, row.selection_key): row
            for row in score_states
        }
        confirmed_live_ids: set[str] = {row.event_id for row in score_states}
        live_game_count = len(confirmed_live_ids)

        # Primary Tennis history stays first. Any confirmed-live physical match
        # missing from that model may receive independent Tennis365 and/or
        # TennisExplorer ranking/form/H2H context. ATP/WTA still requires ESPN
        # structural score authority; lower tours may use Tennis365 structural
        # state directly. Research providers never replace live-score authority.
        lower_tour_candidates = build_lower_tour_live_fallback_candidates(
            live_tennis_markets,
            score_states,
            tennis_candidates,
        )
        if lower_tour_candidates:
            tennis_candidates.extend(lower_tour_candidates)

        coverage = build_tennis_live_coverage(
            live_tennis_markets,
            score_states,
            tennis_candidates,
        )

        tickers = tuple(sorted({
            str(row.kalshi_ticker)
            for row in tennis_candidates
            if row.kalshi_ticker
        }))
        candles, candle_err = get_tennis_candles_live(tickers)
        if candle_err:
            st.warning(candle_err)

        preliminary = build_tennis_reversal_radar(
            tennis_candidates,
            candles,
            confirmed_live_event_ids=confirmed_live_ids,
            live_states=live_state_index,
        )

        # The Odds API score feed remains a secondary live-confirmation fallback
        # for matches ESPN does not cover. Only spend those calls on price/model
        # rows already clearing the WATCH gate.
        if api_key and preliminary:
            active, active_err = get_active_sports(api_key)
            if not active_err:
                tennis_games, _, _ = build_game_universe(api_key, active, "Tennis")
                games_by_pair: dict[str, list[GameEvent]] = {}
                for game in tennis_games:
                    cid = canonical_event_id_from_game(game)
                    parts = cid.split(":", 2)
                    if len(parts) == 3:
                        games_by_pair.setdefault(parts[2], []).append(game)

                watch_ids = {row.event_id for row in preliminary}
                candidate_games: list[tuple[str, GameEvent]] = []
                for event_id in watch_ids:
                    parts = event_id.split(":", 2)
                    if len(parts) != 3:
                        continue
                    matches = games_by_pair.get(parts[2], [])
                    if len(matches) == 1:
                        candidate_games.append((event_id, matches[0]))

                score_cache: dict[str, list[dict]] = {}
                for _, game in candidate_games[:12]:
                    if game.sport_key in score_cache:
                        continue
                    payload, score_err = get_scores(api_key, game.sport_key)
                    if not score_err:
                        score_cache[game.sport_key] = payload

                now_utc = datetime.now(timezone.utc)
                for model_event_id, game in candidate_games:
                    if game.commence_time > now_utc:
                        continue
                    for score_event in score_cache.get(game.sport_key, []):
                        if str(score_event.get("id") or "") != game.event_id:
                            continue
                        if score_event.get("completed") is True:
                            continue
                        scores = score_event.get("scores")
                        if not isinstance(scores, list) or len(scores) < 2:
                            continue
                        last_update = score_event.get("last_update")
                        if not last_update:
                            continue
                        try:
                            updated = datetime.fromisoformat(str(last_update).replace("Z", "+00:00"))
                            if updated.tzinfo is None:
                                updated = updated.replace(tzinfo=timezone.utc)
                            age_s = (now_utc - updated.astimezone(timezone.utc)).total_seconds()
                            if age_s < -30 or age_s > 180:
                                continue
                        except ValueError:
                            continue
                        confirmed_live_ids.add(model_event_id)
                        break

        radar = build_tennis_reversal_radar(
            tennis_candidates,
            candles,
            confirmed_live_event_ids=confirmed_live_ids,
            live_states=live_state_index,
        )
        early_watches = build_early_reversal_watches(
            tennis_candidates, candles, live_state_index,
        )
        extreme_dips = build_extreme_cheap_observations(
            tennis_candidates, candles, live_state_index,
        )
        # Only operator-configured persistent storage may be used for a
        # prospective first-trigger ledger. Streamlit's ephemeral filesystem
        # is intentionally NOT accepted as a production ledger.
        ledger_path = os.environ.get("SPORTSEDGE_TENNIS_LEDGER_PATH")
        ledger_metrics = None
        ledger_error = None
        if ledger_path:
            try:
                ledger = _tennis_ledger.TennisSignalLedger(ledger_path)
                utc_now = datetime.now(timezone.utc)
                for early in early_watches:
                    side = next(
                        (str(candidate.kalshi_side).upper()
                         for candidate in tennis_candidates
                         if str(candidate.kalshi_ticker) == early.ticker),
                        "YES",
                    )
                    ledger.record_first(_tennis_ledger.SignalObservation(
                        event_id=early.event_id, ticker=early.ticker,
                        selection=early.selection,side=side,lane=early.status,
                        model_version=early.model_version, observed_at=utc_now,
                        entry_ask=early.price,model_fair=early.fair,
                        score_state=early.score,
                        evidence={"edge_pp":early.edge_pp,
                                  "trough":early.trough,
                                  "quote_spread_pp":early.quote_spread_pp,
                                  "point_aware":early.point_aware},
                    ))
                ledger_metrics = ledger.metrics()
            except Exception as exc:
                ledger_error = str(exc)
        st.markdown("### 🎯 Extreme Dip Tracker & Already-Moved Surges")
        st.caption(
            "Tracks verified quoted asks as low as 1–4¢ even when a match "
            "looks unfavorable. RECOVERY BUILDING requires supported live value. "
            "MOVED ALREADY flags a rebound past 20¢ for postmortem study, "
            "not as an entry. Candle trade lows are never presented as quoted fills."
        )
        if extreme_dips:
            st.dataframe(pd.DataFrame([
                {
                    "Player": row.selection,
                    "Current ask": f"{row.current_ask:.0%}",
                    "Quoted trough": f"{row.observed_trough:.0%}",
                    "Recovered": f"+{row.rebound_pp:.1f}pp",
                    "Live fair estimate": (
                        f"{row.live_fair:.1%}" if row.live_fair is not None else "unavailable"
                    ),
                    "Live score": row.score,
                    "Status": row.lane,
                }
                for row in extreme_dips[:30]
            ]), use_container_width=True, hide_index=True)
        else:
            st.caption(
                "No current 1–4¢ quoted dip or eligible early recovery. "
                "This is not proof that no cheap prices exist in markets without live score/model coverage."
            )

        recent_signals, recent_error = get_tennis_recent_prospective_signals()
        current_utc = datetime.now(timezone.utc)
        recently_seen = []
        for signal in recent_signals:
            try:
                first = datetime.fromisoformat(signal["first_observed_at"].replace("Z", "+00:00"))
                age_minutes = (current_utc - first).total_seconds() / 60
                if 0 <= age_minutes <= 120:
                    recently_seen.append((signal, age_minutes))
            except (TypeError, KeyError, ValueError, AttributeError):
                continue
        if recently_seen:
            with st.expander(
                f"Recorded cheap Tennis watches in the last 2 hours ({len(recently_seen)})",
                expanded=True,
            ):
                st.caption(
                    "Immutable first-observed signals, NOT live recommendations. "
                    "Prices/score states shown are historical at detection time, "
                    "and some matches may already have ended."
                )
                st.dataframe(pd.DataFrame([
                    {
                        "Seen UTC": s["first_observed_at"][11:16],
                        "Player": s.get("selection", "unknown"),
                        "First ask": f"{s['entry_ask']:.0%}",
                        "Model live fair": f"{s['model_fair']:.1%}",
                        "Lane": s.get("lane", ""),
                        "Score at detection": s.get("score_state", ""),
                    }
                    for s, age in recently_seen[:30]
                ]), use_container_width=True, hide_index=True)
        elif recent_error:
            st.caption("Recent first-trigger history is currently unavailable.")

        st.markdown("### 🔬 Early Tennis Reversal Watch · Research")
        st.caption(
            "Experimental, uncalibrated pre-confirmation signals; not DEEP REVERSAL "
            "entries. Requires live score authority, independent model value, "
            "fresh two-sided quotes and a prior collapse. A point-aware "
            "estimate is used only when server and point score are known."
        )
        if early_watches:
            st.dataframe(pd.DataFrame([
                {"Player":row.selection,"Ask":f"{row.price:.0%}",
                 "Structural fair":f"{row.fair:.0%}",
                 "Estimated edge":f"{row.edge_pp:+.1f}pp",
                 "Trough":f"{row.trough:.0%}",
                 "Rebound":f"+{row.rebound_pp:.1f}pp",
                 "Point-aware":"yes" if row.point_aware else "no",
                 "Score":row.score,"Status":row.status}
                for row in early_watches[:20]
            ]),use_container_width=True,hide_index=True)
        else:
            st.caption("No early research watches currently meet the quality requirements.")
        cloud_report, cloud_report_error = get_tennis_cloud_prospective_report()
        with st.expander("Prospective Tennis reversal results · Git-backed", expanded=False):
            st.caption(
                "Observations are recorded by a separate scheduled, read-only GitHub worker. "
                "First observed quoted asks are NOT actual fills. Kalshi settlement grades "
                "are separate and never inferred from the live score."
            )
            if cloud_report is not None:
                st.caption(
                    f"Persistent samples: {cloud_report.get('total_observations', 0)} · "
                    f"Officially graded: {cloud_report.get('total_settled', 0)} · "
                    f"Pending: {cloud_report.get('total_pending', 0)} · "
                    f"Price/score snapshots: {cloud_report.get('recorded_snapshots', 0)} · "
                    f"Extreme-dip snapshots: {cloud_report.get('extreme_dip_quote_snapshots', 0)} · "
                    f"Already-moved audits: {cloud_report.get('already_moved_postmortems', 0)}"
                )
                run_status = cloud_report.get("last_worker_run")
                if isinstance(run_status, dict):
                    st.caption(
                        f"Last recorded worker heartbeat: {run_status.get('observed_at', 'unknown')} · "
                        f"Healthy: {run_status.get('healthy', False)}"
                    )
                summary_rows = []
                for lane, metrics in (cloud_report.get("by_lane") or {}).items():
                    causal = metrics.get("causal_forward") or {}
                    summary_rows.append({
                        "Lane": lane,
                        "First observations": metrics.get("observations", 0),
                        "Kalshi-settled": metrics.get("settled", 0),
                        "Wins": metrics.get("wins", 0),
                        "Losses": metrics.get("losses", 0),
                        "Win rate": (
                            f"{metrics['win_rate']:.1%}"
                            if metrics.get("win_rate") is not None else "not established"
                        ),
                        "Causal forward samples": causal.get("evaluated", 0),
                        "Forward model Brier": (
                            f"{causal['model_brier']:.3f}"
                            if causal.get("model_brier") is not None else "—"
                        ),
                        "Forward Kalshi Brier": (
                            f"{causal['quote_brier']:.3f}"
                            if causal.get("quote_brier") is not None else "—"
                        ),
                    })
                if summary_rows:
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("Prospective data collection is running, but no first-trigger Tennis signals have been recorded.")
                st.caption("No calibrated accuracy gain or realized ROI is claimed before unseen verified outcomes exist.")
            else:
                st.caption(
                    "No readable durable cloud report yet. The worker or its data branch may "
                    f"not be accessible from this app. {cloud_report_error or ''}"
                )
            st.markdown(
                "[Open Git-backed Tennis research ledger](https://github.com/"
                "jwalton180-jpg/sports-edge-bot/tree/tennis-prospective-data/research/tennis_prospective)"
            )

        if ledger_metrics:
            st.caption(
                f"Prospective first-observation ledger: {ledger_metrics['total']} recorded, "
                f"{ledger_metrics['settled']} settled, {ledger_metrics['pending']} pending. "
                + (f"Observed win rate {ledger_metrics['hit_rate']:.1%}."
                   if ledger_metrics['hit_rate'] is not None
                   else "Win rate not established.")
            )
        elif ledger_error:
            st.warning("Tennis ledger storage unavailable: " + ledger_error)
        else:
            st.caption(
                "Durable first-trigger recording is disabled until "
                "SPORTSEDGE_TENNIS_LEDGER_PATH is configured on persistent storage. "
                "No historical win rate is claimed."
            )

        live_game_count = len(confirmed_live_ids)
        deep = [row for row in radar if row.status == "DEEP REVERSAL"]
        strong = [row for row in radar if row.status == "REVERSAL SIGNAL"]
        watch = [row for row in radar if row.status == "WATCH"]

        st.markdown("### 🎾 Live Tennis Coverage")
        st.caption(
            "Match-level audit of the Kalshi Tennis reversal universe. "
            "A match is called live only when a structural score feed confirms it. "
            "Kalshi start-time-passed matches without score state are shown separately as unverified."
        )
        q1, q2, q3, q4, q5, q6 = st.columns(6)
        q1.metric("Open Kalshi matches", coverage.open_matches)
        q2.metric("Confirmed live", coverage.confirmed_live_matches)
        q3.metric("Score tracked", coverage.score_tracked_matches)
        q4.metric("Model covered live", coverage.model_covered_live_matches)
        q5.metric("Unsupported live", coverage.unsupported_live_matches)
        q6.metric("Needs verification", coverage.start_passed_unverified_matches)

        if coverage.confirmed_live_matches:
            model_pct = (
                coverage.model_covered_live_matches / coverage.confirmed_live_matches
            )
            source_text = " · ".join(
                f"{source}: {count}"
                for source, count in coverage.source_counts
            ) or "no live structural score sources"
            st.caption(
                f"Confirmed-live model coverage: {model_pct:.0%} · sources: {source_text}"
            )

        unsupported_rows = [
            row for row in coverage.rows
            if row.confirmed_live and row.unsupported_reason
        ]
        if unsupported_rows:
            st.warning(
                f"{len(unsupported_rows)} confirmed-live Tennis match(es) need model/score attention."
            )
            with st.expander("Coverage gaps — exact matches and reasons"):
                gap_table = pd.DataFrame([
                    {
                        "Tour": row.tour,
                        "Match": (
                            " vs ".join(row.participants)
                            if len(row.participants) == 2
                            else row.title
                        ),
                        "Score tracked": "yes" if row.score_tracked else "no",
                        "Score source": " + ".join(row.score_sources) or "—",
                        "Model": row.model_name or "—",
                        "Reason": row.unsupported_reason,
                        "Official ITF live": row.official_itf_url or "",
                        "Official ITF tour": row.official_itf_tour_url or "",
                    }
                    for row in unsupported_rows
                ])
                st.dataframe(
                    gap_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Official ITF live": st.column_config.LinkColumn(
                            "Official ITF live",
                            display_text="Open ITF live",
                        ),
                        "Official ITF tour": st.column_config.LinkColumn(
                            "Official ITF tour",
                            display_text="Open tour",
                        ),
                    },
                )
        elif coverage.confirmed_live_matches:
            st.success("No known match-level coverage holes in the current confirmed-live Kalshi Tennis universe.")

        unverified_rows = [
            row for row in coverage.rows
            if row.start_passed_unverified
        ]
        if unverified_rows:
            with st.expander(
                f"Needs verification — {len(unverified_rows)} start-time-passed match(es) without live score confirmation"
            ):
                st.caption(
                    "These are not claimed live. Kalshi occurrence times can be delayed or placeholder times, "
                    "so Sports Edge keeps them visible until a structural score feed confirms play or the market closes."
                )
                unverified_table = pd.DataFrame([
                    {
                        "Tour": row.tour,
                        "Match": (
                            " vs ".join(row.participants)
                            if len(row.participants) == 2
                            else row.title
                        ),
                        "Model covered": "yes" if row.model_covered else "no",
                        "Model": row.model_name or "—",
                        "Reason": "scheduled start passed; no structural live score confirmation",
                        "Official ITF live": row.official_itf_url or "",
                        "Official ITF tour": row.official_itf_tour_url or "",
                    }
                    for row in unverified_rows
                ])
                st.dataframe(
                    unverified_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Official ITF live": st.column_config.LinkColumn(
                            "Official ITF live",
                            display_text="Open ITF live",
                        ),
                        "Official ITF tour": st.column_config.LinkColumn(
                            "Official ITF tour",
                            display_text="Open tour",
                        ),
                    },
                )

        st.caption(
            "Official ITF cross-check: "
            "[World Tennis Tour Live](https://www.itftennis.com/en/world-tennis-tour-live/) · "
            "[Men's World Tennis Tour](https://www.itftennis.com/en/tours/mens-world-tennis-tour/) · "
            "[Women's World Tennis Tour](https://www.itftennis.com/en/tours/womens-world-tennis-tour/). "
            "These official pages are secondary corroboration; Sports Edge only treats machine-readable "
            "sets/games/server state as score confirmation."
        )

        # Put the cheapest already-qualified live opportunities at the top so a
        # fast-moving 4–20c reversal is visible before the full radar table.
        # This is presentation-only: a low price never creates a signal.
        cheap_status_priority = {
            "DEEP REVERSAL": 0,
            "REVERSAL SIGNAL": 1,
            "WATCH": 2,
        }
        cheap_live = sorted(
            [
                row for row in radar
                if row.current_price <= CHEAP_TENNIS_REVERSAL_MAX_PRICE
                and row.status in cheap_status_priority
            ],
            key=lambda row: (
                cheap_status_priority[row.status],
                -float(row.score),
                -float(row.rebound_points),
                float(row.current_price),
            ),
        )

        st.markdown(f"### 🔥 Cheap Live Underdogs ≤{CHEAP_TENNIS_REVERSAL_MAX_PRICE * 100:.0f}¢")
        st.caption(
            "Only already-qualified Tennis reversal rows appear here. "
            "Strong promotion now requires score-conditioned live value, server/break context, "
            "persistent recovery, and executable quote quality; cheap price alone never qualifies."
        )
        if cheap_live:
            cheap_table = pd.DataFrame([
                {
                    "Status": row.status,
                    "Player": row.selection,
                    "Current": f"{row.current_price:.0%}",
                    "Live model": (
                        f"{getattr(row, 'live_probability', 0.0):.0%}"
                        if getattr(row, "live_probability", None) is not None
                        else "—"
                    ),
                    "Live edge": (
                        f"{getattr(row, 'live_edge_points', 0.0):+.1f}pp"
                        if getattr(row, "live_edge_points", None) is not None
                        else "—"
                    ),
                    "Net breaks": (
                        f"{getattr(row, 'net_break_advantage', 0):+d}"
                        if getattr(row, "net_break_advantage", None) is not None
                        else "—"
                    ),
                    "Rebound": f"+{row.rebound_points:.1f}pp",
                    "Spread": (
                        f"{getattr(row, 'current_spread_points', 0.0):.1f}pp"
                        if getattr(row, "current_spread_points", None) is not None
                        else "—"
                    ),
                    "Live score": row.live_score or "—",
                    "Signal score": f"{row.score:.0f}",
                    "Match": row.event_title,
                }
                for row in cheap_live[:8]
            ])
            st.dataframe(cheap_table, use_container_width=True, hide_index=True)
            top_cheap = cheap_live[0]
            if top_cheap.status == "DEEP REVERSAL":
                st.success(
                    f"Top cheap reversal: {top_cheap.selection} at "
                    f"{top_cheap.current_price:.0%} · "
                    f"{top_cheap.event_title}"
                )
        else:
            st.caption(
                f"No ≤{CHEAP_TENNIS_REVERSAL_MAX_PRICE * 100:.0f}¢ Tennis underdog currently clears the live reversal WATCH gate."
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modeled sides", len(tennis_candidates))
        c2.metric("Deep reversals", len(deep))
        c3.metric("Signals", len(strong))
        c4.metric("Watches", len(watch))

        if not radar:
            st.info(
                "No Tennis underdog currently clears the dip + persistent rebound + live-value gate. "
                "The radar will not force a signal when the score state or executable quote does not support it."
            )
            return

        table = pd.DataFrame([
            {
                "Status": row.status,
                "Player": row.selection,
                "Match": row.event_title,
                "Current": f"{row.current_price:.0%}",
                "Peak": f"{row.local_peak:.0%}",
                "Trough": f"{row.trough_price:.0%}",
                "Dip": f"-{row.dip_points:.1f}pp",
                "Rebound": f"+{row.rebound_points:.1f}pp",
                "3m momentum": f"{row.recent_momentum_points:+.1f}pp",
                "Pregame prior": f"{row.model_prior_probability:.0%}",
                "Live model": (
                    f"{getattr(row, 'live_probability', 0.0):.0%}"
                    if getattr(row, "live_probability", None) is not None
                    else "—"
                ),
                "Live edge": (
                    f"{getattr(row, 'live_edge_points', 0.0):+.1f}pp"
                    if getattr(row, "live_edge_points", None) is not None
                    else "—"
                ),
                "Server": (
                    "player"
                    if getattr(row, "serving", None) is True
                    else ("opponent" if getattr(row, "serving", None) is False else "—")
                ),
                "Net breaks": (
                    f"{getattr(row, 'net_break_advantage', 0):+d}"
                    if getattr(row, "net_break_advantage", None) is not None
                    else "—"
                ),
                "Format": f"Bo{getattr(row, 'best_of', 3)}",
                "Spread": (
                    f"{getattr(row, 'current_spread_points', 0.0):.1f}pp"
                    if getattr(row, "current_spread_points", None) is not None
                    else "—"
                ),
                "Recovery confirms": getattr(row, "recovery_confirmations", 0),
                "Model conf.": f"{row.model_confidence:.0%}",
                "H2H": "yes" if row.h2h_context else "—",
                "Live score": row.live_score or "—",
                "Turnaround": "yes" if row.score_turnaround else "—",
                "Score": f"{row.score:.0f}",
            }
            for row in radar[:25]
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)

        for row in radar[:10]:
            label = f"{row.status} · {row.selection} · {row.current_price:.0%} · score {row.score:.0f}"
            with st.expander(label):
                st.write(f"**Match:** {row.event_title}")
                st.write(f"**Kalshi contract:** {row.ticker}")
                for reason in row.reasons:
                    st.caption("• " + reason)
                if row.warnings:
                    st.warning(" · ".join(row.warnings))

    _fragment = getattr(st, "fragment", None)
    if _fragment is not None:
        _fragment(run_every="30s")(_render_tennis_reversal_radar)()
    else:
        _render_tennis_reversal_radar()

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
    st.write("**Sport models:** model evidence is mandatory for parlay qualification. Tennis uses Elo, form trajectory, serve/return trend, workload, and recency-weighted shrunk H2H; MLB Hits, Home Runs, Total Bases, RBIs, H+R+RBI, and Pitcher Strikeouts use player/recent/game-log/opponent/probable-starter context; NFL Spread/Game Total/Team Total use current/prior scoring and defense with empirical volatility; NFL Passing Yards/TDs/Attempts/Completions/Interceptions, Rushing Yards/Attempts, Rushing + Receiving Yards, Receiving Yards, Receptions, and Player Touchdowns use current usage/efficiency, prior-season shrinkage, and conservative matchup context; MLB run lines/totals use independent team scoring/allowance Poisson baselines; MLB/NFL/NBA/WNBA game winners use public team-strength baselines; WNBA spreads/totals add scoring, defense, recent form, home court, and empirical game volatility; WNBA player props add minutes/role, per-minute production, recent efficiency, opponent history, team scoring environment, and empirical variance. Sportsbooks are secondary calibration only.")
    st.write("**Public bettors:** records must clear sample, verification, and CLV gates before they can count as supporting evidence.")
    st.warning("No pick or parlay is guaranteed. Missing, stale, conflicting, or unverified evidence fails closed.")

st.divider()
st.caption(
    f"Actionable Kalshi catalog · {len(markets)} open · {kseries} series · {kpages} pages · Kalshi request max {f'{klat:.0f} ms' if klat else '—'} · "
    f"Deployment {deployment_mode().replace('_', ' ').title()}"
)
