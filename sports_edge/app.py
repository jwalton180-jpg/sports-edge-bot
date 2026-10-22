from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import importlib
import inspect
import io
import hmac
import json
import os
import requests
import uuid
from statistics import median
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from sports_edge.core.startup import deployment_mode
from sports_edge.data.kalshi import KalshiPublicClient
# Streamlit Community Cloud can retain a previous catalog module during a
# hot reload. Resolve the catalog export from the refreshed module so the
# parlay page does not call a stale function signature after partial deploys.
_kc = importlib.import_module("sports_edge.data.kalshi_catalog")
try:
    _kc = importlib.reload(_kc)
except Exception:
    pass
fetch_supported_sport_catalog = _kc.fetch_supported_sport_catalog
from sports_edge.data.mlb import MLBClient
from sports_edge.data.mlb_prop_data import schedule_for_day
from sports_edge.data.nfl import NFLClient
from sports_edge.data.odds import OddsClient
from sports_edge.data.source_registry import get_source_status
from sports_edge.data.pick_store import SupabasePickStore, validate_prediction, parse_timestamp
from sports_edge.models.prop_performance import summarize_performance, summarize_parlays, american_to_decimal, net_units
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
from sports_edge.models.live_board import LiveSignal, build_live_signals as _sportsbook_crosschecks, build_underdog_signals, market_yes_probability, market_side_probability
from sports_edge.models.parlay import PRESETS, kalshi_copy_ticket
from sports_edge.models.parlay_runtime import guard_ticket_action

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

from sports_edge.models.kalshi_model_candidates import model_candidates_from_kalshi as _independent_model_candidates, attach_sportsbook_context, _market_local_date
from sports_edge.models.sleeper_pipeline import audit_app_candidates, scan_sleeper_markets, report_candidates
from sports_edge.models.tennis_radar_candidates import live_tennis_radar_candidates
from sports_edge.sleeper_panel import render_sleeper_panel
from dataclasses import replace

def build_live_signals(*args, **kwargs):
    return [replace(signal, status="WATCH", warnings=signal.warnings+("Sportsbook cross-check only; independent probability qualification required",)) for signal in _sportsbook_crosschecks(*args, **kwargs)]

def model_candidates_from_kalshi(grouped, **kwargs):
    candidates = _independent_model_candidates(grouped, **kwargs)
    inventory = [r.market for rows in grouped.values() for r in rows]
    chosen = kwargs.get("sport_filter", "All")
    research_markets = [m for m in inventory if m.get("sports_edge_sport") in {"NHL", "NCAAF", "Soccer", "NBA"} and chosen in {"All", m.get("sports_edge_sport")}
                        and (kwargs.get("target_local_date") is None or abs(_market_local_date(m, kwargs.get("ticket_timezone", "Pacific/Honolulu"))-kwargs["target_local_date"]) <= timedelta(days=1))]
    if research_markets:
        report = scan_sleeper_markets(research_markets, coverage_complete=bool(globals().get("parlay_catalog_complete", False)), limit=40)
        if kwargs.get("target_local_date") is not None:
            report["records"] = [r for r in report["records"] if not r.get("start_verified") or datetime.fromisoformat(r["event_time"].replace("Z", "+00:00")).astimezone(ZoneInfo(kwargs.get("ticket_timezone", "Pacific/Honolulu"))).date() == kwargs["target_local_date"]]
        candidates.extend(report_candidates(report, research_markets))
        st.session_state["last_ticket_research_report"] = report
    from sports_edge.models.sleeper_promotion import load_reviewed_promotions
    reviewed = load_reviewed_promotions("research/reviewed_model_promotions.json", verification_key=os.getenv("SPORTSEDGE_PROMOTION_REVIEW_KEY"), now=datetime.now(timezone.utc).isoformat())
    return audit_app_candidates(candidates, inventory, coverage_complete=bool(globals().get("parlay_catalog_complete", globals().get("kcursor_exhausted", False))), promotions=reviewed)


_tennis_watch_timeline = import_module_fresh("sports_edge.models.tennis_watch_timeline")
build_tennis_watch_timelines = getattr(
    _tennis_watch_timeline, "build_tennis_watch_timelines", lambda *args, **kwargs: ()
)

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
st.caption("Build: 2026-10-22-prop-picks-performance-v1")


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


def _persistent_pick_storage_configured():
    return bool(_secret("SUPABASE_URL") and _secret("SUPABASE_SERVICE_ROLE_KEY"))


def _ensure_tracker_access():
    if not _persistent_pick_storage_configured():
        return True
    expected = _secret("SPORTSEDGE_TRACKER_ACCESS_KEY")
    if not expected:
        st.warning("Persistent pick storage is disabled because SPORTSEDGE_TRACKER_ACCESS_KEY is not configured. This public app will use session-only tracking.")
        return True
    if st.session_state.get("sportedge_tracker_authorized"):
        return True
    with st.form("private_tracker_unlock_v1"):
        entered = st.text_input("Private tracker access key", type="password")
        submitted = st.form_submit_button("Unlock private tracker", use_container_width=True)
    if submitted:
        if hmac.compare_digest(entered, expected):
            st.session_state["sportedge_tracker_authorized"] = True
            st.rerun()
        else:
            st.error("Tracker access key did not match.")
    st.info("Private tracker is locked. Saved database records are not loaded until the tracker is unlocked.")
    return False


def _pick_store():
    url = _secret("SUPABASE_URL")
    key = _secret("SUPABASE_SERVICE_ROLE_KEY")
    access_key = _secret("SPORTSEDGE_TRACKER_ACCESS_KEY")
    if not url or not key or not access_key or not st.session_state.get("sportedge_tracker_authorized"):
        return None
    try:
        return SupabasePickStore(url, key)
    except Exception:
        return None


def _session_pick_records():
    return st.session_state.setdefault("sportedge_session_pick_records", [])


def _safe_event_start(value):
    try:
        return parse_timestamp(value)
    except (TypeError, ValueError):
        return None


def _record_payload(record):
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _pick_records(store):
    if store is not None:
        return store.list_records()
    return list(_session_pick_records())


def _settlement_by_prediction(records):
    settled = {}
    for record in records:
        if record.get("record_type") != "settlement":
            continue
        payload = _record_payload(record)
        prediction_id = payload.get("prediction_id")
        if prediction_id:
            settled[prediction_id] = payload
    return settled


def _save_research_prediction(payload, store):
    row = validate_prediction(payload)
    row.setdefault("record_origin", "manual")
    if store is not None:
        return store.append_prediction(row)
    prediction_id = str(uuid.uuid4())
    row["prediction_id"] = prediction_id
    row["saved_at"] = datetime.now(timezone.utc).isoformat()
    record = {
        "id": prediction_id,
        "record_type": "prediction",
        "payload": row,
        "created_at": row["saved_at"],
    }
    _session_pick_records().append(record)
    return record


def _safe_error(exc: Exception) -> str:
    msg = str(exc)
    if "apiKey=" in msg:
        msg = msg.split("apiKey=", 1)[0] + "apiKey=REDACTED"
    return msg[:300]


def _parlay_build_failure(error_type: str) -> None:
    st.error(
        "Ticket build could not finish because a sports-data, Kalshi, or model "
        "request failed. No new ticket was produced. Retry when the source "
        "recovers; if it repeats, send the error type below."
    )
    st.caption(f"Parlay diagnostic: {error_type} (no credentials or private URLs shown)")


@st.cache_data(ttl=60, show_spinner=False)
def get_kalshi_markets(sport_filter_value: str):
    selected = None if sport_filter_value == "All" else (sport_filter_value,)
    try:
        result = fetch_supported_sport_catalog(
            page_limit_per_series=50,
            page_size=1000,
            request_pause_s=0.0,
            max_workers=6,
            request_interval_s=0.15,
            sports=selected,
            overview_only=(sport_filter_value == "All"),
        )
    except Exception as exc:
        # A temporary Kalshi outage should not crash the complete Streamlit UI.
        return [], _safe_error(exc), None, 0, False, 0, ()
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
    """Full parlay game/date catalog, with fail-closed provider recovery."""
    selected = None if sport_filter_value == "All" else (sport_filter_value,)
    try:
        result = fetch_supported_sport_catalog(
            page_limit_per_series=50,
            page_size=1000,
            request_pause_s=0.0,
            max_workers=6,
            request_interval_s=0.15,
            sports=selected,
            overview_only=False,
            # Use the bounded production series allowlist. Full /series discovery
            # enumerates thousands of unrelated series and overloads this UI path.
            discover_all_series=False,
        )
    except Exception as exc:
        # Do not reuse an old market catalog or manufacture game identities.
        return [], _safe_error(exc), False, ()
    return (
        list(result.markets),
        result.error,
        result.cursor_exhausted,
        result.incomplete_series,
    )


@st.cache_data(ttl=60, show_spinner=False)
def get_parlay_event_by_ticker(event_ticker: str):
    """Resolve an exact parent event omitted from the bulk open-event page."""
    ticker = str(event_ticker or "").strip()
    if not ticker or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for ch in ticker):
        return {}
    client = KalshiPublicClient()
    try:
        payload = client.http.get_json(f"{client.base}/events/{ticker}")
        if not isinstance(payload, dict):
            return {}
        event = payload.get("event")
        if isinstance(event, dict):
            return event
        # Some compatible API responses return the event object directly.
        return payload if payload.get("event_ticker") or payload.get("ticker") else {}
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def get_parlay_kalshi_events():
    """Best-effort open-event metadata for ticket labels; never a fatal dependency."""
    client = KalshiPublicClient()
    events: dict[str, dict] = {}
    cursor = None
    error = None
    seen_cursors: set[str] = set()
    for _ in range(50):
        try:
            # We only need the parent event names. Nested markets multiply the
            # payload by hundreds and can cause timeouts / transient 429s.
            payload = client.events(status="open", limit=200, cursor=cursor, with_nested_markets=False)
            if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
                raise ValueError("malformed Kalshi event-label payload")
            for event in payload["events"]:
                if not isinstance(event, dict):
                    continue
                key = str(event.get("event_ticker") or event.get("ticker") or "").strip()
                if key:
                    events[key] = event
            next_cursor = str(payload.get("cursor") or "").strip()
            if not next_cursor:
                break
            if next_cursor in seen_cursors:
                raise ValueError("Kalshi event-label pagination repeated a cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        except Exception as exc:
            error = _safe_error(exc)
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


@st.cache_data(ttl=60, show_spinner=False)
def get_tennis_recent_prospective_grades():
    url = (
        "https://raw.githubusercontent.com/jwalton180-jpg/sports-edge-bot/"
        "tennis-prospective-data/research/tennis_prospective/settlements.jsonl"
    )
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
        return tuple(row for row in rows if isinstance(row, dict)), None
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


@st.cache_data(ttl=300, show_spinner=False)
def get_tennis_market_opportunity_report():
    root = (
        "https://raw.githubusercontent.com/jwalton180-jpg/sports-edge-bot/"
        "tennis-market-alert-data/research/tennis_market_tracking/"
    )
    try:
        response = requests.get(root + "market_opportunities.json", timeout=8)
        response.raise_for_status()
        report = response.json()
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            return None, None, "Unrecognized market opportunity report."
        status_response = requests.get(root + "latest_poll.json", timeout=8)
        poll = status_response.json() if status_response.ok else None
        return report, poll, None
    except (requests.RequestException, ValueError, TypeError) as exc:
        return None, None, _safe_error(exc)


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
                "Book cross-check": f"{s.fair_probability:.1%}",
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
        return ["Best Available", "NFL Game Markets", "NFL Passing", "NFL Rushing", "NFL Receiving", "NFL Touchdowns", "NFL 2+ Touchdowns"]
    if sport_filter_value == "NHL":
        return ["Best Available", "NHL Sleeper Markets"]
    if sport_filter_value == "NCAAF":
        return ["Best Available", "NCAA Upset Intelligence"]
    if sport_filter_value == "Soccer":
        return ["Best Available", "Soccer Totals / BTTS"]
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
    "Discover": "Edge Board",
    "Games": "Games",
    "Markets": "Game Lines",
    "Props": "Player Props",
    "Parlays": "Parlay Generator",
    "Live": "Live Feed",
}
more_views = {
    "Scorer Lab": "Scorer Lab",
    "My Picks": "My Picks",
    "Performance": "Performance",
    "Data Sources": "Data Sources",
}
reverse_nav = {value: key for key, value in nav_map.items()}
current_nav = reverse_nav.get(st.session_state.view, "More" if st.session_state.view in more_views.values() else "Discover")
nav_labels = list(nav_map) + ["More"]

st.markdown('<div class="section-eyebrow">EXPLORE</div>', unsafe_allow_html=True)
selected_nav = st.segmented_control(
    "Navigation",
    nav_labels,
    default=current_nav,
    key="main_nav_v2",
    label_visibility="collapsed",
)
if selected_nav == "More":
    more_default = st.session_state.view if st.session_state.view in more_views.values() else "My Picks"
    selected_more = st.segmented_control(
        "More tools",
        list(more_views),
        default=more_default,
        key="more_nav_v1",
        label_visibility="collapsed",
    )
    view = more_views.get(selected_more or more_default, "My Picks")
else:
    view = nav_map.get(selected_nav or current_nav, "Edge Board")
st.session_state.view = view

sport_filter = st.segmented_control(
    "Sport",
    ["All", "MLB", "NBA", "WNBA", "NFL", "Tennis", "NHL", "NCAAF", "Soccer"],
    default="All",
    key="sport_filter_v2",
    label_visibility="collapsed",
 ) or "All"

# NHL goal-scorer pipeline: refresh this dependency explicitly because Streamlit
# Community Cloud can reload app.py before the updated module file is in sys.modules.
if (view == "Scorer Lab" and sport_filter == "NHL") or (sport_filter == "NHL" and view == "Player Props"):
    st.markdown("### 🏒 NHL Goal Scorer Lab")
    st.caption("Official NHL schedule, roster and game logs. Research probabilities are unvalidated and do not imply betting value.")
    try:
        _nhl_scorer_module = import_module_fresh("sports_edge.data.nhl_goal_scorers")
        if _nhl_scorer_module is None:
            raise ImportError("NHL scorer module could not be loaded")
        _nhl_now = datetime.now(ZoneInfo("Pacific/Honolulu"))
        _nhl_day = _nhl_now.date().isoformat()
        _nhl_start = _nhl_now.year if _nhl_now.month >= 7 else _nhl_now.year - 1
        _nhl_current = f"{_nhl_start}{_nhl_start + 1}"
        _nhl_prior = f"{_nhl_start - 1}{_nhl_start}"
        _nhl_report_fn = getattr(_nhl_scorer_module, "goal_scorer_report", None)
        if callable(_nhl_report_fn):
            with st.spinner("Checking official NHL schedule, rosters and player history (read-only)…"):
                _nhl_report = _nhl_report_fn(
                    _nhl_day, current_season=_nhl_current, prior_season=_nhl_prior,
                    max_per_game=2, minimum_games=10, max_skaters_per_team=24,
                )
            _nhl_rows = _nhl_report.get("candidates", [])
            st.caption(
                f"Schedule date: {_nhl_day} · Games: {_nhl_report.get('schedule_games', 0)} · "
                f"Teams: {_nhl_report.get('teams_seen', 0)} · Roster skaters checked: {_nhl_report.get('roster_players', 0)} · "
                f"Player histories checked: {_nhl_report.get('players_checked', 0)} · "
                f"History-qualified: {_nhl_report.get('history_qualified', 0)} · API errors: {_nhl_report.get('api_errors', 0)}"
            )
            _nhl_empty_message = _nhl_report.get("diagnostic", "No candidates returned.")
        else:
            # Backward-compatible path for a stale deployed module that has
            # the original candidate API but not the new diagnostics API.
            _nhl_legacy_fn = getattr(_nhl_scorer_module, "goal_scorer_candidates", None)
            if not callable(_nhl_legacy_fn):
                raise ImportError("Loaded NHL scorer module has neither report nor legacy candidate function")
            with st.spinner("Refreshing stale NHL scorer module and checking available player history…"):
                _nhl_rows = list(_nhl_legacy_fn(
                    _nhl_day, current_season=_nhl_current, prior_season=_nhl_prior, max_per_game=2,
                ))
            _nhl_empty_message = (
                "The loaded NHL module is an older version without pipeline diagnostics. "
                "Refresh the app deployment to load the current history-fallback module."
            )
            st.caption("Compatibility mode: legacy scorer API detected; diagnostics unavailable.")
        if _nhl_rows:
            st.dataframe(pd.DataFrame([{"Game": f"{r['team']} vs {r['opponent']}", "Player": r["name"],
                "Goal 1+ baseline": f"{r['probability']:.1%}", "Games used": r["games"],
                "History season": r.get("season_used", "legacy"), "Status": r["status"],
                "Caveats": "; ".join(r.get("warnings", ["Unvalidated research estimate; no odds/value check"]))}
                for r in _nhl_rows]), use_container_width=True, hide_index=True)
            if _nhl_report_fn:
                st.caption(_nhl_empty_message)
            savable_nhl = [
                (index, row) for index, row in enumerate(_nhl_rows)
                if _safe_event_start(row.get("event_start")) is not None
                and _safe_event_start(row.get("event_start")) > datetime.now(timezone.utc)
            ]
            if savable_nhl:
                selected_nhl_index = st.selectbox(
                    "NHL scorer candidate to track",
                    [index for index, _ in savable_nhl],
                    format_func=lambda index: next(f"{row['name']} · {row['team']} vs {row['opponent']}" for i, row in savable_nhl if i == index),
                    key="track_nhl_scorer_select_v1",
                )
                if st.button("Save NHL scorer research candidate to My Picks", key="track_nhl_scorer_button_v1", use_container_width=True):
                    row = _nhl_rows[selected_nhl_index]
                    try:
                        payload = {
                            "sport": "NHL",
                            "event": row.get("event_title") or f"{row['team']} vs {row['opponent']}",
                            "market": "Anytime goal scorer",
                            "selection": row["name"],
                            "observed_at": datetime.now(timezone.utc).isoformat(),
                            "event_start": row["event_start"],
                            "source": "SportEdge",
                            "record_type": "nhl_goal_scorer",
                            "record_origin": "generated",
                            "probability": row["probability"],
                            "model_version": row.get("model", "official_game_log_poisson_v1"),
                            "sample_size": row.get("games"),
                            "event_id": row.get("game_id"),
                            "notes": "; ".join(row.get("warnings", ())) + "; unvalidated baseline; no executable odds/value check",
                        }
                        tracking_store = _pick_store()
                        _save_research_prediction(payload, tracking_store)
                        if tracking_store is None:
                            st.warning("NHL scorer candidate saved to this session only. Configure the private tracker and Supabase secrets for durable tracking.")
                        else:
                            st.success("NHL scorer research candidate saved before puck drop. No wager was placed.")
                        st.rerun()
                    except Exception as exc:
                        st.warning(f"NHL scorer was not saved: {_safe_error(exc)}")
        else:
            st.warning(_nhl_empty_message)
            st.caption("No candidates are fabricated. Use the diagnostic counts to distinguish missing schedule, roster, API, and history coverage.")
    except Exception as _nhl_error:
        st.warning(f"NHL goal-scorer pipeline unavailable; no picks shown. {type(_nhl_error).__name__}: {_nhl_error}")

if st.button("↻ Refresh live data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

api_key = _secret("THE_ODDS_API_KEY")
markets, kerr, klat, kpages, kcursor_exhausted, kseries, kincomplete = get_kalshi_markets(sport_filter)
kalshi_grouped = group_kalshi_sports(markets)
catalog_health = catalog_diagnostics(markets)

render_sleeper_panel(markets, coverage_complete=kcursor_exhausted and not kerr and sport_filter != "All", scope=sport_filter+"_"+view.replace(" ", "_"))
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
                # Bulk open-event pagination can omit a parent even when its
                # child markets are present. Resolve the exact parent rather
                # than silently dropping every market for that game.
                event = get_parlay_event_by_ticker(event_key)
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
        if sport_filter in {"NHL", "NCAAF", "Soccer"}:
            st.caption("Independent models for this sport are in research mode. A listed Kalshi price is not an independent probability or a recommendation.")
            from sports_edge.data.validation_snapshot import HISTORICAL_RESEARCH_SNAPSHOT
            snapshot = HISTORICAL_RESEARCH_SNAPSHOT[sport_filter]
            with st.expander(f"{sport_filter} — verified historical research results (not live picks)"):
                st.caption(snapshot["scope"])
                st.write(snapshot["interpretation"])
                st.write({
                    "Holdout predictions": snapshot["holdout_predictions"],
                    "Holdout Brier": round(snapshot["holdout_brier"], 5),
                    "Training-baseline Brier": round(snapshot["holdout_train_baseline_brier"], 5),
                    "Holdout calibration error": round(snapshot["holdout_ece"], 5),
                    "Model status": snapshot["status"],
                })
                st.caption("Reproducible GitHub Actions run: 38553755658. Retrospective results never auto-promote a model.")

            research_export = [
                {"sport": r.sport, "family": r.family,
                 "ticker": str(r.market.get("ticker") or ""),
                 "event": str(r.market.get("event_title") or r.market.get("title") or ""),
                 "yes_ask": r.market.get("yes_ask_dollars"),
                 "no_ask": r.market.get("no_ask_dollars"),
                 "status": "RESEARCH_ONLY", "reason": "prospective_model_validation_required"}
                for r in show_rows
            ]
            st.download_button("Export research-only market evidence (JSON)",
                               data=json.dumps(research_export, default=str, indent=2),
                               file_name=f"sportsedge_{sport_filter.lower()}_research.json",
                               mime="application/json", key=f"research_export_{sport_filter}")


elif view == "Player Props":
    _section_hero("PLAYER LAB", "Props with model accountability", "See the real Kalshi prop board, then run independent Sports Edge models only where production-grade player evidence exists.")
    st.markdown(
        '<div class="section-note">Direct Kalshi prop markets first. MLB/NBA/WNBA/NFL player props and Tennis match/set/game/stat props appear here even when sportsbook enrichment is unavailable.</div>',
        unsafe_allow_html=True,
    )
    if sport_filter == "All":
        st.info("Select a sport above to load that sport’s full prop catalog.")
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
            st.info("Independent projection available; qualification requires fresh evidence and scoped validation.")
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
                savable_props = [
                    (index, row) for index, row in enumerate(assessments)
                    if row.qualified and row.leg.verified_start
                ]
                if savable_props:
                    selected_prop_index = st.selectbox(
                        "Qualified prop to track",
                        [index for index, _ in savable_props],
                        format_func=lambda index: next(row.leg.selection for i, row in savable_props if i == index),
                        key=f"track_prop_select_{sport_filter}_{family}",
                    )
                    if st.button("Save qualified prop to My Picks", key=f"track_prop_button_{sport_filter}_{family}", use_container_width=True):
                        selected_assessment = assessments[selected_prop_index]
                        selected_leg = selected_assessment.leg
                        try:
                            payload = {
                                "sport": selected_leg.sport,
                                "event": selected_leg.event_title,
                                "market": selected_leg.market_label,
                                "selection": selected_leg.selection,
                                "observed_at": datetime.now(timezone.utc).isoformat(),
                                "event_start": selected_leg.verified_start,
                                "source": "SportEdge",
                                "record_type": "prop",
                                "record_origin": "generated",
                                "probability": selected_assessment.fair_probability,
                                "model_version": selected_leg.model_name or "unknown",
                                "model_confidence": selected_leg.model_confidence,
                                "sample_size": selected_leg.model_sample_size,
                                "event_id": selected_leg.event_id,
                                "market_key": selected_leg.market_key,
                                "kalshi_ticker": selected_leg.kalshi_ticker,
                                "kalshi_price": selected_leg.kalshi_price,
                                "notes": "; ".join(selected_assessment.warnings) or None,
                            }
                            if selected_leg.median_odds is not None:
                                payload["odds_american"] = selected_leg.median_odds
                            tracking_store = _pick_store()
                            _save_research_prediction(payload, tracking_store)
                            if tracking_store is None:
                                st.warning("Qualified prop saved to this session only. Configure Supabase secrets for durable tracking.")
                            else:
                                st.success("Qualified prop saved to durable My Picks storage. Original model probability and timestamp are preserved.")
                            st.rerun()
                        except Exception as exc:
                            st.warning(f"Prop was not saved: {_safe_error(exc)}")
                else:
                    st.caption("No qualified prop with a verified event start is available to save. No unqualified candidate is promoted.")
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
    _section_hero("DISCOVER", "Sports Edge Intelligence", "The strongest current model-versus-market disagreements, ranked by evidence quality, freshness, and executable Kalshi price.")
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
        ["Best Available", "Qualified Value", "Sleeper Value", "Priced Longshot (5+ legs)"],
        default="Best Available",
        key="intel_builder_v4",
    ) or "Best Available"
    mode = {"Qualified Value":"value", "Sleeper Value":"sleeper"}.get(builder_label, "longshot" if builder_label.startswith("Priced Longshot") else "best")
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
    if ticket_local_date.weekday() == 3 and sport_filter in {"NFL", "All"}:
        st.info(
            "THURSDAY NIGHT FOOTBALL SPOTLIGHT — use NFL + Single game to investigate "
            "the exact matchup, with a maximum of four independently qualified legs. "
            "Review player status near kickoff. Two-TD tails, same-player prop "
            "correlation, executable asks/fees and exit liquidity require extra caution. "
            "Never fill missing legs just to raise payout."
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
    if not available_games:
        sport_market_count = sum(len(rows) for rows in parlay_grouped.values())
        date_match_count = 0
        event_key_count = 0
        for rows in parlay_grouped.values():
            for row in rows:
                market = getattr(row, "market", {}) or {}
                try:
                    if _market_local_date(market, ticket_timezone) == ticket_local_date:
                        date_match_count += 1
                        if str(market.get("event_ticker") or "").strip():
                            event_key_count += 1
                except (TypeError, ValueError, KeyError):
                    continue
        if not sport_market_count:
            st.warning(
                f"No classified Kalshi markets were loaded for {sport_filter}. "
                "Refresh the page; if this repeats, the Kalshi catalog feed or sport classifier needs investigation."
            )
        elif not date_match_count:
            st.warning(
                f"Loaded {sport_market_count} markets, but none matched {ticket_local_date.isoformat()} in Hawaiʻi time. "
                "Try another date within the next 7 days or refresh the catalog."
            )
        else:
            st.warning(
                f"Found {date_match_count} markets for this date ({event_key_count} with event IDs), "
                "but could not resolve any parent game titles. Refresh once; if still empty, report this diagnostic."
            )
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
        min_legs = 1
        default_legs = 6 if mode == "longshot" else 5
        max_legs = None
    target = st.number_input(
        "Target legs",
        min_value=min_legs,
        max_value=max_legs,
        value=default_legs,
        step=1,
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

    if mode in {"value", "sleeper"}:
        st.info("Value uses independent probability against executable cost. It has no favorite-probability floor or cheap-price band; missing evidence and unvalidated models cannot fill the ticket.")
    elif mode == "longshot":
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
        with guard_ticket_action(
            st.session_state, st.spinner("Running sport models against current Kalshi markets…"),
            _parlay_build_failure, replace_previous=True,
        ):
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
                include_nfl_touchdowns=(preset in {"NFL Touchdowns", "NFL 2+ Touchdowns"} or use_all_nfl_models),
                max_nfl_td_players=(
                    focused_nfl_cap if preset in {"NFL Touchdowns", "NFL 2+ Touchdowns"}
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

            if preset == "NFL 2+ Touchdowns":
                model_candidates = [r for r in model_candidates if r.market_key == "player_anytime_td" and __import__("re").search(r"(?:Over|Under) (?:1\.5|2\.5|3\.5) ", r.selection)]

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
                "NFL 2+ Touchdowns",
                "NBA Points", "NBA Rebounds", "NBA Assists", "NBA Threes", "NBA PRA",
                "NHL Sleeper Markets",
                "NCAA Upset Intelligence",
                "Soccer Totals / BTTS",
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
            elif preset in {"NFL Touchdowns", "NFL 2+ Touchdowns"}:
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
            elif preset.startswith("NBA "):
                nba_keys = {"NBA Points": "player_points", "NBA Rebounds": "player_rebounds", "NBA Assists": "player_assists", "NBA Threes": "player_threes", "NBA PRA": "player_points_rebounds_assists"}
                model_candidates = [r for r in model_candidates if r.sport == "NBA" and r.market_key == nba_keys.get(preset)]
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
                ticket_date=ticket_local_date,
                ticket_timezone="Pacific/Honolulu",
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
            with guard_ticket_action(
                st.session_state, st.spinner("Cross-checking a bounded set of exact-game sportsbook markets…"),
                lambda error_type: st.warning(
                    f"Sportsbook cross-check failed ({error_type}); the existing model-first ticket is unchanged."
                ),
            ):
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
                    ticket_date=state.get("ticket_date"),
                    ticket_timezone="Pacific/Honolulu",
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
        # Across ALL matchups, not only Thursday NFL. Research board consumes
        # the already scanned independent-model candidates; it never creates
        # Kalshi quotes, future cash-out prices or missing games.
        if sport_filter in {"All", "NFL", "NCAAF", "NBA", "NHL", "MLB"}:
            from sports_edge.models.all_game_joint_research import (
                SPORTS as _FIVE_SPORTS, all_game_research_board,
            )
            _all_game_board = all_game_research_board(
                state.get("model_candidate_rows") or (),
                now=datetime.now(timezone.utc),
                sports=_FIVE_SPORTS if sport_filter == "All" else (sport_filter,),
                max_preview=50,
            )
            with st.expander(
                "All-game parlay research · NFL / NCAAF / NBA / NHL / MLB",
                expanded=True,
            ):
                st.caption(
                    "Every analyzed matchup in the selected candidate pool, "
                    "not just TNF. An unscanned game is NOT claimed covered. "
                    "Only independent, audited and pre-kickoff legs can "
                    "pass their original policy. The combined probability "
                    "is UNKNOWN without validated independent joint scenarios; "
                    "individual Kalshi asks never imply an executable combo ask."
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Scanned sport candidates", _all_game_board["total"])
                m2.metric("Pregame leg-qualified", _all_game_board["qualified_pregame"])
                m3.metric("Research / blocked", _all_game_board["blocked"])
                st.dataframe(
                    [
                        {
                            "Sport": _sp,
                            "Scanned events": _v["games"],
                            "Candidates": _v["candidates"],
                            "Pregame qualified": _v["qualified_pregame"],
                            "Blocked": _v["blocked"],
                        }
                        for _sp, _v in _all_game_board["sports"].items()
                    ],
                    use_container_width=True, hide_index=True,
                )
                if _all_game_board["same_game_risks"]:
                    st.warning(
                        "Same-game legs share game script, usage, and/or athlete "
                        "exposure; multiplying leg percentages is ONLY an "
                        "independence benchmark, not a joint probability."
                    )
                    st.dataframe(
                        _all_game_board["same_game_risks"],
                        use_container_width=True, hide_index=True,
                    )
                if _all_game_board["rows"]:
                    st.dataframe(
                        _all_game_board["rows"],
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.caption(
                        "No five-sport independent candidates found in this scan. "
                        "Select a sport/date and request a scan; missing evidence "
                        "is never replaced with a social pick or a guess."
                    )
                if _all_game_board["common_blockers"]:
                    st.caption(
                        "Top blockers: " + " · ".join(
                            f"{r}: {n}"
                            for r, n in _all_game_board["common_blockers"][:5]
                        )
                    )
                st.caption(
                    "Joint scenario evaluation is RESEARCH ONLY until "
                    "chronologically calibrated, and early cash-out odds "
                    "require recorded exact combo bid paths. No auto orders."
                )
                from sports_edge.joint_score_evidence_panel import render_five_sport_joint_research_reports
                render_five_sport_joint_research_reports()
        if ticket_local_date.weekday() == 3 and sport_filter in {"NFL", "All"}:
            from sports_edge.models.tnf_research import thursday_research_board
            tnf_board = thursday_research_board(
                state.get("model_candidate_rows") or (),
                local_date=ticket_local_date,
                now=datetime.now(timezone.utc),
            )
            with st.expander("TNF spotlight · independently modeled candidates and blockers", expanded=True):
                st.caption(
                    "Thursday kickoff verified in Eastern time. Pre-game model "
                    "probabilities are NOT predicted cash-out odds. Model-ask-fee edge "
                    "is indicative; only fully reviewed candidates clear the original "
                    "Sports Edge leg qualification. No automatic combo orders."
                )
                x, y, z = st.columns(3)
                x.metric("Thursday candidates", tnf_board["total"])
                y.metric("Pregame qualified", tnf_board["qualified"])
                z.metric("Blocked / research", tnf_board["blocked"])
                if tnf_board["kickoff_unverified"]:
                    st.warning(
                        f'{tnf_board["kickoff_unverified"]} candidate(s) have no '
                        'independently verified kickoff; they cannot qualify.'
                    )
                if tnf_board["role_uncertain"]:
                    st.warning(
                        f'{tnf_board["role_uncertain"]} candidate(s) still require '
                        'role/injury/starter evidence. Recheck close to kickoff.'
                    )
                if tnf_board["dependence_pairs"]:
                    st.warning(
                        f'{tnf_board["dependence_pairs"]} same-player cross-stat '
                        'pair(s) depend on the same on-field workload. '
                        'Multiplying individual leg probabilities is NOT '
                        'a calibrated same-game parlay probability.'
                    )
                if tnf_board["research"]:
                    st.dataframe(tnf_board["research"], use_container_width=True, hide_index=True)
                else:
                    st.caption(
                        "No Thursday NFL model candidates for this exact slate. "
                        "Select NFL, the Thursday date, and build the ticket; "
                        "do not substitute Kalshi-only or social picks."
                    )
                if tnf_board["common_blockers"]:
                    st.caption(
                        "Main blockers: " + " · ".join(
                            f'{reason}: {count}' for reason,count in tnf_board["common_blockers"][:5]
                        )
                    )
                st.caption(
                    "Early-exit cash-out probabilities are unvalidated. "
                    "Use the separate combo quote lab for observed public "
                    "bid depth and prospective paper research."
                )
        if result.legs:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Qualified legs", len(result.legs))
            c2.metric("Joint success probability", "Unavailable")
            c3.metric("Actual combined price", "Unavailable")
            c4.metric("Illustrative leg-price payout", f"{result.market_payout_multiplier:.2f}x")
            st.warning("No validated joint-dependence model or executable combined parlay quote is available. The product of individual leg probabilities is only an independence benchmark, not a joint probability; expected value and a model/market value ratio are withheld.")
            st.caption(
                f"Build: {builder_label.upper()} · Ticket math risk: {result.risk_label} · "
                "independence benchmark and payout illustration only"
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
            if st.button("Save this parlay candidate to My Picks", key="track_intelligent_parlay_v1", use_container_width=True):
                try:
                    verified_starts = [parse_timestamp(row.leg.verified_start) for row in result.legs if row.leg.verified_start]
                    if len(verified_starts) != len(result.legs) or not verified_starts or min(verified_starts) <= datetime.now(timezone.utc):
                        raise ValueError("Every leg needs a verified future event start before the parlay can be tracked.")
                    dependence_unknown = (
                        result.correlation_risk != "LOW"
                        or any("independence" in warning.lower() or "joint probability" in warning.lower() or "correlation" in warning.lower() for warning in result.warnings)
                    )
                    leg_payloads = [{
                        "sport": row.leg.sport,
                        "event": row.leg.event_title,
                        "market": row.leg.market_label,
                        "selection": row.leg.selection,
                        "model_probability": row.fair_probability,
                        "source": row.leg.model_name or "SportEdge",
                        "kalshi_price": row.leg.kalshi_price,
                    } for row in result.legs]
                    payload = {
                        "sport": sport_filter,
                        "event": "Multi-event research parlay",
                        "market": "Parlay research candidate",
                        "selection": " + ".join(row.leg.selection for row in result.legs),
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                        "event_start": min(verified_starts).isoformat(),
                        "source": "SportEdge",
                        "record_type": "parlay",
                        "record_origin": "generated",
                        "leg_count": len(result.legs),
                        "probability": None if dependence_unknown else result.fair_joint_probability,
                        "model_version": "intelligent-parlay",
                        "legs": leg_payloads,
                        "illustrative_payout_multiplier": result.market_payout_multiplier,
                        "notes": "Joint probability unavailable for value purposes unless validated dependence evidence exists." if dependence_unknown else "Research estimate only; verify actual sportsbook parlay price separately.",
                    }
                    tracking_store = _pick_store()
                    _save_research_prediction(payload, tracking_store)
                    if tracking_store is None:
                        st.warning("Parlay candidate saved to this session only. Configure Supabase secrets for durable tracking.")
                    else:
                        st.success("Parlay candidate saved before the earliest event start. This is a research record, not a placed wager or sportsbook quote.")
                    st.rerun()
                except Exception as exc:
                    st.warning(f"Parlay was not saved: {_safe_error(exc)}")
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
            # Do not hide the audit reasons behind an empty ticket.
            # Historical/research probabilities remain explicitly unqualified.
            from sports_edge.models.parlay_diagnostics import parlay_evidence_diagnostics
            evidence = parlay_evidence_diagnostics(state.get("model_candidate_rows") or ())
            if evidence["candidates"]:
                st.markdown("### Why no ticket cleared")
                d1, d2, d3 = st.columns(3)
                d1.metric("Independent models", evidence["modeled"])
                d2.metric("Ask + fee checked", evidence["modeled_and_fee_priced"])
                d3.metric("Reviewed value", evidence["qualified"])
                if evidence["common_blockers"]:
                    st.dataframe(pd.DataFrame([
                        {"Evidence blocker":reason,"Candidates":count}
                        for reason,count in evidence["common_blockers"]]),
                        use_container_width=True,hide_index=True)
                if evidence["research_preview"]:
                    st.markdown("### Model-priced watchlist · research only")
                    st.caption("Research probabilities and indicative edges are NOT validated picks, executable combo quotes or guaranteed returns. Rejected legs never enter recommended tickets.")
                    st.dataframe(pd.DataFrame(evidence["research_preview"]),
                                 use_container_width=True,hide_index=True)
            else:
                st.info("No independently modeled candidate rows. Check sport, date, Kalshi contract family, and supported historical data.")
        # Cash-out is its OWN outcome, and requires the exact combo-level bid
        # and liquidity, not multiplication of leg prices. Runs only on demand.
        from sports_edge.combo_cashout_panel import render_combo_cashout_lab
        render_combo_cashout_lab()
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
        # Live reversal inputs have dedicated score/price/survivability gates;
        # the shared pregame ticket auditor is not applicable here and can
        # crash or suppress otherwise valid live radar observations.
        tennis_candidates = live_tennis_radar_candidates(live_grouped)

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
        recent_signals, recent_error = get_tennis_recent_prospective_signals()
        recent_grades, grades_error = get_tennis_recent_prospective_grades()
        timelines = build_tennis_watch_timelines(
            recent_signals,
            now=datetime.now(timezone.utc),
            live_event_ids=set(confirmed_live_ids),
            grades=recent_grades,
            lookback_minutes=180,
            max_rows=30,
        )
        st.markdown("### ⏱ Tennis Reversal Timeline · First Alert → Recovery")
        st.caption(
            "First observed quoted ask and research-stage transitions are shown "
            "together, including 2–20¢ recoveries. Only score-confirmed, "
            "recent positive-edge observations are highlighted as RESEARCH. "
            "Historical quotes are not available entries and are not fills."
        )
        if timelines:
            st.dataframe(pd.DataFrame([
                {
                    "Player": row.selection,
                    "First seen (UTC)": row.first_seen.strftime("%H:%M"),
                    "First ask": f"{row.first_price:.0%}",
                    "Early positive-edge ask": (
                        f"{row.earliest_positive_edge_price:.0%}"
                        if row.earliest_positive_edge_price is not None
                        else "none"
                    ),
                    "Latest stage ask": f"{row.latest_price:.0%}",
                    "Latest estimated fair": f"{row.latest_model_fair:.1%}",
                    "Stages recorded": " → ".join(row.sequence),
                    "Status": row.status,
                    "Latest observed score": row.latest_score,
                }
                for row in timelines
            ]), use_container_width=True, hide_index=True)
        else:
            st.caption(
                "No recent recorded Tennis reversal signals. This does not "
                "guarantee the absence of cheap prices or opportunities."
            )
        if recent_error:
            st.caption(f"Signal ledger delayed/unavailable: {recent_error}")
        if grades_error:
            st.caption(f"Settlement grading status unavailable: {grades_error}")

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
        market_report, market_poll, market_error = get_tennis_market_opportunity_report()
        with st.expander("Tennis alert follow-through · observed exit quotes", expanded=False):
            st.caption(
                "After each market-tracking notification, SportsEdge records later Kalshi "
                "YES bids on subsequent scheduled polls. These are sampled prices only, "
                "NOT executed trades, realized profit or prediction wins."
            )
            if market_report is None:
                st.caption("No readable post-alert quote report yet. " + str(market_error or ""))
            else:
                st.caption(
                    f"ntfy receipts: {market_report.get('notification_count', 0)} · "
                    f"priced receipts: {market_report.get('entry_price_recorded', 0)} · "
                    f"with later bid samples: {market_report.get('with_post_alert_bid_samples', 0)} · "
                    f"unsampled: {market_report.get('without_post_alert_samples', 0)}"
                )
                if isinstance(market_poll, dict):
                    gap = market_poll.get("poll_gap_minutes")
                    if gap is not None:
                        st.caption(
                            f"Last poll gap: {gap} minutes "
                            + ("(delayed monitoring)" if market_poll.get("late_poll_gap") else "")
                        )
                    st.caption(
                        f"Research price movements: {market_poll.get('tracking_observations', 0)} · "
                        f"score-confirmed eligible markets: {market_poll.get('survivability_cleared', 0)} · "
                        f"poor/unverified score moves suppressed: "
                        f"{market_poll.get('suppressed_unverified_or_late', 0)}"
                    )
                    if market_poll.get("live_score_states", 0) == 0:
                        st.warning(
                            "Independent live Tennis scores unavailable for this scan. "
                            "Market-only movements remain research observations; phone "
                            "reversal alerts are suppressed rather than suggesting "
                            "late or losing-straight-set players."
                        )
                    rejected = market_poll.get("suppression_reasons") or {}
                    if rejected:
                        st.caption(
                            "Score-quality rejection counts: "
                            + ", ".join(f"{k}: {v}" for k,v in sorted(rejected.items()))
                        )
                rows = [
                    {
                        "Market": row.get("ticker", ""),
                        "Lane": row.get("lane", ""),
                        "Alert ask": f"{row['alert_ask']*100:.1f}¢",
                        "Best later observed bid": f"{row['best_observed_bid']*100:.1f}¢",
                        "Gross quote improvement": f"{row['max_observed_quote_improvement_pp']:+.1f}pp",
                        "Bid samples": row.get("post_alert_sample_count", 0),
                    }
                    for row in (market_report.get("alerts") or [])
                    if row.get("status") == "sampled"
                    and row.get("alert_ask") is not None
                    and row.get("best_observed_bid") is not None
                ]
                if rows:
                    st.dataframe(pd.DataFrame(rows[-25:]),use_container_width=True,hide_index=True)
                else:
                    st.caption(
                        "No qualifying later bid observations yet. Older alerts without "
                        "a saved entry quote cannot be retroactively valued."
                    )
                st.caption(
                    "A sampled market bid does not guarantee fillable size. Fees, "
                    "cash-out costs and unsampled price peaks are not measured."
                )

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
                    push_on = bool(run_status.get("push_configured", False))
                    accepted = int(cloud_report.get("push_accepted_notifications", 0) or 0)
                    if push_on:
                        st.caption(
                            f"Phone push bridge: configured · {accepted} provider acceptance(s) recorded. "
                            "Device receipt is not guaranteed by provider acceptance."
                        )
                    else:
                        st.caption(
                            "Phone push bridge: waiting for the GitHub Actions topic secret "
                            "and an iPhone ntfy subscription."
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
            st.markdown(
                "[Set up iPhone Tennis push alerts](https://github.com/"
                "jwalton180-jpg/sports-edge-bot/blob/main/docs/tennis_iphone_push.md)"
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

elif view == "Scorer Lab":
    if sport_filter == "NHL":
        _section_hero("NHL SCORER LAB", "Anytime-goal research", "Official NHL schedule, roster and player-history coverage. Only eligible history-backed research estimates appear; no scorer is invented to fill a quota.")
        st.caption("Current lineup, line combination, power-play role, goalie context and executable anytime-goal odds are not yet complete in this baseline. These estimates are not validated value picks.")
    elif sport_filter == "NFL":
        _section_hero("NFL TD SCORER LAB", "Anytime-touchdown research", "Run the existing independent NFL touchdown model against the actual current Kalshi player-prop catalog. Missing prices or verified starts block tracking and value claims.")
        if st.button("Run NFL touchdown scorer analysis", type="primary", use_container_width=True, key="nfl_td_scorer_scan_v1"):
            with st.spinner("Checking current NFL touchdown markets and independent model evidence…"):
                try:
                    td_candidates = model_candidates_from_kalshi(
                        kalshi_grouped,
                        sport_filter="NFL",
                        include_nfl_touchdowns=True,
                        max_nfl_td_players=24,
                        target_local_date=datetime.now(ZoneInfo("Pacific/Honolulu")).date(),
                        ticket_timezone="Pacific/Honolulu",
                    )
                    td_candidates = [row for row in td_candidates if row.market_key == "player_anytime_td"]
                    td_assessments = [assess_leg(row, "best") for row in td_candidates]
                    td_assessments.sort(key=lambda row: (row.qualified, row.score), reverse=True)
                    st.session_state["nfl_td_scorer_results_v1"] = td_assessments
                except Exception as exc:
                    st.warning(f"NFL touchdown analysis unavailable: {_safe_error(exc)}")
                    st.session_state["nfl_td_scorer_results_v1"] = []
        td_assessments = [
            row for row in st.session_state.get("nfl_td_scorer_results_v1", [])
            if _safe_event_start(row.leg.verified_start) is not None
            and _safe_event_start(row.leg.verified_start) > datetime.now(timezone.utc)
        ]
        if not td_assessments:
            st.info("No NFL touchdown-scorer candidates have been generated in this session. Run the analysis; if the actual catalog lacks eligible markets or independent model evidence, the lab will remain empty.")
        else:
            st.dataframe(pd.DataFrame([{
                "Status": "QUALIFIED RESEARCH" if row.qualified else "WATCH / PASS",
                "Event": row.leg.event_title,
                "Selection": row.leg.selection,
                "Model probability": f"{row.fair_probability:.1%}",
                "Sportsbook price": row.leg.median_odds if row.leg.median_odds is not None else "—",
                "Kalshi price": f"{row.kalshi_probability:.1%}" if row.kalshi_probability is not None else "—",
                "Edge": f"{row.edge_points:+.1f} pp" if row.edge_points is not None else "—",
                "Model confidence": f"{row.leg.model_confidence:.0%}",
                "Event start": row.leg.verified_start or "unverified",
                "Caveats": "; ".join(row.warnings) or "See model evidence",
            } for row in td_assessments]), use_container_width=True, hide_index=True)
            savable_td = [(index, row) for index, row in enumerate(td_assessments) if row.qualified and _safe_event_start(row.leg.verified_start) is not None and _safe_event_start(row.leg.verified_start) > datetime.now(timezone.utc)]
            if savable_td:
                selected_td_index = st.selectbox(
                    "NFL touchdown candidate to track",
                    [index for index, _ in savable_td],
                    format_func=lambda index: next(row.leg.selection for i, row in savable_td if i == index),
                    key="track_nfl_td_select_v1",
                )
                if st.button("Save qualified NFL touchdown candidate to My Picks", key="track_nfl_td_button_v1", use_container_width=True):
                    row = td_assessments[selected_td_index]
                    leg = row.leg
                    try:
                        payload = {
                            "sport": "NFL",
                            "event": leg.event_title,
                            "market": "Anytime touchdown scorer",
                            "selection": leg.selection,
                            "observed_at": datetime.now(timezone.utc).isoformat(),
                            "event_start": leg.verified_start,
                            "source": "SportEdge",
                            "record_type": "nfl_td_scorer",
                            "record_origin": "generated",
                            "probability": row.fair_probability,
                            "model_version": leg.model_name or "unknown",
                            "model_confidence": leg.model_confidence,
                            "sample_size": leg.model_sample_size,
                            "event_id": leg.event_id,
                            "kalshi_ticker": leg.kalshi_ticker,
                            "kalshi_price": leg.kalshi_price,
                            "notes": "; ".join(row.warnings) or "Review current availability and exact odds before any value assessment.",
                        }
                        if leg.median_odds is not None:
                            payload["odds_american"] = leg.median_odds
                        tracking_store = _pick_store()
                        _save_research_prediction(payload, tracking_store)
                        if tracking_store is None:
                            st.warning("NFL touchdown candidate saved to this session only. Configure the private tracker and Supabase secrets for durable tracking.")
                        else:
                            st.success("NFL touchdown research candidate saved before kickoff. No wager was placed.")
                        st.rerun()
                    except Exception as exc:
                        st.warning(f"NFL touchdown candidate was not saved: {_safe_error(exc)}")
            else:
                st.caption("No qualified touchdown candidate with a verified future start is available to save.")
    else:
        _section_hero("SCORER LAB", "Choose NHL or NFL", "NHL anytime-goal and NFL anytime-touchdown research use separate source pipelines and separate records.")
        st.info("Select NHL or NFL in the sport selector above to load the corresponding scorer research pipeline.")

elif view == "My Picks":
    _section_hero("RESEARCH TRACKER", "My Picks", "Save a timestamped research candidate before the event starts, then keep the original price and attribution when results are recorded.")
    if not _ensure_tracker_access():
        st.stop()
    store = _pick_store()
    store_error = None
    uploaded_picks = st.file_uploader("Import provider or manual research picks (CSV)", type=["csv"], key="research_pick_csv_upload_v1")
    if uploaded_picks is not None:
        st.caption("Import requires original observed_at and event_start timestamps. Rows observed at or after event start are rejected; import time is never substituted for a missing original timestamp.")
        if len(uploaded_picks.getvalue()) > 2_000_000:
            st.warning("CSV import is limited to 2 MB per upload.")
        else:
            try:
                import_frame = pd.read_csv(io.BytesIO(uploaded_picks.getvalue()))
                required_columns = {"sport", "event", "market", "selection", "observed_at", "event_start", "source", "record_type"}
                missing_columns = sorted(required_columns - set(import_frame.columns))
                if missing_columns:
                    st.warning("CSV missing required columns: " + ", ".join(missing_columns))
                else:
                    st.dataframe(import_frame.head(8), use_container_width=True, hide_index=True)
                    if st.button("Import valid CSV rows", key="research_pick_csv_import_button"):
                        imported = 0
                        rejected = []
                        optional_columns = {"probability", "odds_american", "model_version", "leg_count", "notes", "model_confidence", "sample_size"}
                        for row_index, raw_row in import_frame.iterrows():
                            try:
                                payload = {}
                                for key in required_columns | optional_columns:
                                    value = raw_row.get(key)
                                    if value is None or pd.isna(value):
                                        continue
                                    payload[key] = value.item() if hasattr(value, "item") else value
                                if "probability" in payload:
                                    payload["probability"] = float(payload["probability"])
                                if "odds_american" in payload:
                                    payload["odds_american"] = int(float(payload["odds_american"]))
                                    if payload["odds_american"] == 0:
                                        raise ValueError("American odds cannot be zero")
                                if payload.get("record_type") == "parlay" and "leg_count" in payload:
                                    payload["leg_count"] = int(float(payload["leg_count"]))
                                now_utc = datetime.now(timezone.utc)
                                if parse_timestamp(payload["observed_at"]) > now_utc:
                                    raise ValueError("observed_at cannot be in the future")
                                if parse_timestamp(payload["event_start"]) <= now_utc:
                                    raise ValueError("past-event imports are rejected to prevent backfilling")
                                payload["record_origin"] = "imported"
                                payload["provenance_status"] = "imported_unverified"
                                _save_research_prediction(payload, store)
                                imported += 1
                            except Exception as exc:
                                rejected.append(f"row {row_index + 2}: {_safe_error(exc)}")
                        st.success(f"Imported {imported} causal record(s).")
                        if rejected:
                            st.warning(f"Rejected {len(rejected)} row(s): " + " · ".join(rejected[:5]))
            except Exception as exc:
                st.warning(f"CSV could not be read: {_safe_error(exc)}")

    try:
        records = _pick_records(store)
    except Exception as exc:
        store_error = type(exc).__name__
        records = list(_session_pick_records())
        store = None
    if store is None:
        st.warning("SESSION-ONLY STORAGE: picks may disappear when this Streamlit session ends or the app redeploys. Configure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Streamlit secrets for durable storage. See docs/pick_storage.md.")
    if store_error:
        st.error(f"Configured storage could not be read ({store_error}). New entries in this view are session-only until storage is healthy.")

    with st.form("research_pick_capture_v1", clear_on_submit=True):
        st.subheader("Save a research candidate")
        record_kind = st.selectbox("Candidate type", ["Individual prop", "Parlay candidate", "NHL goal scorer", "NFL touchdown scorer", "Live / reversal research"])
        sport_value = st.selectbox("Sport", ["NFL", "NCAAF", "NBA", "NHL", "MLB", "Tennis", "WNBA", "Soccer", "Other"])
        event_value = st.text_input("Game / event")
        market_value = st.text_input("Market", placeholder="Player points, anytime TD, anytime goal…")
        selection_value = st.text_input("Selection", placeholder="Player X over 24.5 points")
        event_start_value = st.text_input("Event start (ISO-8601 with timezone)", placeholder="2026-10-22T23:00:00Z")
        source_value = st.selectbox("Source / provider attribution", ["SportEdge", "Manual research", "PropsBot.AI", "Betstamp", "Juice Reel", "Outlier", "The Odds API", "Other provider"])
        probability_known = st.checkbox("Model probability is available", value=True)
        probability_value = st.number_input("Model probability (%)", min_value=0.0, max_value=100.0, value=55.0, step=0.5, disabled=not probability_known)
        odds_value = st.text_input("Original American odds (optional)", placeholder="-110 or +135")
        leg_count_value = st.number_input("Parlay leg count (for parlay candidates)", min_value=1, max_value=12, value=2, step=1, disabled=record_kind != "Parlay candidate")
        model_version_value = st.text_input("Model / version (optional)")
        notes_value = st.text_input("Notes / evidence caveat (optional)")
        save_submitted = st.form_submit_button("Save research candidate", use_container_width=True)

    if save_submitted:
        observed_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "sport": sport_value,
            "event": event_value.strip(),
            "market": market_value.strip(),
            "selection": selection_value.strip(),
            "observed_at": observed_at,
            "event_start": event_start_value.strip(),
            "source": source_value,
            "record_type": {
                "Individual prop": "prop",
                "Parlay candidate": "parlay",
                "NHL goal scorer": "nhl_goal_scorer",
                "NFL touchdown scorer": "nfl_td_scorer",
                "Live / reversal research": "live_reversal",
            }[record_kind],
            "model_version": model_version_value.strip() or None,
            "notes": notes_value.strip() or None,
            "price_type": "american_odds" if odds_value.strip() else "unpriced",
        }
        if probability_known:
            payload["probability"] = float(probability_value) / 100.0
        if odds_value.strip():
            try:
                payload["odds_american"] = int(odds_value.strip().replace(" ", ""))
                if payload["odds_american"] == 0:
                    raise ValueError("American odds cannot be zero")
            except ValueError:
                st.error("Original odds must be a non-zero American price such as -110 or +135.")
                payload = None
        if payload is not None:
            try:
                if payload["record_type"] == "parlay":
                    payload["leg_count"] = int(leg_count_value)
                _save_research_prediction(payload, store)
                st.success("Research candidate saved with its original timestamp. This does not mean a wager was placed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Candidate was not saved: {_safe_error(exc)}")

    try:
        records = _pick_records(store) if store is not None else list(_session_pick_records())
    except Exception as exc:
        st.error(f"Could not refresh saved picks ({type(exc).__name__}).")
        records = list(_session_pick_records())
        store = None
    predictions = [r for r in records if r.get("record_type") == "prediction"]
    settlements = _settlement_by_prediction(records)
    if predictions:
        display_rows = []
        for record in predictions:
            payload = _record_payload(record)
            settlement = settlements.get(record.get("id"), {})
            display_rows.append({
                "Saved": payload.get("saved_at") or record.get("created_at"),
                "Sport": payload.get("sport"),
                "Event": payload.get("event"),
                "Type": payload.get("record_type"),
                "Market": payload.get("market"),
                "Selection": payload.get("selection"),
                "Model probability": payload.get("probability"),
                "Original American odds": payload.get("odds_american"),
                "Source": payload.get("source"),
                "Origin": payload.get("record_origin", "manual"),
                "Provenance": payload.get("provenance_status", "internal_record"),
                "Status": settlement.get("result", "pending").upper(),
                "Model": payload.get("model_version"),
            })
        st.subheader("Saved candidates")
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
        st.download_button("Export My Picks CSV", pd.DataFrame(display_rows).to_csv(index=False).encode("utf-8"), "sportedge_my_picks.csv", "text/csv", use_container_width=True)

    settled_predictions = [
        r for r in predictions
        if r.get("id") not in settlements
        and (lambda p: _safe_event_start(p.get("event_start")) is not None and _safe_event_start(p.get("event_start")) <= datetime.now(timezone.utc))(_record_payload(r))
    ]
    if settled_predictions:
        st.subheader("Record a result")
        st.caption("Manual settlement is not independently verified. Use an official final-results URL and keep the original prediction unchanged.")
        with st.form("research_pick_settlement_v1"):
            chosen_id = st.selectbox("Prediction", [r.get("id") for r in settled_predictions], format_func=lambda pid: next((f"{_record_payload(r).get('sport')} · {_record_payload(r).get('selection')} · {pid[:8]}" for r in settled_predictions if r.get("id") == pid), pid))
            result_value = st.selectbox("Result", ["win", "loss", "push", "void", "cancelled"])
            settlement_url = st.text_input("Official result source URL (HTTPS required)")
            settle_submitted = st.form_submit_button("Append settlement record", use_container_width=True)
        if settle_submitted:
            try:
                if not settlement_url.strip().startswith("https://"):
                    raise ValueError("Use an HTTPS source URL for the result.")
                settled_at = datetime.now(timezone.utc).isoformat()
                if store is not None:
                    store.append_settlement(prediction_id=chosen_id, result=result_value, settled_at=settled_at, source_url=settlement_url.strip(), verification="manual")
                else:
                    original = next(r for r in settled_predictions if r.get("id") == chosen_id)
                    if settled_at < parse_timestamp(_record_payload(original)["event_start"]).isoformat():
                        raise ValueError("Settlement cannot predate event start.")
                    _session_pick_records().append({
                        "id": str(uuid.uuid4()),
                        "record_type": "settlement",
                        "payload": {"prediction_id": chosen_id, "result": result_value, "settled_at": settled_at, "source_url": settlement_url.strip(), "verification": "manual"},
                        "created_at": settled_at,
                    })
                st.success("Settlement appended. The prediction record was not changed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Settlement not recorded: {_safe_error(exc)}")
    else:
        st.caption("No unsettled past-event picks are available to grade yet. Pending future candidates remain untouched.")

elif view == "Performance":
    _section_hero("TRACK RECORD", "Performance", "A transparent record of tracked research candidates. Empty history is shown as empty; provider records and SportEdge records remain attributable.")
    if not _ensure_tracker_access():
        st.stop()
    store = _pick_store()
    try:
        records = _pick_records(store) if store is not None else list(_session_pick_records())
    except Exception as exc:
        st.error(f"Persistent tracking is unavailable ({type(exc).__name__}); session-only records may be incomplete.")
        records = list(_session_pick_records())
    predictions = [r for r in records if r.get("record_type") == "prediction"]
    settlements = _settlement_by_prediction(records)
    if not predictions:
        st.info("No tracked predictions yet. Save candidates in My Picks before the event starts. No hit rate or ROI is available.")
    else:
        perf_rows = []
        for record in predictions:
            payload = _record_payload(record)
            settlement = settlements.get(record.get("id"), {})
            result_value = str(settlement.get("result") or "pending").lower()
            if result_value == "cancelled":
                result_value = "void"
            perf_rows.append({
                "record_type": payload.get("record_type", "prop"),
                "source": payload.get("source", "unknown"),
                "record_origin": payload.get("record_origin", "manual"),
                "sport": payload.get("sport", "unknown"),
                "market": payload.get("market", "unknown"),
                "model_version": payload.get("model_version", "unknown"),
                "observed_at": payload.get("observed_at"),
                "leg_count": payload.get("leg_count"),
                "result": result_value,
                "odds_american": payload.get("odds_american"),
                "probability": payload.get("probability"),
                "outcome": 1 if result_value == "win" else 0 if result_value == "loss" else None,
                "stake_units": payload.get("stake_units", 1.0),
            })
        performance_groups = [
            ("All tracked research", perf_rows),
            ("SportEdge-generated", [r for r in perf_rows if str(r["source"]).lower() == "sportedge" and r["record_origin"] == "generated"]),
        ]
        attribution_pairs = sorted({
            (str(r["source"]), str(r["record_origin"]))
            for r in perf_rows
            if not (str(r["source"]).lower() == "sportedge" and r["record_origin"] == "generated")
        })
        for source_name, origin_name in attribution_pairs:
            performance_groups.append((
                f"{origin_name.title()} · {source_name}",
                [r for r in perf_rows if str(r["source"]) == source_name and str(r["record_origin"]) == origin_name],
            ))
        for group_name, group_rows in performance_groups:
            st.subheader(group_name)
            report = summarize_performance(group_rows)
            a, b, d, e = st.columns(4)
            a.metric("Win rate", "—" if report["hit_rate"] is None else f'{report["hit_rate"]:.1%}', f'{report["wins"]}W / {report["losses"]}L · n={report["hit_rate_denominator"]}')
            b.metric("ROI", "—" if report["roi"] is None else f'{report["roi"]:+.1%}', "Requires complete original odds")
            d.metric("Net units", "—" if report["net_units"] is None else f'{report["net_units"]:+.2f}', f'{report["risked_units"]:.2f} units priced')
            e.metric("Brier score", "—" if report["brier_score"] is None else f'{report["brier_score"]:.3f}', f'n={report["brier_denominator"]}')
            avg_odds_text = "—" if report["average_decimal_odds"] is None else f'{report["average_decimal_odds"]:.3f}'
            avg_stake_text = "—" if report["average_stake_units"] is None else f'{report["average_stake_units"]:.2f}'
            st.caption(f'Pending {report["pending"]} · Pushes {report["pushes"]} · Voids {report["voids"]} · Cancelled {report["cancelled"]} · Settled results with missing odds {report["missing_odds_resolved"]}. Average decimal odds: {avg_odds_text} · Average stake: {avg_stake_text} units. ROI remains unavailable if any resolved win/loss is missing its original price.')
        st.subheader("Breakdown by sport, market, source and model")
        breakdown_rows = []
        breakdown_keys = sorted({(r["sport"], r["market"], r["source"], r["model_version"]) for r in perf_rows})
        for sport_name, market_name, source_name, model_name in breakdown_keys:
            subset = [r for r in perf_rows if (r["sport"], r["market"], r["source"], r["model_version"]) == (sport_name, market_name, source_name, model_name)]
            report = summarize_performance(subset)
            breakdown_rows.append({
                "Sport": sport_name, "Market": market_name, "Source": source_name, "Model": model_name,
                "Wins": report["wins"], "Losses": report["losses"], "Pushes": report["pushes"],
                "Voids": report["voids"], "Cancelled": report["cancelled"], "Pending": report["pending"], "n": report["total_records"],
                "Hit rate": report["hit_rate"], "ROI": report["roi"], "Net units": report["net_units"],
                "Brier": report["brier_score"],
            })
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)
        resolved_rows = []
        for record in predictions:
            payload = _record_payload(record)
            settlement = settlements.get(record.get("id"), {})
            result_value = str(settlement.get("result") or "pending").lower()
            if result_value in {"win", "loss"}:
                resolved_rows.append((payload, result_value, settlement.get("settled_at")))
        has_missing_price = any(payload.get("odds_american") is None for payload, _, _ in resolved_rows)
        if resolved_rows and not has_missing_price:
            equity_rows = []
            running_units = 0.0
            for payload, result_value, settled_at in sorted(resolved_rows, key=lambda row: str(row[2] or "")):
                stake = float(payload.get("stake_units", 1.0))
                decimal_price = american_to_decimal(payload["odds_american"])
                running_units += net_units(result_value, decimal_price, stake)
                equity_rows.append({"Settlement": str(settled_at or payload.get("event_start")), "Cumulative net units": running_units})
            st.subheader("Cumulative net units")
            st.line_chart(pd.DataFrame(equity_rows).set_index("Settlement"))
        elif resolved_rows:
            st.caption("Cumulative-profit chart withheld because at least one settled win/loss is missing its original odds.")
        observed_values = [str(r["observed_at"]) for r in perf_rows if r.get("observed_at")]
        if observed_values:
            st.caption(f"Observation date range: {min(observed_values)} to {max(observed_values)} · probability coverage: {sum(r.get('probability') is not None for r in perf_rows)}/{len(perf_rows)} records.")
        parlay_rows = [r for r in perf_rows if r["record_type"] == "parlay" and type(r.get("leg_count")) is int]
        if parlay_rows:
            st.subheader("Whole-parlay results by leg count")
            parlay_report = summarize_parlays(parlay_rows)
            st.dataframe(pd.DataFrame([{"Legs": legs, "Wins": row["wins"], "Losses": row["losses"], "Pushes": row["pushes"], "Voids": row["voids"], "Pending": row["pending"], "Hit rate": row["hit_rate"], "ROI": row["roi"], "Net units": row["net_units"], "n": row["total_records"]} for legs, row in parlay_report.items()]), use_container_width=True, hide_index=True)
        st.caption("Only candidates captured before event start are eligible. Manually entered settlements are marked manual and are not represented as independently verified. Whole parlays are graded separately from individual legs.")

elif view == "Data Sources":
    _section_hero("INTEGRATIONS", "Data sources", "Connected, configured, partial and research-only sources are kept distinct. A website or API document does not mean credentials are available.")
    source_rows = get_source_status()
    st.dataframe(pd.DataFrame([{"Source": row["name"], "Role": row["role"], "Access": row["access"], "Status": row["status"].replace("_", " ").title(), "Notes": row["detail"]} for row in source_rows]), use_container_width=True, hide_index=True)
    st.markdown("Manual research links: [PropsBot free prop preview](https://propsbot.ai/free-ai-player-prop-generator/) · [PropsBot public ledger / methodology](https://propsbot.ai/about/) · [Outlier free NFL research](https://www.outlier.bet/nfl-free-bets/)")
    st.markdown("Official API documentation: [Betstamp API](https://www.betstamp.com/docs) · [Juice Reel API](https://www.juicereel.com/api-docs)")
    st.caption("PropsBot's High ROI and High Hit Rate signals are separate and provider-reported; their published metrics are not SportEdge results or parlay ROI. PropsBot and Outlier are manual research links only until authorized API/export access is verified. Betstamp requires a provisioned key and plan entitlements. Juice Reel requires user consent and a secure server-side OAuth implementation. No provider is marked connected without actual configuration.")

with st.expander("Data source coverage / integrations", expanded=False):
    source_rows = get_source_status()
    st.caption("Integrated means this repository has a client. Research-only vendors are not scraped or claimed as connected without verified authorized API access.")
    st.dataframe(
        pd.DataFrame([
            {
                "Source": row["name"],
                "Role": row["role"],
                "Access": row["access"],
                "Status": row["status"].replace("_", " ").title(),
                "Notes": row["detail"],
            }
            for row in source_rows
        ]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("For The Odds API, configure THE_ODDS_API_KEY in Streamlit secrets to enable the existing adapter. Source presence does not imply a model edge or validated prediction.")

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
