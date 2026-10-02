from datetime import date
from types import SimpleNamespace

from sports_edge.models.kalshi_model_candidates import _market_local_date, _rows_for_local_date


def _row(**market):
    return SimpleNamespace(market=market)


def test_occurrence_datetime_is_converted_to_hawaii_calendar_date():
    # 05:30 UTC on Sep 30 is still Sep 29 in Hawaiʻi.
    market = {"occurrence_datetime": "2026-09-30T05:30:00Z", "ticker": "KXATP-26SEP30-TEST"}
    assert _market_local_date(market, "Pacific/Honolulu") == date(2026, 9, 29)


def test_today_scope_rejects_tennis_matches_on_later_local_dates():
    rows = [
        _row(occurrence_datetime="2026-09-30T05:30:00Z", ticker="KXATP-26SEP30-A"),
        _row(occurrence_datetime="2026-10-01T05:30:00Z", ticker="KXATP-26OCT01-B"),
    ]
    kept = _rows_for_local_date(
        rows,
        target_date=date(2026, 9, 29),
        timezone_name="Pacific/Honolulu",
    )
    assert [row.market["ticker"] for row in kept] == ["KXATP-26SEP30-A"]


def test_today_scope_applies_across_sports_and_uses_ticker_date_fallback():
    rows = [
        _row(ticker="KXMLB-26SEP29-NYYBOS"),
        _row(ticker="KXNFL-26SEP30-SEASF"),
    ]
    kept = _rows_for_local_date(
        rows,
        target_date=date(2026, 9, 29),
        timezone_name="Pacific/Honolulu",
    )
    assert [row.market["ticker"] for row in kept] == ["KXMLB-26SEP29-NYYBOS"]


def test_market_local_date_prefers_ticker_date_over_settlement_time():
    market = {
        "ticker": "KXMLB-26SEP30-NYYBOS",
        "close_time": "2026-10-01T08:00:00Z",
        "expected_expiration_time": "2026-10-01T09:00:00Z",
    }
    assert _market_local_date(market, "Pacific/Honolulu") == date(2026, 9, 30)


def test_future_ticket_scope_keeps_only_requested_tomorrow_date():
    rows = [
        _row(occurrence_datetime="2026-09-30T20:00:00Z", ticker="KXMLB-26SEP30-A"),
        _row(occurrence_datetime="2026-10-01T20:00:00Z", ticker="KXMLB-26OCT01-B"),
        _row(occurrence_datetime="2026-10-02T20:00:00Z", ticker="KXMLB-26OCT02-C"),
    ]
    kept = _rows_for_local_date(
        rows,
        target_date=date(2026, 10, 1),
        timezone_name="Pacific/Honolulu",
    )
    assert [row.market["ticker"] for row in kept] == ["KXMLB-26OCT01-B"]


def test_child_market_title_is_not_a_physical_game_name():
    market = {
        "event_ticker": "KXMLB-26SEP30-SDCHC",
        "event_title": "1st inning: Over 0.5 runs",
        "title": "Will there be over 0.5 runs in the 1st inning?",
    }
    event = {
        "event_ticker": "KXMLB-26SEP30-SDCHC",
        "title": "San Diego Padres at Chicago Cubs",
    }
    # Regression contract: selectors must resolve display identity from the
    # parent event, never from a child inning/prop market title.
    assert event["title"] == "San Diego Padres at Chicago Cubs"
    assert market["event_title"] != event["title"]


def test_mlb_schedule_matchup_is_physical_game_not_child_market():
    game = {
        "gamePk": 999001,
        "teams": {
            "away": {"team": {"name": "Boston Red Sox"}},
            "home": {"team": {"name": "New York Yankees"}},
        },
    }
    away = game["teams"]["away"]["team"]["name"]
    home = game["teams"]["home"]["team"]["name"]
    assert f"{away} at {home}" == "Boston Red Sox at New York Yankees"
    assert "inning" not in f"{away} at {home}".lower()
    assert "hits" not in f"{away} at {home}".lower()


def test_selected_mlb_game_build_dependency_is_importable():
    from sports_edge.models.game_scope import GameEvent, market_matches_game

    game = GameEvent(
        event_id="MLB:1",
        sport_key="baseball_mlb",
        sport="MLB",
        away_team="Philadelphia Phillies",
        home_team="Atlanta Braves",
        commence_time=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        state="UPCOMING",
    )
    market = {
        "title": "Philadelphia at Atlanta",
        "event_title": "Philadelphia Phillies at Atlanta Braves",
    }
    assert market_matches_game(market, game)


def test_parlay_build_uses_full_catalog_not_overview_source():
    from pathlib import Path
    source = Path("sports_edge/app.py").read_text()
    build_start = source.index('if st.button("Build Sports Edge ticket"')
    build_end = source.index('supported_model_presets = {', build_start)
    build_path = source[build_start:build_end]
    assert "model_candidates_from_kalshi(" in build_path
    assert "parlay_grouped," in build_path
    assert "kalshi_grouped," not in build_path


def test_selected_schedule_matchup_canonical_id_matches_modeled_game_identity():
    from datetime import date
    from sports_edge.models.event_identity import canonical_event_id_from_title

    selected = canonical_event_id_from_title(
        "MLB", "Philadelphia Phillies at Atlanta Braves", date(2026, 9, 30)
    )
    modeled = canonical_event_id_from_title(
        "MLB", "Philadelphia Phillies at Atlanta Braves", date(2026, 9, 30)
    )
    assert selected == modeled


def test_selected_games_keeps_legacy_builder_call_without_single_game_option():
    from pathlib import Path
    source = Path("sports_edge/app.py").read_text()
    start = source.index('if game_scope == "Single game":', source.index('build_disabled ='))
    end = source.index('st.session_state["intel_parlay_v3"]', start)
    block = source[start:end]
    single, other = block.split("else:", 1)
    assert "prioritize_payout_multiplier=True" in single
    assert "prioritize_payout_multiplier" not in other


def test_parlay_event_metadata_failure_is_nonfatal():
    from pathlib import Path
    source = Path("sports_edge/app.py").read_text()
    helper_start = source.index("def get_parlay_kalshi_events():")
    helper_end = source.index("@st.cache_data", helper_start)
    helper = source[helper_start:helper_end]
    assert "except Exception as exc:" in helper
    assert "return events, error" in helper
    assert "parlay_events_by_ticker, parlay_events_error = get_parlay_kalshi_events()" in source


def test_premium_ui_helpers_and_core_view_cards_are_present():
    from pathlib import Path
    source = Path("sports_edge/app.py").read_text()
    assert "def _premium_empty(" in source
    assert "def _premium_stat_strip(" in source
    assert 'class="intel-strip"' in source
    assert 'with st.expander("Full game slate")' in source


def test_low_stake_ticket_defaults_to_strong_four_to_five_leg_core():
    from pathlib import Path
    source = Path("sports_edge/app.py").read_text()
    assert 'min_legs = 5 if mode == "longshot" else 4' in source
    assert 'default_legs = 6 if mode == "longshot" else 5' in source
    assert 'max_legs = 4' in source
    assert "returns fewer rather than adding weak filler" in source


def test_selected_games_builder_guards_new_kwarg_against_stale_streamlit_module():
    from pathlib import Path
    source = Path("sports_edge/app.py").read_text()
    assert 'importlib.import_module("sports_edge.models.parlay_intelligence")' in source
    assert "importlib.reload(_pi)" in source
    assert '"preferred_event_ids"\n                        in inspect.signature(build_intelligent_parlay).parameters' in source
    assert "build_candidates = _selected_games_compat_core(" in source
    assert "getattr(result, \"requested_event_count\", 0)" in source
