from datetime import datetime, timezone

import pytest

from sports_edge.data.tennis365 import Tennis365PlayerContext
from sports_edge.data.tennisexplorer import (
    TennisExplorerPairContext,
    TennisExplorerPlayerContext,
)
from sports_edge.data.tennis_live import TennisLiveScoreState
from sports_edge.models.event_identity import canonical_event_id, canonical_participant
from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models import tennis_lower_tour
from sports_edge.models.tennis_lower_tour import (
    LOWER_TOUR_MODEL,
    build_lower_tour_live_fallback_candidates,
    lower_tour_prior,
)


NOW = datetime(2026, 10, 6, 21, 0, tzinfo=timezone.utc)


def _context(player, *, rank, points, wins, losses):
    return Tennis365PlayerContext(
        player=player,
        rank=rank,
        ranking_points=points,
        recent_wins=wins,
        recent_losses=losses,
    )


def _explorer_player(player, *, rank, wins, losses, h2h_wins=0, h2h_losses=0):
    return TennisExplorerPlayerContext(
        player=player,
        profile_url="https://www.tennisexplorer.com/player/test/",
        rank=rank,
        highest_rank=rank,
        year_wins=max(wins, 0) + 20,
        year_losses=max(losses, 0) + 10,
        recent_wins=wins,
        recent_losses=losses,
        clay_wins=20,
        clay_losses=10,
        hard_wins=5,
        hard_losses=4,
        indoor_wins=0,
        indoor_losses=0,
        grass_wins=0,
        grass_losses=0,
        h2h_wins=h2h_wins,
        h2h_losses=h2h_losses,
    )


def _explorer_pair(
    a="Guido Ivan Justo",
    b="Francisco Comesana",
    *,
    a_rank=188,
    b_rank=54,
    a_wins=7,
    a_losses=3,
    b_wins=6,
    b_losses=4,
    a_h2h_wins=1,
    a_h2h_losses=0,
):
    return TennisExplorerPairContext(
        _explorer_player(
            a,
            rank=a_rank,
            wins=a_wins,
            losses=a_losses,
            h2h_wins=a_h2h_wins,
            h2h_losses=a_h2h_losses,
        ),
        _explorer_player(
            b,
            rank=b_rank,
            wins=b_wins,
            losses=b_losses,
            h2h_wins=a_h2h_losses,
            h2h_losses=a_h2h_wins,
        ),
    )


@pytest.fixture(autouse=True)
def _disable_tennisexplorer_network(monkeypatch):
    monkeypatch.setattr(
        tennis_lower_tour,
        "fetch_tennisexplorer_pair_context",
        lambda *args, **kwargs: None,
    )


def _state(
    a="Guido Ivan Justo",
    b="Francisco Comesana",
    *,
    tour="CHALLENGER",
    source_url="https://livescore.tennis365.com/match/justo-comesana",
):
    event_id = canonical_event_id("Tennis", a, b, "2026-10-06")
    return TennisLiveScoreState(
        event_id=event_id,
        selection_key=canonical_participant("Tennis", a),
        player=a,
        opponent=b,
        tour=tour,
        period=3,
        player_sets=1,
        opponent_sets=1,
        player_games=4,
        opponent_games=1,
        lost_first_set=True,
        won_latest_completed_set=True,
        turnaround=True,
        deciding_set=True,
        current_set_lead=3,
        score_label=f"{a} vs {b} · 6-7 · 7-6 · 4-1",
        fetched_at=NOW,
        best_of=3,
        sets_to_win=2,
        serving=False,
        net_break_advantage=2,
        point_score="0-30",
        score_sources=("Tennis365",),
        source_url=source_url,
    )


def _markets():
    event = "KXATPCHALLENGERMATCH-26OCT06JUSCOM"
    return [
        {
            "series_ticker": "KXATPCHALLENGERMATCH",
            "event_ticker": event,
            "ticker": event + "-JUS",
            "yes_sub_title": "Guido Ivan Justo",
            "yes_ask_dollars": "0.14",
        },
        {
            "series_ticker": "KXATPCHALLENGERMATCH",
            "event_ticker": event,
            "ticker": event + "-COM",
            "yes_sub_title": "Francisco Comesana",
            "yes_ask_dollars": "0.86",
        },
    ]


def _existing_leg(state):
    return ParlayCandidateLeg(
        sport="Tennis",
        event_id=state.event_id,
        event_title=f"{state.player} vs {state.opponent}",
        market_key="model_h2h",
        market_label="Match Winner",
        selection=state.player,
        consensus_probability=0.30,
        book_count=0,
        source_age_s=0.0,
        median_odds=None,
        kalshi_ticker="KX-PRIMARY",
        kalshi_side="YES",
        kalshi_price=0.14,
        kalshi_edge_points=16.0,
        kalshi_status="MODEL",
        evidence_class="MODEL",
        model_probability=0.30,
        model_confidence=0.62,
        model_name="Tennis primary",
        model_sample_size=20,
        model_reasons=("primary",),
        model_warnings=(),
    )


def test_lower_tour_prior_is_conservative_and_independent_of_market_price():
    a = _context("Guido Ivan Justo", rank=310, points=190, wins=6, losses=4)
    b = _context("Francisco Comesana", rank=125, points=480, wins=7, losses=3)
    prior = lower_tour_prior(a, b)
    assert prior is not None
    assert 0.20 < prior.probability_a < 0.50
    assert prior.confidence <= 0.51
    assert prior.sample_size == 10
    assert any("Ranking points" in reason for reason in prior.reasons)
    assert any("excludes the current live match" in warning for warning in prior.warnings)


def test_lower_tour_prior_fails_closed_when_unranked_history_is_sparse():
    a = _context("Player A", rank=None, points=None, wins=2, losses=2)
    b = _context("Player B", rank=None, points=None, wins=3, losses=1)
    assert lower_tour_prior(a, b) is None


def test_fallback_candidates_fill_live_model_hole_only(monkeypatch):
    state = _state()
    contexts = (
        _context("Guido Ivan Justo", rank=310, points=190, wins=6, losses=4),
        _context("Francisco Comesana", rank=125, points=480, wins=7, losses=3),
    )
    monkeypatch.setattr(
        tennis_lower_tour,
        "fetch_tennis365_player_context",
        lambda *args, **kwargs: contexts,
    )
    rows = build_lower_tour_live_fallback_candidates(
        _markets(),
        [state],
        [],
        max_workers=1,
    )
    assert len(rows) == 2
    assert {row.selection for row in rows} == {"Guido Ivan Justo", "Francisco Comesana"}
    assert all(row.model_name == LOWER_TOUR_MODEL for row in rows)
    assert all(row.model_sample_size == 10 for row in rows)
    assert all(row.model_confidence <= 0.51 for row in rows)
    assert all(row.event_id == state.event_id for row in rows)


def test_fallback_does_not_override_primary_model_pair(monkeypatch):
    state = _state()
    monkeypatch.setattr(
        tennis_lower_tour,
        "fetch_tennis365_player_context",
        lambda *args, **kwargs: (
            _context(state.player, rank=310, points=190, wins=6, losses=4),
            _context(state.opponent, rank=125, points=480, wins=7, losses=3),
        ),
    )
    rows = build_lower_tour_live_fallback_candidates(
        _markets(),
        [state],
        [_existing_leg(state)],
        max_workers=1,
    )
    assert rows == []


def test_fallback_can_fill_main_tour_live_model_hole_after_cross_feed_match(monkeypatch):
    state = _state(
        "Iga Swiatek",
        "Iva Jovic",
        tour="WTA",
        source_url="https://livescore.tennis365.com/match/swiatek-jovic",
    )
    event = "KXWTAMATCH-26OCT07SWIJOV"
    markets = [
        {
            "series_ticker": "KXWTAMATCH",
            "event_ticker": event,
            "ticker": event + "-SWI",
            "yes_sub_title": "Iga Swiatek",
            "yes_ask_dollars": "0.72",
        },
        {
            "series_ticker": "KXWTAMATCH",
            "event_ticker": event,
            "ticker": event + "-JOV",
            "yes_sub_title": "Iva Jovic",
            "yes_ask_dollars": "0.29",
        },
    ]
    monkeypatch.setattr(
        tennis_lower_tour,
        "fetch_tennis365_player_context",
        lambda *args, **kwargs: (
            _context("Iga Swiatek", rank=3, points=6409, wins=8, losses=2),
            _context("Iva Jovic", rank=16, points=2436, wins=7, losses=3),
        ),
    )
    rows = build_lower_tour_live_fallback_candidates(
        markets,
        [state],
        [],
        max_workers=1,
    )
    assert len(rows) == 2
    assert {row.selection for row in rows} == {"Iga Swiatek", "Iva Jovic"}
    assert all(row.model_name == LOWER_TOUR_MODEL for row in rows)
    assert all(row.model_sample_size == 10 for row in rows)
    assert all("live Tennis fallback" in " ".join(row.model_warnings) for row in rows)


def test_main_tour_fallback_can_use_verified_tennisexplorer_prior_without_tennis365_url(monkeypatch):
    state = _state(
        "Iga Swiatek",
        "Iva Jovic",
        tour="WTA",
        source_url=None,
    )
    state = TennisLiveScoreState(
        **{
            **state.__dict__,
            "score_sources": ("ESPN",),
        }
    )
    called = False

    def should_not_call(*args, **kwargs):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(
        tennis_lower_tour,
        "fetch_tennis365_player_context",
        should_not_call,
    )
    monkeypatch.setattr(
        tennis_lower_tour,
        "fetch_tennisexplorer_pair_context",
        lambda *args, **kwargs: _explorer_pair(
            "Iga Swiatek",
            "Iva Jovic",
            a_rank=3,
            b_rank=16,
            a_wins=8,
            a_losses=2,
            b_wins=7,
            b_losses=3,
            a_h2h_wins=0,
            a_h2h_losses=0,
        ),
    )
    event = "KXWTAMATCH-26OCT07SWIJOV"
    markets = [
        {
            "series_ticker": "KXWTAMATCH",
            "event_ticker": event,
            "ticker": event + "-SWI",
            "yes_sub_title": "Iga Swiatek",
            "yes_ask_dollars": "0.72",
        },
        {
            "series_ticker": "KXWTAMATCH",
            "event_ticker": event,
            "ticker": event + "-JOV",
            "yes_sub_title": "Iva Jovic",
            "yes_ask_dollars": "0.29",
        },
    ]
    rows = build_lower_tour_live_fallback_candidates(
        markets,
        [state],
        [],
        max_workers=1,
    )
    assert len(rows) == 2
    assert not called
    assert all("TennisExplorer" in " ".join(row.model_reasons) for row in rows)
    assert all("ranking-points evidence incomplete" in " ".join(row.model_warnings) for row in rows)


def test_main_tour_without_espn_structural_state_still_fails_closed(monkeypatch):
    state = _state(
        "Iga Swiatek",
        "Iva Jovic",
        tour="WTA",
        source_url=None,
    )
    monkeypatch.setattr(
        tennis_lower_tour,
        "fetch_tennisexplorer_pair_context",
        lambda *args, **kwargs: _explorer_pair("Iga Swiatek", "Iva Jovic"),
    )
    event = "KXWTAMATCH-26OCT07SWIJOV"
    markets = [
        {
            "series_ticker": "KXWTAMATCH",
            "event_ticker": event,
            "ticker": event + "-SWI",
            "yes_sub_title": "Iga Swiatek",
            "yes_ask_dollars": "0.72",
        },
        {
            "series_ticker": "KXWTAMATCH",
            "event_ticker": event,
            "ticker": event + "-JOV",
            "yes_sub_title": "Iva Jovic",
            "yes_ask_dollars": "0.29",
        },
    ]
    assert build_lower_tour_live_fallback_candidates(
        markets,
        [state],
        [],
        max_workers=1,
    ) == []


def test_tennisexplorer_fills_lower_tour_context_when_tennis365_context_is_missing(monkeypatch):
    state = _state()
    monkeypatch.setattr(
        tennis_lower_tour,
        "fetch_tennis365_player_context",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tennis_lower_tour,
        "fetch_tennisexplorer_pair_context",
        lambda *args, **kwargs: _explorer_pair(),
    )
    rows = build_lower_tour_live_fallback_candidates(
        _markets(),
        [state],
        [],
        max_workers=1,
    )
    assert len(rows) == 2
    justo = next(row for row in rows if row.selection == "Guido Ivan Justo")
    assert 0.12 <= justo.model_probability <= 0.88
    assert justo.model_sample_size == 10
    assert "TennisExplorer" in " ".join(justo.model_reasons)
    assert "stricter fallback reversal gates remain active" in " ".join(justo.model_warnings)


def test_cross_source_form_conflict_reduces_fallback_confidence():
    a = _context("Guido Ivan Justo", rank=188, points=None, wins=8, losses=2)
    b = _context("Francisco Comesana", rank=54, points=None, wins=2, losses=8)
    aligned = lower_tour_prior(
        a,
        b,
        explorer=_explorer_pair(
            a_wins=8, a_losses=2, b_wins=2, b_losses=8, a_h2h_wins=0, a_h2h_losses=0
        ),
        explorer_corroborates=True,
    )
    conflict = lower_tour_prior(
        a,
        b,
        explorer=_explorer_pair(
            a_wins=2, a_losses=8, b_wins=8, b_losses=2, a_h2h_wins=0, a_h2h_losses=0
        ),
        explorer_corroborates=True,
    )
    assert aligned is not None and conflict is not None
    assert aligned.confidence > conflict.confidence
    assert any("conflicts with Tennis365" in warning for warning in conflict.warnings)


def test_tennisexplorer_h2h_adjustment_is_small_and_shrunk():
    a = _context("Guido Ivan Justo", rank=188, points=None, wins=6, losses=4)
    b = _context("Francisco Comesana", rank=54, points=None, wins=6, losses=4)
    base = lower_tour_prior(a, b)
    h2h = lower_tour_prior(
        a,
        b,
        explorer=_explorer_pair(
            a_wins=6, a_losses=4, b_wins=6, b_losses=4, a_h2h_wins=3, a_h2h_losses=0
        ),
    )
    assert base is not None and h2h is not None
    assert h2h.probability_a > base.probability_a
    assert h2h.probability_a - base.probability_a < 0.04
    assert any("Beta-shrunk and low-weighted" in reason for reason in h2h.reasons)


def test_fallback_infers_series_from_ticker_when_live_market_omits_series_ticker(monkeypatch):
    state = _state()
    markets = _markets()
    for market in markets:
        market.pop("series_ticker", None)
    monkeypatch.setattr(
        tennis_lower_tour,
        "fetch_tennis365_player_context",
        lambda *args, **kwargs: (
            _context("Guido Ivan Justo", rank=310, points=190, wins=6, losses=4),
            _context("Francisco Comesana", rank=125, points=480, wins=7, losses=3),
        ),
    )
    rows = build_lower_tour_live_fallback_candidates(
        markets,
        [state],
        [],
        max_workers=1,
    )
    assert len(rows) == 2
    assert {row.selection for row in rows} == {"Guido Ivan Justo", "Francisco Comesana"}
