from datetime import datetime, timezone

from sports_edge.data.tennis_live import TennisLiveScoreState
from sports_edge.models.event_identity import canonical_event_id, canonical_participant
from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.tennis_live_reversal import (
    assess_tennis_reversal,
    build_tennis_reversal_radar,
    executable_path,
)


NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


def _leg(event_id="TENNIS:2026-09-29:alpha|beta", ticker="KX-T1", fair=0.38, confidence=0.72):
    return ParlayCandidateLeg(
        sport="Tennis",
        event_id=event_id,
        event_title="Alpha vs Beta",
        market_key="model_h2h",
        market_label="Match Winner",
        selection="Alpha",
        consensus_probability=fair,
        book_count=0,
        source_age_s=0.0,
        median_odds=None,
        kalshi_ticker=ticker,
        kalshi_side="YES",
        kalshi_price=0.20,
        kalshi_edge_points=(fair - 0.20) * 100,
        kalshi_status="MODEL",
        evidence_class="MODEL",
        model_probability=fair,
        model_confidence=confidence,
        model_name="Tennis adaptive",
        model_sample_size=30,
        model_reasons=(
            "Head-to-head before match: 2-1",
            "Form trajectory delta +0.20",
            "Recent serve/return trend delta +0.035",
        ),
        model_warnings=(),
    )


def _candle(minute, close, low=None, high=None, volume=100):
    ts = int(NOW.timestamp()) - (9 - minute) * 60
    return {
        "end_period_ts": ts,
        "volume_fp": str(volume),
        "price": {
            "close_dollars": f"{close:.4f}",
            "high_dollars": f"{(high if high is not None else close):.4f}",
            "low_dollars": f"{(low if low is not None else close):.4f}",
        },
        "yes_ask": {
            "close_dollars": f"{close:.4f}",
            "high_dollars": f"{(high if high is not None else close):.4f}",
            "low_dollars": f"{(low if low is not None else close):.4f}",
        },
        "yes_bid": {
            "close_dollars": f"{max(0.01, close - 0.02):.4f}",
            "high_dollars": f"{max(0.01, (high if high is not None else close) - 0.02):.4f}",
            "low_dollars": f"{max(0.01, (low if low is not None else close) - 0.02):.4f}",
        },
    }


def _reversal_candles():
    vals = [
        (0, .36, .35, .37),
        (1, .34, .33, .36),
        (2, .29, .27, .34),
        (3, .22, .20, .29),
        (4, .15, .14, .22),
        (5, .16, .14, .17),
        (6, .18, .16, .19),
        (7, .20, .18, .21),
        (8, .22, .20, .23),
        (9, .24, .22, .25),
    ]
    return tuple(_candle(i, close, low, high, volume=120) for i, close, low, high in vals)


def test_executable_yes_path_uses_ask_quotes():
    path = executable_path((_candle(9, .25, .23, .27),), "YES")
    assert len(path) == 1
    assert abs(path[0].close - .25) < 1e-9
    assert abs(path[0].low - .23) < 1e-9
    assert abs(path[0].high - .27) < 1e-9


def test_executable_no_path_uses_complement_of_yes_bid():
    path = executable_path((_candle(9, .25, .23, .27),), "NO")
    assert len(path) == 1
    assert abs(path[0].close - .77) < 1e-9


def test_confirmed_live_major_dip_rebound_can_signal():
    sig = assess_tennis_reversal(_leg(), _reversal_candles(), confirmed_live=True, now=NOW)
    assert sig is not None
    assert sig.status == "REVERSAL SIGNAL"
    assert sig.dip_points >= 20
    assert sig.rebound_points >= 8
    assert sig.h2h_context
    assert sig.trend_context


def test_same_price_collapse_without_rebound_does_not_signal():
    candles = tuple(
        _candle(i, v, v - .01, v + .01)
        for i, v in enumerate([.36,.34,.31,.28,.24,.20,.17,.14,.12,.10])
    )
    sig = assess_tennis_reversal(_leg(), candles, confirmed_live=True, now=NOW)
    assert sig is None or sig.status == "PASS"


def test_rebound_without_model_prior_support_does_not_signal():
    sig = assess_tennis_reversal(_leg(fair=.20), _reversal_candles(), confirmed_live=True, now=NOW)
    assert sig is not None
    assert sig.status == "PASS"


def test_unconfirmed_match_stays_watch_only():
    sig = assess_tennis_reversal(_leg(), _reversal_candles(), confirmed_live=False, now=NOW)
    assert sig is not None
    assert sig.status == "WATCH"
    assert any("not independently confirmed" in w for w in sig.warnings)


def test_radar_keeps_one_side_per_physical_match():
    a = _leg(ticker="KX-A", fair=.40)
    b = _leg(ticker="KX-B", fair=.35)
    hist = {"KX-A": _reversal_candles(), "KX-B": _reversal_candles()}
    rows = build_tennis_reversal_radar(
        [a, b],
        hist,
        confirmed_live_event_ids={a.event_id},
        now=NOW,
    )
    assert len(rows) == 1
    assert rows[0].event_id == a.event_id


def _deep_state(*, lead=0, turnaround=True):
    event_id = canonical_event_id("Tennis", "Rigele TE", "Adam Walton", "2026-10-05")
    return TennisLiveScoreState(
        event_id=event_id,
        selection_key=canonical_participant("Tennis", "Rigele TE"),
        player="Te Rigele",
        opponent="Adam Walton",
        tour="ATP",
        period=3,
        player_sets=1,
        opponent_sets=1,
        player_games=lead if lead > 0 else 0,
        opponent_games=0,
        lost_first_set=True,
        won_latest_completed_set=turnaround,
        turnaround=turnaround,
        deciding_set=True,
        current_set_lead=lead,
        score_label="Adam Walton vs Te Rigele · 7-5 · 3-6 · 0-0",
        fetched_at=NOW,
    )


def _deep_leg():
    event_id = canonical_event_id("Tennis", "Rigele TE", "Adam Walton", "2026-10-05")
    return _leg(event_id=event_id, ticker="KX-RIG", fair=.10, confidence=.62)


def _deep_reversal_candles():
    vals = [
        (0, .18, .17, .19),
        (1, .15, .14, .18),
        (2, .10, .09, .15),
        (3, .07, .06, .10),
        (4, .05, .04, .07),
        (5, .06, .04, .07),
        (6, .08, .06, .09),
        (7, .10, .08, .11),
        (8, .12, .10, .13),
        (9, .13, .11, .14),
    ]
    return tuple(_candle(i, close, low, high, volume=150) for i, close, low, high in vals)


def test_deep_reversal_can_override_negative_pregame_gap_with_score_turnaround():
    # Current 13% is ABOVE the 10% pregame prior, so the generic model-gap
    # gate would reject it. A real score turnaround plus deep price recovery
    # is the intended high-payout exception.
    sig = assess_tennis_reversal(
        _deep_leg(),
        _deep_reversal_candles(),
        live_state=_deep_state(),
        now=NOW,
    )
    assert sig is not None
    assert sig.prior_gap_points < 0
    assert sig.status == "DEEP REVERSAL"
    assert sig.score_turnaround
    assert sig.deciding_set
    assert sig.live_score is not None


def test_deep_price_rebound_without_score_turnaround_is_not_promoted():
    sig = assess_tennis_reversal(
        _deep_leg(),
        _deep_reversal_candles(),
        confirmed_live=True,
        live_state=None,
        now=NOW,
    )
    assert sig is not None
    assert sig.status == "PASS"


def test_deep_deciding_set_lead_can_support_signal_without_opening_set_turnaround():
    state = _deep_state(lead=2, turnaround=False)
    sig = assess_tennis_reversal(
        _deep_leg(),
        _deep_reversal_candles(),
        live_state=state,
        now=NOW,
    )
    assert sig is not None
    assert sig.status == "DEEP REVERSAL"


def test_radar_ranks_deep_reversal_above_generic_signal():
    deep_leg = _deep_leg()
    generic = _leg(
        event_id="TENNIS:2026-09-29:alpha|beta",
        ticker="KX-GENERIC",
        fair=.38,
        confidence=.72,
    )
    states = {
        (deep_leg.event_id, canonical_participant("Tennis", deep_leg.selection)): _deep_state()
    }
    rows = build_tennis_reversal_radar(
        [generic, deep_leg],
        {
            "KX-GENERIC": _reversal_candles(),
            "KX-RIG": _deep_reversal_candles(),
        },
        confirmed_live_event_ids={generic.event_id, deep_leg.event_id},
        live_states=states,
        now=NOW,
    )
    assert rows
    assert rows[0].status == "DEEP REVERSAL"
