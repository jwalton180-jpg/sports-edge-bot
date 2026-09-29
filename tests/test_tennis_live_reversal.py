from datetime import datetime, timezone

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
