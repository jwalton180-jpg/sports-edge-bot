from datetime import datetime, timedelta, timezone

from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.tennis_reversal import analyze_tennis_reversal


NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


def _candidate(price=0.32, fair=0.46, confidence=0.62):
    return ParlayCandidateLeg(
        sport="Tennis",
        event_id="TENNIS:2026-09-29:alpha|beta",
        event_title="Alpha vs Beta",
        market_key="model_h2h",
        market_label="Moneyline",
        selection="Alpha",
        consensus_probability=fair,
        book_count=0,
        source_age_s=0.0,
        median_odds=None,
        kalshi_ticker="KXTEST-ALPHA",
        kalshi_side="YES",
        kalshi_price=price,
        kalshi_edge_points=100 * (fair - price),
        kalshi_status="MODEL",
        evidence_class="MODEL",
        model_probability=fair,
        model_confidence=confidence,
        model_name="Tennis test",
        model_sample_size=40,
        model_reasons=("Head-to-head before match: 2-1",),
    )


def _market(*, live=True):
    occurrence = NOW - timedelta(minutes=80) if live else NOW + timedelta(hours=1)
    return {
        "ticker": "KXTEST-ALPHA",
        "series_ticker": "KXATPMATCH",
        "status": "active",
        "occurrence_datetime": occurrence.isoformat(),
        "close_time": (NOW + timedelta(hours=2)).isoformat(),
    }


def _candle(ts, price, volume=5):
    return {
        "end_period_ts": int(ts.timestamp()),
        "yes_bid": {"close_dollars": f"{price - 0.01:.4f}"},
        "yes_ask": {"close_dollars": f"{price + 0.01:.4f}"},
        "price": {"close_dollars": f"{price:.4f}"},
        "volume_fp": str(volume),
    }


def _reversal_candles():
    prices = [0.46, 0.44, 0.40, 0.35, 0.28, 0.22, 0.24, 0.26, 0.29, 0.30]
    start = NOW - timedelta(minutes=len(prices))
    return [_candle(start + timedelta(minutes=i), p, 8) for i, p in enumerate(prices)]


def test_major_dip_recovery_with_model_edge_qualifies():
    signal = analyze_tennis_reversal(
        candidate=_candidate(price=0.32, fair=0.46, confidence=0.62),
        market=_market(live=True),
        candles=_reversal_candles(),
        now=NOW,
    )
    assert signal is not None
    assert signal.status == "REVERSAL"
    assert signal.drawdown_pp >= 20
    assert signal.recovery_pp >= 9
    assert signal.model_edge_pp >= 13
    assert any("Major in-match market dip" in r for r in signal.reasons)


def test_future_match_is_not_called_live():
    signal = analyze_tennis_reversal(
        candidate=_candidate(),
        market=_market(live=False),
        candles=_reversal_candles(),
        now=NOW,
    )
    assert signal is None


def test_small_dip_is_rejected():
    prices = [0.42, 0.41, 0.40, 0.39, 0.38, 0.39, 0.40, 0.41]
    start = NOW - timedelta(minutes=len(prices))
    candles = [_candle(start + timedelta(minutes=i), p) for i, p in enumerate(prices)]
    signal = analyze_tennis_reversal(
        candidate=_candidate(price=0.41, fair=0.50, confidence=0.65),
        market=_market(live=True),
        candles=candles,
        now=NOW,
    )
    assert signal is None


def test_model_disagreement_is_required():
    signal = analyze_tennis_reversal(
        candidate=_candidate(price=0.32, fair=0.35, confidence=0.62),
        market=_market(live=True),
        candles=_reversal_candles(),
        now=NOW,
    )
    assert signal is None


def test_favorite_after_recovery_is_not_an_underdog_entry():
    signal = analyze_tennis_reversal(
        candidate=_candidate(price=0.54, fair=0.62, confidence=0.70),
        market=_market(live=True),
        candles=_reversal_candles(),
        now=NOW,
    )
    assert signal is None
