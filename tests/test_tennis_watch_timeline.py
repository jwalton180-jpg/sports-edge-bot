"""Causal Tennis stage-board regressions from Guo–Cirstea observed records."""
from datetime import datetime, timezone, timedelta
from sports_edge.models.tennis_watch_timeline import build_tennis_watch_timelines


EVENT="TENNIS:2026-10-11:cirstea sorana|guo hanyu"
TICKER="KXWTAMATCH-26OCT11GUOCIR-GUO"


def _record(*, observation_time, price, fair, lane, signal_id):
    return {
        "signal_id": signal_id,
        "event_id": EVENT,
        "ticker": TICKER,
        "selection": "Hanyu Guo",
        "side": "YES",
        "lane": lane,
        "entry_ask": price,
        "model_fair": fair,
        "first_observed_at": observation_time,
        "score_state": "Sorana Cirstea vs Guo Hanyu · 6-1 · 6-6",
    }


GUO_RECORDS = [
    _record(observation_time="2026-10-13T07:07:03+00:00",
            price=.12,fair=.024881,lane="WATCH",signal_id="guo-watch"),
    _record(observation_time="2026-10-13T07:12:41+00:00",
            price=.05,fair=.1252,lane="EARLY WATCH — RESEARCH",signal_id="guo-early"),
    _record(observation_time="2026-10-13T07:56:52+00:00",
            price=.15,fair=.261874,lane="RECOVERY BUILDING — RESEARCH",
            signal_id="guo-recovery"),
]


def test_guo_cirstea_timeline_preserves_early_five_cent_watch_without_hindsight():
    now=datetime(2026,10,13,8,0,tzinfo=timezone.utc)
    rows=build_tennis_watch_timelines(
        GUO_RECORDS, now=now, live_event_ids={EVENT}
    )
    assert len(rows)==1
    row=rows[0]
    assert row.first_price==.12
    assert row.earliest_positive_edge_price==.05
    assert row.earliest_positive_edge_at.isoformat()=="2026-10-13T07:12:41+00:00"
    assert row.latest_price==.15
    assert row.latest_model_fair==.261874
    assert row.status=="RECENT RESEARCH SIGNAL — VERIFY LIVE QUOTE"
    assert row.confirmed_live
    assert len(row.sequence)==3


def test_negative_edge_watch_is_never_highlighted_as_positive_reversal():
    now=datetime(2026,10,13,7,10,tzinfo=timezone.utc)
    rows=build_tennis_watch_timelines(
        GUO_RECORDS[:1],now=now,live_event_ids={EVENT}
    )
    assert len(rows)==1
    assert rows[0].status=="WATCH — NO ESTIMATED EDGE"
    assert rows[0].earliest_positive_edge_price is None


def test_completed_or_unverified_live_match_is_historical_not_actionable():
    now=datetime(2026,10,13,8,0,tzinfo=timezone.utc)
    rows=build_tennis_watch_timelines(GUO_RECORDS,now=now,live_event_ids=set())
    assert len(rows)==1
    assert rows[0].status=="HISTORICAL — LIVE NOT VERIFIED"
    assert not rows[0].confirmed_live


def test_verified_terminal_grade_supersedes_historical_and_live_watch():
    now=datetime(2026,10,13,8,0,tzinfo=timezone.utc)
    rows=build_tennis_watch_timelines(
        GUO_RECORDS,now=now,live_event_ids={EVENT},
        grades=[{"signal_id":"guo-early","outcome":"WIN"}],
    )
    assert len(rows)==1
    assert rows[0].status=="VERIFIED WIN — HISTORICAL"
    assert rows[0].verified_grade=="WIN"


def test_watch_goes_stale_and_cannot_be_presented_as_fresh_entry():
    now=datetime(2026,10,13,8,16,tzinfo=timezone.utc)
    rows=build_tennis_watch_timelines(
        GUO_RECORDS,now=now,live_event_ids={EVENT}
    )
    assert len(rows)==1
    assert rows[0].status=="OLDER WATCH — RECHECK PRICE"


def test_no_future_signal_leakage_and_exact_event_identity():
    now=datetime(2026,10,13,7,10,tzinfo=timezone.utc)
    future_records=GUO_RECORDS
    rows=build_tennis_watch_timelines(
        future_records,now=now,live_event_ids={EVENT}
    )
    assert len(rows)==1
    assert len(rows[0].sequence)==1
    assert rows[0].first_price==.12
