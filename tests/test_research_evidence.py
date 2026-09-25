from datetime import datetime, timedelta, timezone

from sports_edge.research.line_history import LineSnapshot, append_line_snapshot, closing_line_report
from sports_edge.research.tipsters import PublicTrackRecord, consensus_support


def test_tipster_requires_sample_verification_and_positive_clv():
    good = PublicTrackRecord(
        handle="verified",
        wins=120,
        losses=80,
        verified_posts=200,
        avg_clv_points=1.8,
    )
    weak = PublicTrackRecord(
        handle="small",
        wins=18,
        losses=8,
        verified_posts=26,
        avg_clv_points=2.0,
    )
    assert good.tail_eligible()
    assert not weak.tail_eligible()
    summary = consensus_support([good, weak])
    assert summary["eligible_count"] == 1
    assert summary["quality"] > 0.7


def test_line_history_append_is_idempotent_and_clv_is_prestart(tmp_path):
    now = datetime(2026, 9, 24, 20, 0, tzinfo=timezone.utc)
    start = now + timedelta(hours=1)
    snap = LineSnapshot(
        observed_at=start - timedelta(minutes=5),
        source="kalshi",
        event_id="e1",
        market_key="h2h",
        selection="Team A",
        price=0.61,
    )
    path = tmp_path / "lines.jsonl"
    first = append_line_snapshot(path, snap)
    second = append_line_snapshot(path, snap)
    assert first["appended"] is True
    assert second["appended"] is False

    report = closing_line_report(
        [snap],
        entry_time=now,
        event_start=start,
        entry_price=0.55,
    )
    assert report["eligible"] is True
    assert report["clv_probability_points"] > 0
