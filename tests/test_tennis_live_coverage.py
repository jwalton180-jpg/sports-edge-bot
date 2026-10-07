from datetime import datetime, timezone

from sports_edge.data.tennis_live import TennisLiveScoreState
from sports_edge.models.event_identity import canonical_event_id, canonical_participant
from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.tennis_live_coverage import build_tennis_live_coverage, ITF_OFFICIAL_LIVE_URL


NOW = datetime(2026, 10, 6, 22, 0, tzinfo=timezone.utc)


def _market(event, series, yes, no, *, start="2026-10-06T21:00:00Z", close="2026-10-06T23:30:00Z"):
    return {
        "event_ticker": event,
        "ticker": event + "-" + canonical_participant("Tennis", yes).replace(" ", "")[:8].upper(),
        "series_ticker": series,
        "event_title": f"{yes} vs {no}",
        "yes_sub_title": yes,
        "no_sub_title": no,
        "occurrence_datetime": start,
        "close_time": close,
    }


def _state(a, b, *, tour="ATP", sources=("ESPN",), conflict=False):
    return TennisLiveScoreState(
        event_id=canonical_event_id("Tennis", a, b, "2026-10-06"),
        selection_key=canonical_participant("Tennis", a),
        player=a,
        opponent=b,
        tour=tour,
        period=2,
        player_sets=1,
        opponent_sets=0,
        player_games=2,
        opponent_games=1,
        lost_first_set=False,
        won_latest_completed_set=True,
        turnaround=False,
        deciding_set=False,
        current_set_lead=1,
        score_label=f"{a} vs {b}",
        fetched_at=NOW,
        serving=True,
        net_break_advantage=0,
        score_sources=sources,
        score_conflict=conflict,
    )


def _candidate(a, b, *, model="Tennis primary"):
    return ParlayCandidateLeg(
        sport="Tennis",
        event_id=canonical_event_id("Tennis", a, b, "2026-10-06"),
        event_title=f"{a} vs {b}",
        market_key="model_h2h",
        market_label="Match Winner",
        selection=a,
        consensus_probability=.6,
        book_count=0,
        source_age_s=0.0,
        median_odds=None,
        kalshi_ticker="KX-TEST",
        kalshi_side="YES",
        kalshi_price=.5,
        kalshi_edge_points=10.0,
        kalshi_status="MODEL",
        evidence_class="MODEL",
        model_probability=.6,
        model_confidence=.65,
        model_name=model,
        model_sample_size=20,
        model_reasons=(),
        model_warnings=(),
    )


def test_coverage_counts_physical_matches_not_contract_sides():
    markets = [
        _market("KXATPMATCH-26OCT06AB", "KXATPMATCH", "Alpha", "Beta"),
        _market("KXATPMATCH-26OCT06AB", "KXATPMATCH", "Beta", "Alpha"),
    ]
    summary = build_tennis_live_coverage(markets, [_state("Alpha", "Beta")], [_candidate("Alpha", "Beta")], now=NOW)
    assert summary.open_matches == 1
    assert summary.live_or_due_matches == 1
    assert summary.score_tracked_matches == 1
    assert summary.model_covered_matches == 1
    assert summary.unsupported_matches == 0
    assert summary.source_counts == (("ESPN", 1),)


def test_started_match_without_score_state_is_exposed_as_coverage_hole():
    markets = [_market("KXATPMATCH-26OCT06AB", "KXATPMATCH", "Alpha", "Beta")]
    summary = build_tennis_live_coverage(markets, [], [_candidate("Alpha", "Beta")], now=NOW)
    row = summary.rows[0]
    assert row.live_or_due
    assert not row.score_tracked
    assert row.model_covered
    assert row.unsupported_reason == "scheduled start passed; no live score state"
    assert summary.unsupported_matches == 1


def test_upcoming_match_inside_grace_is_not_mislabeled_live():
    markets = [_market(
        "KXWTAMATCH-26OCT06AB", "KXWTAMATCH", "Alpha", "Beta",
        start="2026-10-06T21:55:00Z", close="2026-10-07T00:00:00Z"
    )]
    summary = build_tennis_live_coverage(markets, [], [], now=NOW, start_grace_minutes=10)
    assert not summary.rows[0].live_or_due
    assert summary.live_or_due_matches == 0
    assert summary.unsupported_matches == 0


def test_live_score_without_model_is_exposed_as_model_hole():
    markets = [_market("KXITFMATCH-26OCT06AB", "KXITFMATCH", "Alpha", "Beta")]
    summary = build_tennis_live_coverage(
        markets, [_state("Alpha", "Beta", tour="ITF", sources=("Tennis365",))], [], now=NOW
    )
    row = summary.rows[0]
    assert row.tour == "ITF Men"
    assert row.score_tracked
    assert not row.model_covered
    assert row.unsupported_reason == "live score tracked; no usable independent model prior"
    assert row.official_itf_url == ITF_OFFICIAL_LIVE_URL


def test_score_conflict_is_fail_closed_and_visible():
    markets = [_market("KXITFWMATCH-26OCT06AB", "KXITFWMATCH", "Alpha", "Beta")]
    summary = build_tennis_live_coverage(
        markets,
        [_state("Alpha", "Beta", tour="ITF-W", sources=("Tennis365", "ITF Official"), conflict=True)],
        [_candidate("Alpha", "Beta")],
        now=NOW,
    )
    row = summary.rows[0]
    assert row.tour == "ITF Women"
    assert row.score_conflict
    assert row.unsupported_reason == "live score sources disagree"
    assert summary.unsupported_matches == 1
    assert row.official_itf_tour_url and "womens-world-tennis-tour" in row.official_itf_tour_url
