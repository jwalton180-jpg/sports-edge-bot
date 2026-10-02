from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.parlay_intelligence import assess_leg
from sports_edge.research.social_intelligence import (
    SOCIAL_RESEARCH_SOURCES,
    community_methodology_fit,
    research_source_rows,
)


def leg(
    *,
    market_key="batter_hits",
    selection="Hitter 1+ hit",
    fair=0.64,
    price=0.54,
    confidence=0.72,
    reasons=("stable role", "favorable matchup", "recent form"),
):
    return ParlayCandidateLeg(
        sport="MLB",
        event_id="MLB-G1",
        event_title="A vs B",
        market_key=market_key,
        market_label="Test",
        selection=selection,
        consensus_probability=fair,
        book_count=0,
        source_age_s=0.0,
        median_odds=None,
        kalshi_ticker=f"KX-{market_key}",
        kalshi_side="YES",
        kalshi_price=price,
        kalshi_edge_points=100.0 * (fair - price),
        kalshi_status="MODEL",
        evidence_class="MODEL",
        model_probability=fair,
        model_confidence=confidence,
        model_name="test model",
        model_sample_size=30,
        model_reasons=tuple(reasons),
        model_warnings=(),
    )


def test_stable_context_rich_leg_has_stronger_public_sharp_methodology_fit():
    stable = leg()
    volatile = leg(
        market_key="batter_home_runs",
        selection="Hitter home run",
        fair=0.61,
        price=0.51,
        reasons=("recent power",),
    )
    stable_score, _ = community_methodology_fit(
        stable,
        fair_probability=0.64,
        kalshi_probability=0.54,
        mode="best",
    )
    volatile_score, _ = community_methodology_fit(
        volatile,
        fair_probability=0.61,
        kalshi_probability=0.51,
        mode="best",
    )
    assert stable_score > volatile_score


def test_public_sharp_layer_does_not_rescue_negative_ev_leg():
    bad = leg(fair=0.50, price=0.56)
    assessed = assess_leg(bad, "best")
    assert assessed.qualified is False
    assert assessed.community_methodology_score >= 0.0
    assert any("edge below" in warning for warning in assessed.warnings)


def test_public_sharp_layer_is_visible_but_secondary_on_assessment():
    row = assess_leg(leg(), "best")
    assert row.qualified is True
    assert 0.0 <= row.community_methodology_score <= 1.0
    assert row.community_methodology_notes
    assert any("Public-sharp methodology fit" in reason for reason in row.reasons)


def test_social_registry_is_transparent_and_cautioned():
    rows = research_source_rows()
    assert len(rows) == len(SOCIAL_RESEARCH_SOURCES)
    assert len(rows) >= 5
    assert all(row["Source"] for row in rows)
    assert all(row["Caution"] for row in rows)
    assert any("RIBBRIT" in row["Source"] for row in rows)
    assert any("LT_Picks" in row["Source"] for row in rows)
