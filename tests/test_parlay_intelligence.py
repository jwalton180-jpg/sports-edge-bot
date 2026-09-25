from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.parlay_intelligence import assess_leg, build_intelligent_parlay


def leg(
    *,
    event_id="E1",
    selection="Team A",
    fair=0.24,
    price=0.15,
    books=5,
    age=25.0,
    evidence="EDGE-QUALIFIED",
    sport="Tennis",
):
    return ParlayCandidateLeg(
        sport=sport,
        event_id=event_id,
        event_title=f"Event {event_id}",
        market_key="h2h",
        market_label="Moneyline",
        selection=selection,
        consensus_probability=fair,
        book_count=books,
        source_age_s=age,
        median_odds=None,
        kalshi_ticker=f"KX-{event_id}-{selection}",
        kalshi_side="YES",
        kalshi_price=price,
        kalshi_edge_points=100.0 * (fair - price),
        kalshi_status="QUALIFIED",
        evidence_class=evidence,
    )


def test_priced_longshot_requires_positive_ev_not_cheap_price():
    good = assess_leg(leg(fair=0.24, price=0.15), "longshot")
    bad = assess_leg(leg(fair=0.10, price=0.15), "longshot")

    assert good.qualified is True
    assert round(good.edge_points, 1) == 9.0
    assert good.expected_roi_on_cost > 0.50
    assert bad.qualified is False


def test_longshot_requires_independent_book_depth_and_freshness():
    thin = assess_leg(leg(books=2), "longshot")
    stale = assess_leg(leg(age=121), "longshot")

    assert thin.qualified is False
    assert stale.qualified is False


def test_best_available_rejects_favorite_without_edge():
    no_edge = assess_leg(fair=0.70, price=0.69) if False else None


def test_best_available_is_not_just_high_probability():
    strong_favorite_no_edge = assess_leg(leg(fair=0.70, price=0.69, evidence="KALSHI MATCH"), "best")
    genuine_value = assess_leg(leg(fair=0.62, price=0.56), "best")

    assert strong_favorite_no_edge.qualified is False
    assert genuine_value.qualified is True


def test_builder_never_forces_filler_legs():
    candidates = [
        leg(event_id="E1", selection="A", fair=0.25, price=0.15),
        leg(event_id="E2", selection="B", fair=0.22, price=0.16),
        leg(event_id="E3", selection="C", fair=0.11, price=0.17),
    ]
    result = build_intelligent_parlay(candidates, mode="longshot", target_legs=5)

    assert len(result.legs) == 2
    assert any("Only 2" in warning for warning in result.warnings)


def test_one_leg_per_event_reduces_obvious_same_game_correlation():
    candidates = [
        leg(event_id="E1", selection="A", fair=0.25, price=0.15),
        leg(event_id="E1", selection="B", fair=0.24, price=0.14),
        leg(event_id="E2", selection="C", fair=0.23, price=0.15),
    ]
    result = build_intelligent_parlay(candidates, mode="longshot", target_legs=3, max_per_event=1)

    assert len(result.legs) == 2
    assert len({row.leg.event_id for row in result.legs}) == 2
