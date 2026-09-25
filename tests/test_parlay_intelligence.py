from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.parlay_intelligence import assess_leg, build_intelligent_parlay


def leg(
    *,
    event_id="E1",
    selection="Team A",
    fair=0.24,
    price=0.15,
    books=0,
    age=0.0,
    model_conf=0.72,
    sport="Tennis",
    with_model=True,
):
    return ParlayCandidateLeg(
        sport=sport,
        event_id=event_id,
        event_title=f"Event {event_id}",
        market_key="model_h2h",
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
        kalshi_status="MODEL",
        evidence_class="MODEL",
        model_probability=fair if with_model else None,
        model_confidence=model_conf if with_model else 0.0,
        model_name="test sport model" if with_model else None,
        model_sample_size=20 if with_model else 0,
        model_reasons=("independent sport features",) if with_model else (),
    )


def test_book_only_leg_cannot_qualify():
    row = leg(fair=0.70, price=0.55, books=5, age=20, with_model=False)
    assessed = assess_leg(row, "best")
    assert assessed.qualified is False
    assert any("sport-specific model" in warning for warning in assessed.warnings)


def test_model_backed_longshot_can_qualify_without_sportsbooks():
    good = assess_leg(leg(fair=0.24, price=0.15, books=0), "longshot")
    assert good.qualified is True
    assert round(good.edge_points, 1) == 9.0
    assert good.expected_roi_on_cost > 0.50


def test_priced_longshot_rejects_negative_ev_even_with_model():
    bad = assess_leg(leg(fair=0.10, price=0.15), "longshot")
    assert bad.qualified is False


def test_stale_sportsbook_crosscheck_does_not_veto_valid_model():
    row = assess_leg(leg(fair=0.24, price=0.15, books=4, age=121), "longshot")
    assert row.qualified is True
    assert any("stale" in warning.lower() for warning in row.warnings)


def test_best_available_is_not_just_high_probability():
    no_edge = assess_leg(leg(fair=0.70, price=0.69), "best")
    genuine_value = assess_leg(leg(fair=0.62, price=0.56), "best")
    assert no_edge.qualified is False
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
