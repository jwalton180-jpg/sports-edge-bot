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


def test_deep_model_evidence_can_qualify_smaller_positive_best_edge():
    row = assess_leg(leg(fair=0.63, price=0.61, model_conf=0.78), "best")
    assert row.qualified is True
    assert row.edge_points > 0
    assert any("Deep Evidence" in reason for reason in row.reasons)


def test_low_confidence_model_keeps_standard_best_edge_gate():
    row = assess_leg(leg(fair=0.63, price=0.61, model_conf=0.55), "best")
    assert row.qualified is False
    assert any("+3.0 pp" in warning for warning in row.warnings)


def test_deep_model_never_qualifies_zero_or_negative_edge():
    row = assess_leg(leg(fair=0.63, price=0.64, model_conf=0.80), "best")
    assert row.qualified is False


def test_best_builder_can_use_multiple_deep_evidence_positive_value_legs():
    candidates = [
        leg(event_id="E1", selection="A", fair=0.63, price=0.61, model_conf=0.78),
        leg(event_id="E2", selection="B", fair=0.61, price=0.59, model_conf=0.76),
        leg(event_id="E3", selection="C", fair=0.60, price=0.58, model_conf=0.75),
    ]
    result = build_intelligent_parlay(candidates, mode="best", target_legs=3)
    assert len(result.legs) == 3
    assert all(row.edge_points > 0 for row in result.legs)


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



def test_extreme_model_market_gap_requires_secondary_confirmation():
    row = leg(fair=0.55, price=0.10, books=0)
    assessed = assess_leg(row, "longshot")
    assert assessed.qualified is False
    assert any("extreme model-market dislocation" in warning for warning in assessed.warnings)


def test_extreme_gap_with_conflicting_fresh_books_still_fails_closed():
    row = leg(fair=0.55, price=0.10, books=4, age=20)
    row = ParlayCandidateLeg(
        **{
            **row.__dict__,
            "consensus_probability": 0.25,
        }
    )
    assessed = assess_leg(row, "longshot")
    assert assessed.qualified is False
    assert any("conflict" in warning for warning in assessed.warnings)


def test_extreme_gap_can_pass_when_fresh_secondary_confirmation_agrees():
    row = leg(fair=0.55, price=0.10, books=4, age=20)
    row = ParlayCandidateLeg(
        **{
            **row.__dict__,
            "consensus_probability": 0.48,
        }
    )
    assessed = assess_leg(row, "longshot")
    assert assessed.qualified is True


def test_mlb_same_game_distinct_markets_can_both_enter_default_ticket():
    ml = leg(event_id="MLB-G1", selection="San Diego to win", fair=0.66, price=0.58, sport="MLB")
    total = leg(event_id="MLB-G1", selection="San Diego Over 4.5 Team Total", fair=0.64, price=0.55, sport="MLB")
    other = leg(event_id="MLB-G2", selection="Atlanta to win", fair=0.64, price=0.57, sport="MLB")
    result = build_intelligent_parlay([ml, total, other], mode="best", target_legs=3)
    assert len(result.legs) == 3
    assert sum(row.leg.event_id == "MLB-G1" for row in result.legs) == 2
    assert result.correlation_risk == "MEDIUM"
    assert any("Same-game legs are dependent" in warning for warning in result.warnings)


def test_mlb_default_caps_same_game_concentration_at_three_qualified_contracts():
    rows = [
        leg(event_id="MLB-G1", selection=f"Market {i}", fair=0.66 - i * 0.01, price=0.55, sport="MLB")
        for i in range(4)
    ]
    result = build_intelligent_parlay(rows, mode="best", target_legs=4)
    assert len(result.legs) == 3
    assert result.correlation_risk == "HIGH"


def test_mlb_same_game_builder_never_uses_negative_ev_filler():
    good = leg(event_id="MLB-G1", selection="San Diego to win", fair=0.66, price=0.58, sport="MLB")
    bad = leg(event_id="MLB-G1", selection="Weak prop", fair=0.54, price=0.57, sport="MLB")
    result = build_intelligent_parlay([good, bad], mode="best", target_legs=2)
    assert len(result.legs) == 1
    assert result.legs[0].leg.selection == "San Diego to win"


def test_single_game_multiplier_priority_prefers_cheaper_qualified_positive_value_legs():
    rows = [
        leg(event_id="MLB-G1", selection="Safer", fair=0.70, price=0.62, sport="MLB"),
        leg(event_id="MLB-G1", selection="Value A", fair=0.54, price=0.44, sport="MLB"),
        leg(event_id="MLB-G1", selection="Value B", fair=0.54, price=0.39, sport="MLB"),
        leg(event_id="MLB-G1", selection="Value C", fair=0.54, price=0.36, sport="MLB"),
        leg(event_id="MLB-G1", selection="Value D", fair=0.54, price=0.34, sport="MLB"),
    ]
    result = build_intelligent_parlay(
        rows,
        mode="best",
        target_legs=4,
        max_per_event=4,
        prioritize_payout_multiplier=True,
    )
    assert len(result.legs) == 4
    assert "Safer" not in {row.leg.selection for row in result.legs}
    prices = [row.kalshi_probability for row in result.legs]
    assert prices == sorted(prices)


def test_single_game_multiplier_priority_still_rejects_negative_value():
    good = leg(event_id="MLB-G1", selection="Good", fair=0.55, price=0.45, sport="MLB")
    bad = leg(event_id="MLB-G1", selection="Cheap but bad", fair=0.20, price=0.25, sport="MLB")
    result = build_intelligent_parlay(
        [good, bad],
        mode="best",
        target_legs=4,
        max_per_event=4,
        prioritize_payout_multiplier=True,
    )
    assert [row.leg.selection for row in result.legs] == ["Good"]


def test_stable_volume_leg_beats_high_variance_leg_for_single_game_core():
    stable = leg(event_id="MLB-G1", selection="Hitter 1+ hit", fair=0.64, price=0.54, sport="MLB")
    stable = ParlayCandidateLeg(**{**stable.__dict__, "market_key": "batter_hits"})
    volatile = leg(event_id="MLB-G1", selection="Hitter home run", fair=0.61, price=0.40, sport="MLB")
    volatile = ParlayCandidateLeg(**{**volatile.__dict__, "market_key": "batter_home_runs"})
    result = build_intelligent_parlay(
        [volatile, stable],
        mode="best",
        target_legs=1,
        max_per_event=4,
        prioritize_payout_multiplier=True,
    )
    assert result.legs[0].leg.selection == "Hitter 1+ hit"
    assert result.legs[0].involvement_rating == "HIGH"
    assert result.legs[0].variance_rating == "LOW"


def test_material_role_uncertainty_blocks_leg():
    row = leg(event_id="NFL-G1", selection="Player receptions", fair=0.66, price=0.56, sport="NFL")
    row = ParlayCandidateLeg(
        **{
            **row.__dict__,
            "market_key": "receptions",
            "model_warnings": ("Player has a snap restriction",),
        }
    )
    assessed = assess_leg(row, "best")
    assert assessed.qualified is False
    assert assessed.role_check == "BLOCK"
    assert any("role/availability" in warning for warning in assessed.warnings)


def test_ticket_reports_payout_risk_and_failure_map():
    rows = [
        ParlayCandidateLeg(**{**leg(event_id="E1", selection="A", fair=0.70, price=0.60).__dict__, "market_key": "model_h2h"}),
        ParlayCandidateLeg(**{**leg(event_id="E2", selection="B", fair=0.65, price=0.55).__dict__, "market_key": "model_h2h"}),
    ]
    result = build_intelligent_parlay(rows, mode="best", target_legs=2)
    assert result.market_payout_multiplier > 1.0
    assert result.risk_label
    assert result.strongest_leg in {"A", "B"}
    assert result.weakest_leg in {"A", "B"}
    assert result.highest_variance_leg in {"A", "B"}
    assert result.primary_failure_scenario


def test_selected_games_prioritize_one_qualified_leg_per_requested_event_before_extras():
    rows = [
        leg(event_id="E1", selection="E1 strongest", fair=0.76, price=0.60, sport="MLB"),
        leg(event_id="E1", selection="E1 extra", fair=0.74, price=0.59, sport="MLB"),
        leg(event_id="E2", selection="E2 strongest", fair=0.74, price=0.58, sport="MLB"),
        leg(event_id="E2", selection="E2 extra", fair=0.72, price=0.57, sport="MLB"),
        leg(event_id="E3", selection="E3 strongest", fair=0.66, price=0.59, sport="MLB"),
        leg(event_id="E4", selection="E4 qualified", fair=0.58, price=0.54, sport="MLB"),
    ]
    result = build_intelligent_parlay(
        rows,
        mode="best",
        target_legs=5,
        max_per_event=3,
        preferred_event_ids=["E1", "E2", "E3", "E4"],
    )
    assert len(result.legs) == 5
    assert {row.leg.event_id for row in result.legs} == {"E1", "E2", "E3", "E4"}
    assert result.requested_event_count == 4
    assert result.represented_event_count == 4
    assert result.missing_event_ids == ()


def test_selected_games_never_force_bad_leg_and_report_unrepresented_event():
    rows = [
        leg(event_id="E1", selection="E1 A", fair=0.70, price=0.60, sport="MLB"),
        leg(event_id="E1", selection="E1 B", fair=0.68, price=0.59, sport="MLB"),
        leg(event_id="E2", selection="E2 A", fair=0.68, price=0.58, sport="MLB"),
        leg(event_id="E2", selection="E2 B", fair=0.66, price=0.57, sport="MLB"),
        leg(event_id="E3", selection="E3 A", fair=0.65, price=0.57, sport="MLB"),
        leg(event_id="E4", selection="E4 bad price", fair=0.50, price=0.56, sport="MLB"),
    ]
    result = build_intelligent_parlay(
        rows,
        mode="best",
        target_legs=5,
        max_per_event=3,
        preferred_event_ids=["E1", "E2", "E3", "E4"],
    )
    assert len(result.legs) == 5
    assert "E4" not in {row.leg.event_id for row in result.legs}
    assert result.requested_event_count == 4
    assert result.represented_event_count == 3
    assert result.missing_event_ids == ("E4",)
    assert any("3/4 selected game(s) are represented" in warning for warning in result.warnings)


def test_best_available_rejects_coinflipish_positive_ev_leg_even_with_deep_evidence():
    row = leg(fair=0.54, price=0.50, model_conf=0.80)
    assessed = assess_leg(row, "best")
    assert assessed.qualified is False
    assert any("below 55%" in warning for warning in assessed.warnings)


def test_best_available_prefers_higher_fair_probability_after_stability_gates():
    safer = leg(event_id="E1", selection="Safer", fair=0.68, price=0.62, model_conf=0.78)
    cheaper = leg(event_id="E2", selection="Cheaper", fair=0.58, price=0.50, model_conf=0.78)
    result = build_intelligent_parlay([cheaper, safer], mode="best", target_legs=1)
    assert result.legs[0].leg.selection == "Safer"


def test_best_available_and_longshot_remain_distinct_modes():
    row = leg(fair=0.57, price=0.30, model_conf=0.80)
    assert assess_leg(row, "best").qualified is True
    assert assess_leg(row, "longshot").qualified is True
