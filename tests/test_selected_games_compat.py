from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.parlay_intelligence import assess_leg
from sports_edge.models.selected_games_compat import selected_games_compat_candidates


def leg(
    event_id: str,
    selection: str,
    *,
    fair: float,
    price: float,
    market_key: str = "model_h2h",
) -> ParlayCandidateLeg:
    return ParlayCandidateLeg(
        sport="MLB",
        event_id=event_id,
        event_title=f"Event {event_id}",
        market_key=market_key,
        market_label="Market",
        selection=selection,
        consensus_probability=fair,
        book_count=0,
        source_age_s=0.0,
        median_odds=None,
        kalshi_ticker=f"KX-{event_id}-{selection}",
        kalshi_side="YES",
        kalshi_price=price,
        kalshi_edge_points=100.0 * (fair - price),
        kalshi_status="MODEL",
        evidence_class="MODEL",
        model_probability=fair,
        model_confidence=0.78,
        model_name="test model",
        model_sample_size=20,
        model_reasons=("independent sport features",),
    )


def compat(rows, preferred, target):
    return selected_games_compat_candidates(
        rows,
        preferred_event_ids=preferred,
        mode="best",
        target_legs=target,
        max_per_event=3,
        assess=assess_leg,
    )


def test_compat_core_preserves_each_qualified_selected_game_then_fills_extra():
    rows = [
        leg("E1", "E1 strongest", fair=0.76, price=0.60),
        leg("E1", "E1 extra", fair=0.74, price=0.59),
        leg("E2", "E2 strongest", fair=0.74, price=0.58),
        leg("E2", "E2 extra", fair=0.72, price=0.57),
        leg("E3", "E3 strongest", fair=0.66, price=0.59),
        leg("E4", "E4 qualified", fair=0.58, price=0.54),
    ]
    picked = compat(rows, ["E1", "E2", "E3", "E4"], 5)
    assert len(picked) == 5
    assert {row.event_id for row in picked} == {"E1", "E2", "E3", "E4"}


def test_compat_core_never_forces_an_unqualified_selected_game():
    rows = [
        leg("E1", "E1 A", fair=0.70, price=0.60),
        leg("E1", "E1 B", fair=0.68, price=0.59),
        leg("E2", "E2 A", fair=0.68, price=0.58),
        leg("E2", "E2 B", fair=0.66, price=0.57),
        leg("E3", "E3 A", fair=0.65, price=0.57),
        leg("E4", "E4 bad price", fair=0.50, price=0.56),
    ]
    picked = compat(rows, ["E1", "E2", "E3", "E4"], 5)
    assert len(picked) == 5
    assert "E4" not in {row.event_id for row in picked}


def test_compat_core_uses_stability_first_ranking_within_each_game():
    volatile = leg(
        "E1",
        "Home run",
        fair=0.70,
        price=0.50,
        market_key="batter_home_runs",
    )
    stable = leg(
        "E1",
        "One plus hit",
        fair=0.64,
        price=0.55,
        market_key="batter_hits",
    )
    picked = compat([volatile, stable], ["E1"], 1)
    assert [row.selection for row in picked] == ["One plus hit"]


def test_compat_core_uses_strongest_games_when_coverage_exceeds_target():
    rows = [
        leg("E4", "Weakest", fair=0.56, price=0.54),
        leg("E3", "Third", fair=0.62, price=0.56),
        leg("E2", "Second", fair=0.70, price=0.58),
        leg("E1", "Strongest", fair=0.74, price=0.58),
    ]
    picked = compat(rows, ["E4", "E3", "E2", "E1"], 2)
    assert {row.event_id for row in picked} == {"E1", "E2"}
