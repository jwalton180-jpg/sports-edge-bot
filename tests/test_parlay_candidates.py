from datetime import datetime, timezone

from sports_edge.models.game_scope import GameEvent
from sports_edge.models.parlay_candidates import (
    candidate_legs_from_props,
    combo_blueprint,
    generate_candidate_parlay,
)
from sports_edge.models.props import PropConsensus


def game(event_id: str, home: str, away: str, sport: str = "MLB") -> GameEvent:
    return GameEvent(
        event_id=event_id,
        sport_key="baseball_mlb" if sport == "MLB" else "americanfootball_nfl",
        sport=sport,
        home_team=home,
        away_team=away,
        commence_time=datetime(2026, 9, 25, 1, 0, tzinfo=timezone.utc),
        state="SOON",
    )


def quote(player: str, p: float, label: str = "Hits", key: str = "batter_hits") -> PropConsensus:
    return PropConsensus(
        market_key=key,
        market_label=label,
        player=player,
        side="Over",
        line=0.5,
        fair_probability=p,
        book_count=4,
        median_age_s=12,
        median_price=-145,
        warnings=(),
    )


def test_consensus_only_props_can_generate_parlay_without_kalshi_edge():
    g1 = game("g1", "Yankees", "Red Sox")
    g2 = game("g2", "Dodgers", "Giants")
    rows = []
    rows.extend(candidate_legs_from_props(g1, [quote("Aaron Judge", 0.68)], [], mode="high_confidence"))
    rows.extend(candidate_legs_from_props(g2, [quote("Shohei Ohtani", 0.66)], [], mode="high_confidence"))

    assert len(rows) == 2
    assert all(r.kalshi_status == "CONSENSUS ONLY" for r in rows)

    parlay = generate_candidate_parlay(rows, target_legs=2, mode="high_confidence")
    assert len(parlay.legs) == 2
    assert parlay.estimated_independent_probability > 0
    assert parlay.correlation_risk == "LOW"


def test_generator_limits_same_game_concentration():
    g = game("g1", "Yankees", "Red Sox")
    rows = candidate_legs_from_props(
        g,
        [
            quote("Aaron Judge", 0.68),
            quote("Juan Soto", 0.66),
            quote("Giancarlo Stanton", 0.64),
        ],
        [],
        mode="high_confidence",
    )
    parlay = generate_candidate_parlay(rows, target_legs=3, mode="high_confidence", max_per_event=2)
    assert len(parlay.legs) == 2
    assert parlay.correlation_risk == "MEDIUM"
    assert any("Only 2 usable" in w for w in parlay.warnings)


def test_longshot_mode_accepts_lower_probability_td_style_leg():
    g = game("n1", "Chiefs", "Bills", sport="NFL")
    td = PropConsensus(
        market_key="player_anytime_td",
        market_label="Anytime TD",
        player="Player One",
        side="Yes",
        line=None,
        fair_probability=0.42,
        book_count=3,
        median_age_s=9,
        median_price=140,
        warnings=(),
    )
    assert candidate_legs_from_props(g, [td], [], mode="high_confidence") == []
    rows = candidate_legs_from_props(g, [td], [], mode="longshot")
    assert len(rows) == 1


def test_combo_blueprint_is_copyable_without_exact_kalshi_ticker():
    g = game("g1", "Yankees", "Red Sox")
    rows = candidate_legs_from_props(g, [quote("Aaron Judge", 0.68)], [], mode="high_confidence")
    text = combo_blueprint(rows)
    assert "KALSHI COMBO BLUEPRINT" in text
    assert "Aaron Judge" in text
    assert "Find matching component in Kalshi Combo Builder" in text
