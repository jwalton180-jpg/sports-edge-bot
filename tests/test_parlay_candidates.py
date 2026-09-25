from datetime import datetime, timedelta, timezone

from sports_edge.models.game_scope import build_game_events
from sports_edge.models.parlay_candidates import (
    candidate_legs_from_h2h,
    candidate_legs_from_props,
    combo_blueprint,
    generate_candidate_parlay,
)
from sports_edge.models.props import PropConsensus


NOW = datetime(2026, 9, 24, 20, 0, tzinfo=timezone.utc)


def make_game(event_id="g1", home="New York Yankees", away="Boston Red Sox", sport="MLB"):
    key = {
        "MLB": "baseball_mlb",
        "NFL": "americanfootball_nfl",
        "Tennis": "tennis_wta_singapore",
    }[sport]
    raw = {
        "id": event_id,
        "home_team": home,
        "away_team": away,
        "commence_time": (NOW + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
    }
    return build_game_events([raw], sport_key=key, sport=sport, now=NOW)[0]


def prop(player, p, market_key="batter_hits", label="Hits", side="Over", line=0.5, books=3):
    return PropConsensus(
        market_key=market_key,
        market_label=label,
        player=player,
        side=side,
        line=line,
        fair_probability=p,
        book_count=books,
        median_age_s=12,
        median_price=-130,
        warnings=(),
    )


def h2h_event():
    ts = (NOW - timedelta(seconds=10)).isoformat().replace("+00:00", "Z")
    return {
        "id": "nfl1",
        "home_team": "Kansas City Chiefs",
        "away_team": "Buffalo Bills",
        "commence_time": (NOW + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "bookmakers": [
            {
                "key": f"book{i}",
                "last_update": ts,
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": ts,
                        "outcomes": [
                            {"name": "Kansas City Chiefs", "price": -150},
                            {"name": "Buffalo Bills", "price": 130},
                        ],
                    }
                ],
            }
            for i in range(3)
        ],
    }


def test_prop_consensus_can_generate_without_any_kalshi_match():
    game = make_game()
    quotes = [
        prop("Aaron Judge", 0.68),
        prop("Juan Soto", 0.64),
        prop("Giancarlo Stanton", 0.61),
    ]
    candidates = candidate_legs_from_props(game, quotes, [], mode="high_confidence")
    assert len(candidates) == 3
    assert all(c.kalshi_ticker is None for c in candidates)

    parlay = generate_candidate_parlay(candidates, target_legs=2, mode="high_confidence", max_per_event=2)
    assert len(parlay.legs) == 2
    assert parlay.estimated_independent_probability > 0


def test_moneyline_consensus_can_generate_without_any_kalshi_match():
    game = make_game(
        event_id="nfl1",
        home="Kansas City Chiefs",
        away="Buffalo Bills",
        sport="NFL",
    )
    candidates = candidate_legs_from_h2h(game, h2h_event(), [], mode="high_confidence", now=NOW)
    assert len(candidates) == 1
    assert candidates[0].selection == "Kansas City Chiefs"
    assert candidates[0].kalshi_ticker is None


def test_generator_prefers_distinct_games_before_same_game_extra_legs():
    g1 = make_game("g1")
    g2 = make_game("g2", home="Los Angeles Dodgers", away="San Diego Padres")
    c1 = candidate_legs_from_props(g1, [prop("Aaron Judge", 0.68), prop("Juan Soto", 0.65)], [], mode="high_confidence")
    c2 = candidate_legs_from_props(g2, [prop("Shohei Ohtani", 0.67)], [], mode="high_confidence")
    p = generate_candidate_parlay(c1 + c2, target_legs=2, mode="high_confidence", max_per_event=1)
    assert len(p.legs) == 2
    assert len({x.event_id for x in p.legs}) == 2


def test_combo_blueprint_marks_unmatched_legs_for_combo_builder_search():
    game = make_game()
    leg = candidate_legs_from_props(game, [prop("Aaron Judge", 0.68)], [], mode="high_confidence")[0]
    text = combo_blueprint([leg])
    assert "Find matching component in Kalshi Combo Builder" in text
    assert "Aaron Judge" in text



def test_consensus_only_leg_is_explicitly_not_edge_qualified():
    game = make_game()
    leg = candidate_legs_from_props(game, [prop("Aaron Judge", 0.68)], [], mode="high_confidence")[0]
    assert leg.evidence_class == "CONSENSUS BASELINE"
    assert leg.kalshi_edge_points is None


def test_edge_builder_does_not_fill_with_consensus_favorites():
    game = make_game()
    candidates = candidate_legs_from_props(
        game,
        [prop("Aaron Judge", 0.70), prop("Juan Soto", 0.66)],
        [],
        mode="edge",
    )
    p = generate_candidate_parlay(
        candidates,
        target_legs=2,
        mode="edge",
        require_edge=True,
    )
    assert p.legs == ()
    assert any("edge-qualified" in warning for warning in p.warnings)


def test_tennis_moneyline_candidates_remain_tennis():
    game = make_game(
        event_id="ten1",
        home="Player Alpha",
        away="Player Beta",
        sport="Tennis",
    )
    ts = (NOW - timedelta(seconds=8)).isoformat().replace("+00:00", "Z")
    payload = {
        "id": "ten1",
        "home_team": "Player Alpha",
        "away_team": "Player Beta",
        "bookmakers": [
            {
                "key": f"book{i}",
                "last_update": ts,
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": ts,
                        "outcomes": [
                            {"name": "Player Alpha", "price": -175},
                            {"name": "Player Beta", "price": 150},
                        ],
                    }
                ],
            }
            for i in range(3)
        ],
    }
    rows = candidate_legs_from_h2h(game, payload, [], mode="high_confidence", now=NOW)
    assert rows
    assert all(row.sport == "Tennis" for row in rows)
    assert all("Yankees" not in row.event_title for row in rows)


def test_mixed_sports_diversification_uses_multiple_sports_when_available():
    mlb_game = make_game("m1", sport="MLB")
    nfl_game = make_game("n1", home="Kansas City Chiefs", away="Buffalo Bills", sport="NFL")
    tennis_game = make_game("t1", home="Player Alpha", away="Player Beta", sport="Tennis")

    mlb = candidate_legs_from_props(mlb_game, [prop("Aaron Judge", 0.68)], [], mode="high_confidence")[0]
    nfl = candidate_legs_from_props(
        nfl_game,
        [prop("Runner One", 0.64, market_key="player_rush_yds", label="Rushing Yards")],
        [],
        mode="high_confidence",
    )[0]

    ts = (NOW - timedelta(seconds=5)).isoformat().replace("+00:00", "Z")
    payload = {
        "bookmakers": [
            {
                "key": f"book{i}",
                "last_update": ts,
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": ts,
                        "outcomes": [
                            {"name": "Player Alpha", "price": -140},
                            {"name": "Player Beta", "price": 120},
                        ],
                    }
                ],
            }
            for i in range(3)
        ]
    }
    tennis = candidate_legs_from_h2h(tennis_game, payload, [], mode="high_confidence", now=NOW)[0]

    p = generate_candidate_parlay(
        [mlb, nfl, tennis],
        target_legs=3,
        mode="high_confidence",
        diversify_sports=True,
        max_per_event=1,
    )
    assert {leg.sport for leg in p.legs} == {"MLB", "NFL", "Tennis"}
