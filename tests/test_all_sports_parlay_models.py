from sports_edge.models import kalshi_model_candidates as kmc
from sports_edge.models.kalshi_sports import (
    KalshiSportMarket,
    classify_kalshi_market,
    group_kalshi_sports,
)
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models.player_prop_models import (
    basketball_prop_evidence,
    mlb_prop_evidence,
    nfl_prop_evidence,
)


def market(
    ticker,
    *,
    family_series=None,
    title="Player A: 1+ hits?",
    event=None,
    floor=0.5,
    yes=0.40,
    no=0.60,
):
    series = family_series or ticker.split("-", 1)[0]
    return {
        "ticker": ticker,
        "series_ticker": series,
        "event_ticker": event or ticker.rsplit("-", 1)[0],
        "event_title": "Underlying game",
        "title": title,
        "yes_sub_title": title.split("?", 1)[0],
        "no_sub_title": title.split("?", 1)[0],
        "floor_strike": floor,
        "strike_type": "greater",
        "yes_ask_dollars": yes,
        "no_ask_dollars": no,
    }


def evidence(sport="MLB", fair=0.62):
    return ModelEvidence(
        sport=sport,
        model_name=f"{sport} test player model",
        fair_probability=fair,
        confidence=0.72,
        sample_size=30,
        factors=("independent player history",),
    )


def test_current_nfl_rushing_and_reception_series_classify():
    assert classify_kalshi_market(
        market(
            "KXNFLRSHYDS-26SEP27KCMIA-KCEJOHNSON10-25",
            title="Emmett Johnson: 25+ rushing yards",
            event="KXNFLRSHYDS-26SEP27KCMIA",
            floor=24.5,
        )
    ) == ("NFL", "Rushing Yards")
    assert classify_kalshi_market(
        market(
            "KXNFLREC-26SEP27KCMIA-MIAMWASHINGTON6-5",
            title="Malik Washington: 5+ receptions",
            event="KXNFLREC-26SEP27KCMIA",
            floor=4.5,
        )
    ) == ("NFL", "Receptions")
    assert classify_kalshi_market(
        market(
            "KXNFLRSHATT-26SEP27NYJDET-DETJGIBBS0-18",
            title="Jahmyr Gibbs: 18+ rushing attempts",
            event="KXNFLRSHATT-26SEP27NYJDET",
            floor=17.5,
        )
    ) == ("NFL", "Rushing Attempts")


def test_prop_series_share_underlying_game_correlation_key():
    hit = market(
        "KXMLBHIT-26SEP251605BALNYYG1-NYYBRICE22-1",
        title="Ben Rice: 1+ hits?",
        event="KXMLBHIT-26SEP251605BALNYYG1",
    )
    ks = market(
        "KXMLBKS-26SEP251605BALNYYG1-BALTROGERS28-7",
        title="Trevor Rogers: 7+ strikeouts?",
        event="KXMLBKS-26SEP251605BALNYYG1",
        floor=6.5,
    )
    assert kmc.canonical_game_key(hit) == kmc.canonical_game_key(ks)
    assert kmc.canonical_game_key(hit) == "26SEP251605BALNYYG1"


def test_mlb_hits_preset_builds_model_yes_and_no_candidates(monkeypatch):
    row = market(
        "KXMLBHIT-26SEP251605BALNYYG1-NYYBRICE22-1",
        title="Ben Rice: 1+ hits?",
        event="KXMLBHIT-26SEP251605BALNYYG1",
        yes=0.50,
        no=0.50,
    )
    grouped = group_kalshi_sports([row])
    monkeypatch.setattr(kmc, "player_prop_evidence", lambda *a, **k: evidence("MLB", 0.62))

    candidates = kmc.model_candidates_for_preset(
        grouped,
        sport_filter="MLB",
        preset="MLB Hits",
    )
    assert len(candidates) == 2
    yes = next(x for x in candidates if x.kalshi_side == "YES")
    no = next(x for x in candidates if x.kalshi_side == "NO")
    assert yes.model_probability == 0.62
    assert no.model_probability == 0.38
    assert yes.selection == "Ben Rice 1+ Hits"
    assert "under 1 Hits" in no.selection
    assert yes.event_id == no.event_id == "26SEP251605BALNYYG1"


def test_preset_contract_count_distinguishes_no_contract_from_no_model():
    grouped = group_kalshi_sports(
        [
            market(
                "KXNBAGAME-26OCT20PHINYK-PHI",
                title="Philadelphia wins",
                event="KXNBAGAME-26OCT20PHINYK",
                floor=None,
            )
        ]
    )
    assert kmc.preset_contract_count(
        grouped, sport_filter="NBA", preset="NBA Points"
    ) == 0
    assert kmc.preset_contract_count(
        grouped, sport_filter="NBA", preset="Best Available"
    ) > 0


def test_every_advertised_player_preset_has_model_family_mapping():
    expected = {
        "MLB Hits", "MLB Home Runs", "MLB Strikeouts",
        "NFL Passing", "NFL Rushing", "NFL Receiving", "NFL Touchdowns",
        "NBA Points", "NBA Rebounds", "NBA Assists", "NBA Threes", "NBA PRA",
        "WNBA Points", "WNBA Rebounds", "WNBA Assists", "WNBA Threes", "WNBA PRA",
    }
    assert expected <= set(kmc.PRESET_MODEL_FAMILIES)


def test_mlb_hit_model_uses_player_performance_not_market_price(monkeypatch):
    row = {
        "gamesPlayed": 100,
        "hits": 120,
        "atBats": 400,
        "plateAppearances": 450,
        "homeRuns": 20,
    }
    monkeypatch.setattr(
        "sports_edge.models.player_prop_models._mlb_index",
        lambda group, season: {"ben rice": row},
    )
    ev = mlb_prop_evidence("Ben Rice", "Hits", line=0.5, season=2026)
    assert ev is not None
    assert ev.usable
    assert 0 < ev.fair_probability < 1
    assert "hit-rate" in ev.model_name.lower()


def test_nfl_passing_yards_blends_current_and_prior_game_logs(monkeypatch):
    current = tuple({"passing_yards": v} for v in (240, 260, 280))
    prior = tuple({"passing_yards": v} for v in (210, 225, 250, 270, 300, 235))
    monkeypatch.setattr(
        "sports_edge.models.player_prop_models._nfl_player_rows",
        lambda year, player: current if year == 2026 else prior,
    )
    ev = nfl_prop_evidence("Quarterback", "Passing Yards", line=249.5, season=2026)
    assert ev is not None
    assert ev.sample_size == 9
    assert ev.confidence >= 0.4
    assert any("2025 stabilizer" in factor for factor in ev.factors)


def test_wnba_points_model_uses_current_game_distribution(monkeypatch):
    rows = tuple({"points": v} for v in (18, 24, 27, 31, 22, 29, 26, 30))
    monkeypatch.setattr(
        "sports_edge.models.player_prop_models._basketball_player_rows",
        lambda sport, player: rows,
    )
    ev = basketball_prop_evidence("WNBA", "Player", "Points", line=24.5)
    assert ev is not None
    assert ev.usable
    assert ev.sample_size == 8
    assert not ev.warnings


def test_nba_prior_season_prop_model_is_confidence_capped(monkeypatch):
    rows = tuple({"points": v} for v in (18, 24, 27, 31, 22, 29, 26, 30))
    monkeypatch.setattr(
        "sports_edge.models.player_prop_models._basketball_player_rows",
        lambda sport, player: rows,
    )
    ev = basketball_prop_evidence("NBA", "Player", "Points", line=24.5)
    assert ev is not None
    assert ev.confidence <= 0.48
    assert any("prior-season" in warning for warning in ev.warnings)
