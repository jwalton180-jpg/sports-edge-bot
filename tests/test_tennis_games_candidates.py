from datetime import datetime, timezone
from types import SimpleNamespace

from sports_edge.models.kalshi_model_candidates import (
    _tennis_best_of,
    _tennis_games_total_candidates,
)
from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.model_evidence import ModelEvidence


def _match_row(player, opponent):
    return KalshiSportMarket(
        sport="Tennis",
        family="Match Winner",
        market={
            "series_ticker": "KXATPMATCH",
            "event_ticker": "KXATPMATCH-99JAN01ALPBET",
            "ticker": f"KXATPMATCH-99JAN01ALPBET-{player[:3].upper()}",
            "yes_sub_title": player,
            "no_sub_title": opponent,
            "yes_ask_dollars": "0.55",
            "no_ask_dollars": "0.46",
        },
    )


def _total_row(*, start="2099-01-02T05:00:00Z", rules=True):
    market = {
        "series_ticker": "KXATPGTOTAL",
        "event_ticker": "KXATPGTOTAL-99JAN01ALPBET",
        "ticker": "KXATPGTOTAL-99JAN01ALPBET-23",
        "title": "Over 22.5 games",
        "yes_sub_title": "Over 22.5 games",
        "floor_strike": 22.5,
        "occurrence_datetime": start,
        "yes_ask_dollars": "0.40",
        "no_ask_dollars": "0.61",
    }
    if rules:
        market["rules_primary"] = (
            "If the number of completed games is above 22.5 in the "
            "professional tennis match in the 2099 ATP Beijing Round Of 32."
        )
    return KalshiSportMarket(sport="Tennis", family="Games Total", market=market)


class _FakeGamesModel:
    def project_games_total(self, player_a, player_b, **kwargs):
        assert {player_a, player_b} == {"Alpha Player", "Beta Player"}
        assert kwargs["line"] == 22.5
        assert kwargs["best_of"] == 3
        return SimpleNamespace(
            evidence=ModelEvidence(
                sport="Tennis",
                model_name="test tennis total",
                fair_probability=0.64,
                confidence=0.70,
                sample_size=100,
                factors=("independent score distribution",),
            ),
            market_key="tennis_games_total",
            market_label="Games Total",
        )


def test_games_total_uses_sibling_match_identity_and_emits_both_sides(monkeypatch):
    monkeypatch.setattr(
        "sports_edge.models.kalshi_model_candidates._tennis_games_model",
        lambda gender, year: _FakeGamesModel(),
    )
    rows = [
        _match_row("Alpha Player", "Beta Player"),
        _match_row("Beta Player", "Alpha Player"),
        _total_row(),
    ]

    built = _tennis_games_total_candidates(
        rows,
        now=datetime(2099, 1, 1, 0, 0, tzinfo=timezone.utc),
    )

    assert len(built) == 2
    assert {row.selection for row in built} == {"Over 22.5 Games", "Under 22.5 Games"}
    assert built[0].event_id == built[1].event_id
    assert "alpha player" in built[0].event_id
    assert "beta player" in built[0].event_id
    yes = next(row for row in built if row.kalshi_side == "YES")
    no = next(row for row in built if row.kalshi_side == "NO")
    assert yes.model_probability == 0.64
    assert no.model_probability == 0.36
    assert yes.kalshi_price == 0.40
    assert no.kalshi_price == 0.61


def test_games_total_refuses_when_scheduled_start_has_arrived(monkeypatch):
    monkeypatch.setattr(
        "sports_edge.models.kalshi_model_candidates._tennis_games_model",
        lambda gender, year: _FakeGamesModel(),
    )
    rows = [
        _match_row("Alpha Player", "Beta Player"),
        _match_row("Beta Player", "Alpha Player"),
        _total_row(start="2099-01-01T00:00:00Z"),
    ]
    assert _tennis_games_total_candidates(
        rows,
        now=datetime(2099, 1, 1, 0, 0, tzinfo=timezone.utc),
    ) == []


def test_games_total_refuses_ambiguous_mens_match_format(monkeypatch):
    called = False

    def fake_model(gender, year):
        nonlocal called
        called = True
        return _FakeGamesModel()

    monkeypatch.setattr(
        "sports_edge.models.kalshi_model_candidates._tennis_games_model",
        fake_model,
    )
    rows = [
        _match_row("Alpha Player", "Beta Player"),
        _match_row("Beta Player", "Alpha Player"),
        _total_row(rules=False),
    ]
    assert _tennis_games_total_candidates(
        rows,
        now=datetime(2099, 1, 1, 0, 0, tzinfo=timezone.utc),
    ) == []
    assert not called


def test_mens_grand_slam_format_is_explicitly_best_of_five():
    market = {
        "rules_primary": (
            "Professional tennis match in the 2099 Wimbledon Round Of 32."
        )
    }
    assert _tennis_best_of(market, "men") == 5


def test_mens_grand_slam_qualifying_stays_best_of_three():
    market = {
        "rules_primary": (
            "Professional tennis match in the 2099 US Open Qualifying Round."
        )
    }
    assert _tennis_best_of(market, "men") == 3


def test_womens_match_format_is_best_of_three_without_extra_metadata():
    assert _tennis_best_of({}, "women") == 3


def test_games_total_does_not_use_expiration_as_pregame_proof(monkeypatch):
    monkeypatch.setattr(
        "sports_edge.models.kalshi_model_candidates._tennis_games_model",
        lambda gender, year: _FakeGamesModel(),
    )
    total = _total_row()
    total.market.pop("occurrence_datetime")
    total.market["expected_expiration_time"] = "2099-01-03T05:00:00Z"
    rows = [
        _match_row("Alpha Player", "Beta Player"),
        _match_row("Beta Player", "Alpha Player"),
        total,
    ]
    assert _tennis_games_total_candidates(
        rows,
        now=datetime(2099, 1, 1, 0, 0, tzinfo=timezone.utc),
    ) == []
