from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.model_evidence import ModelEvidence
from sports_edge.models import kalshi_model_candidates as candidates


class _FakeWtaModel:
    def probability(self, player_a, player_b, **kwargs):
        assert {player_a, player_b} == {"Iga Swiatek", "Iva Jovic"}
        assert kwargs["level"] == "WTA Tour"
        return ModelEvidence(
            sport="Tennis",
            model_name="test WTA model",
            fair_probability=0.72,
            confidence=0.75,
            sample_size=20,
            factors=("test",),
        )


def _row(selection):
    event = "KXWTAMATCH-26OCT05JOVSWI"
    code = "SWI" if selection == "Iga Swiatek" else "JOV"
    return KalshiSportMarket(
        sport="Tennis",
        family="Match Winner",
        market={
            "event_ticker": event,
            "ticker": f"{event}-{code}",
            # Real live Kalshi rows can omit series_ticker.
            "yes_sub_title": selection,
            "no_sub_title": selection,
            "yes_ask_dollars": "0.50",
            "no_ask_dollars": "0.51",
            "title": f"{selection} wins",
        },
    )


def test_wta_match_model_infers_gender_from_ticker_when_series_ticker_missing(monkeypatch):
    calls = []

    def fake_model(gender, year):
        calls.append((gender, year))
        return _FakeWtaModel()

    monkeypatch.setattr(candidates, "_tennis_model", fake_model)
    built = candidates._tennis_event_candidates([
        _row("Iga Swiatek"),
        _row("Iva Jovic"),
    ])

    assert calls == [("women", 2026)]
    assert len(built) == 2
    assert {row.selection for row in built} == {"Iga Swiatek", "Iva Jovic"}
    assert all(row.model_name == "test WTA model" for row in built)
