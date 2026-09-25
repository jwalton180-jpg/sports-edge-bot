from unittest.mock import patch

from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.kalshi_model_candidates import model_candidates_from_kalshi
from sports_edge.models.model_evidence import ModelEvidence


class FakeTennisModel:
    def probability(self, player_a, player_b, **kwargs):
        return ModelEvidence(
            sport="Tennis",
            model_name="fake tennis model",
            fair_probability=0.64,
            confidence=0.75,
            sample_size=18,
            factors=("surface/form/serve-return fixture",),
        )


def tmarket(event, player, price):
    return KalshiSportMarket(
        sport="Tennis",
        family="Match Winner",
        market={
            "ticker": f"KXATPMATCH-{event}-{player.replace(' ', '').upper()}",
            "series_ticker": "KXATPMATCH",
            "event_ticker": event,
            "event_title": f"{event} tennis match",
            "yes_sub_title": player,
            "yes_ask_dollars": price,
            "no_ask_dollars": 1 - price,
            "close_time": "2026-09-26T12:00:00Z",
        },
    )


def test_multiple_kalshi_tennis_events_reach_model_candidate_pool():
    grouped = {
        "Tennis": [
            tmarket("E1", "Player A", 0.55),
            tmarket("E1", "Player B", 0.45),
            tmarket("E2", "Player C", 0.48),
            tmarket("E2", "Player D", 0.52),
            tmarket("E3", "Player E", 0.30),
            tmarket("E3", "Player F", 0.70),
        ]
    }
    with patch("sports_edge.models.kalshi_model_candidates._tennis_model", return_value=FakeTennisModel()):
        rows = model_candidates_from_kalshi(grouped, sport_filter="Tennis")

    assert len(rows) == 6
    assert {row.event_id for row in rows} == {"E1", "E2", "E3"}
    assert all(row.model_probability is not None for row in rows)
    assert all(row.book_count == 0 for row in rows)
