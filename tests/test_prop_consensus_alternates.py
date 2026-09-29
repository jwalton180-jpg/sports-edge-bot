from datetime import datetime, timezone

from sports_edge.models.props import MARKET_LABELS, prop_consensus


NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


def _payload():
    books = []
    for idx in range(3):
        books.append(
            {
                "key": f"book{idx}",
                "last_update": "2026-09-29T11:59:30Z",
                "markets": [
                    {
                        "key": "player_pass_attempts_alternate",
                        "last_update": "2026-09-29T11:59:30Z",
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "Josh Allen",
                                "price": -110,
                                "point": 34.5,
                            },
                            {
                                "name": "Under",
                                "description": "Josh Allen",
                                "price": -110,
                                "point": 34.5,
                            },
                        ],
                    },
                    {
                        "key": "player_rush_reception_yds_alternate",
                        "last_update": "2026-09-29T11:59:30Z",
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "James Cook",
                                "price": -105,
                                "point": 89.5,
                            },
                            {
                                "name": "Under",
                                "description": "James Cook",
                                "price": -115,
                                "point": 89.5,
                            },
                        ],
                    },
                ],
            }
        )
    return {"bookmakers": books}


def test_nfl_alternate_market_labels_match_model_labels():
    assert MARKET_LABELS["player_pass_attempts_alternate"] == "Pass Attempts"
    assert MARKET_LABELS["player_pass_completions_alternate"] == "Pass Completions"
    assert MARKET_LABELS["player_pass_interceptions_alternate"] == "Pass Interceptions"
    assert MARKET_LABELS["player_rush_attempts_alternate"] == "Rush Attempts"
    assert (
        MARKET_LABELS["player_rush_reception_yds_alternate"]
        == "Rushing + Receiving Yards"
    )


def test_prop_consensus_accepts_nfl_alternate_milestone_markets():
    rows = prop_consensus(
        _payload(),
        market_keys=(
            "player_pass_attempts_alternate",
            "player_rush_reception_yds_alternate",
        ),
        now=NOW,
    )
    assert rows
    by_key = {(row.market_key, row.player, row.side, row.line): row for row in rows}
    assert (
        "player_pass_attempts_alternate",
        "Josh Allen",
        "Over",
        34.5,
    ) in by_key
    assert (
        "player_rush_reception_yds_alternate",
        "James Cook",
        "Over",
        89.5,
    ) in by_key
    assert all(row.book_count == 3 for row in rows)
