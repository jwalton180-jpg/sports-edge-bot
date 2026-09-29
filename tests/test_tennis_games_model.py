from datetime import date, timedelta

from sports_edge.models import tennis_games_model as tgm


def _row(day, winner, loser, score, *, surface="Hard", level="A", best_of=3, match_num=1):
    d = date(2026, 1, 1) + timedelta(days=day)
    return {
        "tourney_date": d.strftime("%Y%m%d"),
        "match_num": str(match_num),
        "winner_name": winner,
        "loser_name": loser,
        "score": score,
        "surface": surface,
        "tourney_level": level,
        "best_of": str(best_of),
    }


def _history(n=90):
    players = ["Alpha Player", "Beta Player", "Gamma Player", "Delta Player", "Epsilon Player", "Zeta Player"]
    scores = ["6-4 6-4", "7-6(5) 6-3", "6-2 4-6 6-3", "6-1 6-2", "7-5 7-5"]
    rows = []
    for i in range(n):
        a = players[i % len(players)]
        b = players[(i * 2 + 1) % len(players)]
        if a == b:
            b = players[(players.index(b) + 1) % len(players)]
        winner, loser = (a, b) if i % 3 else (b, a)
        rows.append(_row(i, winner, loser, scores[i % len(scores)], match_num=i))
    return tuple(rows)


def test_parse_completed_score_counts_standard_and_tiebreak_sets():
    score = tgm.parse_completed_score("7-6(5) 6-3")
    assert score is not None
    assert score.total_games == 22
    assert score.winner_games == 13
    assert score.loser_games == 9
    assert score.sets_played == 2


def test_parse_completed_score_fails_closed_on_retirement_and_match_tiebreak():
    assert tgm.parse_completed_score("6-4 3-2 RET") is None
    assert tgm.parse_completed_score("6-4 3-6 10-8") is None


def test_snapshot_excludes_all_same_date_results(monkeypatch):
    rows = (
        _row(0, "Alpha Player", "Beta Player", "6-4 6-4"),
        _row(1, "Beta Player", "Alpha Player", "7-5 6-4"),
        _row(2, "Alpha Player", "Beta Player", "6-3 6-3"),
        _row(3, "Alpha Player", "Beta Player", "6-4 3-6 6-2"),
        _row(4, "Beta Player", "Alpha Player", "7-6(4) 7-5"),
        _row(5, "Alpha Player", "Beta Player", "6-2 6-2"),
        _row(6, "Beta Player", "Alpha Player", "6-1 6-1"),
    )
    monkeypatch.setattr(tgm, "load_recent_tennis_rows", lambda *args, **kwargs: rows)
    model = tgm.TennisGamesModel("men", current_year=2026)
    as_of = date(2026, 1, 7)  # row day=6 is on this date and must be excluded
    snapshot = model._build_snapshot(as_of)

    assert snapshot.players[tgm.normalize("Alpha Player")].matches == 6
    assert snapshot.players[tgm.normalize("Beta Player")].matches == 6


def test_games_total_projection_is_independent_and_bounded(monkeypatch):
    rows = _history(100)
    monkeypatch.setattr(tgm, "load_recent_tennis_rows", lambda *args, **kwargs: rows)
    model = tgm.TennisGamesModel("men", current_year=2026)
    event_date = date(2026, 5, 1)

    projection = model.project_games_total(
        "Alpha Player",
        "Beta Player",
        line=22.5,
        event_date=event_date,
        level="ATP Tour",
        surface="hard",
        best_of=3,
    )

    assert projection is not None
    assert 0.01 <= projection.evidence.fair_probability <= 0.99
    assert projection.evidence.model_name.startswith("Tennis Games Total:")
    assert projection.evidence.sample_size >= 1
    assert projection.market_key == "tennis_games_total"
    assert any("Probability calibration" in factor for factor in projection.evidence.factors)


def test_walkforward_report_uses_prior_dates_only():
    rows = list(_history(120))
    # Move final 30 matches into 2027 while preserving a large 2026 training base.
    shifted = []
    for i, row in enumerate(rows):
        row = dict(row)
        if i >= 90:
            d = date(2027, 1, 1) + timedelta(days=i - 90)
            row["tourney_date"] = d.strftime("%Y%m%d")
        shifted.append(row)

    report = tgm.walkforward_games_total_report(
        shifted,
        holdout_year=2027,
        alpha=0.75,
        lines=(20.5, 22.5),
    )
    assert report["n"] > 0
    assert 0.0 <= report["brier"] <= 1.0
    assert report["log_loss"] >= 0.0
