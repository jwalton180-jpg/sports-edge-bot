from dataclasses import replace

from sports_edge.models.model_evidence import ModelEvidence, model_dominant_fair, team_record_model
from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.kalshi_model_candidates import attach_sportsbook_context


def _leg(*, fair=0.62, books=0, age=0.0, selection="Team A"):
    return ParlayCandidateLeg(
        sport="NFL",
        event_id="E1",
        event_title="Team B @ Team A",
        market_key="model_h2h",
        market_label="Moneyline",
        selection=selection,
        consensus_probability=fair,
        book_count=books,
        source_age_s=age,
        median_odds=None,
        kalshi_ticker="KXNFLGAME-E1-A",
        kalshi_side="YES",
        kalshi_price=0.55,
        kalshi_edge_points=7.0,
        kalshi_status="MODEL",
        evidence_class="MODEL",
        model_probability=0.62,
        model_confidence=0.72,
        model_name="NFL public team-strength model",
        model_sample_size=8,
        model_reasons=("season record",),
    )


def test_model_dominant_fair_does_not_require_books():
    model = ModelEvidence(
        sport="Tennis",
        model_name="tennis test",
        fair_probability=0.64,
        confidence=0.75,
        sample_size=18,
        factors=("surface/form",),
    )
    fair, quality, reasons = model_dominant_fair(model)
    assert fair == 0.64
    assert quality > 0.5
    assert any("not required" in reason.lower() for reason in reasons)


def test_sportsbook_crosscheck_cannot_dominate_model():
    model = ModelEvidence(
        sport="MLB",
        model_name="mlb test",
        fair_probability=0.70,
        confidence=0.80,
        sample_size=80,
    )
    fair, _, reasons = model_dominant_fair(
        model,
        sportsbook_probability=0.20,
        sportsbook_book_count=5,
        sportsbook_age_s=10,
    )
    # Book weight is capped at 25%, so a wildly different market cannot
    # turn a model favorite into a book-driven underdog.
    assert fair >= 0.575
    assert fair < 0.70
    assert any("cross-check" in reason.lower() for reason in reasons)


def test_stale_sportsbook_crosscheck_is_ignored():
    model = ModelEvidence(
        sport="NBA",
        model_name="nba test",
        fair_probability=0.61,
        confidence=0.70,
        sample_size=25,
    )
    fair, _, _ = model_dominant_fair(
        model,
        sportsbook_probability=0.30,
        sportsbook_book_count=5,
        sportsbook_age_s=121,
    )
    assert fair == 0.61


def test_team_record_model_rewards_better_record_and_context():
    row = team_record_model(
        sport="MLB",
        team_a="A",
        team_b="B",
        win_pct_a=0.620,
        win_pct_b=0.450,
        games_a=100,
        games_b=100,
        home_a=True,
        recent_pct_a=0.70,
        recent_pct_b=0.40,
        differential_per_game_a=0.8,
        differential_per_game_b=-0.4,
    )
    assert row.usable
    assert row.fair_probability > 0.60
    assert row.confidence > 0.60


def test_attach_books_enriches_model_leg_but_preserves_model_probability():
    model_leg = _leg()
    book_leg = replace(
        _leg(fair=0.58, books=4, age=30),
        model_probability=None,
        model_confidence=0.0,
        model_name=None,
        model_sample_size=0,
        model_reasons=(),
        evidence_class="EDGE-QUALIFIED",
    )
    merged = attach_sportsbook_context([model_leg], [book_leg])
    assert len(merged) == 1
    row = merged[0]
    assert row.model_probability == 0.62
    assert row.consensus_probability == 0.58
    assert row.book_count == 4
    assert row.evidence_class == "MODEL + BOOKS"


def test_ambiguous_book_join_fails_closed_to_model_only():
    model_leg = _leg()
    book_a = replace(_leg(fair=0.58, books=4), model_probability=None, model_name=None)
    book_b = replace(book_a, event_id="E2", event_title="Other Event")
    merged = attach_sportsbook_context([model_leg], [book_a, book_b])
    assert merged[0].book_count == 0
    assert merged[0].evidence_class == "MODEL"
