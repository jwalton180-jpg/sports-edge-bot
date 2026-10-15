from dataclasses import replace
from sports_edge.models.combo_archetypes import combo_pattern
from sports_edge.models.parlay_diagnostics import parlay_evidence_diagnostics
from tests.test_parlay_intelligence import leg


def test_slip_patterns_are_hypotheses_not_predicted_win_rates():
    examples=[
        ("NBA","spread","BOS Celtics win by over 18.5 points","", "19+"),
        ("NBA","spread","Toronto win by over 30.5 points","", "31+"),
        ("NBA","game_total","NO Over 244.5 points","Game Total","high game-total"),
        ("NFL","player_anytime_td","Brian Robinson: 2+ touchdowns","Player Touchdowns","scorer tail"),
        ("NFL","player_receptions","Juwan Johnson: 3+ receptions","Receptions","volume ladder"),
        ("NHL","team_total","Over 1.5 goals","Team Total","team-goals"),
        ("NCAAF","model_h2h","Georgia to win","Moneyline","upset-risk"),
        ("Soccer","game_total","No Over 5.5 goals","Game Total","extreme-total"),
        ("Tennis","match_games_total","Over 16.5 games","Games Total","match-games"),
        ("MLB","game_total","Over 2.5 runs","Game Total","run thresholds"),
    ]
    for sport,key,selection,label,wanted in examples:
        assert wanted.lower() in combo_pattern(sport,key,selection,label).lower()
    assert "Other" in combo_pattern("Esports","winner","Team wins")
    assert "Other" in combo_pattern("KBO","h2h","Team wins")


def test_priced_research_preview_shows_pattern_without_promoting_it():
    candidate=replace(leg(event_id="E1",selection="BOS Celtics win by over 18.5 points",
                          fair=.3,price=.13,sport="NBA"),
                market_key="spread",market_label="Spread",
                entry_fee_per_contract=.01,execution_audit_status="RESEARCH_ONLY",
                execution_audit_reasons=("prospective_validation_required",))
    d=parlay_evidence_diagnostics([candidate])
    assert d["qualified"]==0
    assert d["research_preview"][0]["Review"]=="RESEARCH ONLY"
    assert "19+" in d["research_preview"][0]["Pattern hypothesis"]
