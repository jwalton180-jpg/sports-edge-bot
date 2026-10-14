from dataclasses import replace
import pytest

from sports_edge.models.parlay_diagnostics import parlay_evidence_diagnostics
from tests.test_parlay_intelligence import leg


def test_mixed_sport_empty_ticket_shows_blocking_evidence_and_research_not_picks():
    first=replace(leg(selection="Tennis B",fair=.37,price=.17),sport="Tennis",
        execution_audit_status="RESEARCH_ONLY",
        execution_audit_reasons=("scoped_prospective_model_validation_required",),
        entry_fee_per_contract=.01)
    second=replace(leg(selection="NBA C",fair=.65,price=.50),sport="NBA",
        execution_audit_status="RESEARCH_ONLY",
        execution_audit_reasons=("confirmed_starters_and_injuries",),
        entry_fee_per_contract=.02)
    out=parlay_evidence_diagnostics([first,second])
    assert out["candidates"]==2
    assert out["modeled"]==2
    assert out["priced"]==2
    assert out["modeled_and_fee_priced"]==2
    assert out["qualified"]==0
    assert out["sports"]=={"Tennis":1,"NBA":1}
    assert out["research_preview"][0]["Review"]=="RESEARCH ONLY"
    assert out["research_preview"][0]["Indicative net edge"]>0
    assert {reason for reason,_ in out["common_blockers"]}=={
        "confirmed_starters_and_injuries","scoped_prospective_model_validation_required"}


def test_missing_fee_or_probability_never_gets_imaginary_net_edge():
    unpriced=replace(leg(fair=.7,price=.2),entry_fee_per_contract=None,
        execution_audit_status="RESEARCH_ONLY")
    unavailable=replace(unpriced,model_probability=None)
    out=parlay_evidence_diagnostics([unpriced,unavailable])
    assert out["modeled"]==1
    assert out["modeled_and_fee_priced"]==0
    assert out["research_preview"][0]["Indicative net edge"] is None
    assert out["qualified"]==0
    with pytest.raises(ValueError):
        parlay_evidence_diagnostics([],max_preview=51)


def test_all_sport_empty_ticket_renders_diagnostics_without_waiving_audit():
    from pathlib import Path
    source=Path("sports_edge/app.py").read_text()
    assert 'evidence = parlay_evidence_diagnostics(state.get("model_candidate_rows") or ())' in source
    assert "return audit_app_candidates(candidates, inventory" in source
    assert "not validated picks" in source.lower() or "not validated" in source.lower()
