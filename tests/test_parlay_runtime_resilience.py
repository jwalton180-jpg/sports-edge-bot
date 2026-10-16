from contextlib import nullcontext
from pathlib import Path

from sports_edge.models.parlay_runtime import guard_ticket_action


def test_failed_new_build_clears_stale_ticket_without_raising():
    state = {"intel_parlay_v3": {"legs": ["yesterday"]}}
    errors = []
    with guard_ticket_action(state, nullcontext(), errors.append, replace_previous=True):
        raise TimeoutError("do not display private provider URL")
    assert "intel_parlay_v3" not in state
    assert errors == ["TimeoutError"]


def test_optional_enrichment_failure_keeps_primary_ticket():
    original = {"legs": ["verified"]}
    state = {"intel_parlay_v3": original}
    errors = []
    with guard_ticket_action(state, nullcontext(), errors.append):
        raise ValueError("invalid sportsbook payload")
    assert state["intel_parlay_v3"] is original
    assert errors == ["ValueError"]


def test_successful_build_publishes_only_new_ticket():
    state = {"intel_parlay_v3": {"legs": ["old"]}}
    errors = []
    with guard_ticket_action(state, nullcontext(), errors.append, replace_previous=True):
        assert "intel_parlay_v3" not in state
        state["intel_parlay_v3"] = {"legs": ["new"]}
    assert state["intel_parlay_v3"]["legs"] == ["new"]
    assert not errors


def test_catalog_metadata_and_ticket_error_guards_wired_into_ui():
    app = Path("sports_edge/app.py").read_text()
    start = app.index("def get_parlay_kalshi_events():")
    end = app.index("@st.cache_data", start)
    helper = app[start:end]
    assert "with_nested_markets=False" in helper
    assert "return events, error" in helper
    assert "seen_cursors" in helper
    assert 'with guard_ticket_action(' in app
    assert "replace_previous=True" in app
    assert "def _parlay_build_failure(" in app
    assert 'return [], _safe_error(exc), False, ()' in app
