"""The unattended collector must be importable, not just its model dependencies.

The full production gate previously compiled only the Streamlit app and package,
allowing a SyntaxError in scripts/tennis_reversal_worker.py to reach production.
"""
from importlib import import_module
from inspect import signature


def test_unattended_tennis_worker_imports_and_has_expected_entrypoint():
    worker = import_module("scripts.tennis_reversal_worker")
    assert callable(worker.main)
    assert callable(worker.observe_once)
    assert "store" in signature(worker.observe_once).parameters
    assert callable(worker._fresh_quote)
    assert callable(worker._match_state)
