"""Fail-soft Streamlit ticket actions without promoting stale tickets.

No prices, model probabilities, or order/account state are changed here.
"""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def guard_ticket_action(session_state, progress, report_error, *, replace_previous=False):
    """Keep a provider/model failure from killing the parlay page.

    New builds clear the previous ticket immediately: a failed refresh must not
    leave a stale, apparently current recommendation. Optional cross-checks
    retain the existing independently built ticket when enrichment fails.
    """
    if replace_previous:
        session_state.pop("intel_parlay_v3", None)
    try:
        with progress:
            yield
    except Exception as exc:
        if replace_previous:
            session_state.pop("intel_parlay_v3", None)
        # Only surface the exception CLASS; raw API failures can contain keys.
        report_error(type(exc).__name__)
