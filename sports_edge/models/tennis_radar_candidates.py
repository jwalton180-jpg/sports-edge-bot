"""Model inputs for a read-only LIVE Tennis reversal radar, never ticket picks.

The app-level ticket audit requires a pregame start, an exhaustive catalog,
and reviewed forward model promotions. Live reversal observations instead
receive dedicated score, freshness, survival and executable-price gates in
tennis_live_reversal. Applying the pregame audit to this UI crashes or removes
the very live candidates it exists to inspect.

This helper returns model candidates ONLY. It does not qualify bets, place
orders or bypass the ticket builder's audit.
"""
from __future__ import annotations

from sports_edge.models.kalshi_model_candidates import (
    model_candidates_from_kalshi as _independent_candidates,
)


def live_tennis_radar_candidates(grouped):
    tennis_rows = tuple(grouped.get("Tennis") or ())
    if not tennis_rows:
        return []
    # Enforce the exact sport scope even if a caller accidentally passes
    # unrelated grouped inventory. No mutable shared ticket-audit state.
    raw = _independent_candidates({"Tennis": list(tennis_rows)}, sport_filter="Tennis",
                                  include_tennis_games_total=False)
    return [
        row for row in raw
        if row.sport == "Tennis"
        and row.market_key == "model_h2h"
        and row.kalshi_ticker
        and row.kalshi_side in ("YES", "NO")
        and row.model_probability is not None
    ]
