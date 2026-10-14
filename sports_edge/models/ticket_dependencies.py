"""Conservative dependence checks; no calibrated joint model is assumed."""
import re


def incompatible(a, b):
    if a.kalshi_ticker and a.kalshi_ticker == b.kalshi_ticker:
        return "same_contract_redundant_or_opposite_sides"
    if a.event_id != b.event_id:
        return None
    # Existing prop selections name the same participant before the threshold.
    def player(leg):
        return re.split(r"\s+(?:Over|Under|\d+\+)",leg.selection, maxsplit=1,flags=re.I)[0].lower().strip()
    if a.market_key == b.market_key and player(a) == player(b):
        return "shared_participant_thresholds_redundant_or_conflicting"
    # Opponent winners and opposing game scripts may not coexist in a ticket.
    if a.market_key in {"model_h2h", "h2h"} and b.market_key == a.market_key:
        return "incompatible_event_winners"
    return None


def probability_bounds(probabilities):
    p = list(probabilities)
    return (max(0.,sum(p)-len(p)+1), min(p)) if p else (0.,0.)
