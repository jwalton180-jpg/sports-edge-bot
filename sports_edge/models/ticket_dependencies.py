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


def same_player_cross_market_dependence(a, b):
    """Flag strong but uncalibrated same-player cross-stat correlation.

    This is a risk label, not an exclusion or a fabricated joint probability.
    A 3+ receptions and 40+ yards combo may have both legs independently
    correct but its likelihood is not their simple product.
    """
    if not a.event_id or a.event_id != b.event_id or a.sport != b.sport:
        return False
    if a.market_key == b.market_key:
        return False
    def is_player_market(leg):
        key=str(leg.market_key or "").lower()
        return key.startswith(("player_","batter_","pitcher_")) or key in (
            "player_anytime_td","rushing_yards","receiving_yards",
            "passing_yards","receptions")
    if not (is_player_market(a) and is_player_market(b)):
        return False
    def athlete(selection):
        s=str(selection or "").strip()
        colon=re.match(r"^(.+?):\s*(?:over|under|\d)",s,re.I)
        if colon:
            return re.sub(r"[^a-z0-9]","",colon.group(1).casefold())
        prefix=re.match(r"^(.+?)\s+(?:over|under)\s+\d",s,re.I)
        if prefix:
            return re.sub(r"[^a-z0-9]","",prefix.group(1).casefold())
        return ""
    x,y=athlete(a.selection),athlete(b.selection)
    return bool(x and x==y)
