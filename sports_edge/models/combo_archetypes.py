"""Observed combo market *hypotheses*; zero outcome-based promotion.

These names describe the kinds of listed contracts worth investigating,
not profitable systems. Screenshots of closed positions are not causal model
predictions and a cash-out payout is not an official binary settlement.
"""
from __future__ import annotations
import re

def combo_pattern(sport: str, market_key: str, selection: str,
                  market_label: str = "") -> str:
    sport=str(sport or "").upper()
    key=str(market_key or "").casefold()
    phrase=" ".join((str(selection or ""),str(market_label or ""))).casefold()
    if sport=="NBA":
        match=re.search(r"\b(?:win|wins)\s+by\s+(?:over|more than)\s+(18\.5|19\.5|30\.5)",phrase)
        if match:
            return f"NBA full-game win by {int(float(match.group(1)))+1}+ · independently modeled research"
        if "total" in key or "total" in phrase:
            return "NBA unusually high game-total NO · no independent model yet"
    if sport=="NFL":
        if "touchdown" in key or "touchdown" in phrase:
            return "NFL individual TD scorer tail · independently modeled research" if ("2+" in phrase or "over 1.5" in phrase) else "NFL anytime TD scorer"
        if "reception" in key or "receiving" in key or "rushing" in key or "passing" in key:
            return "NFL player volume ladder · independent model when verified"
    if sport=="NHL":
        if "team" in key and ("total" in key or "goal" in phrase):
            return "NHL modest team-goals threshold · independently modeled research"
        if "total" in key or "goal" in phrase:
            return "NHL game/team scoring threshold · verify exact scope"
        if key in ("model_h2h","h2h"):
            return "NHL game winner · model must be verified"
    if sport=="NCAAF":
        if key in ("model_h2h","h2h"):
            return "NCAAF winner and upset-risk intelligence"
        if "total" in key or "total" in phrase:
            return "NCAAF team/game totals · no independent promoted model"
    if sport=="SOCCER":
        if "total" in key or "goal" in phrase:
            return "Soccer modest Over or extreme-total NO · league-specific validation"
        if key in ("model_h2h","h2h"):
            return "Soccer match winner · competition-specific model required"
    if sport=="MLB":
        if "total" in key or "run" in phrase:
            return "Baseball team/game run thresholds · exact league model required"
    if sport=="TENNIS":
        if "game" in key or "games" in phrase:
            return "Tennis conservative match-games Over · sport-specific model"
        if key=="model_h2h":
            return "Tennis match winner / separately validated live reversal"
    return "Other contract · independent model and exact Kalshi rule review required"
