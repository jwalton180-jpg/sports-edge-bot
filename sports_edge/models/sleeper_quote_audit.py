"""Research-only Kalshi sleeper quote comparison.

Do not promote projections to verified tickets without a separate, independently
validated model, confirmed contract rules, fresh executable asks and exact fees.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from math import isfinite
from sports_edge.models.value_audit import audit_value, ValueAudit

@dataclass(frozen=True)
class SleeperQuote:
    audit: ValueAudit
    outcome: str
    model_version: str
    indicative_net_edge: float | None
    status: str

def screen_sleeper_quote(market:dict,*,outcome:str,model_version:str,
                         independent_probability:float|None,side:str,
                         expected_ticker:str,expected_outcome:str,
                         market_observed_at:str|None,now:datetime|None=None,
                         verified_fee_fraction:float|None=None,
                         rules_verified:bool=False,
                         market_mapping_verified:bool=False)->SleeperQuote:
    """Explicit exact market/outcome match, no fuzzy ticker/title inference.

    An indicative positive edge is never permission to label a recommendation.
    """
    if side not in ("YES","NO"): raise ValueError("explicit YES/NO side required")
    current=now or datetime.now(timezone.utc)
    if current.tzinfo is None: raise ValueError("timezone required")
    current=current.astimezone(timezone.utc)
    fresh=False
    if market_observed_at:
        try:
            seen=datetime.fromisoformat(market_observed_at.replace("Z","+00:00"))
            fresh=seen.tzinfo is not None and timedelta(0)<=current-seen.astimezone(timezone.utc)<=timedelta(minutes=2)
        except (ValueError,TypeError):
            pass
    matched=bool(market_mapping_verified and expected_ticker and expected_outcome
                 and market.get("ticker")==expected_ticker and outcome==expected_outcome)
    # Unvalidated research is *always* research-only regardless of edge.
    audited=audit_value(market,side=side,independent_probability=independent_probability,
                       model_validated=False,evidence_fresh=fresh,
                       coverage_complete=matched,rules_verified=rules_verified and matched,
                       fee_fraction=verified_fee_fraction,observed_at=current)
    # The base audit does not know if mapping was independently checked.
    indicative=audited.edge if matched and fresh and rules_verified else None
    return SleeperQuote(audit=audited,outcome=outcome,model_version=model_version,
                        indicative_net_edge=indicative,status="RESEARCH_ONLY")

def screen_sleeper_board(candidates:list[dict],*,now:datetime|None=None)->list[SleeperQuote]:
    """Sort only by verified indicative edges, with unpriced research last."""
    results=[screen_sleeper_quote(now=now,**candidate) for candidate in candidates]
    return sorted(results,key=lambda r:(r.indicative_net_edge is None,
                                        -(r.indicative_net_edge or 0),
                                        r.audit.ticker,r.outcome))
