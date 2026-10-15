"""Thursday NFL parlay *research* focus without bypassing ticket audits.

Selects verified Thursday kickoffs (US Eastern) and presents independent
model-vs-executable-ask evidence. Quotes, fees and role validation remain
mandatory. This does NOT compute a same-game joint probability, forecast an
early profitable cashout, or turn archived research into live picks.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, date
from math import isfinite
from zoneinfo import ZoneInfo

EASTERN=ZoneInfo("America/New_York")
HAWAII=ZoneInfo("Pacific/Honolulu")


def kickoff(row):
    raw=getattr(row, "verified_start", None)
    if not isinstance(raw,str) or not raw.strip():
        return None
    try:
        when=datetime.fromisoformat(raw.replace("Z","+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        return None
    return when


def market_group(leg):
    key=str(getattr(leg,"market_key","") or "").lower()
    if key in ("player_receptions","player_reception_yds"):
        return "Receiving / receptions"
    if key in ("player_rush_yds","player_rush_attempts","player_rush_reception_yds"):
        return "Rushing / workload"
    if key in ("player_pass_yds","player_pass_tds","player_pass_attempts",
               "player_pass_completions","player_pass_interceptions"):
        return "Passing / QB volume"
    if key=="player_anytime_td":
        return "Touchdown scorer / 2+ TD"
    if key in ("nfl_spread","nfl_game_total","nfl_team_total","model_h2h"):
        return "Game / team scoring"
    return "Other NFL model market"


def _probability(value):
    try:
        p=float(value)
    except (TypeError,ValueError):
        return None
    return p if isfinite(p) and 0<=p<=1 else None


def thursday_research_board(candidates, *, local_date: date, now: datetime, max_rows=24):
    if not isinstance(local_date,date) or isinstance(local_date,datetime):
        raise ValueError("local_date_must_be_a_date")
    if now.tzinfo is None:
        raise ValueError("aware_clock_required")
    if not 1<=max_rows<=50:
        raise ValueError("bounded_preview_required")
    if local_date.weekday()!=3:
        return {"scope":"not_thursday","total":0,"research":[],
                "qualified":0,"blocked":0,"role_uncertain":0,
                "kickoff_unverified":0,"already_started":0,
                "dependence_pairs":0,"common_blockers":[]}
    from sports_edge.models.parlay_intelligence import assess_leg
    from sports_edge.models.ticket_dependencies import same_player_cross_market_dependence
    seen=set()
    table=[]
    blockers=Counter()
    qualified_legs=[]
    unverified=0
    started=0
    uncertain=0
    total=0
    for leg in candidates:
        if getattr(leg,"sport",None)!="NFL":
            continue
        event_time=kickoff(leg)
        if (event_time is not None and
            (event_time.astimezone(EASTERN).weekday()!=3 or
             event_time.astimezone(HAWAII).date()!=local_date)):
            continue
        key=(str(getattr(leg,"kalshi_ticker",None) or ""),str(getattr(leg,"kalshi_side",None) or ""),
             str(getattr(leg,"selection",None) or ""))
        if key in seen:continue
        seen.add(key)
        total+=1
        start_known=event_time is not None
        before_kickoff=start_known and event_time>now
        if not start_known:
            unverified+=1
        elif not before_kickoff:
            started+=1
        prob=_probability(getattr(leg,"model_probability",None))
        ask=_probability(getattr(leg,"kalshi_price",None))
        fee=_probability(getattr(leg,"entry_fee_per_contract",None))
        aud=str(getattr(leg,"execution_audit_status",None) or "")
        reasons=tuple(str(s) for s in (getattr(leg,"execution_audit_reasons",None) or ()))
        if not start_known:
            reasons+=("official_kickoff_unverified",)
        elif not before_kickoff:
            reasons+=("game_already_started_no_new_pregame_pick",)
        if prob is None:
            reasons+=("independent_model_probability_missing",)
        if ask is None or ask<=0 or ask>=1:
            reasons+=("executable_ask_missing",)
        if fee is None:
            reasons+=("entry_fee_unverified",)
        if aud!="QUALIFIED_VALUE":
            reasons+=("independent_model_prospective_or_execution_gate_not_qualified",)
        validated=False
        if before_kickoff and prob is not None and 0<ask<1 if ask is not None else False:
            if aud=="QUALIFIED_VALUE" and fee is not None:
                assessment=assess_leg(leg,"best")
                validated=bool(assessment.qualified)
                if not validated:
                    reasons+=("best_available_ticket_policy_not_met",)
        if any("role" in r.lower() or "injur" in r.lower() or "starter" in r.lower()
               for r in reasons):
            uncertain+=1
        for reason in set(reasons):
            blockers[reason]+=1
        if validated:
            qualified_legs.append(leg)
        edge_pp=100*(prob-ask-fee) if prob is not None and ask is not None and fee is not None else None
        table.append({
            "Status":"PREGAME QUALIFIED" if validated else "RESEARCH ONLY",
            "Category":market_group(leg),"Game":str(getattr(leg,"event_title","") or ""),
            "Selection":str(getattr(leg,"selection","") or ""),
            "Side":str(getattr(leg,"kalshi_side","") or ""),
            "Ticker":str(getattr(leg,"kalshi_ticker","") or ""),
            "Model P":round(prob,4) if prob is not None else None,
            "Kalshi ask":round(ask,4) if ask is not None else None,
            "Fee":round(fee,4) if fee is not None else None,
            "Model-ask-fee (pp)":round(edge_pp,2) if edge_pp is not None else None,
            "Verified start":event_time.isoformat() if start_known else "UNVERIFIED",
            "Blocking evidence":"; ".join(sorted(set(reasons))) if not validated else "none",
        })
    pairs=sum(same_player_cross_market_dependence(a,b)
              for i,a in enumerate(qualified_legs) for b in qualified_legs[i+1:])
    table.sort(key=lambda r:(
        r["Status"]=="PREGAME QUALIFIED",
        r["Model-ask-fee (pp)"] is not None,
        r["Model-ask-fee (pp)"] if r["Model-ask-fee (pp)"] is not None else -999,
    ),reverse=True)
    return {
        "scope":"thursday_pregame_model_only",
        "total":total,"research":table[:max_rows],
        "qualified":len(qualified_legs),"blocked":total-len(qualified_legs),
        "kickoff_unverified":unverified,"already_started":started,
        "role_uncertain":uncertain,"dependence_pairs":pairs,
        "common_blockers":blockers.most_common(8),
        "not_a_parlay_probability":True,
        "cashout_probability_not_calibrated":True,
    }
