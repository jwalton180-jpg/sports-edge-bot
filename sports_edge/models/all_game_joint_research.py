"""All-game NFL/NCAAF/NBA/NHL/MLB correlated-parlay research. No trades.

This does NOT infer a joint probability from multiplying individual leg
probabilities. A model probability for a ticket is only available from a
complete, causal independent *joint* scenario archive. Until one exists, the
app reports dependence and the valid Frechet bounds without fake certainty.
No cash-out forecasts: these require actual future combo-level bid paths.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import isfinite, prod
from typing import Iterable

from sports_edge.models.ticket_dependencies import (
    incompatible, probability_bounds, same_player_cross_market_dependence,
)

SPORTS=("NFL","NCAAF","NBA","NHL","MLB")
MAX_CANDIDATES=500
MAX_PREVIEW=100
MIN_SCENARIOS=500
MAX_SCENARIOS=20000
MAX_TICKET_LEGS=25


def _aware(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else None
    if not isinstance(value, str):
        return None
    try:
        t=datetime.fromisoformat(value.replace("Z","+00:00"))
        return t.astimezone(timezone.utc) if t.tzinfo else None
    except ValueError:
        return None


def _prob(value: object) -> float | None:
    try:
        p=float(value)
        return p if isfinite(p) and 0<=p<=1 else None
    except (TypeError,ValueError):
        return None


def _category(sport: str, market_key: str) -> str:
    key=market_key.lower()
    if sport in {"NFL","NCAAF"}:
        if "pass" in key: return "QB passing / attempts / TDs"
        if "rush" in key: return "Rushing / workload"
        if "rece" in key or "target" in key: return "Receiving / targets"
        if "td" in key or "touchdown" in key: return "Touchdown ladder / team scoring"
        return "Spread / moneyline / totals / other"
    if sport=="NBA":
        if any(x in key for x in ("assist","rebound","point","three","3pt","double")):
            return "Player points / assists / rebounds / 3s"
        return "Score / margin / total / other"
    if sport=="NHL":
        if any(x in key for x in ("shot","goal","assist","point","save")):
            return "Skater shots / points / goals / goalie saves"
        return "Team goals / game total / other"
    if sport=="MLB":
        if any(x in key for x in ("batter","pitcher","strikeout","hit","base","homerun")):
            return "Hits / total bases / HR / strikeouts"
        return "Runs / run line / moneyline / other"
    return "Other"


def same_event_risk(a, b) -> str | None:
    """Qualitative risk, not estimated numeric correlation or causal effect."""
    if a.sport not in SPORTS or b.sport not in SPORTS:
        return None
    if not a.event_id or a.event_id!=b.event_id or a.sport!=b.sport:
        return None
    blocked=incompatible(a,b)
    if blocked: return "INCOMPATIBLE: "+blocked
    if same_player_cross_market_dependence(a,b):return "SHARED_PLAYER_USAGE"
    return "SHARED_GAME_SCRIPT"


def all_game_research_board(candidates: Iterable, *, now: datetime,
                            sports: Iterable[str] = SPORTS,
                            max_preview: int = 50) -> dict:
    """Inspect ALL candidate games; never silently force an unqualified leg.

    Candidates must already carry independent model outputs and exact Kalshi
    executable audit. Verified kickoff is required for pregame qualification.
    No Thursday / TNF or 1-per-game restrictions.
    """
    current=_aware(now)
    if current is None:raise ValueError("aware_now_required")
    requested=tuple(dict.fromkeys(sports))
    if not requested or any(s not in SPORTS for s in requested):
        raise ValueError("supported_sports_only")
    if not 1<=max_preview<=MAX_PREVIEW:raise ValueError("bounded_preview_required")
    candidates=list(candidates)
    if len(candidates)>MAX_CANDIDATES:raise ValueError("bounded_candidate_count")
    from sports_edge.models.parlay_intelligence import assess_leg
    per_sport={s:{"candidates":0,"qualified_pregame":0,"blocked":0,"games":0} for s in requested}
    rows=[]
    grouped=defaultdict(list)
    blockers=Counter()
    seen=set()
    qualified=defaultdict(list)
    for leg in candidates:
        sport=str(getattr(leg,"sport",""))
        if sport not in per_sport:continue
        ticker=str(getattr(leg,"kalshi_ticker",None) or "")
        side=str(getattr(leg,"kalshi_side",None) or "")
        event=str(getattr(leg,"event_id","") or "")
        sel=str(getattr(leg,"selection","") or "")
        key=(sport,event,ticker,side,sel)
        if key in seen:continue
        seen.add(key)
        per_sport[sport]["candidates"]+=1
        start=_aware(getattr(leg,"verified_start",None))
        reasons=list(getattr(leg,"execution_audit_reasons",()) or ())
        model=_prob(getattr(leg,"model_probability",None))
        ask=_prob(getattr(leg,"kalshi_price",None))
        fee=_prob(getattr(leg,"entry_fee_per_contract",None))
        if not event:
            reasons.append("event_identity_unverified")
        if start is None:
            reasons.append("verified_kickoff_missing")
        elif start<=current:
            reasons.append("game_started_no_pregame_promotion")
        if model is None or not getattr(leg,"model_name",None):
            reasons.append("independent_sport_model_missing")
        if ask is None or not (0<ask<1) or not ticker:
            reasons.append("exact_executable_kalshi_ask_missing")
        if fee is None:
            reasons.append("entry_fee_unverified")
        if getattr(leg,"execution_audit_status",None)!="QUALIFIED_VALUE":
            reasons.append("execution_audit_not_qualified")
        approved=False
        if not reasons and start is not None and start>current:
            assessment=assess_leg(leg,"best")
            approved=bool(assessment.qualified)
            if not approved:
                reasons.append("original_ticket_policy_rejected")
        if approved:
            qualified[(sport,event)].append(leg)
            per_sport[sport]["qualified_pregame"]+=1
        else:
            per_sport[sport]["blocked"]+=1
        for r in set(reasons):blockers[r]+=1
        row={
            "Sport":sport,"Game":str(getattr(leg,"event_title","") or event),
            "Event ID":event,
            "Status":"PREGAME LEG QUALIFIED" if approved else "RESEARCH ONLY",
            "Type":_category(sport,str(getattr(leg,"market_key","") or "")),
            "Selection":sel,"Ticker":ticker,
            "Independent model P":model,"Kalshi ask":ask,
            "Entry fee":fee,
            "Individual edge pp":round(100*(model-ask-fee),3) if
                 model is not None and ask is not None and fee is not None else None,
            "Kickoff UTC":start.isoformat() if start else "UNVERIFIED",
            "Blockers":"; ".join(sorted(set(reasons))),
        }
        rows.append(row)
        if event:grouped[(sport,event)].append(row)
    for sport in requested:
        per_sport[sport]["games"]=len({ev for sp,ev in grouped if sp==sport})
    risk=[]
    for (sport,event),legs in qualified.items():
        if len(legs)>1:
            n=0
            classes=Counter()
            for i,a in enumerate(legs):
                for b in legs[i+1:]:
                    typ=same_event_risk(a,b)
                    if typ:
                        n+=1
                        classes[typ]+=1
            risk.append({"Sport":sport,"Event":event,
                         "Individually qualified legs":len(legs),
                         "Dependent pairs":n,
                         "Risk classifications":dict(classes),
                         "Joint P":"UNKNOWN — scenario model not validated",
                         "Combo EV":"UNKNOWN — exact combo ask/fees required"})
    rows.sort(key=lambda r:(r["Status"]=="PREGAME LEG QUALIFIED",
                           r["Individual edge pp"] is not None,
                           r["Individual edge pp"] if r["Individual edge pp"] is not None else -999),
              reverse=True)
    return {
        "scope":"ALL_GAME_FIVE_SPORT_RESEARCH_ONLY",
        "sports":per_sport,"total":sum(x["candidates"] for x in per_sport.values()),
        "qualified_pregame":sum(x["qualified_pregame"] for x in per_sport.values()),
        "blocked":sum(x["blocked"] for x in per_sport.values()),
        "rows":rows[:max_preview],"event_count":len(grouped),
        "same_game_risks":risk[:30],
        "common_blockers":blockers.most_common(12),
        "joint_probability_status":"UNKNOWN_UNLESS_INDEPENDENT_JOINT_SCENARIOS",
        "cashout_probability_status":"UNVALIDATED_NO_PROSPECTIVE_COMBO_QUOTE_PATH",
        "does_not_generate_live_combo_orders":True,
    }


def evaluate_joint_scenarios(legs: Iterable, *, bundle: dict,
                             now: datetime,
                             combo_quote: dict | None = None) -> dict:
    """Compute exact conjunction from pregame independent model simulations.

    Every scenario must contain every selected exact contract-and-side boolean.
    All legs across games share scenario ID (joint draws, not pairwise glue).
    Quote is optional and must be the EXACT combination executable ask + fee.
    Never return a verified positive-EV trade: results are research-only.
    """
    leg_list=list(legs)
    if not 2<=len(leg_list)<=MAX_TICKET_LEGS:
        raise ValueError("requires_2_to_25_legs")
    if not isinstance(bundle,dict) or bundle.get("source_kind")!="INDEPENDENT_PREGAME_JOINT_SCENARIOS":
        raise ValueError("independent_joint_model_source_required")
    current=_aware(now)
    issued=_aware(bundle.get("generated_at_utc"))
    if current is None or issued is None or issued>current:
        raise ValueError("causal_evaluation_clock_required")
    if bundle.get("review_status")!="RESEARCH_ONLY" or not bundle.get("model_version"):
        raise ValueError("research_model_version_required")
    keys=[];probs=[];dates=set();sports=set()
    for l in leg_list:
        sport=getattr(l,"sport",None)
        if sport not in SPORTS:raise ValueError("unsupported_sport")
        start=_aware(getattr(l,"verified_start",None))
        if start is None or issued>=start or start<=current:
            raise ValueError("scenario_or_current_time_after_game_start")
        market=getattr(l,"kalshi_ticker",None)
        side=str(getattr(l,"kalshi_side",None) or "").upper()
        fair=_prob(getattr(l,"model_probability",None))
        if not market or side not in ("YES","NO") or fair is None:
            raise ValueError("complete_independent_leg_identity_required")
        if getattr(l,"execution_audit_status",None)!="QUALIFIED_VALUE":
            raise ValueError("individual_leg_not_execution_qualified")
        if not getattr(l,"event_id",None):
            raise ValueError("event_identity_required")
        dates.add(start);sports.add(sport)
        keys.append(f"{market}|{side}")
        probs.append(fair)
    if len(set(keys))!=len(keys) or len({k.split("|")[0] for k in keys})!=len(keys):
        raise ValueError("duplicate_or_conflicting_leg_contracts")
    draws=bundle.get("draws")
    if not isinstance(draws,list) or not MIN_SCENARIOS<=len(draws)<=MAX_SCENARIOS:
        raise ValueError("insufficient_or_unbounded_joint_draws")
    matched=0
    ids=set()
    for draw in draws:
        if not isinstance(draw,dict) or not isinstance(draw.get("outcomes"),dict):
            raise ValueError("invalid_scenario")
        sid=draw.get("scenario_id")
        if not isinstance(sid,int) or sid in ids:
            raise ValueError("duplicate_or_missing_joint_scenario_id")
        ids.add(sid)
        outcomes=draw["outcomes"]
        if set(outcomes)!=set(keys) or any(type(x) is not bool for x in outcomes.values()):
            raise ValueError("incomplete_or_nonboolean_joint_draw")
        matched+=all(outcomes[k] for k in keys)
    joint=matched/len(draws)
    lb,ub=probability_bounds(probs)
    independent=prod(probs)
    # An uncalibrated simulation may fail marginal reconciliation; mark
    # it as unusable, not an impressive new edge.
    marginals={k:sum(draw["outcomes"][k] for draw in draws)/len(draws) for k in keys}
    if any(abs(marginals[k]-p)>.10 for k,p in zip(keys,probs)):
        raise ValueError("joint_scenario_marginals_disagree_with_frozen_leg_models")
    if not lb-1e-9<=joint<=ub+1e-9:
        raise ValueError("joint_scenarios_violate_probability_bounds")
    result={
        "status":"UNVALIDATED_JOINT_RESEARCH",
        "sports":sorted(sports),"distinct_game_count":len({l.event_id for l in leg_list}),
        "joint_model_probability":round(joint,6),
        "independence_benchmark_not_a_joint_forecast":round(independent,6),
        "model_draws":len(draws),"model_version":str(bundle["model_version"])[:100],
        "unrestricted_joint_probability_bounds":(round(lb,6),round(ub,6)),
        "joint_market_cost":None,"model_minus_exact_combo_cost":None,
        "qualified_for_live":False,"cashout_probability":None,
        "warnings":["Simulation outputs require chronological sports-specific calibration and prospective validation.",
                    "No unfilled private RFQ or missing combo price can be invented."],
    }
    if combo_quote is not None:
        if not isinstance(combo_quote,dict):
            raise ValueError("invalid_exact_combo_quote")
        ask=_prob(combo_quote.get("executable_ask_dollars"))
        fee=_prob(combo_quote.get("entry_fee_per_contract"))
        quote_at=_aware(combo_quote.get("observed_at_utc"))
        if (not combo_quote.get("exact_combo_ticker") or
            combo_quote.get("source")!="KALSHI_EXACT_COMBO_EXECUTABLE_ASK" or
            quote_at is None or quote_at>current or (current-quote_at).total_seconds()>60 or
            ask is None or fee is None or not 0<ask<1 or ask+fee>=1):
            raise ValueError("unverified_or_stale_exact_combo_ask_fee")
        result["joint_market_cost"]=round(ask+fee,6)
        result["model_minus_exact_combo_cost"]=round(joint-ask-fee,6)
        result["warnings"].append("Quote alone does not verify fill or forecast early cash-out.")
    return result
