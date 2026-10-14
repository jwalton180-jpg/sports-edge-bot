"""Application-wide read-only evidence/quote pipeline.

All listed rows, including rejected rows, are retained. Independent projections
never consume Kalshi prices. New models stay research-only until scoped review.
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_CEILING
from math import floor, isfinite
import hashlib
import json
import re

from sports_edge.data.kalshi import KalshiPublicClient
from sports_edge.data.sleeper_sources import SleeperSources, name, nominal_day
from sports_edge.models.kalshi_sports import group_kalshi_sports
from sports_edge.models.sleeper_counts import nhl_count, nfl_scored_touchdowns, utc
from sports_edge.models.ncaaf_vulnerability import review_upset
from sports_edge.models.sleeper_quote_audit import screen_sleeper_quote

VERSION = "sleeper-evidence-v2"


def production_commit():
    import os
    from pathlib import Path
    import subprocess
    sha = os.getenv("GITHUB_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        try:
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], timeout=2, stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"
    return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else "unknown"


@dataclass(frozen=True)
class MarketSpec:
    ticker: str
    sport: str
    family: str
    participant: str
    threshold: int | None
    event_time: str
    rules_hash: str
    rules_supported: bool
    reasons: tuple[str, ...]


def contract_spec(row):
    m = row.market
    title = str(m.get("title", ""))
    primary, secondary = str(m.get("rules_primary", "")), str(m.get("rules_secondary", ""))
    rules = primary+"\n"+secondary
    problems = []
    try:
        utc(m["occurrence_datetime"])
    except (KeyError, ValueError, TypeError):
        problems.append("scheduled_start_missing")
    if m.get("market_type") != "binary" or str(m.get("notional_value_dollars", "1.0000")) not in {"1", "1.0", "1.00", "1.0000"}:
        problems.append("non_binary_or_variable_payoff_unsupported")
    if not primary or not secondary:
        problems.append("contract_rules_missing")
    if any(x in rules.lower() for x in ("ladder", "escalator", "slingshot", "payout increases")):
        problems.append("non_binary_ladder_payoff_unsupported")
    threshold = None
    participant = title.split(":", 1)[0].strip() if ":" in title else str(m.get("yes_sub_title", ""))
    if row.family == "Team Total":
        match = re.match(r"^(.+?)\s+(?:over|under)\s+[0-9.]+",title,re.I)
        if match:
            participant=match.group(1)
    if ":" in title and name(participant) not in name(primary):
        problems.append("participant_contract_rule_mismatch")
    if row.family in {"Player Goals", "Shots on Goal", "Player Assists", "Player Points", "Goalie Saves", "Player Touchdowns", "Team Total", "Game Total", "Points", "Rebounds", "Assists", "Three-Pointers", "Points + Rebounds + Assists"}:
        try:
            strike = float(m["floor_strike"])
            if not isfinite(strike) or strike < 0 or m.get("strike_type") != "greater" or not strike % 1 == .5:
                raise ValueError("ambiguous count threshold")
            threshold = floor(strike)+1
            # The rules must independently repeat the exact threshold.
            if not (re.search(r"\b"+str(threshold)+r"\+", primary) or
                    re.search(r"(?:over|more than)\s+"+re.escape(f"{strike:g}")+r"\b", primary, re.I)):
                raise ValueError("rule/strike threshold mismatch")
        except (ValueError, TypeError, KeyError):
            problems.append("count_threshold_or_rule_mismatch")
    if row.sport == "NFL" and row.family == "Player Touchdowns":
        if "scores" not in primary.lower() or "touchdown" not in primary.lower() or "snap" not in secondary.lower():
            problems.append("scored_touchdown_participation_rules_unverified")
    if row.sport == "Soccer":
        if "90 minutes" not in primary.lower() or "does not include extra time or penalties" not in primary.lower():
            problems.append("soccer_regulation_scope_unverified")
    if row.sport == "NHL" and row.family == "Team Total":
        # Model ingest must reproduce the contract's administrative shootout
        # goal, rather than assuming the score is regulation-only.
        low = rules.lower()
        excluded = any(x in low for x in ("excludes shootout", "shootout goals do not count", "excluding shootout"))
        included = "shootout winner is credited with one additional goal" in low and "counts toward this market" in low
        if "overtime" not in low or not (excluded or included):
            problems.append("nhl_overtime_shootout_scope_unverified")
    return MarketSpec(str(m.get("ticker", "")), row.sport, row.family, participant, threshold,
                      str(m.get("occurrence_datetime", "")), hashlib.sha256(rules.encode()).hexdigest(),
                      not problems, tuple(problems))


def entry_fee(price, series, *, contracts=1):
    """Immediate-match quadratic fee; cent rounding for the entire order.

    Flat/unknown fee schedules fail closed. No assumed fee for other brokers.
    """
    if type(contracts) is not int or contracts < 1:
        raise ValueError("positive whole-contract quantity required")
    if series.get("fee_type") not in {"quadratic", "quadratic_with_maker_fees", "quadratic_with_combo_maker_fees"}:
        return None
    try:
        p, multiplier = Decimal(str(price)), Decimal(str(series["fee_multiplier"]))
        if not p.is_finite() or not multiplier.is_finite() or not 0 < p < 1 or multiplier < 0:
            return None
    except (KeyError, ValueError, TypeError, __import__("decimal").InvalidOperation):
        return None
    total = (Decimal(".07")*multiplier*contracts*p*(1-p)*100).to_integral_value(rounding=ROUND_CEILING)/100
    return float(total/Decimal(contracts))


def execution_quote(market, side, series, *, received_at, now, contracts=1):
    issues = []
    if str(market.get("status", "")).lower() not in {"active", "open"}:
        issues.append("market_not_tradable")
    try:
        if not timedelta(0) <= utc(now)-utc(received_at) <= timedelta(seconds=120):
            raise ValueError("stale")
    except (ValueError, TypeError):
        issues.append("quote_stale_or_timestamp_missing")
    prefix = side.lower()
    price = bid = size = None
    try:
        price = float(market[prefix+"_ask_dollars"])
        if not isfinite(price) or not 0 < price < 1:
            raise ValueError("invalid ask")
    except (KeyError, ValueError, TypeError):
        price = None
        issues.append("executable_ask_missing")
    try:
        size = float(market[prefix+"_ask_size_fp"])
        if not isfinite(size) or size < contracts:
            raise ValueError("thin")
    except (KeyError, ValueError, TypeError):
        issues.append("ask_size_unverified_or_insufficient")
    try:
        bid = float(market[prefix+"_bid_dollars"])
        if not isfinite(bid) or not 0 <= bid < 1 or price is None or bid > price:
            raise ValueError("crossed")
    except (KeyError, ValueError, TypeError):
        bid = None
        issues.append("bid_or_spread_unverified")
    fee = entry_fee(price, series, contracts=contracts) if price is not None else None
    if fee is None:
        issues.append("fee_schedule_unverified")
    return {"ask": price, "bid": bid, "spread": price-bid if price is not None and bid is not None else None,
            "available_size": size, "contracts": contracts, "entry_fee_per_contract": fee,
            "break_even": price+fee if price is not None and fee is not None else None,
            "received_at": received_at, "source_updated_at": market.get("updated_time"),
            "reasons": issues}


def research_projection(row, spec, sources):
    """Bounded sport-specific routing; unsupported markets return no estimate."""
    m = row.market
    if not spec.rules_supported:
        return None, {}
    if row.sport in {"NBA", "WNBA"} and row.family in {"Points", "Rebounds", "Assists", "Three-Pointers", "Points + Rebounds + Assists"}:
        from sports_edge.models.basketball_sleeper import basketball_sleeper
        from sports_edge.data.basketball_prop_data import _url
        context = sources.basketball_context(m, row.sport, spec.participant)
        event=utc(spec.event_time)
        season=event.year+int(row.sport=="NBA" and event.month>=7)
        context["source_urls"]=[_url(row.sport,season-1),_url(row.sport,season)]
        context["source_retrieved_at"]=None
        context["freshness_limitations"]=["existing_cached_player_loader_retrieval_time_not_recorded"]
        return basketball_sleeper(sport=row.sport, player=spec.participant, family=row.family,
            threshold=spec.threshold, before_time=context["start"], season_type=context["season_type"]), context
    if row.sport == "NHL":
        fields = {"Player Goals":"goals", "Shots on Goal":"shots", "Player Assists":"assists",
                  "Player Points":"points", "Goalie Saves":"saves", "Team Total":"team_goals"}
        field = fields.get(row.family)
        if not field or spec.threshold is None:
            return None, {}
        context = sources.nhl_team(m, spec.participant) if field == "team_goals" else sources.nhl_player(m, spec.participant)
        return nhl_count(context["rows"], market=field, threshold=spec.threshold, before_time=context.get("start",spec.event_time)), context
    if row.sport == "NFL" and row.family == "Player Touchdowns" and spec.threshold in {1,2,3,4}:
        from sports_edge.data.nfl_prop_data import player_context
        from sports_edge.data.nfl_scoring_credits import attach_scored_totals
        from sports_edge.models.sleeper_counts import nfl_all_scored_touchdowns
        ctx = player_context(spec.participant, nominal_day(m), m.get("event_ticker"))
        if ctx is None:
            raise ValueError("nfl_player_history_or_schedule_unresolved")
        rows = [dict(r, source="nflverse", player_id=ctx.player_id) for r in (*ctx.prior_rows, *ctx.current_rows)]
        if any(not r.get("date") for r in rows):
            raise ValueError("nfl_history_causal_dates_missing")
        if not ctx.scheduled_start:
            raise ValueError("independent_nfl_kickoff_missing")
        rows,credit_sources=attach_scored_totals(rows)
        return nfl_all_scored_touchdowns(rows, threshold=spec.threshold, before_time=ctx.scheduled_start), {
            "scoring_scope":"all_credited_touchdown_scorers", "scoring_sources":credit_sources,
            "start":ctx.scheduled_start,
            "event_id": m["event_ticker"], "participant": ctx.player_name, "opponent":ctx.opponent,
            "missing":["red_zone_role", "active_status", "offensive_line", "opponent_personnel"],
            "source_urls":["https://github.com/nflverse/nflverse-data/releases/tag/stats_player"]}
    if row.sport == "Soccer" and str(m.get("ticker", "")).startswith("KXEPL"):
        from sports_edge.models.soccer_sleeper_research import soccer_market_projection
        context = sources.espn_match_history(m, "Soccer")
        field = {"Game Total":"total_goals", "Both Teams Score":"both_teams_score"}.get(row.family)
        if row.family == "Team Total":
            if name(spec.participant) == name(context["home_name"]):
                field = "home_goals"
            elif name(spec.participant) == name(context["away_name"]):
                field = "away_goals"
        if field:
            return soccer_market_projection(context["rows"], home_id=context["home_id"], away_id=context["away_id"],
                before_time=context["start"], market=field, threshold=spec.threshold), context
    if row.sport == "NCAAF" and row.family == "Moneyline":
        from sports_edge.models.team_challengers import ncaaf_opponent_challenger
        context = sources.espn_match_history(m, "NCAAF")
        projection = ncaaf_opponent_challenger(context["rows"], home_id=context["home_id"], away_id=context["away_id"], before_time=context["start"])
        selected = name(m.get("yes_sub_title", ""))
        home_aliases = {name(x) for x in context.get("home_aliases", [context["home_name"]])}
        away_aliases = {name(x) for x in context.get("away_aliases", [context["away_name"]])}
        home_selected = selected in home_aliases
        away_selected = selected in away_aliases
        if home_selected == away_selected:
            raise ValueError("ncaaf_selection_identity_unresolved")
        if away_selected and projection.probability is not None:
            projection = replace(projection, probability=1-projection.probability)
        context["upset_review"] = asdict(review_upset({}, event_id=context["event_id"], now=datetime.now(timezone.utc).isoformat()))
        return projection, context
    return None, {}


def scan_sleeper_markets(markets, *, coverage_complete, limit=60, client=None, sources=None, now=None, legacy_candidates=()):
    client, sources = client or KalshiPublicClient(), sources or SleeperSources()
    current = now or datetime.now(timezone.utc)
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError("bounded positive scan limit required")
    rows = [r for group in group_kalshi_sports(markets).values() for r in group]
    series_cache, records, analyzed = {}, [], 0
    legacy = {(c.kalshi_ticker,c.kalshi_side):c for c in legacy_candidates}
    for row in rows:
        spec, context, projection = contract_spec(row), {}, None
        issues = list(spec.reasons)
        fresh, received = row.market, None
        if not coverage_complete:
            issues.append("incomplete_market_coverage")
        try:
            pregame = utc(spec.event_time) > current
        except (ValueError, TypeError):
            pregame = False
        if not pregame:
            issues.append("live_or_start_unknown_requires_live_model")
        elif analyzed >= limit:
            issues.append("scan_limit_not_analyzed")
        else:
            analyzed += 1
            try:
                response = client.market(spec.ticker)
                fresh, received = response.data["market"], response.received_at.isoformat()
                if fresh.get("ticker") != spec.ticker:
                    raise ValueError("quote_ticker_mismatch")
                fresh_row = replace(row, market=fresh)
                fresh_spec = contract_spec(fresh_row)
                if fresh_spec != spec:
                    raise ValueError("contract_changed_refresh_required")
                series_key = spec.ticker.split("-",1)[0]
                if series_key not in series_cache:
                    sr = client.series(series_key).data["series"]
                    if sr.get("ticker") != series_key:
                        raise ValueError("fee_series_mismatch")
                    series_cache[series_key] = sr
                projection, context = research_projection(row, spec, sources)
            except requests_error_types() as exc:
                issues.append("source_or_quote_unavailable:"+(str(exc)[:100] if isinstance(exc,ValueError) else type(exc).__name__))
        # Once independently matched, the official game start replaces
        # Kalshi occurrence_datetime (often the expected postgame close).
        # Neither a missed matchup nor an already-live game may inherit an
        # outdated pregame projection or legacy price-following fallback.
        if row.sport in {"NHL", "NCAAF", "Soccer", "NBA", "WNBA"} or (row.sport == "NFL" and row.family == "Player Touchdowns"):
            verified_start = context.get("start")
            if verified_start:
                try:
                    true_start = utc(verified_start)
                    if true_start > utc(spec.event_time):
                        issues.append("official_start_after_market_occurrence")
                        projection = None
                        pregame = False
                    elif true_start <= current:
                        issues.append("official_game_already_started")
                        issues.append("independent_start_passed_requires_live_model")
                        projection = None
                        pregame = False
                    spec = replace(spec, event_time=true_start.isoformat())
                except (ValueError, TypeError):
                    issues.append("official_start_invalid")
                    projection = None
                    pregame = False
            else:
                issues.append("official_game_start_unverified")
                projection = None
                pregame = False
        probability = getattr(projection, "probability", None)
        base = legacy.get((spec.ticker,"YES"))
        if projection is None and base is not None and spec.rules_supported and pregame:
            from sports_edge.models.new_sport_research import Projection
            projection = Projection(row.sport, base.selection, base.model_probability, base.model_name or "legacy", "UNVALIDATED_RESEARCH",
                                    base.model_reasons+base.model_warnings, base.model_sample_size)
            probability = projection.probability
            context = {"missing":["scoped_validation_and_current_availability_review"], "provenance":"existing_independent_model_projection"}
        if probability is None:
            issues.append("independent_model_or_evidence_unavailable")
        # No data ingestion or historical replay can manufacture prospective
        # validation. All new projections remain research pending scoped review.
        issues.append("prospective_validation_required")
        observed = now or datetime.now(timezone.utc)
        source_dates = []
        for r in context.get("rows", []):
            value = r.get("event_time") or (r.get("date", "")+"T00:00:00Z")
            try:
                if utc(value) < utc(spec.event_time)-timedelta(hours=24):
                    source_dates.append(value)
            except (ValueError, TypeError):
                pass
        for side in ("YES", "NO"):
            q = execution_quote(fresh, side, series_cache.get(spec.ticker.split("-",1)[0], {}),
                                received_at=received, now=observed.isoformat())
            p = probability if side == "YES" else (1-probability if probability is not None else None)
            screening = screen_sleeper_quote(fresh, outcome=f"{spec.family}:{spec.participant}:{spec.threshold}",
                expected_outcome=f"{spec.family}:{spec.participant}:{spec.threshold}",expected_ticker=spec.ticker,
                model_version=getattr(projection,"model","unsupported"),independent_probability=p, side=side,
                market_observed_at=received, now=observed, entry_fee_per_contract=q["entry_fee_per_contract"],
                rules_verified=spec.rules_supported, market_mapping_verified=projection is not None)
            edge = screening.indicative_net_edge
            reasons = list(dict.fromkeys(issues+q["reasons"]+context.get("missing", [])))
            if edge is not None and edge <= 0:
                reasons.append("no_positive_net_edge")
            low, high = getattr(projection, "lower", None), getattr(projection, "upper", None)
            if side == "NO" and low is not None:
                low, high = 1-high, 1-low
            records.append({**asdict(spec), "side":side, "probability":p,
                            "sensitivity_range":[low,high], "uncertainty_type":"model_sensitivity_not_calibrated_CI",
                            "quote":q, "net_edge":edge, "status":"REJECTED" if edge is not None and edge <= 0 else "RESEARCH_ONLY",
                            "category":"research_only", "reasons":reasons,
                            "model":getattr(projection,"model","unsupported"),
                            "sample":getattr(projection,"sample",getattr(projection,"sample_size",0)),
                            "source_as_of":getattr(projection,"source_as_of",None) or (max(source_dates, key=utc) if source_dates else None),
                            "model_reasons":list(getattr(projection,"reasons",())),
                            "context":{k:v for k,v in context.items() if k != "rows"},
                            "market_occurrence_time":row.market.get("occurrence_datetime"),
                            "start_verified":bool(context.get("start")),
                            "observed_at":observed.isoformat()})
    records.sort(key=lambda r:(r["net_edge"] is None,-(r["net_edge"] or 0.),r["ticker"],r["side"]))
    return {"schema_version":VERSION, "production_commit":production_commit(), "generated_at":current.isoformat(),
            "coverage":{"catalog_complete":coverage_complete, "listed_markets":len(rows),
                        "analyzed_markets":analyzed, "limit":limit, "scan_complete":analyzed == len(rows)},
            "sources":sources.provenance, "records":records,
            "safety":{"read_only":True,"order_submission":False},
            "payout":{"executable_combo":False,"label":"illustrative_only","dependence_model":"unvalidated"}}


def run_application_sleeper_scan(markets, *, coverage_complete, limit=60):
    """Reuse the independent MLB/NFL/WNBA/Tennis models for the same board.

    NBA minutes research is routed separately and rejects preseason. This
    bounded scan retains every unprocessed inventory item with a reason.
    """
    from sports_edge.models.kalshi_model_candidates import model_candidates_from_kalshi
    legacy = model_candidates_from_kalshi(group_kalshi_sports(markets[:limit]), sport_filter="All",
        include_mlb_home_runs=True, include_mlb_total_bases=True, include_mlb_rbis=True,
        include_mlb_hrr=True, include_mlb_strikeouts=True, include_mlb_game_lines=True,
        include_nfl_passing_yards=True, include_nfl_passing_tds=True, include_nfl_pass_attempts=True,
        include_nfl_pass_completions=True, include_nfl_pass_interceptions=True,
        include_nfl_rushing_yards=True, include_nfl_rush_attempts=True, include_nfl_rush_receiving_yards=True,
        include_nfl_receiving_yards=True, include_nfl_receptions=True, include_nfl_touchdowns=True,
        include_nfl_game_lines=True, include_wnba_game_lines=True, include_wnba_player_props=True,
        include_tennis_games_total=True)
    return scan_sleeper_markets(markets, coverage_complete=coverage_complete, limit=limit, legacy_candidates=legacy)


def report_candidates(report, original_markets):
    from sports_edge.models.parlay_candidates import ParlayCandidateLeg
    originals = {m.get("ticker"):m for m in original_markets}
    candidates = []
    for r in report["records"]:
        if r["probability"] is None or r["quote"]["ask"] is None:
            continue
        m = originals[r["ticker"]]
        candidates.append(ParlayCandidateLeg(sport=r["sport"], event_id=m.get("event_ticker",r["ticker"]),
            event_title=m.get("event_title", m.get("title", "")), market_key={"Points":"player_points", "Rebounds":"player_rebounds", "Assists":"player_assists", "Three-Pointers":"player_threes", "Points + Rebounds + Assists":"player_points_rebounds_assists"}.get(r["family"],r["family"].lower().replace(" ","_")),
            market_label=r["family"], selection=f"{r['participant']} {r['side']} {r['threshold'] or ''} {r['family']}",
            consensus_probability=r["probability"], book_count=0, source_age_s=0, median_odds=None,
            kalshi_ticker=r["ticker"], kalshi_side=r["side"], kalshi_price=r["quote"]["ask"],
            kalshi_edge_points=100*r["net_edge"] if r["net_edge"] is not None else None,
            kalshi_status="RESEARCH_ONLY", evidence_class="MODEL", model_probability=r["probability"],
            model_confidence=0., model_name=r["model"], model_sample_size=r["sample"],
            model_reasons=tuple(r["model_reasons"]), model_warnings=tuple(r["reasons"]),
            execution_audit_status=r["status"], execution_audit_reasons=tuple(r["reasons"]),
            verified_start=r["event_time"] if r.get("start_verified") else None,
            entry_fee_per_contract=r["quote"]["entry_fee_per_contract"]))
    return candidates


def requests_error_types():
    import requests
    return (requests.RequestException, KeyError, ValueError, TypeError, IndexError, StopIteration, AttributeError)


def export_evidence(report):
    """Only the deliberately public report schema is exported; never session/account data."""
    exported = {k:report[k] for k in ("schema_version", "generated_at", "coverage", "sources", "records", "safety", "payout")}
    exported["production_commit"] = report.get("production_commit", "unknown")
    return json.dumps(exported, indent=2, allow_nan=False)


def audit_app_candidates(candidates, markets, *, coverage_complete, now=None, promotions=None):
    """Mandatory app-level audit using timestamped catalog snapshots.

    Metadata discovery provides fees; unknown rates remain unverified. Legacy
    model confidence is not interchangeable with scoped statistical validation.
    This function is pure: no hidden refresh, server or order submission.
    """
    current = (now or datetime.now(timezone.utc)).isoformat()
    by_ticker = {r.market.get("ticker"):r for group in group_kalshi_sports(markets).values() for r in group}
    result = []
    for c in candidates:
        row = by_ticker.get(c.kalshi_ticker)
        issues, fee = [reason for reason in c.execution_audit_reasons if reason not in {"prospective_validation_required","scoped_prospective_model_validation_required"}], None
        if not row:
            issues.append("exact_market_missing_from_current_catalog")
        else:
            spec = contract_spec(row)
            issues.extend(spec.reasons)
            m = row.market
            q = execution_quote(m, c.kalshi_side or "YES", {"fee_type":m.get("sports_edge_fee_type"), "fee_multiplier":m.get("sports_edge_fee_multiplier")},
                                received_at=m.get("sports_edge_received_at"), now=current)
            fee = q["entry_fee_per_contract"]
            issues.extend(q["reasons"])
            if q["ask"] is None or c.kalshi_price is None or abs(q["ask"]-c.kalshi_price) > .000001:
                issues.append("candidate_quote_mismatch")
            if c.model_probability is None:
                issues.append("independent_model_probability_missing")
            try:
                if not c.verified_start:
                    issues.append("independent_scheduled_start_unverified")
                if utc(c.verified_start or spec.event_time) <= utc(current):
                    issues.append("pregame_projection_after_start")
            except (ValueError, TypeError):
                pass
        if not coverage_complete:
            issues.append("incomplete_market_coverage")
        scope_key=(c.sport, row.family if row else "", c.model_name,
                   f"{row.family}:{spec.threshold}:{c.kalshi_side}" if row else "")
        if scope_key not in (promotions or {}):
            issues.append("scoped_prospective_model_validation_required")
        elif c.sport == "NCAAF":
            issues.append("fresh_complete_upset_vulnerability_review_required")
        if c.model_probability is not None and c.kalshi_price is not None and fee is not None and c.model_probability <= c.kalshi_price+fee:
            issues.append("no_positive_net_edge")
        result.append(replace(c, execution_audit_status="RESEARCH_ONLY" if issues else "QUALIFIED_VALUE", execution_audit_reasons=tuple(dict.fromkeys(issues)), entry_fee_per_contract=fee,
                              validation_reviewed=scope_key in (promotions or {})))
    return result
