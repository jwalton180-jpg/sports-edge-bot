"""Review-ready prospective study export from immutable on-demand scan/settlement records.

This prepares evidence, not an authenticated lock or independently reviewed model.
An editable SQLite file or downloaded JSONL is NOT trusted proof of when a call
was made. Independent archive timestamps, baseline registration and source review
are required before prospective promotion.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite

from sports_edge.models.sleeper_counts import utc


def _valid_probability(value):
    return type(value) in (float, int) and isfinite(value) and 0 <= value <= 1


def prospective_review_export(jsonl: str) -> dict:
    """Use first pre-event YES-side observation per contract/model/outcome.

    NO complements cannot inflate sample size, nor may rescans of an outcome
    become additional independent predictions. Conflicting settlements fail
    closed; voids never become binary wins.
    """
    if len(jsonl.encode("utf-8")) > 20_000_000:
        raise ValueError("review input exceeds 20MB bound")
    scans = {}
    settlements = {}
    for line in jsonl.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        key, kind, payload = entry["id"], entry["kind"], entry["payload"]
        if kind == "scan":
            if key in scans:
                raise ValueError("duplicate scan identity")
            scans[key] = payload
        elif kind == "settlement":
            pair = (payload["scan_id"], payload["ticker"])
            if pair in settlements:
                raise ValueError("duplicate or conflicting official settlement")
            settlements[pair] = payload
        else:
            raise ValueError("unknown evidence record")
    observations = []
    for scan_id, report in scans.items():
        for row in report.get("records", []):
            # Both price sides are preserved in raw export. One canonical YES
            # outcome is enough for fair-probability calibration.
            if row.get("side") != "YES":
                continue
            observations.append((scan_id, report, row))
    def observed(entry):
        try:
            return utc(entry[2]["observed_at"])
        except (ValueError, KeyError, TypeError):
            return datetime.max.replace(tzinfo=timezone.utc)
    observations.sort(key=lambda entry: (observed(entry), entry[0], str(entry[2].get("ticker"))))
    seen, prepared, duplicates = set(), [], 0
    for scan_id, report, r in observations:
        ticker = str(r.get("ticker") or "")
        model = str(r.get("model") or "")
        outcome = f'{r.get("family")}:{r.get("threshold")}:YES'
        identity = (ticker, model, outcome)
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        reasons = []
        try:
            event = utc(r["event_time"])
            locked = utc(r["observed_at"])
            if locked >= event:
                reasons.append("not_recorded_before_event")
        except (KeyError, ValueError, TypeError):
            event = locked = None
            reasons.append("event_or_observation_timestamp_missing")
        try:
            source_as_of = utc(r["source_as_of"])
            if locked is not None and source_as_of > locked:
                reasons.append("source_newer_than_prediction")
        except (KeyError, ValueError, TypeError):
            source_as_of = None
            reasons.append("source_timestamp_unverified")
        probability = r.get("probability")
        if not _valid_probability(probability):
            reasons.append("independent_probability_unavailable")
        if report.get("coverage", {}).get("scan_complete") is not True or report.get("coverage", {}).get("catalog_complete") is not True:
            reasons.append("incomplete_discovery_or_scan")
        if r.get("quote", {}).get("reasons"):
            reasons.append("unverified_executable_quote")
        result = settlements.get((scan_id, ticker))
        verified = result is not None and result.get("settlement_verified") is True
        won = None
        settled_at = None
        if verified and result.get("result") in ("yes", "no"):
            try:
                settled_at = utc(result["settled_at"])
                if event is None or locked is None or settled_at < event or settled_at <= locked:
                    reasons.append("noncausal_settlement")
                else:
                    won = result["result"] == "yes"
            except (ValueError, TypeError, KeyError):
                reasons.append("settlement_time_unverified")
        elif verified and result.get("result") in ("void", "fair_price"):
            reasons.append("nonbinary_settlement_excluded")
        else:
            reasons.append("awaiting_verified_binary_settlement")
        if not ticker or not model:
            reasons.append("market_or_model_identity_missing")
        if r.get("context", {}).get("event_id") is None:
            reasons.append("verified_event_identity_missing")
        # The on-demand app does not sign/externally anchor this record or
        # preregister a frozen statistical comparator. Never manufacture one.
        reasons += ["external_prediction_timestamp_attestation_missing",
                    "independent_baseline_registration_missing",
                    "underlying_source_authentication_pending"]
        prediction_id = hashlib.sha256(
            json.dumps([scan_id, ticker, model, outcome, r.get("observed_at")],
                       separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        prepared.append({
            "prediction_id": prediction_id, "scan_id": scan_id,
            "sport": r.get("sport"), "family": r.get("family"),
            "model_version": model, "ticker": ticker,
            "event_id": r.get("context", {}).get("event_id"),
            "outcome": outcome, "side": "YES",
            "locked_at": r.get("observed_at"), "event_time": r.get("event_time"),
            "source_as_of": r.get("source_as_of"),
            "source_reference": r.get("context", {}).get("source_urls", []),
            "model_probability": probability,
            "baseline_probability": None,
            "observed_yes_ask": r.get("quote", {}).get("ask"),
            "fee_per_contract": r.get("quote", {}).get("entry_fee_per_contract"),
            "settled_at": settled_at.isoformat() if settled_at is not None else None,
            "settlement_source_url": result.get("source_url") if result else None,
            "won": won,
            "source_verified": False,
            "settlement_verified": bool(won is not None and verified),
            "status": "PROSPECTIVE_REVIEW_PENDING",
            "evaluation_kind": "unattested_on_demand_prospective_archive",
            "promotion_eligible": False,
            "reasons": list(dict.fromkeys(reasons))
        })
    counts = Counter(r["sport"] for r in prepared)
    graded = sum(r["won"] is not None for r in prepared)
    return {
        "schema_version": "prospective-review-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "NONE_UNTIL_EXTERNAL_TIMESTAMP_AND_SOURCE_REVIEW",
        "read_only": True, "qualified_for_live": False,
        "coverage": {"original_scans": len(scans), "canonical_predictions": len(prepared),
                     "repeat_scans_excluded": duplicates, "officially_graded": graded,
                     "ungraded": len(prepared)-graded, "by_sport": dict(sorted(counts.items()))},
        "records": prepared,
        "blockers": [
            "Downloaded or local-only evidence has no independent timestamp attestation",
            "Future predictions and outcomes must be collected and verified before promotion",
            "Calibration comparator requires pre-registration",
            "No score or realized profit is inferred from a market quote",
        ],
    }
