"""Brief independent GitHub Actions pre-event sampling; no always-on worker.

Public, read-only Kalshi and sports APIs only. A GitHub artifact timestamp is
external evidence of archival time, not proof of data-source authenticity.
Never places orders, loads Kalshi credentials, or promotes a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

from sports_edge.models.sleeper_counts import utc

PROTOCOL = "SPORTSEDGE-PROSPECTIVE-V1-20261014"
SPORTS = ("NHL", "NCAAF", "Soccer")
END_UTC = datetime(2026, 12, 15, tzinfo=timezone.utc)


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def select_markets(markets, *, now: datetime, per_sport: int = 14,
                   horizon_hours: int = 60):
    """Frozen deterministic sample of pregame listings, not a cherry-picked slate."""
    if now.tzinfo is None or per_sport < 1 or per_sport > 30:
        raise ValueError("invalid sampling configuration")
    picked, inventory, by_sport = [], {}, {s: [] for s in SPORTS}
    for m in markets:
        sport = m.get("sports_edge_sport")
        if sport not in by_sport:
            continue
        inventory[sport] = inventory.get(sport, 0) + 1
        try:
            start = utc(m["occurrence_datetime"])
        except (KeyError, TypeError, ValueError):
            continue
        if not now < start <= now + timedelta(hours=horizon_hours):
            continue
        ticker = m.get("ticker")
        if not ticker or m.get("status") not in ("active", "open"):
            continue
        by_sport[sport].append(m)
    for sport in SPORTS:
        # Hash selection is preregistered and independent of price, model,
        # game winner, and apparent value. Selection is not full coverage.
        unique = {m["ticker"]: m for m in by_sport[sport]}
        ranked = sorted(unique.values(), key=lambda m: (
            hashlib.sha256((PROTOCOL + "|" + m["ticker"]).encode()).hexdigest(),
            m["ticker"]))
        picked.extend(ranked[:per_sport])
    return picked, {"inventory_by_sport": inventory,
                    "eligible_by_sport": {s:len({m["ticker"] for m in by_sport[s]}) for s in SPORTS},
                    "sample_by_sport": {s:sum(m["sports_edge_sport"] == s for m in picked) for s in SPORTS},
                    "sampling": "SHA256(protocol|ticker), earliest 14 per sport",
                    "subset_not_full_market_coverage": True,
                    "protocol": PROTOCOL}


def capture(*, now=None, output_dir="prospective-out"):
    from sports_edge.data.kalshi_catalog import fetch_supported_sport_catalog
    from sports_edge.models.sleeper_pipeline import scan_sleeper_markets, export_evidence
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("UTC timestamp required")
    if current >= END_UTC:
        return {"status": "collection_period_ended", "snapshot_written": False}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    catalog = fetch_supported_sport_catalog(
        sports=SPORTS, overview_only=False,
        discover_all_series=False, page_size=200,
        page_limit_per_series=6, max_workers=4)
    selected, selection = select_markets(catalog.markets, now=current)
    # Full-coverage claims are always false for a bounded prospective sample.
    report = scan_sleeper_markets(selected, coverage_complete=False,
                                 limit=42, now=current)
    report["sampling"] = selection
    report["catalog_status"] = {
        "cursor_exhausted": bool(catalog.cursor_exhausted),
        "error": catalog.error,
        "pages": catalog.pages,
        "listed_markets": len(catalog.markets)}
    report["study"] = {
        "protocol": PROTOCOL, "evaluation_kind": "prospective_research_snapshot",
        "promoted": False, "trading": False, "archived_by": "GitHub Actions artifact",
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "commit": os.environ.get("GITHUB_SHA"),
        "source_authentication": "pending_review",
        "baseline": "YES executable ask when available; never fed to model"}
    payload = json.loads(export_evidence(report))
    payload.update({k:report[k] for k in ("sampling","catalog_status","study")})
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    # Avoid empty/projectionless captures being mislabeled model validation.
    model_count = sum(r.get("side") == "YES" and r.get("probability") is not None
                      for r in report["records"])
    manifest = {
        "schema_version": PROTOCOL,
        "capture_sha256": digest,
        "captured_at": current.isoformat(),
        "git_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "model_predictions": model_count,
        "no_promotions": True, "read_only": True,
        "market_sample": selection}
    (output/"snapshot.json").write_bytes(canonical_bytes(payload))
    (output/"manifest.json").write_bytes(canonical_bytes(manifest))
    print(json.dumps({"model_predictions": model_count, "selected":len(selected),
                      "digest":digest, "catalog_incomplete":not catalog.cursor_exhausted,
                      "catalog_error": catalog.error}, sort_keys=True))
    return {"snapshot_written": True, "manifest": manifest}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",default="prospective-out")
    args=parser.parse_args()
    result=capture(output_dir=args.output_dir)
    print(json.dumps(result, default=str))
    # No trade, runtime or promotion. Fail if HTTP inventory incomplete:
    # the artifact itself still holds the diagnostic record.
    if result.get("snapshot_written") and result["manifest"]["model_predictions"] == 0:
        print("No independent predictions available in sample; no qualification.")


if __name__ == "__main__":
    main()
