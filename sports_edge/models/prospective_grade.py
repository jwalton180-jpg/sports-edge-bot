"""Read-only validation of GitHub-hosted pregame captures against official Kalshi finals.

External run/artifact timestamps guard prospective ordering. This never
authorizes production promotion: source and settlement reviews are separate.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import zipfile

import requests
from sports_edge.models.prospective_capture import canonical_bytes, PROTOCOL
from sports_edge.models.sleeper_counts import utc

ARTIFACT_PREFIX="sportsedge-live-research-snapshot-"
SUPPORTED_PROTOCOLS = {"SPORTSEDGE-PROSPECTIVE-V1-20261014", "SPORTSEDGE-PROSPECTIVE-V2-20261014", "SPORTSEDGE-PROSPECTIVE-V3-20261014", "SPORTSEDGE-PROSPECTIVE-V4-20261014", "SPORTSEDGE-PROSPECTIVE-V5-20261014", "SPORTSEDGE-PROSPECTIVE-V6-20261014"}


def verify_snapshot(snapshot: dict, manifest: dict, artifact: dict):
    """Artifact creation/hosting time is external; local timestamps alone aren't."""
    protocol = manifest.get("schema_version")
    if protocol not in SUPPORTED_PROTOCOLS or not manifest.get("no_promotions"):
        raise ValueError("unknown protocol or attempted promotion")
    digest=hashlib.sha256(canonical_bytes(snapshot)).hexdigest()
    if digest!=manifest.get("capture_sha256"):
        raise ValueError("snapshot changed after manifest")
    official_run=artifact.get("workflow_run") or {}
    run_id=str(official_run.get("id") or "")
    if not run_id or str(manifest.get("github_run_id"))!=run_id:
        raise ValueError("unmatched GitHub Actions run identity")
    if official_run.get("head_sha")!=manifest.get("git_sha"):
        raise ValueError("provenance commit mismatch")
    created=utc(artifact["created_at"])
    captured=utc(manifest["captured_at"])
    if captured>created:
        raise ValueError("artifact claims future pre-upload capture")
    if not (0<=(created-captured).total_seconds()<=3600):
        raise ValueError("unbounded capture-to-archive delay")
    if snapshot.get("study",{}).get("protocol")!=protocol:
        raise ValueError("study protocol mismatch")
    return created


def first_archived_predictions(pairs: list[tuple[dict,dict,dict]]):
    """One first *externally archived* YES forecast per contract/model/outcome."""
    ordered=[]
    for snapshot,manifest,artifact in pairs:
        created=verify_snapshot(snapshot,manifest,artifact)
        for r in snapshot.get("records",()):
            if r.get("side")!="YES" or r.get("probability") is None:
                continue
            key=(r.get("sport"),r.get("ticker"),r.get("model"),
                 r.get("family"),r.get("threshold"))
            if not all(key[:4]):
                continue
            issues=[]
            try:
                event=utc(r["event_time"])
                observed=utc(r["observed_at"])
                # Provider receipt/source times must be prior to archive and
                # event start. Never trust self-reported observation alone.
                source=utc(r["source_as_of"])
                if not (source<=observed<=created<event):
                    issues.append("causality_or_archive_after_start")
            except (KeyError,TypeError,ValueError):
                issues.append("missing_causal_source_timestamp")
            p=r["probability"]
            if type(p) not in (int,float) or not __import__("math").isfinite(p) or not 0<=p<=1:
                issues.append("invalid_probability")
            if snapshot.get("sampling",{}).get("protocol")!=manifest["schema_version"]:
                issues.append("inconsistent_sampling_protocol")
            if r.get("quote",{}).get("reasons") or r.get("quote",{}).get("ask") is None:
                issues.append("market_baseline_unusable")
            ordered.append((created,key,r,artifact["id"],issues))
    ordered.sort(key=lambda x:(x[0],str(x[1])))
    seen=set();out=[]
    for archived,key,r,artifact_id,issues in ordered:
        if key in seen:continue
        seen.add(key)
        out.append({"sport":key[0],"ticker":key[1],"model_version":key[2],
                    "outcome":f"{key[3]}:{key[4]}:YES",
                    "archived_at":archived.isoformat(),"artifact_id":artifact_id,
                    "event_time":r.get("event_time"),
                    "prediction":r["probability"],
                    "market_ask_reference":r.get("quote",{}).get("ask"),
                    "market_fee":r.get("quote",{}).get("entry_fee_per_contract"),
                    "source_as_of":r.get("source_as_of"),
                    "source_refs":r.get("context",{}).get("source_urls",()),
                    "reasons":issues,"won":None,"settled_at":None,
                    "settlement_verified":False,"qualified_for_live":False})
    return out


def grade_official(rows: list[dict], client, *, now=None, maximum=150):
    """Use official binary settlement only; never infer results from price changes."""
    current=now or datetime.now(timezone.utc)
    if not 1<=maximum<=500:raise ValueError("bounded GET budget required")
    checked=0
    for row in rows:
        try:
            if utc(row["event_time"]) >= current:
                row["reasons"].append("event_not_started")
                continue
        except (TypeError,KeyError,ValueError):
            row["reasons"].append("event_time_unavailable")
            continue
        if row["reasons"]:
            continue
        if checked>=maximum:
            row["reasons"].append("settlement_check_budget_exhausted")
            continue
        checked+=1
        try:
            m=client.market(row["ticker"]).data["market"]
            if m.get("ticker")!=row["ticker"]:
                raise ValueError("settlement_ticker_mismatch")
            result=m.get("result")
            if m.get("status") not in ("settled","finalized") or result not in ("yes","no"):
                row["reasons"].append("official_binary_settlement_pending")
                continue
            payoff=float(m.get("settlement_value_dollars","nan"))
            if payoff!=int(result=="yes"):
                row["reasons"].append("nonbinary_or_conflicting_settlement")
                continue
            settled=m.get("settlement_ts") or m.get("settlement_time")
            if not settled or utc(settled)<utc(row["event_time"]):
                row["reasons"].append("official_settlement_timestamp_missing_or_noncausal")
                continue
            row["won"]=(result=="yes")
            row["settled_at"]=settled
            row["settlement_verified"]=True
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            row["reasons"].append("official_settlement_fetch_error:"+type(exc).__name__)
    return rows


def summarize(rows):
    by_outcome=defaultdict(list)
    for r in rows:
        if (r["won"] is not None and not r["reasons"] and
            type(r["prediction"]) in (int,float)):
            by_outcome[(r["sport"],r["model_version"],r["outcome"])].append(r)
    stats=[]
    for (sport,model,outcome),rr in sorted(by_outcome.items()):
        n=len(rr)
        p_brier=sum((r["prediction"]-int(r["won"]))**2 for r in rr)/n
        q=[r for r in rr if type(r["market_ask_reference"]) in (int,float)
           and 0<=r["market_ask_reference"]<=1]
        q_brier=sum((r["market_ask_reference"]-int(r["won"]))**2 for r in q)/len(q) if q else None
        stats.append({"sport":sport,"model":model,"outcome":outcome,
                      "n":n,"model_brier":p_brier,"market_ask_reference_brier":q_brier,
                      "distinct_dates":len({utc(r["event_time"]).date() for r in rr}),
                      "source_authenticated":False,"promoted":False})
    return {"protocol":PROTOCOL,"evaluation_kind":"independently_archived_prospective_research",
            "qualified_for_live":False,"archived_predictions":len(rows),
            "graded_unblocked":sum(x["n"] for x in stats),
            "ungraded_or_blocked":len(rows)-sum(x["n"] for x in stats),
            "by_model_and_outcome":stats,
            "limitations":["Source records not independently authenticated by a reviewer",
                           "No model may be promoted by this read-only grading job",
                           "Incomplete subsets cannot establish market-wide coverage",
                           "No verified execution or realized P&L"]}


def github_artifacts(repo,token,*,maximum=80):
    session=requests.Session()
    session.headers.update({"Accept":"application/vnd.github+json",
                            "Authorization":"Bearer "+token,"X-GitHub-Api-Version":"2022-11-28"})
    url=f"https://api.github.com/repos/{repo}/actions/artifacts"
    response=session.get(url,params={"per_page":100},timeout=25)
    response.raise_for_status()
    artifacts=[a for a in response.json().get("artifacts",[]) if
               a.get("name","").startswith(ARTIFACT_PREFIX) and not a.get("expired")]
    pairs=[]
    for artifact in sorted(artifacts,key=lambda a:a["created_at"])[:maximum]:
        z=session.get(artifact["archive_download_url"],timeout=45)
        z.raise_for_status()
        with zipfile.ZipFile(BytesIO(z.content)) as archive:
            pairs.append((json.loads(archive.read("snapshot.json")),
                          json.loads(archive.read("manifest.json")),artifact))
    return pairs


def run(*,output_dir="prospective-grade",max_artifacts=80,max_gets=150):
    from sports_edge.data.kalshi import KalshiPublicClient
    token=os.environ.get("GITHUB_TOKEN")
    repo=os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo or "/" not in repo:
        raise ValueError("GitHub Actions read token and repository required")
    pairs=github_artifacts(repo,token,maximum=max_artifacts)
    rows=first_archived_predictions(pairs)
    graded=grade_official(rows, KalshiPublicClient(),maximum=max_gets)
    report=summarize(graded)
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    (output/"assessment.json").write_bytes(canonical_bytes(report))
    (output/"outcomes.jsonl").write_text("".join(json.dumps(x,sort_keys=True,allow_nan=False)+"\n" for x in graded))
    print(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    run()
