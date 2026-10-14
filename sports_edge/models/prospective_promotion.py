"""Strict prospective-only sport-model promotion assessment.

No retrospective replay or unlabeled/unaudited predictions may authorize a
recommended Kalshi leg. Promotion is advisory; no trading side effects.
"""
from __future__ import annotations
from datetime import datetime, timezone
from math import isfinite, sqrt

def _time(value: str) -> datetime:
    x = datetime.fromisoformat(value.replace("Z","+00:00"))
    if x.tzinfo is None:
        raise ValueError("timezone required")
    return x

def assess_prospective_promotion(rows: list[dict], *, min_predictions: int = 300,
                                 min_distinct_dates: int = 15, max_ece: float = .06) -> dict:
    """Require causal, uniquely locked, officially settled, preregistered predictions.

    Paired Brier improvement must clear a 95% normal confidence bound over
    a fixed baseline. A real deployment must additionally authenticate source
    provenance; caller-supplied strings alone are not cryptographic proof.
    """
    if min_predictions < 100 or min_distinct_dates < 5:
        raise ValueError("unsafe validation threshold")
    invalid = []
    valid = []
    seen = set()
    prediction_ids = set()
    versions = set()
    sports = set()
    outcomes = set()
    for idx, r in enumerate(rows):
        try:
            if r.get("evaluation_kind") != "prospective" or r.get("status") != "IMMUTABLE_LOCK":
                raise ValueError("not a locked prospective prediction")
            if not all(r.get(k) for k in ("prediction_id","source_id","settlement_source_id",
                                          "model_version","event_id","sport","outcome")):
                raise ValueError("missing source or prediction identity")
            if r["prediction_id"] in prediction_ids:
                raise ValueError("duplicate immutable prediction id")
            prediction_ids.add(r["prediction_id"])
            key = (r["sport"],r["event_id"],r["outcome"],r["model_version"])
            if key in seen:
                raise ValueError("duplicate prediction")
            seen.add(key)
            locked, event, settled = (_time(r[k]) for k in ("locked_at","event_time","settled_at"))
            source_as_of = _time(r["source_as_of"])
            if not source_as_of <= locked < event <= settled <= datetime.now(timezone.utc):
                raise ValueError("future leakage or noncausal settlement")
            p, baseline = float(r["model_probability"]),float(r["baseline_probability"])
            if not all(isfinite(v) and 0 <= v <= 1 for v in (p,baseline)) or type(r.get("won")) is not bool:
                raise ValueError("invalid probability or outcome")
            if r.get("settlement_verified") is not True or r.get("source_verified") is not True:
                raise ValueError("unverified official source")
            valid.append((p,baseline,int(r["won"]),event.date()))
            sports.add(r["sport"])
            versions.add(r["model_version"])
            outcomes.add(r["outcome"])
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            invalid.append((idx,str(exc)))
    if invalid or len(sports) != 1 or len(versions) != 1 or len(outcomes) != 1:
        return {"qualified":False,"reason":"invalid_or_mixed_prospective_evidence",
                "graded":len(valid),"rejected":len(invalid)}
    n = len(valid)
    if n < min_predictions or len({r[3] for r in valid}) < min_distinct_dates:
        return {"qualified":False,"reason":"insufficient_prospective_coverage",
                "graded":n,"distinct_dates":len({r[3] for r in valid})}
    differences = [(base-y)**2-(p-y)**2 for p,base,y,_ in valid]
    mean = sum(differences)/n
    # Cluster by event date; player-game ladders on the same slate are not
    # independent validation observations.
    dates = {v[3] for v in valid}
    cluster_sums = [sum(d-mean for d,v in zip(differences,valid) if v[3] == day) for day in dates]
    cluster_var = len(dates)/(len(dates)-1)*sum(x*x for x in cluster_sums)/(n*n)
    improvement_lower95 = mean-1.96*sqrt(cluster_var)
    brier = sum((p-y)**2 for p,_,y,_ in valid)/n
    baseline_brier = sum((base-y)**2 for _,base,y,_ in valid)/n
    buckets = [[] for _ in range(10)]
    for p,_,y,_ in valid:
        buckets[min(9,int(p*10))].append((p,y))
    ece = sum(len(b)/n*abs(sum(p for p,_ in b)/len(b)-sum(y for _,y in b)/len(b))
              for b in buckets if b)
    positive_count = sum(y for _,_,y,_ in valid)
    qualified = improvement_lower95 > 0 and ece <= max_ece and min(positive_count,n-positive_count) >= 20
    return {"qualified":False,"statistical_gate_passed":qualified,
            "reason":"statistical_gate_passed_requires_authenticated_source_review"
            if qualified else "calibration_or_baseline_improvement_failed",
            "sport":next(iter(sports)),"model_version":next(iter(versions)),
            "outcome":next(iter(outcomes)),"positive_count":positive_count,
            "graded":n,"brier":brier,"baseline_brier":baseline_brier,
            "paired_brier_improvement_lower95":improvement_lower95,"ece":ece,
            "source_authentication_required":True}
