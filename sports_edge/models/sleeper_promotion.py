"""Scoped, expiring promotion reviews bound to immutable prospective evidence.

Default: no reviewed models. Historical results never authorize recommendations.
A review signature is supplied by the deployment operator after source review;
the scanner cannot create one or read/emit the review signing key.
"""
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
from sports_edge.models.prospective_promotion import assess_prospective_promotion
from sports_edge.models.sleeper_counts import utc


@dataclass(frozen=True)
class PromotionScope:
    sport: str
    family: str
    model: str
    outcome: str
    approved_at: str
    expires_at: str


def load_reviewed_promotions(path, *, verification_key, now):
    """Reject unsigned, edited, mixed-scope, expired or failed prospective reviews."""
    if not verification_key or not Path(path).exists():
        return {}
    result={}
    try:
        reviews=json.loads(Path(path).read_text())
        for envelope in reviews:
            payload=envelope["payload"]
            encoded=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
            signature=hmac.new(verification_key.encode(),encoded,hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature,envelope["signature"]):
                continue
            rows=payload["prospective_rows"]
            digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
            if digest!=payload["prospective_sha256"] or payload.get("source_authenticated") is not True:
                continue
            scope=PromotionScope(**payload["scope"])
            if not utc(scope.approved_at)<=utc(now)<utc(scope.expires_at) or not payload.get("source_review_url","").startswith("https://github.com/"):
                continue
            gate=assess_prospective_promotion(rows)
            if (gate.get("statistical_gate_passed") is True and gate.get("sport")==scope.sport
                and gate.get("model_version")==scope.model and gate.get("outcome")==scope.outcome):
                result[(scope.sport,scope.family,scope.model,scope.outcome)]=scope
    except (ValueError, TypeError, KeyError, OSError):
        return {}
    return result
