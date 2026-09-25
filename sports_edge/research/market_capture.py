from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import json, os
from pathlib import Path
from sports_edge.core.math import clamp

@dataclass(frozen=True)
class MarketObservation:
    observed_at: datetime
    ticker: str
    yes_bid: float | None
    yes_ask: float | None
    last: float | None
    source: str = "kalshi_ws"
    sequence: int | None = None

    @property
    def executable_yes(self) -> float | None:
        return self.yes_ask


def _prob(v) -> float | None:
    if v is None: return None
    x=float(v)
    if x > 1.0: x /= 100.0
    if not 0 <= x <= 1: return None
    return x


def observation_from_ws(payload: dict, received_at: datetime) -> MarketObservation | None:
    msg=payload.get("msg") if isinstance(payload.get("msg"),dict) else payload
    ticker=msg.get("market_ticker") or msg.get("ticker")
    if not ticker: return None
    bid=_prob(msg.get("yes_bid")); ask=_prob(msg.get("yes_ask"))
    last=_prob(msg.get("price") if msg.get("price") is not None else msg.get("last_price"))
    if bid is not None and ask is not None and bid > ask:
        return None
    return MarketObservation(received_at.astimezone(timezone.utc), str(ticker), bid, ask, last,
                             sequence=int(payload["seq"]) if payload.get("seq") is not None else None)


def prospective_value_record(obs: MarketObservation, fair_probability: float, *, model_version: str,
                             data_quality: float, max_quote_age_s: float = 12.0,
                             evaluated_at: datetime | None = None) -> dict:
    now=(evaluated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age=max(0.0,(now-obs.observed_at).total_seconds())
    ask=obs.executable_yes
    reasons=[]
    if ask is None: reasons.append("missing_executable_yes_ask")
    if age > max_quote_age_s: reasons.append("stale_quote")
    if data_quality < .75: reasons.append("low_data_quality")
    f=clamp(float(fair_probability),.001,.999)
    edge=(f-ask) if ask is not None else None
    return {
        "observed_at":obs.observed_at.isoformat(), "evaluated_at":now.isoformat(), "ticker":obs.ticker,
        "yes_bid":obs.yes_bid, "yes_ask":ask, "last":obs.last, "fair_probability":f,
        "edge":edge, "ev_per_contract":edge, "quote_age_s":age, "data_quality":float(data_quality),
        "model_version":str(model_version), "eligible":not reasons, "rejections":reasons,
        "sequence":obs.sequence, "source":obs.source,
    }


def append_prospective_record(path: str | os.PathLike, record: dict) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf-8") as f:
        f.write(json.dumps(record,sort_keys=True,separators=(",",":"))+"\n")
