from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os


@dataclass(frozen=True)
class LineSnapshot:
    observed_at: datetime
    source: str
    event_id: str
    market_key: str
    selection: str
    price: float
    is_executable: bool = True

    def normalized(self) -> "LineSnapshot":
        ts = self.observed_at if self.observed_at.tzinfo else self.observed_at.replace(tzinfo=timezone.utc)
        p = float(self.price)
        p = p / 100.0 if p > 1 else p
        if not 0 < p < 1:
            raise ValueError("price must be strictly between 0 and 1")
        for name, value in (
            ("source", self.source),
            ("event_id", self.event_id),
            ("market_key", self.market_key),
            ("selection", self.selection),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} required")
        return LineSnapshot(
            ts.astimezone(timezone.utc),
            str(self.source),
            str(self.event_id),
            str(self.market_key),
            str(self.selection),
            p,
            bool(self.is_executable),
        )


def snapshot_id(snapshot: LineSnapshot) -> str:
    n = snapshot.normalized()
    raw = "|".join(
        (
            n.observed_at.isoformat(),
            n.source,
            n.event_id,
            n.market_key,
            n.selection,
            f"{n.price:.8f}",
            str(int(n.is_executable)),
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def append_line_snapshot(path: str | os.PathLike, snapshot: LineSnapshot) -> dict:
    n = snapshot.normalized()
    sid = snapshot_id(n)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line).get("snapshot_id") == sid:
                        return {"snapshot_id": sid, "appended": False}
                except (json.JSONDecodeError, TypeError):
                    raise ValueError("corrupt line-history evidence")
    rec = {
        "snapshot_id": sid,
        "observed_at": n.observed_at.isoformat(),
        "source": n.source,
        "event_id": n.event_id,
        "market_key": n.market_key,
        "selection": n.selection,
        "price": n.price,
        "is_executable": n.is_executable,
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
    return {"snapshot_id": sid, "appended": True}


def closing_line_report(
    snapshots: list[LineSnapshot],
    *,
    entry_time: datetime,
    event_start: datetime,
    entry_price: float,
    max_prestart_age_s: float = 900.0,
) -> dict:
    et = entry_time if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)
    st = event_start if event_start.tzinfo else event_start.replace(tzinfo=timezone.utc)
    et = et.astimezone(timezone.utc)
    st = st.astimezone(timezone.utc)
    if et >= st:
        raise ValueError("entry must precede event start")
    ep = float(entry_price)
    ep = ep / 100 if ep > 1 else ep
    if not 0 < ep < 1:
        raise ValueError("entry_price must be strictly between 0 and 1")

    valid = []
    for raw in snapshots:
        s = raw.normalized()
        if s.is_executable and et <= s.observed_at <= st:
            valid.append(s)
    if not valid:
        return {"eligible": False, "rejections": ["missing_prestart_closing_line"]}

    close = max(valid, key=lambda x: x.observed_at)
    age = (st - close.observed_at).total_seconds()
    if age > max_prestart_age_s:
        return {
            "eligible": False,
            "rejections": ["stale_closing_line"],
            "closing_age_s": age,
        }
    return {
        "eligible": True,
        "rejections": [],
        "entry_price": ep,
        "closing_price": close.price,
        "clv_probability_points": close.price - ep,
        "closing_observed_at": close.observed_at.isoformat(),
        "closing_age_s": age,
        "source": close.source,
    }
