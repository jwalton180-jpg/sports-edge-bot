from __future__ import annotations
import csv, os
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ["timestamp_utc","sport","market","selection","market_price","fair_probability","edge_points","model_version","result","closing_price"]

def append_paper_pick(path: str | os.PathLike, row: dict) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists()
    payload = {k: row.get(k, "") for k in FIELDS}
    payload["timestamp_utc"] = payload["timestamp_utc"] or datetime.now(timezone.utc).isoformat()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists: w.writeheader()
        w.writerow(payload)
