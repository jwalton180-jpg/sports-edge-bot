from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    required: bool
    latency_ms: float | None = None
    detail: str = ""

@dataclass(frozen=True)
class StartupReport:
    mode: str
    generated_at: float
    probes: tuple[ProbeResult, ...]
    @property
    def ready(self) -> bool: return all(p.ok for p in self.probes if p.required)
    @property
    def degraded(self) -> bool: return self.ready and any(not p.ok for p in self.probes if not p.required)

def deployment_mode() -> str:
    mode=os.getenv("SPORTS_EDGE_MODE","on_demand").strip().lower()
    return mode if mode in {"on_demand","always_on"} else "on_demand"

def session_cache_dir() -> Path:
    configured=os.getenv("SPORTS_EDGE_SESSION_CACHE")
    root=Path(configured) if configured else Path(tempfile.gettempdir())/"sports_edge_session"
    root.mkdir(parents=True,exist_ok=True)
    return root

def _probe(name: str, required: bool, fn: Callable[[], Any]) -> ProbeResult:
    start=time.perf_counter()
    try:
        fn(); return ProbeResult(name=name,ok=True,required=required,latency_ms=(time.perf_counter()-start)*1000)
    except Exception as exc:
        return ProbeResult(name=name,ok=False,required=required,latency_ms=(time.perf_counter()-start)*1000,detail=str(exc)[:180])

def run_startup_checks(kalshi=None, mlb=None, nfl=None, odds=None) -> StartupReport:
    probes=[]
    if kalshi is not None: probes.append(_probe("Kalshi markets",True,lambda:kalshi.markets(status="open",limit=1)))
    if mlb is not None:
        from datetime import datetime, timezone
        probes.append(_probe("MLB feed",False,lambda:mlb.schedule(datetime.now(timezone.utc).date())))
    if nfl is not None: probes.append(_probe("NFL feed",False,nfl.scoreboard))
    if odds is not None:
        configured=bool(getattr(odds,"configured",False))
        probes.append(ProbeResult("Sportsbook odds",configured,False,detail="configured" if configured else "optional API key not configured"))
    return StartupReport(mode=deployment_mode(),generated_at=time.time(),probes=tuple(probes))
