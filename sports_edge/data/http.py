from __future__ import annotations
import time
from dataclasses import dataclass
from datetime import datetime, timezone
import requests

@dataclass
class HttpResult:
    data: dict | list
    received_at: datetime
    latency_ms: float
    status_code: int
    headers: dict

class HttpClient:
    def __init__(self, user_agent: str = "SportsEdgeBot/0.1", timeout: float = 8.0):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self.timeout = timeout

    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None) -> HttpResult:
        t0 = time.perf_counter()
        r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        latency = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        return HttpResult(r.json(), datetime.now(timezone.utc), latency, r.status_code, dict(r.headers))
