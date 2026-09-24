from __future__ import annotations
from .http import HttpClient

# Unofficial ESPN public endpoint used only as a failover/cross-check; production priority is Kalshi live stats where available.
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

class NFLClient:
    def __init__(self, http: HttpClient | None = None): self.http = http or HttpClient()
    def scoreboard(self, dates: str | None = None):
        params = {"dates": dates} if dates else None
        return self.http.get_json(f"{ESPN_BASE}/scoreboard", params=params)
    def summary(self, event_id: str):
        return self.http.get_json("https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary", params={"event": event_id})
