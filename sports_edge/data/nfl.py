from __future__ import annotations
from .http import HttpClient

# Public ESPN score/summary feed used only as a failover/cross-check.
# Kalshi live-data remains preferred when a verified milestone mapping exists.
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SportsEdgeReadOnly/1.0; +https://streamlit.io)",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.espn.com/",
}

class NFLClient:
    def __init__(self, http: HttpClient | None = None):
        self.http = http or HttpClient()

    def scoreboard(self, dates: str | None = None):
        params = {"dates": dates} if dates else None
        return self.http.get_json(f"{ESPN_BASE}/scoreboard", params=params, headers=ESPN_HEADERS)

    def summary(self, event_id: str):
        return self.http.get_json(
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary",
            params={"event": event_id},
            headers=ESPN_HEADERS,
        )
