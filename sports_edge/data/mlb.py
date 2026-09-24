from __future__ import annotations
from datetime import date
from .http import HttpClient

BASE = "https://statsapi.mlb.com/api/v1"

class MLBClient:
    def __init__(self, http: HttpClient | None = None): self.http = http or HttpClient()
    def schedule(self, day: str | date):
        d = day.isoformat() if hasattr(day, "isoformat") else str(day)
        return self.http.get_json(f"{BASE}/schedule", params={"sportId": 1, "date": d, "hydrate": "probablePitcher,team,linescore"})
    def live_feed(self, game_pk: int):
        return self.http.get_json(f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live")
    def boxscore(self, game_pk: int):
        return self.http.get_json(f"{BASE}/game/{int(game_pk)}/boxscore")
