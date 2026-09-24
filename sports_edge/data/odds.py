from __future__ import annotations
import os
from .http import HttpClient
from sports_edge.core.math import american_to_implied, no_vig_two_way

BASE = "https://api.the-odds-api.com/v4"

class OddsClient:
    def __init__(self, api_key: str | None = None, http: HttpClient | None = None):
        self.api_key = api_key or os.getenv("THE_ODDS_API_KEY")
        self.http = http or HttpClient()
    @property
    def configured(self): return bool(self.api_key)
    def sports(self):
        headers = {"x-api-key": self.api_key} if self.api_key else None
        return self.http.get_json(f"{BASE}/sports", params={"apiKey": self.api_key} if self.api_key else None)
    def odds(self, sport_key: str, markets: str = "h2h", regions: str = "us"):
        if not self.api_key: raise RuntimeError("THE_ODDS_API_KEY is not configured")
        return self.http.get_json(f"{BASE}/sports/{sport_key}/odds", params={"regions": regions, "markets": markets, "oddsFormat": "american", "apiKey": self.api_key})
    def events(self, sport_key: str):
        if not self.api_key: raise RuntimeError("THE_ODDS_API_KEY is not configured")
        return self.http.get_json(f"{BASE}/sports/{sport_key}/events", params={"apiKey": self.api_key})
    def event_odds(self, sport_key: str, event_id: str, markets: str, regions: str = "us", bookmakers: str | None = None):
        if not self.api_key: raise RuntimeError("THE_ODDS_API_KEY is not configured")
        params={"regions":regions,"markets":markets,"oddsFormat":"american","apiKey":self.api_key}
        if bookmakers: params["bookmakers"]=bookmakers
        return self.http.get_json(f"{BASE}/sports/{sport_key}/events/{event_id}/odds", params=params)

def two_way_book_fair(outcome_a_odds: float, outcome_b_odds: float) -> tuple[float, float]:
    return no_vig_two_way(american_to_implied(outcome_a_odds), american_to_implied(outcome_b_odds))
