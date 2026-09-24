from __future__ import annotations

from typing import Any

from .http import HttpClient, HttpResult

# Public ESPN score/summary feeds used only as failover/cross-check sources.
# Kalshi live-data remains preferred when a verified milestone mapping exists.
ESPN_SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_CDN_SCOREBOARD = "https://cdn.espn.com/core/nfl/scoreboard"
ESPN_CDN_GAME = "https://cdn.espn.com/core/nfl/game"
ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SportsEdgeReadOnly/1.0; +https://streamlit.io)",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.espn.com/",
}


def _find_event_list(node: Any) -> list[dict] | None:
    """Find an ESPN-style event list inside a nested CDN payload."""
    if isinstance(node, dict):
        events = node.get("events")
        if isinstance(events, list):
            usable = [e for e in events if isinstance(e, dict) and isinstance(e.get("competitions"), list)]
            if usable:
                return usable
        for value in node.values():
            found = _find_event_list(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_event_list(value)
            if found:
                return found
    return None


class NFLClient:
    def __init__(self, http: HttpClient | None = None):
        self.http = http or HttpClient()

    def scoreboard(self, dates: str | None = None):
        params = {"dates": dates} if dates else None
        try:
            return self.http.get_json(
                f"{ESPN_SITE_BASE}/scoreboard",
                params=params,
                headers=ESPN_HEADERS,
            )
        except Exception as primary_error:
            # ESPN's site-api intermittently returns 403 from cloud data centers.
            # The public CDN endpoint is used as a read-only fallback and is
            # normalized back to the same top-level {"events": [...]} shape.
            cdn_params = {"xhr": "1"}
            if dates:
                cdn_params["dates"] = dates
            fallback = self.http.get_json(
                ESPN_CDN_SCOREBOARD,
                params=cdn_params,
                headers=ESPN_HEADERS,
            )
            events = _find_event_list(fallback.data)
            if not events:
                raise RuntimeError("NFL public fallback returned no recognizable events") from primary_error
            return HttpResult(
                data={"events": events, "_source": "espn_cdn"},
                received_at=fallback.received_at,
                latency_ms=fallback.latency_ms,
                status_code=fallback.status_code,
                headers=fallback.headers,
            )

    def summary(self, event_id: str):
        try:
            return self.http.get_json(
                f"{ESPN_SITE_BASE}/summary",
                params={"event": event_id},
                headers=ESPN_HEADERS,
            )
        except Exception:
            # Rich public game package. This payload is intentionally returned
            # raw because downstream live fusion should parse only verified
            # fields it explicitly needs.
            return self.http.get_json(
                ESPN_CDN_GAME,
                params={"xhr": "1", "gameId": event_id},
                headers=ESPN_HEADERS,
            )
