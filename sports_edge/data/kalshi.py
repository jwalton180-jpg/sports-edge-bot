from __future__ import annotations
import base64, os, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from .http import HttpClient
from sports_edge.core.types import MarketQuote

DEFAULT_BASE = "https://external-api.kalshi.com/trade-api/v2"
DEFAULT_WS = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"

class KalshiPublicClient:
    def __init__(self, base_url: str | None = None, http: HttpClient | None = None):
        self.base = (base_url or os.getenv("KALSHI_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.http = http or HttpClient()

    def exchange_status(self):
        return self.http.get_json(f"{self.base}/exchange/status")

    def sports_filters(self):
        return self.http.get_json(f"{self.base}/search/filters/sports")

    def markets(self, status: str = "open", limit: int = 200, cursor: str | None = None, series_ticker: str | None = None):
        params = {"status": status, "limit": min(limit, 200)}
        if cursor: params["cursor"] = cursor
        if series_ticker: params["series_ticker"] = series_ticker
        return self.http.get_json(f"{self.base}/markets", params=params)

    def series_list(
        self,
        category: str | None = None,
        *,
        include_volume: bool = True,
        min_updated_ts: int | None = None,
    ):
        params = {}
        if category:
            params["category"] = category
        if include_volume:
            params["include_volume"] = "true"
        if min_updated_ts is not None:
            params["min_updated_ts"] = int(min_updated_ts)
        return self.http.get_json(f"{self.base}/series", params=params or None)

    def market(self, ticker: str):
        return self.http.get_json(f"{self.base}/markets/{ticker}")

    def orderbook(self, ticker: str, depth: int | None = None):
        params = {"depth": depth} if depth else None
        return self.http.get_json(f"{self.base}/markets/{ticker}/orderbook", params=params)

    def events(self, status: str = "open", limit: int = 200, cursor: str | None = None, with_nested_markets: bool = True):
        params = {"status": status, "limit": min(limit, 200), "with_nested_markets": str(with_nested_markets).lower()}
        if cursor: params["cursor"] = cursor
        return self.http.get_json(f"{self.base}/events", params=params)

    def game_stats(self, milestone_id: str):
        return self.http.get_json(f"{self.base}/live_data/milestone/{milestone_id}/game_stats")

    def event_live_data(self, event_ticker: str, range_hint: str | None = None):
        params = {"range": range_hint} if range_hint else None
        return self.http.get_json(f"{self.base}/live_data/events/{event_ticker}", params=params)

    def milestone_live_data(self, milestone_id: str):
        return self.http.get_json(f"{self.base}/live_data/milestone/{milestone_id}")

    @staticmethod
    def _dollars(v):
        try:
            if v is None: return None
            x = float(v)
            return x / 100.0 if x > 1.0 else x
        except Exception:
            return None

    def quote_from_market(self, m: dict, sport: str = "Sports") -> MarketQuote:
        return MarketQuote(ticker=m.get("ticker", ""), sport=sport, title=m.get("title") or m.get("subtitle") or m.get("ticker", ""), yes_bid=self._dollars(m.get("yes_bid_dollars", m.get("yes_bid"))), yes_ask=self._dollars(m.get("yes_ask_dollars", m.get("yes_ask"))), no_bid=self._dollars(m.get("no_bid_dollars", m.get("no_bid"))), no_ask=self._dollars(m.get("no_ask_dollars", m.get("no_ask"))), last_price=self._dollars(m.get("last_price_dollars", m.get("last_price"))), volume=float(m.get("volume_fp", m.get("volume", 0)) or 0), open_interest=float(m.get("open_interest_fp", m.get("open_interest", 0)) or 0), observed_at=_parse_ts(m.get("updated_time") or m.get("last_updated_ts")), raw=m)

def _parse_ts(value):
    if not value: return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None

class KalshiSigner:
    def __init__(self, access_key: str, private_key_path: str):
        from cryptography.hazmat.primitives import serialization
        self.access_key = access_key
        self.private_key = serialization.load_pem_private_key(Path(private_key_path).read_bytes(), password=None)
    def headers(self, method: str, path_with_query: str) -> dict[str, str]:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        ts = str(int(time.time() * 1000)); path = path_with_query.split("?", 1)[0]
        msg = f"{ts}{method.upper()}{path}".encode()
        sig = self.private_key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        return {"KALSHI-ACCESS-KEY": self.access_key, "KALSHI-ACCESS-TIMESTAMP": ts, "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}
