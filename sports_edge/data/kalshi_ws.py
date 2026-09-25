from __future__ import annotations
import asyncio, json, os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Iterable
from .kalshi import KalshiSigner, DEFAULT_WS

@dataclass(frozen=True)
class WsEnvelope:
    received_at: datetime
    payload: dict

class SequenceGap(RuntimeError):
    pass

class SequenceGuard:
    """Tracks server sequence numbers per subscription id. False means data cannot be trusted until resynced."""
    def __init__(self):
        self.last: dict[int, int] = {}
        self.broken: set[int] = set()
    def observe(self, sid: int, seq: int) -> bool:
        sid, seq = int(sid), int(seq)
        prev=self.last.get(sid)
        ok=prev is None or seq == prev + 1
        if not ok: self.broken.add(sid)
        self.last[sid]=seq
        return ok
    def reset(self, sid: int, seq: int | None = None):
        sid=int(sid); self.broken.discard(sid)
        if seq is None: self.last.pop(sid,None)
        else: self.last[sid]=int(seq)

def subscription_message(request_id: int, channels: Iterable[str], market_tickers: Iterable[str]) -> dict:
    tickers=list(dict.fromkeys(str(t) for t in market_tickers if t))
    if not tickers: raise ValueError("At least one market ticker is required")
    return {"id":int(request_id),"cmd":"subscribe","params":{"channels":list(channels),"market_tickers":tickers,"send_initial_snapshot":True}}

class KalshiReadOnlyWebSocket:
    """Authenticated read-only market stream. No order-submit/cancel methods exist here."""
    def __init__(self, access_key: str | None = None, private_key_path: str | None = None, ws_url: str | None = None):
        self.access_key = access_key or os.getenv("KALSHI_ACCESS_KEY")
        self.private_key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH")
        self.ws_url = ws_url or os.getenv("KALSHI_WS_URL") or DEFAULT_WS
        if not self.access_key or not self.private_key_path:
            raise RuntimeError("KALSHI_ACCESS_KEY and KALSHI_PRIVATE_KEY_PATH are required for WebSocket market data")
        self.signer = KalshiSigner(self.access_key, self.private_key_path)
        self.sequence = SequenceGuard()

    def _headers(self):
        return self.signer.headers("GET", "/trade-api/ws/v2")

    def validate_sequence(self, payload: dict) -> None:
        sid, seq = payload.get("sid"), payload.get("seq")
        if sid is None or seq is None: return
        if not self.sequence.observe(int(sid), int(seq)):
            raise SequenceGap(f"sequence gap sid={sid} seq={seq}")

    async def stream_once(self, market_tickers: Iterable[str], callback: Callable[[WsEnvelope], Awaitable[None]],
                          channels: tuple[str, ...] = ("ticker","trade","orderbook_delta")) -> None:
        import websockets
        headers=self._headers(); msg=subscription_message(1, channels, market_tickers)
        async with websockets.connect(self.ws_url, additional_headers=headers, ping_interval=10, ping_timeout=20, max_queue=4096) as ws:
            await ws.send(json.dumps(msg))
            async for raw in ws:
                payload=json.loads(raw)
                self.validate_sequence(payload)
                await callback(WsEnvelope(datetime.now(timezone.utc), payload))

    async def stream(self, market_tickers: Iterable[str], callback: Callable[[WsEnvelope], Awaitable[None]],
                     channels: tuple[str, ...] = ("ticker","trade","orderbook_delta"),
                     reconnect_min_s: float = 0.5, reconnect_max_s: float = 8.0) -> None:
        delay=max(.1,float(reconnect_min_s))
        while True:
            try:
                self.sequence=SequenceGuard()
                await self.stream_once(market_tickers, callback, channels)
                delay=max(.1,float(reconnect_min_s))
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(delay)
                delay=min(float(reconnect_max_s), delay*2)
