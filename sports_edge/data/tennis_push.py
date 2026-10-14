"""Optional, read-only iPhone push delivery for SportsEdge Tennis reversals.

ntfy topics on ntfy.sh are *not encrypted/private*: use a random topic and
publish only public market observations. Neither tokens nor topic URLs are
ever logged or persisted in the Git research ledger. HTTP acceptance does not
guarantee that the phone displayed the notification.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
from urllib.parse import urlparse

import requests


SUPPORTED = {
    "EXTREME DIP — TRACKING ONLY": (2, "Cheap dip • TRACK ONLY"),
    "PRICE RECOVERY — TRACKING ONLY": (2, "Market price recovery • TRACK ONLY"),
    "RECOVERY BUILDING — RESEARCH": (3, "Recovery forming • RESEARCH"),
    "EARLY WATCH — RESEARCH": (3, "Early reversal • RESEARCH"),
    "WATCH": (3, "Reversal watch • NOT CONFIRMED"),
    "REVERSAL SIGNAL": (4, "Reversal signal"),
    "DEEP REVERSAL": (4, "Deep reversal confirmed"),
}
PUBLIC_KALSHI_URL = "https://kalshi.com/markets"


@dataclass(frozen=True)
class NtfySettings:
    topic: str
    server: str = "https://ntfy.sh"
    token: str | None = None

    @classmethod
    def from_environ(cls, environ=None) -> "NtfySettings | None":
        env = environ if environ is not None else os.environ
        topic = str(env.get("SPORTSEDGE_NTFY_TOPIC") or "").strip()
        if not topic:
            return None
        # Long topics help protect an otherwise public/unprotected topic from
        # guessing. We refuse weak/demo topic names to avoid publishing to them.
        if not re.fullmatch(r"[A-Za-z0-9_-]{24,64}", topic):
            raise ValueError("ntfy topic must contain 24–64 URL-safe characters")
        server = str(env.get("SPORTSEDGE_NTFY_SERVER") or "https://ntfy.sh").strip().rstrip("/")
        parsed = urlparse(server)
        if parsed.scheme != "https" or not parsed.hostname or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("ntfy server must be an HTTPS origin")
        return cls(topic=topic, server=server,
                   token=str(env.get("SPORTSEDGE_NTFY_TOKEN") or "").strip() or None)


@dataclass(frozen=True)
class PushEvent:
    alert_id: str
    selection: str
    ticker: str
    lane: str
    price: float
    fair: float | None
    score: str
    observed_at: str
    source_label: str = "SportsEdge"
    observed_trough: float | None = None
    rebound_pp: float | None = None
    quote_spread_pp: float | None = None

    def is_eligible(self, *, now: datetime, allow_extreme: bool = True) -> bool:
        if self.lane not in SUPPORTED or not self.alert_id or not self.ticker:
            return False
        try:
            detected = datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))
            age = (now.astimezone(timezone.utc) - detected.astimezone(timezone.utc)).total_seconds()
        except (TypeError, AttributeError, ValueError):
            return False
        if not (0 <= age <= 240):
            return False
        if not (0.01 <= self.price <= 0.25):
            return False
        if self.lane == "EXTREME DIP — TRACKING ONLY":
            return allow_extreme and self.price <= 0.04
        if self.lane == "PRICE RECOVERY — TRACKING ONLY":
            return allow_extreme and 0.04 < self.price <= 0.20 and self.fair is None
        if self.fair is None:
            return False
        # Research lanes and WATCHs require independent evidence even to alert.
        floor = .06 if self.lane in {"EARLY WATCH — RESEARCH", "WATCH"} else .04
        return self.fair - self.price >= floor

    def public_message(self) -> str:
        def cent(value: float) -> str:
            return f"{100*value:.1f}¢"
        fair = cent(self.fair) if self.fair is not None else "not available"
        prefix = ("TRACKING ONLY — not a bet." if self.lane in {"EXTREME DIP — TRACKING ONLY", "PRICE RECOVERY — TRACKING ONLY"}
                  else "Research estimate — not a guaranteed outcome.")
        movement = ""
        if self.lane in {"PRICE RECOVERY — TRACKING ONLY", "EXTREME DIP — TRACKING ONLY"}:
            if self.observed_trough is not None and self.rebound_pp is not None:
                movement += (
                    f"Observed quote trough {cent(self.observed_trough)} "
                    f"| rebound +{self.rebound_pp:.1f} percentage points\n"
                )
            if self.quote_spread_pp is not None:
                movement += f"Observed bid/ask spread {self.quote_spread_pp:.1f}pp\n"
        return (
            f"{self.selection} | quoted ASK {cent(self.price)} | model live fair {fair}\n"
            f"{movement}"
            f"{self.score}\n"
            f"{prefix}\n"
            f"Ticker: {self.ticker}\n"
            f"Observed UTC: {self.observed_at}"
        )


def post_ntfy(event: PushEvent, settings: NtfySettings, *,
              session=None, timeout: float = 8.0) -> str:
    """Publish to an already configured topic; return provider message ID.

    Do not include topic, token, full URLs, request bodies or raw exceptions
    in errors. Provider accepts a message, not a guarantee of device delivery.
    """
    priority, label = SUPPORTED[event.lane]
    payload = {
        "topic": settings.topic,
        "title": f"SportsEdge Tennis: {label}",
        "message": event.public_message(),
        "priority": priority,
        "tags": ["tennis"],
        "click": PUBLIC_KALSHI_URL,
    }
    headers = {"Authorization": f"Bearer {settings.token}"} if settings.token else {}
    client = session or requests
    try:
        response = client.post(
            f"{settings.server}/",
            json=payload, headers=headers, timeout=timeout,
        )
    except requests.RequestException:
        raise RuntimeError("push transport failure") from None
    if response.status_code not in (200, 201, 202):
        raise RuntimeError(f"push provider rejected request (HTTP {response.status_code})")
    try:
        doc = response.json()
        return str(doc.get("id") or "accepted") if isinstance(doc, dict) else "accepted"
    except ValueError:
        return "accepted"
