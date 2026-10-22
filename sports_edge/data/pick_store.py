"""Append-only optional Supabase-backed research-pick storage.

The Supabase service-role key is used only in server-side Streamlit Python and
must be stored in Streamlit secrets, never committed or exposed to the browser.
Without configured credentials the app must use an explicitly session-only
fallback rather than pretending records are durable.
"""
from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import Any, Mapping
from urllib.parse import quote

import requests


SETTLEMENTS = {"win", "loss", "push", "void", "cancelled"}


def parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_prediction(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(payload)
    required = ("sport", "event", "market", "selection", "observed_at", "event_start", "source", "record_type")
    if any(not str(row.get(key) or "").strip() for key in required):
        raise ValueError("prediction is missing required attribution fields")
    observed = parse_timestamp(row["observed_at"])
    start = parse_timestamp(row["event_start"])
    if observed >= start:
        raise ValueError("prediction must be recorded before event start")
    probability = row.get("probability")
    if probability is not None:
        p = float(probability)
        if not 0 <= p <= 1:
            raise ValueError("probability must be in [0, 1]")
        row["probability"] = p
    return row


class SupabasePickStore:
    """Minimal PostgREST adapter for an append-only sportedge_pick_records table."""

    def __init__(self, url: str, service_role_key: str, *, session=None, timeout: float = 8.0):
        self.url = str(url).strip().rstrip("/")
        self.key = str(service_role_key).strip()
        if not self.url.startswith("https://") or not self.key:
            raise ValueError("Supabase HTTPS URL and service-role key are required")
        self.session = session or requests.Session()
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.url and self.key)

    def _request(self, method: str, *, params=None, json_body=None):
        endpoint = self.url + "/rest/v1/sportedge_pick_records"
        headers = {
            "apikey": self.key,
            "Authorization": "Bearer " + self.key,
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        response = self.session.request(
            method, endpoint, headers=headers, params=params, json=json_body, timeout=self.timeout
        )
        response.raise_for_status()
        if not response.content:
            return []
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Supabase returned an unexpected records payload")
        return payload

    def append(self, record_type: str, payload: Mapping[str, Any], *, record_id: str | None = None):
        if record_type not in {"prediction", "settlement", "note"}:
            raise ValueError("unsupported append-only record type")
        row = {
            "id": record_id or str(uuid.uuid4()),
            "record_type": record_type,
            "payload": dict(payload),
        }
        result = self._request("POST", json_body=row)
        return result[0] if result else row

    def append_prediction(self, payload: Mapping[str, Any]):
        row = validate_prediction(payload)
        row["prediction_id"] = str(row.get("prediction_id") or uuid.uuid4())
        row["saved_at"] = datetime.now(timezone.utc).isoformat()
        return self.append("prediction", row, record_id=row["prediction_id"])

    def append_settlement(self, *, prediction_id: str, result: str, settled_at: str,
                          source_url: str, verification: str = "manual"):
        if result not in SETTLEMENTS:
            raise ValueError("unsupported settlement result")
        if not str(source_url).startswith("https://"):
            raise ValueError("settlement requires an HTTPS source URL")
        settled = parse_timestamp(settled_at)
        existing = self.list_records()
        predictions = [
            row for row in existing
            if row.get("record_type") == "prediction"
            and row.get("id") == prediction_id
        ]
        if not predictions:
            raise ValueError("unknown prediction id")
        if any(
            row.get("record_type") == "settlement"
            and (row.get("payload") or {}).get("prediction_id") == prediction_id
            for row in existing
        ):
            raise ValueError("prediction already has a settlement record")
        payload = predictions[0].get("payload") or {}
        if settled < parse_timestamp(payload["event_start"]):
            raise ValueError("settlement cannot predate event start")
        return self.append("settlement", {
            "prediction_id": prediction_id,
            "result": result,
            "settled_at": settled.isoformat(),
            "source_url": source_url,
            "verification": verification,
        })

    def list_records(self, *, limit: int = 5000):
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError("limit must be an integer from 1 to 10000")
        return self._request("GET", params={
            "select": "id,record_type,payload,created_at",
            "order": "created_at.asc",
            "limit": limit,
        })
