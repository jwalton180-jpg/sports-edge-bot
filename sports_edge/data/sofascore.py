from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

import requests


SOFASCORE_BASE_URL = "https://api.sofascore.com/api/v1"
SOFASCORE_SPORT_SLUGS = {
    "Tennis": "tennis",
    "MLB": "baseball",
    "NFL": "american-football",
    "NBA": "basketball",
    "WNBA": "basketball",
}


@dataclass(frozen=True)
class SofaScoreEvent:
    event_id: str
    sport: str
    home: str
    away: str
    status: str
    home_score: float | None
    away_score: float | None
    start_at: datetime | None


@dataclass(frozen=True)
class SofaScoreFeedResult:
    sport: str
    source_url: str
    available: bool
    blocked: bool
    status_code: int | None
    events: tuple[SofaScoreEvent, ...]
    error: str | None = None


def _number(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_score(team_score: dict) -> float | None:
    for key in ("current", "display", "normaltime"):
        value = _number((team_score or {}).get(key))
        if value is not None:
            return value
    return None


def _start_at(value) -> datetime | None:
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def parse_sofascore_events(payload: dict, *, sport: str) -> tuple[SofaScoreEvent, ...]:
    rows: list[SofaScoreEvent] = []
    for event in (payload or {}).get("events", []) or []:
        home = str(((event.get("homeTeam") or {}).get("name")) or "").strip()
        away = str(((event.get("awayTeam") or {}).get("name")) or "").strip()
        event_id = str(event.get("id") or "").strip()
        if not event_id or not home or not away:
            continue
        status = event.get("status") or {}
        status_text = str(
            status.get("description")
            or status.get("type")
            or status.get("status")
            or ""
        ).strip()
        rows.append(
            SofaScoreEvent(
                event_id=event_id,
                sport=sport,
                home=home,
                away=away,
                status=status_text,
                home_score=_event_score(event.get("homeScore") or {}),
                away_score=_event_score(event.get("awayScore") or {}),
                start_at=_start_at(event.get("startTimestamp")),
            )
        )
    return tuple(rows)


def fetch_sofascore_live_events(
    sport: str,
    *,
    timeout: float = 8.0,
) -> SofaScoreFeedResult:
    slug = SOFASCORE_SPORT_SLUGS.get(str(sport or ""))
    if not slug:
        return SofaScoreFeedResult(
            sport=sport,
            source_url="",
            available=False,
            blocked=False,
            status_code=None,
            events=(),
            error="unsupported SofaScore sport",
        )

    url = f"{SOFASCORE_BASE_URL}/sport/{slug}/events/live"
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "SportsEdgeReadOnly/1.0",
                "Accept": "application/json",
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return SofaScoreFeedResult(
            sport=sport,
            source_url=url,
            available=False,
            blocked=False,
            status_code=None,
            events=(),
            error=f"{type(exc).__name__}: {exc}",
        )

    if response.status_code in {401, 403, 429}:
        return SofaScoreFeedResult(
            sport=sport,
            source_url=url,
            available=False,
            blocked=True,
            status_code=response.status_code,
            events=(),
            error=f"upstream access denied ({response.status_code})",
        )
    if response.status_code != 200:
        return SofaScoreFeedResult(
            sport=sport,
            source_url=url,
            available=False,
            blocked=False,
            status_code=response.status_code,
            events=(),
            error=f"unexpected HTTP {response.status_code}",
        )

    try:
        payload = response.json()
    except ValueError:
        return SofaScoreFeedResult(
            sport=sport,
            source_url=url,
            available=False,
            blocked=False,
            status_code=200,
            events=(),
            error="invalid JSON response",
        )

    return SofaScoreFeedResult(
        sport=sport,
        source_url=url,
        available=True,
        blocked=False,
        status_code=200,
        events=parse_sofascore_events(payload, sport=sport),
        error=None,
    )


def fetch_sofascore_live_snapshot(
    sports: tuple[str, ...] = ("Tennis", "MLB", "NFL", "NBA", "WNBA"),
    *,
    timeout: float = 8.0,
    max_workers: int = 5,
) -> tuple[SofaScoreFeedResult, ...]:
    requested = tuple(dict.fromkeys(sports))
    if not requested:
        return ()
    results: dict[str, SofaScoreFeedResult] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(requested)))) as pool:
        future_map = {
            pool.submit(fetch_sofascore_live_events, sport, timeout=timeout): sport
            for sport in requested
        }
        for future in as_completed(future_map):
            sport = future_map[future]
            try:
                results[sport] = future.result()
            except Exception as exc:
                results[sport] = SofaScoreFeedResult(
                    sport=sport,
                    source_url="",
                    available=False,
                    blocked=False,
                    status_code=None,
                    events=(),
                    error=f"{type(exc).__name__}: {exc}",
                )
    return tuple(results[sport] for sport in requested if sport in results)
