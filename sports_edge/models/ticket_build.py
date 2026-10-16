from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, datetime
import inspect
from zoneinfo import ZoneInfo
from typing import Any

from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.selected_games_compat import selected_games_compat_candidates


def build_ticket_for_scope(
    candidates: Iterable[ParlayCandidateLeg],
    *,
    builder: Callable[..., Any],
    assessor: Callable[[ParlayCandidateLeg, str], Any],
    mode: str,
    target_legs: int,
    sport_filter: str,
    game_scope: str,
    preferred_event_ids: Iterable[str] = (),
    ticket_date: str | date | None = None,
    ticket_timezone: str = "Pacific/Honolulu",
) -> Any:
    """Build tickets with stale, explicitly dated events excluded."""
    rows = list(candidates)
    if ticket_date is not None:
        target_date = date.fromisoformat(ticket_date) if isinstance(ticket_date, str) else ticket_date
        zone = ZoneInfo(ticket_timezone)
        valid_rows = []
        for row in rows:
            raw_start = getattr(row, "verified_start", None)
            if raw_start:
                try:
                    parsed = datetime.fromisoformat(str(raw_start).replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        continue
                    if parsed.astimezone(zone).date() != target_date:
                        continue
                except (TypeError, ValueError, OverflowError):
                    continue
            valid_rows.append(row)
        rows = valid_rows
    if game_scope == "Single game":
        return builder(
            rows,
            mode=mode,
            target_legs=min(target_legs, 4),
            max_per_event=4,
            diversify_sports=False,
            prioritize_payout_multiplier=True,
        )

    kwargs: dict[str, Any] = {
        "mode": mode,
        "target_legs": target_legs,
        "max_per_event": 3 if sport_filter == "MLB" else 1,
        "diversify_sports": sport_filter == "All",
    }
    preferred = tuple(
        dict.fromkeys(
            str(event_id).strip()
            for event_id in preferred_event_ids
            if str(event_id).strip()
        )
    )
    try:
        supports_preferred = "preferred_event_ids" in inspect.signature(builder).parameters
    except (TypeError, ValueError):
        supports_preferred = False

    if game_scope == "Selected games" and preferred:
        if supports_preferred:
            kwargs["preferred_event_ids"] = preferred
        else:
            rows = selected_games_compat_candidates(
                rows,
                preferred_event_ids=preferred,
                mode=mode,
                target_legs=target_legs,
                max_per_event=kwargs["max_per_event"],
                assess=assessor,
            )

    return builder(rows, **kwargs)


def ticket_state_matches(
    state: dict[str, Any] | None,
    *,
    mode: str,
    preset: str,
    sport: str,
    target: int,
    ticket_date: str,
    game_scope: str,
    selected_games: Iterable[str],
) -> bool:
    """Prevent a ticket built for old controls from rendering as current."""
    if not state:
        return False
    return (
        state.get("mode") == mode
        and state.get("preset") == preset
        and state.get("sport") == sport
        and state.get("target") == target
        and state.get("ticket_date") == ticket_date
        and state.get("game_scope") == game_scope
        and tuple(state.get("selected_games") or ()) == tuple(selected_games)
    )
