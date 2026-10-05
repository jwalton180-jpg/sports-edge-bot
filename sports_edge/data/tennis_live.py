from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import time

import requests

from sports_edge.data.kalshi import KalshiPublicClient
from sports_edge.models.event_identity import canonical_event_id, canonical_participant


ESPN_TENNIS_SCOREBOARDS = (
    ("ATP", "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"),
    ("WTA", "https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard"),
)


@dataclass(frozen=True)
class TennisLiveScoreState:
    event_id: str
    selection_key: str
    player: str
    opponent: str
    tour: str
    period: int
    player_sets: int
    opponent_sets: int
    player_games: int | None
    opponent_games: int | None
    lost_first_set: bool
    won_latest_completed_set: bool
    turnaround: bool
    deciding_set: bool
    current_set_lead: int
    score_label: str
    fetched_at: datetime


def _iso_date(value) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date().isoformat()
    except (TypeError, ValueError):
        return None


def _line_value(lines: list[dict], index: int) -> int | None:
    if index < 0 or index >= len(lines):
        return None
    try:
        return int(float(lines[index].get("value")))
    except (TypeError, ValueError, AttributeError):
        return None


def parse_espn_live_tennis_states(
    payload: dict,
    *,
    tour: str,
    fetched_at: datetime | None = None,
) -> tuple[TennisLiveScoreState, ...]:
    """Parse live singles competitions from ESPN's nested tournament payload.

    Tournament-level status is intentionally ignored. Each nested competition
    has its own live/final state and per-set linescores.
    """
    fetched_at = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    out: list[TennisLiveScoreState] = []

    for event in (payload or {}).get("events", []) or []:
        for grouping in event.get("groupings", []) or []:
            for comp in grouping.get("competitions", []) or []:
                status = comp.get("status") or {}
                status_type = status.get("type") or {}
                state = str(status_type.get("state") or "").strip().lower()
                if state != "in" or status_type.get("completed") is True:
                    continue

                competitors = list(comp.get("competitors", []) or [])
                if len(competitors) != 2:
                    continue
                event_date = _iso_date(comp.get("date") or comp.get("startDate"))
                if not event_date:
                    continue

                names: list[str] = []
                lines: list[list[dict]] = []
                for competitor in competitors:
                    athlete = competitor.get("athlete") or {}
                    name = str(
                        athlete.get("displayName")
                        or athlete.get("fullName")
                        or athlete.get("shortName")
                        or ""
                    ).strip()
                    names.append(name)
                    lines.append(list(competitor.get("linescores", []) or []))
                if not all(names):
                    continue

                try:
                    period = max(1, int(status.get("period") or 1))
                except (TypeError, ValueError):
                    period = 1

                event_id = canonical_event_id("Tennis", names[0], names[1], event_date)
                completed_sets = max(0, period - 1)

                set_winners: list[int | None] = []
                for set_i in range(completed_sets):
                    a = _line_value(lines[0], set_i)
                    b = _line_value(lines[1], set_i)
                    if a is None or b is None or a == b:
                        set_winners.append(None)
                    else:
                        set_winners.append(0 if a > b else 1)

                current_index = period - 1
                current = (
                    _line_value(lines[0], current_index),
                    _line_value(lines[1], current_index),
                )

                score_parts: list[str] = []
                max_sets = max(len(lines[0]), len(lines[1]), period)
                for set_i in range(max_sets):
                    a = _line_value(lines[0], set_i)
                    b = _line_value(lines[1], set_i)
                    if a is None and b is None:
                        continue
                    score_parts.append(
                        f"{a if a is not None else '-'}-{b if b is not None else '-'}"
                    )
                pair_score = " · ".join(score_parts) if score_parts else f"Set {period}"

                for player_i in (0, 1):
                    opp_i = 1 - player_i
                    player_sets = sum(1 for winner in set_winners if winner == player_i)
                    opponent_sets = sum(1 for winner in set_winners if winner == opp_i)
                    lost_first = bool(set_winners and set_winners[0] == opp_i)
                    latest_won = bool(set_winners and set_winners[-1] == player_i)
                    turnaround = lost_first and latest_won
                    deciding = (
                        period >= 3
                        and player_sets == opponent_sets
                        and player_sets > 0
                    )
                    player_games = current[player_i]
                    opponent_games = current[opp_i]
                    current_lead = (
                        (player_games - opponent_games)
                        if player_games is not None and opponent_games is not None
                        else 0
                    )
                    out.append(
                        TennisLiveScoreState(
                            event_id=event_id,
                            selection_key=canonical_participant("Tennis", names[player_i]),
                            player=names[player_i],
                            opponent=names[opp_i],
                            tour=str(tour or "").upper(),
                            period=period,
                            player_sets=player_sets,
                            opponent_sets=opponent_sets,
                            player_games=player_games,
                            opponent_games=opponent_games,
                            lost_first_set=lost_first,
                            won_latest_completed_set=latest_won,
                            turnaround=turnaround,
                            deciding_set=deciding,
                            current_set_lead=current_lead,
                            score_label=f"{names[0]} vs {names[1]} · {pair_score}",
                            fetched_at=fetched_at,
                        )
                    )

    dedup = {(row.event_id, row.selection_key): row for row in out}
    return tuple(dedup.values())


def fetch_espn_live_tennis_states(
    *,
    timeout: float = 12.0,
) -> tuple[TennisLiveScoreState, ...]:
    """Fetch current ATP/WTA singles score state from ESPN's public scoreboards.

    This is a live-state corroboration layer only; it never replaces the
    independent Tennis probability model. Challenger/ITF matches remain
    eligible for price-path WATCH status when no ESPN score state is available.
    """
    fetched_at = datetime.now(timezone.utc)
    out: list[TennisLiveScoreState] = []
    for tour, url in ESPN_TENNIS_SCOREBOARDS:
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "application/json"},
                timeout=timeout,
            )
            response.raise_for_status()
            out.extend(
                parse_espn_live_tennis_states(
                    response.json(),
                    tour=tour,
                    fetched_at=fetched_at,
                )
            )
        except (requests.RequestException, ValueError, TypeError):
            continue

    dedup = {(row.event_id, row.selection_key): row for row in out}
    return tuple(dedup.values())


TENNIS_MATCH_SERIES = (
    "KXATPMATCH",
    "KXATPCHALLENGERMATCH",
    "KXWTAMATCH",
    "KXWTACHALLENGERMATCH",
    "KXITFMATCH",
    "KXITFWMATCH",
)


def _payload(result):
    return result.data if hasattr(result, "data") else result


def fetch_open_tennis_match_markets(
    *,
    max_workers: int = 6,
) -> tuple[dict, ...]:
    """Fetch only current open Tennis match-winner markets.

    This is intentionally much smaller than the full Tennis catalog so the
    live-reversal scanner can refresh frequently without reloading set/game
    derivative markets.
    """
    def one(series: str) -> list[dict]:
        client = KalshiPublicClient()
        payload = _payload(client.markets(
            status="open",
            series_ticker=series,
            limit=1000,
        ))
        return list((payload or {}).get("markets", []) or [])

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), len(TENNIS_MATCH_SERIES)))) as pool:
        futures = {pool.submit(one, series): series for series in TENNIS_MATCH_SERIES}
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception:
                # Live radar is fail-soft across series. The UI reports coverage
                # from the returned rows rather than inventing missing markets.
                continue

    dedup = {str(row.get("ticker") or ""): row for row in rows if row.get("ticker")}
    return tuple(dedup.values())


def fetch_tennis_candle_history(
    tickers: list[str] | tuple[str, ...],
    *,
    lookback_minutes: int = 60,
    period_interval: int = 1,
    now: datetime | None = None,
    max_workers: int = 5,
) -> dict[str, tuple[dict, ...]]:
    """Fetch executable price history in public Kalshi batches.

    Kalshi accepts up to 100 tickers per batch request. Chunks are kept under
    that limit and independently fail closed.
    """
    unique = list(dict.fromkeys(str(x).strip() for x in tickers if str(x).strip()))
    if not unique:
        return {}

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    end_ts = int(now.timestamp())
    start_ts = end_ts - max(5, int(lookback_minutes)) * 60
    chunks = [unique[i:i + 100] for i in range(0, len(unique), 100)]

    def one(chunk: list[str]) -> dict[str, tuple[dict, ...]]:
        client = KalshiPublicClient()
        result = client.batch_market_candlesticks(
            chunk,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=max(1, int(period_interval)),
            include_latest_before_start=True,
        )
        payload = _payload(result)
        out: dict[str, tuple[dict, ...]] = {}
        for item in (payload or {}).get("markets", []) or []:
            ticker = str(item.get("market_ticker") or item.get("ticker") or "").strip()
            if not ticker:
                continue
            candles = tuple(item.get("candlesticks", []) or [])
            out[ticker] = candles
        return out

    out: dict[str, tuple[dict, ...]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), len(chunks)))) as pool:
        futures = [pool.submit(one, chunk) for chunk in chunks]
        for future in as_completed(futures):
            try:
                out.update(future.result())
            except Exception:
                continue
    return out
