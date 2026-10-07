from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sports_edge.data.tennis_live import TennisLiveScoreState
from sports_edge.models.event_identity import canonical_participant
from sports_edge.models.parlay_candidates import ParlayCandidateLeg


ITF_OFFICIAL_LIVE_URL = "https://www.itftennis.com/en/world-tennis-tour-live/"
ITF_MEN_TOUR_URL = "https://www.itftennis.com/en/tours/mens-world-tennis-tour/"
ITF_WOMEN_TOUR_URL = "https://www.itftennis.com/en/tours/womens-world-tennis-tour/"


@dataclass(frozen=True)
class TennisLiveCoverageRow:
    event_ticker: str
    title: str
    tour: str
    participants: tuple[str, ...]
    live_or_due: bool
    score_tracked: bool
    model_covered: bool
    model_name: str | None
    score_sources: tuple[str, ...]
    score_conflict: bool
    unsupported_reason: str | None
    official_itf_url: str | None
    official_itf_tour_url: str | None


@dataclass(frozen=True)
class TennisLiveCoverageSummary:
    rows: tuple[TennisLiveCoverageRow, ...]
    open_matches: int
    live_or_due_matches: int
    score_tracked_matches: int
    model_covered_matches: int
    unsupported_matches: int
    source_counts: tuple[tuple[str, int], ...]


def _text(value) -> str:
    return str(value or "").strip()


def _selection(value) -> str:
    value = _text(value)
    if value.lower() in {"", "yes", "no"}:
        return ""
    if value.lower().startswith("no —"):
        return ""
    return value


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((
        canonical_participant("Tennis", a),
        canonical_participant("Tennis", b),
    )))


def _pair_from_event_id(event_id: str) -> tuple[str, str] | None:
    parts = str(event_id or "").split(":", 2)
    if len(parts) != 3 or "|" not in parts[2]:
        return None
    a, b = parts[2].split("|", 1)
    if not a or not b:
        return None
    return tuple(sorted((a, b)))


def _parse_time(value) -> datetime | None:
    raw = _text(value)
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _tour(series: str) -> str:
    series = str(series or "").upper()
    if series.startswith("KXATPCHALLENGER"):
        return "ATP Challenger"
    if series.startswith("KXWTACHALLENGER"):
        return "WTA Challenger"
    if series.startswith("KXITFWMATCH"):
        return "ITF Women"
    if series.startswith("KXITFMATCH"):
        return "ITF Men"
    if series.startswith("KXWTAMATCH"):
        return "WTA"
    if series.startswith("KXATPMATCH"):
        return "ATP"
    return "Tennis"


def _official_urls(tour: str) -> tuple[str | None, str | None]:
    if tour == "ITF Men":
        return ITF_OFFICIAL_LIVE_URL, ITF_MEN_TOUR_URL
    if tour == "ITF Women":
        return ITF_OFFICIAL_LIVE_URL, ITF_WOMEN_TOUR_URL
    return None, None


def _inventory(markets: list[dict] | tuple[dict, ...]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for market in markets or ():
        event = _text(market.get("event_ticker") or market.get("ticker"))
        if not event:
            continue
        grouped.setdefault(event, []).append(market)

    rows: list[dict] = []
    for event_ticker, event_markets in grouped.items():
        names: list[str] = []
        seen: set[str] = set()
        for market in event_markets:
            for key in ("yes_sub_title", "no_sub_title", "yes_title", "no_title"):
                name = _selection(market.get(key))
                participant = canonical_participant("Tennis", name)
                if name and participant and participant not in seen:
                    seen.add(participant)
                    names.append(name)
        first = event_markets[0]
        series = _text(first.get("series_ticker") or event_ticker.split("-", 1)[0]).upper()
        title = _text(first.get("event_title") or first.get("title") or event_ticker)
        start = None
        close = None
        for market in event_markets:
            start = start or _parse_time(market.get("occurrence_datetime"))
            close = close or _parse_time(market.get("close_time") or market.get("expected_expiration_time"))
        rows.append({
            "event_ticker": event_ticker,
            "title": title,
            "tour": _tour(series),
            "participants": tuple(names[:2]),
            "pair": _pair_key(names[0], names[1]) if len(names) == 2 else None,
            "start": start,
            "close": close,
        })
    return rows


def build_tennis_live_coverage(
    markets: list[dict] | tuple[dict, ...],
    states: list[TennisLiveScoreState] | tuple[TennisLiveScoreState, ...],
    candidates: list[ParlayCandidateLeg] | tuple[ParlayCandidateLeg, ...],
    *,
    now: datetime | None = None,
    start_grace_minutes: int = 10,
) -> TennisLiveCoverageSummary:
    """Audit match-level coverage for the Kalshi Tennis reversal universe.

    ``live_or_due`` means a score feed currently confirms the match as live OR
    Kalshi's scheduled occurrence time has passed while the market remains open.
    The second case is deliberately labelled as due rather than definitely live
    because delayed matches exist; it is still the right bucket for coverage
    holes that need attention.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    grace = timedelta(minutes=max(0, int(start_grace_minutes)))

    state_pairs: dict[tuple[str, str], list[TennisLiveScoreState]] = {}
    for state in states or ():
        pair = _pair_key(state.player, state.opponent)
        state_pairs.setdefault(pair, []).append(state)

    candidate_pairs: dict[tuple[str, str], list[ParlayCandidateLeg]] = {}
    for candidate in candidates or ():
        pair = _pair_from_event_id(candidate.event_id)
        if pair is None:
            continue
        candidate_pairs.setdefault(pair, []).append(candidate)

    rows: list[TennisLiveCoverageRow] = []
    source_counter: Counter[str] = Counter()
    for item in _inventory(markets):
        pair = item["pair"]
        matched_states = state_pairs.get(pair, []) if pair is not None else []
        matched_candidates = candidate_pairs.get(pair, []) if pair is not None else []
        score_tracked = bool(matched_states)
        model_covered = bool(matched_candidates)
        score_conflict = any(bool(state.score_conflict) for state in matched_states)
        sources = tuple(sorted({
            source
            for state in matched_states
            for source in tuple(state.score_sources or ())
            if source
        }))
        for source in sources:
            source_counter[source] += 1

        start = item["start"]
        close = item["close"]
        scheduled_due = bool(
            start is not None
            and start <= now + grace
            and (close is None or close >= now - grace)
        )
        live_or_due = score_tracked or scheduled_due

        reason = None
        if live_or_due:
            if pair is None:
                reason = "participant identity unresolved"
            elif not score_tracked:
                reason = "scheduled start passed; no live score state"
            elif score_conflict:
                reason = "live score sources disagree"
            elif not model_covered:
                reason = "live score tracked; no usable independent model prior"

        model_names = sorted({
            str(row.model_name or "").strip()
            for row in matched_candidates
            if str(row.model_name or "").strip()
        })
        model_name = " + ".join(model_names) if model_names else None
        official_live, official_tour = _official_urls(item["tour"])
        rows.append(TennisLiveCoverageRow(
            event_ticker=item["event_ticker"],
            title=item["title"],
            tour=item["tour"],
            participants=item["participants"],
            live_or_due=live_or_due,
            score_tracked=score_tracked,
            model_covered=model_covered,
            model_name=model_name,
            score_sources=sources,
            score_conflict=score_conflict,
            unsupported_reason=reason,
            official_itf_url=official_live,
            official_itf_tour_url=official_tour,
        ))

    rows.sort(key=lambda row: (not row.live_or_due, row.tour, row.title, row.event_ticker))
    live_rows = [row for row in rows if row.live_or_due]
    return TennisLiveCoverageSummary(
        rows=tuple(rows),
        open_matches=len(rows),
        live_or_due_matches=len(live_rows),
        score_tracked_matches=sum(1 for row in live_rows if row.score_tracked),
        model_covered_matches=sum(1 for row in live_rows if row.model_covered),
        unsupported_matches=sum(1 for row in live_rows if row.unsupported_reason is not None),
        source_counts=tuple(sorted(source_counter.items())),
    )
