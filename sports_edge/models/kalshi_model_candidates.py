from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import re
from typing import Iterable
from zoneinfo import ZoneInfo

from sports_edge.data.public_team_data import team_game_model
from sports_edge.models.event_identity import (
    canonical_event_id,
    canonical_event_id_from_title,
)
from sports_edge.models.game_scope import normalize
from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.live_board import market_side_probability
from sports_edge.models.basketball_prop_models import project_wnba_player_prop
from sports_edge.models.mlb_hits_model import project_mlb_hits
from sports_edge.models.mlb_lines_model import (
    project_mlb_game_total,
    project_mlb_spread,
    project_mlb_team_total,
)
from sports_edge.models.mlb_hr_model import project_mlb_home_runs
from sports_edge.models.mlb_run_production_model import project_mlb_hrr, project_mlb_rbis
from sports_edge.models.mlb_total_bases_model import project_mlb_total_bases
from sports_edge.models.mlb_strikeouts_model import project_mlb_pitcher_strikeouts
from sports_edge.models.nfl_lines_model import (
    project_nfl_game_total,
    project_nfl_spread,
    project_nfl_team_total,
)
from sports_edge.models.nfl_prop_models import (
    project_nfl_pass_attempts,
    project_nfl_pass_completions,
    project_nfl_pass_interceptions,
    project_nfl_passing_tds,
    project_nfl_passing_yards,
    project_nfl_receiving_yards,
    project_nfl_receptions,
    project_nfl_rush_attempts,
    project_nfl_rush_receiving_yards,
    project_nfl_rushing_yards,
    project_nfl_touchdowns,
)
from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.tennis_research import TennisResearchModel, tennis_level_from_series
from sports_edge.models.tennis_games_model import TennisGamesModel
from sports_edge.models.wnba_lines_model import (
    project_wnba_game_total,
    project_wnba_spread,
    project_wnba_team_total,
)


_TICKER_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_TICKER_DATE_RE = re.compile(r"(?:^|-)(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})", re.I)


def _parse_date(market: dict) -> date:
    """Prefer Kalshi's event/ticker date over settlement timestamps.

    Sports contracts often remain open/settle 1–3 days after the game. Using
    close_time as the event date silently breaks public schedule joins.
    """
    for key in ("event_ticker", "ticker"):
        raw = str(market.get(key) or "").upper()
        match = _TICKER_DATE_RE.search(raw)
        if not match:
            continue
        yy, mon, dd = match.groups()
        try:
            return date(2000 + int(yy), _TICKER_MONTHS[mon.upper()], int(dd))
        except ValueError:
            continue

    for key in ("close_time", "expected_expiration_time", "open_time"):
        raw = market.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
        except ValueError:
            continue
    return datetime.now(timezone.utc).date()


def _event_key(market: dict) -> str:
    return str(market.get("event_ticker") or market.get("ticker") or "").strip()


def _market_series_ticker(market: dict) -> str:
    """Resolve series even when Kalshi omits series_ticker on live rows."""
    explicit = str(market.get("series_ticker") or "").strip().upper()
    if explicit:
        return explicit
    for key in ("event_ticker", "ticker"):
        raw = str(market.get(key) or "").strip().upper()
        if raw:
            return raw.split("-", 1)[0]
    return ""


def _market_local_date(market: dict, timezone_name: str) -> date:
    """Resolve the actual event calendar date in the ticket timezone.

    Prefer Kalshi's explicit occurrence timestamp. Older/non-tennis contracts
    that do not expose it fall back to the event/ticker date, never settlement
    time when a ticker date is available.
    """
    raw = market.get("occurrence_datetime")
    if raw:
        parsed = _parse_timestamp(raw)
        if parsed is not None:
            try:
                return parsed.astimezone(ZoneInfo(timezone_name)).date()
            except (KeyError, ValueError):
                pass
    return _parse_date(market)


def _rows_for_local_date(
    rows: list[KalshiSportMarket],
    *,
    target_date: date,
    timezone_name: str,
) -> list[KalshiSportMarket]:
    return [
        row for row in rows
        if _market_local_date(row.market, timezone_name) == target_date
    ]


def _canonical_game_id(sport: str, game_title: str, event_date: date) -> str:
    """Backward-compatible wrapper around shared physical-event identity."""
    return canonical_event_id_from_title(sport, game_title, event_date)


def _event_title(market: dict) -> str:
    return str(market.get("event_title") or market.get("title") or _event_key(market)).strip()


def _explicit_tennis_surface(market: dict) -> str | None:
    text = normalize(" ".join(
        str(market.get(k) or "")
        for k in ("series_title", "series_tags", "event_title", "subtitle")
    ))
    for surface in ("clay", "grass", "hard", "carpet"):
        if re.search(rf"\b{surface}\b", text):
            return surface
    return None


def _clean_selection(value: str | None) -> str:
    raw = str(value or "").strip()
    if normalize(raw) in {"", "yes", "no"}:
        return ""
    return raw


def _event_choices(rows: Iterable[KalshiSportMarket]) -> list[tuple[str, str, float, dict]]:
    """Return (selection, side, price, market) choices for one event."""
    choices: list[tuple[str, str, float, dict]] = []
    seen: set[str] = set()

    for row in rows:
        market = row.market
        yes = _clean_selection(
            market.get("yes_sub_title")
            or market.get("yes_title")
            or market.get("yes_label")
        )
        if yes:
            p = market_side_probability(market, "YES")
            key = normalize(yes)
            if p is not None and key and key not in seen:
                choices.append((yes, "YES", p, market))
                seen.add(key)

        no = _clean_selection(
            market.get("no_sub_title")
            or market.get("no_title")
            or market.get("no_label")
        )
        if no and not no.lower().startswith("no —"):
            p = market_side_probability(market, "NO")
            key = normalize(no)
            if p is not None and key and key not in seen:
                choices.append((no, "NO", p, market))
                seen.add(key)

    return choices


@lru_cache(maxsize=4)
def _tennis_model(gender: str, year: int) -> TennisResearchModel:
    return TennisResearchModel(gender, current_year=year)


@lru_cache(maxsize=4)
def _tennis_games_model(gender: str, year: int) -> TennisGamesModel:
    return TennisGamesModel(gender, current_year=year)


def _tennis_event_signature(market: dict) -> str:
    raw = str(market.get("event_ticker") or "").strip().upper()
    suffix = raw.split("-", 1)[1] if "-" in raw else raw
    series = _market_series_ticker(market) or raw.split("-", 1)[0]
    if series.startswith("KXWTA"):
        namespace = "WTA"
    elif series.startswith("KXITF"):
        namespace = "ITF"
    elif series.startswith("KXATP"):
        namespace = "ATP"
    else:
        namespace = "TENNIS"
    return f"{namespace}:{suffix}" if suffix else ""


def _tennis_matchups_by_signature(
    rows: list[KalshiSportMarket],
) -> dict[str, tuple[str, str]]:
    """Resolve exactly two participants for each physical Tennis match.

    Match-winner contracts are preferred because they expose both player names.
    Games-spread titles are only a fallback. The shared event suffix is a join
    key, never the final canonical event identity.
    """
    names: dict[str, list[str]] = {}
    for row in rows:
        if row.family != "Match Winner":
            continue
        signature = _tennis_event_signature(row.market)
        if not signature:
            continue
        bucket = names.setdefault(signature, [])
        for key in ("yes_sub_title", "no_sub_title", "yes_title", "no_title"):
            value = _clean_selection(row.market.get(key))
            if value and normalize(value) not in {normalize(x) for x in bucket}:
                bucket.append(value)

    for row in rows:
        if row.family != "Games Spread":
            continue
        signature = _tennis_event_signature(row.market)
        if not signature or len(names.get(signature, [])) == 2:
            continue
        title = str(row.market.get("title") or "").strip()
        match = re.match(
            r"Will\s+(.+?)\s+win\s+at\s+least\s+[0-9.]+\s+more\s+games\s+than\s+(.+?)\??$",
            title,
            re.I,
        )
        if not match:
            continue
        bucket = names.setdefault(signature, [])
        for value in (match.group(1).strip(), match.group(2).strip()):
            if value and normalize(value) not in {normalize(x) for x in bucket}:
                bucket.append(value)

    return {
        signature: (values[0], values[1])
        for signature, values in names.items()
        if len(values) == 2 and normalize(values[0]) != normalize(values[1])
    }


def _parse_timestamp(value: object) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _tennis_scheduled_start(market: dict) -> datetime | None:
    # Require the explicit scheduled occurrence. Expiration/close timestamps can
    # lag the actual match and are not proof that a contract is still pregame.
    return _parse_timestamp(market.get("occurrence_datetime"))


def _tennis_best_of(market: dict, gender: str) -> int | None:
    if gender == "women":
        return 3

    text = normalize(" ".join(
        str(market.get(key) or "")
        for key in (
            "rules_primary",
            "rules_secondary",
            "event_title",
            "subtitle",
            "series_title",
            "series_tags",
        )
    ))
    if not text:
        return None

    grand_slam = any(
        term in text
        for term in (
            "australian open",
            "french open",
            "roland garros",
            "wimbledon",
            "us open",
            "u s open",
        )
    )
    if grand_slam and "qualif" not in text:
        return 5

    # Men's ATP/Challenger/ITF non-Slam singles are best-of-three. Require
    # explicit tennis competition/rule context so missing metadata fails closed.
    if any(term in text for term in (" atp ", " challenger", " itf ", " tennis ")):
        return 3
    return None


def _tennis_games_total_candidates(
    rows: list[KalshiSportMarket],
    *,
    now: datetime | None = None,
) -> list[ParlayCandidateLeg]:
    matchups = _tennis_matchups_by_signature(rows)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)

    out: list[ParlayCandidateLeg] = []
    for row in rows:
        if row.family != "Games Total":
            continue
        market = row.market
        signature = _tennis_event_signature(market)
        participants = matchups.get(signature)
        if participants is None:
            continue

        start = _tennis_scheduled_start(market)
        # A scheduled start in the past is not proof a match is live, but it is
        # enough uncertainty to refuse a pregame model. Future start is required.
        if start is None or start <= current + timedelta(minutes=10):
            continue

        try:
            line = float(market.get("floor_strike"))
        except (TypeError, ValueError):
            continue
        if line <= 0:
            continue

        player_a, player_b = participants
        series = _market_series_ticker(market)
        gender, level = tennis_level_from_series(series)
        best_of = _tennis_best_of(market, gender)

        # Production scope is deliberately narrower than the visible Kalshi
        # family. The matchup-aware WTA challenger lost to a strength-blind
        # chronological prior on the 2025 holdout, and BO5 totals have not yet
        # passed a dedicated alternate-line validation. Fail closed on both.
        if gender != "men" or "itf" in level.lower() or best_of != 3:
            continue

        model = _tennis_games_model(gender, start.date().year)
        projection = model.project_games_total(
            player_a,
            player_b,
            line=line,
            event_date=start.date(),
            level=level,
            surface=_explicit_tennis_surface(market),
            best_of=best_of,
            game_title=f"{player_a} vs {player_b}",
        )
        if projection is None or not projection.evidence.usable:
            continue

        base = projection.evidence
        no_evidence = replace(
            base,
            fair_probability=1.0 - base.fair_probability,
            factors=tuple([
                f"Complement of Over {line:g} games model probability",
                *base.factors,
            ]),
        )
        yes_price = market_side_probability(market, "YES")
        no_price = market_side_probability(market, "NO")

        # Preserve the same canonical identity scheme as existing Tennis
        # match-winner candidates. The suffix only resolved participants.
        canonical_date = _parse_date(market)
        event_id = canonical_event_id("Tennis", player_a, player_b, canonical_date)
        event_title = f"{player_a} vs {player_b}"

        for side, price, evidence, selection in (
            ("YES", yes_price, base, f"Over {line:g} Games"),
            ("NO", no_price, no_evidence, f"Under {line:g} Games"),
        ):
            if price is None:
                continue
            out.append(
                ParlayCandidateLeg(
                    sport="Tennis",
                    event_id=event_id,
                    event_title=event_title,
                    market_key=projection.market_key,
                    market_label=projection.market_label,
                    selection=selection,
                    consensus_probability=evidence.fair_probability,
                    book_count=0,
                    source_age_s=0.0,
                    median_odds=None,
                    kalshi_ticker=str(market.get("ticker") or ""),
                    kalshi_side=side,
                    kalshi_price=price,
                    kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                    kalshi_status="MODEL",
                    evidence_class="MODEL",
                    model_probability=evidence.fair_probability,
                    model_confidence=evidence.confidence,
                    model_name=evidence.model_name,
                    model_sample_size=evidence.sample_size,
                    model_reasons=evidence.factors,
                    model_warnings=evidence.warnings,
                )
            )
    return out


def _tennis_event_candidates(rows: list[KalshiSportMarket]) -> list[ParlayCandidateLeg]:
    choices = _event_choices(rows)
    if len(choices) != 2:
        return []

    first_market = choices[0][3]
    series = _market_series_ticker(first_market)
    gender, level = tennis_level_from_series(series)
    event_date = _parse_date(first_market)

    a_name, a_side, a_price, a_market = choices[0]
    b_name, b_side, b_price, b_market = choices[1]
    model = _tennis_model(gender, event_date.year)
    surface = _explicit_tennis_surface(first_market)
    evidence_a = model.probability(
        a_name,
        b_name,
        level=level,
        as_of=event_date,
        surface=surface,
    )
    if evidence_a is None or not evidence_a.usable:
        return []

    evidence_b = replace(
        evidence_a,
        fair_probability=1.0 - evidence_a.fair_probability,
        factors=tuple(
            [
                f"Opponent-side complement of {a_name} model probability",
                *evidence_a.factors,
            ]
        ),
    )

    out: list[ParlayCandidateLeg] = []
    for name, side, price, market, evidence in (
        (a_name, a_side, a_price, a_market, evidence_a),
        (b_name, b_side, b_price, b_market, evidence_b),
    ):
        out.append(
            ParlayCandidateLeg(
                sport="Tennis",
                event_id=canonical_event_id("Tennis", a_name, b_name, event_date),
                event_title=_event_title(market),
                market_key="model_h2h",
                market_label="Match Winner",
                selection=name,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _team_event_candidates(sport: str, rows: list[KalshiSportMarket]) -> list[ParlayCandidateLeg]:
    choices = _event_choices(rows)
    if len(choices) != 2:
        return []

    a_name, a_side, a_price, a_market = choices[0]
    b_name, b_side, b_price, b_market = choices[1]
    event_date = _parse_date(a_market)
    evidence_a = team_game_model(sport, a_name, b_name, event_date=event_date)
    if evidence_a is None or not evidence_a.usable:
        return []
    evidence_b = replace(
        evidence_a,
        fair_probability=1.0 - evidence_a.fair_probability,
        factors=tuple(
            [
                f"Opponent-side complement of {a_name} team-strength probability",
                *evidence_a.factors,
            ]
        ),
    )

    out: list[ParlayCandidateLeg] = []
    for name, side, price, market, evidence in (
        (a_name, a_side, a_price, a_market, evidence_a),
        (b_name, b_side, b_price, b_market, evidence_b),
    ):
        out.append(
            ParlayCandidateLeg(
                sport=sport,
                event_id=canonical_event_id(sport, a_name, b_name, event_date),
                event_title=_event_title(market),
                market_key="model_h2h",
                market_label="Moneyline",
                selection=name,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _market_volume(market: dict) -> float:
    try:
        return float(market.get("volume_fp", market.get("volume", 0)) or 0)
    except (TypeError, ValueError):
        return 0.0


def _mlb_hit_candidate_for_market(row: KalshiSportMarket) -> list[ParlayCandidateLeg]:
    market = row.market
    title = str(market.get("title") or "")
    player_name = title.split(":", 1)[0].strip()
    if not player_name:
        return []

    floor = market.get("floor_strike")
    try:
        milestone = int(float(floor) + 0.5)
    except (TypeError, ValueError):
        match = re.search(r":\s*(\d+)\+\s*hits", title, re.I)
        if not match:
            return []
        milestone = int(match.group(1))

    event_date = _parse_date(market)
    projection = project_mlb_hits(
        player_name=player_name,
        milestone_hits=milestone,
        event_date=event_date,
        event_ticker=str(market.get("event_ticker") or ""),
    )
    if projection is None or not projection.evidence.usable:
        return []

    yes_price = market_side_probability(market, "YES")
    no_price = market_side_probability(market, "NO")
    base = projection.evidence
    no_evidence = replace(
        base,
        fair_probability=1.0 - base.fair_probability,
        factors=tuple([
            f"Complement of {projection.player_name} {milestone}+ hits model probability",
            *base.factors,
        ]),
    )

    out: list[ParlayCandidateLeg] = []
    for side, price, evidence, selection_side in (
        ("YES", yes_price, base, "Over"),
        ("NO", no_price, no_evidence, "Under"),
    ):
        if price is None:
            continue
        selection = (
            f"{projection.player_name} {selection_side} "
            f"{projection.line:g} Hits"
        )
        out.append(
            ParlayCandidateLeg(
                sport="MLB",
                event_id=_canonical_game_id(
                    projection.evidence.sport,
                    projection.game_title,
                    _parse_date(market),
                ),
                event_title=projection.game_title,
                market_key="batter_hits",
                market_label="Hits",
                selection=selection,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _mlb_hit_candidates(
    rows: list[KalshiSportMarket],
    *,
    max_players: int | None = None,
    max_workers: int = 4,
) -> list[ParlayCandidateLeg]:
    """Bound MLB hit-model latency without weakening per-player analysis.

    Kalshi can expose several milestones per hitter. We rank distinct
    hitter/game groups by live Kalshi volume, optionally cap the number of
    hitters analyzed for a parlay request, then process different games in
    parallel. Rows for the same game stay in one worker so cached schedule and
    probable-pitcher data are shared instead of refetched concurrently.
    """
    player_groups: dict[tuple[str, str], list[KalshiSportMarket]] = {}
    for row in rows:
        market = row.market
        title = str(market.get("title") or "")
        player_name = title.split(":", 1)[0].strip()
        if not player_name:
            continue
        key = (_event_key(market), normalize(player_name))
        player_groups.setdefault(key, []).append(row)

    ranked_groups = sorted(
        player_groups.values(),
        key=lambda group: max((_market_volume(r.market) for r in group), default=0.0),
        reverse=True,
    )
    if max_players is not None and max_players > 0:
        ranked_groups = ranked_groups[:max_players]

    by_event: dict[str, list[list[KalshiSportMarket]]] = {}
    for group in ranked_groups:
        event_id = _event_key(group[0].market)
        by_event.setdefault(event_id, []).append(group)

    if not by_event:
        return []

    def build_event(groups: list[list[KalshiSportMarket]]) -> list[ParlayCandidateLeg]:
        event_rows: list[ParlayCandidateLeg] = []
        for group in groups:
            for row in group:
                event_rows.extend(_mlb_hit_candidate_for_market(row))
        return event_rows

    out: list[ParlayCandidateLeg] = []
    worker_count = max(1, min(max_workers, len(by_event)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(build_event, groups) for groups in by_event.values()]
        for future in as_completed(futures):
            try:
                out.extend(future.result())
            except Exception:
                continue

    return out


def _mlb_hr_candidate_for_market(row: KalshiSportMarket) -> list[ParlayCandidateLeg]:
    market = row.market
    title = str(market.get("title") or "")
    player_name = title.split(":", 1)[0].strip()
    if not player_name:
        return []

    floor = market.get("floor_strike")
    try:
        milestone = int(float(floor) + 0.5)
    except (TypeError, ValueError):
        match = re.search(r":\s*(\d+)\+\s*home runs?", title, re.I)
        if not match:
            match = re.search(r":\s*(\d+)\+\s*hrs?", title, re.I)
        if not match:
            return []
        milestone = int(match.group(1))

    projection = project_mlb_home_runs(
        player_name=player_name,
        milestone_home_runs=milestone,
        event_date=_parse_date(market),
        event_ticker=str(market.get("event_ticker") or ""),
    )
    if projection is None or not projection.evidence.usable:
        return []

    yes_price = market_side_probability(market, "YES")
    no_price = market_side_probability(market, "NO")
    base = projection.evidence
    no_evidence = replace(
        base,
        fair_probability=1.0 - base.fair_probability,
        factors=tuple([
            f"Complement of {projection.player_name} {milestone}+ home run model probability",
            *base.factors,
        ]),
    )

    out: list[ParlayCandidateLeg] = []
    for side, price, evidence, selection_side in (
        ("YES", yes_price, base, "Over"),
        ("NO", no_price, no_evidence, "Under"),
    ):
        if price is None:
            continue
        selection = (
            f"{projection.player_name} {selection_side} "
            f"{projection.line:g} Home Runs"
        )
        out.append(
            ParlayCandidateLeg(
                sport="MLB",
                event_id=_canonical_game_id(
                    projection.evidence.sport,
                    projection.game_title,
                    _parse_date(market),
                ),
                event_title=projection.game_title,
                market_key="batter_home_runs",
                market_label="Home Runs",
                selection=selection,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _mlb_hr_candidates(
    rows: list[KalshiSportMarket],
    *,
    max_players: int | None = None,
    max_workers: int = 4,
) -> list[ParlayCandidateLeg]:
    player_groups: dict[tuple[str, str], list[KalshiSportMarket]] = {}
    for row in rows:
        market = row.market
        title = str(market.get("title") or "")
        player_name = title.split(":", 1)[0].strip()
        if not player_name:
            continue
        key = (_event_key(market), normalize(player_name))
        player_groups.setdefault(key, []).append(row)

    ranked_groups = sorted(
        player_groups.values(),
        key=lambda group: max((_market_volume(r.market) for r in group), default=0.0),
        reverse=True,
    )
    if max_players is not None and max_players > 0:
        ranked_groups = ranked_groups[:max_players]

    by_event: dict[str, list[list[KalshiSportMarket]]] = {}
    for group in ranked_groups:
        event_id = _event_key(group[0].market)
        by_event.setdefault(event_id, []).append(group)

    if not by_event:
        return []

    def build_event(groups: list[list[KalshiSportMarket]]) -> list[ParlayCandidateLeg]:
        event_rows: list[ParlayCandidateLeg] = []
        for group in groups:
            for row in group:
                event_rows.extend(_mlb_hr_candidate_for_market(row))
        return event_rows

    out: list[ParlayCandidateLeg] = []
    worker_count = max(1, min(max_workers, len(by_event)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(build_event, groups) for groups in by_event.values()]
        for future in as_completed(futures):
            try:
                out.extend(future.result())
            except Exception:
                continue
    return out


def _mlb_tb_candidate_for_market(row: KalshiSportMarket) -> list[ParlayCandidateLeg]:
    market = row.market
    title = str(market.get("title") or "")
    player_name = title.split(":", 1)[0].strip()
    if not player_name:
        return []

    floor = market.get("floor_strike")
    try:
        milestone = int(float(floor) + 0.5)
    except (TypeError, ValueError):
        match = re.search(r":\s*(\d+)\+\s*total bases?", title, re.I)
        if not match:
            match = re.search(r":\s*(\d+)\+\s*tb", title, re.I)
        if not match:
            return []
        milestone = int(match.group(1))

    projection = project_mlb_total_bases(
        player_name=player_name,
        milestone_total_bases=milestone,
        event_date=_parse_date(market),
        event_ticker=str(market.get("event_ticker") or ""),
    )
    if projection is None or not projection.evidence.usable:
        return []

    yes_price = market_side_probability(market, "YES")
    no_price = market_side_probability(market, "NO")
    base = projection.evidence
    no_evidence = replace(
        base,
        fair_probability=1.0 - base.fair_probability,
        factors=tuple([
            f"Complement of {projection.player_name} {milestone}+ total bases model probability",
            *base.factors,
        ]),
    )

    out: list[ParlayCandidateLeg] = []
    for side, price, evidence, selection_side in (
        ("YES", yes_price, base, "Over"),
        ("NO", no_price, no_evidence, "Under"),
    ):
        if price is None:
            continue
        selection = (
            f"{projection.player_name} {selection_side} "
            f"{projection.line:g} Total Bases"
        )
        out.append(
            ParlayCandidateLeg(
                sport="MLB",
                event_id=_canonical_game_id(
                    projection.evidence.sport,
                    projection.game_title,
                    _parse_date(market),
                ),
                event_title=projection.game_title,
                market_key="batter_total_bases",
                market_label="Total Bases",
                selection=selection,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _mlb_tb_candidates(
    rows: list[KalshiSportMarket],
    *,
    max_players: int | None = None,
    max_workers: int = 4,
) -> list[ParlayCandidateLeg]:
    player_groups: dict[tuple[str, str], list[KalshiSportMarket]] = {}
    for row in rows:
        market = row.market
        title = str(market.get("title") or "")
        player_name = title.split(":", 1)[0].strip()
        if not player_name:
            continue
        key = (_event_key(market), normalize(player_name))
        player_groups.setdefault(key, []).append(row)

    ranked_groups = sorted(
        player_groups.values(),
        key=lambda group: max((_market_volume(r.market) for r in group), default=0.0),
        reverse=True,
    )
    if max_players is not None and max_players > 0:
        ranked_groups = ranked_groups[:max_players]

    by_event: dict[str, list[list[KalshiSportMarket]]] = {}
    for group in ranked_groups:
        event_id = _event_key(group[0].market)
        by_event.setdefault(event_id, []).append(group)

    if not by_event:
        return []

    def build_event(groups: list[list[KalshiSportMarket]]) -> list[ParlayCandidateLeg]:
        event_rows: list[ParlayCandidateLeg] = []
        for group in groups:
            for row in group:
                event_rows.extend(_mlb_tb_candidate_for_market(row))
        return event_rows

    out: list[ParlayCandidateLeg] = []
    worker_count = max(1, min(max_workers, len(by_event)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(build_event, groups) for groups in by_event.values()]
        for future in as_completed(futures):
            try:
                out.extend(future.result())
            except Exception:
                continue
    return out


def _mlb_run_candidate_for_market(row: KalshiSportMarket) -> list[ParlayCandidateLeg]:
    market = row.market
    title = str(market.get("title") or "")
    player_name = title.split(":", 1)[0].strip()
    if not player_name:
        return []

    floor = market.get("floor_strike")
    try:
        milestone = int(float(floor) + 0.5)
    except (TypeError, ValueError):
        match = re.search(r":\s*(\d+)\+", title)
        if not match:
            return []
        milestone = int(match.group(1))

    event_date = _parse_date(market)
    if row.family == "RBIs":
        projection = project_mlb_rbis(
            player_name=player_name,
            milestone_rbis=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )
    elif row.family == "Hits + Runs + RBIs":
        projection = project_mlb_hrr(
            player_name=player_name,
            milestone_hrr=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )
    else:
        return []

    if projection is None or not projection.evidence.usable:
        return []

    yes_price = market_side_probability(market, "YES")
    no_price = market_side_probability(market, "NO")
    base = projection.evidence
    no_evidence = replace(
        base,
        fair_probability=1.0 - base.fair_probability,
        factors=tuple([
            f"Complement of {projection.player_name} {milestone}+ {projection.market_label} model probability",
            *base.factors,
        ]),
    )

    out: list[ParlayCandidateLeg] = []
    for side, price, evidence, selection_side in (
        ("YES", yes_price, base, "Over"),
        ("NO", no_price, no_evidence, "Under"),
    ):
        if price is None:
            continue
        selection = (
            f"{projection.player_name} {selection_side} "
            f"{projection.line:g} {projection.market_label}"
        )
        out.append(
            ParlayCandidateLeg(
                sport="MLB",
                event_id=_canonical_game_id(
                    projection.evidence.sport,
                    projection.game_title,
                    event_date,
                ),
                event_title=projection.game_title,
                market_key=projection.market_key,
                market_label=projection.market_label,
                selection=selection,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _mlb_run_candidates(
    rows: list[KalshiSportMarket],
    *,
    max_players: int | None = None,
    max_workers: int = 4,
) -> list[ParlayCandidateLeg]:
    groups: dict[tuple[str, str], list[KalshiSportMarket]] = {}
    for row in rows:
        market = row.market
        title = str(market.get("title") or "")
        player_name = title.split(":", 1)[0].strip()
        if not player_name:
            continue
        groups.setdefault((_event_key(market), normalize(player_name)), []).append(row)

    ranked = sorted(
        groups.values(),
        key=lambda group: max((_market_volume(r.market) for r in group), default=0.0),
        reverse=True,
    )
    if not ranked:
        return []

    def build_group(group: list[KalshiSportMarket]) -> list[ParlayCandidateLeg]:
        group_rows: list[ParlayCandidateLeg] = []
        for row in group:
            group_rows.extend(_mlb_run_candidate_for_market(row))
        return group_rows

    # The cap counts valid modeled players, not raw high-volume player groups.
    # Completed/live/unresolvable groups often lead the volume ranking and must
    # not prevent the scan from reaching eligible pregame players behind them.
    if max_players is not None and max_players > 0:
        out: list[ParlayCandidateLeg] = []
        valid_players = 0
        batch_size = max(1, max_workers)
        # Bound live latency even when the highest-volume groups are already
        # in-play/completed. We search past invalid groups, but fail closed
        # after a finite 4x discovery window instead of walking the full slate.
        scan_limit = min(len(ranked), max_players * 4)
        ranked_scan = ranked[:scan_limit]
        for start in range(0, len(ranked_scan), batch_size):
            batch = ranked_scan[start:start + batch_size]
            worker_count = max(1, min(max_workers, len(batch)))
            batch_results: list[list[ParlayCandidateLeg]] = [[] for _ in batch]
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                future_to_index = {
                    pool.submit(build_group, group): idx
                    for idx, group in enumerate(batch)
                }
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        batch_results[idx] = future.result()
                    except Exception:
                        batch_results[idx] = []

            for group_rows in batch_results:
                if not group_rows:
                    continue
                out.extend(group_rows)
                valid_players += 1
                if valid_players >= max_players:
                    return out
        return out

    by_event: dict[str, list[list[KalshiSportMarket]]] = {}
    for group in ranked:
        by_event.setdefault(_event_key(group[0].market), []).append(group)

    def build_event(player_groups: list[list[KalshiSportMarket]]) -> list[ParlayCandidateLeg]:
        event_rows: list[ParlayCandidateLeg] = []
        for group in player_groups:
            event_rows.extend(build_group(group))
        return event_rows

    out: list[ParlayCandidateLeg] = []
    worker_count = max(1, min(max_workers, len(by_event)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(build_event, groups) for groups in by_event.values()]
        for future in as_completed(futures):
            try:
                out.extend(future.result())
            except Exception:
                continue
    return out


def _mlb_k_candidate_for_market(row: KalshiSportMarket) -> list[ParlayCandidateLeg]:
    market = row.market
    title = str(market.get("title") or "")
    player_name = title.split(":", 1)[0].strip()
    if not player_name:
        return []

    floor = market.get("floor_strike")
    try:
        milestone = int(float(floor) + 0.5)
    except (TypeError, ValueError):
        match = re.search(r":\s*(\d+)\+\s*(?:strikeouts?|ks?)", title, re.I)
        if not match:
            return []
        milestone = int(match.group(1))

    projection = project_mlb_pitcher_strikeouts(
        player_name=player_name,
        milestone_strikeouts=milestone,
        event_date=_parse_date(market),
        event_ticker=str(market.get("event_ticker") or ""),
    )
    if projection is None or not projection.evidence.usable:
        return []

    yes_price = market_side_probability(market, "YES")
    no_price = market_side_probability(market, "NO")
    base = projection.evidence
    no_evidence = replace(
        base,
        fair_probability=1.0 - base.fair_probability,
        factors=tuple([
            f"Complement of {projection.player_name} {milestone}+ strikeout model probability",
            *base.factors,
        ]),
    )

    out: list[ParlayCandidateLeg] = []
    for side, price, evidence, selection_side in (
        ("YES", yes_price, base, "Over"),
        ("NO", no_price, no_evidence, "Under"),
    ):
        if price is None:
            continue
        selection = (
            f"{projection.player_name} {selection_side} "
            f"{projection.line:g} Strikeouts"
        )
        out.append(
            ParlayCandidateLeg(
                sport="MLB",
                event_id=_canonical_game_id(
                    projection.evidence.sport,
                    projection.game_title,
                    _parse_date(market),
                ),
                event_title=projection.game_title,
                market_key="pitcher_strikeouts",
                market_label="Pitcher Strikeouts",
                selection=selection,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _mlb_k_candidates(
    rows: list[KalshiSportMarket],
    *,
    max_pitchers: int | None = None,
    max_workers: int = 4,
) -> list[ParlayCandidateLeg]:
    pitcher_groups: dict[tuple[str, str], list[KalshiSportMarket]] = {}
    for row in rows:
        market = row.market
        title = str(market.get("title") or "")
        player_name = title.split(":", 1)[0].strip()
        if not player_name:
            continue
        key = (_event_key(market), normalize(player_name))
        pitcher_groups.setdefault(key, []).append(row)

    ranked_groups = sorted(
        pitcher_groups.values(),
        key=lambda group: max((_market_volume(r.market) for r in group), default=0.0),
        reverse=True,
    )
    if max_pitchers is not None and max_pitchers > 0:
        ranked_groups = ranked_groups[:max_pitchers]

    by_event: dict[str, list[list[KalshiSportMarket]]] = {}
    for group in ranked_groups:
        event_id = _event_key(group[0].market)
        by_event.setdefault(event_id, []).append(group)

    if not by_event:
        return []

    def build_event(groups: list[list[KalshiSportMarket]]) -> list[ParlayCandidateLeg]:
        event_rows: list[ParlayCandidateLeg] = []
        for group in groups:
            for row in group:
                event_rows.extend(_mlb_k_candidate_for_market(row))
        return event_rows

    out: list[ParlayCandidateLeg] = []
    worker_count = max(1, min(max_workers, len(by_event)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(build_event, groups) for groups in by_event.values()]
        for future in as_completed(futures):
            try:
                out.extend(future.result())
            except Exception:
                continue
    return out


def _mlb_line_candidate_for_market(row: KalshiSportMarket) -> list[ParlayCandidateLeg]:
    market = row.market
    title = str(market.get("title") or "").strip()
    event_title = str(market.get("event_title") or "")
    try:
        line = float(market.get("floor_strike"))
    except (TypeError, ValueError):
        match = re.search(r"(\d+(?:\.\d+)?)", title)
        if not match:
            return []
        line = float(match.group(1))

    event_date = _parse_date(market)
    event_ticker = str(market.get("event_ticker") or "")
    projection = None

    if row.family == "Spread":
        match = re.match(r"(.+?)\s+wins(?:\s+the\s+game)?\s+by\s+over\s+", title, re.I)
        if not match:
            return []
        projection = project_mlb_spread(
            team_name=match.group(1).strip(),
            line=line,
            event_date=event_date,
            event_ticker=event_ticker,
            event_title=event_title,
        )
    elif row.family == "Game Total":
        projection = project_mlb_game_total(
            line=line,
            event_date=event_date,
            event_ticker=event_ticker,
            event_title=event_title,
        )
    elif row.family == "Team Total":
        match = re.match(r"(?:Will\s+)?(.+?)\s+(?:score\s+)?over\s+", title, re.I)
        if not match:
            return []
        projection = project_mlb_team_total(
            team_name=match.group(1).strip(),
            line=line,
            event_date=event_date,
            event_ticker=event_ticker,
            event_title=event_title,
        )

    if projection is None or not projection.evidence.usable:
        return []

    base = projection.evidence
    no_evidence = replace(
        base,
        fair_probability=1.0 - base.fair_probability,
        factors=tuple([
            f"Complement of {projection.selection_label} model probability",
            *base.factors,
        ]),
    )
    yes_price = market_side_probability(market, "YES")
    no_price = market_side_probability(market, "NO")

    if projection.market_key == "mlb_spread":
        yes_selection = projection.selection_label
        no_selection = projection.selection_label.replace(" > ", " ≤ ")
    elif projection.market_key == "mlb_game_total":
        yes_selection = f"Over {line:g} Game Total"
        no_selection = f"Under {line:g} Game Total"
    else:
        team_name = projection.selection_label.rsplit(" over ", 1)[0]
        yes_selection = f"{team_name} Over {line:g} Team Total"
        no_selection = f"{team_name} Under {line:g} Team Total"

    out: list[ParlayCandidateLeg] = []
    for side, price, evidence, selection in (
        ("YES", yes_price, base, yes_selection),
        ("NO", no_price, no_evidence, no_selection),
    ):
        if price is None:
            continue
        out.append(
            ParlayCandidateLeg(
                sport="MLB",
                event_id=_canonical_game_id("MLB", projection.game_title, event_date),
                event_title=projection.game_title,
                market_key=projection.market_key,
                market_label=projection.market_label,
                selection=selection,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _mlb_line_candidates(rows: list[KalshiSportMarket]) -> list[ParlayCandidateLeg]:
    out: list[ParlayCandidateLeg] = []
    for row in rows:
        out.extend(_mlb_line_candidate_for_market(row))
    return out


def _nfl_line_candidate_for_market(row: KalshiSportMarket) -> list[ParlayCandidateLeg]:
    market = row.market
    title = str(market.get("title") or "").strip()
    try:
        line = float(market.get("floor_strike"))
    except (TypeError, ValueError):
        match = re.search(r"(\d+(?:\.\d+)?)", title)
        if not match:
            return []
        line = float(match.group(1))

    event_date = _parse_date(market)
    event_ticker = str(market.get("event_ticker") or "")
    projection = None

    if row.family == "Spread":
        match = re.match(r"(.+?)\s+wins(?:\s+the\s+game)?\s+by\s+over\s+", title, re.I)
        if not match:
            return []
        projection = project_nfl_spread(
            team_name=match.group(1).strip(),
            line=line,
            event_date=event_date,
            event_ticker=event_ticker,
        )
    elif row.family == "Game Total":
        projection = project_nfl_game_total(
            line=line,
            event_date=event_date,
            event_ticker=event_ticker,
        )
    elif row.family == "Team Total":
        match = re.match(r"(.+?)\s+over\s+", title, re.I)
        if not match:
            return []
        projection = project_nfl_team_total(
            team_name=match.group(1).strip(),
            line=line,
            event_date=event_date,
            event_ticker=event_ticker,
        )

    if projection is None or not projection.evidence.usable:
        return []

    base = projection.evidence
    no_evidence = replace(
        base,
        fair_probability=1.0 - base.fair_probability,
        factors=tuple([
            f"Complement of {projection.selection_label} model probability",
            *base.factors,
        ]),
    )
    yes_price = market_side_probability(market, "YES")
    no_price = market_side_probability(market, "NO")

    if projection.market_key == "nfl_spread":
        yes_selection = projection.selection_label
        no_selection = projection.selection_label.replace(" > ", " ≤ ")
    elif projection.market_key == "nfl_game_total":
        yes_selection = f"Over {line:g} Game Total"
        no_selection = f"Under {line:g} Game Total"
    else:
        team_name = projection.selection_label.rsplit(" over ", 1)[0]
        yes_selection = f"{team_name} Over {line:g} Team Total"
        no_selection = f"{team_name} Under {line:g} Team Total"

    out: list[ParlayCandidateLeg] = []
    for side, price, evidence, selection in (
        ("YES", yes_price, base, yes_selection),
        ("NO", no_price, no_evidence, no_selection),
    ):
        if price is None:
            continue
        out.append(
            ParlayCandidateLeg(
                sport="NFL",
                event_id=_canonical_game_id("NFL", projection.game_title, event_date),
                event_title=projection.game_title,
                market_key=projection.market_key,
                market_label=projection.market_label,
                selection=selection,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _nfl_line_candidates(rows: list[KalshiSportMarket]) -> list[ParlayCandidateLeg]:
    out: list[ParlayCandidateLeg] = []
    for row in rows:
        out.extend(_nfl_line_candidate_for_market(row))
    return out


def _nfl_prop_candidate_for_market(row: KalshiSportMarket) -> list[ParlayCandidateLeg]:
    market = row.market
    title = str(market.get("title") or "")
    player_name = title.split(":", 1)[0].strip()
    if not player_name:
        return []
    low_name = normalize(player_name)
    if not low_name or "d st" in low_name or low_name in {"no touchdown", "no team"}:
        return []

    floor = market.get("floor_strike")
    try:
        milestone = int(float(floor) + 0.5)
    except (TypeError, ValueError):
        match = re.search(r":\s*(\d+)\+", title)
        if not match:
            return []
        milestone = int(match.group(1))

    event_date = _parse_date(market)
    projection = None
    if row.family == "Passing Yards":
        projection = project_nfl_passing_yards(
            player_name=player_name,
            milestone_yards=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )
    elif row.family == "Passing TDs":
        projection = project_nfl_passing_tds(
            player_name=player_name,
            milestone_tds=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )
    elif row.family == "Pass Attempts":
        projection = project_nfl_pass_attempts(
            player_name=player_name,
            milestone_attempts=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )
    elif row.family == "Pass Completions":
        projection = project_nfl_pass_completions(
            player_name=player_name,
            milestone_completions=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )
    elif row.family == "Pass Interceptions":
        projection = project_nfl_pass_interceptions(
            player_name=player_name,
            milestone_interceptions=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )
    elif row.family == "Rushing Yards":
        projection = project_nfl_rushing_yards(
            player_name=player_name,
            milestone_yards=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )
    elif row.family == "Rush Attempts":
        projection = project_nfl_rush_attempts(
            player_name=player_name,
            milestone_attempts=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )
    elif row.family == "Rushing + Receiving Yards":
        projection = project_nfl_rush_receiving_yards(
            player_name=player_name,
            milestone_yards=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )
    elif row.family == "Receiving Yards":
        projection = project_nfl_receiving_yards(
            player_name=player_name,
            milestone_yards=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )
    elif row.family == "Receptions":
        projection = project_nfl_receptions(
            player_name=player_name,
            milestone_receptions=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )
    elif row.family == "Player Touchdowns":
        projection = project_nfl_touchdowns(
            player_name=player_name,
            milestone_tds=milestone,
            event_date=event_date,
            event_ticker=str(market.get("event_ticker") or ""),
        )

    if projection is None or not projection.evidence.usable:
        return []

    yes_price = market_side_probability(market, "YES")
    no_price = market_side_probability(market, "NO")
    base = projection.evidence
    no_evidence = replace(
        base,
        fair_probability=1.0 - base.fair_probability,
        factors=tuple([
            f"Complement of {projection.player_name} {milestone}+ {projection.market_label} model probability",
            *base.factors,
        ]),
    )

    out: list[ParlayCandidateLeg] = []
    for side, price, evidence, selection_side in (
        ("YES", yes_price, base, "Over"),
        ("NO", no_price, no_evidence, "Under"),
    ):
        if price is None:
            continue
        selection = (
            f"{projection.player_name} {selection_side} "
            f"{projection.line:g} {projection.market_label}"
        )
        out.append(
            ParlayCandidateLeg(
                sport="NFL",
                event_id=_canonical_game_id(
                    projection.evidence.sport,
                    projection.game_title,
                    _parse_date(market),
                ),
                event_title=projection.game_title,
                market_key=projection.market_key,
                market_label=projection.market_label,
                selection=selection,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _nfl_prop_candidates(
    rows: list[KalshiSportMarket],
    *,
    max_players: int | None = None,
    max_workers: int = 4,
) -> list[ParlayCandidateLeg]:
    player_groups: dict[tuple[str, str], list[KalshiSportMarket]] = {}
    for row in rows:
        market = row.market
        title = str(market.get("title") or "")
        player_name = title.split(":", 1)[0].strip()
        if not player_name:
            continue
        key = (_event_key(market), normalize(player_name))
        player_groups.setdefault(key, []).append(row)

    ranked_groups = sorted(
        player_groups.values(),
        key=lambda group: max((_market_volume(r.market) for r in group), default=0.0),
        reverse=True,
    )
    if max_players is not None and max_players > 0:
        ranked_groups = ranked_groups[:max_players]

    by_event: dict[str, list[list[KalshiSportMarket]]] = {}
    for group in ranked_groups:
        event_id = _event_key(group[0].market)
        by_event.setdefault(event_id, []).append(group)
    if not by_event:
        return []

    def build_event(groups: list[list[KalshiSportMarket]]) -> list[ParlayCandidateLeg]:
        event_rows: list[ParlayCandidateLeg] = []
        for group in groups:
            for row in group:
                event_rows.extend(_nfl_prop_candidate_for_market(row))
        return event_rows

    out: list[ParlayCandidateLeg] = []
    worker_count = max(1, min(max_workers, len(by_event)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(build_event, groups) for groups in by_event.values()]
        for future in as_completed(futures):
            try:
                out.extend(future.result())
            except Exception:
                continue
    return out


def _wnba_player_candidate_for_market(row: KalshiSportMarket) -> list[ParlayCandidateLeg]:
    market = row.market
    title = str(market.get("title") or "").strip()
    player_name = title.split(":", 1)[0].strip()
    if not player_name:
        return []

    try:
        milestone = int(float(market.get("floor_strike")) + 0.5)
    except (TypeError, ValueError):
        match = re.search(r":\s*(\d+)\+", title)
        if not match:
            return []
        milestone = int(match.group(1))

    projection = project_wnba_player_prop(
        player_name=player_name,
        family=row.family,
        milestone=milestone,
        event_date=_parse_date(market),
        event_ticker=str(market.get("event_ticker") or ""),
    )
    if projection is None or not projection.evidence.usable:
        return []

    base = projection.evidence
    no_evidence = replace(
        base,
        fair_probability=1.0 - base.fair_probability,
        factors=tuple([
            f"Complement of {projection.player_name} {projection.milestone}+ {projection.market_label} model probability",
            *base.factors,
        ]),
    )
    yes_price = market_side_probability(market, "YES")
    no_price = market_side_probability(market, "NO")

    out: list[ParlayCandidateLeg] = []
    for side, price, evidence, selection_side in (
        ("YES", yes_price, base, "Over"),
        ("NO", no_price, no_evidence, "Under"),
    ):
        if price is None:
            continue
        out.append(
            ParlayCandidateLeg(
                sport="WNBA",
                event_id=_canonical_game_id(
                    "WNBA",
                    projection.game_title,
                    _parse_date(market),
                ),
                event_title=projection.game_title,
                market_key=projection.market_key,
                market_label=projection.market_label,
                selection=(
                    f"{projection.player_name} {selection_side} "
                    f"{projection.line:g} {projection.market_label}"
                ),
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _wnba_player_candidates(
    rows: list[KalshiSportMarket],
    *,
    max_players: int | None = None,
) -> list[ParlayCandidateLeg]:
    groups: dict[tuple[str, str], list[KalshiSportMarket]] = {}
    for row in rows:
        title = str(row.market.get("title") or "")
        player_name = title.split(":", 1)[0].strip()
        if not player_name:
            continue
        groups.setdefault((_event_key(row.market), normalize(player_name)), []).append(row)

    ranked = sorted(
        groups.values(),
        key=lambda group: max((_market_volume(x.market) for x in group), default=0.0),
        reverse=True,
    )

    out: list[ParlayCandidateLeg] = []
    modeled_players = 0
    scan_limit = len(ranked)
    if max_players is not None and max_players > 0:
        scan_limit = min(len(ranked), max_players * 3)

    for group in ranked[:scan_limit]:
        built: list[ParlayCandidateLeg] = []
        for row in group:
            built.extend(_wnba_player_candidate_for_market(row))
        if not built:
            continue
        out.extend(built)
        modeled_players += 1
        if max_players is not None and max_players > 0 and modeled_players >= max_players:
            break
    return out


def _wnba_line_candidate_for_market(row: KalshiSportMarket) -> list[ParlayCandidateLeg]:
    market = row.market
    title = str(market.get("title") or "").strip()
    try:
        line = float(market.get("floor_strike"))
    except (TypeError, ValueError):
        match = re.search(r"(\d+(?:\.\d+)?)", title)
        if not match:
            return []
        line = float(match.group(1))

    event_date = _parse_date(market)
    event_ticker = str(market.get("event_ticker") or "")
    projection = None

    if row.family == "Spread":
        match = re.match(r"(.+?)\s+wins the game by over\s+", title, re.I)
        if not match:
            return []
        projection = project_wnba_spread(
            team_name=match.group(1).strip(),
            line=line,
            event_date=event_date,
            event_ticker=event_ticker,
        )
    elif row.family == "Game Total":
        projection = project_wnba_game_total(
            line=line,
            event_date=event_date,
            event_ticker=event_ticker,
        )
    elif row.family == "Team Total":
        match = re.match(r"(.+?)\s+over\s+", title, re.I)
        if not match:
            return []
        projection = project_wnba_team_total(
            team_name=match.group(1).strip(),
            line=line,
            event_date=event_date,
            event_ticker=event_ticker,
        )

    if projection is None or not projection.evidence.usable:
        return []

    base = projection.evidence
    no_evidence = replace(
        base,
        fair_probability=1.0 - base.fair_probability,
        factors=tuple([
            f"Complement of {projection.selection_label} model probability",
            *base.factors,
        ]),
    )
    yes_price = market_side_probability(market, "YES")
    no_price = market_side_probability(market, "NO")

    if projection.market_key == "wnba_spread":
        yes_selection = projection.selection_label
        no_selection = projection.selection_label.replace(" > ", " ≤ ")
    elif projection.market_key == "wnba_game_total":
        yes_selection = f"Over {line:g} Game Total"
        no_selection = f"Under {line:g} Game Total"
    else:
        team_name = projection.selection_label.rsplit(" over ", 1)[0]
        yes_selection = f"{team_name} Over {line:g} Team Total"
        no_selection = f"{team_name} Under {line:g} Team Total"

    out: list[ParlayCandidateLeg] = []
    for side, price, evidence, selection in (
        ("YES", yes_price, base, yes_selection),
        ("NO", no_price, no_evidence, no_selection),
    ):
        if price is None:
            continue
        out.append(
            ParlayCandidateLeg(
                sport="WNBA",
                event_id=_canonical_game_id("WNBA", projection.game_title, event_date),
                event_title=projection.game_title,
                market_key=projection.market_key,
                market_label=projection.market_label,
                selection=selection,
                consensus_probability=evidence.fair_probability,
                book_count=0,
                source_age_s=0.0,
                median_odds=None,
                kalshi_ticker=str(market.get("ticker") or ""),
                kalshi_side=side,
                kalshi_price=price,
                kalshi_edge_points=100.0 * (evidence.fair_probability - price),
                kalshi_status="MODEL",
                evidence_class="MODEL",
                model_probability=evidence.fair_probability,
                model_confidence=evidence.confidence,
                model_name=evidence.model_name,
                model_sample_size=evidence.sample_size,
                model_reasons=evidence.factors,
                model_warnings=evidence.warnings,
            )
        )
    return out


def _wnba_line_candidates(rows: list[KalshiSportMarket]) -> list[ParlayCandidateLeg]:
    out: list[ParlayCandidateLeg] = []
    for row in rows:
        out.extend(_wnba_line_candidate_for_market(row))
    return out


def model_candidates_from_kalshi(
    grouped: dict[str, list[KalshiSportMarket]],
    *,
    sport_filter: str,
    include_mlb_hits: bool = True,
    max_mlb_hit_players: int | None = None,
    include_mlb_home_runs: bool = False,
    max_mlb_hr_players: int | None = None,
    include_mlb_total_bases: bool = False,
    max_mlb_tb_players: int | None = None,
    include_mlb_rbis: bool = False,
    max_mlb_rbi_players: int | None = None,
    include_mlb_hrr: bool = False,
    max_mlb_hrr_players: int | None = None,
    include_mlb_strikeouts: bool = False,
    max_mlb_k_pitchers: int | None = None,
    include_mlb_game_lines: bool = False,
    include_nfl_passing_yards: bool = False,
    max_nfl_passing_players: int | None = None,
    include_nfl_passing_tds: bool = False,
    include_nfl_pass_attempts: bool = False,
    include_nfl_pass_completions: bool = False,
    include_nfl_pass_interceptions: bool = False,
    include_nfl_rushing_yards: bool = False,
    include_nfl_rush_attempts: bool = False,
    include_nfl_rush_receiving_yards: bool = False,
    max_nfl_rushing_players: int | None = None,
    include_nfl_receiving_yards: bool = False,
    include_nfl_receptions: bool = False,
    max_nfl_receiving_players: int | None = None,
    include_nfl_touchdowns: bool = False,
    max_nfl_td_players: int | None = None,
    include_nfl_game_lines: bool = False,
    include_wnba_game_lines: bool = False,
    include_wnba_player_props: bool = False,
    max_wnba_players: int | None = None,
    include_tennis_match_winner: bool = True,
    include_tennis_games_total: bool = False,
    target_local_date: date | None = None,
    ticket_timezone: str = "Pacific/Honolulu",
) -> list[ParlayCandidateLeg]:
    sports = (
        ("MLB", "NBA", "WNBA", "NFL", "Tennis")
        if sport_filter == "All"
        else (sport_filter,)
    )
    all_rows: list[ParlayCandidateLeg] = []

    for sport in sports:
        rows = grouped.get(sport, [])
        if target_local_date is not None:
            rows = _rows_for_local_date(
                rows,
                target_date=target_local_date,
                timezone_name=ticket_timezone,
            )
        target_family = "Match Winner" if sport == "Tennis" else "Moneyline"
        events: dict[str, list[KalshiSportMarket]] = {}
        for row in rows:
            if row.family != target_family:
                continue
            key = _event_key(row.market)
            if key:
                events.setdefault(key, []).append(row)

        for event_rows in events.values():
            if sport == "Tennis":
                if include_tennis_match_winner:
                    all_rows.extend(_tennis_event_candidates(event_rows))
            else:
                all_rows.extend(_team_event_candidates(sport, event_rows))

        if sport == "Tennis" and include_tennis_games_total:
            # Experimental/secondary model families must fail closed. A malformed
            # live contract or upstream payload must never take down the entire
            # Ticket Builder (moneyline and other sports can still be modeled).
            try:
                all_rows.extend(_tennis_games_total_candidates(rows))
            except (TypeError, ValueError, KeyError, AttributeError):
                pass

        if sport == "MLB" and include_mlb_hits:
            hit_rows = [row for row in rows if row.family == "Hits"]
            all_rows.extend(
                _mlb_hit_candidates(
                    hit_rows,
                    max_players=max_mlb_hit_players,
                )
            )

        if sport == "MLB" and include_mlb_home_runs:
            hr_rows = [row for row in rows if row.family == "Home Runs"]
            all_rows.extend(
                _mlb_hr_candidates(
                    hr_rows,
                    max_players=max_mlb_hr_players,
                )
            )

        if sport == "MLB" and include_mlb_total_bases:
            tb_rows = [row for row in rows if row.family == "Total Bases"]
            all_rows.extend(
                _mlb_tb_candidates(
                    tb_rows,
                    max_players=max_mlb_tb_players,
                )
            )

        if sport == "MLB" and include_mlb_rbis:
            rbi_rows = [row for row in rows if row.family == "RBIs"]
            all_rows.extend(
                _mlb_run_candidates(
                    rbi_rows,
                    max_players=max_mlb_rbi_players,
                )
            )

        if sport == "MLB" and include_mlb_hrr:
            hrr_rows = [row for row in rows if row.family == "Hits + Runs + RBIs"]
            all_rows.extend(
                _mlb_run_candidates(
                    hrr_rows,
                    max_players=max_mlb_hrr_players,
                )
            )

        if sport == "MLB" and include_mlb_strikeouts:
            k_rows = [row for row in rows if row.family == "Strikeouts"]
            all_rows.extend(
                _mlb_k_candidates(
                    k_rows,
                    max_pitchers=max_mlb_k_pitchers,
                )
            )

        if sport == "MLB" and include_mlb_game_lines:
            line_rows = [
                row for row in rows
                if row.family in {"Spread", "Game Total", "Team Total"}
            ]
            all_rows.extend(_mlb_line_candidates(line_rows))

        if sport == "NFL" and include_nfl_passing_yards:
            passing_rows = [row for row in rows if row.family == "Passing Yards"]
            all_rows.extend(
                _nfl_prop_candidates(
                    passing_rows,
                    max_players=max_nfl_passing_players,
                )
            )

        if sport == "NFL" and include_nfl_passing_tds:
            passing_td_rows = [row for row in rows if row.family == "Passing TDs"]
            all_rows.extend(
                _nfl_prop_candidates(
                    passing_td_rows,
                    max_players=max_nfl_passing_players,
                )
            )

        if sport == "NFL" and include_nfl_pass_attempts:
            pass_attempt_rows = [row for row in rows if row.family == "Pass Attempts"]
            all_rows.extend(_nfl_prop_candidates(pass_attempt_rows, max_players=max_nfl_passing_players))

        if sport == "NFL" and include_nfl_pass_completions:
            pass_completion_rows = [row for row in rows if row.family == "Pass Completions"]
            all_rows.extend(_nfl_prop_candidates(pass_completion_rows, max_players=max_nfl_passing_players))

        if sport == "NFL" and include_nfl_pass_interceptions:
            pass_int_rows = [row for row in rows if row.family == "Pass Interceptions"]
            all_rows.extend(_nfl_prop_candidates(pass_int_rows, max_players=max_nfl_passing_players))

        if sport == "NFL" and include_nfl_rushing_yards:
            rushing_rows = [row for row in rows if row.family == "Rushing Yards"]
            all_rows.extend(
                _nfl_prop_candidates(
                    rushing_rows,
                    max_players=max_nfl_rushing_players,
                )
            )

        if sport == "NFL" and include_nfl_rush_attempts:
            rush_attempt_rows = [row for row in rows if row.family == "Rush Attempts"]
            all_rows.extend(_nfl_prop_candidates(rush_attempt_rows, max_players=max_nfl_rushing_players))

        if sport == "NFL" and include_nfl_rush_receiving_yards:
            rush_receive_rows = [row for row in rows if row.family == "Rushing + Receiving Yards"]
            all_rows.extend(_nfl_prop_candidates(rush_receive_rows, max_players=max_nfl_rushing_players))

        if sport == "NFL" and include_nfl_receiving_yards:
            receiving_rows = [row for row in rows if row.family == "Receiving Yards"]
            all_rows.extend(
                _nfl_prop_candidates(
                    receiving_rows,
                    max_players=max_nfl_receiving_players,
                )
            )

        if sport == "NFL" and include_nfl_receptions:
            reception_rows = [row for row in rows if row.family == "Receptions"]
            all_rows.extend(
                _nfl_prop_candidates(
                    reception_rows,
                    max_players=max_nfl_receiving_players,
                )
            )

        if sport == "NFL" and include_nfl_touchdowns:
            td_rows = [row for row in rows if row.family == "Player Touchdowns"]
            all_rows.extend(
                _nfl_prop_candidates(
                    td_rows,
                    max_players=max_nfl_td_players,
                )
            )

        if sport == "NFL" and include_nfl_game_lines:
            line_rows = [
                row for row in rows
                if row.family in {"Spread", "Game Total", "Team Total"}
            ]
            all_rows.extend(_nfl_line_candidates(line_rows))

        if sport == "WNBA" and include_wnba_game_lines:
            line_rows = [
                row for row in rows
                if row.family in {"Spread", "Game Total", "Team Total"}
            ]
            all_rows.extend(_wnba_line_candidates(line_rows))

        if sport == "WNBA" and include_wnba_player_props:
            prop_rows = [
                row for row in rows
                if row.family in {
                    "Points",
                    "Rebounds",
                    "Assists",
                    "Three-Pointers",
                    "Points + Rebounds + Assists",
                }
            ]
            all_rows.extend(
                _wnba_player_candidates(
                    prop_rows,
                    max_players=max_wnba_players,
                )
            )

    # Model candidates are allowed to include both sides; the EV gate chooses.
    return all_rows


def attach_sportsbook_context(
    model_candidates: list[ParlayCandidateLeg],
    book_candidates: list[ParlayCandidateLeg],
) -> list[ParlayCandidateLeg]:
    """Attach books only when physical event and selection both match.

    Canonical event identity prevents same-named selections from other games
    or Tennis matches contaminating model evidence. Ambiguous joins still fail
    closed.
    """
    by_key: dict[tuple[str, str, str], list[ParlayCandidateLeg]] = {}
    for row in book_candidates:
        key = (row.sport, row.event_id, normalize(row.selection))
        by_key.setdefault(key, []).append(row)

    out: list[ParlayCandidateLeg] = []
    for row in model_candidates:
        hits = by_key.get((row.sport, row.event_id, normalize(row.selection)), [])
        if len(hits) != 1:
            out.append(row)
            continue
        book = hits[0]
        out.append(
            replace(
                row,
                consensus_probability=book.consensus_probability,
                book_count=book.book_count,
                source_age_s=book.source_age_s,
                median_odds=book.median_odds,
                evidence_class="MODEL + BOOKS",
            )
        )
    return out
