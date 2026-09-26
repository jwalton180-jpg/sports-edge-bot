from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timezone
from functools import lru_cache
import re
from typing import Iterable

from sports_edge.data.public_team_data import team_game_model
from sports_edge.models.game_scope import normalize
from sports_edge.models.kalshi_sports import KalshiSportMarket
from sports_edge.models.live_board import market_side_probability
from sports_edge.models.mlb_hits_model import project_mlb_hits
from sports_edge.models.mlb_hr_model import project_mlb_home_runs
from sports_edge.models.mlb_run_production_model import project_mlb_hrr, project_mlb_rbis
from sports_edge.models.mlb_total_bases_model import project_mlb_total_bases
from sports_edge.models.mlb_strikeouts_model import project_mlb_pitcher_strikeouts
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


def _canonical_game_id(sport: str, game_title: str, event_date: date) -> str:
    """Stable game identity shared across Kalshi prop series for correlation control."""
    parts = [
        normalize(part)
        for part in re.split(r"\s+(?:vs\.?|@)\s+", str(game_title or "").strip(), maxsplit=1, flags=re.I)
        if normalize(part)
    ]
    if len(parts) == 2:
        participants = "|".join(sorted(parts))
    else:
        participants = normalize(game_title) or "unknown"
    return f"{sport.upper()}:{event_date.isoformat()}:{participants}"


def _event_title(market: dict) -> str:
    return str(market.get("event_title") or market.get("title") or _event_key(market)).strip()


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


def _tennis_event_candidates(rows: list[KalshiSportMarket]) -> list[ParlayCandidateLeg]:
    choices = _event_choices(rows)
    if len(choices) != 2:
        return []

    first_market = choices[0][3]
    series = str(first_market.get("series_ticker") or "")
    gender, level = tennis_level_from_series(series)
    event_date = _parse_date(first_market)

    a_name, a_side, a_price, a_market = choices[0]
    b_name, b_side, b_price, b_market = choices[1]
    model = _tennis_model(gender, event_date.year)
    evidence_a = model.probability(a_name, b_name, level=level, as_of=event_date)
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
                event_id=_event_key(market),
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
                event_id=_event_key(market),
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
        for start in range(0, len(ranked), batch_size):
            batch = ranked[start:start + batch_size]
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
) -> list[ParlayCandidateLeg]:
    sports = (
        ("MLB", "NBA", "WNBA", "NFL", "Tennis")
        if sport_filter == "All"
        else (sport_filter,)
    )
    all_rows: list[ParlayCandidateLeg] = []

    for sport in sports:
        rows = grouped.get(sport, [])
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
                all_rows.extend(_tennis_event_candidates(event_rows))
            else:
                all_rows.extend(_team_event_candidates(sport, event_rows))

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

    # Model candidates are allowed to include both sides; the EV gate chooses.
    return all_rows


def attach_sportsbook_context(
    model_candidates: list[ParlayCandidateLeg],
    book_candidates: list[ParlayCandidateLeg],
) -> list[ParlayCandidateLeg]:
    """Attach books only when a selection match is unique within the sport.

    Ambiguous joins fail closed: the model candidate stays model-only rather
    than inheriting evidence from the wrong game.
    """
    by_key: dict[tuple[str, str], list[ParlayCandidateLeg]] = {}
    for row in book_candidates:
        key = (row.sport, normalize(row.selection))
        by_key.setdefault(key, []).append(row)

    out: list[ParlayCandidateLeg] = []
    for row in model_candidates:
        hits = by_key.get((row.sport, normalize(row.selection)), [])
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
