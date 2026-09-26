from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import re
from statistics import mean, pstdev

import requests

from sports_edge.core.math import clamp
from sports_edge.data.public_team_data import _truthy, _wnba_schedule_rows
from sports_edge.models.game_scope import normalize
from sports_edge.models.model_evidence import ModelEvidence


@dataclass(frozen=True)
class WNBALineProjection:
    evidence: ModelEvidence
    market_key: str
    market_label: str
    game_title: str
    selection_label: str
    line: float
    projected_value: float
    projected_sd: float


@dataclass(frozen=True)
class _TeamGame:
    game_date: date
    pf: float
    pa: float
    home: bool

    @property
    def margin(self) -> float:
        return self.pf - self.pa

    @property
    def total(self) -> float:
        return self.pf + self.pa


@dataclass(frozen=True)
class _Matchup:
    away_id: int
    home_id: int
    away_name: str
    home_name: str
    away_abbr: str
    home_abbr: str
    away_games: tuple[_TeamGame, ...]
    home_games: tuple[_TeamGame, ...]

    @property
    def title(self) -> str:
        return f"{self.away_name} @ {self.home_name}"


def _f(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normal_over(mean_value: float, sd: float, line: float) -> float:
    if sd <= 0:
        return 0.5
    z = (line - mean_value) / (sd * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + math.erf(z))
    return clamp(1.0 - cdf, 0.01, 0.99)


def _clean_code(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _row_date(row: dict) -> date | None:
    raw = str(row.get("game_date") or row.get("start_date") or "")[:10]
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _team_display(row: dict, side: str) -> str:
    return str(
        row.get(f"{side}_display_name")
        or row.get(f"{side}_short_display_name")
        or row.get(f"{side}_name")
        or row.get(f"{side}_location")
        or row.get(f"{side}_abbreviation")
        or ""
    ).strip()


def _match_event_row(rows: tuple[dict, ...], event_date: date, event_ticker: str) -> dict | None:
    ticker = _clean_code(event_ticker)
    hits: list[dict] = []
    for row in rows:
        if _row_date(row) != event_date:
            continue
        away = _clean_code(row.get("away_abbreviation"))
        home = _clean_code(row.get("home_abbreviation"))
        if not away or not home:
            continue
        if away + home in ticker or home + away in ticker:
            hits.append(row)
    return hits[0] if len(hits) == 1 else None


def _history_for_team(
    rows: tuple[dict, ...],
    *,
    team_id: int,
    before: date,
) -> tuple[_TeamGame, ...]:
    out: list[_TeamGame] = []
    for row in rows:
        gd = _row_date(row)
        if gd is None or gd >= before:
            continue
        if str(row.get("season_type") or row.get("type_id") or "") not in {"2", "3"}:
            continue
        if not _truthy(row.get("status_type_completed")):
            continue
        try:
            home_id = int(row.get("home_id"))
            away_id = int(row.get("away_id"))
        except (TypeError, ValueError):
            continue
        if team_id not in {home_id, away_id}:
            continue
        hs, aws = _f(row.get("home_score")), _f(row.get("away_score"))
        if hs is None or aws is None:
            continue
        if team_id == home_id:
            out.append(_TeamGame(gd, hs, aws, True))
        else:
            out.append(_TeamGame(gd, aws, hs, False))
    return tuple(sorted(out, key=lambda g: g.game_date))


def _resolve_matchup(event_date: date, event_ticker: str) -> _Matchup | None:
    rows = _wnba_schedule_rows()
    event = _match_event_row(rows, event_date, event_ticker)
    if event is None:
        return None
    try:
        away_id = int(event.get("away_id"))
        home_id = int(event.get("home_id"))
    except (TypeError, ValueError):
        return None
    away_name = _team_display(event, "away")
    home_name = _team_display(event, "home")
    away_abbr = str(event.get("away_abbreviation") or "").strip()
    home_abbr = str(event.get("home_abbreviation") or "").strip()
    if not away_name or not home_name:
        return None
    away_games = _history_for_team(rows, team_id=away_id, before=event_date)
    home_games = _history_for_team(rows, team_id=home_id, before=event_date)
    if min(len(away_games), len(home_games)) < 8:
        return None
    return _Matchup(
        away_id=away_id,
        home_id=home_id,
        away_name=away_name,
        home_name=home_name,
        away_abbr=away_abbr,
        home_abbr=home_abbr,
        away_games=away_games,
        home_games=home_games,
    )


def _team_aliases(name: str, abbr: str) -> set[str]:
    values = {normalize(name), normalize(abbr)}
    parts = normalize(name).split()
    if parts:
        values.add(parts[-1])
    if len(parts) >= 2:
        values.add(" ".join(parts[-2:]))
    return {v for v in values if v}


def _which_team(matchup: _Matchup, name: str) -> str | None:
    q = normalize(name)
    if not q:
        return None
    away_aliases = _team_aliases(matchup.away_name, matchup.away_abbr)
    home_aliases = _team_aliases(matchup.home_name, matchup.home_abbr)
    away_hit = q in away_aliases or any(q == x for x in away_aliases)
    home_hit = q in home_aliases or any(q == x for x in home_aliases)
    if away_hit == home_hit:
        return None
    return "away" if away_hit else "home"


def _weighted_rate(games: tuple[_TeamGame, ...], attr: str) -> float:
    season = [float(getattr(g, attr)) for g in games]
    recent = season[-8:]
    return 0.72 * mean(season) + 0.28 * mean(recent)


def _score_projection(matchup: _Matchup) -> tuple[float, float, float, float, float]:
    away_pf = _weighted_rate(matchup.away_games, "pf")
    away_pa = _weighted_rate(matchup.away_games, "pa")
    home_pf = _weighted_rate(matchup.home_games, "pf")
    home_pa = _weighted_rate(matchup.home_games, "pa")

    away_score = 0.55 * away_pf + 0.45 * home_pa
    home_score = 0.55 * home_pf + 0.45 * away_pa

    # Approx. 2.5-point WNBA home-court margin, applied symmetrically so the
    # total expectation is not mechanically inflated.
    away_score -= 1.25
    home_score += 1.25

    margin_mean = home_score - away_score
    total_mean = home_score + away_score

    margins = [g.margin for g in matchup.away_games[-20:]] + [
        -g.margin for g in matchup.home_games[-20:]
    ]
    totals = [g.total for g in matchup.away_games[-20:]] + [
        g.total for g in matchup.home_games[-20:]
    ]
    margin_sd = clamp(pstdev(margins) if len(margins) >= 4 else 12.0, 9.0, 18.0)
    total_sd = clamp(pstdev(totals) if len(totals) >= 4 else 14.0, 10.0, 24.0)
    return away_score, home_score, margin_mean, total_mean, max(margin_sd, total_sd)


def _team_score_sd(games: tuple[_TeamGame, ...]) -> float:
    values = [g.pf for g in games[-24:]]
    return clamp(pstdev(values) if len(values) >= 4 else 10.0, 7.0, 16.0)


def _confidence(matchup: _Matchup, *, line_type: str) -> float:
    n = min(len(matchup.away_games), len(matchup.home_games))
    value = 0.48 + 0.16 * clamp(n / 36.0, 0.0, 1.0)
    if line_type == "team_total":
        value -= 0.025
    return clamp(value, 0.48, 0.66)


def project_wnba_spread(
    *,
    team_name: str,
    line: float,
    event_date: date,
    event_ticker: str,
) -> WNBALineProjection | None:
    try:
        matchup = _resolve_matchup(event_date, event_ticker)
        if matchup is None:
            return None
        side = _which_team(matchup, team_name)
        if side is None:
            return None
        away_score, home_score, home_margin, total_mean, pooled_sd = _score_projection(matchup)
        team_margin = home_margin if side == "home" else -home_margin
        margin_values = [g.margin for g in matchup.away_games[-20:]] + [
            g.margin for g in matchup.home_games[-20:]
        ]
        margin_sd = clamp(pstdev(margin_values) if len(margin_values) >= 4 else 12.0, 9.0, 18.0)
        prob = _normal_over(team_margin, margin_sd, line)
        chosen_name = matchup.home_name if side == "home" else matchup.away_name
        factors = (
            f"Projected score {matchup.away_name} {away_score:.1f} – {matchup.home_name} {home_score:.1f}",
            f"Projected {chosen_name} margin {team_margin:+.1f} points",
            f"Margin volatility {margin_sd:.1f} points from recent team results",
            f"History depth {len(matchup.away_games)} / {len(matchup.home_games)} games",
        )
        evidence = ModelEvidence(
            sport="WNBA",
            model_name="WNBA Spread: scoring/defense + recent form + home court",
            fair_probability=prob,
            confidence=_confidence(matchup, line_type="spread"),
            sample_size=min(len(matchup.away_games), len(matchup.home_games)),
            factors=factors,
            warnings=("pregame model; injuries/confirmed rotations are not yet separately modeled",),
        )
        return WNBALineProjection(
            evidence=evidence,
            market_key="wnba_spread",
            market_label="Spread",
            game_title=matchup.title,
            selection_label=f"{chosen_name} margin > {line:g}",
            line=line,
            projected_value=team_margin,
            projected_sd=margin_sd,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None


def project_wnba_game_total(
    *,
    line: float,
    event_date: date,
    event_ticker: str,
) -> WNBALineProjection | None:
    try:
        matchup = _resolve_matchup(event_date, event_ticker)
        if matchup is None:
            return None
        away_score, home_score, home_margin, total_mean, pooled_sd = _score_projection(matchup)
        totals = [g.total for g in matchup.away_games[-20:]] + [
            g.total for g in matchup.home_games[-20:]
        ]
        total_sd = clamp(pstdev(totals) if len(totals) >= 4 else 14.0, 10.0, 24.0)
        prob = _normal_over(total_mean, total_sd, line)
        evidence = ModelEvidence(
            sport="WNBA",
            model_name="WNBA Game Total: scoring/defense + recent form",
            fair_probability=prob,
            confidence=_confidence(matchup, line_type="total"),
            sample_size=min(len(matchup.away_games), len(matchup.home_games)),
            factors=(
                f"Projected score {matchup.away_name} {away_score:.1f} – {matchup.home_name} {home_score:.1f}",
                f"Projected game total {total_mean:.1f} points",
                f"Total volatility {total_sd:.1f} points from recent team results",
                f"History depth {len(matchup.away_games)} / {len(matchup.home_games)} games",
            ),
            warnings=("pregame model; injuries/confirmed rotations are not yet separately modeled",),
        )
        return WNBALineProjection(
            evidence=evidence,
            market_key="wnba_game_total",
            market_label="Game Total",
            game_title=matchup.title,
            selection_label=f"Over {line:g} total points",
            line=line,
            projected_value=total_mean,
            projected_sd=total_sd,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None


def project_wnba_team_total(
    *,
    team_name: str,
    line: float,
    event_date: date,
    event_ticker: str,
) -> WNBALineProjection | None:
    try:
        matchup = _resolve_matchup(event_date, event_ticker)
        if matchup is None:
            return None
        side = _which_team(matchup, team_name)
        if side is None:
            return None
        away_score, home_score, home_margin, total_mean, pooled_sd = _score_projection(matchup)
        if side == "home":
            projected = home_score
            games = matchup.home_games
            chosen_name = matchup.home_name
        else:
            projected = away_score
            games = matchup.away_games
            chosen_name = matchup.away_name
        score_sd = _team_score_sd(games)
        prob = _normal_over(projected, score_sd, line)
        evidence = ModelEvidence(
            sport="WNBA",
            model_name="WNBA Team Total: scoring/defense + recent form + home court",
            fair_probability=prob,
            confidence=_confidence(matchup, line_type="team_total"),
            sample_size=min(len(matchup.away_games), len(matchup.home_games)),
            factors=(
                f"Projected score {matchup.away_name} {away_score:.1f} – {matchup.home_name} {home_score:.1f}",
                f"Projected {chosen_name} score {projected:.1f} points",
                f"{chosen_name} scoring volatility {score_sd:.1f} points",
                f"History depth {len(matchup.away_games)} / {len(matchup.home_games)} games",
            ),
            warnings=("pregame model; injuries/confirmed rotations are not yet separately modeled",),
        )
        return WNBALineProjection(
            evidence=evidence,
            market_key="wnba_team_total",
            market_label="Team Total",
            game_title=matchup.title,
            selection_label=f"{chosen_name} over {line:g} points",
            line=line,
            projected_value=projected,
            projected_sd=score_sd,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None
