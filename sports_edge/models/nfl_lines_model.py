from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import math
import re
from statistics import mean, pstdev

import requests

from sports_edge.core.math import clamp
from sports_edge.data.nfl_prop_data import nfl_season, schedule_rows
from sports_edge.data.public_team_data import NFL_TEAM_ALIASES
from sports_edge.models.game_scope import normalize
from sports_edge.models.model_evidence import ModelEvidence


_KALSHI_CODE = {"JAX": "JAC", "LA": "LAR", "WSH": "WAS"}


@dataclass(frozen=True)
class NFLLinesProjection:
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
    away: str
    home: str
    event_date: date
    away_current: tuple[_TeamGame, ...]
    home_current: tuple[_TeamGame, ...]
    away_prior: tuple[_TeamGame, ...]
    home_prior: tuple[_TeamGame, ...]

    @property
    def title(self) -> str:
        return f"{self.away} @ {self.home}"


def _f(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _d(value) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _normal_over(mu: float, sd: float, line: float) -> float:
    if sd <= 0:
        return 0.5
    z = (line - mu) / (sd * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + math.erf(z))
    return clamp(1.0 - cdf, 0.01, 0.99)


def _kalshi_team_code(code: str) -> str:
    return _KALSHI_CODE.get(code, code)


def _aliases(code: str) -> set[str]:
    vals = {normalize(code)}
    for value in NFL_TEAM_ALIASES.get(code, ()):
        vals.add(normalize(value))
    return {x for x in vals if x}


def _team_from_name(name: str, matchup: _Matchup) -> str | None:
    q = normalize(name)
    if not q:
        return None
    away_hit = q in _aliases(matchup.away)
    home_hit = q in _aliases(matchup.home)
    if away_hit == home_hit:
        return None
    return matchup.away if away_hit else matchup.home


def _ticker_matches(away: str, home: str, ticker: str) -> bool:
    t = re.sub(r"[^A-Z0-9]", "", str(ticker or "").upper())
    a = _kalshi_team_code(away)
    h = _kalshi_team_code(home)
    return a + h in t or h + a in t


def _history(
    rows: tuple[dict, ...],
    *,
    team: str,
    season: int,
    before: date,
) -> tuple[_TeamGame, ...]:
    out: list[_TeamGame] = []
    for row in rows:
        if str(row.get("season") or "") != str(season):
            continue
        if str(row.get("game_type") or "").strip().upper() not in {"REG", "POST"}:
            continue
        gd = _d(row.get("gameday"))
        if gd is None or gd >= before:
            continue
        home = str(row.get("home_team") or "").strip()
        away = str(row.get("away_team") or "").strip()
        if team not in {home, away}:
            continue
        hs, aws = _f(row.get("home_score")), _f(row.get("away_score"))
        if hs is None or aws is None:
            continue
        if team == home:
            out.append(_TeamGame(gd, hs, aws, True))
        else:
            out.append(_TeamGame(gd, aws, hs, False))
    return tuple(sorted(out, key=lambda g: g.game_date))


@lru_cache(maxsize=64)
def _resolve_matchup(event_date: date, event_ticker: str) -> _Matchup | None:
    rows = schedule_rows()
    season = nfl_season(event_date)
    matches: list[tuple[str, str]] = []
    for row in rows:
        if str(row.get("season") or "") != str(season):
            continue
        if str(row.get("game_type") or "").strip().upper() not in {"REG", "POST"}:
            continue
        if _d(row.get("gameday")) != event_date:
            continue
        away = str(row.get("away_team") or "").strip()
        home = str(row.get("home_team") or "").strip()
        if away and home and _ticker_matches(away, home, event_ticker):
            matches.append((away, home))
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        return None
    away, home = matches[0]

    away_current = _history(rows, team=away, season=season, before=event_date)
    home_current = _history(rows, team=home, season=season, before=event_date)
    away_prior = _history(rows, team=away, season=season - 1, before=date(season, 3, 1))
    home_prior = _history(rows, team=home, season=season - 1, before=date(season, 3, 1))

    # Early-season models need a real prior; later in the season current form
    # can stand on its own.
    if min(len(away_current) + len(away_prior), len(home_current) + len(home_prior)) < 8:
        return None

    return _Matchup(
        away=away,
        home=home,
        event_date=event_date,
        away_current=away_current,
        home_current=home_current,
        away_prior=away_prior,
        home_prior=home_prior,
    )


def _blended_rate(
    current: tuple[_TeamGame, ...],
    prior: tuple[_TeamGame, ...],
    attr: str,
) -> float:
    current_vals = [float(getattr(g, attr)) for g in current]
    prior_vals = [float(getattr(g, attr)) for g in prior[-18:]]
    if current_vals and prior_vals:
        current_weight = min(0.78, 0.38 + 0.08 * len(current_vals))
        base = current_weight * mean(current_vals) + (1.0 - current_weight) * mean(prior_vals)
        recent = mean(current_vals[-3:])
        recent_weight = 0.12 * clamp(len(current_vals) / 5.0, 0.0, 1.0)
        return (1.0 - recent_weight) * base + recent_weight * recent
    if current_vals:
        return mean(current_vals)
    return mean(prior_vals)


def _distribution_values(matchup: _Matchup, attr: str) -> list[float]:
    games = (
        list(matchup.away_prior[-12:])
        + list(matchup.home_prior[-12:])
        + list(matchup.away_current)
        + list(matchup.home_current)
    )
    return [float(getattr(g, attr)) for g in games]


def _score_projection(matchup: _Matchup) -> tuple[float, float, float, float, float, float]:
    away_pf = _blended_rate(matchup.away_current, matchup.away_prior, "pf")
    away_pa = _blended_rate(matchup.away_current, matchup.away_prior, "pa")
    home_pf = _blended_rate(matchup.home_current, matchup.home_prior, "pf")
    home_pa = _blended_rate(matchup.home_current, matchup.home_prior, "pa")

    away_score = 0.55 * away_pf + 0.45 * home_pa
    home_score = 0.55 * home_pf + 0.45 * away_pa

    # Conservative home-field adjustment. We do not treat market price as a
    # feature; this is purely a public-results model.
    away_score -= 0.75
    home_score += 0.75

    home_margin = home_score - away_score
    total_mean = home_score + away_score

    margins = _distribution_values(matchup, "margin")
    totals = _distribution_values(matchup, "total")
    margin_sd = clamp(pstdev(margins) if len(margins) >= 8 else 13.0, 9.0, 20.0)
    total_sd = clamp(pstdev(totals) if len(totals) >= 8 else 14.0, 10.0, 24.0)
    return away_score, home_score, home_margin, total_mean, margin_sd, total_sd


def _team_score_sd(current: tuple[_TeamGame, ...], prior: tuple[_TeamGame, ...]) -> float:
    vals = [g.pf for g in prior[-12:]] + [g.pf for g in current]
    return clamp(pstdev(vals) if len(vals) >= 8 else 10.5, 7.0, 17.0)


def _confidence(matchup: _Matchup, *, kind: str) -> float:
    current_n = min(len(matchup.away_current), len(matchup.home_current))
    prior_n = min(len(matchup.away_prior), len(matchup.home_prior))
    value = 0.46
    value += 0.10 * clamp(current_n / 5.0, 0.0, 1.0)
    value += 0.08 * clamp(prior_n / 14.0, 0.0, 1.0)
    if kind == "team_total":
        value -= 0.02
    return clamp(value, 0.45, 0.66)


def _warnings(matchup: _Matchup) -> tuple[str, ...]:
    out = [
        "pregame team-results model; injuries, weather, pace, and confirmed personnel are not yet separately modeled"
    ]
    if min(len(matchup.away_current), len(matchup.home_current)) < 5:
        out.append("early-season NFL sample; prior-season scoring baseline receives material weight")
    return tuple(out)


def project_nfl_spread(
    *,
    team_name: str,
    line: float,
    event_date: date,
    event_ticker: str,
) -> NFLLinesProjection | None:
    try:
        matchup = _resolve_matchup(event_date, event_ticker)
        if matchup is None:
            return None
        team = _team_from_name(team_name, matchup)
        if team is None:
            return None
        away_score, home_score, home_margin, total_mean, margin_sd, total_sd = _score_projection(matchup)
        team_margin = home_margin if team == matchup.home else -home_margin
        prob = _normal_over(team_margin, margin_sd, line)
        evidence = ModelEvidence(
            sport="NFL",
            model_name="NFL Spread: scoring/defense + current/prior form + home field",
            fair_probability=prob,
            confidence=_confidence(matchup, kind="spread"),
            sample_size=min(
                len(matchup.away_current) + len(matchup.away_prior),
                len(matchup.home_current) + len(matchup.home_prior),
            ),
            factors=(
                f"Projected score {matchup.away} {away_score:.1f} – {matchup.home} {home_score:.1f}",
                f"Projected {team} margin {team_margin:+.1f} points",
                f"Margin volatility {margin_sd:.1f} points",
                f"Current history {len(matchup.away_current)} / {len(matchup.home_current)} games; prior {len(matchup.away_prior)} / {len(matchup.home_prior)}",
            ),
            warnings=_warnings(matchup),
        )
        return NFLLinesProjection(
            evidence=evidence,
            market_key="nfl_spread",
            market_label="Spread",
            game_title=matchup.title,
            selection_label=f"{team} margin > {line:g}",
            line=line,
            projected_value=team_margin,
            projected_sd=margin_sd,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None


def project_nfl_game_total(
    *,
    line: float,
    event_date: date,
    event_ticker: str,
) -> NFLLinesProjection | None:
    try:
        matchup = _resolve_matchup(event_date, event_ticker)
        if matchup is None:
            return None
        away_score, home_score, home_margin, total_mean, margin_sd, total_sd = _score_projection(matchup)
        prob = _normal_over(total_mean, total_sd, line)
        evidence = ModelEvidence(
            sport="NFL",
            model_name="NFL Game Total: scoring/defense + current/prior form",
            fair_probability=prob,
            confidence=_confidence(matchup, kind="total"),
            sample_size=min(
                len(matchup.away_current) + len(matchup.away_prior),
                len(matchup.home_current) + len(matchup.home_prior),
            ),
            factors=(
                f"Projected score {matchup.away} {away_score:.1f} – {matchup.home} {home_score:.1f}",
                f"Projected game total {total_mean:.1f} points",
                f"Total volatility {total_sd:.1f} points",
            ),
            warnings=_warnings(matchup),
        )
        return NFLLinesProjection(
            evidence=evidence,
            market_key="nfl_game_total",
            market_label="Game Total",
            game_title=matchup.title,
            selection_label=f"Over {line:g} Game Total",
            line=line,
            projected_value=total_mean,
            projected_sd=total_sd,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None


def project_nfl_team_total(
    *,
    team_name: str,
    line: float,
    event_date: date,
    event_ticker: str,
) -> NFLLinesProjection | None:
    try:
        matchup = _resolve_matchup(event_date, event_ticker)
        if matchup is None:
            return None
        team = _team_from_name(team_name, matchup)
        if team is None:
            return None
        away_score, home_score, home_margin, total_mean, margin_sd, total_sd = _score_projection(matchup)
        if team == matchup.home:
            projected = home_score
            sd = _team_score_sd(matchup.home_current, matchup.home_prior)
        else:
            projected = away_score
            sd = _team_score_sd(matchup.away_current, matchup.away_prior)
        prob = _normal_over(projected, sd, line)
        evidence = ModelEvidence(
            sport="NFL",
            model_name="NFL Team Total: scoring/defense + current/prior form + home field",
            fair_probability=prob,
            confidence=_confidence(matchup, kind="team_total"),
            sample_size=min(
                len(matchup.away_current) + len(matchup.away_prior),
                len(matchup.home_current) + len(matchup.home_prior),
            ),
            factors=(
                f"Projected score {matchup.away} {away_score:.1f} – {matchup.home} {home_score:.1f}",
                f"Projected {team} score {projected:.1f} points",
                f"{team} scoring volatility {sd:.1f} points",
            ),
            warnings=_warnings(matchup),
        )
        return NFLLinesProjection(
            evidence=evidence,
            market_key="nfl_team_total",
            market_label="Team Total",
            game_title=matchup.title,
            selection_label=f"{team} over {line:g} points",
            line=line,
            projected_value=projected,
            projected_sd=sd,
        )
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None
