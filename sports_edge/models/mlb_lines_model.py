from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import math
import re

import requests

from sports_edge.core.math import clamp
from sports_edge.data.mlb_prop_data import schedule_for_day, team_season_stat
from sports_edge.models.game_scope import normalize
from sports_edge.models.model_evidence import ModelEvidence


@dataclass(frozen=True)
class MLBLineProjection:
    evidence: ModelEvidence
    market_key: str
    market_label: str
    game_title: str
    selection_label: str
    line: float
    projected_value: float


@dataclass(frozen=True)
class _Matchup:
    away_id: int
    home_id: int
    away_name: str
    home_name: str
    away_abbr: str
    home_abbr: str

    @property
    def title(self) -> str:
        return f"{self.away_name} @ {self.home_name}"


_CODE_ALIASES = {
    "CWS": {"CWS", "CHW"},
    "KCR": {"KCR", "KC"},
    "ARI": {"ARI", "AZ"},
    "WSH": {"WSH", "WAS"},
    "SDP": {"SDP", "SD"},
    "SFG": {"SFG", "SF"},
    "TBR": {"TBR", "TB"},
}


def _f(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _codes(abbr: str) -> set[str]:
    clean = re.sub(r"[^A-Z0-9]", "", str(abbr or "").upper())
    return _CODE_ALIASES.get(clean, {clean})


def _ticker_matches(away_abbr: str, home_abbr: str, ticker: str) -> bool:
    t = re.sub(r"[^A-Z0-9]", "", str(ticker or "").upper())
    return any(a + h in t or h + a in t for a in _codes(away_abbr) for h in _codes(home_abbr))


def _title_matches(away_name: str, home_name: str, title: str) -> bool:
    q = normalize(title)
    if not q:
        return False
    away = normalize(away_name)
    home = normalize(home_name)
    return bool(away and home and away in q and home in q)


@lru_cache(maxsize=128)
def _resolve_matchup(event_date: date, event_ticker: str, event_title: str) -> _Matchup | None:
    hits: list[_Matchup] = []
    for game in schedule_for_day(event_date.isoformat()):
        teams = game.get("teams") or {}
        away = teams.get("away") or {}
        home = teams.get("home") or {}
        away_team = away.get("team") or {}
        home_team = home.get("team") or {}
        try:
            away_id = int(away_team.get("id"))
            home_id = int(home_team.get("id"))
        except (TypeError, ValueError):
            continue
        away_name = str(away_team.get("name") or "").strip()
        home_name = str(home_team.get("name") or "").strip()
        away_abbr = str(away_team.get("abbreviation") or "").strip()
        home_abbr = str(home_team.get("abbreviation") or "").strip()
        if not away_name or not home_name:
            continue
        if not (
            _ticker_matches(away_abbr, home_abbr, event_ticker)
            or _title_matches(away_name, home_name, event_title)
        ):
            continue
        hits.append(_Matchup(away_id, home_id, away_name, home_name, away_abbr, home_abbr))
    unique = {(m.away_id, m.home_id): m for m in hits}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _team_rates(team_id: int, season: int) -> tuple[float, float, int] | None:
    hitting = team_season_stat(team_id, "hitting", season)
    pitching = team_season_stat(team_id, "pitching", season)
    if not hitting or not pitching:
        return None

    runs_for = _f(hitting.get("runs"))
    runs_allowed = _f(pitching.get("runs"))
    gp = _f(hitting.get("gamesPlayed")) or _f(pitching.get("gamesPlayed"))
    if runs_for is None or runs_allowed is None or gp is None or gp < 8:
        return None
    return runs_for / gp, runs_allowed / gp, int(gp)


def _poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return clamp(total, 0.0, 1.0)


def _poisson_over(line: float, lam: float) -> float:
    threshold = math.floor(line) + 1
    return clamp(1.0 - _poisson_cdf(threshold - 1, lam), 0.01, 0.99)


def _margin_over(line: float, team_lam: float, opp_lam: float) -> float:
    total = 0.0
    # MLB run tails above 20 are negligible for these means; retain explicit tail safety.
    for team_runs in range(0, 21):
        p_team = math.exp(-team_lam) * team_lam**team_runs / math.factorial(team_runs)
        for opp_runs in range(0, 21):
            if team_runs - opp_runs <= line:
                continue
            p_opp = math.exp(-opp_lam) * opp_lam**opp_runs / math.factorial(opp_runs)
            total += p_team * p_opp
    return clamp(total, 0.01, 0.99)


def _projection(event_date: date, event_ticker: str, event_title: str):
    matchup = _resolve_matchup(event_date, event_ticker, event_title)
    if matchup is None:
        return None
    away = _team_rates(matchup.away_id, event_date.year)
    home = _team_rates(matchup.home_id, event_date.year)
    if away is None or home is None:
        return None
    away_for, away_allowed, away_gp = away
    home_for, home_allowed, home_gp = home

    away_mu = 0.55 * away_for + 0.45 * home_allowed - 0.08
    home_mu = 0.55 * home_for + 0.45 * away_allowed + 0.08
    away_mu = clamp(away_mu, 2.0, 7.5)
    home_mu = clamp(home_mu, 2.0, 7.5)
    return matchup, away_mu, home_mu, min(away_gp, home_gp)


def _confidence(games: int, kind: str) -> float:
    value = 0.49 + 0.16 * clamp(games / 120.0, 0.0, 1.0)
    if kind == "team_total":
        value -= 0.02
    return clamp(value, 0.48, 0.66)


def _warnings() -> tuple[str, ...]:
    return (
        "pregame scoring model; confirmed lineups, starting-pitcher quality, bullpen availability, park, and weather are not separately modeled here",
    )


def _team_side(matchup: _Matchup, team_name: str) -> str | None:
    q = normalize(team_name)
    if not q:
        return None
    away_aliases = {normalize(matchup.away_name), normalize(matchup.away_abbr), *[normalize(x) for x in _codes(matchup.away_abbr)]}
    home_aliases = {normalize(matchup.home_name), normalize(matchup.home_abbr), *[normalize(x) for x in _codes(matchup.home_abbr)]}
    away_hit = q in away_aliases or any(q.startswith(x + " ") for x in away_aliases if x)
    home_hit = q in home_aliases or any(q.startswith(x + " ") for x in home_aliases if x)
    if away_hit == home_hit:
        return None
    return "away" if away_hit else "home"


def project_mlb_spread(
    *,
    team_name: str,
    line: float,
    event_date: date,
    event_ticker: str,
    event_title: str = "",
) -> MLBLineProjection | None:
    try:
        result = _projection(event_date, event_ticker, event_title)
        if result is None:
            return None
        matchup, away_mu, home_mu, games = result
        side = _team_side(matchup, team_name)
        if side is None:
            return None
        team_mu, opp_mu = (away_mu, home_mu) if side == "away" else (home_mu, away_mu)
        team = matchup.away_name if side == "away" else matchup.home_name
        prob = _margin_over(line, team_mu, opp_mu)
        evidence = ModelEvidence(
            sport="MLB",
            model_name="MLB Run Line: team scoring/allowance Poisson baseline",
            fair_probability=prob,
            confidence=_confidence(games, "spread"),
            sample_size=games,
            factors=(
                f"Projected score {matchup.away_name} {away_mu:.2f} – {matchup.home_name} {home_mu:.2f}",
                f"Projected {team} margin {team_mu - opp_mu:+.2f} runs",
                f"Season scoring depth {games} games minimum",
            ),
            warnings=_warnings(),
        )
        return MLBLineProjection(evidence, "mlb_spread", "Spread", matchup.title, f"{team} margin > {line:g}", line, team_mu - opp_mu)
    except (requests.RequestException, TypeError, ValueError, KeyError, OverflowError):
        return None


def project_mlb_game_total(
    *,
    line: float,
    event_date: date,
    event_ticker: str,
    event_title: str = "",
) -> MLBLineProjection | None:
    try:
        result = _projection(event_date, event_ticker, event_title)
        if result is None:
            return None
        matchup, away_mu, home_mu, games = result
        total_mu = away_mu + home_mu
        prob = _poisson_over(line, total_mu)
        evidence = ModelEvidence(
            sport="MLB",
            model_name="MLB Game Total: team scoring/allowance Poisson baseline",
            fair_probability=prob,
            confidence=_confidence(games, "total"),
            sample_size=games,
            factors=(
                f"Projected score {matchup.away_name} {away_mu:.2f} – {matchup.home_name} {home_mu:.2f}",
                f"Projected game total {total_mu:.2f} runs",
                f"Season scoring depth {games} games minimum",
            ),
            warnings=_warnings(),
        )
        return MLBLineProjection(evidence, "mlb_game_total", "Game Total", matchup.title, f"Over {line:g} Game Total", line, total_mu)
    except (requests.RequestException, TypeError, ValueError, KeyError, OverflowError):
        return None


def project_mlb_team_total(
    *,
    team_name: str,
    line: float,
    event_date: date,
    event_ticker: str,
    event_title: str = "",
) -> MLBLineProjection | None:
    try:
        result = _projection(event_date, event_ticker, event_title)
        if result is None:
            return None
        matchup, away_mu, home_mu, games = result
        side = _team_side(matchup, team_name)
        if side is None:
            return None
        team_mu = away_mu if side == "away" else home_mu
        team = matchup.away_name if side == "away" else matchup.home_name
        prob = _poisson_over(line, team_mu)
        evidence = ModelEvidence(
            sport="MLB",
            model_name="MLB Team Total: team scoring/allowance Poisson baseline",
            fair_probability=prob,
            confidence=_confidence(games, "team_total"),
            sample_size=games,
            factors=(
                f"Projected score {matchup.away_name} {away_mu:.2f} – {matchup.home_name} {home_mu:.2f}",
                f"Projected {team} score {team_mu:.2f} runs",
                f"Season scoring depth {games} games minimum",
            ),
            warnings=_warnings(),
        )
        return MLBLineProjection(evidence, "mlb_team_total", "Team Total", matchup.title, f"{team} over {line:g} runs", line, team_mu)
    except (requests.RequestException, TypeError, ValueError, KeyError, OverflowError):
        return None
