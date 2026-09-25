from __future__ import annotations

from functools import lru_cache
from math import erfc, exp, factorial, sqrt
from statistics import mean, pstdev

from sports_edge.core.math import clamp
from sports_edge.data.public_player_data import (
    mlb_recent_rows,
    mlb_season_rows,
    nba_player_rows,
    nfl_weekly_rows,
    normalize_person,
    resolve_person,
    unique_person_index,
    wnba_player_rows,
)
from sports_edge.models.model_evidence import ModelEvidence


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _poisson_tail(lam: float, threshold: float) -> float:
    """P(X > threshold) for a nonnegative integer Poisson variable."""
    lam = max(0.01, float(lam))
    k = max(0, int(threshold + 0.5000001))
    cdf = 0.0
    for x in range(k):
        cdf += exp(-lam) * (lam ** x) / factorial(x)
    return clamp(1.0 - cdf, 0.005, 0.995)


def _normal_over(mean_value: float, stddev: float, line: float) -> float:
    sd = max(float(stddev), 1e-6)
    z = (float(line) - float(mean_value)) / (sd * sqrt(2.0))
    return clamp(0.5 * erfc(z), 0.005, 0.995)


def _empirical_projection(
    current: list[float],
    prior: list[float],
    *,
    line: float,
    count_model: bool,
) -> tuple[float, float, int, float, float] | None:
    """Return probability, confidence, sample size, projected mean, projected sd."""
    current = [float(x) for x in current]
    prior = [float(x) for x in prior]
    total = len(current) + len(prior)
    if total < 4:
        return None

    cur_mean = mean(current) if current else None
    prior_mean = mean(prior) if prior else None
    if cur_mean is not None and prior_mean is not None:
        # Current season drives the projection, with prior data stabilizing
        # small early-season samples.
        cur_w = clamp(len(current) / (len(current) + 4.0), 0.35, 0.78)
        projected_mean = cur_w * cur_mean + (1.0 - cur_w) * prior_mean
    elif cur_mean is not None:
        projected_mean = cur_mean
    else:
        projected_mean = float(prior_mean)

    pool = prior[-16:] + current[-10:]
    sd = pstdev(pool) if len(pool) >= 2 else max(1.0, abs(projected_mean) * 0.30)

    if count_model:
        probability = _poisson_tail(projected_mean, line)
    else:
        # Blend a smooth distribution with direct hit-rate evidence. This is
        # deliberately conservative for small samples.
        p_dist = _normal_over(projected_mean, max(sd, 0.15 * max(1.0, projected_mean)), line)
        wins = sum(1 for value in pool if value > line)
        p_emp = (wins + 1.0) / (len(pool) + 2.0)
        probability = clamp(0.60 * p_dist + 0.40 * p_emp, 0.01, 0.99)

    current_support = clamp(len(current) / 6.0, 0.0, 1.0)
    total_support = clamp(total / 16.0, 0.0, 1.0)
    confidence = clamp(0.34 + 0.28 * total_support + 0.18 * current_support, 0.0, 0.80)
    return probability, confidence, total, projected_mean, sd


@lru_cache(maxsize=4)
def _mlb_index(group: str, season: int) -> dict[str, dict]:
    rows = mlb_season_rows(group, season)
    return unique_person_index(rows, lambda row: row.get("player_name"))


@lru_cache(maxsize=8)
def _mlb_recent_index(group: str, season: int, games: int) -> dict[str, dict]:
    rows = mlb_recent_rows(group, season, games)
    return unique_person_index(rows, lambda row: row.get("player_name"))


def mlb_prop_evidence(
    player: str,
    family: str,
    *,
    line: float,
    season: int,
) -> ModelEvidence | None:
    group = "pitching" if family == "Strikeouts" else "hitting"
    row = resolve_person(_mlb_index(group, season), player)
    if row is None:
        return None

    recent_games = 5 if family == "Strikeouts" else 15
    recent = resolve_person(_mlb_recent_index(group, season, recent_games), player)

    if family == "Hits":
        games = int(_float(row.get("gamesPlayed")))
        hits = _float(row.get("hits"))
        at_bats = _float(row.get("atBats"))
        if games < 20 or at_bats < 50:
            return None
        season_rate = hits / max(1.0, games)
        recent_rate = None
        recent_n = 0
        if recent is not None:
            recent_n = int(_float(recent.get("gamesPlayed")))
            recent_hits = _float(recent.get("hits"))
            if recent_n >= 5:
                recent_rate = recent_hits / recent_n
        rate = 0.72 * season_rate + 0.28 * recent_rate if recent_rate is not None else season_rate
        p = _poisson_tail(rate, line)
        confidence = clamp(
            0.42 + 0.26 * min(1.0, games / 100.0) + (0.05 if recent_rate is not None else 0.0),
            0.0, 0.76,
        )
        factors = [
            f"2026 MLB: {hits:.0f} hits in {games} games",
            f"Contact rate {hits / max(1.0, at_bats):.3f} over {at_bats:.0f} AB",
            f"Season hit rate {season_rate:.2f}/game",
        ]
        if recent_rate is not None:
            factors.append(f"Recent {recent_n}-game hit rate {recent_rate:.2f}/game (28% blend)")
        factors.append(f"Projected hit rate {rate:.2f}/game vs {line + 0.5:g}+ threshold")
        name = "MLB season + recent hit-rate model"
        sample = games
    elif family == "Home Runs":
        games = int(_float(row.get("gamesPlayed")))
        homers = _float(row.get("homeRuns"))
        pa = _float(row.get("plateAppearances"))
        if games < 20 or pa < 50:
            return None
        season_rate = homers / max(1.0, games)
        recent_rate = None
        recent_n = 0
        if recent is not None:
            recent_n = int(_float(recent.get("gamesPlayed")))
            recent_hr = _float(recent.get("homeRuns"))
            if recent_n >= 5:
                recent_rate = recent_hr / recent_n
        rate = 0.78 * season_rate + 0.22 * recent_rate if recent_rate is not None else season_rate
        p = _poisson_tail(rate, line)
        confidence = clamp(
            0.40 + 0.25 * min(1.0, games / 100.0) + (0.04 if recent_rate is not None else 0.0),
            0.0, 0.72,
        )
        factors = [
            f"2026 MLB: {homers:.0f} HR in {games} games",
            f"HR/PA {homers / max(1.0, pa):.3f} over {pa:.0f} PA",
            f"Season HR rate {season_rate:.3f}/game",
        ]
        if recent_rate is not None:
            factors.append(f"Recent {recent_n}-game HR rate {recent_rate:.3f}/game (22% blend)")
        factors.append(f"Projected HR rate {rate:.3f}/game vs {line + 0.5:g}+ threshold")
        name = "MLB season + recent home-run model"
        sample = games
    elif family == "Strikeouts":
        starts = int(_float(row.get("gamesStarted")))
        strikeouts = _float(row.get("strikeOuts"))
        innings = str(row.get("inningsPitched") or "")
        if starts < 5:
            return None
        season_rate = strikeouts / max(1.0, starts)
        recent_rate = None
        recent_starts = 0
        if recent is not None:
            recent_starts = int(_float(recent.get("gamesStarted")))
            recent_ks = _float(recent.get("strikeOuts"))
            if recent_starts >= 2:
                recent_rate = recent_ks / recent_starts
        rate = 0.70 * season_rate + 0.30 * recent_rate if recent_rate is not None else season_rate
        p = _poisson_tail(rate, line)
        confidence = clamp(
            0.42 + 0.27 * min(1.0, starts / 24.0) + (0.05 if recent_rate is not None else 0.0),
            0.0, 0.78,
        )
        factors = [
            f"2026 MLB: {strikeouts:.0f} strikeouts across {starts} starts",
            f"Season K/start {season_rate:.2f}; innings pitched {innings or 'n/a'}",
        ]
        if recent_rate is not None:
            factors.append(f"Recent {recent_starts}-start K/start {recent_rate:.2f} (30% blend)")
        factors.append(f"Projected K/start {rate:.2f} vs {line + 0.5:g}+ threshold")
        name = "MLB season + recent starter strikeout model"
        sample = starts
    else:
        return None

    return ModelEvidence(
        sport="MLB",
        model_name=name,
        fair_probability=clamp(p, 0.01, 0.99),
        confidence=confidence,
        sample_size=sample,
        factors=tuple(factors),
    )


@lru_cache(maxsize=4)
def _nfl_index(year: int) -> dict[str, tuple[dict, ...]]:
    buckets: dict[str, list[dict]] = {}
    for row in nfl_weekly_rows(year):
        if str(row.get("season_type") or "") != "REG":
            continue
        name = normalize_person(row.get("player_display_name"))
        if name:
            buckets.setdefault(name, []).append(row)
    return {key: tuple(rows) for key, rows in buckets.items()}


def _nfl_player_rows(year: int, player: str) -> tuple[dict, ...]:
    index = _nfl_index(year)
    exact = normalize_person(player)
    if exact in index:
        return index[exact]
    # Conservative first+last reduction only when unique.
    tokens = exact.split()
    if len(tokens) >= 2:
        alias = f"{tokens[0]} {tokens[-1]}"
        hits = [rows for key, rows in index.items() if key == alias or (
            len(key.split()) >= 2 and f"{key.split()[0]} {key.split()[-1]}" == alias
        )]
        if len(hits) == 1:
            return hits[0]
    return ()


def nfl_prop_evidence(
    player: str,
    family: str,
    *,
    line: float,
    season: int = 2026,
) -> ModelEvidence | None:
    current_rows = list(_nfl_player_rows(season, player))
    prior_rows = list(_nfl_player_rows(season - 1, player))

    if family == "Passing Yards":
        getter = lambda r: _float(r.get("passing_yards"))
        count_model = False
    elif family == "Passing TDs":
        getter = lambda r: _float(r.get("passing_tds"))
        count_model = True
    elif family == "Rushing Yards":
        getter = lambda r: _float(r.get("rushing_yards"))
        count_model = False
    elif family == "Receiving Yards":
        getter = lambda r: _float(r.get("receiving_yards"))
        count_model = False
    elif family == "Receptions":
        getter = lambda r: _float(r.get("receptions"))
        count_model = False
    elif family == "Player Touchdowns":
        getter = lambda r: (
            _float(r.get("rushing_tds"))
            + _float(r.get("receiving_tds"))
            + _float(r.get("special_teams_tds"))
        )
        count_model = True
    else:
        return None

    current = [getter(r) for r in current_rows]
    prior = [getter(r) for r in prior_rows]
    projection = _empirical_projection(
        current,
        prior,
        line=line,
        count_model=count_model,
    )
    if projection is None:
        return None
    probability, confidence, sample, projected_mean, sd = projection
    if not current:
        confidence = min(confidence, 0.46)

    factors = (
        f"2026 NFL game sample {len(current)}; 2025 stabilizer sample {len(prior)}",
        f"Projected {family.lower()} mean {projected_mean:.2f}",
        f"Observed game-to-game SD {sd:.2f}; threshold {line + 0.5:g}+",
    )
    warnings = () if len(current) >= 2 else ("limited current-season sample",)
    return ModelEvidence(
        sport="NFL",
        model_name=f"NFL {family.lower()} game-log distribution",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=factors,
        warnings=warnings,
    )


@lru_cache(maxsize=2)
def _basketball_index(sport: str) -> dict[str, tuple[dict, ...]]:
    rows = nba_player_rows() if sport == "NBA" else wnba_player_rows()
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        if sport == "NBA":
            name = row.get("athlete_display_name")
            if str(row.get("did_not_play") or "").strip().lower() in {"true", "1"}:
                continue
        else:
            name = f"{row.get('first_name') or ''} {row.get('family_name') or ''}".strip()
            if not str(row.get("minutes") or "").strip():
                continue
        key = normalize_person(name)
        if key:
            buckets.setdefault(key, []).append(row)
    return {key: tuple(vals) for key, vals in buckets.items()}


def _basketball_player_rows(sport: str, player: str) -> tuple[dict, ...]:
    index = _basketball_index(sport)
    exact = normalize_person(player)
    if exact in index:
        return index[exact]
    tokens = exact.split()
    if len(tokens) >= 2:
        alias = f"{tokens[0]} {tokens[-1]}"
        hits = [
            vals for key, vals in index.items()
            if len(key.split()) >= 2 and f"{key.split()[0]} {key.split()[-1]}" == alias
        ]
        if len(hits) == 1:
            return hits[0]
    return ()


def basketball_prop_evidence(
    sport: str,
    player: str,
    family: str,
    *,
    line: float,
) -> ModelEvidence | None:
    if sport not in {"NBA", "WNBA"}:
        return None
    rows = list(_basketball_player_rows(sport, player))
    if len(rows) < 5:
        return None

    def get(row: dict) -> float:
        if sport == "NBA":
            if family == "Points":
                return _float(row.get("points"))
            if family == "Rebounds":
                return _float(row.get("rebounds"))
            if family == "Assists":
                return _float(row.get("assists"))
            if family == "Three-Pointers":
                return _float(row.get("three_point_field_goals_made"))
            if family == "Points + Rebounds + Assists":
                return _float(row.get("points")) + _float(row.get("rebounds")) + _float(row.get("assists"))
        else:
            if family == "Points":
                return _float(row.get("points"))
            if family == "Rebounds":
                return _float(row.get("rebounds_total"))
            if family == "Assists":
                return _float(row.get("assists"))
            if family == "Three-Pointers":
                return _float(row.get("three_pointers_made"))
            if family == "Points + Rebounds + Assists":
                return _float(row.get("points")) + _float(row.get("rebounds_total")) + _float(row.get("assists"))
        raise KeyError(family)

    try:
        values = [get(row) for row in rows]
    except KeyError:
        return None

    # WNBA 2026 is current-season evidence. NBA 2026 file is the completed
    # 2025-26 season and is a prior-season baseline for October 2026 games.
    current = values if sport == "WNBA" else []
    prior = [] if sport == "WNBA" else values
    projection = _empirical_projection(current, prior, line=line, count_model=False)
    if projection is None:
        return None
    probability, confidence, sample, projected_mean, sd = projection
    warnings: tuple[str, ...] = ()
    if sport == "NBA":
        confidence = min(confidence, 0.48)
        warnings = ("prior-season NBA baseline; current season has not established a sample",)

    return ModelEvidence(
        sport=sport,
        model_name=f"{sport} {family.lower()} game-log distribution",
        fair_probability=probability,
        confidence=confidence,
        sample_size=sample,
        factors=(
            f"{sample} player game(s) in public box-score sample",
            f"Projected {family.lower()} mean {projected_mean:.2f}",
            f"Game-to-game SD {sd:.2f}; threshold {line + 0.5:g}+",
        ),
        warnings=warnings,
    )


def player_prop_evidence(
    sport: str,
    player: str,
    family: str,
    *,
    line: float,
    season: int = 2026,
) -> ModelEvidence | None:
    if sport == "MLB":
        return mlb_prop_evidence(player, family, line=line, season=season)
    if sport == "NFL":
        return nfl_prop_evidence(player, family, line=line, season=season)
    if sport in {"NBA", "WNBA"}:
        return basketball_prop_evidence(sport, player, family, line=line)
    return None
