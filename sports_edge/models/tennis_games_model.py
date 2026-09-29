from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import groupby
import math
import re
from statistics import mean
from typing import Iterable

from sports_edge.core.math import clamp
from sports_edge.data.tennis_history import load_recent_tennis_rows
from sports_edge.models.game_scope import normalize
from sports_edge.models.model_evidence import ModelEvidence


# Selected on the 2025 chronological holdout and re-checked on 2026.
# Shrinking toward 50% worsened Brier/log loss for men; women were nearly tied
# at 0.90 in 2025 and favored 1.00 again in 2026, so production stays unshrunk.
TENNIS_GAMES_TOTAL_CALIBRATION_ALPHA = 1.00


@dataclass(frozen=True)
class TennisGamesProjection:
    evidence: ModelEvidence
    market_key: str
    market_label: str
    game_title: str
    line: float
    projected_total: float
    projected_sd: float
    raw_probability: float


@dataclass(frozen=True)
class _ScoreOutcome:
    total_games: int
    winner_games: int
    loser_games: int
    sets_played: int


@dataclass(frozen=True)
class _HistoricalSample:
    match_date: date
    favorite_probability: float
    total_games: int
    surface: str
    level: str
    best_of: int


@dataclass
class _PlayerState:
    elo: float = 1500.0
    matches: int = 0
    recent_totals: deque[float] = field(default_factory=lambda: deque(maxlen=16))


@dataclass(frozen=True)
class _Snapshot:
    players: dict[str, _PlayerState]
    alias_index: dict[str, set[str]]
    samples: tuple[_HistoricalSample, ...]


_SET_RE = re.compile(r"^(\d+)-(\d+)(?:\(\d+\))?$")
_BAD_SCORE_TOKENS = ("RET", "W/O", "WALKOVER", "DEF", "ABD", "ABN", "DEFAULT")


def parse_completed_score(value: str | None) -> _ScoreOutcome | None:
    """Parse a conventional completed tennis score into game counts.

    Match-tiebreak formats such as 10-8 are rejected because different market
    rulebooks can count them differently. Retirements/walkovers also fail
    closed rather than contaminating a game-total distribution.
    """
    raw = str(value or "").strip().upper()
    if not raw or any(token in raw for token in _BAD_SCORE_TOKENS):
        return None

    winner_games = 0
    loser_games = 0
    sets_played = 0
    for token in raw.split():
        token = token.strip().strip(",;")
        match = _SET_RE.fullmatch(token)
        if not match:
            return None
        a, b = int(match.group(1)), int(match.group(2))
        if max(a, b) > 7:
            return None
        if a == b or max(a, b) < 6:
            return None
        winner_games += a
        loser_games += b
        sets_played += 1

    if sets_played < 2:
        return None
    return _ScoreOutcome(
        total_games=winner_games + loser_games,
        winner_games=winner_games,
        loser_games=loser_games,
        sets_played=sets_played,
    )


def _date(row: dict) -> date | None:
    raw = str(row.get("tourney_date") or "")
    if len(raw) != 8 or not raw.isdigit():
        return None
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def _best_of(row: dict, score: _ScoreOutcome | None = None) -> int:
    try:
        value = int(float(row.get("best_of")))
        if value in {3, 5}:
            return value
    except (TypeError, ValueError):
        pass
    if score is not None and score.sets_played > 3:
        return 5
    return 3


def _level_bucket(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"f", "i", "itf"} or "itf" in raw or "futures" in raw:
        return "itf"
    if raw == "c" or "challenger" in raw or "125" in raw:
        return "challenger"
    if raw == "g" or "slam" in raw or "grand slam" in raw:
        return "slam"
    return "tour"


def _elo_probability(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def _update_elo(winner: _PlayerState, loser: _PlayerState, *, level: str) -> None:
    expected = _elo_probability(winner.elo, loser.elo)
    k = 28.0 if level in {"tour", "slam"} else (24.0 if level == "challenger" else 20.0)
    delta = k * (1.0 - expected)
    winner.elo += delta
    loser.elo -= delta


def _alias_forms(value: str | None) -> tuple[str, ...]:
    q = normalize(value)
    if not q:
        return ()
    tokens = q.split()
    if tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv"}:
        tokens = tokens[:-1]
    if not tokens:
        return ()
    out = {" ".join(tokens)}
    if len(tokens) == 2:
        out.add(f"{tokens[1]} {tokens[0]}")
    elif len(tokens) >= 3:
        out.add(f"{tokens[0]} {tokens[-1]}")
        out.add(f"{tokens[-1]} {tokens[0]}")
    return tuple(sorted(out))


def _alias_index(players: dict[str, _PlayerState]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for canonical in players:
        for alias in _alias_forms(canonical):
            out.setdefault(alias, set()).add(canonical)
    return out


def _resolve_player(value: str | None, snapshot: _Snapshot) -> str | None:
    exact = normalize(value)
    if not exact:
        return None
    if exact in snapshot.players:
        return exact
    hits: set[str] = set()
    for alias in _alias_forms(value):
        hits.update(snapshot.alias_index.get(alias, set()))
    return next(iter(hits)) if len(hits) == 1 else None


def _level_weight(sample_level: str, target_level: str) -> float:
    if target_level in {"itf", "challenger"}:
        return 1.0 if sample_level == target_level else 0.0
    if sample_level == target_level:
        return 1.0
    if {sample_level, target_level} <= {"tour", "slam"}:
        return 0.82
    return 0.0


def _weighted_total_probabilities(
    *,
    samples: Iterable[_HistoricalSample],
    player_a: _PlayerState,
    player_b: _PlayerState,
    target_probability_a: float,
    target_date: date,
    target_level: str,
    target_surface: str | None,
    best_of: int,
    lines: Iterable[float],
) -> tuple[dict[float, float], float, float, int, float] | None:
    """Build one weighted empirical distribution and evaluate many total lines."""
    favorite_p = max(target_probability_a, 1.0 - target_probability_a)
    pool = list(samples)[-1800:]
    weighted: list[tuple[float, float]] = []

    surface = str(target_surface or "").strip().lower()
    for sample in pool:
        if sample.best_of != best_of:
            continue
        level_w = _level_weight(sample.level, target_level)
        if level_w <= 0:
            continue

        age_days = max(0, (target_date - sample.match_date).days)
        recency_w = 0.5 ** (age_days / 420.0)
        strength_w = math.exp(-abs(sample.favorite_probability - favorite_p) / 0.10)
        surface_w = 1.0
        if surface:
            surface_w = 1.25 if sample.surface == surface else 0.78
        weight = recency_w * strength_w * level_w * surface_w
        if weight > 0.005:
            weighted.append((weight, float(sample.total_games)))

    if len(weighted) < 30:
        return None

    sum_w = sum(w for w, _ in weighted)
    sum_w2 = sum(w * w for w, _ in weighted)
    if sum_w <= 0 or sum_w2 <= 0:
        return None

    sample_mu = sum(w * total for w, total in weighted) / sum_w
    a_recent = list(player_a.recent_totals)
    b_recent = list(player_b.recent_totals)
    projected_mu = sample_mu
    if len(a_recent) >= 4 and len(b_recent) >= 4:
        player_mu = 0.5 * (mean(a_recent[-10:]) + mean(b_recent[-10:]))
        delta = clamp(player_mu - sample_mu, -3.0, 3.0)
        projected_mu = sample_mu + 0.35 * delta

    shift = projected_mu - sample_mu
    line_values = tuple(float(line) for line in lines)
    probabilities = {
        line: clamp(
            sum(w for w, total in weighted if total + shift > line) / sum_w,
            0.01,
            0.99,
        )
        for line in line_values
    }
    variance = sum(
        w * ((total + shift) - projected_mu) ** 2
        for w, total in weighted
    ) / sum_w
    sd = max(2.0, math.sqrt(max(0.0, variance)))
    effective_n = (sum_w * sum_w) / sum_w2
    return probabilities, projected_mu, sd, len(weighted), effective_n


def _weighted_total_probability(
    *,
    samples: Iterable[_HistoricalSample],
    player_a: _PlayerState,
    player_b: _PlayerState,
    target_probability_a: float,
    target_date: date,
    target_level: str,
    target_surface: str | None,
    best_of: int,
    line: float,
) -> tuple[float, float, float, int, float] | None:
    result = _weighted_total_probabilities(
        samples=samples,
        player_a=player_a,
        player_b=player_b,
        target_probability_a=target_probability_a,
        target_date=target_date,
        target_level=target_level,
        target_surface=target_surface,
        best_of=best_of,
        lines=(float(line),),
    )
    if result is None:
        return None
    probabilities, projected_mu, sd, pool_n, effective_n = result
    return probabilities[float(line)], projected_mu, sd, pool_n, effective_n

def _calibrated_probability(raw: float, alpha: float) -> float:
    return clamp(0.5 + alpha * (raw - 0.5), 0.01, 0.99)


def _apply_row(
    row: dict,
    *,
    players: dict[str, _PlayerState],
    samples: list[_HistoricalSample],
) -> None:
    match_date = _date(row)
    winner_name = normalize(row.get("winner_name"))
    loser_name = normalize(row.get("loser_name"))
    if match_date is None or not winner_name or not loser_name or winner_name == loser_name:
        return

    score = parse_completed_score(row.get("score"))
    if score is None:
        return

    winner = players.setdefault(winner_name, _PlayerState())
    loser = players.setdefault(loser_name, _PlayerState())
    level = _level_bucket(row.get("tourney_level"))
    p_winner = _elo_probability(winner.elo, loser.elo)

    if min(winner.matches, loser.matches) >= 3:
        samples.append(
            _HistoricalSample(
                match_date=match_date,
                favorite_probability=max(p_winner, 1.0 - p_winner),
                total_games=score.total_games,
                surface=str(row.get("surface") or "").strip().lower(),
                level=level,
                best_of=_best_of(row, score),
            )
        )

    winner.recent_totals.append(float(score.total_games))
    loser.recent_totals.append(float(score.total_games))
    _update_elo(winner, loser, level=level)
    winner.matches += 1
    loser.matches += 1


class TennisGamesModel:
    """Independent pregame Tennis game-total model from completed match scores.

    No sportsbook probability or Kalshi price enters the fair-value model.
    Every target snapshot uses only matches with tourney_date < event_date.
    """

    def __init__(self, gender: str, *, current_year: int | None = None):
        if gender not in {"men", "women"}:
            raise ValueError("gender must be men or women")
        self.gender = gender
        year = current_year or datetime.utcnow().year
        self.start_year = max(2000, year - 2)
        self.end_year = year
        self.rows = tuple(
            sorted(
                load_recent_tennis_rows(gender, self.start_year, self.end_year),
                key=lambda row: (
                    str(row.get("tourney_date") or ""),
                    str(row.get("match_num") or ""),
                    str(row.get("winner_name") or ""),
                ),
            )
        )
        self._snapshots: dict[date, _Snapshot] = {}

    def _build_snapshot(self, as_of: date) -> _Snapshot:
        cached = self._snapshots.get(as_of)
        if cached is not None:
            return cached

        players: dict[str, _PlayerState] = {}
        samples: list[_HistoricalSample] = []
        for row in self.rows:
            match_date = _date(row)
            if match_date is None:
                continue
            if match_date >= as_of:
                break
            _apply_row(row, players=players, samples=samples)

        snapshot = _Snapshot(players=players, alias_index=_alias_index(players), samples=tuple(samples))
        self._snapshots[as_of] = snapshot
        return snapshot

    def project_games_total(
        self,
        player_a: str,
        player_b: str,
        *,
        line: float,
        event_date: date,
        level: str,
        surface: str | None = None,
        best_of: int = 3,
        game_title: str | None = None,
    ) -> TennisGamesProjection | None:
        if best_of not in {3, 5}:
            return None
        snapshot = self._build_snapshot(event_date)
        akey = _resolve_player(player_a, snapshot)
        bkey = _resolve_player(player_b, snapshot)
        if not akey or not bkey or akey == bkey:
            return None
        a = snapshot.players.get(akey)
        b = snapshot.players.get(bkey)
        if a is None or b is None or min(a.matches, b.matches) < 5:
            return None

        p_a = _elo_probability(a.elo, b.elo)
        level_key = _level_bucket(level)
        estimate = _weighted_total_probability(
            samples=snapshot.samples,
            player_a=a,
            player_b=b,
            target_probability_a=p_a,
            target_date=event_date,
            target_level=level_key,
            target_surface=surface,
            best_of=best_of,
            line=float(line),
        )
        if estimate is None:
            return None

        raw, projected_total, projected_sd, pool_n, effective_n = estimate
        fair = _calibrated_probability(raw, TENNIS_GAMES_TOTAL_CALIBRATION_ALPHA)
        player_depth = min(a.matches, b.matches)
        recent_depth = min(len(a.recent_totals), len(b.recent_totals))
        confidence = clamp(
            0.40
            + 0.13 * clamp(player_depth / 20.0, 0.0, 1.0)
            + 0.13 * clamp(effective_n / 120.0, 0.0, 1.0)
            + 0.08 * clamp(recent_depth / 10.0, 0.0, 1.0),
            0.0,
            0.72,
        )
        if level_key == "itf":
            confidence = min(confidence, 0.54)
        elif level_key == "challenger":
            confidence = min(confidence, 0.62)

        warnings: list[str] = []
        if not surface:
            warnings.append("surface not explicit in Kalshi metadata; distribution uses cross-surface history")
        if recent_depth < 6:
            warnings.append("limited recent completed-score depth for one or both players")
        if best_of == 5:
            warnings.append("best-of-five totals use a smaller historical pool and require explicit match-format confirmation")

        title = game_title or f"{player_a} vs {player_b}"
        evidence = ModelEvidence(
            sport="Tennis",
            model_name="Tennis Games Total: leakage-safe empirical score distribution",
            fair_probability=fair,
            confidence=confidence,
            sample_size=int(round(effective_n)),
            factors=(
                f"Pregame Elo matchup probability {player_a} {p_a:.1%}",
                f"Projected total {projected_total:.1f} games with empirical SD {projected_sd:.1f}",
                f"Comparable historical pool {pool_n} matches; effective weighted sample {effective_n:.1f}",
                f"Player completed-match depth {a.matches} / {b.matches}",
                f"Probability calibration {TENNIS_GAMES_TOTAL_CALIBRATION_ALPHA:.2f}× distance from 50% (raw {raw:.1%})",
            ),
            warnings=tuple(warnings),
        )
        if not evidence.usable:
            return None
        return TennisGamesProjection(
            evidence=evidence,
            market_key="tennis_games_total",
            market_label="Games Total",
            game_title=title,
            line=float(line),
            projected_total=projected_total,
            projected_sd=projected_sd,
            raw_probability=raw,
        )



TENNIS_GAMES_VALIDATION_LINES: tuple[float, ...] = (
    14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5,
    22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5,
)


def _eligible_validation_match(
    row: dict,
    score: _ScoreOutcome,
    *,
    best_of_filter: int | None,
) -> bool:
    return best_of_filter is None or _best_of(row, score) == best_of_filter


def walkforward_games_total_report(
    rows: Iterable[dict],
    *,
    holdout_year: int,
    alpha: float,
    lines: tuple[float, ...] = TENNIS_GAMES_VALIDATION_LINES,
    use_surface: bool = False,
    best_of_filter: int | None = 3,
) -> dict:
    """Prequential score for one calibration alpha using only prior-date history."""
    ordered = sorted(
        [row for row in rows if _date(row) is not None],
        key=lambda row: (
            str(row.get("tourney_date") or ""),
            str(row.get("match_num") or ""),
            str(row.get("winner_name") or ""),
        ),
    )
    players: dict[str, _PlayerState] = {}
    samples: list[_HistoricalSample] = []
    scored: list[tuple[float, int, float]] = []
    line_values = tuple(float(line) for line in lines)

    for day, day_iter in groupby(ordered, key=_date):
        day_rows = list(day_iter)
        if day is None:
            continue

        # Score every match on the date before adding any same-date results.
        if day.year == holdout_year:
            for row in day_rows:
                winner_name = normalize(row.get("winner_name"))
                loser_name = normalize(row.get("loser_name"))
                score = parse_completed_score(row.get("score"))
                if (
                    not winner_name
                    or not loser_name
                    or score is None
                    or not _eligible_validation_match(
                        row,
                        score,
                        best_of_filter=best_of_filter,
                    )
                ):
                    continue
                winner = players.get(winner_name)
                loser = players.get(loser_name)
                if winner is None or loser is None or min(winner.matches, loser.matches) < 5:
                    continue

                p_winner = _elo_probability(winner.elo, loser.elo)
                level = _level_bucket(row.get("tourney_level"))
                surface = (
                    str(row.get("surface") or "").strip().lower() or None
                    if use_surface else None
                )
                best_of = _best_of(row, score)
                estimate = _weighted_total_probabilities(
                    samples=samples,
                    player_a=winner,
                    player_b=loser,
                    target_probability_a=p_winner,
                    target_date=day,
                    target_level=level,
                    target_surface=surface,
                    best_of=best_of,
                    lines=line_values,
                )
                if estimate is None:
                    continue
                probabilities = estimate[0]
                for line in line_values:
                    probability = _calibrated_probability(probabilities[line], alpha)
                    scored.append((probability, int(score.total_games > line), line))

        for row in day_rows:
            _apply_row(row, players=players, samples=samples)

    if not scored:
        return {"n": 0, "brier": None, "log_loss": None, "accuracy": None, "by_line": {}}

    eps = 1e-12
    brier = sum((p - y) ** 2 for p, y, _ in scored) / len(scored)
    log_loss = -sum(
        y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1.0 - p))
        for p, y, _ in scored
    ) / len(scored)
    accuracy = sum(int((p >= 0.5) == bool(y)) for p, y, _ in scored) / len(scored)
    by_line: dict[str, dict] = {}
    for line in line_values:
        subset = [(p, y) for p, y, ln in scored if ln == line]
        if not subset:
            continue
        by_line[f"{line:g}"] = {
            "n": len(subset),
            "brier": sum((p - y) ** 2 for p, y in subset) / len(subset),
            "mean_p": sum(p for p, _ in subset) / len(subset),
            "hit_rate": sum(y for _, y in subset) / len(subset),
        }
    return {
        "n": len(scored),
        "brier": brier,
        "log_loss": log_loss,
        "accuracy": accuracy,
        "by_line": by_line,
    }


def walkforward_games_total_calibration_grid(
    rows: Iterable[dict],
    *,
    holdout_year: int,
    alphas: tuple[float, ...] = (0.60, 0.75, 0.90, 1.00),
    lines: tuple[float, ...] = TENNIS_GAMES_VALIDATION_LINES,
    use_surface: bool = False,
    best_of_filter: int | None = 3,
) -> dict[str, dict]:
    """Score several calibration alphas from one leakage-safe prequential pass."""
    ordered = sorted(
        [row for row in rows if _date(row) is not None],
        key=lambda row: (
            str(row.get("tourney_date") or ""),
            str(row.get("match_num") or ""),
            str(row.get("winner_name") or ""),
        ),
    )
    players: dict[str, _PlayerState] = {}
    samples: list[_HistoricalSample] = []
    raw_scores: list[tuple[float, int, float]] = []
    line_values = tuple(float(line) for line in lines)

    for day, day_iter in groupby(ordered, key=_date):
        day_rows = list(day_iter)
        if day is None:
            continue

        if day.year == holdout_year:
            for row in day_rows:
                winner_name = normalize(row.get("winner_name"))
                loser_name = normalize(row.get("loser_name"))
                score = parse_completed_score(row.get("score"))
                if (
                    not winner_name
                    or not loser_name
                    or score is None
                    or not _eligible_validation_match(
                        row,
                        score,
                        best_of_filter=best_of_filter,
                    )
                ):
                    continue
                winner = players.get(winner_name)
                loser = players.get(loser_name)
                if winner is None or loser is None or min(winner.matches, loser.matches) < 5:
                    continue

                p_winner = _elo_probability(winner.elo, loser.elo)
                level = _level_bucket(row.get("tourney_level"))
                surface = (
                    str(row.get("surface") or "").strip().lower() or None
                    if use_surface else None
                )
                best_of = _best_of(row, score)
                estimate = _weighted_total_probabilities(
                    samples=samples,
                    player_a=winner,
                    player_b=loser,
                    target_probability_a=p_winner,
                    target_date=day,
                    target_level=level,
                    target_surface=surface,
                    best_of=best_of,
                    lines=line_values,
                )
                if estimate is None:
                    continue
                probabilities = estimate[0]
                for line in line_values:
                    raw_scores.append(
                        (probabilities[line], int(score.total_games > line), line)
                    )

        for row in day_rows:
            _apply_row(row, players=players, samples=samples)

    reports: dict[str, dict] = {}
    eps = 1e-12
    for alpha in alphas:
        key = f"{float(alpha):.2f}"
        if not raw_scores:
            reports[key] = {"n": 0, "brier": None, "log_loss": None, "accuracy": None}
            continue
        scored = [
            (_calibrated_probability(raw, float(alpha)), outcome, line)
            for raw, outcome, line in raw_scores
        ]
        reports[key] = {
            "n": len(scored),
            "brier": sum((p - y) ** 2 for p, y, _ in scored) / len(scored),
            "log_loss": -sum(
                y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1.0 - p))
                for p, y, _ in scored
            ) / len(scored),
            "accuracy": sum(int((p >= 0.5) == bool(y)) for p, y, _ in scored) / len(scored),
        }
    return reports


def walkforward_games_total_benchmark(
    rows: Iterable[dict],
    *,
    holdout_year: int,
    lines: tuple[float, ...] = TENNIS_GAMES_VALIDATION_LINES,
    use_surface: bool = False,
    best_of_filter: int | None = 3,
) -> dict:
    """Compare the matchup model with a strength-blind chronological prior.

    The benchmark knows line, best-of format, level bucket and recency, but
    deliberately ignores player identity, Elo strength, surface and player
    recent-total history. Both probabilities are formed only from earlier dates.
    """
    ordered = sorted(
        [row for row in rows if _date(row) is not None],
        key=lambda row: (
            str(row.get("tourney_date") or ""),
            str(row.get("match_num") or ""),
            str(row.get("winner_name") or ""),
        ),
    )
    players: dict[str, _PlayerState] = {}
    samples: list[_HistoricalSample] = []
    scored: list[tuple[float, float, int, float]] = []
    line_values = tuple(float(line) for line in lines)

    for day, day_iter in groupby(ordered, key=_date):
        day_rows = list(day_iter)
        if day is None:
            continue

        if day.year == holdout_year:
            for row in day_rows:
                winner_name = normalize(row.get("winner_name"))
                loser_name = normalize(row.get("loser_name"))
                score = parse_completed_score(row.get("score"))
                if (
                    not winner_name
                    or not loser_name
                    or score is None
                    or not _eligible_validation_match(
                        row,
                        score,
                        best_of_filter=best_of_filter,
                    )
                ):
                    continue
                winner = players.get(winner_name)
                loser = players.get(loser_name)
                if winner is None or loser is None or min(winner.matches, loser.matches) < 5:
                    continue

                p_winner = _elo_probability(winner.elo, loser.elo)
                level = _level_bucket(row.get("tourney_level"))
                surface = (
                    str(row.get("surface") or "").strip().lower() or None
                    if use_surface else None
                )
                best_of = _best_of(row, score)

                # Build the strength-blind prior once per target match.
                baseline_weight = 0.0
                baseline_over = {line: 0.0 for line in line_values}
                baseline_n = 0
                for sample in samples[-1800:]:
                    if sample.best_of != best_of:
                        continue
                    level_w = _level_weight(sample.level, level)
                    if level_w <= 0:
                        continue
                    age_days = max(0, (day - sample.match_date).days)
                    weight = (0.5 ** (age_days / 420.0)) * level_w
                    if weight <= 0.005:
                        continue
                    baseline_weight += weight
                    baseline_n += 1
                    for line in line_values:
                        baseline_over[line] += weight * float(sample.total_games > line)

                if baseline_n < 30 or baseline_weight <= 0:
                    continue

                estimate = _weighted_total_probabilities(
                    samples=samples,
                    player_a=winner,
                    player_b=loser,
                    target_probability_a=p_winner,
                    target_date=day,
                    target_level=level,
                    target_surface=surface,
                    best_of=best_of,
                    lines=line_values,
                )
                if estimate is None:
                    continue

                probabilities = estimate[0]
                for line in line_values:
                    model_p = _calibrated_probability(
                        probabilities[line],
                        TENNIS_GAMES_TOTAL_CALIBRATION_ALPHA,
                    )
                    baseline_p = clamp(
                        baseline_over[line] / baseline_weight,
                        0.01,
                        0.99,
                    )
                    scored.append(
                        (model_p, baseline_p, int(score.total_games > line), line)
                    )

        for row in day_rows:
            _apply_row(row, players=players, samples=samples)

    if not scored:
        return {"n": 0, "by_line": {}}

    eps = 1e-12

    def metrics(values: list[tuple[float, int]]) -> dict[str, float]:
        return {
            "brier": sum((p - y) ** 2 for p, y in values) / len(values),
            "log_loss": -sum(
                y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1.0 - p))
                for p, y in values
            ) / len(values),
            "accuracy": sum(int((p >= 0.5) == bool(y)) for p, y in values) / len(values),
        }

    model_metrics = metrics([(model_p, outcome) for model_p, _, outcome, _ in scored])
    baseline_metrics = metrics([(base_p, outcome) for _, base_p, outcome, _ in scored])
    by_line: dict[str, dict] = {}
    for line in line_values:
        subset = [row for row in scored if row[3] == line]
        if not subset:
            continue
        model_line = metrics([(row[0], row[2]) for row in subset])
        baseline_line = metrics([(row[1], row[2]) for row in subset])
        by_line[f"{line:g}"] = {
            "n": len(subset),
            "model": model_line,
            "baseline": baseline_line,
            "brier_improvement": baseline_line["brier"] - model_line["brier"],
            "log_loss_improvement": baseline_line["log_loss"] - model_line["log_loss"],
        }

    return {
        "n": len(scored),
        "model": model_metrics,
        "baseline": baseline_metrics,
        "brier_improvement": baseline_metrics["brier"] - model_metrics["brier"],
        "log_loss_improvement": baseline_metrics["log_loss"] - model_metrics["log_loss"],
        "by_line": by_line,
    }
