from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import exp, log
from statistics import mean

from sports_edge.core.math import clamp
from sports_edge.data.tennis_history import load_recent_tennis_rows
from sports_edge.models.game_scope import normalize
from sports_edge.models.model_evidence import ModelEvidence


@dataclass
class PlayerState:
    elo: float = 1500.0
    matches: int = 0
    recent_results: deque = None
    recent_dates: deque = None
    serve_points_won: deque = None
    return_points_won: deque = None
    minutes: deque = None
    surface_elo: dict = None
    level_elo: dict = None
    recent_levels: deque = None

    def __post_init__(self):
        if self.recent_results is None:
            self.recent_results = deque(maxlen=20)
        if self.recent_dates is None:
            self.recent_dates = deque(maxlen=20)
        if self.serve_points_won is None:
            self.serve_points_won = deque(maxlen=20)
        if self.return_points_won is None:
            self.return_points_won = deque(maxlen=20)
        if self.minutes is None:
            self.minutes = deque(maxlen=20)
        if self.surface_elo is None:
            self.surface_elo = {}
        if self.level_elo is None:
            self.level_elo = {}
        if self.recent_levels is None:
            self.recent_levels = deque(maxlen=12)


class TennisResearchModel:
    def __init__(
        self,
        gender: str,
        *,
        current_year: int | None = None,
        as_of: date | None = None,
    ):
        self.gender = gender
        year = current_year or (as_of.year if as_of is not None else datetime.now(timezone.utc).year)
        self.start_year = max(2000, year - 1)
        self.end_year = year
        # Match archives are date-granular. Excluding the entire as-of date is
        # deliberately conservative: without a trustworthy match timestamp we
        # cannot know which same-day results were available before prediction.
        self.as_of = as_of
        self.players: dict[str, PlayerState] = defaultdict(PlayerState)
        self._fit()

    @staticmethod
    def _float(row: dict, key: str) -> float | None:
        try:
            value = row.get(key)
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _date(row: dict) -> date | None:
        raw = str(row.get("tourney_date") or "")
        if len(raw) != 8 or not raw.isdigit():
            return None
        try:
            return datetime.strptime(raw, "%Y%m%d").date()
        except ValueError:
            return None

    @staticmethod
    def _name(value: str | None) -> str:
        return normalize(value)

    def _fit(self) -> None:
        rows = list(load_recent_tennis_rows(self.gender, self.start_year, self.end_year))
        rows.sort(key=lambda r: (str(r.get("tourney_date") or ""), str(r.get("match_num") or "")))

        for row in rows:
            match_date = self._date(row)
            if self.as_of is not None and (match_date is None or match_date >= self.as_of):
                continue

            winner = self._name(row.get("winner_name"))
            loser = self._name(row.get("loser_name"))
            if not winner or not loser:
                continue
            w = self.players[winner]
            l = self.players[loser]

            expected = 1.0 / (1.0 + 10 ** ((l.elo - w.elo) / 400.0))
            level = str(row.get("tourney_level") or "").upper()
            surface = str(row.get("surface") or "").strip().lower()
            k = 28.0 if level in {"G", "M", "A"} else (24.0 if level in {"C", "D"} else 20.0)
            delta = k * (1.0 - expected)
            w.elo += delta
            l.elo -= delta

            # Maintain chronological context ratings separately from overall Elo.
            # Sparse contexts start from the player's pre-match overall rating,
            # then update only on matches in that context.
            if surface:
                ws = w.surface_elo.get(surface, w.elo - delta)
                ls = l.surface_elo.get(surface, l.elo + delta)
                surface_expected = 1.0 / (1.0 + 10 ** ((ls - ws) / 400.0))
                surface_delta = 22.0 * (1.0 - surface_expected)
                w.surface_elo[surface] = ws + surface_delta
                l.surface_elo[surface] = ls - surface_delta
            if level:
                wl = w.level_elo.get(level, w.elo - delta)
                ll = l.level_elo.get(level, l.elo + delta)
                level_expected = 1.0 / (1.0 + 10 ** ((ll - wl) / 400.0))
                level_delta = 20.0 * (1.0 - level_expected)
                w.level_elo[level] = wl + level_delta
                l.level_elo[level] = ll - level_delta
                w.recent_levels.append(level)
                l.recent_levels.append(level)
            w.matches += 1
            l.matches += 1
            w.recent_results.append(1.0)
            l.recent_results.append(0.0)

            d = self._date(row)
            if d is not None:
                w.recent_dates.append(d)
                l.recent_dates.append(d)

            w_svpt = self._float(row, "w_svpt")
            w_1st = self._float(row, "w_1stWon")
            w_2nd = self._float(row, "w_2ndWon")
            l_svpt = self._float(row, "l_svpt")
            l_1st = self._float(row, "l_1stWon")
            l_2nd = self._float(row, "l_2ndWon")
            if w_svpt and w_svpt > 0 and w_1st is not None and w_2nd is not None:
                wsp = clamp((w_1st + w_2nd) / w_svpt, 0.0, 1.0)
                w.serve_points_won.append(wsp)
                l.return_points_won.append(1.0 - wsp)
            if l_svpt and l_svpt > 0 and l_1st is not None and l_2nd is not None:
                lsp = clamp((l_1st + l_2nd) / l_svpt, 0.0, 1.0)
                l.serve_points_won.append(lsp)
                w.return_points_won.append(1.0 - lsp)

            minutes = self._float(row, "minutes")
            if minutes and minutes > 0:
                w.minutes.append(minutes)
                l.minutes.append(minutes)

    @staticmethod
    def _elo_probability(a: float, b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))

    @staticmethod
    def _mean(values) -> float | None:
        vals = list(values)
        return mean(vals) if vals else None

    def _fatigue(self, state: PlayerState, today: date) -> tuple[int, float]:
        recent_matches = 0
        for d in state.recent_dates:
            try:
                if 0 <= (today - d).days <= 7:
                    recent_matches += 1
            except TypeError:
                continue
        recent_minutes = sum(list(state.minutes)[-5:]) if state.minutes else 0.0
        return recent_matches, recent_minutes

    def probability(
        self,
        player_a: str,
        player_b: str,
        *,
        level: str = "tour",
        surface: str | None = None,
        as_of: date | None = None,
    ) -> ModelEvidence | None:
        akey = self._name(player_a)
        bkey = self._name(player_b)
        if not akey or not bkey:
            return None
        a = self.players.get(akey)
        b = self.players.get(bkey)
        if a is None or b is None or min(a.matches, b.matches) < 3:
            return None

        p = self._elo_probability(a.elo, b.elo)
        factors = [
            f"Elo {player_a} {a.elo:.0f} vs {player_b} {b.elo:.0f}",
        ]

        surface_key = str(surface or "").strip().lower()
        if surface_key and surface_key in a.surface_elo and surface_key in b.surface_elo:
            surface_p = self._elo_probability(a.surface_elo[surface_key], b.surface_elo[surface_key])
            p = 0.62 * p + 0.38 * surface_p
            factors.append(
                f"{surface_key.title()} Elo {a.surface_elo[surface_key]:.0f} vs {b.surface_elo[surface_key]:.0f}"
            )

        level_code = {"atp tour": "A", "wta tour": "A", "challenger": "C", "itf": "F"}.get(level.lower())
        if level_code and level_code in a.level_elo and level_code in b.level_elo:
            level_p = self._elo_probability(a.level_elo[level_code], b.level_elo[level_code])
            p = 0.78 * p + 0.22 * level_p
            factors.append(
                f"{level}-level Elo {a.level_elo[level_code]:.0f} vs {b.level_elo[level_code]:.0f}"
            )

        form_a = self._mean(a.recent_results)
        form_b = self._mean(b.recent_results)
        if form_a is not None and form_b is not None:
            form_delta = clamp(form_a - form_b, -0.6, 0.6)
            z = log(p / (1.0 - p)) + 0.55 * form_delta
            p = 1.0 / (1.0 + exp(-z))
            factors.append(f"Recent-form delta {form_delta:+.2f}")

        serve_a = self._mean(a.serve_points_won)
        serve_b = self._mean(b.serve_points_won)
        ret_a = self._mean(a.return_points_won)
        ret_b = self._mean(b.return_points_won)
        stat_pairs = [x for x in (serve_a, serve_b, ret_a, ret_b) if x is not None]
        if len(stat_pairs) == 4:
            sr_delta = clamp((serve_a + ret_a) - (serve_b + ret_b), -0.20, 0.20)
            z = log(p / (1.0 - p)) + 2.2 * sr_delta
            p = 1.0 / (1.0 + exp(-z))
            factors.append(f"Serve/return composite delta {sr_delta:+.3f}")

        today = as_of or datetime.now(timezone.utc).date()
        matches_a, mins_a = self._fatigue(a, today)
        matches_b, mins_b = self._fatigue(b, today)
        fatigue_delta = (matches_b - matches_a) * 0.025 + clamp((mins_b - mins_a) / 900.0, -0.12, 0.12)
        if abs(fatigue_delta) > 1e-6:
            z = log(p / (1.0 - p)) + fatigue_delta
            p = 1.0 / (1.0 + exp(-z))
            factors.append(
                f"7-day workload {player_a} {matches_a} match(es) vs {player_b} {matches_b}"
            )

        level_key = level.lower()
        shrink = 1.0
        if "itf" in level_key:
            shrink = 0.72
        elif "challenger" in level_key or "125" in level_key:
            shrink = 0.84
        p = 0.5 + (p - 0.5) * shrink
        if shrink < 1.0:
            factors.append(f"{level} calibration shrink applied")

        sample = min(a.matches, b.matches)
        stat_depth = min(len(a.serve_points_won), len(b.serve_points_won), 12)
        confidence = clamp(
            0.40
            + 0.28 * clamp(sample / 20.0, 0.0, 1.0)
            + 0.14 * clamp(stat_depth / 10.0, 0.0, 1.0)
            + 0.08 * clamp(min(len(a.recent_results), len(b.recent_results)) / 10.0, 0.0, 1.0),
            0.0,
            0.88,
        )
        if "itf" in level_key:
            confidence = min(confidence, 0.68)
        elif "challenger" in level_key:
            confidence = min(confidence, 0.76)

        warnings: list[str] = []
        if stat_depth < 4:
            warnings.append("limited recent serve/return samples")
        if sample < 8:
            warnings.append("thin player-history sample")

        return ModelEvidence(
            sport="Tennis",
            model_name="Tennis Elo + form + serve/return + workload",
            fair_probability=clamp(p, 0.03, 0.97),
            confidence=confidence,
            sample_size=sample,
            factors=tuple(factors),
            warnings=tuple(warnings),
        )


def tennis_level_from_series(series_ticker: str | None) -> tuple[str, str]:
    series = str(series_ticker or "").upper()
    if series.startswith("KXITF"):
        return ("women" if "ITFW" in series else "men", "ITF")
    if "CHALLENGER" in series:
        return ("women" if series.startswith("KXWTA") else "men", "Challenger")
    if series.startswith("KXWTA"):
        return ("women", "WTA Tour")
    return ("men", "ATP Tour")
