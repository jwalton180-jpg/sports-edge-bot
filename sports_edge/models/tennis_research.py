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


@dataclass(frozen=True)
class HeadToHeadMeeting:
    match_date: date
    winner: str
    loser: str
    surface: str
    level: str


@dataclass
class PlayerState:
    elo: float = 1500.0
    matches: int = 0
    recent_results: deque = None
    recent_dates: deque = None
    serve_points_won: deque = None
    return_points_won: deque = None
    minutes: deque = None

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


class TennisResearchModel:
    NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

    def __init__(self, gender: str, *, current_year: int | None = None):
        self.gender = gender
        year = current_year or datetime.now(timezone.utc).year
        self.start_year = max(2000, year - 1)
        self.end_year = year
        self.players: dict[str, PlayerState] = defaultdict(PlayerState)
        self.alias_index: dict[str, set[str]] = {}
        self.head_to_head: dict[tuple[str, str], list[HeadToHeadMeeting]] = defaultdict(list)
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

    @classmethod
    def _alias_forms(cls, value: str | None) -> tuple[str, ...]:
        """Conservative player-name aliases; ambiguous aliases never resolve."""
        normalized = normalize(value)
        if not normalized:
            return ()
        tokens = normalized.split()
        if tokens and tokens[-1] in cls.NAME_SUFFIXES:
            tokens = tokens[:-1]
        if not tokens:
            return ()

        forms = {" ".join(tokens)}
        if len(tokens) == 2:
            forms.add(f"{tokens[1]} {tokens[0]}")
        elif len(tokens) >= 3:
            # Public feeds often omit middle names/initials. Use only the
            # first+last reduction and its reverse; resolution must be unique.
            forms.add(f"{tokens[0]} {tokens[-1]}")
            forms.add(f"{tokens[-1]} {tokens[0]}")
        return tuple(sorted(forms))

    def _rebuild_alias_index(self) -> None:
        index: dict[str, set[str]] = {}
        for canonical in self.players:
            for alias in self._alias_forms(canonical):
                index.setdefault(alias, set()).add(canonical)
        self.alias_index = index

    def _resolve_name(self, value: str | None) -> str | None:
        exact = self._name(value)
        if not exact:
            return None
        if exact in self.players:
            return exact

        hits: set[str] = set()
        for alias in self._alias_forms(value):
            hits.update(self.alias_index.get(alias, set()))
        return next(iter(hits)) if len(hits) == 1 else None

    def _fit(self) -> None:
        rows = list(load_recent_tennis_rows(self.gender, self.start_year, self.end_year))
        rows.sort(key=lambda r: (str(r.get("tourney_date") or ""), str(r.get("match_num") or "")))

        for row in rows:
            winner = self._name(row.get("winner_name"))
            loser = self._name(row.get("loser_name"))
            if not winner or not loser:
                continue
            w = self.players[winner]
            l = self.players[loser]

            expected = 1.0 / (1.0 + 10 ** ((l.elo - w.elo) / 400.0))
            level = str(row.get("tourney_level") or "").upper()
            k = 28.0 if level in {"G", "M", "A"} else (24.0 if level in {"C", "D"} else 20.0)
            delta = k * (1.0 - expected)
            w.elo += delta
            l.elo -= delta
            w.matches += 1
            l.matches += 1
            w.recent_results.append(1.0)
            l.recent_results.append(0.0)

            d = self._date(row)
            if d is not None:
                w.recent_dates.append(d)
                l.recent_dates.append(d)
                pair = tuple(sorted((winner, loser)))
                self.head_to_head[pair].append(
                    HeadToHeadMeeting(
                        match_date=d,
                        winner=winner,
                        loser=loser,
                        surface=str(row.get("surface") or "").strip().lower(),
                        level=level,
                    )
                )

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

        self._rebuild_alias_index()

    @staticmethod
    def _elo_probability(a: float, b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))

    @staticmethod
    def _mean(values) -> float | None:
        vals = list(values)
        return mean(vals) if vals else None

    @staticmethod
    def _recent_mean(values, n: int) -> float | None:
        vals = list(values)
        if not vals:
            return None
        tail = vals[-n:]
        return mean(tail) if tail else None

    def _trajectory_adjustment(self, a: PlayerState, b: PlayerState) -> tuple[float, str | None]:
        short_a = self._recent_mean(a.recent_results, 5)
        short_b = self._recent_mean(b.recent_results, 5)
        long_a = self._mean(a.recent_results)
        long_b = self._mean(b.recent_results)
        if None in (short_a, short_b, long_a, long_b):
            return 0.0, None
        delta = clamp((short_a - long_a) - (short_b - long_b), -0.60, 0.60)
        return 0.28 * delta, f"Form trajectory delta {delta:+.2f} (last 5 vs longer baseline)"

    def _serve_return_trend_adjustment(self, a: PlayerState, b: PlayerState) -> tuple[float, str | None]:
        a_short_s = self._recent_mean(a.serve_points_won, 5)
        a_short_r = self._recent_mean(a.return_points_won, 5)
        b_short_s = self._recent_mean(b.serve_points_won, 5)
        b_short_r = self._recent_mean(b.return_points_won, 5)
        a_long_s = self._mean(a.serve_points_won)
        a_long_r = self._mean(a.return_points_won)
        b_long_s = self._mean(b.serve_points_won)
        b_long_r = self._mean(b.return_points_won)
        vals = (a_short_s, a_short_r, b_short_s, b_short_r, a_long_s, a_long_r, b_long_s, b_long_r)
        if any(v is None for v in vals):
            return 0.0, None
        a_trend = (a_short_s + a_short_r) - (a_long_s + a_long_r)
        b_trend = (b_short_s + b_short_r) - (b_long_s + b_long_r)
        delta = clamp(a_trend - b_trend, -0.12, 0.12)
        return 1.25 * delta, f"Recent serve/return trend delta {delta:+.3f}"

    def _h2h_adjustment(
        self,
        akey: str,
        bkey: str,
        *,
        today: date,
        level: str,
        surface: str | None,
    ) -> tuple[float, tuple[str, ...], int]:
        store = getattr(self, "head_to_head", {}) or {}
        meetings = list(store.get(tuple(sorted((akey, bkey))), ()))
        eligible = [m for m in meetings if m.match_date < today]
        if not eligible:
            return 0.0, (), 0

        weighted_a = 0.0
        weighted_b = 0.0
        same_surface = 0
        surface_key = str(surface or "").strip().lower()
        level_key = str(level or "").strip().lower()

        for meeting in eligible:
            age_days = max(0, (today - meeting.match_date).days)
            # Two-year half-life: old meetings retain context but decay.
            weight = 0.5 ** (age_days / 730.0)
            if surface_key and meeting.surface and meeting.surface == surface_key:
                weight *= 1.25
                same_surface += 1
            if level_key and meeting.level:
                meeting_level = meeting.level.lower()
                if (
                    ("challenger" in level_key and meeting_level in {"c", "d"})
                    or ("itf" in level_key and meeting_level in {"f", "i"})
                    or ("tour" in level_key and meeting_level in {"g", "m", "a"})
                ):
                    weight *= 1.10

            if meeting.winner == akey:
                weighted_a += weight
            elif meeting.winner == bkey:
                weighted_b += weight

        total_weight = weighted_a + weighted_b
        if total_weight <= 0:
            return 0.0, (), 0

        # Beta(2,2) prior aggressively shrinks tiny H2H samples toward 50%.
        posterior_a = (2.0 + weighted_a) / (4.0 + total_weight)
        strength = total_weight / (total_weight + 3.0)
        centered = posterior_a - 0.5
        adjustment = clamp(1.6 * centered * strength, -0.28, 0.28)

        a_wins = sum(1 for m in eligible if m.winner == akey)
        b_wins = sum(1 for m in eligible if m.winner == bkey)
        factors = [
            f"Head-to-head before match: {a_wins}-{b_wins}; recency-weighted share {posterior_a:.1%}",
            f"H2H logit adjustment {adjustment:+.3f} after small-sample shrinkage",
        ]
        if same_surface:
            factors.append(f"{same_surface} prior H2H meeting(s) on the same surface received extra weight")
        return adjustment, tuple(factors), len(eligible)

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
        as_of: date | None = None,
        surface: str | None = None,
    ) -> ModelEvidence | None:
        a_input = self._name(player_a)
        b_input = self._name(player_b)
        akey = self._resolve_name(player_a)
        bkey = self._resolve_name(player_b)
        if not akey or not bkey or akey == bkey:
            return None
        a = self.players.get(akey)
        b = self.players.get(bkey)
        if a is None or b is None or min(a.matches, b.matches) < 3:
            return None

        base_elo = self._elo_probability(a.elo, b.elo)
        p = base_elo
        factors = [
            f"Elo {player_a} {a.elo:.0f} vs {player_b} {b.elo:.0f}",
        ]
        if a_input != akey:
            factors.append(f"Unique historical-name alias resolved for {player_a}")
        if b_input != bkey:
            factors.append(f"Unique historical-name alias resolved for {player_b}")

        form_a = self._mean(a.recent_results)
        form_b = self._mean(b.recent_results)
        if form_a is not None and form_b is not None:
            form_delta = clamp(form_a - form_b, -0.6, 0.6)
            z = log(p / (1.0 - p)) + 0.55 * form_delta
            p = 1.0 / (1.0 + exp(-z))
            factors.append(f"Recent-form delta {form_delta:+.2f}")

        trajectory_adj, trajectory_factor = self._trajectory_adjustment(a, b)
        if trajectory_factor:
            z = log(p / (1.0 - p)) + trajectory_adj
            p = 1.0 / (1.0 + exp(-z))
            factors.append(trajectory_factor)

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

        sr_trend_adj, sr_trend_factor = self._serve_return_trend_adjustment(a, b)
        if sr_trend_factor:
            z = log(p / (1.0 - p)) + sr_trend_adj
            p = 1.0 / (1.0 + exp(-z))
            factors.append(sr_trend_factor)

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
        if "itf" in level_key:
            # Chronological 2026 holdout: supplemental form/serve/workload
            # adjustments did not improve ITF probability quality over Elo.
            p = base_elo
            factors.append(
                "ITF calibration: fair probability uses Elo-only; form/serve/workload remain diagnostics"
            )
        elif "challenger" in level_key or "125" in level_key:
            # Pooled Challenger holdout slightly favored a conservative blend
            # over either the enhanced model or Elo alone.
            p = 0.5 * p + 0.5 * base_elo
            factors.append(
                "Challenger calibration: 50/50 enhanced-model and Elo blend"
            )

        h2h_adj, h2h_factors, h2h_count = self._h2h_adjustment(
            akey,
            bkey,
            today=today,
            level=level,
            surface=surface,
        )
        if h2h_count:
            z = log(p / (1.0 - p)) + h2h_adj
            p = 1.0 / (1.0 + exp(-z))
            factors.extend(h2h_factors)

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
        if h2h_count >= 3:
            confidence = min(0.90, confidence + min(0.04, 0.01 * h2h_count))
        if "itf" in level_key:
            confidence = min(confidence, 0.55)
        elif "challenger" in level_key or "125" in level_key:
            confidence = min(confidence, 0.65)

        warnings: list[str] = []
        if stat_depth < 4:
            warnings.append("limited recent serve/return samples")
        if sample < 8:
            warnings.append("thin player-history sample")
        if h2h_count == 1:
            warnings.append("single prior H2H meeting is heavily shrunk and should not dominate the call")

        return ModelEvidence(
            sport="Tennis",
            model_name="Tennis Elo + form trajectory + serve/return trend + workload + H2H",
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
