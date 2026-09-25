from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable
from collections import defaultdict, deque
from .tennis import EloState, matchup_probability

LEVEL_WEIGHT = {"itf": .72, "challenger": .86, "atp": 1.0, "wta": 1.0, "slam": 1.08, "bkc": .96, "davis": .96}

@dataclass(frozen=True)
class TennisMatch:
    played_at: datetime
    winner: str
    loser: str
    surface: str
    level: str = "atp"
    winner_serve_pts_won: float | None = None
    loser_serve_pts_won: float | None = None
    retired: bool = False

@dataclass(frozen=True)
class TennisPrediction:
    played_at: datetime
    player_a: str
    player_b: str
    probability_a: float
    outcome_a: int
    level: str
    surface: str
    prior_form_delta: float = 0.0
    prior_form_samples_a: int = 0
    prior_form_samples_b: int = 0
    prior_form_age_days_a: float | None = None
    prior_form_age_days_b: float | None = None


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fatigue(last_played: datetime | None, now: datetime) -> float:
    """Small conservative rest adjustment; no boost beyond normal rest."""
    if last_played is None: return 0.0
    days = max(0.0, (_utc(now) - _utc(last_played)).total_seconds() / 86400)
    if days < 1: return -.10
    if days < 2: return -.05
    return 0.0


class _RollingForm:
    """Strictly lagged, time-decayed and opponent-quality-aware form store."""
    def __init__(self, window: int = 12, half_life_days: float = 60.0):
        self.window = window
        self.half_life_days = half_life_days
        self.overall = defaultdict(lambda: deque(maxlen=window))
        self.surface = defaultdict(lambda: deque(maxlen=window))

    def add(self, player: str, surface: str, serve: float, ret: float,
            played_at: datetime, opponent_rating: float, base_rating: float = 1500.0) -> None:
        if not (0.0 <= serve <= 1.0 and 0.0 <= ret <= 1.0):
            return
        quality = max(.75, min(1.25, 1.0 + (opponent_rating-base_rating)/800.0))
        obs=(serve, ret, _utc(played_at), quality)
        self.overall[player].append(obs)
        self.surface[(player, surface.lower())].append(obs)

    def _weighted_mean(self, xs, now: datetime) -> tuple[float, float | None]:
        vals=list(xs)
        if not vals: return 0.0, None
        num=den=0.0; ages=[]
        for sv,rt,ts,quality in vals:
            age=max(0.0, (_utc(now)-ts).total_seconds()/86400.0)
            decay=2 ** (-age/self.half_life_days)
            w=decay*quality
            num += (((sv-.5)+(rt-.5))/2)*w; den += 1.0; ages.append(age)
        return (num/den if den else 0.0), min(ages) if ages else None

    def signal(self, player: str, surface: str, now: datetime) -> tuple[float,int,float | None]:
        surf=list(self.surface.get((player, surface.lower()), ()))
        overall=list(self.overall.get(player, ()))
        if not overall: return 0.0, 0, None
        ov,ov_age=self._weighted_mean(overall, now)
        if not surf: return ov, len(overall), ov_age
        sf,sf_age=self._weighted_mean(surf, now)
        w=min(1.0, len(surf)/4.0)
        return w*sf+(1-w)*ov, len(surf), sf_age


def chronological_predictions(matches: Iterable[TennisMatch], base: float = 1500.0) -> list[TennisPrediction]:
    state = EloState(base=base)
    last: dict[str, datetime] = {}
    form = _RollingForm()
    out: list[TennisPrediction] = []
    ordered = sorted(matches, key=lambda m: (_utc(m.played_at), min(m.winner, m.loser), max(m.winner, m.loser)))
    seen_at: dict[datetime, set[str]] = {}
    for m in ordered:
        ts = _utc(m.played_at)
        players = seen_at.setdefault(ts, set())
        if m.winner in players or m.loser in players:
            raise ValueError(f"ambiguous same-timestamp tennis history at {ts.isoformat()}")
        players.update((m.winner, m.loser))
        player_a, player_b = sorted((m.winner, m.loser))
        ra = state.rating(player_a, m.surface); rb = state.rating(player_b, m.surface)
        fatigue = _fatigue(last.get(player_a), m.played_at) - _fatigue(last.get(player_b), m.played_at)
        fa, na, age_a = form.signal(player_a, m.surface, m.played_at); fb, nb, age_b = form.signal(player_b, m.surface, m.played_at)
        form_delta = max(-.18, min(.18, fa-fb))
        p = matchup_probability(ra, rb, serve_return_delta=form_delta, fatigue_delta=fatigue)
        outcome_a = int(player_a == m.winner)
        out.append(TennisPrediction(ts, player_a, player_b, p, outcome_a, m.level.lower(), m.surface.lower(), form_delta, na, nb, age_a, age_b))
        if not m.retired:
            pre_w = state.rating(m.winner, m.surface); pre_l = state.rating(m.loser, m.surface)
            if m.winner_serve_pts_won is not None and m.loser_serve_pts_won is not None:
                form.add(m.winner, m.surface, m.winner_serve_pts_won, 1-m.loser_serve_pts_won, ts, pre_l, base)
                form.add(m.loser, m.surface, m.loser_serve_pts_won, 1-m.winner_serve_pts_won, ts, pre_w, base)
            state.update(m.winner, m.loser, m.surface, LEVEL_WEIGHT.get(m.level.lower(), .82))
        last[m.winner] = ts; last[m.loser] = ts
    return out


def calibration_report(predictions: Iterable[TennisPrediction], bins: int = 10) -> dict:
    ps = list(predictions)
    if not ps: return {"n": 0, "brier": None, "log_loss": None, "bins": []}
    eps = 1e-12
    brier = sum((p.probability_a-p.outcome_a)**2 for p in ps)/len(ps)
    ll = -sum(p.outcome_a*math.log(max(eps,p.probability_a))+(1-p.outcome_a)*math.log(max(eps,1-p.probability_a)) for p in ps)/len(ps)
    rows=[]
    for i in range(bins):
        lo=i/bins; hi=(i+1)/bins
        xs=[p for p in ps if lo <= p.probability_a < hi or (i==bins-1 and p.probability_a==1)]
        if xs: rows.append({"lo":lo,"hi":hi,"n":len(xs),"mean_p":sum(x.probability_a for x in xs)/len(xs),"hit_rate":sum(x.outcome_a for x in xs)/len(xs)})
    return {"n":len(ps),"brier":brier,"log_loss":ll,"bins":rows}


def split_report(predictions: Iterable[TennisPrediction]) -> dict[str, dict]:
    groups: dict[str,list[TennisPrediction]]={}
    for p in predictions: groups.setdefault(p.level,[]).append(p)
    return {k: calibration_report(v) for k,v in groups.items()}


def underdog_band_report(predictions: Iterable[TennisPrediction], low: float = 0.03, high: float = 0.30, min_n: int = 30) -> dict:
    if not (0.0 <= low < high <= 0.5):
        raise ValueError("underdog band must satisfy 0 <= low < high <= 0.5")
    rows=[]
    for p in predictions:
        q = min(p.probability_a, 1.0-p.probability_a)
        if low <= q <= high:
            y = p.outcome_a if p.probability_a <= 0.5 else 1-p.outcome_a
            rows.append((q, int(y), p.level))
    if not rows:
        return {"n":0,"mean_p":None,"hit_rate":None,"brier":None,"calibration_gap":None,
                "levels":{},"sample_floor":min_n,"validated":False}
    n=len(rows); mean_p=sum(q for q,_,_ in rows)/n; hit=sum(y for _,y,_ in rows)/n
    brier=sum((q-y)**2 for q,y,_ in rows)/n
    levels={}
    for _,_,level in rows: levels[level]=levels.get(level,0)+1
    return {"n":n,"mean_p":mean_p,"hit_rate":hit,"brier":brier,
            "calibration_gap":hit-mean_p,"levels":levels,
            "sample_floor":min_n,"validated":n>=min_n}
