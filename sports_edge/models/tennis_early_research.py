"""Independent, research-only early Tennis reversal evidence.

Never modifies the qualified DEEP REVERSAL lane. Point-level inference is
structural, not calibrated; it is never an actionable confirmation on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from math import isfinite

from sports_edge.models.tennis_live_probability import (
    _hold_probabilities,
    _skill_from_pregame_probability,
    _match_probability_for_skill,
    estimate_live_match_probability,
)
from sports_edge.models.tennis_live_reversal import executable_path, _drawdown
from sports_edge.data.tennis_live import TennisLiveScoreState
from sports_edge.models.parlay_candidates import ParlayCandidateLeg
from sports_edge.models.event_identity import canonical_participant

POINTS = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4, "ADV": 4}


@lru_cache(maxsize=512)
def _point_win_rate_from_hold(hold: float) -> float:
    """Inverts i.i.d. serve-point scoring to an exact tennis game hold chance."""
    def game(p: float) -> float:
        q = 1 - p
        # Win before deuce plus reach deuce then win from deuce.
        win_predeuce = p**4 * (1 + 4*q + 10*q*q)
        reach_deuce = 20 * p**3 * q**3
        deuce_win = p*p/(p*p + q*q)
        return win_predeuce + reach_deuce * deuce_win
    lo, hi = 0.01, 0.99
    for _ in range(38):
        mid = (lo + hi) / 2
        if game(mid) < hold:
            lo = mid
        else:
            hi = mid
    return (lo + hi)/2


def _game_win_probability(server_point_win: float, server_points: int, returner_points: int) -> float:
    """Server wins current game from observed points; advantage/deuce exact."""
    p = server_point_win
    q = 1-p
    @lru_cache(maxsize=None)
    def rec(a: int,b: int) -> float:
        if a >= 4 and a-b >= 2:
            return 1.
        if b >= 4 and b-a >= 2:
            return 0.
        if a >= 3 and b >= 3:
            if a==b:
                return p*p/(p*p+q*q)
            deuce = p*p/(p*p+q*q)
            if a>b:
                return p + q*deuce
            return p*deuce
        return p*rec(a+1,b)+q*rec(a,b+1)
    return rec(server_points,returner_points)


def parse_point_score(raw: str | None) -> tuple[int,int] | None:
    if not raw:
        return None
    parts=str(raw).upper().replace("–","-").split("-")
    if len(parts)!=2:
        return None
    a,b=(POINTS.get(x.strip()) for x in parts)
    if a is None or b is None:
        return None
    # 40-40 is deuce, 40-AD is receiver advantage, etc.
    if a==4 and b==4:
        return None
    return a,b


def point_aware_live_probability(prior: float, state: TennisLiveScoreState | None) -> float | None:
    """Replace only the current *game* transition with point-conditional odds.

    No point-state inference for a tiebreak, ambiguous score or unknown server.
    Never uses Kalshi prices or hindsight. Structural research estimate only.
    """
    if (state is None or state.at_tiebreak or state.score_conflict
            or not isinstance(state.serving,bool)):
        return None
    points=parse_point_score(state.point_score)
    if points is None:
        return None
    baseline=estimate_live_match_probability(prior,state)
    if baseline is None:
        return None
    best_of=state.best_of
    skill=_skill_from_pregame_probability(round(max(.01,min(.99,float(prior))),4),best_of)
    hold_a,hold_b=_hold_probabilities(skill)
    server_point=_point_win_rate_from_hold(hold_a if state.serving else hold_b)
    sp,rp=points if state.serving else tuple(reversed(points))
    current_game_server_win=_game_win_probability(server_point,sp,rp)
    p_a_current_game=current_game_server_win if state.serving else 1-current_game_server_win
    next_server=not state.serving
    common=dict(best_of=best_of,sets_a=state.player_sets,sets_b=state.opponent_sets,server_a=next_server)
    if state.player_games is None or state.opponent_games is None:
        return None
    if state.player_games>=7 or state.opponent_games>=7:
        return None
    a_win=_match_probability_for_skill(skill,games_a=state.player_games+1,games_b=state.opponent_games,**common)
    b_win=_match_probability_for_skill(skill,games_a=state.player_games,games_b=state.opponent_games+1,**common)
    return max(.001,min(.999,p_a_current_game*a_win+(1-p_a_current_game)*b_win))


@dataclass(frozen=True)
class EarlyReversalWatch:
    event_id: str
    ticker: str
    selection: str
    price: float
    fair: float
    edge_pp: float
    trough: float
    rebound_pp: float
    score: str
    point_aware: bool
    quote_spread_pp: float
    fetched_at: datetime
    status: str = "EARLY WATCH — RESEARCH"
    model_version: str = "tennis-early-research-v1"


def build_early_reversal_watches(
    candidates: list[ParlayCandidateLeg],
    candles: dict[str,list[dict]],
    live_states: dict[tuple[str,str],TennisLiveScoreState],
    *,
    now: datetime | None = None,
) -> tuple[EarlyReversalWatch,...]:
    now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    output=[]
    for leg in candidates:
        if leg.sport!="Tennis" or leg.market_key!="model_h2h" or leg.model_probability is None:
            continue
        if not leg.kalshi_ticker or not leg.kalshi_side or (leg.model_sample_size or 0)<6:
            continue
        state=_resolve_live_state(leg, live_states)
        if state is None or state.score_conflict or not state.score_sources:
            continue
        if abs((now-state.fetched_at).total_seconds())>150:
            continue
        path=executable_path(candles.get(str(leg.kalshi_ticker),()),leg.kalshi_side)
        if len(path)<4:
            continue
        last=path[-1]
        if not (last.quoted and last.spread is not None and 0<=(now.timestamp()-last.end_ts)<=150):
            continue
        if not (.02<=last.close<=.20) or last.spread>min(.08,max(.04,last.close*.45)):
            continue
        peak,trough,idx,dd=_drawdown(path)
        if dd<.08 or trough>=last.close or idx>=len(path)-1:
            continue
        if last.close-trough<.005:
            continue
        point_fair=point_aware_live_probability(float(leg.model_probability),state)
        structural=estimate_live_match_probability(float(leg.model_probability),state)
        if structural is None:
            continue
        fair=point_fair if point_fair is not None else structural.probability
        edge=(fair-last.close)*100
        if edge<max(6.,last.close*30):
            continue
        if (leg.model_confidence or 0)<.45:
            continue
        # No early watch when a known structural break strongly contradicts.
        if state.net_break_advantage is not None and state.net_break_advantage<=-1:
            continue
        output.append(EarlyReversalWatch(
            event_id=leg.event_id,ticker=str(leg.kalshi_ticker),selection=leg.selection,
            price=last.close,fair=fair,edge_pp=edge,trough=trough,
            rebound_pp=(last.close-trough)*100,score=state.score_label,
            point_aware=point_fair is not None,quote_spread_pp=last.spread*100,
            fetched_at=now,
        ))
    output.sort(key=lambda x:(-x.edge_pp,x.price))
    return tuple(output)



def _resolve_live_state(
    leg: ParlayCandidateLeg,
    live_states: dict[tuple[str, str], TennisLiveScoreState],
) -> TennisLiveScoreState | None:
    """Exact pair preferred; unique pair fallback accounts for UTC/local-date mismatches."""
    player_key = canonical_participant("Tennis", leg.selection)
    exact = live_states.get((leg.event_id, player_key))
    if exact is not None:
        return exact
    parts = str(leg.event_id).split(":", 2)
    if len(parts) != 3:
        return None
    pair = [
        state for state in live_states.values()
        if state.selection_key == player_key
        and len(str(state.event_id).split(":", 2)) == 3
        and str(state.event_id).split(":", 2)[2] == parts[2]
    ]
    return pair[0] if len(pair) == 1 else None


@dataclass(frozen=True)
class ExtremeCheapObservation:
    event_id: str
    ticker: str
    selection: str
    current_ask: float
    observed_trough: float
    rebound_pp: float
    live_fair: float | None
    score: str
    lane: str
    score_sources: tuple[str, ...]
    latest_age_s: float


def build_extreme_cheap_observations(
    candidates: list[ParlayCandidateLeg] | tuple[ParlayCandidateLeg, ...],
    candles: dict[str, list[dict] | tuple[dict, ...]],
    live_states: dict[tuple[str, str], TennisLiveScoreState],
    *,
    now: datetime | None = None,
) -> tuple[ExtremeCheapObservation, ...]:
    """Discover 1–4¢ live prices without pretending the longshot is a pick.

    Continue watching recoveries up to 20¢ when the last-hour quoted trough was
    <=4¢. Promote only to *research-only* recovery-building after a fresh
    independent live probability, a material rebound, and no known break-state
    contradiction. Existing confirmed lanes are unchanged.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    out: list[ExtremeCheapObservation] = []
    for leg in candidates:
        if leg.sport != "Tennis" or leg.market_key != "model_h2h":
            continue
        if leg.model_probability is None or not leg.kalshi_ticker or not leg.kalshi_side:
            continue
        state = _resolve_live_state(leg, live_states)
        if state is None or not state.score_sources or state.score_conflict:
            continue
        if abs((now - state.fetched_at).total_seconds()) > 150 or state.at_tiebreak:
            continue
        path = executable_path(candles.get(str(leg.kalshi_ticker), ()), leg.kalshi_side)
        if not path:
            continue
        latest = path[-1]
        age = now.timestamp() - latest.end_ts
        if not (0 <= age <= 150 and latest.quoted and latest.spread is not None):
            continue
        if not (0.01 <= latest.close <= 0.20):
            continue
        spread_limit = min(.08, max(.04, latest.close * .45))
        if latest.spread > spread_limit:
            continue
        # This is the actual quoted trough in the currently retrieved lookback,
        # not an inferred entry or a candle that only traded at 1¢.
        trough = min(p.low for p in path if p.quoted)
        if trough > .04:
            continue
        rebound = max(0., (latest.close - trough) * 100.)
        live = estimate_live_match_probability(float(leg.model_probability), state)
        fair = live.probability if live is not None else None
        phase = "EXTREME DIP — TRACKING ONLY"
        if (latest.close > .04 and rebound >= 1.0
                and fair is not None and fair - latest.close >= .04
                and (leg.model_confidence or 0) >= .45
                and (leg.model_sample_size or 0) >= 6
                and state.net_break_advantage is not None
                and state.net_break_advantage >= 0):
            phase = "RECOVERY BUILDING — RESEARCH"
        out.append(ExtremeCheapObservation(
            event_id=leg.event_id,
            ticker=str(leg.kalshi_ticker),
            selection=leg.selection,
            current_ask=latest.close,
            observed_trough=trough,
            rebound_pp=rebound,
            live_fair=fair,
            score=state.score_label,
            lane=phase,
            score_sources=tuple(state.score_sources),
            latest_age_s=age,
        ))
    out.sort(key=lambda r:(r.lane.startswith("RECOVERY"), -r.rebound_pp, -r.current_ask),reverse=True)
    return tuple(out)
