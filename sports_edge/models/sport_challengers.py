"""Independent, historical-only NHL player goal challenger."""
from __future__ import annotations
from math import exp
from sports_edge.models.new_sport_research import Projection

def nhl_shot_challenger(prior: list[dict], *, threshold: int = 2) -> Projection:
    """Empirical Bayes shooting and recent opportunity, with no market odds."""
    model="nhl_shot_shrink_v2"
    if not 1<=threshold<=6 or len(prior)<20 or any(
        r.get("source")!="NHL" or type(r.get("shots")) is not int or
        type(r.get("goals")) is not int or r["shots"]<r["goals"] or r["goals"]<0
        for r in prior):
        return Projection("NHL",f"{threshold}+ player goals",None,model,
                          "RESEARCH_ONLY",("missing_verified_shots_or_history",),len(prior))
    shots=sum(r["shots"] for r in prior)
    goals=sum(r["goals"] for r in prior)
    shooting=(goals+4.5)/(shots+50.)
    recent=prior[-10:]
    recent_volume=(sum(r["shots"] for r in recent)+60)/(len(recent)+20)
    season_volume=(shots+60)/(len(prior)+20)
    rate=shooting*(.6*recent_volume+.4*season_volume)
    from math import factorial
    probability=1-exp(-rate)*sum(rate**k/factorial(k) for k in range(threshold))
    return Projection("NHL",f"{threshold}+ player goals",max(0.,min(1.,probability)),
                      model,"UNVALIDATED_RESEARCH",
                      ("shooting_prior_assumption","goalie_and_ice_time_unadjusted"),len(prior))
