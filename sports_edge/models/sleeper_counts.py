"""Independent count tails with dispersion and explicit sensitivity bounds.

Bounds are model sensitivity ranges, NOT calibrated confidence intervals.
These challengers never self-promote. Inputs are completed pre-event records.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import exp, isfinite, lgamma, log
from statistics import mean, pvariance


@dataclass(frozen=True)
class CountEstimate:
    probability: float | None
    lower: float | None
    upper: float | None
    mean: float | None
    variance: float | None
    sample: int
    model: str
    reasons: tuple[str, ...]
    status: str = "UNVALIDATED_RESEARCH"


def utc(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timezone required")
    return dt.astimezone(timezone.utc)


def count_tail(mu, variance, threshold):
    if type(threshold) is not int or threshold < 1 or not all(isfinite(x) and x >= 0 for x in (mu, variance)):
        raise ValueError("invalid count distribution")
    if mu == 0:
        return 0.
    if variance <= mu * 1.03:
        term = exp(-mu)
        cdf = term
        for x in range(1, threshold):
            term *= mu / x
            cdf += term
    else:
        r = max(.25, mu * mu / (variance - mu))
        p = r / (r + mu)
        cdf = sum(exp(lgamma(x+r)-lgamma(r)-lgamma(x+1)+r*log(p)+x*log(1-p))
                  for x in range(threshold))
    return max(0., min(1., 1-cdf))


def unavailable(model, reason, sample=0):
    return CountEstimate(None, None, None, None, None, sample, model, (reason,), "RESEARCH_ONLY")


def estimate_counts(rows, *, field, threshold, before_time, source, model,
                    minimum=20, prior_mean=.5, exposure_field=None):
    """Identity must be homogeneous; future/target-day results are excluded.

    Date-only sources require a full day exclusion. Missing stats reject the
    projection rather than silently substituting zeros.
    """
    cutoff = utc(before_time)
    prior = []
    seen = set()
    identity = set()
    for row in rows:
        try:
            when = utc(row["event_time"]) if row.get("event_time") else utc(row["date"]+"T00:00:00Z") + timedelta(days=1)
        except (ValueError, KeyError, TypeError):
            return unavailable(model, "historical_source_timestamp_missing")
        if when >= cutoff or (row.get("event_time") and when + timedelta(hours=8) >= cutoff):
            continue
        key = str(row.get("game_id") or row.get("event_id") or "")
        identity.add(str(row.get("player_id") or row.get("team_id") or ""))
        v = row.get(field)
        if not key or key in seen or row.get("source") != source or type(v) is not int or not 0 <= v <= 200:
            return unavailable(model, "duplicate_unverified_or_missing_stat", len(prior))
        seen.add(key)
        if source == "NHL" and all(type(row.get(k)) is int for k in ("goals", "assists", "points")) and row["points"] != row["goals"]+row["assists"]:
            return unavailable(model, "inconsistent_official_points")
        prior.append((when, row))
    if len(identity) > 1:
        return unavailable(model, "mixed_participant_history")
    prior.sort(key=lambda pair: pair[0])
    history = [r for _, r in prior][-60:]
    n = len(history)
    if type(threshold) is not int or not 1 <= threshold <= 100 or n < minimum:
        return unavailable(model, "insufficient_history_or_invalid_threshold", n)
    values = [r[field] for r in history]
    full = (sum(values)+12*prior_mean)/(n+12)
    recent = values[-10:]
    mu = .65*full + .35*(sum(recent)+12*prior_mean)/(len(recent)+12)
    reasons = ["pre_event_history_only", "negative_binomial_when_overdispersed",
               "sensitivity_bounds_not_confidence_intervals"]
    if exposure_field:
        exposures = [r.get(exposure_field) for r in history]
        if any(v is None or not isfinite(v) or v <= 0 for v in exposures):
            reasons.append("deployment_exposure_missing")
        else:
            ratio = mean(exposures[-5:])/mean(exposures)
            mu *= max(.85, min(1.15, ratio))
            reasons.append("recent_exposure_ratio_capped_15_percent")
    variance = max(mu, .65*pvariance(values)+.35*mu)
    poisson = count_tail(mu, mu, threshold)
    dispersed = count_tail(mu, variance, threshold)
    # Empirical tail receives a fixed 20% weight; small samples shrink to the
    # model prior. No tuning against this event's outcome or market price.
    empirical = (sum(v >= threshold for v in values)+12*dispersed)/(n+12)
    p = .8*dispersed+.2*empirical
    tails = [poisson, dispersed, empirical,
             count_tail(mu*.8, variance, threshold), count_tail(mu*1.2, max(variance,mu*1.2), threshold)]
    return CountEstimate(p, min(tails), max(tails), mu, variance, n, model, tuple(reasons))


def nfl_scored_touchdowns(rows, *, threshold=2, before_time):
    scored = []
    for r in rows:
        copied = dict(r)
        try:
            vals = [float(r[k]) for k in ("rushing_tds", "receiving_tds")]
            if any(not isfinite(v) or v < 0 or not v.is_integer() for v in vals):
                raise ValueError("invalid touchdowns")
            copied["scored_tds"] = int(sum(vals))
        except (KeyError, TypeError, ValueError):
            copied["scored_tds"] = None
        scored.append(copied)
    return estimate_counts(scored, field="scored_tds", threshold=threshold,
                           before_time=before_time, source="nflverse",
                           model="nfl_scored_td_dispersion_v2", minimum=12, prior_mean=.25)


def nhl_count(rows, *, market, threshold, before_time):
    priors = {"goals": .12, "shots": 2.5, "assists": .35, "points": .55,
              "saves": 27., "team_goals": 3.}
    if market not in priors:
        raise ValueError("unsupported NHL count")
    exposure = "toi_minutes" if market != "team_goals" else None
    estimate = estimate_counts(rows, field=market, threshold=threshold,
                               before_time=before_time, source="NHL",
                               model=f"nhl_{market}_dispersion_v2", prior_mean=priors[market],
                               exposure_field=exposure)
    return estimate
