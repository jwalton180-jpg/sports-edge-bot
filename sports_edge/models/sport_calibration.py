"""Time-split Platt calibration; no model promotion without prospective evidence."""
from __future__ import annotations
from math import exp, log, isfinite
from sports_edge.models.sport_walkforward import calibration_report, _date

def _sigmoid(z):
    z = max(-35., min(35., z))
    return 1. / (1. + exp(-z))

def _logit(p):
    p = max(1e-5, min(1-1e-5, float(p)))
    return log(p/(1-p))

def fit_platt(train: list[dict], *, iterations: int = 700, learning_rate: float = .08):
    """L2-regularized logistic fit on training predictions only."""
    if len(train) < 100:
        raise ValueError("insufficient calibration training observations")
    if len({(r["sport"],r["event_id"],r["outcome"]) for r in train}) != len(train):
        raise ValueError("duplicate training event")
    x = []
    for r in train:
        p = r["probability"]
        if r.get("evaluation_kind") != "historical_walkforward" or type(r.get("won")) is not bool or not isinstance(p,(int,float)) or not isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid calibration input")
        x.append((_logit(p), int(r["won"])))
    a, b = 0., 1.
    for _ in range(iterations):
        da = sum(_sigmoid(a+b*v)-y for v,y in x)/len(x) + .02*a
        db = sum((_sigmoid(a+b*v)-y)*v for v,y in x)/len(x) + .02*(b-1)
        a -= learning_rate*da
        b -= learning_rate*db
    return a, b

def chronological_calibration(rows: list[dict], *, min_train: int = 100,
                              min_test: int = 50, test_fraction: float = .25):
    """One frozen temporal split; fit only on earlier completed games.

    Output is diagnostic. Even strong historical holdout metrics cannot certify
    prospective reliability or authorize live recommendations.
    """
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("invalid test fraction")
    ordered = sorted(rows,key=lambda r: (_date(r["event_time"]),r["event_id"]))
    n = len(ordered)
    split = int(n*(1-test_fraction))
    if split < min_train or n-split < min_test:
        return {"status":"INSUFFICIENT_HISTORY","qualified_for_live":False,
                "n":n,"min_train":min_train,"min_test":min_test}
    # Same-time events never leak between training and test.
    while split > 0 and _date(ordered[split-1]["event_time"]) == _date(ordered[split]["event_time"]):
        split -= 1
    if split < min_train or n-split < min_test:
        return {"status":"INSUFFICIENT_HISTORY","qualified_for_live":False,"n":n}
    train, test = ordered[:split], ordered[split:]
    if len({r["sport"] for r in rows}) != 1:
        raise ValueError("calibration must be sport-specific")
    a,b = fit_platt(train)
    scored = [dict(r,probability=_sigmoid(a+b*_logit(r["probability"]))) for r in test]
    raw = calibration_report(test)
    calibrated = calibration_report(scored)
    sport = test[0]["sport"]
    return {"status":"HISTORICAL_HOLDOUT_ONLY","qualified_for_live":False,
            "train_n":len(train),"test_n":len(test),
            "train_cutoff":train[-1]["event_time"],
            "test_start":test[0]["event_time"],
            "coefficients":{"intercept":a,"slope":b},
            "raw":raw[sport],"calibrated":calibrated[sport],
            "reason":"requires_future_immutable_prospective_predictions"}
