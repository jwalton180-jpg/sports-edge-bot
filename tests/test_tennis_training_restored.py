from datetime import datetime, timezone, timedelta
from sports_edge.models.tennis_training import TennisMatch, chronological_predictions, calibration_report, split_report

def dt(n): return datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(days=n)

def test_chronological_no_current_match_leakage():
    a=TennisMatch(dt(0),'A','B','hard','wta',.99,.01)
    p=chronological_predictions([a])[0]
    assert p.probability_a == .5

def test_prior_result_moves_next_prediction():
    ms=[TennisMatch(dt(0),'A','B','hard','wta'), TennisMatch(dt(3),'A','B','hard','wta')]
    ps=chronological_predictions(ms)
    assert ps[0].probability_a == .5 and ps[1].probability_a > .5

def test_retirement_does_not_train_elo():
    ms=[TennisMatch(dt(0),'A','B','clay','itf',retired=True), TennisMatch(dt(3),'A','B','clay','itf')]
    ps=chronological_predictions(ms)
    assert ps[1].probability_a == .5

def test_separate_itf_calibration_report():
    ps=chronological_predictions([TennisMatch(dt(0),'A','B','hard','itf'), TennisMatch(dt(1),'C','D','hard','wta')])
    r=split_report(ps)
    assert r['itf']['n']==1 and r['wta']['n']==1
    assert calibration_report(ps)['n']==2

def test_prior_serve_return_form_moves_future_prediction_without_current_leakage():
    ms=[TennisMatch(dt(0),'A','B','hard','wta',.72,.42), TennisMatch(dt(3),'A','C','hard','wta',.01,.99)]
    ps=chronological_predictions(ms)
    assert ps[1].prior_form_samples_a == 1
    assert ps[1].prior_form_samples_b == 0
    assert ps[1].prior_form_delta > 0
    control=chronological_predictions([ms[0], TennisMatch(dt(3),'A','C','hard','wta')])
    assert ps[1].probability_a == control[1].probability_a
