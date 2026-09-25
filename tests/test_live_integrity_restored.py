from sports_edge.data.kalshi_ws import KalshiReadOnlyWebSocket, SequenceGap
from sports_edge.models.underdog import score_underdog

def bare_ws():
    x=object.__new__(KalshiReadOnlyWebSocket)
    x.sequence=__import__('sports_edge.data.kalshi_ws',fromlist=['SequenceGuard']).SequenceGuard()
    return x

def test_sequence_accepts_contiguous():
    x=bare_ws(); x.validate_sequence({"sid":1,"seq":10}); x.validate_sequence({"sid":1,"seq":11})

def test_sequence_rejects_gap():
    x=bare_ws(); x.validate_sequence({"sid":1,"seq":10})
    try: x.validate_sequence({"sid":1,"seq":12})
    except SequenceGap: pass
    else: raise AssertionError("sequence gap must fail closed")

def test_underdog_signal_asymmetric():
    s=score_underdog(.12,.24,data_quality=.9,source_age_s=1)
    assert s.tier=="ASYMMETRIC" and s.edge_points==12 and round(s.ev_per_contract,6)==.12

def test_underdog_stale_pass():
    assert score_underdog(.12,.24,data_quality=.9,source_age_s=99).tier=="PASS"
