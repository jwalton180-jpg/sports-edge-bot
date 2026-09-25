from collections import defaultdict, deque

from sports_edge.models.tennis_research import PlayerState, TennisResearchModel


def _model() -> TennisResearchModel:
    model = TennisResearchModel.__new__(TennisResearchModel)
    model.gender = "men"
    model.start_year = 2025
    model.end_year = 2026
    model.players = defaultdict(PlayerState)
    model.alias_index = {}
    for name, elo, recent, serve, ret in [
        ("Alpha Player", 1650.0, [1,1,1,1,0,1], [0.68,0.67,0.69,0.66], [0.35,0.34,0.36,0.35]),
        ("Beta Player", 1500.0, [0,0,1,0,0,1], [0.61,0.62,0.60,0.61], [0.31,0.30,0.32,0.31]),
    ]:
        state=model.players[model._name(name)]
        state.elo=elo
        state.matches=20
        state.recent_results=deque(recent,maxlen=20)
        state.serve_points_won=deque(serve,maxlen=20)
        state.return_points_won=deque(ret,maxlen=20)
        state.recent_dates=deque(maxlen=20)
        state.minutes=deque(maxlen=20)
    model._rebuild_alias_index()
    return model


def test_tour_keeps_enhanced_probability():
    model=_model()
    ev=model.probability("Alpha Player","Beta Player",level="ATP Tour")
    assert ev is not None
    elo=model._elo_probability(1650.0,1500.0)
    assert abs(ev.fair_probability-elo) > 1e-4
    assert ev.confidence <= 0.88


def test_challenger_blends_enhanced_model_with_elo_and_caps_confidence():
    model=_model()
    tour=model.probability("Alpha Player","Beta Player",level="ATP Tour")
    chall=model.probability("Alpha Player","Beta Player",level="Challenger")
    elo=model._elo_probability(1650.0,1500.0)
    assert tour is not None and chall is not None
    expected=0.5*tour.fair_probability + 0.5*elo
    assert abs(chall.fair_probability-expected) < 1e-9
    assert chall.confidence <= 0.65
    assert any("50/50" in factor for factor in chall.factors)


def test_itf_uses_elo_only_and_caps_confidence():
    model=_model()
    ev=model.probability("Alpha Player","Beta Player",level="ITF")
    elo=model._elo_probability(1650.0,1500.0)
    assert ev is not None
    assert abs(ev.fair_probability-elo) < 1e-9
    assert ev.confidence <= 0.55
    assert any("Elo-only" in factor for factor in ev.factors)
