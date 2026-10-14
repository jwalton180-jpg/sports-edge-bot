"""Regression: the live radar must not run the pregame app ticket audit.

The Streamlit traceback passed through model_candidates_from_kalshi() wrapper
into audit_app_candidates(). This test reproduces that audit failure and proves
the read-only radar's bounded independent-model path still operates.
"""
from pathlib import Path
from unittest.mock import patch

from sports_edge.models.tennis_radar_candidates import live_tennis_radar_candidates
from tests.test_model_candidates import FakeTennisModel, tmarket
from sports_edge.models.sleeper_pipeline import audit_app_candidates
from sports_edge.models.kalshi_sports import KalshiSportMarket


def _grouped():
    return {
        "Tennis":[
            tmarket("E1", "Player A", .20),
            tmarket("E1", "Player B", .80),
            tmarket("E2", "Player C", .55),
            tmarket("E2", "Player D", .45),
        ],
        "NFL":[KalshiSportMarket("NFL","Moneyline",{"ticker":"KXNFLGAME-X"})],
    }


def test_live_radar_does_not_hit_failing_pregame_ticket_audit():
    grouped=_grouped()
    with patch("sports_edge.models.kalshi_model_candidates._tennis_model",
               return_value=FakeTennisModel()), patch(
                   "sports_edge.models.sleeper_pipeline.audit_app_candidates",
                   side_effect=AttributeError("simulated live pregame audit crash")) as audit:
        rows=live_tennis_radar_candidates(grouped)
    audit.assert_not_called()
    assert len(rows)==4
    assert all(r.sport=="Tennis" and r.market_key=="model_h2h" for r in rows)
    assert {r.kalshi_side for r in rows}=={"YES"}
    assert all(r.model_probability is not None for r in rows)


def test_tennis_radar_screens_nonmodel_and_unsupported_rows(monkeypatch):
    from dataclasses import replace
    from sports_edge.models import tennis_radar_candidates as lane
    with patch("sports_edge.models.kalshi_model_candidates._tennis_model",
               return_value=FakeTennisModel()):
        baseline=live_tennis_radar_candidates(_grouped())
    raw=[*baseline, replace(baseline[0],model_probability=None),
         replace(baseline[0],market_key="player_points"),
         replace(baseline[0],kalshi_ticker=None),
         replace(baseline[0],kalshi_side=None),
         replace(baseline[0],sport="NFL")]
    monkeypatch.setattr(lane,"_independent_candidates",lambda grouped,**kwargs:raw)
    result=live_tennis_radar_candidates(_grouped())
    assert result==baseline


def test_empty_live_radar_never_invokes_independent_model(monkeypatch):
    from sports_edge.models import tennis_radar_candidates as lane
    monkeypatch.setattr(lane,"_independent_candidates",
                        lambda *args,**kw:(_ for _ in ()).throw(AssertionError("should not run")))
    assert live_tennis_radar_candidates({"Tennis":[]})==[]


def test_streamlit_radar_uses_live_only_input_not_pregame_ticket_wrapper():
    import ast
    app=Path("sports_edge/app.py").read_text()
    tree=ast.parse(app)
    # The surrounding live radar is nested inside the on-demand app.
    segment=app[app.index("def _render_tennis_reversal_radar():"):]
    segment=segment[:segment.index("        # ESPN remains the preferred ATP/WTA")]
    assert "live_tennis_radar_candidates(live_grouped)" in segment
    assert "row for row in model_candidates_from_kalshi(" not in segment
    # The ticket path must still call the mandatory audit.
    assert "return audit_app_candidates(candidates, inventory" in app
