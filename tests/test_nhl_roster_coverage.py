from sports_edge.data import historical_sport_evidence as h
import pytest

def test_nhl_roster_requires_complete_noncherrypicked_population(monkeypatch):
    rows=[{"playerId":i+100000} for i in range(320)]
    monkeypatch.setattr(h, "_get_json", lambda *a, **k: {"data":rows,"total":320})
    h.nhl_complete_skater_ids.cache_clear()
    assert len(h.nhl_complete_skater_ids(20242025)) == 320
    h.nhl_complete_skater_ids.cache_clear()
    monkeypatch.setattr(h, "_get_json", lambda *a, **k: {"data":rows[:2],"total":320})
    with pytest.raises(ValueError, match="incomplete"):
        h.nhl_complete_skater_ids(20242025)
