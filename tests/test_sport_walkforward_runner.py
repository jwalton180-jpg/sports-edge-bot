from datetime import date
from scripts.run_sport_walkforward import days, collect_team

def test_date_range_is_inclusive():
    assert list(days(date(2025,1,1),date(2025,1,3))) == [
        date(2025,1,1),date(2025,1,2),date(2025,1,3)]

def test_collection_does_not_skip_weekdays(monkeypatch):
    from scripts import run_sport_walkforward as module
    calls=[]
    def fake(sport, day):
        calls.append(day)
        return ()
    monkeypatch.setattr(module, "completed_team_games", fake)
    assert collect_team("NCAAF",date(2025,9,4),date(2025,9,6)) == []
    assert calls == ["20250904","20250905","20250906"]
