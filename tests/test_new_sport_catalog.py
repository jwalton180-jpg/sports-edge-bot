from sports_edge.data.kalshi_catalog import (
    SUPPORTED_PREFIXES, PRODUCTION_SERIES_BY_SPORT, OVERVIEW_SERIES_BY_SPORT,
    _sport_for_series, _series_is_actionable,
)

def test_new_sports_are_in_real_catalog_discovery():
    cases = [("KXNHLPLAYERGOALS", "NHL"), ("KXNCAAFGAME", "NCAAF"),
             ("KXCFBSPREAD", "NCAAF"), ("KXEPLTOTAL", "Soccer")]
    for ticker, sport in cases:
        assert _sport_for_series({"ticker":ticker}) == sport
        assert _series_is_actionable({"ticker":ticker,"title":"Match total goals"},sport)
        assert sport in PRODUCTION_SERIES_BY_SPORT
        assert sport in OVERVIEW_SERIES_BY_SPORT
    assert {"NHL","NCAAF","Soccer"}.issubset({sport for _,sport in SUPPORTED_PREFIXES})

def test_futures_stay_excluded():
    assert not _series_is_actionable({"ticker":"KXNHLSEASON", "title":"Season champion"},"NHL")
    assert not _series_is_actionable({"ticker":"KXEPLWINNER", "title":"Tournament winner"},"Soccer")
