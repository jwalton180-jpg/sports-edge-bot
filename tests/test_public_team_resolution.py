from sports_edge.data.public_team_data import (
    _metadata_aliases,
    _mlb_match_team_id,
    _resolve_competitors,
)


def comp(display, location, name, abbr):
    return {
        "team": {
            "displayName": display,
            "shortDisplayName": name,
            "location": location,
            "name": name,
            "abbreviation": abbr,
        }
    }


def test_espn_resolver_matches_kalshi_city_to_full_nfl_team():
    competitors = [
        comp("Kansas City Chiefs", "Kansas City", "Chiefs", "KC"),
        comp("Denver Broncos", "Denver", "Broncos", "DEN"),
    ]
    resolved = _resolve_competitors("Kansas City", "Denver", competitors)
    assert resolved is not None
    a, b = resolved
    assert a["team"]["displayName"] == "Kansas City Chiefs"
    assert b["team"]["displayName"] == "Denver Broncos"


def test_resolver_handles_multiword_city_without_generic_city_alias():
    competitors = [
        comp("Oklahoma City Thunder", "Oklahoma City", "Thunder", "OKC"),
        comp("Houston Rockets", "Houston", "Rockets", "HOU"),
    ]
    resolved = _resolve_competitors("Oklahoma City", "Houston", competitors)
    assert resolved is not None
    assert resolved[0]["team"]["displayName"] == "Oklahoma City Thunder"


def test_resolver_fails_closed_when_city_is_ambiguous_within_event():
    competitors = [
        comp("Los Angeles Lakers", "Los Angeles", "Lakers", "LAL"),
        comp("Los Angeles Clippers", "Los Angeles", "Clippers", "LAC"),
    ]
    assert _resolve_competitors("Los Angeles", "LA Clippers", competitors) is None


def test_structured_aliases_include_location_nickname_and_abbreviation():
    team = {
        "displayName": "Las Vegas Aces",
        "location": "Las Vegas",
        "name": "Aces",
        "abbreviation": "LVA",
    }
    aliases = _metadata_aliases(team)
    assert "las vegas" in aliases
    assert "aces" in aliases
    assert "lva" in aliases
    assert "las vegas aces" in aliases


def test_mlb_city_alias_resolves_uniquely():
    aliases = {
        118: {"kansas city", "royals", "kansas city royals", "kc"},
        121: {"new york", "mets", "new york mets", "nym"},
    }
    assert _mlb_match_team_id("Kansas City", aliases) == 118


def test_mlb_ambiguous_location_fails_closed():
    aliases = {
        147: {"new york", "yankees", "new york yankees", "nyy"},
        121: {"new york", "mets", "new york mets", "nym"},
    }
    assert _mlb_match_team_id("New York", aliases) is None
