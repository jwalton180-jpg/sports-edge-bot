from sports_edge.data.sleeper_sources import SleeperSources,kalshi_event_day
from sports_edge.models.prospective_capture import model_family_supported

def test_kalshi_occurrence_is_not_nhl_game_start():
    source=SleeperSources()
    calls=[]
    def fake(url,params=None):
        calls.append(url)
        assert url.endswith("/schedule/2026-10-14")
        return {"gameWeek":[{"games":[{
            "gameType":2,"gameScheduleState":"OK",
            "gameDate":"2026-10-14","startTimeUTC":"2026-10-14T23:30:00Z",
            "awayTeam":{"abbrev":"MTL"},"homeTeam":{"abbrev":"WSH"}
        }]}]}
    source.get=fake
    market={"event_ticker":"KXNHLGOAL-26OCT14MTLWSH",
            "occurrence_datetime":"2026-10-15T02:30:00Z"}
    match=source.nhl_match(market)
    assert match["startTimeUTC"]=="2026-10-14T23:30:00Z"
    assert kalshi_event_day(market["event_ticker"]).isoformat()=="2026-10-14"

def test_bad_event_day_or_game_pair_refused():
    source=SleeperSources()
    source.get=lambda *a,**k:{"gameWeek":[{"games":[{
         "gameType":2,"gameScheduleState":"OK",
         "gameDate":"2026-10-14","startTimeUTC":"2026-10-14T23:30:00Z",
         "awayTeam":{"abbrev":"TOR"},"homeTeam":{"abbrev":"WSH"}
    }]}]}
    try:source.nhl_match({"event_ticker":"KXNHLGOAL-26OCT14MTLWSH",
        "occurrence_datetime":"2026-10-15T02:30:00Z"})
    except ValueError:pass
    else:assert False

def test_espn_uses_original_event_day_and_validates_end_window():
    source=SleeperSources()
    calls=[]
    def fake(url,params=None):
        calls.append((url,params))
        if url.endswith("/scoreboard"):
            return {"events":[{
                "id":"1","date":"2026-10-14T23:30:00Z",
                "competitions":[{"competitors":[
                    {"id":"1","homeAway":"home","team":{"displayName":"Missouri State","shortDisplayName":"Missouri State","location":"Missouri"}},
                    {"id":"2","homeAway":"away","team":{"displayName":"Kennesaw State","shortDisplayName":"Kennesaw State","location":"Kennesaw"}}
                ]}]
            }]}
        return {"events":[]}
    source.get=fake
    context=source.espn_match_history({
        "ticker":"KXNCAAFGAME-26OCT14KENNMOSU-KENN",
        "event_ticker":"KXNCAAFGAME-26OCT14KENNMOSU",
        "occurrence_datetime":"2026-10-15T02:30:00Z",
        "rules_primary":"If Kennesaw State defeats Missouri State the market resolves to Yes."
    },"NCAAF")
    assert context["start"]=="2026-10-14T23:30:00Z"
    assert calls[0][1]["dates"]=="20261014"

def test_sampling_excludes_unmodeled_mls_and_college_team_totals():
    assert not model_family_supported({"ticker":"KXMLSTOTAL-26OCT14XXX-1"},"Soccer")
    assert not model_family_supported({"ticker":"KXNCAAFTEAMTOTAL-26OCT14XXX-1"},"NCAAF")
    assert model_family_supported({"ticker":"KXEPLTOTAL-26OCT14XXX-1"},"Soccer")
    assert model_family_supported({"ticker":"KXNHLPTS-26OCT14XXX-1"},"NHL")
