from __future__ import annotations

def parse_mlb_recent_plays(feed: dict, limit: int = 12) -> list[dict]:
    plays=(feed.get("liveData",{}).get("plays",{}).get("allPlays") or [])[-limit:]
    out=[]
    for p in plays:
        about=p.get("about",{}); result=p.get("result",{}); matchup=p.get("matchup",{})
        out.append({
            "index": about.get("atBatIndex"),
            "inning": about.get("inning"),
            "half": about.get("halfInning"),
            "batter": matchup.get("batter",{}).get("fullName"),
            "pitcher": matchup.get("pitcher",{}).get("fullName"),
            "event": result.get("event"),
            "description": result.get("description"),
            "rbi": result.get("rbi"),
        })
    return out


def parse_espn_nfl_recent_plays(summary: dict, limit: int = 14) -> list[dict]:
    drives=summary.get("drives",{}).get("previous",[]) or []
    plays=[]
    for d in drives[-4:]:
        for p in d.get("plays",[]) or []:
            plays.append({
                "id":p.get("id"),
                "period":p.get("period",{}).get("number"),
                "clock":p.get("clock",{}).get("displayValue"),
                "text":p.get("text"),
                "awayScore":p.get("awayScore"),
                "homeScore":p.get("homeScore"),
                "scoringPlay":bool(p.get("scoringPlay")),
            })
    return plays[-limit:]
