"""Read-only, bounded completed-game score archive for five sports.

Records are source-backed *final scoreboards*, not Kalshi market settlements.
They are validated before any joint simulation; player props are not inferred.
No bookmaker, Kalshi, or private account data is requested here.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlencode

import requests

ESPN_BASE="https://site.api.espn.com/apis/site/v2/sports"
PATHS={
    "NFL":"football/nfl",
    "NCAAF":"football/college-football",
    "NBA":"basketball/nba",
    "NHL":"hockey/nhl",
    "MLB":"baseball/mlb",
}
MAX_DAYS=45


def utc(raw):
    if not isinstance(raw,str):raise ValueError("timezone_aware_date_required")
    try:t=datetime.fromisoformat(raw.replace("Z","+00:00"))
    except ValueError:raise ValueError("invalid_time") from None
    if t.tzinfo is None:raise ValueError("naive_time_forbidden")
    return t.astimezone(timezone.utc)


def daily_url(sport, day):
    if sport not in PATHS:raise ValueError("unsupported_sport")
    if type(day) is not date:raise ValueError("date_required")
    return f"{ESPN_BASE}/{PATHS[sport]}/scoreboard?" + urlencode({"dates":day.strftime("%Y%m%d"),"limit":500})


def completed_scores(sport, data):
    if sport not in PATHS or not isinstance(data,dict) or not isinstance(data.get("events"),list):
        raise ValueError("malformed_source_or_sport")
    results=[]
    for e in data["events"]:
        if not isinstance(e,dict):raise ValueError("malformed_event")
        try:
            comps=e["competitions"]
            if not isinstance(comps,list) or len(comps)!=1:raise ValueError("unverified_competition")
            c=comps[0]
            state=e.get("status",{}).get("type",{}).get("completed") is True
            if not state:
                continue
            if not isinstance(c.get("competitors"),list) or len(c["competitors"])!=2:
                raise ValueError("missing_teams")
            teams={x["homeAway"]:x for x in c["competitors"]}
            if set(teams)!={"home","away"}:raise ValueError("unverified_home_away")
            def val(obj):
                v=obj["score"]
                if isinstance(v,dict):v=v.get("value",v.get("displayValue"))
                s=str(v)
                if not s.isdigit():raise ValueError("invalid_completed_score")
                x=int(s)
                if x>300:raise ValueError("score_out_of_range")
                return x
            home,away=teams["home"],teams["away"]
            score_home,score_away=val(home),val(away)
            h_id=str(home["team"]["id"]);a_id=str(away["team"]["id"])
            if not h_id or not a_id or h_id==a_id:raise ValueError("invalid_team_ids")
            event_time=utc(e["date"])
            event_id=str(e["id"])
            if not event_id:raise ValueError("invalid_event_id")
            results.append({
                "source":"ESPN_COMPLETED_SCOREBOARD","sport":sport,
                "event_id":event_id,"event_time":event_time.isoformat(),
                "home_id":h_id,"away_id":a_id,
                "home_score":score_home,"away_score":score_away,
                "source_scope":"FULL_GAME_FINAL_SCORE_NOT_KALSHI_SETTLEMENT",
            })
        except (KeyError,TypeError,IndexError,ValueError) as exc:
            raise ValueError("malformed_final_game_record") from exc
    if len({(r["sport"],r["event_id"]) for r in results})!=len(results):
        raise ValueError("duplicate_completed_events")
    return results


def validate_archive(doc, *, as_of=None):
    if not isinstance(doc,dict) or doc.get("schema")!="SPORTEDGE-FIVE-SPORT-SCORE-ARCHIVE-V1":
        raise ValueError("archive_schema_missing")
    game_rows=doc.get("games")
    if not isinstance(game_rows,list) or len(game_rows)>25000:
        raise ValueError("missing_or_unbounded_games")
    seen=set();clean=[]
    for r in game_rows:
        if not isinstance(r,dict) or r.get("sport") not in PATHS:
            raise ValueError("unsupported_or_malformed_game")
        if r.get("source")!="ESPN_COMPLETED_SCOREBOARD" or r.get("source_scope")!="FULL_GAME_FINAL_SCORE_NOT_KALSHI_SETTLEMENT":
            raise ValueError("source_provenance_missing")
        event_time=utc(r.get("event_time"))
        if as_of is not None and event_time>=utc(as_of):
            raise ValueError("archive_contains_future_game_for_cutoff")
        key=(r.get("sport"),r.get("event_id"))
        if not isinstance(r.get("event_id"),str) or not r["event_id"] or key in seen:
            raise ValueError("duplicate_or_invalid_game_id")
        seen.add(key)
        h,a=r.get("home_id"),r.get("away_id")
        if not isinstance(h,str) or not isinstance(a,str) or not h or not a or h==a:
            raise ValueError("invalid_teams")
        for score in (r.get("home_score"),r.get("away_score")):
            if type(score) is not int or not 0<=score<=300:
                raise ValueError("invalid_score")
        clean.append(r)
    return sorted(clean,key=lambda r:(r["event_time"],r["sport"],r["event_id"]))


def collect_range(*, sport, start, end, get=None):
    """No implicit network in imports; only a deliberate bounded invocation."""
    if sport not in PATHS:raise ValueError("unsupported_sport")
    if type(start) is not date or type(end) is not date or start>end or (end-start).days>=MAX_DAYS:
        raise ValueError("date_range_must_be_1_to_45_days")
    if end>=datetime.now(timezone.utc).date():
        raise ValueError("only_past_utc_dates_can_be_finalized")
    rows={};pages=[]
    sess=requests.Session() if get is None else None
    try:
        for offset in range((end-start).days+1):
            day=start+timedelta(days=offset)
            url=daily_url(sport,day)
            if get is None:
                response=sess.get(url,timeout=20,headers={"User-Agent":"SportEdge-readonly-research/1.0","Accept":"application/json"})
                response.raise_for_status()
                raw=response.content
                data=response.json()
            else:
                data=get(url)
                raw=json.dumps(data,sort_keys=True,separators=(",",":")).encode()
            games=completed_scores(sport,data)
            pages.append({"date":day.isoformat(),"url":url,"sha256":sha256(raw).hexdigest(),
                          "source_completed_games":len(games)})
            for game in games:
                k=(sport,game["event_id"])
                if k in rows and rows[k]!=game:raise ValueError("conflicting_game_results")
                rows[k]=game
        if not pages:raise ValueError("no_source_days")
        return {"schema":"SPORTEDGE-FIVE-SPORT-SCORE-ARCHIVE-V1",
                "source":"ESPN_PUBLIC_COMPLETED_SCOREBOARDS",
                "sports":[sport],"range":[start.isoformat(),end.isoformat()],
                "fetched_at_utc":datetime.now(timezone.utc).isoformat(),
                "provenance":pages,
                "games":sorted(rows.values(),key=lambda r:(r["event_time"],r["event_id"])),
                "research_only":True,"no_kalshi_market_settlement_claim":True}
    finally:
        if sess is not None:sess.close()


def merge_archives(docs):
    if not docs:raise ValueError("archive_required")
    rows={};proof=[]
    for doc in docs:
        if doc.get("research_only") is not True or doc.get("no_kalshi_market_settlement_claim") is not True:
            raise ValueError("archive_permissions_incomplete")
        games=validate_archive(doc)
        sources=doc.get("provenance")
        if not isinstance(sources,list) or not sources:
            raise ValueError("source_download_provenance_missing")
        proof.extend(sources)
        for r in games:
            k=(r["sport"],r["event_id"])
            if k in rows and rows[k]!=r:
                raise ValueError("conflicting_games_from_source")
            rows[k]=r
    return {"schema":"SPORTEDGE-FIVE-SPORT-SCORE-ARCHIVE-V1",
            "source":"ESPN_PUBLIC_COMPLETED_SCOREBOARDS",
            "sports":sorted({r["sport"] for r in rows.values()}),
            "range":None,
            "fetched_at_utc":datetime.now(timezone.utc).isoformat(),
            "provenance":proof,
            "games":sorted(rows.values(),key=lambda r:(r["event_time"],r["sport"],r["event_id"])),
            "research_only":True,"no_kalshi_market_settlement_claim":True}
