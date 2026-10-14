"""Bounded on-demand public sports adapters. No account/order endpoints.

Only official NHL records and ESPN finalized results enter these models.
An unresolvable participant or schedule discrepancy is a rejected join.
"""
from datetime import timedelta
from datetime import datetime, timezone
import unicodedata
import requests
from sports_edge.models.sleeper_counts import utc
from sports_edge.models.game_scope import normalize

NHL = "https://api-web.nhle.com/v1"
ESPN = "https://site.api.espn.com/apis/site/v2/sports"


def nominal_day(market):
    import re
    from datetime import date
    months={m:i for i,m in enumerate(("JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"),1)}
    match=re.search(r"-(\d{2})([A-Z]{3})(\d{2})", str(market.get("event_ticker", "")).upper())
    if not match or match[2] not in months:
        raise ValueError("verified_nominal_event_date_missing")
    return date(2000+int(match[1]),months[match[2]],int(match[3]))


def kalshi_event_day(ticker):
    """Compatibility helper for independent prospective capture."""
    return nominal_day({"event_ticker": ticker})


def name(value):
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return normalize(value)


class SleeperSources:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SportEdge-ReadOnly/2.0"})
        self.cache = {}
        self.provenance = []

    def get(self, url, params=None):
        key = (url, tuple(sorted((params or {}).items())))
        if key not in self.cache:
            if len(self.cache) >= 120:
                raise ValueError("source_request_budget_reached")
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise ValueError("invalid source payload")
            self.cache[key] = data
            self.provenance.append({"url": r.url, "retrieved_at": datetime.now(timezone.utc).isoformat()})
        return self.cache[key]

    def nhl_match(self, market):
        start = utc(market["occurrence_datetime"])
        ticker = market.get("event_ticker", "")
        day = nominal_day(market).isoformat()
        # Schedule UTC date may differ from the nominal North American date.
        data = self.get(f"{NHL}/schedule/{day}")
        matches = []
        for week in data.get("gameWeek", []):
            for g in week.get("games", []):
                if week.get("date") != day or g.get("gameType") != 2 or g.get("gameScheduleState") != "OK":
                    continue
                codes = [g[k]["abbrev"] for k in ("awayTeam", "homeTeam")]
                pairs = ["".join(codes), "".join(reversed(codes))]
                # Kalshi's TB code differs from NHL's TBL.
                pairs += [p.replace("TBL", "TB") for p in pairs]
                if any(p in ticker for p in pairs) and 0 <= (start-utc(g["startTimeUTC"])).total_seconds() <= 8*3600:
                    matches.append(g)
        if len(matches) != 1:
            raise ValueError("nhl_schedule_identity_or_start_conflict")
        return matches[0]

    def nhl_player(self, market, player):
        g = self.nhl_match(market)
        matches = []
        for team in (g["awayTeam"], g["homeTeam"]):
            roster = self.get(f"{NHL}/roster/{team['abbrev']}/{g['season']}")
            for section in ("forwards", "defensemen", "goalies"):
                for p in roster.get(section, []):
                    full = p["firstName"]["default"]+" "+p["lastName"]["default"]
                    if name(full) == name(player):
                        matches.append((p["id"], section, team["abbrev"]))
        if len(matches) != 1:
            raise ValueError("nhl_player_identity_ambiguous_or_missing")
        pid, role, team = matches[0]
        logs = []
        for season in (g["season"]-10001, g["season"]):
            raw = self.get(f"{NHL}/player/{pid}/game-log/{season}/2")
            schedules={}
            for r in raw.get("gameLog", []):
                played_for=r.get("teamAbbrev")
                if not played_for:
                    raise ValueError("historical_player_team_missing")
                if played_for not in schedules:
                    data=self.get(f"{NHL}/club-schedule-season/{played_for}/{season}")
                    schedules[played_for]={str(item["id"]):item for item in data.get("games", [])}
                scheduled=schedules[played_for].get(str(r["gameId"]))
                if not scheduled or scheduled.get("gameState") not in {"OFF","FINAL"}:
                    continue
                mm, ss = str(r.get("toi", "0:00")).split(":")
                item = {"source": "NHL", "player_id": pid, "game_id": str(r["gameId"]),
                        "date": r["gameDate"], "event_time":scheduled["startTimeUTC"], "toi_minutes": int(mm)+int(ss)/60,
                        "opponent": r.get("opponentAbbrev"), "started": r.get("gamesStarted") == 1}
                for f in ("goals", "shots", "assists", "points"):
                    item[f] = r.get(f)
                if type(r.get("shotsAgainst")) is int and type(r.get("goalsAgainst")) is int:
                    item["saves"] = r["shotsAgainst"]-r["goalsAgainst"]
                if role == "goalies" and not item["started"]:
                    continue
                logs.append(item)
        return {"event_id": str(g["id"]), "start": g["startTimeUTC"], "participant": player,
                "opponent": g["homeTeam" if team == g["awayTeam"]["abbrev"] else "awayTeam"]["abbrev"],
                "rows": logs, "role": role, "deployment_verified": False,
                "missing": ["confirmed_starting_goalie"] if role == "goalies" else ["confirmed_line_and_power_play_role"]}

    def nhl_team(self, market, team_name):
        game = self.nhl_match(market)
        matches = []
        for t in (game["homeTeam"], game["awayTeam"]):
            aliases = [t["abbrev"], t["placeName"]["default"], t["commonName"]["default"],
                       t["placeName"]["default"]+" "+t["commonName"]["default"]]
            if t["abbrev"] == "VGK":
                aliases.append("Vegas")
            if name(team_name) in {name(x) for x in aliases}:
                matches.append(t)
        if len(matches) != 1:
            raise ValueError("nhl_team_identity_ambiguous")
        t = matches[0]
        rows = []
        secondary = str(market.get("rules_secondary", "")).lower()
        include_shootout = "shootout winner is credited with one additional goal" in secondary and "counts toward this market" in secondary
        for season in (game["season"]-10001, game["season"]):
            data = self.get(f"{NHL}/club-schedule-season/{t['abbrev']}/{season}")
            for g in data.get("games", []):
                if g.get("gameType") != 2 or g.get("gameState") not in {"OFF", "FINAL"}:
                    continue
                side = "homeTeam" if g["homeTeam"]["id"] == t["id"] else "awayTeam"
                score = g[side].get("score")
                if type(score) is not int:
                    raise ValueError("missing_final_team_score")
                # Remove the administrative shootout-winning goal; team count
                # model's explicit scope is regulation+OT excluding shootout.
                if not include_shootout and g.get("gameOutcome", {}).get("lastPeriodType") == "SO" and score > g["awayTeam" if side == "homeTeam" else "homeTeam"]["score"]:
                    score -= 1
                rows.append({"source": "NHL", "team_id": t["id"], "game_id": str(g["id"]),
                             "event_time": g["startTimeUTC"], "team_goals": score})
        return {"event_id": str(game["id"]), "start": game["startTimeUTC"], "participant": team_name,
                "rows": rows, "missing": ["opponent_goalie_and_lineups"], "deployment_verified": False,
                "score_scope":"regulation_plus_OT_plus_shootout_bonus" if include_shootout else "regulation_plus_OT_excluding_shootout"}

    @staticmethod
    def espn_result(event, sport):
        comp = event["competitions"][0]
        status = comp.get("status", event.get("status", {})).get("type", {})
        if status.get("completed") is not True or len(comp["competitors"]) != 2:
            return None
        home = next(x for x in comp["competitors"] if x["homeAway"] == "home")
        away = next(x for x in comp["competitors"] if x["homeAway"] == "away")
        def score(x):
            v = x["score"]
            return int(v.get("value", v.get("displayValue"))) if isinstance(v, dict) else int(v)
        return {"source": "ESPN", "sport": sport, "event_id": str(event["id"]),
                "event_time": event["date"], "home_id": str(home["id"]), "away_id": str(away["id"]),
                "home_score": score(home), "away_score": score(away)}

    def espn_match_history(self, market, sport):
        path = {"Soccer": "soccer/eng.1", "NCAAF": "football/college-football"}[sport]
        start = utc(market["occurrence_datetime"])
        day=nominal_day(market)
        events={}
        for candidate_day in (day,day+timedelta(days=1)):
            data=self.get(f"{ESPN}/{path}/scoreboard", {"dates":candidate_day.strftime("%Y%m%d"), "limit":500})
            events.update({str(e["id"]):e for e in data.get("events", [])})
        rules = str(market.get("rules_primary", ""))
        candidates = []
        for e in events.values():
            if not 0 <= (start-utc(e["date"])).total_seconds() <= 8*3600:
                continue
            teams = e["competitions"][0]["competitors"]
            # Exact names/abbreviations bounded by words, not fuzzy guesses.
            import re
            if len(teams) == 2 and all(any(re.search(r"\b"+re.escape(name(t["team"].get(k, "")))+r"\b", name(rules))
                                                         for k in ("displayName", "shortDisplayName", "location")
                                                         if t["team"].get(k)) for t in teams):
                candidates.append(e)
        if len(candidates) != 1:
            raise ValueError("espn_schedule_identity_or_start_conflict")
        e = candidates[0]
        comp = e["competitions"][0]
        home = next(t for t in comp["competitors"] if t["homeAway"] == "home")
        away = next(t for t in comp["competitors"] if t["homeAway"] == "away")
        start = utc(e["date"])
        collected = {}
        for t in (home, away):
            for season in (start.year-1, start.year):
                data = self.get(f"{ESPN}/{path}/teams/{t['id']}/schedule", {"season": season})
                for past in data.get("events", []):
                    if utc(past["date"])+timedelta(hours=24) >= start:
                        continue
                    row = self.espn_result(past, sport)
                    if row:
                        collected[row["event_id"]] = row
        def aliases(team):
            info = team["team"]
            return [info[k] for k in ("displayName", "shortDisplayName", "abbreviation", "location") if info.get(k)]
        return {"event_id": str(e["id"]), "start": e["date"], "home_id": str(home["id"]),
                "home_aliases": aliases(home), "away_aliases": aliases(away),
                "away_id": str(away["id"]), "home_name": home["team"]["displayName"],
                "away_name": away["team"]["displayName"], "rows": sorted(collected.values(), key=lambda r:r["event_time"]),
                "missing": ["confirmed_lineups", "weather", "injuries"], "deployment_verified": False}

    def nba_margin_context(self, market):
        """Exact NBA regular-season event and past complete-game schedule joins.

        Market occurrence is NOT assumed to be tipoff. Sources and identity
        come from ESPN; no Kalshi probabilities or sportsbook spreads enter.
        """
        from sports_edge.data.public_team_data import _metadata_aliases
        from sports_edge.models.game_scope import normalize
        nominal=nominal_day(market)
        occurrence=utc(market["occurrence_datetime"])
        ticker=str(market.get("event_ticker") or "").upper()
        events={}
        for day in (nominal,nominal+timedelta(days=1)):
            payload=self.get(f"{ESPN}/basketball/nba/scoreboard",
                             {"dates":day.strftime("%Y%m%d"),"limit":100})
            for e in payload.get("events",[]):
                events[str(e["id"])]=e
        matched=[]
        for e in events.values():
            try:
                teams=e["competitions"][0]["competitors"]
                codes=[t["team"]["abbreviation"].upper() for t in teams]
                start=utc(e["date"])
                pairs=["".join(codes),"".join(reversed(codes))]
                if (len(teams)==2 and len(set(codes))==2 and
                    0 <= (occurrence-start).total_seconds() <= 8*3600 and
                    any(code in ticker for code in pairs)):
                    matched.append(e)
            except (KeyError, TypeError, ValueError):
                continue
        if len(matched)!=1:
            raise ValueError("nba_margin_schedule_identity_or_time_conflict")
        e=matched[0]
        kind=e.get("season",{}).get("type") or e.get("seasonType",{}).get("type")
        if str(kind) != "2":
            raise ValueError("nba_margin_preseason_or_playoffs_unmodeled")
        comp=e["competitions"][0]["competitors"]
        home=next(c for c in comp if c["homeAway"]=="home")
        away=next(c for c in comp if c["homeAway"]=="away")
        season=utc(e["date"]).year+int(utc(e["date"]).month>=7)
        rows={}
        for team in (home,away):
            for year in (season-1,season):
                data=self.get(f"{ESPN}/basketball/nba/teams/{team['id']}/schedule",
                              {"season":year,"limit":250})
                for past in data.get("events",[]):
                    season_info=past.get("season") or data.get("season") or {}
                    typ=season_info.get("type") if isinstance(season_info,dict) else None
                    if str(typ)!="2":
                        continue
                    result=self.espn_result(past,"NBA")
                    if result is None:
                        continue
                    if utc(result["event_time"])+timedelta(hours=24)>=utc(e["date"]):
                        continue
                    result["season_type"]="regular"
                    key=result["event_id"]
                    if key in rows and rows[key]!=result:
                        raise ValueError("conflicting_espn_nba_completed_game")
                    rows[key]=result
        def aliases(team):
            info=team["team"]
            variants=_metadata_aliases(info) | {normalize(info.get("abbreviation",""))}
            # Kalshi uses well-known abbreviations in its NBA market labels
            # (e.g. "LA Lakers", "GS Warriors"). Never fuzzy-match teams.
            short_city={"LAL":"LA","LAC":"LA","GSW":"GS","NYK":"NY",
                        "NOP":"NO","SAS":"SA","OKC":"OKC"}.get(
                            str(info.get("abbreviation") or "").upper())
            nickname=normalize(info.get("name",""))
            if short_city and nickname:
                variants.add(normalize(short_city+" "+nickname))
            return sorted(variants)
        games=sorted(rows.values(),key=lambda g:g["event_time"])
        return {"event_id":str(e["id"]),"start":e["date"],"season_type":"regular",
                "home_id":str(home["id"]),"away_id":str(away["id"]),
                "home_name":home["team"]["displayName"],
                "away_name":away["team"]["displayName"],
                "home_aliases":aliases(home),"away_aliases":aliases(away),
                "rows":games,"source_urls":[f"{ESPN}/basketball/nba/scoreboard",
                                          f"{ESPN}/basketball/nba/teams/{home['id']}/schedule",
                                          f"{ESPN}/basketball/nba/teams/{away['id']}/schedule"],
                "missing":["confirmed_starters_and_injuries","rotation_and_minutes",
                           "rest_travel_and_game_script","prospective_margin_calibration"]}

    def basketball_context(self, market, sport, player):
        start = utc(market["occurrence_datetime"])
        league = sport.lower()
        day=nominal_day(market)
        events={}
        for candidate_day in (day,day+timedelta(days=1)):
            data=self.get(f"{ESPN}/basketball/{league}/scoreboard", {"dates":candidate_day.strftime("%Y%m%d"), "limit":500})
            events.update({str(e["id"]):e for e in data.get("events", [])})
        matches = []
        for e in events.values():
            teams = e["competitions"][0]["competitors"]
            codes = [t["team"]["abbreviation"] for t in teams]
            if len(codes) != 2:
                continue
            if 0 <= (start-utc(e["date"])).total_seconds() <= 8*3600 and any("".join(pair) in market.get("event_ticker", "") for pair in (codes, list(reversed(codes)))):
                matches.append(e)
        if len(matches) != 1:
            raise ValueError("basketball_schedule_identity_conflict")
        e = matches[0]
        kind = e.get("season", {}).get("type")
        if kind is None:
            kind = e.get("seasonType", {}).get("type")
        return {"event_id":str(e["id"]), "participant":player,"start":e["date"],
                "season_type":{1:"preseason",2:"regular",3:"playoffs"}.get(kind,"unknown"),
                "missing":["confirmed_minutes", "active_roster", "opponent_personnel"]}
