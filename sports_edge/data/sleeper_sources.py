"""Bounded on-demand public sports adapters. No account/order endpoints.

Only official NHL records and ESPN finalized results enter these models.
An unresolvable participant or schedule discrepancy is a rejected join.
"""
from datetime import timedelta
from datetime import datetime, timezone
import unicodedata
import re
import requests


def kalshi_event_day(ticker):
    """Extract the event date printed in the Kalshi event ticker, not its expiry."""
    match = re.search(r"-(\\d{2}[A-Z]{3}\\d{2})(?:[A-Z]|-|$)", str(ticker))
    if not match:
        raise ValueError("kalshi_event_date_missing")
    return datetime.strptime(match.group(1), "%y%b%d").date()

from sports_edge.models.sleeper_counts import utc
from sports_edge.models.game_scope import normalize

NHL = "https://api-web.nhle.com/v1"
ESPN = "https://site.api.espn.com/apis/site/v2/sports"


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
        expiry = utc(market["occurrence_datetime"])
        ticker = str(market.get("event_ticker", ""))
        game_day = kalshi_event_day(ticker)
        # occurrence_datetime is commonly the expected game-end/expiry, NOT puck drop.
        data = self.get(f"{NHL}/schedule/{game_day.isoformat()}")
        matches = []
        for week in data.get("gameWeek", []):
            for g in week.get("games", []):
                if g.get("gameType") != 2 or g.get("gameScheduleState") != "OK":
                    continue
                codes = [g[k]["abbrev"] for k in ("awayTeam", "homeTeam")]
                pairs = ["".join(codes), "".join(reversed(codes))]
                # Kalshi's TB code differs from NHL's TBL.
                pairs += [p.replace("TBL", "TB") for p in pairs]
                game_start = utc(g["startTimeUTC"])
                # Require exact NHL schedule date and plausible post-start market
                # occurrence. Never treat the latter as the pregame cutoff.
                delay = (expiry-game_start).total_seconds()
                if (any(p in ticker for p in pairs)
                    and g.get("gameDate") == game_day.isoformat()
                    and 0 <= delay <= 8*3600):
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
        expiry = utc(market["occurrence_datetime"])
        game_day = kalshi_event_day(market.get("event_ticker", ""))
        # ESPN scoreboard dates use the original listed event day, not the
        # exchange's later settlement/occurrence clock.
        data = self.get(f"{ESPN}/{path}/scoreboard", {"dates": game_day.strftime("%Y%m%d"), "limit": 500})
        rules = str(market.get("rules_primary", ""))
        candidates = []
        for e in data.get("events", []):
            delay=(expiry-utc(e["date"])).total_seconds()
            if not 0 <= delay <= 8*3600:
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
        return {"event_id": str(e["id"]), "start": e["date"], "home_id": str(home["id"]),
                "away_id": str(away["id"]), "home_name": home["team"]["displayName"],
                "away_name": away["team"]["displayName"], "rows": sorted(collected.values(), key=lambda r:r["event_time"]),
                "missing": ["confirmed_lineups", "weather", "injuries"], "deployment_verified": False}

    def basketball_context(self, market, sport, player):
        start = utc(market["occurrence_datetime"])
        league = sport.lower()
        data = self.get(f"{ESPN}/basketball/{league}/scoreboard", {"dates":start.strftime("%Y%m%d"), "limit":500})
        matches = []
        for e in data.get("events", []):
            teams = e["competitions"][0]["competitors"]
            codes = [t["team"]["abbreviation"] for t in teams]
            if len(codes) != 2:
                continue
            if abs((utc(e["date"])-start).total_seconds()) <= 300 and any("".join(pair) in market.get("event_ticker", "") for pair in (codes, list(reversed(codes)))):
                matches.append(e)
        if len(matches) != 1:
            raise ValueError("basketball_schedule_identity_conflict")
        e = matches[0]
        kind = e.get("season", {}).get("type")
        if kind is None:
            kind = e.get("seasonType", {}).get("type")
        return {"event_id":str(e["id"]), "participant":player,
                "season_type":{1:"preseason",2:"regular",3:"playoffs"}.get(kind,"unknown"),
                "missing":["confirmed_minutes", "active_roster", "opponent_personnel"]}
