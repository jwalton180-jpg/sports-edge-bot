"""Complete scorer counts from nflverse's credited touchdown player IDs.

Passing TDs credit the receiver/scorer, not the passer. Return/defensive/fumble
scores are captured once through the same credited scorer, avoiding overlaps in
weekly defensive/special-teams summary components. GET-only, bounded caches.
"""
import csv
import gzip
import hashlib
import io
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import requests
from sports_edge.models.sleeper_counts import utc

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"

@dataclass(frozen=True)
class ScoringCredits:
    counts: dict
    games: frozenset
    provenance: dict

def normalize_scoring_credits(rows):
    counts=Counter(); games=set(); seen=set()
    for r in rows:
        if r.get("season_type")!="REG":
            continue
        game=str(r.get("game_id") or "")
        if not game:
            raise ValueError("game identity missing")
        games.add(game)
        try:
            scored=float(r.get("touchdown") or 0)
        except (ValueError,TypeError):
            raise ValueError("touchdown indicator malformed")
        if scored not in {0.,1.}:
            raise ValueError("invalid touchdown indicator")
        if scored!=1 or r.get("play_type")=="no_play":
            continue
        player=str(r.get("td_player_id") or "")
        play=str(r.get("play_id") or "")
        key=(game,play)
        if not player or player in {"NA","nan","NaN"} or not play or key in seen:
            raise ValueError("missing or duplicate credited touchdown")
        seen.add(key)
        counts[(game,player)]+=1
    return dict(counts),frozenset(games)

@lru_cache(maxsize=4)
def scoring_credits(season):
    response=requests.get(URL.format(season=int(season)),timeout=30,
                          headers={"User-Agent":"SportEdge-ReadOnly/3.0"})
    response.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as stream:
        reader=csv.DictReader(io.TextIOWrapper(stream))
        required={"season_type","game_id","touchdown","td_player_id","play_id","play_type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("credited scorer schema incomplete")
        counts,games=normalize_scoring_credits(reader)
    return ScoringCredits(counts,games,{"url":URL.format(season=season),
        "sha256":hashlib.sha256(response.content).hexdigest(),
        "retrieved_at":datetime.now(timezone.utc).isoformat()})

def attach_scored_totals(rows, *, now=None):
    from sports_edge.data.nfl_prop_data import schedule_rows
    from zoneinfo import ZoneInfo
    current=now or datetime.now(timezone.utc)
    schedule={str(g.get("game_id")):g for g in schedule_rows() if g.get("game_type")=="REG"}
    out=[]; provenance={}
    for row in rows:
        game=str(row.get("game_id") or "")
        player=str(row.get("player_id") or "")
        g=schedule.get(game)
        if not game or not player or not g or not g.get("gametime"):
            raise ValueError("scorer schedule identity missing")
        kickoff=datetime.fromisoformat(str(g["gameday"])+"T"+str(g["gametime"])).replace(tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)
        if kickoff+timedelta(hours=8)>=current:
            continue
        season=int(g["season"])
        credits=scoring_credits(season)
        if game not in credits.games:
            raise ValueError("completed game absent from scoring archive")
        count=credits.counts.get((game,player),0)
        out.append(dict(row,source="nflverse",event_time=kickoff.isoformat(),all_scored_tds=count))
        provenance[season]=credits.provenance
    return out,list(provenance.values())
