"""Strict historical walkforward for the NHL player prop menu."""
from __future__ import annotations
from sports_edge.models.nhl_multimarket import nhl_market_menu

def nhl_prop_walkforward(log:list[dict])->list[dict]:
    ordered=sorted(log,key=lambda r:(r["date"],r["game_id"]))
    seen=set()
    out=[]
    for row in ordered:
        key=(row["player_id"],row["game_id"])
        if key in seen: raise ValueError("duplicate official game log")
        seen.add(key)
        prior=[r for r in ordered if r["date"]<row["date"]]
        if not prior: continue
        for proj in nhl_market_menu(prior):
            if proj.probability is None: continue
            # A missing outcome cannot be silently counted as zero.
            field=proj.outcome.split(" player ",1)[1]
            threshold=int(proj.outcome.split("+",1)[0])
            if type(row.get(field)) is not int: continue
            out.append({"sport":"NHL","event_id":str(row["game_id"])+":"+str(row["player_id"]),
                        "event_time":row["date"]+"T12:00:00Z",
                        "predicted_at":prior[-1]["date"]+"T23:59:59Z",
                        "source_id":"NHL:"+str(row["game_id"]),
                        "model_version":proj.model,"outcome":f"player_{field}_{threshold}_plus",
                        "probability":proj.probability,"won":row[field]>=threshold,
                        "evaluation_kind":"historical_walkforward","status":"RESEARCH_ONLY"})
    return out
