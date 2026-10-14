"""Run source-backed retrospective walk-forward ON DEMAND, never live or on a timer.

Usage: python scripts/run_sport_walkforward.py --output-dir /tmp/sportsedge-evidence
Requires public ESPN/NHL endpoints; aborts on missing provider responses.
"""
from __future__ import annotations
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

from sports_edge.data.historical_sport_evidence import completed_team_games, nhl_player_goal_log, nhl_complete_skater_ids
from sports_edge.models.sport_walkforward import team_walkforward, nhl_player_walkforward, calibration_report
from sports_edge.models.sport_calibration import chronological_calibration
from sports_edge.models.challenger_evaluation import challenger_team_rows, challenger_nhl_rows, matched_holdout

def days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)

def collect_team(sport: str, start: date, end: date):
    out = []
    for day in days(start, end):
        # ESPN covers weeknight NCAA football as well as weekend contests.
        out.extend(completed_team_games(sport, day.strftime("%Y%m%d")))
    return out

def run(output_dir: str, nhl_player_ids: list[int] | None = None):
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    data = {}
    predictions = []
    challengers = []
    sources = {
        "NCAAF": (date(2024, 8, 15), date(2025, 12, 20)),
        "Soccer": (date(2023, 8, 1), date(2025, 5, 31)),
    }
    for sport, (start, end) in sources.items():
        games = collect_team(sport, start, end)
        data[sport] = {"events_fetched": len(games), "source": "ESPN",
                       "period": [start.isoformat(), end.isoformat()]}
        predictions.extend(team_walkforward(games, sport=sport))
        challengers.extend(challenger_team_rows(games, sport))
    all_ids = nhl_complete_skater_ids(20242025)
    selected = nhl_player_ids if nhl_player_ids else list(all_ids)
    if not set(selected).issubset(set(all_ids)) or len(set(selected)) != len(selected):
        raise ValueError("NHL IDs must be unique members of official season roster")
    total = 0
    # Bounded concurrency; all failures abort the report rather than bias the sample.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(nhl_player_goal_log, player_id, 20242025): player_id
                   for player_id in selected}
        for future in as_completed(futures):
            logs = future.result()
            total += len(logs)
            predictions.extend(nhl_player_walkforward(list(logs)))
            challengers.extend(challenger_nhl_rows(list(logs)))
    data["NHL"] = {"player_count": len(selected), "official_roster_count": len(all_ids),
                   "game_logs": total, "source": "NHL official API",
                   "sampling": "full_roster" if len(selected) == len(all_ids) else "explicit_subset",
                   "representative": len(selected) == len(all_ids)}
    calibrations = {sport: chronological_calibration([r for r in predictions if r["sport"] == sport])
                    for sport in ("NHL", "NCAAF", "Soccer")}
    comparisons = {sport: matched_holdout([r for r in predictions if r["sport"]==sport],
                                            [r for r in challengers if r["sport"]==sport])
                   for sport in ("NHL","NCAAF","Soccer")}
    result = {"method": "retrospective_strict_walkforward",
              "live_recommendation_eligible": False,
              "data_coverage": data, "metrics": calibration_report(predictions),
              "temporal_calibration": calibrations,
              "challenger_matched_holdout": comparisons}
    (dest / "research_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (dest / "historical_predictions.jsonl").open("w") as fh:
        for row in predictions:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with (dest / "historical_challenger_predictions.jsonl").open("w") as fh:
        for row in challengers:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nhl-player-ids", default="",
                        help="Comma-separated verified NHL player IDs. Sample is never called representative.")
    args = parser.parse_args()
    ids = [int(v) for v in args.nhl_player_ids.split(",") if v.strip()]
    print(json.dumps(run(args.output_dir, ids), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
