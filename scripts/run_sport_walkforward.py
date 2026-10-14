"""Run source-backed retrospective walk-forward ON DEMAND, never live or on a timer.

Usage: python scripts/run_sport_walkforward.py --output-dir /tmp/sportsedge-evidence
Requires public ESPN/NHL endpoints; aborts on missing provider responses.
"""
from __future__ import annotations
import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from sports_edge.data.historical_sport_evidence import completed_team_games, nhl_player_goal_log
from sports_edge.models.sport_walkforward import team_walkforward, nhl_player_walkforward, calibration_report
from sports_edge.models.sport_calibration import chronological_calibration

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
    sources = {
        "NCAAF": (date(2025, 8, 20), date(2025, 12, 20)),
        "Soccer": (date(2024, 8, 1), date(2025, 5, 31)),
    }
    for sport, (start, end) in sources.items():
        games = collect_team(sport, start, end)
        data[sport] = {"events_fetched": len(games), "source": "ESPN",
                       "period": [start.isoformat(), end.isoformat()]}
        predictions.extend(team_walkforward(games, sport=sport))
    if nhl_player_ids:
        total = 0
        for player_id in nhl_player_ids:
            logs = nhl_player_goal_log(player_id, 20242025)
            total += len(logs)
            predictions.extend(nhl_player_walkforward(list(logs)))
        data["NHL"] = {"player_count": len(nhl_player_ids), "game_logs": total,
                       "source": "NHL official API", "sampling": "explicit_player_id_sample",
                       "representative": False}
    else:
        data["NHL"] = {"source": "NHL official API", "status": "missing_full_roster_sample",
                       "representative": False}
    calibrations = {sport: chronological_calibration([r for r in predictions if r["sport"] == sport])
                    for sport in ("NHL", "NCAAF", "Soccer")}
    result = {"method": "retrospective_strict_walkforward",
              "live_recommendation_eligible": False,
              "data_coverage": data, "metrics": calibration_report(predictions),
              "temporal_calibration": calibrations}
    (dest / "research_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (dest / "historical_predictions.jsonl").open("w") as fh:
        for row in predictions:
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
