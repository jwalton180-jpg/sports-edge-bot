"""NBA/WNBA research ladders with minutes sensitivity; preseason fails closed."""
from dataclasses import replace
from sports_edge.models.sleeper_counts import estimate_counts, unavailable, utc
from sports_edge.data.basketball_prop_data import player_history, stat_value, minutes_value

FIELDS = {"Points":"points", "Rebounds":"rebounds", "Assists":"assists",
          "Three-Pointers":"threes", "Points + Rebounds + Assists":"pra"}


def basketball_sleeper(*, sport, player, family, threshold, before_time, season_type):
    model = f"{sport.lower()}_{family.lower().replace(' ', '_')}_minutes_v1"
    if sport not in {"NBA", "WNBA"} or family not in FIELDS:
        return unavailable(model, "unsupported_market")
    if season_type not in {"regular", "playoffs"}:
        return unavailable(model, "preseason_or_unknown_rotation_requires_separate_model")
    cutoff = utc(before_time)
    season = cutoff.year + int(sport == "NBA" and cutoff.month >= 7)
    history = []
    for year in (season-1, season):
        history.extend(player_history(sport, player, year, cutoff.date().isoformat()))
    rows = []
    metric = FIELDS[family]
    for r in history:
        v = stat_value(r, metric)
        rows.append({"source":"sportsdataverse", "player_id":str(r.get("athlete_id") or player),
                     "game_id":str(r.get("game_id") or ""), "date":str(r.get("game_date", ""))[:10],
                     metric:int(v) if v is not None and v == int(v) else None,
                     "minutes":minutes_value(r)})
    result = estimate_counts(rows, field=metric, threshold=threshold, before_time=before_time,
                             source="sportsdataverse", model=model, prior_mean={"points":15,"rebounds":5,"assists":3,"threes":1.5,"pra":23}[metric],
                             exposure_field="minutes")
    return replace(result, reasons=result.reasons+("confirmed_minutes_injuries_and_blowout_scenario_missing",))
