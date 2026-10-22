# Player prop history workspace

The Props tab's **Player history · recent games and descriptive hit rates** panel is opt-in: when a real catalog contract exists, users select it and explicitly request history; when no current contract exists, they can enter a player, supported stat family, target date and optional threshold for manual historical research. Manual research does not create a market quote or betting candidate. It never invents a market or fills an empty chart with simulated values.

## Source and supported market mappings

| Sport | Source | Mapped families | Fallback behavior |
|---|---|---|---|
| NBA / WNBA | SportsDataverse player box-score CSV releases | Points, rebounds, assists, three-pointers, points + rebounds + assists | If current-season history is empty, prior-season history may be shown with an explicit label. |
| NFL | nflverse player-week CSV plus schedule joins | Passing yards/TDs/attempts/completions/interceptions, rushing yards/attempts, rushing + receiving yards, receiving yards, receptions, player touchdowns | Current-season games are used when present; prior-season rows are used only when current-season history is empty and are explicitly labeled. |
| MLB | Official MLB Stats API | Hits, home runs, total bases, RBIs, runs, hits + runs + RBIs, pitcher strikeouts | If current-season game log is empty, prior-season game log may be shown with an explicit label. |
| NHL / Tennis | No general prop-history adapter in this panel yet | Unsupported here | Shows an unsupported/missing-data state. NHL goal-scorer research remains in its separate Scorer Lab pipeline. |

## Integrity rules

- For catalog-backed history, event date is parsed from the market/event ticker. Market close time is not substituted for the event date. Manual research requires an explicit target date from the user.
- Only games dated before the target event are eligible.
- The report shows actual observed values, opponent/home-away where available, season hit rate, last-five hit rate, home/away splits, sample denominators and the available date range.
- The table shows the latest 20 eligible games; summary hit rates use the complete eligible history returned by the selected source.
- Historical hit rates are descriptive, not next-event probabilities and not sufficient proof of betting value.
- Missing players, unsupported market mappings, unavailable sources and ambiguous joins fail closed.
- The panel does not imply injury, starting-role, projected-minutes, line-combination, goalie or matchup adjustments unless a separately identified source/model provides that evidence.
- Source retrieval time is shown separately from any upstream data publication time; the providers do not expose a single reliable universal freshness timestamp for every row.

The adapter uses public/free research datasets where available, with bounded provider caches already present in the source modules. Availability, schemas and provider terms can change; a source error results in a clear unavailable state rather than fabricated statistics.
