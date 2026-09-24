# SPORTS EDGE build status — v0.1.7

## Verified local baseline
- 29 automated tests passing.
- Python package compiles cleanly.
- Streamlit boot was not executable inside the current offline build container because Streamlit is not preinstalled and the container has no package-network access; `requirements.txt` pins the runtime dependencies for a normal connected machine.

## Completed now
- Premium dark Streamlit terminal shell.
- Read-only Kalshi public REST adapter on the current production host.
- Kalshi sports-filter hooks plus current live-data/event/milestone/game-stats endpoints.
- Corrected game-stats route to `/live_data/milestone/{milestone_id}/game_stats`.
- Authenticated read-only Kalshi WebSocket transport with initial snapshots, heartbeat handling via the client transport, reconnect/backoff, sequence tracking and fail-closed gap detection. No order methods exist.
- MLB schedule/boxscore adapter and corrected live-feed route via Stats API v1.1.
- NFL live cross-check adapter; Kalshi live game data remains the preferred source where available.
- Corrected The Odds API v4 host/path; event-level odds/prop hooks added.
- Vig removal, fair-price math, Kalshi expected value, edge grading and quality gates.
- Freshness-aware live-source fusion and disagreement detection.
- Dedicated 3%–30% Underdog Radar with asymmetric-value gating and stale/data-quality rejection.
- Tennis overall/surface Elo and matchup primitives; ATP/WTA/Challenger/ITF kept as separate competition contexts.
- MLB game + at-least-one-hit probability primitives.
- NFL game + player-prop distribution primitives.
- Gaussian-copula correlated parlay engine and independence baseline.
- Sportsbook no-vig consensus estimator with recency weighting and stale-line exclusion.
- Public-tipster evidence-quality primitives that penalize tiny samples and require verifiable records/CLV when available.
- Internal prospective paper-study logger only; not surfaced as a user feature.

## Integrity rules
- Cheap price is never called value unless an independent/calibrated fair probability exists.
- Stale, conflicting or low-quality data fail closed.
- No real-order submission code.
- No “guaranteed” parlays and no manufactured 80% claim.
- Historical validation must be chronological/walk-forward, with Brier/log loss/calibration/CLV/ROI and coverage reported together.

## Overnight priorities
1. Resolve current Kalshi sports markets to event + milestone IDs and fuse their live state/play-by-play.
2. Build chronological tennis training pipeline: surface Elo, serve/return, opponent quality, fatigue/travel, retirement handling, competition-level transitions and separate ITF calibration.
3. Build MLB feature store: Statcast contact/pitch quality, platoon/pitch mix, starter/bullpen, park/weather, confirmed lineups; model hits/HR/K/outs and game lines.
4. Build NFL feature store: EPA/CPOE/success, neutral pace, OL/DL, pressure/blitz/coverage, usage/routes/snaps, red-zone role, actives/injuries, weather and game script; model sides/totals/player props.
5. Add sportsbook line-history/closing-line capture and cross-book disagreement diagnostics.
6. Build parlay candidate search with sport-specific correlation constraints for High-Confidence and 5+ leg Longshot modes.
7. Add calibration and prospective paper-study dashboards to research artifacts only, not the user app.
