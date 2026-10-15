# Five-sport public-score joint research and prospective workflow

RESEARCH ONLY: NFL, NCAAF, NBA, NHL, MLB. No account access, orders,
Railway or background processes.

## Source ingestion and limitations

The historical_score collector fetches public ESPN league scoreboards,
strictly completed games, each with date, event identity, both team
identities, final scores, URL provenance and source response SHA256.
ESPN was checked live October 15, 2026 and returned historical completed
games in all five sports. ESPN is not a governing league, and these
full-game scores DO NOT certify Kalshi settlement of specific contracts.
Bounded 1-45 day downloads per requested sport; explicit merge preserving
provenance for longer historical windows. Future scoreboards, unrecognized
sources, incomplete games and conflicting IDs are rejected.

Commands from a checkout with project requirements installed:

Collect 45 or fewer UTC historical calendar days for each of the five
sports, for example:
  python -m scripts.five_sport_joint_research collect --sport NFL --start 2025-10-01 --end 2025-11-14 --output research-private/nfl-oct.json

Combine separately gathered sport archives:
  python -m scripts.five_sport_joint_research merge --archive research-private/nfl-oct.json --archive research-private/ncaaf-oct.json --output research-private/combined.json

Evaluate independent time-cutoff joint score probabilities:
  python -m scripts.five_sport_joint_research backtest --archive research-private/combined.json --holdout-start 2025-11-01T00:00:00Z --output research-private/holdout.json

No unknown or unsupported sport/market gets fabricated selections.

## Correlated joint model

Causal team offense/defense averages and sampled PAIRED historical
home/away score residuals generate whole game score pairs.
Only past dated completed scores can inform a new event. Two same-game
score markets are evaluated on each draw, not multiplied as independent.
Supported at this stage: home/away winners, full-game scoring totals,
home or away team totals with half-point binary thresholds.
PLAYER PROPS ARE NOT MODELED and no cash-out probabilities are produced.
The model is uncalibrated and all output is RESEARCH ONLY.

Chronological historical holdout reports the Brier score of
home-win AND game-over, alongside a product-of-marginals independence
benchmark and an earlier-game league base-rate benchmark. Predictions
never see same-day final scores. Fewer than 30 genuinely holdout
observations returns INSUFFICIENT_RETROSPECTIVE_HOLDOUT.
The model still requires chronological/prospective calibration.

## Immutable on-demand prospective research

Fetch upcoming public scheduled game identities:
  python -m scripts.five_sport_joint_research fixtures --sport NFL --day 2026-10-18 --output research-private/nfl-fixtures.json
Freeze model predictions before kickoff with SHA256 and no overwrite:
  python -m scripts.five_sport_joint_research freeze --archive research-private/combined.json --fixtures research-private/nfl-fixtures.json --output research-private/frozen-nfl.json
After the games, collect their completed score archive and grade it:
  python -m scripts.five_sport_joint_research grade --frozen research-private/frozen-nfl.json --final-archive research-private/final-nfl.json --output research-private/graded-nfl.json

Frozen call identity and SHA256 are verified. The grade has a separate
file and cannot change the pregame prediction. Missing game outcomes
remain pending/unscorable, not winning records. There is no auto-worker.

## Streamlit

The five-sport all-game Parlay Generator research section accepts
ONLY a user-selected sanitized SCORE-BACKTEST JSON for retrospective
Brier display. Never upload personal Kalshi account history, API key,
private fill data or credentials. Uploading to a hosted service
transfers the uploaded report to that host.

## Still required before actual combined-market recommendations

Sport-specific historical player statistical and lineup datasets,
pregame causal player usage models, true contemporaneous exact
Kalshi combined-market asks+fees, joint model chronological validation,
and prospective immutable calls with later combo bids/depth.
No promotional sports-tipster figures or arbitrary correlation
coefficients count as edge evidence. No model here is live-qualified.


## Five-sport combined season-specific report

The source window differs by league's actual season dates. A global
historical holdout cutoff is inappropriate for NBA, NHL and MLB when
compared with NFL and NCAAF. For a single combined UI report, prepare
a local JSON object with exactly five keys NFL, NCAAF, NBA, NHL, MLB,
each a timezone-aware cutoff string, for example:

  {"NFL":"2025-10-05T00:00:00Z",
   "NCAAF":"2025-09-25T00:00:00Z",
   "NBA":"2025-11-12T00:00:00Z",
   "NHL":"2025-10-30T00:00:00Z",
   "MLB":"2025-08-24T00:00:00Z"}

Then use:
  python -m scripts.five_sport_joint_research backtest --archive research-private/combined.json --cutoffs-file research-private/sport-cutoffs.json --output research-private/five-sport-report.json

The model's on-demand future call path also requires completed, verified
sport history from the PRIOR 45 calendar days. A previous season-only
archive returns STALE_SPORT_HISTORY_FOR_FUTURE_PREDICTION and freezes
an explicitly UNSCORABLE probability, not a misleading live call.

## Initial real-source evidence

On October 15, 2026 a bounded 45-day archive was collected on demand
for each of the five sports (225 source daily pages, 1,762 unique
completed games, 742 retrospective holdout game predictions). Exact
source and held-out report SHA256 digests, per-sport Brier and
reproduction metadata are in
docs/five_sport_45day_actual_score_holdouts_20261015.json.
This initial experiment shows NO statistically established predictive
or Kalshi-price advantage and remains strictly research-only.
