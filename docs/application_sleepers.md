# Application-wide sleeper evidence

The existing on-demand Streamlit application exposes the same **Sleeper value • every sport and both sides** panel on every navigation screen. Select a sport and Hawaiʻi date, run the scan, then use **Export sleeper evidence (JSON)** or **Export all recorded scans and settlements (JSONL)**. These are public-market evidence exports, not account or trading interfaces. No new externally hosted API address is claimed.

## Coverage and model status

| Sport | Independent projections routed to evidence scanning | Important limits |
| --- | --- | --- |
| NHL | Goals, shots, assists, points, goalie saves, team goals | Real aliases include KXNHLGOAL/ANYGOAL, AST, PTS, SAVE/SAVES. Shots are modeled if a matching contract is actually listed; a model is not proof of a listed series. Exact schedule joins, goalie starts, line and power-play confirmation matter. |
| NFL | Existing passing/rushing/receiving/receptions/attempts plus scored touchdown counts, including 2+ | Scored TDs include rushing and receiving, exclude passing TDs. Dispersed tails and empirical-tail sensitivity replace simple extrapolation for timestamped histories. Active status, red-zone roles, OL and opponent personnel remain explicit missing inputs. |
| NCAA football | Opponent-adjusted winner research and mandatory upset review | Complete QB, OL, offense, underdog matchup, recent competitiveness, venue/travel, coaching and weather evidence is required. Missing reviews and multiple material red flags block Best Available. The bounded team-history join is not a complete national opponent graph. |
| Soccer | EPL BTTS, team goals and alternate match totals | Only EPL histories are modeled by this challenger. Regulation plus stoppage-time contracts must exclude extra time and penalties. Other leagues and player goals stay unsupported until sourced and validated. |
| NBA/WNBA | Points, rebounds, assists, threes and PRA count research; existing WNBA models | Minutes sensitivity is explicit. Unknown/preseason rotations reject regular-season projections. Confirmed availability, minutes and opponent personnel are still required for qualification. |
| MLB | Existing hits, home runs, total bases, RBIs, H+R+RBI, strikeouts and game-line models | Reused independent projections retain original warnings and require exact rules, executable quotes and scoped validation. |
| Tennis | Existing winner and supported games-total models | Existing lower-tour and live reversal/survivability code and notification workflows are preserved. This pregame evidence scanner rejects live matches. No new live-stat model is implied. |

**All new challengers remain unvalidated research.** Positive model-price gaps are not recommendations. App-produced candidates pass an execution audit before ticket assessment; absent scoped prospective reviews they remain research rather than borrowing legacy confidence labels as proof of calibration. This can reduce or empty recommendation tickets, with the actual rejection reasons retained. Existing projection functions, navigation and notification infrastructure remain present.

## Price and settlement integrity

- Public Kalshi market and series reads only. Actual asks and available size are required; a last trade or bid cannot substitute for an ask.
- Quote receipt time and source update time are separate. A stale `updated_time` does not by itself mean a freshly fetched unchanged quote is stale.
- Taker costs use verified quadratic series metadata and `ceil(0.07 * multiplier * quantity * price * (1-price) * 100)/100`. Break-even is ask plus the per-contract entry fee. Unknown/flat fee schedules are unpriced. The displayed size and quote are not a promise of a fill.
- Count floors must be half-integers and agree with the primary rule's threshold. Variable-payoff ladders/escalators are not treated as $1 binary contracts.
- NHL team histories reproduce the contract's OT/shootout treatment. The observed KXNHLTEAMTOTAL rule credits the shootout winner with one additional goal; models for that rule use official final scores including the bonus.
- Exact participants, nominal event dates and independently verified UTC kickoff times are required. Some observed Kalshi occurrence fields described a later time, so the pipeline preserves that field separately and uses the independent schedule for kickoff. Started games require a live model. The Hawaiʻi filter verifies the actual start date after resolving nearby inventory; unverified starts keep coverage incomplete. Settlement/expiration dates cannot replace kickoff.
- Main overview discovery is bounded. The full ticket catalog opts into metadata discovery and exhausts cursors; scan caps, source failures and unknown starts are exposed. Incomplete coverage is never an exhaustive “no opportunities” claim.

## Tickets

**Qualified Value** and **Sleeper Value** rank positive net value without the favorite-probability floor or longshot price band. Best Available preserves the existing higher-probability core. NFL **2+ Touchdowns** is a dedicated focus. The multi-game leg target is user-specified with no arbitrary maximum; the established single-game four-leg maximum remains. The builder rejects opposite sides of a contract, redundant same-player thresholds and incompatible winner selections. Joint products are explicitly independence benchmarks with dependence-unrestricted probability bounds. Payout is **illustrative** unless an actual executable combo quote and costs exist. No create-combo, RFQ, wager or order endpoint is invoked.

## Prospective records and promotion

Every scan retains both sides, rejections, missing inputs, versions, quotes and timestamped provenance. SQLite records cannot be updated or deleted through the ledger; official settlements are appended separately. Local hosted storage may reset: full JSONL export is available after each scan. The app warns if persistence fails. No durable cloud ledger is falsely claimed.

**Grade saved predictions from official Kalshi results** performs bounded GET-only grading of settled YES/NO markets. Open, void, fair-price and ambiguous outcomes cannot be invented as binary wins. Metrics group by sport, family, threshold, model, side and price band. No realized returns are reported without actual execution evidence, and repeat scans must be deduplicated before statistical promotion.

Promotion requires immutable causal future calls, one model/sport/outcome scope, sufficient dates and both outcome classes, calibration and a date-clustered baseline-improvement bound. Retrospective replay cannot promote. A reviewed, expiring scope must be signed and bound to the prospective payload's hash and authenticated source review. The application optionally reads `research/reviewed_model_promotions.json` and an operator-held `SPORTSEDGE_PROMOTION_REVIEW_KEY`; no approved reviews are shipped and the scanner cannot sign one. The signing key is never exported. NCAA favorites additionally require a fresh complete vulnerability review.

## Validation commands

```bash
python -m pytest -q
python -m compileall -q sports_edge scripts app.py launch_on_demand.py
python -m scripts.run_sleeper_replay --output-dir research-output --sample-size 4
python -m scripts.run_sleeper_replay --output-dir research-output --sample-size 0
```

The first replay is labeled a nonrepresentative source-integration smoke; the second enumerates the entire official historical NHL population plus eligible NFL player histories. Calibration uses chronological training/inner-validation/holdout splits. Both are retrospective and omit live market-price performance. Source errors prevent a passing replay rather than shrinking the sample silently.

Primary references: [Kalshi public orderbook semantics](https://docs.kalshi.com/getting_started/orderbook_responses), [Kalshi fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf), [NHL API](https://api-web.nhle.com/v1/), [NHL official statistics](https://api.nhle.com/stats/rest/en/), [nflverse source releases](https://github.com/nflverse/nflverse-data/releases), [ESPN public scoreboards](https://site.api.espn.com/apis/site/v2/sports/), [sportsdataverse releases](https://github.com/sportsdataverse/sportsdataverse-data/releases).

No Railway, paid service, scheduled continuous runtime, shared Ubuntu host change, trading action or unrelated project modification is introduced.

### Exact NFL scorer scope

The offensive-only v2 replay is retained as a benchmark, rather than rewritten. The current v3 scorer model uses nflverse play-by-play `td_player_id` credits for each actual touchdown, covering offensive, return, defensive and fumble scores exactly once and excluding the passer for a passing TD. Cancelled plays and duplicate or missing credits fail closed. Missing games cannot be assumed to have zero touchdowns. The complete-population replay separately reports v2 and v3. Source payload hashes and retrieval times accompany v3 inputs. None of this automatically promotes v3 to recommendations.

Authenticated statistical review is a separate usability signal from heuristic confidence; approved scopes can pass the model-usability guard without inventing a new confidence percentage. Fresh market, kickoff, availability and contract-input guards still apply.

## First-outcome prospective review export (not live promotion)

After running a sleeper scan, use **Export prospective review dataset (JSON)**. It canonically retains the earliest YES-side model forecast per contract, model and outcome; later rescans and the complementary NO side cannot inflate samples. Verified binary official results can be attached on a later session using the existing bounded GET-only grading. Void and fair-price resolutions never count as ordinary wins. The output includes source timestamps, prediction identities, indicative prices, missing evidence and a breakdown by sport.

**This export is deliberately unattested.** Neither the downloaded JSON file nor ephemeral local SQLite proves the prediction was externally timestamped before the game. This prevents retrospective editing from being confused with a prospective holdout. An independently time-stamped remote archive, a preregistered baseline, authenticated source review, and sufficient future official results are still required before the signed model-promotion gate can pass. No 24/7 process or paid hosting is introduced.


## NBA 19+ point blowout focus (October 14)
The on-demand sleeper scan has a dedicated **NBA 19+ point victory • research** surface for the actual listed Kalshi `KXNBASPREAD` full-game over-18.5 contract. Winning outright is NOT confused with winning by 19+. Exact half-point strike, selected team and rule text, independently confirmed game start, regular-season-only historical samples, fresh YES/NO asks, fees and sensitivity are required. Missing current availability, schedule identity, preseason or unsupported rules fail closed. The model uses only past official completed ESPN games; no live market price enters the estimate. It remains `UNVALIDATED_RESEARCH` until separately archived prospective forecasts and official Kalshi settlements show a credible edge over the executable price baseline.

The pre-registered historical benchmark was evaluated on 2025-26 completed games with chronological 2024-25 history: 1,231 games and 2,462 team-side outcomes across 167 dates. Model Brier 0.101666 vs historical unconditional home/away baseline 0.110137; date-clustered Brier advantage bootstrap [0.005187, 0.011595]. There is NO validated Kalshi edge or guarantee. Apparent 87.4% sign accuracy is almost identical to always predicting no 19+ blowout (87.3%), so it cannot serve as a confidence badge. Archived artifacts include score rows, source SHA hashes, per-prediction probabilities and calibration report. Future immutable snapshots must be scored before promotion.
