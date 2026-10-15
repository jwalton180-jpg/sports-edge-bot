# All-game, five-sport correlated parlay research (October 2026)

## Scope and truthfulness

**Sports:** every scanned matchup in NFL (including TNF, Sunday and MNF),
NCAAF, NBA, NHL, and MLB. Existing Tennis, WNBA and soccer lanes remain
unchanged; absence from this board does not imply they are unsupported
elsewhere in SportEdge.

**Model candidates:** only independent probability models, verified game
identity and kickoff, exact Kalshi ticker/side/ask, entry fees, original
execution qualification and injury/role policy may clear individual leg
review. Calendar day does not restrict NFL research to Thursdays. A game
with no verified candidate does not get a fabricated ticket.

**Correlation:** all selected legs in one physical event share exposure.
Different players depend on team opportunities/game script. Same-player
cross-stat legs additionally receive a shared-usage warning. This is
not a numeric correlation estimate. We do not automatically multiply
Goff passing yards and St. Brown receiving yards and call that the
joint probability of the ticket. Across games, independence is still
a hypothesis, not proven by these flags.

**Joint evaluator:** `evaluate_joint_scenarios` accepts 2–25 exact
side-matched contracts, plus 500–20,000 complete independent pregame
scenario draws with stable IDs and causal timestamps. It estimates the
frequency of all-leg success from the whole jointly simulated scenario
rather than multiplied marginals. Scenario marginals must approximately
agree with the individually frozen probability estimates. Missing legs,
missing kickoff, bad identities, duplicate markets, unsupported sports,
non-finite probabilities, post-kickoff information and market-derived
"simulations" fail closed. Cross-game tickets require *joint* multi-game
scenarios, not stitched pairwise correlations.

**No invented fair value:** direct exact combo contract identification,
exact same ticker/side constituents, current ask (≤60 seconds), and
its Kalshi entry fee are required before computing an *indicative*
joint model minus executable cost. A set of component asks never
substitutes for an actual Kalshi combination ask. This is still research
only; no fit, holdout or prospective accuracy for these joint draws is
claimed, and a quoted ask does not guarantee a fill.

**Cash-out:** `cashout_probability=None` until prospective exact combo
bid/depth paths and settlement-time exits are recorded and calibrated.
The independent single-leg probability is not the probability that
the user will have an opportunity to sell the combo profitably.

## UI

The existing Parlay Generator scan now includes a five-sport all-game
research expander. Each sport reports scanned event count, candidate
count, pregame individually eligible legs and blocked evidence. A separate
same-event risk table identifies dependency classes and **UNKNOWN**
combined joint probability. The previously implemented TNF spotlight
remains intact as a special additional view on Thursdays.

The board derives only from the existing candidate scan. It does not
promise full calendar coverage if that scan did not find models/markets.
Do not mistake individually qualified legs for a simultaneously
qualified same-game combo; combo fair P, pricing, correlated EV and
cash-out remain independent gates.

## Required promotion evidence (still missing)

- Historical datasets with exact market rules and synchronized game
  player/box-score events across **each** sport and threshold family.
- Independent causal scenario calibrations and proper out-of-sample
  model selection; no Kalshi quotes in model feature generation.
- Unseen historical holdouts by season/date, with same-game
  joint Brier/log scores and marginal reconciliation checks.
- Frozen prospective predictions and actual exact-combo quote depth,
  fees and sizing, including missing opportunities and losing tickets.
- Explicit approval before promoting a research estimate to a
  validated recommendation. No 24/7 runtime, Railway or live trading.

## Safety invariants

Never use social creators' headline win rates or promotional payouts
as calibrated probability evidence. Do not manufacture slate picks
from unsupported games; no order submission, live trading, or changes
to BTC, Late Caller, AutoProfit or other protected projects.
