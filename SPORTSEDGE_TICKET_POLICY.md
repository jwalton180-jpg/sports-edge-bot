# SportEdge Ticket Construction Policy

This policy applies to every supported sport and to both pre-event and live ticket construction.

1. Start with the strongest high-involvement opportunities before niche props. Primary players, primary team outcomes, and repeatable opportunity come first.
2. Represent both sides of an event when qualified opportunities exist; do not force balance when the evidence is one-sided.
3. Role/opportunity stability outranks recent box-score noise.
4. Prefer volume before variance.
5. Prefer survivable thresholds when the model and market offer them.
6. Never add a weak leg merely to reach a payout target. Build the strongest core, then increase risk gradually.
7. Limit narrow game-script dependence; normally no more than three legs from one physical event without overwhelming evidence.
8. Use correlation deliberately and disclose it.
9. Kalshi is the tradable universe. Every leg requires an exact current contract and price, then SportEdge fair probability is compared with Kalshi implied probability.
10. Sportsbooks are secondary evidence for price, movement, disagreement, and news reaction; they never replace the independent sport model.
11. Public/social analysis is corroboration only, never the primary model.
12. Every leg receives an involvement assessment: HIGH, MEDIUM, or LOW.
13. Pre-event availability/role checks are required before calling a ticket ready.
14. Material role uncertainty blocks a recommendation.
15. Every ticket includes a failure map: strongest leg, weakest leg, highest-variance leg, correlation risk, and primary failure scenario.
16. Risk language must match the math; longshots are never described as safe.
17. Final ticket output exposes exact contract, price, fair probability, edge, involvement, variance, role status, combined probability benchmark, current payout multiplier, value ratio, correlation, and failure map.
18. Live markets must use current game state; stale pregame probabilities cannot be promoted as live fair values after material state changes.
19. Postmortems classify process outcomes specifically: variance, role miss, matchup miss, bad price, correlation, injury/news, game script, or stale data.
20. Universal principle: obvious high-volume/high-opportunity edges first; sophistication never outranks repeatability and price quality.

Implementation note: the code must fail closed when exact identity, independent model evidence, current price, or material role/availability requirements are not satisfied.


## Low-stake default profile

For the current SportEdge combo workflow, the normal stake profile is USD 1–3. Multi-game Best Available tickets therefore default to a 5-leg strong core with a minimum user-selectable target of 4 legs; Priced Longshot defaults to 6 with a minimum of 5. Single Game remains capped at 4 legs. These are target sizes, not quotas: if enough strong, model-qualified, positive-EV legs do not clear the gates, return fewer legs rather than adding weak filler.
