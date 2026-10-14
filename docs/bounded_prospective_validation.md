# Bounded future-record model validation

SportEdge remains an on-demand Streamlit app. This study uses only the public
GitHub-hosted Actions runner, twice daily until **December 14, 2026**, to freeze
read-only, price-independent NHL / NCAA football / EPL pre-event projections and
executable-ask reference prices. There is no Railway, shared Ubuntu load,
always-on worker, betting order, or account credential.

Source population: open Kalshi series supported by existing discovery; eligibility
within 96 hours of listed market occurrence (with independent kickoff checked afterward); 14 markets per sport per scheduled capture, restricted to independently modelled contract families,
deterministically selected by SHA256("SPORTSEDGE-PROSPECTIVE-V1-20261014|ticker")
before viewing outcomes or estimated edge. This is a sample, NOT a complete
market inventory. Unavailable models stay rejected; repeat scans are deduplicated
during assessment. The 2026-12-15 cutoff stops data collection even though the
workflow may still be scheduled. Disable the workflow after the study window.

Each Actions snapshot artifact has the GitHub server archive timestamp, repo
commit, run ID, digest, market sample, model outputs, quote and source provenance.
The bounded grader verifies external artifact creation before the event, digest
and source clock ordering; official Kalshi binary outcomes only. It compares
locked independent probability against the contemporaneous YES ask reference,
not against an invented "fair" probability. Voids, missing settlement times,
missing participation/rules, outdated quotes and incomplete retrieval are
flagged, never silently counted as wins.

There is **no validated win rate now**. GitHub Actions might delay or skip a
scheduled job; only successful snapshots count. Calibration and Brier reports
remain RESEARCH_ONLY. Promotion remains disabled until sufficiently many
future, immutable, representative, independently reviewed observations can
pass the separately locked per-model/payout/rules validation gate, with
pre-registered benchmark and measured execution.

In GitHub Actions open **Bounded prospective sport model validation** and
download the two artifacts for each run: `sportsedge-live-research-snapshot-...`
and `sportsedge-prospective-review-...`. Reports intentionally exclude
credentials, holdings, signing keys and private account data.

## Causal date correction (v2, Oct 14)

Kalshi `occurrence_datetime` can represent expected game expiry rather than kickoff; for example a listed NHL market at 02:30 UTC can concern a game actually starting at 23:30 UTC the previous day. The v2 capture protocol therefore uses the official NHL/ESPN matched event's **actual scheduled start time** for lookback cutoff, pregame admissibility and grading. Contract date encoded in the event ticker selects the official schedule day; team identities and an independently bounded post-start expiry gap must agree. Any unresolved schedule or already-started game fails closed. V1 observations with no model forecasts remain archived and are still digest-verifiable but are not promoted. V2 sampling covers only currently independently modeled NHL player/team families, NCAA game winner and EPL totals/BTTS/team goals. Other families stay explicitly unsupported.

## Scoped prospective capture (v3, Oct 14)

The bounded study now enumerates only independently modeled NHL, NCAA football, and EPL series; it no longer consumes entire unrelated NCAA spread/total inventories just to sample NCAA game winners. Normal application discovery and its pagination remain unchanged. The study fetches at most 24 pages of 500 contracts per included series and reports cursor incompleteness rather than claiming full coverage. The 96-hour listed-occurrence horizon allows weekend Premier League research to enter a midweek capture, while the independent official kickoff check still excludes any game already started. Sampling protocol V3 has a new deterministic hash cohort; the grader retains verification of archived V1/V2 artifacts without converting historical observations into prospective proof or authorizing model promotion.


## NBA 19+ alternate-spread prospective cohort (V4)
V4 adds a fixed, price-blind research cohort from actual listed `KXNBASPREAD` contracts with exact 18.5-point half-strike and `greater` direction. At most six NBA contracts are drawn by SHA256 from a 96-hour occurrence window per snapshot, alongside the existing NHL/NCAA/EPL cohorts. The independent NBA team schedule determines actual tipoff; preseason is unsupported, and its apparent lines must not be graded as regular-season evidence. Existing V1/V2/V3 artifacts remain verifiable. This lane compares the frozen 19+ probability against the contemporaneous Kalshi executable YES ask including fees, then independently graded official binary settlement. Retrospective margin improvement is not market-price validation and cannot promote a model.
