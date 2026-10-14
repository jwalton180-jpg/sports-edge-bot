# Bounded future-record model validation

SportEdge remains an on-demand Streamlit app. This study uses only the public
GitHub-hosted Actions runner, twice daily until **December 14, 2026**, to freeze
read-only, price-independent NHL / NCAA football / EPL pre-event projections and
executable-ask reference prices. There is no Railway, shared Ubuntu load,
always-on worker, betting order, or account credential.

Source population: open Kalshi series supported by existing discovery; eligibility
within 60 hours of game start; 14 markets per sport per scheduled capture,
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
