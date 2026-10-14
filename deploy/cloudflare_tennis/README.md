# Cloudflare Free Tennis Sentinel — independent deployment

**Not live by merging a PR.** This is a separate JavaScript Cloudflare Worker + D1 database, unrelated to the Streamlit app, Railway, or the Ubuntu VM. No Kalshi API key is used: only public read-only market endpoints. Never configure trading access or run this on the shared host.

## Architecture and limits

A one-minute UTC Cron Trigger queries six Kalshi match-winner series (ATP/WTA/Challenger/ITF men and women), ESPN ATP/WTA live scores, and Tennis365 Challenger/ITF live singles. Strict page exhaustion, unique two-player event matching, current score consistency, and the existing PR #71 match-survivability rules gate every alert. Minute-by-minute *observed public bid/ask quotes* live in D1 per series. Candidate alert requires at least four observed quote points, a verified >=7-percentage-point / >=45% collapse to <=10¢ and genuine >=2pp rebound on two upward readings (or strong single >=6pp / >=5pp reprice), fresh score, sufficiently tight spread, and current ask from 1–20¢. There are no assumed fills, predicted probabilities, ROI, or profits.

Existing GitHub Actions observers remain running as fallback. They are **not changed or disabled**. Both can alert independently, so duplicates are possible until 24 hours of prospective cloud reliability evidence supports a coordinated transition.

Cloudflare Free at the time of writing: Workers 100,000 requests/day, scheduled CPU 10ms/invocation, 50 subrequests/invocation; D1 5 million rows read/day, 100,000 written/day, total 5 GB, single database max 500 MB. **10ms CPU is a hard viability risk** when parsing full Tennis365 HTML or large ESPN results. A real cloud deployment and Cloudflare analytics are required before promising reliability. Cloudflare may throttle at daily caps; it will not automatically create paid resources using this setup.

## First deployment

1. Create or use a Cloudflare **Free** account (do not upgrade). Create a Cloudflare API token scoped to **Workers Scripts: Edit** and **D1: Edit** for your own account. Never paste credentials in chat. Workers deployment may require additional account-read permission.
2. In your GitHub repo **Settings > Secrets and variables > Actions**, add `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`. Keep the existing 24–64 character `SPORTSEDGE_NTFY_TOPIC`; optionally `SPORTSEDGE_NTFY_TOKEN`. Add `SPORTSEDGE_HEALTH_TOKEN` (long random value) for restricted health checks.
3. Go to **Actions > Deploy Cloudflare Tennis Sentinel > Run workflow**. The workflow checks required secrets, reuses or creates only `sportsedge-tennis-sentinel` D1, applies the checked-in schema, deploys the Worker, and configures ntfy secrets. Never run the old Railway deployment.
4. Review GitHub Actions deployment steps and the Cloudflare Workers logs/analytics. Authorized health check: `curl -H "Authorization: Bearer <your health token>" https://sportsedge-tennis-sentinel.<account>.workers.dev/health`. Check last-run age (<180 seconds), `status`, score coverage, errors, alert records. The 1-minute cron is **not** continuous streaming and may skip 1-2 min under scheduler/load.
5. The workflow optionally sends an **explicitly labeled test push** to the existing ntfy topic; ntfy HTTP success proves acceptance, **not** that iOS received it. Confirm the phone notification yourself. Run 24 hours of side-by-side comparisons with GitHub Actions; check gaps, official settlement coverage and Cloudflare CPU/error limits before replacing any fallback.

If ESPN or Tennis365 block Cloudflare egress or change their schemas, the affected tour's pushes fail closed and health is degraded; do not equate lack of alerts with lack of opportunities. Since Tennis365 is not an official ITF feed, results should be reviewed for coverage; ITF and SofaScore can inform future adapters once their accessible live score formats are verified. Low-tour coverage should never be faked from markets alone.

## Safety, storage and outcomes

- Only public GETs to Kalshi's `/markets`, `/markets/{ticker}`; POST is exclusively `ntfy.sh` after eligibility gate.
- Database `alerts`: quoted ask/bid at alert, trough/peak, spread, source-backed score, notification provider accepted/failed, subsequently observed quoted bid and best bid; official `yes/no` resolution when settlement is verified. These are **not realized trading returns**.
- Database `runs`: minute timestamps and degraded reasons. Database `series_state`: most recent 90 minutes of observed bid/ask snapshots. No Kalshi or phone secrets in data rows, code, HTTP logs or output.
- Rollout never stops, edits or deploys PRIMARY Sixty, Late Caller, AutoProfit, shared Ubuntu, Railway, or the existing GitHub observer workflows.