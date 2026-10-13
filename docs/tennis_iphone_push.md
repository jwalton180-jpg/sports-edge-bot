# SportsEdge Tennis: iPhone push alerts (ntfy)

The Tennis prospective observer can send **free ntfy push alerts** to a subscribed iPhone without placing orders. Alerts contain only public-market research.

## Activate on iPhone

1. Install **ntfy** from the iOS App Store, allow notifications, and open it.
2. In ntfy, tap **Subscribe to topic** on server https://ntfy.sh . Choose a unique, unguessable **24–64-character topic** containing only letters, numbers, hyphens and underscores. Keep the topic private.
3. In GitHub, navigate to **jwalton180-jpg/sports-edge-bot → Settings → Secrets and variables → Actions → New repository secret**. Set the secret named **SPORTSEDGE_NTFY_TOPIC** to exactly the topic you subscribed to on the phone.
4. Let the next worker observation process a new signal. Enable iPhone notifications for ntfy and verify your subscribed topic.

**Security:** ntfy.sh topics are *not encrypted or access-controlled by default*. Anyone knowing the topic can subscribe. Choose a long random topic and don't share it, commit it, or paste it in chat. The notifications contain only public Kalshi market information, not your positions or credentials.

Optional GitHub secret **SPORTSEDGE_NTFY_TOKEN** provides Bearer authentication for a server configured to require it.

## Alert categories

- **1–4¢ extreme dip**: low priority and explicitly **TRACKING ONLY**, not a bet recommendation.
- **Early reversal / WATCH (generally 2–20¢)**: provisional research; requires fresh quoted ask, usable structural score and meaningful independent model edge.
- **Recovery building**: research-only score-backed improvement.
- **REVERSAL SIGNAL / DEEP REVERSAL**: higher notification priority, existing model-first gates unchanged.

Each push includes player, quoted ASK, estimated live model fair (if available), structural score, UTC observation time, and Kalshi ticker. The link opens Kalshi's markets page; **verify the live price before acting**. An observed quote is not a fill.

SportsEdge suppresses repeat pushes for the same ticker/stage and permits escalation when a stronger stage qualifies. Acceptance receipts are recorded on the dedicated research-data branch. Provider HTTP acceptance does not guarantee iOS receipt; a rare outage between acceptance and persistence may result in a duplicate alert.

GitHub Actions scheduling is **best effort**. Intra-run sampling can improve opportunities but is not a continuously connected or guaranteed real-time service. Historical signals are never replayed as newly live after configuration.

## Checking delivery

In the Tennis prospective observer GitHub Actions run, inspect **push_configured**, **push_accepted**, and **push_failed** in the worker JSON. The public research heartbeat stores counts and errors but never the topic or token. With no configured secret, no notification is sent. Verify the ntfy app's topic subscription and iOS notification settings if acceptance is recorded but the phone shows nothing.
