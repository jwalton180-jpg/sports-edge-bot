# Provider access matrix

This matrix records the actual integration boundary. A provider is not considered connected merely because a website or API exists. No credentials are embedded in the repository.

| Provider | Confirmed capability / access | SportEdge status | Next requirement |
|---|---|---|---|
| PropsBot.AI | Public-facing prop research is discoverable; a supported public API or authorized export entitlement was not verified in this audit. | Research reference only | Provider confirmation of API/export terms and access. |
| Betstamp Pro API | Official API docs describe markets, players, teams, fixtures, and pricing endpoints. Authentication uses a provisioned API key; access is entitlement/plan-dependent. | Not connected; key absent | Obtain an authorized key and confirm allowed leagues, plan, quota, and intended usage. |
| Juice Reel API | Official OAuth 2.0 documentation supports consented access to a connected user's open/settled bet records with scopes. Token exchange/refresh must be server-side. | Not connected; no OAuth backend/token store | Register an app, implement secure backend OAuth/PKCE and durable token storage, obtain explicit user consent. |
| Outlier | Public product research exists; a supported authorized public API was not verified in this audit. | Research reference only | Provider confirmation of API/export access and terms. |
| The Odds API | Existing SportEdge client; requires THE_ODDS_API_KEY; free-tier quota and prop-market coverage depend on the account. | Optional existing integration | Configure key in Streamlit secrets and verify quota/market availability. |
| Official NHL API | Existing read-only schedule/roster/game-log integration; current model output remains unvalidated research unless lineup/role/odds checks are available. | Partial integration | Improve active lineup, role/PP, goalie, and current odds evidence before value claims. |
| ESPN public feeds | Existing read-only scoreboard/context clients; availability and schema may vary. | Integrated where implemented | Continue freshness checks and fail-closed behavior. |

Official documentation checked:
- Betstamp API: https://www.betstamp.com/docs and https://www.betstamp.com/docs/getting-started
- Juice Reel API: https://www.juicereel.com/api-docs

Access notes:
- Betstamp Pro API is not assumed free; docs say keys are provisioned and league entitlements are enforced. No key was available to test.
- Juice Reel's documented API is not unrestricted access to other users' records: a connected user must authorize the scopes requested. Its client secret and rotating refresh tokens must stay server-side.
- PropsBot.AI and Outlier are not scraped and no fake connector responses are returned. Until authorized integration terms are verified, use manual research links/imports only.
- No provider ROI is merged into SportEdge's internal record. Provider-published metrics must remain labeled provider-reported unless independently verified.
