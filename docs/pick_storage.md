## Optional durable pick storage (free-tier compatible)

SportEdge's hosted Streamlit filesystem is not a durable database. The My Picks and Performance pages use a server-side Supabase PostgREST adapter only when both of these Streamlit secrets are configured:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "store-only-on-server"
```

Never commit these values or place them in browser-visible code. Use a private Supabase project, restrict dashboard access, rotate keys when exposed, and review the provider's current free-tier limits. Without both secrets, the app must clearly label tracking as session-only and offer CSV export; it must not claim cross-session persistence.

Create the append-only table in the Supabase SQL editor:

```sql
create table if not exists public.sportedge_pick_records (
  id text primary key,
  record_type text not null check (record_type in ('prediction', 'settlement', 'note')),
  payload jsonb not null,
  created_at timestamptz not null default now()
);
create index if not exists sportedge_pick_records_type_created_idx
  on public.sportedge_pick_records (record_type, created_at);
alter table public.sportedge_pick_records enable row level security;
```

The service-role key bypasses row-level security and must remain server-side. Prediction and settlement events are separate inserts; application code never updates or deletes an existing row. This is append-only at the application layer; database administrators can still alter the database, so this is not tamper-proof audit storage.
