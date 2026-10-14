-- Isolated SportsEdge Tennis-only D1 data. No order, balance, position or P&L tables.
CREATE TABLE IF NOT EXISTS series_state (
  series TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
  id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  selection TEXT NOT NULL,
  alert_at INTEGER NOT NULL,
  observed_at INTEGER NOT NULL,
  alert_ask REAL NOT NULL,
  alert_bid REAL NOT NULL,
  trough REAL NOT NULL,
  peak REAL NOT NULL,
  spread REAL NOT NULL,
  score_source TEXT NOT NULL,
  score_label TEXT NOT NULL,
  score_reason TEXT NOT NULL,
  score_confidence TEXT NOT NULL,
  push_status TEXT NOT NULL,
  provider_id TEXT,
  push_accepted_at INTEGER,
  poll_delay_ms INTEGER, -- scheduled-to-sample, if clock comparable
  provider_accept_delay_ms INTEGER, -- scheduled-to-ntfy HTTP acceptance, not iOS arrival
  last_bid REAL,
  best_bid REAL,
  final_result TEXT CHECK(final_result IN ('yes','no')),
  settled_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_alerts_recent ON alerts(alert_at DESC);
CREATE TABLE IF NOT EXISTS runs (
  observed_at INTEGER PRIMARY KEY,
  status TEXT NOT NULL,
  markets INTEGER NOT NULL,
  scores INTEGER NOT NULL,
  eligible INTEGER NOT NULL,
  alerts INTEGER NOT NULL,
  errors TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_latest ON runs(observed_at DESC);