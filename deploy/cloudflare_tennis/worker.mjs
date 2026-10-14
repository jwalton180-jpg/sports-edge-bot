/**
 * SportsEdge Tennis Reversal Sentinel — Cloudflare Workers Free / D1.
 * Read-only Kalshi public REST. Never accepts trading credentials, never calls
 * an order endpoint. Designed for one-minute Cron Triggers, not a persistent WS.
 * Fail closed for missing/ambiguous/stale scores or incomplete market pages.
 */

export const SERIES = [
  'KXATPMATCH', 'KXATPCHALLENGERMATCH', 'KXWTAMATCH',
  'KXWTACHALLENGERMATCH', 'KXITFMATCH', 'KXITFWMATCH'
];
const KALSHI = 'https://external-api.kalshi.com/trade-api/v2';
const SCORE_URLS = [
  ['ATP', 'https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard'],
  ['WTA', 'https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard']
];
const TENNIS365 = 'https://livescore.tennis365.com/';
const SAMPLE_TTL = 90 * 60 * 1000;
const MAX_MARKETS_PER_SERIES = 3000;
const MAX_TRACKED_PER_SERIES = 650;
const MAX_ALERTS_PER_CYCLE = 3;
const headers = { 'Accept': 'application/json', 'User-Agent': 'SportsEdge-Tennis-ReadOnly/1.0' };

export const canon = value => String(value ?? '').normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
  .toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim().replace(/\s+/g, ' ');
const number = v => (v === '' || v == null ? NaN : Number(v));
const int = v => Number.isInteger(number(v)) ? number(v) : null;
const iso = ms => new Date(ms).toISOString();
const validPrice = p => Number.isFinite(p) && p > 0 && p < 1;
const pairKey = (a, b) => [canon(a), canon(b)].sort().join('|');
const escapeText = text => String(text ?? '').replace(/<[^>]*>/g, ' ').replace(/&amp;/g, '&').replace(/&quot;/g, '"')
  .replace(/&#39;/g, "'").replace(/&nbsp;/g, ' ').replace(/\s+/g, ' ').trim();
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

function price(m, stem) {
  const dollars = m[`${stem}_dollars`];
  const p = dollars !== undefined && dollars !== null ? number(dollars) : number(m[stem]) / 100;
  return validPrice(p) ? Math.round(p * 10000) / 10000 : null;
}

export function priceQuote(m) {
  const bid = price(m, 'yes_bid'), ask = price(m, 'yes_ask');
  return bid !== null && ask !== null && bid <= ask ? { bid, ask } : null;
}

function bestOf(event, grouping, comp, tour) {
  const g = String(grouping?.grouping?.slug ?? '').toLowerCase();
  const t = String(comp?.type?.slug ?? '').toLowerCase();
  const round = String(comp?.round?.displayName ?? '').toLowerCase();
  return tour === 'ATP' && event.major && (g === 'mens-singles' || t === 'mens-singles')
    && !round.includes('qual') ? 5 : 3;
}
function netBreak(pGames, oGames, serving) {
  if (serving === null || serving === undefined || !Number.isInteger(pGames) || !Number.isInteger(oGames)) return null;
  if (pGames < 0 || oGames < 0 || pGames > 7 || oGames > 7 || (pGames === 6 && oGames === 6)) return null;
  const n = pGames + oGames;
  const playerServedFirst = n % 2 === 0 ? serving : !serving;
  return pGames - (playerServedFirst ? Math.ceil(n / 2) : Math.floor(n / 2));
}
function statePair(player, opponent, tour, pSets, oSets, pGames, oGames, best, serving, source, fetched, label, conflict = false) {
  return { player, opponent, pair: pairKey(player, opponent), selection: canon(player), tour,
    playerSets: pSets, opponentSets: oSets, playerGames: pGames, opponentGames: oGames,
    bestOf: best, setsToWin: Math.floor(best / 2) + 1, netBreak: netBreak(pGames, oGames, serving),
    atTiebreak: pGames === 6 && oGames === 6, source, fetched, label, conflict };
}

export function parseEspn(doc, tour, fetched) {
  const out = [];
  for (const event of (doc?.events ?? [])) for (const grouping of (event.groupings ?? [])) {
    for (const comp of (grouping.competitions ?? [])) {
      if (comp?.status?.type?.state !== 'in' || comp.status.type.completed === true) continue;
      const rows = comp.competitors ?? [];
      if (rows.length !== 2) continue;
      const names = rows.map(x => String(x?.athlete?.displayName || x?.athlete?.fullName || x?.athlete?.shortName || '').trim());
      if (!names[0] || !names[1] || names[0] === names[1]) continue;
      const period = int(comp.status.period);
      if (!period || period < 1 || period > 5) continue;
      const best = bestOf(event, grouping, comp, tour);
      if (period > best) continue;
      const ls = rows.map(x => x.linescores ?? []);
      const won = [0, 0]; let coherent = true;
      for (let s = 0; s < period - 1; s++) {
        const a = int(ls[0]?.[s]?.value), b = int(ls[1]?.[s]?.value);
        if (a === null || b === null || a === b) { coherent = false; break; }
        won[a > b ? 0 : 1]++;
      }
      if (!coherent) continue;
      const g = [int(ls[0]?.[period - 1]?.value), int(ls[1]?.[period - 1]?.value)];
      if (g.some(x => x === null)) continue;
      const label = names.join(' vs ') + ' · ' + Array.from({length:period}, (_, k) =>
        `${int(ls[0]?.[k]?.value) ?? '-'}-${int(ls[1]?.[k]?.value) ?? '-'}`).join(' · ');
      for (let i = 0; i < 2; i++) out.push(statePair(names[i], names[1-i], tour,
        won[i], won[1-i], g[i], g[1-i], best,
        typeof rows[i].possession === 'boolean' ? rows[i].possession : null,
        'ESPN', fetched, label));
    }
  }
  return out;
}

function span(block, id) {
  const m = block.match(new RegExp(`<span\\b[^>]*\\bid=["']${id}["'][^>]*>([\\s\\S]*?)<\\/span>`, 'i'));
  return m ? escapeText(m[1]) : null;
}
function classNumbers(block, side) {
  const region = block;
  return [...region.matchAll(new RegExp(`<span\\b[^>]*class=["'][^"']*\\btennis_${side}_set_design\\b[^"']*["'][^>]*>([\\s\\S]*?)<\\/span>`, 'gi'))]
    .map(x => int(escapeText(x[1]))).filter(x => x !== null);
}
/** Parse lower-tour tennis365 server-rendered singles; never infer missing set details. */
export function parseTennis365(html, fetched) {
  const out = [];
  const start = [...String(html).matchAll(/<div\s+id=["'](\d+)["']\s+class=["'][^"']*\btennis_score_grid\b[^"']*["'][^>]*>/gi)];
  for (let k = 0; k < start.length; k++) {
    const id = start[k][1], block = html.slice(start[k].index, start[k+1]?.index ?? html.length);
    if (!block.includes('live_icon')) continue;
    const periodMatch = block.match(/<strong>\s*(?:Set\s*)?(\d+)\s*<\/strong>/i);
    const period = int(periodMatch?.[1]);
    if (!period || period > 3) continue;
    const href = block.match(/<a\b[^>]*href=["']([^"']+)["'][^>]*class=["'][^"']*\bmatches_grid_anchor\b/i)?.[1] || '';
    const hrefL = href.toLowerCase();
    const tour = hrefL.includes('-challenger-') ? 'CHALLENGER' : hrefL.includes('-itf-') ? 'ITF' : hrefL.includes('-wta-') ? 'WTA' : 'ATP';
    // ESPN is authoritative for main-tour score; lower-tour feed is standalone.
    if (!['ITF', 'CHALLENGER'].includes(tour)) continue;
    const names = [span(block, `hn-${id}`), span(block, `an-${id}`)];
    if (names.some(x => !x || x.includes('/'))) continue;
    const sets = [int(span(block, `hs-${id}`)), int(span(block, `as-${id}`))];
    const games = [int(span(block, `csh-${id}`)), int(span(block, `csa-${id}`))];
    if ([...sets, ...games].some(x => x === null)) continue;
    const history = [classNumbers(block, 'home').slice(0, period-1), classNumbers(block, 'away').slice(0, period-1)];
    if (history.some(x => x.length !== period-1)) continue;
    const won = [0, 0]; let coherent = true;
    for (let s = 0; s < period-1; s++) {
      const [a,b] = [history[0][s], history[1][s]];
      if (a === b) { coherent = false; break; }
      won[a > b ? 0 : 1]++;
    }
    if (!coherent || won.some((w,i) => w !== sets[i])) continue;
    const hasBall = ['home','away'].map(side => {
      const m = block.match(new RegExp(`<span\\b[^>]*class=["'][^"']*\\btennis_${side}_ball\\b[^"']*["'][^>]*>([\\s\\S]*?)<\\/span>`, 'i'));
      return !!m && /<img\b/i.test(m[1]);
    });
    const serve = hasBall[0] === hasBall[1] ? [null, null] : hasBall;
    const label = names.join(' vs ') + ' · ' + Array.from({length:period}, (_, s) =>
      s === period-1 ? games.join('-') : `${history[0][s]}-${history[1][s]}`).join(' · ');
    for (let i=0; i<2; i++) out.push(statePair(names[i], names[1-i], tour,
      sets[i], sets[1-i], games[i], games[1-i], 3, serve[i], 'Tennis365', fetched, label));
  }
  return out;
}

/** Carry forward PR #71. Missing break evidence in an elimination set fails closed. */
export function scoreReason(s, now) {
  if (!s) return 'unverified_live_score';
  if (s.conflict || !s.source) return 'conflicted_or_unverified_score';
  if (!Number.isFinite(s.fetched) || now < s.fetched || now - s.fetched > 120000) return 'stale_live_score';
  const g = s.playerGames, o = s.opponentGames;
  if (!Number.isInteger(g) || !Number.isInteger(o) || g < 0 || o < 0 || g > 7 || o > 7) return 'invalid_game_score';
  if (![3,5].includes(s.bestOf) || s.setsToWin !== (Math.floor(s.bestOf/2)+1)) return 'untrusted_match_format';
  if (Math.min(s.playerSets,s.opponentSets)<0 || Math.max(s.playerSets,s.opponentSets)>=s.setsToWin) return 'completed_or_invalid_sets';
  if (s.playerSets + s.opponentSets >= s.bestOf) return 'impossible_score';
  if (s.atTiebreak) return 'tiebreak_high_variance';
  if (o >= 5 && g < o) return 'near_game_or_match_loss';
  if (s.opponentSets === s.setsToWin-1) {
    if (g-o < 2) return 'elimination_set_without_control';
    if (s.netBreak === null || s.netBreak < 1) return 'elimination_set_break_unverified';
  } else if (s.playerSets === s.opponentSets) {
    if (g-o < 2) return 'no_deciding_set_control';
  } else if (s.playerSets > s.opponentSets) {
    if (g-o < -1) return 'current_set_unfavorable';
  } else return 'unverified_match_progress';
  return 'eligible';
}

/** Only complete, strictly matched, unique 2-side events qualify. */
export function makeScoreIndex(markets, scores, now) {
  const sides = new Map(), states = new Map(), eligible = new Map();
  for (const m of markets) if (m.event_ticker && m.yes_sub_title) {
    const e = String(m.event_ticker), name = canon(m.yes_sub_title);
    if (!sides.has(e)) sides.set(e, new Set());
    sides.get(e).add(name);
  }
  for (const s of scores) {
    const k = s.pair + '/' + s.selection;
    if (!states.has(k)) states.set(k, []);
    states.get(k).push(s);
  }
  for (const m of markets) {
    const names = sides.get(String(m.event_ticker)) ?? new Set();
    if (names.size !== 2) continue;
    const p = [...names];
    const found = states.get(p.sort().join('|') + '/' + canon(m.yes_sub_title)) ?? [];
    if (found.length !== 1) continue;
    const s = found[0];
    if (scoreReason(s, now) === 'eligible') eligible.set(m.ticker, s);
  }
  return eligible;
}

/** Full-causal observed quotes only. Require collapse then >=2 upward samples or a >=5pp single repricing. */
export function evaluateMovement(history, now) {
  const recent = (history ?? []).filter(x => Array.isArray(x) && now - x[0] >= 0 && now - x[0] <= SAMPLE_TTL);
  if (recent.length < 4) return null;
  const last = recent.at(-1), ask = last[2] / 10000, bid = last[1] / 10000;
  if (now-last[0] > 125000 || ask < .01 || ask > .20 || bid <= 0 || bid > ask) return null;
  const spread = ask - bid;
  if (spread > Math.min(.06, Math.max(.02, ask * .45))) return null;
  let trough = Infinity, ti = -1, peak = 0;
  for (let i=0;i<recent.length;i++) {
    const p=recent[i][2]/10000;
    if (p < trough) { trough = p; ti = i; }
  }
  if (ti < 1 || ti === recent.length-1 || trough > .10) return null;
  for (let i=0;i<ti;i++) peak = Math.max(peak, recent[i][2]/10000);
  if (peak-trough < .07 || (peak-trough)/Math.max(peak,.01) < .45) return null;
  const rebound = ask-trough;
  if (rebound < .02) return null;
  let upward = 0;
  for (let i=ti+1;i<recent.length;i++) if (recent[i][2]-recent[i-1][2] >= 50) upward++;
  const prev = recent.at(-2)[2]/10000;
  if ((upward < 2 && !(rebound >= .06 && ask-prev >= .05)) || ask < prev) return null;
  return { ask, bid, trough, peak, rebound, spread, troughAt: recent[ti][0], observedAt: last[0] };
}

async function getJson(url) {
  const res = await fetch(url, { headers, signal: AbortSignal.timeout(9000) });
  if (!res.ok) throw new Error('upstream HTTP '+res.status);
  return res.json();
}
async function getText(url) {
  const res = await fetch(url, {headers: {...headers, Accept:'text/html'}, signal: AbortSignal.timeout(9000)});
  if (!res.ok) throw new Error('score HTTP '+res.status);
  return res.text();
}
async function marketsFor(series) {
  let cursor = '', prev = '', found = [];
  for (let p=0;p<3;p++) {
    const u = new URL(`${KALSHI}/markets`);
    u.searchParams.set('status','open'); u.searchParams.set('series_ticker', series);
    u.searchParams.set('limit','1000'); if (cursor) u.searchParams.set('cursor',cursor);
    const data=await getJson(u.toString());
    if (!Array.isArray(data?.markets)) throw new Error('invalid market shape: '+series);
    found.push(...data.markets);
    if (found.length > MAX_MARKETS_PER_SERIES) throw new Error('market limit: '+series);
    const next=String(data.cursor ?? '');
    if (!next) return found;
    if (next === cursor || next === prev) throw new Error('repeated cursor: '+series);
    prev = cursor; cursor = next;
  }
  throw new Error('incomplete market pagination: '+series);
}
async function scoresNow(now) {
  const results=await Promise.allSettled([
    ...SCORE_URLS.map(([, url])=>getJson(url)), getText(TENNIS365)
  ]);
  const scored = []; const sources = {};
  for (let i=0;i<results.length;i++) {
    const result = results[i], name = i<2 ? SCORE_URLS[i][0] : 'Tennis365';
    if (result.status !== 'fulfilled') { sources[name] = 'unavailable'; continue; }
    try {
      scored.push(...(i<2 ? parseEspn(result.value,name,now) : parseTennis365(result.value,now)));
      sources[name]='ok';
    } catch (_) { sources[name]='parse_error'; }
  }
  return {scores:scored, sources};
}

function sample(m, at) {
  const q=priceQuote(m); if (!q) return null;
  return [at, Math.round(q.bid*10000), Math.round(q.ask*10000)];
}
function idFor(m, movement) {
  // Stable across repeated cron observations of a single collapse.
  return `${m.ticker}|${movement.troughAt}`;
}
function notifyText(row, score) {
  const c=n=>(n*100).toFixed(1)+'¢';
  return `${row.selection} | Kalshi quoted ASK ${c(row.ask)} | BID ${c(row.bid)}\n`+
    `Observed trough ${c(row.trough)} → rebound +${(row.rebound*100).toFixed(1)}pp\n`+
    `Score: ${score.label} (${score.source}; survivability checked)\n`+
    `Watch only: no calibrated win probability, fills or guaranteed returns.\n`+
    `Ticker: ${row.ticker}\nObserved UTC: ${iso(row.observedAt)}`;
}
async function pushNtfy(env, row, score) {
  const topic=String(env.SPORTSEDGE_NTFY_TOPIC ?? '');
  if (!/^[A-Za-z0-9_-]{24,64}$/.test(topic)) throw new Error('invalid ntfy topic');
  const origin=String(env.SPORTSEDGE_NTFY_SERVER ?? 'https://ntfy.sh');
  if (!/^https:\/\/[a-zA-Z0-9.-]+(?::\d+)?$/.test(origin)) throw new Error('invalid ntfy server');
  const auth=env.SPORTSEDGE_NTFY_TOKEN ? {Authorization: 'Bearer '+env.SPORTSEDGE_NTFY_TOKEN} : {};
  const res=await fetch(origin+'/',{method:'POST', headers:{'content-type':'application/json',...auth},
    body:JSON.stringify({topic,title:'SportsEdge Tennis: reversal recovery WATCH',
      message:notifyText(row,score),priority:4,tags:['tennis'],click:'https://kalshi.com/markets'}),
    signal:AbortSignal.timeout(8000)});
  if (![200,201,202].includes(res.status)) throw new Error('ntfy HTTP '+res.status);
  // Provider HTTP acceptance is not proof of device delivery.
  const doc=await res.json().catch(()=>({}));
  return String(doc?.id ?? 'accepted');
}

const stmt = (db, sql, ...args) => db.prepare(sql).bind(...args);
async function saveRun(env, started, stats) {
  const status = stats.errors.length ? 'degraded' : 'ok';
  await stmt(env.TENNIS_DB, 'INSERT INTO runs(observed_at,status,markets,scores,eligible,alerts,errors) VALUES(?,?,?,?,?,?,?)',
    started, status, stats.markets, stats.scores, stats.eligible, stats.alerts, JSON.stringify(stats.errors.slice(0,10))).run();
}
async function followup(env, markets, now) {
  // No hypothetical P&L: follow-on *quoted bids* and official settlement only.
  const existing=await stmt(env.TENNIS_DB, "SELECT id,ticker,alert_at,best_bid,final_result,observed_at FROM alerts WHERE alert_at > ? AND (final_result IS NULL OR observed_at > ?) ORDER BY alert_at DESC LIMIT 6",
    now-2*86400000, now-6*3600000).all();
  const byTicker = new Map(markets.map(m=>[m.ticker,m]));
  for (const a of existing.results ?? []) {
    const quote=priceQuote(byTicker.get(a.ticker) ?? {});
    if (quote && now >= a.alert_at) {
      await stmt(env.TENNIS_DB, 'UPDATE alerts SET last_bid=?, best_bid=MAX(COALESCE(best_bid,0),?), observed_at=? WHERE id=?',
        quote.bid, quote.bid, now, a.id).run();
    }
  }
  // An official terminal check is bounded to max 3 GETs per cron tick.
  let terminal=0;
  for (const a of (existing.results ?? [])) {
    if (terminal >= 3 || a.final_result !== null || now-a.alert_at < 15*60000) continue;
    terminal++;
    try {
      const data=await getJson(`${KALSHI}/markets/${encodeURIComponent(a.ticker)}`);
      const m=data?.market;
      const result=String(m?.result ?? '').toLowerCase();
      if (!['yes','no'].includes(result) || String(m?.status ?? '').toLowerCase() !== 'settled') continue;
      await stmt(env.TENNIS_DB,'UPDATE alerts SET final_result=?,settled_at=? WHERE id=?',result,now,a.id).run();
    } catch (_) { /* only authoritative settlement labels; retry later */ }
  }
}

export async function runCycle(env, now=Date.now()) {
  if (!env.TENNIS_DB) throw new Error('TENNIS_DB required');
  if (String(env.LIVE_TRADING)!=='false' || String(env.ORDER_SUBMISSION_ENABLED)!=='false' || String(env.READ_ONLY_KALSHI)!=='true') throw new Error('read-only safety flags required');
  if (!/^[A-Za-z0-9_-]{24,64}$/.test(String(env.SPORTSEDGE_NTFY_TOPIC ?? ''))) throw new Error('ntfy topic not configured');
  const started = now;
  const stats = {markets:0,scores:0,eligible:0,alerts:0,errors:[],score_sources:{}};
  const scoreData=await scoresNow(now);
  for (const [source, status] of Object.entries(scoreData.sources)) if (status !== 'ok') stats.errors.push(source+':score_source_'+status);
  stats.scores=scoreData.scores.length; stats.score_sources=scoreData.sources;
  const data=await Promise.allSettled(SERIES.map(async series => {
    // An incomplete series must never create an apparent market or missing-price signal.
    const markets=await marketsFor(series);
    const row=await stmt(env.TENNIS_DB,'SELECT payload FROM series_state WHERE series=?',series).first();
    return {series, markets, state:row?.payload ? JSON.parse(row.payload) : {}};
  }));
  const complete=[];
  for (let i=0;i<data.length;i++) {
    if (data[i].status==='fulfilled') complete.push(data[i].value);
    else stats.errors.push(SERIES[i]+':market_or_state_unavailable');
  }
  const allMarkets=complete.flatMap(x=>x.markets);
  const scoreIndex=makeScoreIndex(allMarkets,scoreData.scores,now);
  stats.markets=allMarkets.length; stats.eligible=scoreIndex.size;
  let pushed=0;
  for (const {series, markets, state} of complete) {
    const next={};
    const watchRows=[];
    let tracked = 0;
    for (const m of markets) {
      const t=String(m.ticker || ''); if (!t || !m.yes_sub_title || !m.event_ticker) continue;
      const s=sample(m,now); if (!s) continue;
      // Keep all previously watched and qualifying live match sides, plus
      // <=40c entrants before the plunge. Never silently cap an overloaded series.
      if (s[2] > 4000 && !state[t] && !scoreIndex.has(t)) continue;
      const previous=(state[t]?.path ?? []).filter(x=>Array.isArray(x) && now-x[0] <= SAMPLE_TTL && now-x[0]>=0);
      // Sampled public REST book quotes, not trade prints and not assumed fills.
      if (!previous.length || previous.at(-1)[1]!==s[1] || previous.at(-1)[2]!==s[2]) previous.push(s);
      const path=previous.slice(-90);
      const prior=state[t] ?? {};
      const record={path, lastAlertAt:prior.lastAlertAt ?? 0, lastAlertAsk:prior.lastAlertAsk ?? 0};
      next[t]=record;
      tracked++;
      if (tracked > MAX_TRACKED_PER_SERIES) {
        // A D1 row can hold at most 2 MB. No partial/truncated coverage claims.
        stats.errors.push(series+':tracking_capacity_exceeded');
        watchRows.length=0;
        break;
      }
      const movement=evaluateMovement(path,now);
      if (!movement || !scoreIndex.has(t) || pushed+watchRows.length>=MAX_ALERTS_PER_CYCLE) continue;
      if (now - record.lastAlertAt < 15*60000) continue;
      // A new peak need not indicate an independent reversal: throttle duplicates.
      watchRows.push({ ...movement, ticker:t, selection:String(m.yes_sub_title), id:idFor(m,movement), score:scoreIndex.get(t), record });
    }
    // Persist data before push, so network/provider failure cannot erase the observed prices.
    if (tracked > MAX_TRACKED_PER_SERIES) continue;
    await stmt(env.TENNIS_DB,'INSERT INTO series_state(series,payload,updated_at) VALUES(?,?,?) ON CONFLICT(series) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at',
      series,JSON.stringify(next),now).run();
    let newSuccessfulPush = false;
    for (const alert of watchRows) {
      if (pushed>=MAX_ALERTS_PER_CYCLE) break;
      const inserted=await stmt(env.TENNIS_DB,
        `INSERT OR IGNORE INTO alerts(id,ticker,selection,alert_at,observed_at,alert_ask,alert_bid,trough,peak,spread,score_source,score_label,score_reason,score_confidence,push_status,poll_delay_ms)
         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
        alert.id,alert.ticker,alert.selection,now,now,alert.ask,alert.bid,alert.trough,alert.peak,
        alert.spread,alert.score.source,alert.score.label,'eligible','source-backed, unique pair','pending',
        Math.abs(Date.now()-now)<86400000 ? Math.max(0,Date.now()-now) : null).run();
      if (!inserted.meta?.changes) {
        // Only explicit provider failures are safely retryable. A pending request
        // may already have reached ntfy before a process crash; do not duplicate it.
        const previousPush=await stmt(env.TENNIS_DB,
          'SELECT push_status,push_attempts,last_attempt_at FROM alerts WHERE id=?',alert.id).first();
        if (previousPush?.push_status!=='failed' || previousPush.push_attempts>=3 ||
            now-(previousPush.last_attempt_at ?? 0)<60000) continue;
      }
      // Record attempt BEFORE network publish. If D1 is unavailable, fail closed.
      await stmt(env.TENNIS_DB,
        'UPDATE alerts SET push_status=?,push_attempts=push_attempts+1,last_attempt_at=? WHERE id=?',
        'pending',now,alert.id).run();
      pushed++;
      // Repeated failures do not automatically infer delivery. Keep explicit failed status.
      try {
        const receipt=await pushNtfy(env,alert,alert.score);
        const acceptedAt=Date.now();
        await stmt(env.TENNIS_DB,'UPDATE alerts SET push_status=?,provider_id=?,push_accepted_at=?,provider_accept_delay_ms=? WHERE id=?',
          'accepted',receipt,acceptedAt,Math.abs(acceptedAt-now)<86400000 ? Math.max(0,acceptedAt-now) : null,alert.id).run();
        alert.record.lastAlertAt=now; alert.record.lastAlertAsk=alert.ask;
        newSuccessfulPush=true;
        stats.alerts++;
      } catch (_) {
        stats.errors.push('ntfy_rejected_or_unavailable');
        await stmt(env.TENNIS_DB,'UPDATE alerts SET push_status=? WHERE id=?','failed',alert.id).run();
      }
    }
    if (newSuccessfulPush) {
      await stmt(env.TENNIS_DB,'UPDATE series_state SET payload=?,updated_at=? WHERE series=?', JSON.stringify(next),now,series).run();
    }
  }
  try {await followup(env,allMarkets,now);} catch (_) {stats.errors.push('followup_unavailable');}
  try {await saveRun(env,started,stats);} catch (_) {stats.errors.push('run_ledger_unavailable');}
  return {...stats, observed_at:iso(now),status:stats.errors.length?'degraded':'ok'};
}

export default {
  async scheduled(controller, env, ctx) { ctx.waitUntil(runCycle(env, controller.scheduledTime).catch(e=>{
    // No secrets, topics, or URLs in worker logs.
    console.error('tennis sentinel cycle failed: ', String(e?.message || 'unavailable').slice(0,90));
  })); },
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method!=='GET' || url.pathname!=='/health' || !env.SPORTSEDGE_HEALTH_TOKEN ||
      request.headers.get('Authorization')!==`Bearer ${env.SPORTSEDGE_HEALTH_TOKEN}`) return new Response('Not Found',{status:404});
    if (!env.TENNIS_DB) return Response.json({status:'unconfigured'},{status:503});
    const row=await env.TENNIS_DB.prepare('SELECT observed_at,status,markets,scores,eligible,alerts,errors FROM runs ORDER BY observed_at DESC LIMIT 1').first();
    if (!row) return Response.json({status:'not_started'},{status:503});
    return Response.json({ ...row, last_run_utc:iso(row.observed_at), fresh: Date.now()-row.observed_at<180000 },
      {headers:{'Cache-Control':'no-store'}});
  }
};