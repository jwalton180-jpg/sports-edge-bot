import test from 'node:test';
import assert from 'node:assert/strict';
import {
  canon, scoreReason, parseEspn, parseTennis365, evaluateMovement,
  makeScoreIndex, priceQuote, runCycle, SERIES
} from '../deploy/cloudflare_tennis/worker.mjs';

const t=Date.parse('2026-10-14T08:00:00Z');
const basicScore={player:'Alice Test',opponent:'Bea Sample',selection:'alice test',pair:'alice test|bea sample',tour:'ITF',
 playerSets:0,opponentSets:1,playerGames:3,opponentGames:1,bestOf:3,setsToWin:2,
 netBreak:1,atTiebreak:false,source:'Tennis365',fetched:t,label:'A vs B · 3-6 · 3-1',conflict:false};
const values=(asks, bids) => asks.map((a,i)=>[t+i*60000, Math.round((bids?.[i] ?? a-.01)*10000),Math.round(a*10000)]);

test('price accepts only uncrossed genuine two-sided quotes',()=>{
 assert.deepEqual(priceQuote({yes_bid_dollars:'0.13',yes_ask_dollars:'0.16'}),{bid:.13,ask:.16});
 assert.deepEqual(priceQuote({yes_bid:13,yes_ask:16}),{bid:.13,ask:.16});
 assert.equal(priceQuote({yes_bid:18,yes_ask:16}),null);
 assert.equal(priceQuote({yes_bid:12}),null);
 assert.equal(priceQuote({yes_bid:0,yes_ask:2}),null);
});

test('survivability refuses doomed straight sets, decider with no control, tiebreak and stale scores',()=>{
 assert.equal(scoreReason(basicScore,t),'eligible');
 assert.equal(scoreReason({...basicScore,playerGames:1,opponentGames:5},t),'near_game_or_match_loss');
 assert.equal(scoreReason({...basicScore,playerGames:1,opponentGames:3},t),'elimination_set_without_control');
 assert.equal(scoreReason({...basicScore,netBreak:null},t),'elimination_set_break_unverified');
 assert.equal(scoreReason({...basicScore,playerSets:0,opponentSets:2},t),'completed_or_invalid_sets');
 assert.equal(scoreReason({...basicScore,atTiebreak:true},t),'tiebreak_high_variance');
 assert.equal(scoreReason({...basicScore,fetched:t-121000},t),'stale_live_score');
 assert.equal(scoreReason(null,t),'unverified_live_score');
 assert.equal(scoreReason({...basicScore,source:null},t),'conflicted_or_unverified_score');
});

test('a collapse alone is not a reversal; two improvements or exceptional one-jump are required',()=>{
 assert.equal(evaluateMovement(values([.22,.19,.03,.04]),t+3*60000),null);
 assert.equal(evaluateMovement(values([.22,.19,.03,.05,.07]),t+4*60000)?.trough,.03);
 assert.equal(evaluateMovement(values([.22,.19,.03,.10]),t+3*60000)?.rebound.toFixed(2),'0.07');
 assert.equal(evaluateMovement(values([.22,.19,.03,.05,.07],[.20,.18,.02,.02,.02]),t+4*60000),null);
 assert.equal(evaluateMovement(values([.22,.19,.03,.05,.25]),t+4*60000),null);
 assert.equal(evaluateMovement(values([.06,.05,.03,.05,.08]),t+4*60000),null);
 assert.equal(evaluateMovement(values([.22,.19,.03,.05,.07]),t+9*60000),null);
});

test('exact unique score-pair matching rejects missing and ambiguous sources',()=>{
 const ms=[{ticker:'A',event_ticker:'E',yes_sub_title:'Alice Test'},{ticker:'B',event_ticker:'E',yes_sub_title:'Bea Sample'}];
 assert.equal(makeScoreIndex(ms,[basicScore],t).size,1);
 assert.equal(makeScoreIndex(ms,[basicScore,basicScore],t).size,0);
 assert.equal(makeScoreIndex(ms.slice(0,1),[basicScore],t).size,0);
 assert.equal(makeScoreIndex(ms,[{...basicScore,player:'Alice Something',selection:'alice something'}],t).size,0);
 assert.equal(canon('MÁJA  Chwalińska'),'maja chwalinska');
});

test('Tennis365 lower-tour HTML parser preserves finished sets and server/break evidence',()=>{
 const html=`<div id="123" class="tennis_score_grid live"><span class="live_icon">L</span><strong>2</strong>
 <a href="/atp-itf-example" class="matches_grid_anchor">Match</a>
 <span id="hn-123">Alice Test</span><span id="an-123">Bea Sample</span>
 <span id="hs-123">0</span><span id="as-123">1</span><span id="csh-123">3</span><span id="csa-123">1</span>
 <span class="tennis_home_set_score"><span><span class="tennis_home_set_design">3</span></span></span>
 <span class="tennis_away_set_score"><span><span class="tennis_away_set_design">6</span></span></span>
 <span class="tennis_home_ball"><img src="ball.png"></span><span class="tennis_away_ball"></span>
 </div>`;
 const parsed=parseTennis365(html,t);
 assert.equal(parsed.length,2);
 assert.equal(parsed[0].source,'Tennis365');
 assert.equal(parsed[0].playerSets,0);
 assert.equal(parsed[0].opponentSets,1);
 assert.equal(parsed[0].netBreak,1);
 assert.equal(scoreReason(parsed[0],t),'eligible');
 assert.equal(parseTennis365(html.replace('6</span>', '2</span>'),t).length,0);
});

test('ESPN structural parser refuses completed matches and trusts best-of only in majors',()=>{
 const base={events:[{groupings:[{grouping:{slug:'mens-singles'},competitions:[{
  status:{type:{state:'in',completed:false},period:2},
  competitors:[{athlete:{displayName:'Alice Test'},possession:true,linescores:[{value:3},{value:3}]},
   {athlete:{displayName:'Bea Sample'},possession:false,linescores:[{value:6},{value:1}]}]
 }]}]}]};
 assert.equal(parseEspn(base,'ATP',t).length,2);
 assert.equal(scoreReason(parseEspn(base,'ATP',t)[0],t),'eligible');
 base.events[0].groupings[0].competitions[0].status.type.completed=true;
 assert.equal(parseEspn(base,'ATP',t).length,0);
});

function fakeDb(){
 const series=new Map(), alerts=new Map(), runs=[];
 function query(sql,args){
  if (sql.startsWith('SELECT payload')) return {first:async()=>series.has(args[0]) ? {payload:series.get(args[0])}:null};
  if (sql.startsWith('INSERT INTO series_state')) return {run:async()=>{series.set(args[0],args[1]);return {meta:{changes:1}}}};
  if (sql.startsWith('UPDATE series_state')) return {run:async()=>{series.set(args[2],args[0]);return {meta:{changes:1}}}};
  if (sql.startsWith('INSERT OR IGNORE INTO alerts')) return {run:async()=>{
   if (alerts.has(args[0])) return {meta:{changes:0}};
   alerts.set(args[0],{id:args[0],ticker:args[1],alert_at:args[3],observed_at:args[4],best_bid:null,final_result:null,push_status:'pending',push_attempts:0,last_attempt_at:null});
   return {meta:{changes:1}};
  }};
  if (sql.startsWith('SELECT push_status')) return {first:async()=>alerts.get(args[0])??null};
  if (sql.startsWith('UPDATE alerts SET push_status')) return {run:async()=>{let a=alerts.get(args.at(-1));a.push_status=args[0];if(sql.includes('push_attempts+1')){a.push_attempts++;a.last_attempt_at=args[1]}return {meta:{changes:1}}}};
  if (sql.startsWith('UPDATE alerts SET last_bid')) return {run:async()=>({meta:{changes:1}})};
  if (sql.startsWith('UPDATE alerts SET final_result')) return {run:async()=>({meta:{changes:1}})};
  if (sql.startsWith('SELECT id,ticker')) return {all:async()=>({results:[...alerts.values()]})};
  if (sql.startsWith('INSERT INTO runs')) return {run:async()=>{runs.push(args);return {meta:{changes:1}}}};
  throw new Error('UNEXPECTED SQL '+sql);
 }
 return {series,alerts,runs,prepare(sql){return {bind(...args){return query(sql,args)}}}};
}

test('complete, score-qualified reversal pushes exactly once; settlement and money never invented',async()=>{
 const db=fakeDb(); const env={TENNIS_DB:db,LIVE_TRADING:'false',ORDER_SUBMISSION_ENABLED:'false',READ_ONLY_KALSHI:'true',SPORTSEDGE_NTFY_TOPIC:'a'.repeat(32)};
 let ask=.22, notifies=0, marketsOk=true, scoreOk=true;
 const oldFetch=globalThis.fetch;
 globalThis.fetch=async (url, options={})=>{
  const s=String(url);
  const reply=obj=>({ok:true,status:200,json:async()=>obj});
  if(s.includes('/markets?')) {
   if (!marketsOk) return {ok:false,status:503};
   const series=new URL(s).searchParams.get('series_ticker');
   const markets=series===SERIES[0] ? [
    {ticker:'ALICE',event_ticker:'E',status:'open',yes_sub_title:'Alice Test',yes_bid_dollars:String(ask-.01),yes_ask_dollars:String(ask)},
    {ticker:'BEA',event_ticker:'E',status:'open',yes_sub_title:'Bea Sample',yes_bid_dollars:'.40',yes_ask_dollars:'.45'}
   ]:[];
   return reply({markets,cursor:''});
  }
  if(s.includes('espn.com')) {
   if(!scoreOk) return {ok:false,status:403};
   if(s.includes('/wta/')) return reply({events:[]});
   return reply({events:[{groupings:[{grouping:{slug:'mens-singles'},competitions:[{
    status:{type:{state:'in',completed:false},period:2},
    competitors:[{athlete:{displayName:'Alice Test'},possession:true,linescores:[{value:3},{value:3}]},
      {athlete:{displayName:'Bea Sample'},possession:false,linescores:[{value:6},{value:1}]}]
   }]}]}]});
  }
  if(s.includes('tennis365')) return {ok:true,status:200,text:async()=>''};
  if(s==='https://ntfy.sh/') {notifies++;return reply({id:'provider-accepted'})}
  if(s.includes('/markets/ALICE')) return reply({market:{status:'active'}});
  throw new Error('Unrecognized URL '+s);
 };
 try {
  for(let i=0;i<6;i++) {
   ask=[.25,.22,.03,.05,.07,.09][i];
   const x=await runCycle(env,t+i*60000); 
   if(i===0) assert.equal(x.alerts,0);
   if(i>=4) assert.equal(x.errors.length,0);
  }
  assert.equal(notifies,1);
  assert.equal(db.alerts.size,1);
  assert.equal([...db.alerts.values()][0].push_status,'accepted');
  assert.equal([...db.alerts.values()][0].push_attempts,1);
  assert.ok(JSON.parse(db.series.get(SERIES[0])).ALICE.lastAlertAt>0,'successful alert throttling persisted');
  assert.equal([...db.alerts.values()][0].final_result,null);
  assert.equal(db.runs.length,6);
  scoreOk=false; ask=.12;
  const result=await runCycle(env,t+6*60000);
  assert.equal(result.status,'degraded');
  assert.equal(notifies,1);
  marketsOk=false;
  const bad=await runCycle(env,t+7*60000);
  assert.equal(bad.status,'degraded');
  assert.equal(notifies,1);
 } finally {globalThis.fetch=oldFetch}
});

test('failed ntfy delivery retries only after 60s and persists accepted throttle',async()=>{
 const db=fakeDb(), env={TENNIS_DB:db,LIVE_TRADING:'false',ORDER_SUBMISSION_ENABLED:'false',READ_ONLY_KALSHI:'true',SPORTSEDGE_NTFY_TOPIC:'c'.repeat(32)};
 let ask=.22, sends=0;
 const original=globalThis.fetch;
 globalThis.fetch=async url=>{
  const u=String(url);
  const ok=obj=>({ok:true,status:200,json:async()=>obj});
  if(u.includes('/markets?')){
   const series=new URL(u).searchParams.get('series_ticker');
   return ok({markets:series===SERIES[0] ? [
    {ticker:'ALICE',event_ticker:'E',status:'open',yes_sub_title:'Alice Test',yes_bid_dollars:String(ask-.01),yes_ask_dollars:String(ask)},
    {ticker:'BEA',event_ticker:'E',status:'open',yes_sub_title:'Bea Sample',yes_bid_dollars:'.40',yes_ask_dollars:'.45'}]:[],cursor:''});
  }
  if(u.includes('espn.com'))return ok({events:u.includes('/wta/')?[]:[{groupings:[{grouping:{slug:'mens-singles'},competitions:[{
   status:{type:{state:'in',completed:false},period:2},competitors:[
    {athlete:{displayName:'Alice Test'},possession:true,linescores:[{value:3},{value:3}]},
    {athlete:{displayName:'Bea Sample'},possession:false,linescores:[{value:6},{value:1}]}]
   }]}]}]});
  if(u.includes('tennis365')) return {ok:true,status:200,text:async()=>''};
  if(u==='https://ntfy.sh/'){sends++; return sends===1?{ok:false,status:503,json:async()=>({})}:ok({id:'delivered'});}
  throw new Error('Unexpected upstream request');
 };
 try{
  for(const [i,p] of [.25,.22,.03,.05,.07].entries()){ask=p;await runCycle(env,t+i*60000)}
  assert.equal(sends,1);
  assert.equal(db.alerts.size,1);
  assert.equal([...db.alerts.values()][0].push_status,'failed');
  assert.equal([...db.alerts.values()][0].push_attempts,1);
  assert.equal(JSON.parse(db.series.get(SERIES[0])).ALICE.lastAlertAt,0);
  ask=.09;await runCycle(env,t+5*60000);
  assert.equal(sends,2,'one retry on documented provider failure');
  assert.equal([...db.alerts.values()][0].push_status,'accepted');
  assert.equal([...db.alerts.values()][0].push_attempts,2);
  assert.ok(JSON.parse(db.series.get(SERIES[0])).ALICE.lastAlertAt>0);
  ask=.11;await runCycle(env,t+6*60000);
  assert.equal(sends,2,'no duplicate accepted push');
 }finally{globalThis.fetch=original}
});

test('strict safety flags fail before network reads',async()=>{
 const db=fakeDb();
 await assert.rejects(runCycle({TENNIS_DB:db,LIVE_TRADING:'true',ORDER_SUBMISSION_ENABLED:'false',READ_ONLY_KALSHI:'true'},t),/read-only/);
 await assert.rejects(runCycle({TENNIS_DB:db,LIVE_TRADING:'false',ORDER_SUBMISSION_ENABLED:'true',READ_ONLY_KALSHI:'true'},t),/read-only/);
});