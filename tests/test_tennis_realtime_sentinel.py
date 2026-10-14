import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from scripts import tennis_realtime_sentinel as mod

NOW=datetime(2026,10,14,10,0,tzinfo=timezone.utc)


def test_ticker_quotes_require_both_contemporaneous_sides():
    message={"type":"ticker","msg":{
        "market_ticker":"KXITFWMATCH-MATCH",
        "yes_bid_dollars":"0.11", "yes_ask_dollars":"0.14",
    }}
    assert mod.extract_ticker_quote(message)==("KXITFWMATCH-MATCH",.11,.14)
    assert mod.extract_ticker_quote({"type":"trade","msg":message["msg"]}) is None
    assert mod.extract_ticker_quote({"type":"ticker","msg":{
        "market_ticker":"x","yes_ask_dollars":".14"
    }}) is None
    assert mod.extract_ticker_quote({"type":"ticker","msg":{
        "market_ticker":"x","yes_bid_dollars":".16","yes_ask_dollars":".14"
    }}) is None
    assert mod.extract_ticker_quote({"type":"ticker","msg":{
        "market_ticker":"x","yes_bid":11,"yes_ask":14
    }})==("x",.11,.14)


def test_sentinel_requires_absolute_persistent_storage(tmp_path):
    try:
        mod.TennisSentinel("relative/storage")
        assert False, "should fail closed"
    except ValueError:
        pass
    obj=mod.TennisSentinel(tmp_path/"volume")
    assert obj.store.root == tmp_path/"volume"


def test_websocket_quote_is_not_pushable_without_verified_fresh_score(monkeypatch,tmp_path):
    obj=mod.TennisSentinel(tmp_path/"volume")
    obj.markets={"KX-TEST":{"ticker":"KX-TEST","yes_sub_title":"Player","status":"active"}}
    obj.candles={"KX-TEST":({"end_period_ts":int(NOW.timestamp())-60},)}
    sent=[]
    def fake_send(*args,**kwargs):
        sent.append(kwargs)
        return {"accepted":1,"failed":0}
    monkeypatch.setattr(mod,"send_tennis_pushes",fake_send)
    monkeypatch.setattr(mod,"build_price_movements",lambda *args,**kwargs:(
        SimpleNamespace(lane="PRICE RECOVERY — TRACKING ONLY",
                        ticker="KX-TEST",selection="Player",current_ask=.14,
                        observed_trough=.03,rebound_pp=11,quote_spread_pp=2),
    ))
    frame=SimpleNamespace(received_at=NOW,payload={
        "type":"ticker","msg":{"market_ticker":"KX-TEST",
                               "yes_bid_dollars":".12","yes_ask_dollars":".14"}})
    asyncio.run(obj.on_frame(frame))
    assert sent==[]
    obj.allowed={"KX-TEST":SimpleNamespace(score_label="1-6 6-1 3-0")}
    monkeypatch.setattr(mod,"survivability_reason",lambda state,now:"eligible")
    asyncio.run(obj.on_frame(frame))
    assert len(sent)==1
    assert sent[0]["allowed_tickers"]=={"KX-TEST"}
    assert "model edge not established" in sent[0]["price_observations"][0].score


def test_no_wide_or_crossed_quote_can_send(monkeypatch,tmp_path):
    obj=mod.TennisSentinel(tmp_path/"vol")
    obj.markets={"KX-TEST":{"ticker":"KX-TEST","status":"active"}}
    obj.candles={"KX-TEST":({"end_period_ts":int(NOW.timestamp())-60},)}
    obj.allowed={"KX-TEST":SimpleNamespace(score_label="1-6 6-1 3-0")}
    monkeypatch.setattr(mod,"survivability_reason",lambda state,now:"eligible")
    sent=[]
    monkeypatch.setattr(mod,"send_tennis_pushes",lambda *a,**k:sent.append(1))
    for bid,ask in [(".01",".14"),(".16",".14")]:
        asyncio.run(obj.on_frame(SimpleNamespace(received_at=NOW,payload={
            "type":"ticker","msg":{"market_ticker":"KX-TEST",
            "yes_bid_dollars":bid,"yes_ask_dollars":ask}
        })))
    assert not sent


def test_refresh_uses_separate_market_and_independent_score_sources(monkeypatch,tmp_path):
    obj=mod.TennisSentinel(tmp_path/"vol")
    market={"ticker":"KXITFWMATCH-SIDE","yes_sub_title":"Player",
            "status":"active","yes_ask_dollars":".14"}
    fake_state=object()
    monkeypatch.setattr(mod,"fetch_open_tennis_match_markets",lambda strict: (market,))
    monkeypatch.setattr(mod,"fetch_live_tennis_states",lambda timeout: (fake_state,))
    monkeypatch.setattr(mod,"qualified_tickers",lambda markets,states,now:
        ({market["ticker"]:fake_state}, {}))
    calls=[]
    def candles(tickers,**kwargs):
        calls.extend(tickers)
        return {market["ticker"]:({"end_period_ts":int(NOW.timestamp())-60},)}
    monkeypatch.setattr(mod,"fetch_tennis_candle_history",candles)
    tickers=asyncio.run(obj.refresh())
    assert tickers==(market["ticker"],)
    assert calls==[market["ticker"]]
    assert obj.metrics["score_eligible"]==1


def test_worker_rejects_trading_enabled(monkeypatch,tmp_path):
    monkeypatch.setenv("SPORTSEDGE_SENTINEL_DATA_DIR",str(tmp_path))
    monkeypatch.setenv("SPORTSEDGE_NTFY_TOPIC","A"*28)
    monkeypatch.setenv("LIVE_TRADING","true")
    try:
        mod.main()
        assert False, "must fail closed"
    except RuntimeError as exc:
        assert "Trading" in str(exc)
