from datetime import datetime, timedelta, timezone

from sports_edge.models.tennis_price_movement import (
    build_price_movements, price_watch_tickers,
)

NOW=datetime(2026,10,13,19,0,tzinfo=timezone.utc)


def market(*, ticker="KXW15SHENZHEN-ZHALAN-ZHA", ask=.12, name="Ziyan Zhang"):
    return {
        "ticker":ticker, "yes_sub_title":name,
        "yes_ask_dollars":str(ask), "status":"open",
        "event_ticker":"KXW15SHENZHEN-ZHALAN",
    }


def candle(price, minute, *, spread=.01, quote=True, end=None):
    stamp=int((NOW-timedelta(minutes=minute)).timestamp()) if end is None else end
    return {
        "end_period_ts":stamp,
        "price":{"close_dollars":str(price)},
        "yes_ask":{"close_dollars":str(price)} if quote else {},
        "yes_bid":{"close_dollars":str(price-spread)} if quote else {},
        "volume_fp":"100",
    }


def path(prices, spreads=None):
    return tuple(candle(v,len(prices)-1-i,spread=(spreads[i] if spreads else .01))
                 for i,v in enumerate(prices))


def test_shenzhen_price_recovery_discovered_without_live_score_or_model():
    m=market(ask=.12)
    candles={m["ticker"]:path([.68,.48,.19,.03,.05,.08,.12])}
    rows=build_price_movements([m],candles,now=NOW)
    assert len(rows)==1
    assert rows[0].lane=="PRICE RECOVERY — TRACKING ONLY"
    assert rows[0].live_fair is None
    assert "UNVERIFIED" in rows[0].score
    assert rows[0].rebound_pp==9


def test_extreme_low_dip_discovered_without_live_score_or_model():
    m=market(ask=.03,name="Clemence Mercier")
    candles={m["ticker"]:path([.42,.25,.12,.04,.03])}
    rows=build_price_movements([m],candles,now=NOW)
    assert len(rows)==1
    assert rows[0].lane=="EXTREME DIP — TRACKING ONLY"
    assert rows[0].current_ask==.03


def test_no_quote_stale_quote_or_market_mismatch_fails_closed():
    m=market(ask=.12)
    history=list(path([.62,.3,.03,.05,.08,.12]))
    unquoted=list(history)
    unquoted[-1]=candle(.12,0,quote=False)
    assert not build_price_movements([m],{m["ticker"]:unquoted},now=NOW)
    stale=list(history)
    stale[-1]=candle(.12,0,end=int((NOW-timedelta(minutes=4)).timestamp()))
    assert not build_price_movements([m],{m["ticker"]:stale},now=NOW)
    assert not build_price_movements([market(ask=.24)],{m["ticker"]:history},now=NOW)
    assert not build_price_movements([m],{m["ticker"]:path([.12,.12,.12,.12])},now=NOW)


def test_price_watch_tickers_includes_unmodeled_cheap_matches():
    found=price_watch_tickers([
        market(ask=.02),
        market(ticker="KX-W50-SAI",ask=.19,name="Sara Saito"),
        market(ticker="KX-W50-MER",ask=.31,name="Clemence Mercier"),
        {**market(ticker="KX-closed",ask=.05),"status":"closed"},
    ])
    assert found==("KXW15SHENZHEN-ZHALAN-ZHA","KX-W50-SAI")



def test_lightweight_observer_persists_diagnostic_without_scores(monkeypatch,tmp_path):
    from scripts import tennis_price_only_worker as worker
    from sports_edge.models.tennis_prospective_store import ProspectiveResearchStore
    m=market(ask=.12)
    monkeypatch.setattr(worker,"fetch_open_tennis_match_markets",lambda strict: (m,))
    monkeypatch.setattr(
        worker,"fetch_tennis_candle_history",
        lambda tickers, **kwargs: {m["ticker"]:path([.68,.48,.19,.03,.05,.08,.12])}
    )
    seen=[]
    def fake_send(store, **kwargs):
        seen.extend(kwargs["price_observations"])
        return {"configured":True,"accepted":1,"failed":0,"errors":[]}
    monkeypatch.setattr(worker,"send_tennis_pushes",fake_send)
    store=ProspectiveResearchStore(tmp_path/"research")
    result=worker.poll_once(store,now=NOW)
    assert result["healthy"]
    assert result["open_contracts"]==1
    assert result["tracking_observations"]==1
    assert result["push_accepted"]==1
    assert len(seen)==1
    assert "UNVERIFIED" in seen[0].score
    assert (tmp_path/"research"/"latest_poll.json").exists()
    assert not store.signals


def test_lightweight_observer_fails_closed_when_market_coverage_errors(monkeypatch,tmp_path):
    from scripts import tennis_price_only_worker as worker
    from sports_edge.models.tennis_prospective_store import ProspectiveResearchStore
    def failed(strict):
        raise RuntimeError("one Tennis market series unavailable")
    monkeypatch.setattr(worker,"fetch_open_tennis_match_markets",failed)
    store=ProspectiveResearchStore(tmp_path/"research")
    result=worker.poll_once(store,now=NOW)
    assert not result["healthy"]
    assert not result["push_accepted"]
    assert result["errors"]
