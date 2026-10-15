from datetime import datetime, timedelta, timezone
from dataclasses import replace
from types import SimpleNamespace

from sports_edge.models.kalshi_sports import KalshiSportMarket, classify_kalshi_market
from sports_edge.models.nba_margin_sleeper import nba_margin_tail
from sports_edge.models.sleeper_pipeline import contract_spec, scan_sleeper_markets
from sports_edge.data.sleeper_sources import SleeperSources

NOW=datetime(2026,11,20,18,tzinfo=timezone.utc)

def game(i,home="H",away="A",date=None,hs=120,aws=105):
    dt=date or NOW-timedelta(days=130-i*2)
    return {"source":"ESPN","sport":"NBA","season_type":"regular","event_id":str(i),
            "event_time":dt.isoformat(),"home_id":home,"away_id":away,
            "home_score":hs+i%5,"away_score":aws+i%3}

def history():
    # Independent games per team; avoid the synthetic shortcut of a single
    # fake head-to-head matchup repeated forty times.
    a=[game(i,home="H",away=f"T{i%8}",hs=120,aws=108) for i in range(32)]
    b=[game(i+100,home=f"T{i%8}",away="A",date=NOW-timedelta(days=130-i*2),hs=112,aws=105) for i in range(32)]
    return a+b

def market():
    return {"ticker":"KXNBASPREAD-26NOV20HMA-H19","series_ticker":"KXNBASPREAD",
            "event_ticker":"KXNBASPREAD-26NOV20HMA","status":"active","market_type":"binary",
            "title":"Home Team win by over 18.5 points","yes_sub_title":"Home Team win by over 18.5 points",
            "rules_primary":"If Home Team wins by more than 18.5 points in the Home vs Away Pro Basketball game, resolves Yes.",
            "rules_secondary":"Resolves based on the official final result including overtime.",
            "floor_strike":18.5,"strike_type":"greater",
            "occurrence_datetime":"2026-11-21T02:30:00Z","yes_ask_dollars":".15",
            "yes_bid_dollars":".13","no_ask_dollars":".88",
            "no_bid_dollars":".85","yes_ask_size_fp":"25","no_ask_size_fp":"30"}

def test_exact_19_plus_rule_and_disallow_ambiguous_or_regulation_only():
    m=market()
    assert classify_kalshi_market(m)==("NBA","Spread")
    spec=contract_spec(KalshiSportMarket("NBA","Spread",m))
    assert spec.rules_supported and spec.threshold==19 and spec.participant=="Home Team"
    for changed in ({"floor_strike":20.5},{"strike_type":"less"},{"rules_primary":"If Home Team wins by more than 16.5 points, resolves Yes."},
                    {"yes_sub_title":"Home Team wins game"},{"rules_secondary":"Only first half counts"}):
        assert not contract_spec(KalshiSportMarket("NBA","Spread",{**m,**changed})).rules_supported

def test_margin_tail_monotone_symmetric_and_never_confident_without_validation():
    h=history()
    p=nba_margin_tail(h,home_id="H",away_id="A",selected_id="H",before_time=NOW.isoformat())
    hard=nba_margin_tail(h,home_id="H",away_id="A",selected_id="H",before_time=NOW.isoformat(),line=30.5)
    opponent=nba_margin_tail(h,home_id="H",away_id="A",selected_id="A",before_time=NOW.isoformat())
    assert 0 < hard.probability < p.probability < 1
    assert 0 < opponent.probability < 1
    assert p.lower <= p.probability <= p.upper
    assert p.status=="UNVALIDATED_RESEARCH"
    assert "prospective_model_vs_executable_ask_validation_required" in p.reasons

def test_tail_rejects_invalid_history_and_target_future_scores():
    h=history()
    original=nba_margin_tail(h,home_id="H",away_id="A",selected_id="H",before_time=NOW.isoformat())
    future=game(9999,home="H",away="Z",date=NOW-timedelta(hours=2),hs=240,aws=0)
    assert nba_margin_tail(h+[future],home_id="H",away_id="A",selected_id="H",before_time=NOW.isoformat())==original
    assert nba_margin_tail(h+[dict(h[0],home_score=None)],home_id="H",away_id="A",selected_id="H",before_time=NOW.isoformat()).probability is None
    assert nba_margin_tail(h+[dict(h[0],home_score=199)],home_id="H",away_id="A",selected_id="H",before_time=NOW.isoformat()).probability is None
    assert nba_margin_tail(h[:5],home_id="H",away_id="A",selected_id="H",before_time=NOW.isoformat()).probability is None
    assert nba_margin_tail(h,home_id="H",away_id="A",selected_id="H",before_time=NOW.isoformat(),season_type="preseason").probability is None

class FakeClient:
    def market(self,ticker):
        return SimpleNamespace(data={"market":market()},received_at=NOW)
    def series(self,ticker):
        return SimpleNamespace(data={"series":{"ticker":ticker,"fee_type":"quadratic","fee_multiplier":1}})

class FakeSources:
    provenance=[]
    def nba_margin_context(self,m):
        return {"rows":history(),"start":"2026-11-20T23:00:00Z",
                "season_type":"regular","home_id":"H","away_id":"A",
                "home_aliases":["home team"],"away_aliases":["away team"],
                "missing":["injury_status_unconfirmed"]}

def test_scanner_retains_both_sides_and_does_not_auto_promote():
    report=scan_sleeper_markets([market()],coverage_complete=True,client=FakeClient(),sources=FakeSources(),now=NOW)
    assert len(report["records"])==2
    assert all(r["probability"] is not None for r in report["records"])
    assert sum(r["probability"] for r in report["records"])==1
    assert all(r["status"]!="QUALIFIED_VALUE" for r in report["records"])
    assert all(r["threshold"]==19 and r["start_verified"] for r in report["records"])

def test_scanner_blocks_game_already_started():
    class Started(FakeSources):
        def nba_margin_context(self,m):
            return {**super().nba_margin_context(m),"start":(NOW-timedelta(minutes=1)).isoformat()}
    report=scan_sleeper_markets([market()],coverage_complete=True,client=FakeClient(),sources=Started(),now=NOW)
    assert all(r["probability"] is None for r in report["records"])

def test_nba_schedule_exact_join_and_preseason_gate():
    s=SleeperSources()
    m=market()
    home={"homeAway":"home","id":"H","team":{"displayName":"Home Team","abbreviation":"HM","name":"Team"}}
    away={"homeAway":"away","id":"A","team":{"displayName":"Away Team","abbreviation":"A","name":"Team"}}
    event={"id":"EV","date":"2026-11-20T23:00:00Z","season":{"type":2},"competitions":[{"competitors":[away,home]}]}
    def get(url,params=None):
        if "scoreboard" in url:return {"events":[event]}
        return {"events":[]}
    s.get=get
    ctx=s.nba_margin_context(m)
    assert ctx["start"]==event["date"]
    assert ctx["home_id"]=="H" and ctx["away_id"]=="A"
    event["season"]["type"]=1
    try:s.nba_margin_context(m)
    except ValueError as exc:assert "preseason" in str(exc)
    else:assert False


def test_real_kalshi_shorthand_names_are_exact_official_aliases():
    m=dict(market(),event_ticker="KXNBASPREAD-26NOV20GSWLAL",
           ticker="KXNBASPREAD-26NOV20GSWLAL-LAL19",
           yes_sub_title="LA Lakers win by over 18.5 points")
    home={"homeAway":"home","id":"LAL","team":{
        "displayName":"Los Angeles Lakers","name":"Lakers",
        "location":"Los Angeles","abbreviation":"LAL"}}
    away={"homeAway":"away","id":"GSW","team":{
        "displayName":"Golden State Warriors","name":"Warriors",
        "location":"Golden State","abbreviation":"GSW"}}
    event={"id":"NBA1","date":"2026-11-20T23:00:00Z","season":{"type":2},
           "competitions":[{"competitors":[away,home]}]}
    source=SleeperSources()
    source.get=lambda url,params=None:({"events":[event]} if "scoreboard" in url else {"events":[]})
    ctx=source.nba_margin_context(m)
    assert "la lakers" in ctx["home_aliases"]
    assert "gs warriors" in ctx["away_aliases"]


def test_exact_20_and_31_point_margin_contracts_and_sides():
    for line,required in ((19.5,20),(30.5,31)):
        x=dict(market(),floor_strike=line,
               title=f"Home Team win by over {line} points",
               yes_sub_title=f"Home Team win by over {line} points",
               rules_primary=f"If Home Team wins by more than {line} points in the Home vs Away Pro Basketball game, resolves Yes.")
        spec=contract_spec(KalshiSportMarket("NBA","Spread",x))
        assert spec.rules_supported and spec.threshold==required
        class QuoteClient(FakeClient):
            def market(self,ticker):
                return SimpleNamespace(data={"market":x},received_at=NOW)
        scan=scan_sleeper_markets([x],coverage_complete=True,client=QuoteClient(),sources=FakeSources(),now=NOW)
        assert {r["side"] for r in scan["records"]}=={"YES","NO"}
        assert all(r["probability"] is not None and r["model"]=="nba_full_game_margin_tail_v2" for r in scan["records"])
        assert abs(sum(r["probability"] for r in scan["records"])-1)<1e-9
        assert all(r["status"]!="QUALIFIED_VALUE" for r in scan["records"])
        bad=dict(x,rules_primary="If Home Team wins by more than 18.5 points, resolves Yes.")
        assert not contract_spec(KalshiSportMarket("NBA","Spread",bad)).rules_supported


def test_nba_tail_probability_monotone_between_19_20_31():
    a=[nba_margin_tail(history(),home_id="H",away_id="A",selected_id="H",
                       before_time=NOW.isoformat(),line=line) for line in (18.5,19.5,30.5)]
    assert a[0].probability > a[1].probability > a[2].probability > 0
    assert all(m.status=="UNVALIDATED_RESEARCH" for m in a)
    assert nba_margin_tail(history(),home_id="H",away_id="A",selected_id="H",
                           before_time=NOW.isoformat(),line=25.5).probability is None


def test_historical_nba_tail_replay_has_distinct_threshold_labels():
    from scripts.run_nba_margin_replay import replay
    from datetime import timedelta
    # Synthetic history is intentionally insufficient for statistical promotion.
    past=history()
    for line,threshold in ((18.5,19),(19.5,20),(30.5,31)):
        report,records=replay(past,line=line)
        assert report["required_margin"]==threshold
        assert report["qualified_for_live"] is False
        assert report["no_promotions"] is True
        assert all(r["threshold"]==threshold for r in records)
