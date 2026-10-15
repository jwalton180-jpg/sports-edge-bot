from datetime import timedelta
from types import SimpleNamespace
import pytest
from sports_edge.models.nba_high_total import nba_high_total
from sports_edge.models.kalshi_sports import KalshiSportMarket,classify_kalshi_market
from sports_edge.models.sleeper_pipeline import contract_spec,scan_sleeper_markets
from sports_edge.models.prospective_capture import select_markets,model_family_supported
from tests.test_nba_margin_sleeper import NOW,history,FakeSources,market as margin_market


def market(line=244.5):
    m=margin_market()
    m.update(ticker=f"KXNBATOTAL-26NOV20HMA-{int(line)+1}",
             series_ticker="KXNBATOTAL",event_ticker="KXNBATOTAL-26NOV20HMA",
             title=f"Over {line} points",
             yes_sub_title=f"Over {line} points",
             rules_primary=f"Resolves Yes if over {line} total points are scored in the Home vs Away pro basketball game.",
             rules_secondary="Official final score includes overtime.",
             floor_strike=line)
    return m


class Client:
    def __init__(self,m):self.m=m
    def market(self,ticker):
        return SimpleNamespace(data={"market":self.m},received_at=NOW)
    def series(self,ticker):
        return SimpleNamespace(data={"series":{"ticker":ticker,"fee_type":"quadratic","fee_multiplier":1}})


def test_nba_high_total_no_exact_rules_and_executable_two_sides():
    m=market()
    assert classify_kalshi_market(m)==("NBA","Game Total")
    spec=contract_spec(KalshiSportMarket("NBA","Game Total",m))
    assert spec.rules_supported and spec.threshold==245
    report=scan_sleeper_markets([m],coverage_complete=True,
        sources=FakeSources(),client=Client(m),now=NOW)
    records=report["records"]
    assert len(records)==2 and {r["side"] for r in records}=={"YES","NO"}
    assert all(r["model"]=="nba_high_total_yes_no_v1" for r in records)
    assert all(r["status"]!="QUALIFIED_VALUE" for r in records)
    assert all(r["start_verified"] for r in records)
    assert abs(sum(r["probability"] for r in records)-1)<1e-9
    assert all(r["quote"]["entry_fee_per_contract"] is not None for r in records)


def test_high_total_prediction_monotone_and_no_more_likely_over_250_than_240():
    p=[nba_high_total(history(),home_id="H",away_id="A",before_time=NOW.isoformat(),line=x)
       for x in (239.5,244.5,249.5)]
    assert 0<p[2].probability<p[1].probability<p[0].probability<1
    assert all(r.status=="UNVALIDATED_RESEARCH" for r in p)
    assert all(r.lower<=r.probability<=r.upper for r in p)


def test_future_scores_conflicting_duplicates_preseason_and_unscoped_line_fail_closed():
    prior=history()
    a=nba_high_total(prior,home_id="H",away_id="A",before_time=NOW.isoformat(),line=244.5)
    future={**prior[0],"event_id":"FUTURE",
        "event_time":(NOW-timedelta(hours=3)).isoformat(),"home_score":200}
    assert nba_high_total(prior+[future],home_id="H",away_id="A",before_time=NOW.isoformat(),line=244.5)==a
    assert nba_high_total(prior+[dict(prior[0],home_score=200)],home_id="H",away_id="A",
                          before_time=NOW.isoformat(),line=244.5).probability is None
    assert nba_high_total(prior,home_id="H",away_id="A",before_time=NOW.isoformat(),line=244.5,
                          season_type="preseason").probability is None
    assert nba_high_total(prior,home_id="H",away_id="A",before_time=NOW.isoformat(),line=220.5).probability is None
    assert nba_high_total(prior[:5],home_id="H",away_id="A",before_time=NOW.isoformat(),line=244.5).probability is None


def test_wrong_total_threshold_rule_and_period_are_never_supported():
    m=market()
    for changed in ({"floor_strike":245.5},{"strike_type":"less"},
                    {"rules_primary":"Resolves Yes if over 220.5 points are scored."}):
        spec=contract_spec(KalshiSportMarket("NBA","Game Total",{**m,**changed}))
        assert not spec.rules_supported
    bad=dict(m,rules_secondary="Regulation only, excluding overtime.")
    report=scan_sleeper_markets([bad],coverage_complete=True,
        sources=FakeSources(),client=Client(bad),now=NOW)
    assert all(r["probability"] is None for r in report["records"])


def test_nba_high_total_capture_is_hash_sampled_not_price_ranked():
    rows=[]
    for line in (238.5,239.5,244.5,249.5,259.5,260.5):
        for i in range(3):
            rows.append(dict(market(line),ticker=f"KXNBATOTAL-test-{line}-{i}",
                             sports_edge_sport="NBA",
                             occurrence_datetime="2026-10-15T16:00:00Z",
                             yes_ask_dollars=str(.01+i*.2)))
    assert not model_family_supported(rows[0],"NBA")
    assert model_family_supported(rows[3],"NBA")
    from datetime import datetime,timezone
    now=datetime(2026,10,15,12,tzinfo=timezone.utc)
    picked,meta=select_markets(rows,now=now)
    assert len(picked)==3 and meta["sample_by_sport"]["NBA"]==3
    more,_=select_markets([dict(r,yes_ask_dollars=".99") for r in reversed(rows)],now=now)
    assert {r["ticker"] for r in picked}=={r["ticker"] for r in more}


def test_total_historical_replay_is_threshold_scoped_not_prospective():
    from scripts.run_nba_high_total_replay import replay
    for line in (239.5,244.5,249.5):
        result,rows=replay(history(),line=line)
        assert result.get("qualified_for_live") is False
        assert all(r["line"]==line for r in rows)
