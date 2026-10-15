from datetime import datetime,timezone,timedelta
import json
import pytest
from sports_edge.models.prospective_capture import select_markets,canonical_bytes,PROTOCOL,END_UTC
NOW=datetime(2026,10,15,12,tzinfo=timezone.utc)
def market(ticker,sport,offset):
    return {"ticker":ticker,"sports_edge_sport":sport,
            "occurrence_datetime":(NOW+timedelta(hours=offset)).isoformat(),
            "status":"active","yes_ask_dollars":".11"}
def test_stable_choice_is_price_and_order_independent():
    pool=[market(f"KXNHLGOAL-{i}","NHL",4) for i in range(24)]
    other=[dict(m,yes_ask_dollars=".90") for m in reversed(pool)]
    a,d=select_markets(pool,now=NOW)
    b,_=select_markets(other,now=NOW)
    assert len(a)==14
    assert [r["ticker"] for r in a]==[r["ticker"] for r in b]
    assert d["subset_not_full_market_coverage"] and d["protocol"]==PROTOCOL
def test_excludes_started_and_far_future():
    pool=[market("KXNHLGOAL-early","NHL",-1),market("KXNCAAFGAME-good","NCAAF",4),
          market("KXEPLTOTAL-late","Soccer",125),market("KXNHLPTS-missing","NHL",2)]
    del pool[3]["occurrence_datetime"]
    picked,_=select_markets(pool,now=NOW)
    assert [r["ticker"] for r in picked]==["KXNCAAFGAME-good"]
def test_weekend_sports_capture_uses_extended_but_still_bounded_horizon():
    pool=[market("KXEPLTOTAL-weekend","Soccer",75),
          market("KXEPLTOTAL-too_far","Soccer",97)]
    picked,meta=select_markets(pool,now=NOW)
    assert [r["ticker"] for r in picked]==["KXEPLTOTAL-weekend"]
    assert meta["eligible_by_sport"]["Soccer"]==1

def test_canonical_hash_and_expiration():
    assert canonical_bytes({"b":2,"a":1})==canonical_bytes({"a":1,"b":2})
    with pytest.raises(ValueError):canonical_bytes({"p":float("nan")})
    assert END_UTC>NOW


def test_nba_19_20_31_margin_sampling_is_price_blind_and_stratified():
    from sports_edge.models.prospective_capture import model_family_supported
    rows=[dict(market(f"KXNBASPREAD-{line}-{i}","NBA",24),
               floor_strike=line,strike_type="greater") for line in (18.5,19.5,30.5) for i in range(15)]
    wrong=[dict(rows[0],ticker="KXNBASPREAD-wrong",floor_strike=16.5)]
    assert all(model_family_supported(r,"NBA") for r in rows)
    assert not model_family_supported(wrong[0],"NBA")
    chosen,meta=select_markets(rows+wrong,now=NOW)
    repriced=[dict(r,yes_ask_dollars=".99") for r in reversed(rows)]
    chosen_again,_=select_markets(repriced,now=NOW)
    assert len(chosen)==6 and meta["sample_by_sport"]["NBA"]==6
    assert sorted(m["floor_strike"] for m in chosen)==[18.5,18.5,19.5,19.5,30.5,30.5]
    assert {m["ticker"] for m in chosen}=={m["ticker"] for m in chosen_again}

