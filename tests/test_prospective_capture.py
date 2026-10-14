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
    pool=[market(f"KXNHL-{i}","NHL",4) for i in range(24)]
    other=[dict(m,yes_ask_dollars=".90") for m in reversed(pool)]
    a,d=select_markets(pool,now=NOW)
    b,_=select_markets(other,now=NOW)
    assert len(a)==14
    assert [r["ticker"] for r in a]==[r["ticker"] for r in b]
    assert d["subset_not_full_market_coverage"] and d["protocol"]==PROTOCOL
def test_excludes_started_and_far_future():
    pool=[market("early","NHL",-1),market("good","NCAAF",4),
          market("late","Soccer",75),market("missing","NHL",2)]
    del pool[3]["occurrence_datetime"]
    picked,_=select_markets(pool,now=NOW)
    assert [r["ticker"] for r in picked]==["good"]
def test_canonical_hash_and_expiration():
    assert canonical_bytes({"b":2,"a":1})==canonical_bytes({"a":1,"b":2})
    with pytest.raises(ValueError):canonical_bytes({"p":float("nan")})
    assert END_UTC>NOW
