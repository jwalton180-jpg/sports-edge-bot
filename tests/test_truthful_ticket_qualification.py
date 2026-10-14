from sports_edge.models.kalshi_sports import KalshiSideCandidate, choose_kalshi_ticket
from sports_edge.models.value_audit import audit_value

def candidate(ticker, price, event):
    return KalshiSideCandidate(sport="NHL", family="Player Goals", ticker=ticker,
        event_key=event, event_title=event, side="YES", selection=ticker,
        price=price, volume=1000, market={"ticker":ticker,"status":"open","yes_ask":round(price*100)})

def decision(c, fair):
    return audit_value(c.market, side="YES", independent_probability=fair,
        model_validated=True, evidence_fresh=True, coverage_complete=True,
        rules_verified=True, fee_fraction=0.01)

def test_unvalidated_discovery_never_creates_recommended_ticket():
    c=candidate("A",0.12,"E1")
    assert choose_kalshi_ticket([c],mode="longshot",target_legs=5)==[]

def test_sleeper_qualifies_cheap_unjustified_and_favorite_rejected():
    rows=[candidate("A",0.12,"E1"),candidate("B",0.03,"E2"),candidate("C",0.96,"E3")]
    audits={("A","YES"):decision(rows[0],0.2),("B","YES"):decision(rows[1],0.01),
            ("C","YES"):decision(rows[2],0.7)}
    assert [x.ticker for x in choose_kalshi_ticket(rows,mode="longshot",target_legs=10,
        verified_value=audits)]==["A"]

def test_quote_change_blocks_stale_value():
    c=candidate("A",0.12,"E1")
    audit=decision(c,0.2)
    newer=candidate("A",0.17,"E1")
    assert choose_kalshi_ticket([newer],mode="best",target_legs=2,
        verified_value={("A","YES"):audit})==[]

def test_event_and_ticker_deduplication():
    rows=[candidate("A",0.12,"E1"),candidate("B",0.2,"E1"),candidate("C",0.25,"E2")]
    audits={(c.ticker,c.side):decision(c,0.6) for c in rows}
    selected=choose_kalshi_ticket(rows,mode="best",target_legs=5,verified_value=audits)
    assert len(selected)==2 and len({c.event_key for c in selected})==2
