from sports_edge.models.value_audit import audit_value

def market(price=12, status="open"):
    return {"ticker":"KXTEST-ABC", "status":status, "yes_ask":price, "no_ask":100-price}

def audit(m, p, **kw):
    return audit_value(m, side="YES", independent_probability=p, model_validated=True,
                       evidence_fresh=True, coverage_complete=True, rules_verified=True,
                       fee_fraction=0.01, **kw)

def test_supported_sleeper_qualifies():
    result=audit(market(12), 0.20)
    assert result.status=="QUALIFIED_VALUE" and result.edge > 0

def test_cheap_unjustified_rejected():
    result=audit(market(3), 0.01)
    assert result.status=="REJECTED"

def test_overpriced_favorite_rejected():
    result=audit(market(96), 0.75)
    assert result.status=="REJECTED"

def test_missing_evidence_fails_closed():
    result=audit_value(market(),side="YES",independent_probability=None,
                       model_validated=False,evidence_fresh=False,
                       coverage_complete=False,rules_verified=False)
    assert result.status=="RESEARCH_ONLY"
    assert "independent_probability_missing" in result.reasons

def test_missing_fees_not_invented():
    result=audit_value(market(),side="YES",independent_probability=0.9,
                       model_validated=True,evidence_fresh=True,
                       coverage_complete=True,rules_verified=True)
    assert result.status=="RESEARCH_ONLY"

def test_no_side_price_substitution():
    result=audit({"ticker":"T","status":"open","yes_bid":90}, 0.9)
    assert result.status=="RESEARCH_ONLY" and result.executable_price is None
