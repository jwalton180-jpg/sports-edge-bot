from datetime import datetime,timezone,timedelta
from sports_edge.models.sleeper_quote_audit import screen_sleeper_quote,screen_sleeper_board

NOW=datetime(2026,10,14,12,tzinfo=timezone.utc)
MARKET={"ticker":"KX-ABC","status":"active","yes_ask_dollars":"0.40","no_ask_dollars":"0.61"}
def args():
    return dict(market=MARKET,outcome="player_shots_3_plus",
                expected_outcome="player_shots_3_plus",expected_ticker="KX-ABC",
                model_version="research_v1",independent_probability=.65,side="YES",
                market_observed_at=NOW.isoformat(),now=NOW,
                verified_fee_fraction=.02,rules_verified=True,
                market_mapping_verified=True)
def test_research_never_becomes_qualified():
    q=screen_sleeper_quote(**args())
    assert q.indicative_net_edge>0
    assert q.status=="RESEARCH_ONLY" and q.audit.status=="RESEARCH_ONLY"
    assert "independent_model_unvalidated" in q.audit.reasons
def test_stale_or_mismatched_market_has_no_edge():
    stale=dict(args(),market_observed_at=(NOW-timedelta(minutes=5)).isoformat())
    assert screen_sleeper_quote(**stale).indicative_net_edge is None
    mismatch=dict(args(),expected_ticker="KX-OTHER")
    assert screen_sleeper_quote(**mismatch).indicative_net_edge is None
    assert "incomplete_market_coverage" in screen_sleeper_quote(**mismatch).audit.reasons
def test_missing_fee_or_rules_cannot_be_actionable():
    a=dict(args(),verified_fee_fraction=None)
    assert screen_sleeper_quote(**a).audit.break_even is None
    b=dict(args(),rules_verified=False)
    assert screen_sleeper_quote(**b).indicative_net_edge is None
def test_sorting_and_side_rejection():
    board=screen_sleeper_board([args(),dict(args(),market_observed_at=None)],now=NOW)
    assert board[0].indicative_net_edge is not None
    try: screen_sleeper_quote(**dict(args(),side="MAYBE"))
    except ValueError: pass
    else: assert False
