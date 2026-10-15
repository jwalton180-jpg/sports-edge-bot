from datetime import datetime,timezone,timedelta
from decimal import Decimal
import pytest

from sports_edge.models.combo_quote_archive import (
    SCHEMA, validate_sanitized_combo_snapshot, quote_depth_research,
    first_observed_paper_target
)

NOW=datetime(2026,10,15,7,tzinfo=timezone.utc)
TICK="KXMVECROSSCATEGORY-26OCT15-TST1"


def sample(*, t=NOW, yes=None, no=None, status="active"):
    yes = yes if yes is not None else [
        {"bid_dollars":"0.40","size_fp":"1.25"},
        {"bid_dollars":"0.30","size_fp":"3.50"},
    ]
    no = no if no is not None else [
        {"bid_dollars":"0.55","size_fp":"1.00"}]
    at=t.isoformat()
    return {
        "schema":SCHEMA,
        "kind":"PUBLIC_MARKET_ONLY_NON_EXECUTED_QUOTE_SNAPSHOT",
        "no_account_history":True, "no_API_credentials":True, "research_only":True,
        "recorded_at_utc":at,"snapshot_count":1,
        "markets":[{
            "ticker":TICK,"observed_at_utc":at,"market_status":status,
            "status":"OBSERVED_PUBLIC_BIDS_NOT_GUARANTEED_FILLED",
            "exact_combo_yes_bid_levels":yes,
            "exact_combo_no_bid_levels":no,
            "fills_verified":False,"full_size_exit_verified":False,
            "leg_tickers_and_sides":[{"ticker":"KXNFLGAME-X","side":"yes"}],
        }],
    }


def test_sanitized_snapshot_safely_parses_exact_bid_depth():
    row=validate_sanitized_combo_snapshot(sample())[0]
    assert row["ticker"]==TICK
    assert row["leg_count"]==1
    assert row["yes_levels"][0]==(Decimal(".40"),Decimal("1.25"))
    assert "leg_tickers_and_sides" not in row
    assert "account_id" not in row


def test_full_size_fractional_book_walk_and_fees_paper_only():
    row=validate_sanitized_combo_snapshot(sample())[0]
    res=quote_depth_research(row,side="YES",contracts="2.50",
        entry_total_including_fees=".50",exit_fee_per_contract=".02",now=NOW)
    assert res["status"]=="FULL_DEPTH_OBSERVED_NOT_EXECUTABLE_GUARANTEE"
    assert Decimal(res["paper_net_profit_for_quoted_part"])==Decimal(".825")
    assert res["contracts_with_book_depth"]=="2.50"
    assert res["executed"] is False and res["qualified_for_live"] is False


def test_partial_or_missing_side_quotes_do_not_fabricate_full_exit():
    row=validate_sanitized_combo_snapshot(sample())[0]
    partial=quote_depth_research(row,side="NO",contracts="6",entry_total_including_fees="1",
        exit_fee_per_contract=".01",now=NOW)
    assert partial["status"]=="PARTIAL_DEPTH_ONLY"
    assert partial["full_requested_depth"] is False
    assert Decimal(partial["contracts_not_priced"])==5
    row["no_levels"]=[]
    absent=quote_depth_research(row,side="NO",contracts="1",entry_total_including_fees=".1",
        exit_fee_per_contract=".01",now=NOW)
    assert absent["status"]=="NO_MATCHING_SIDE_BID"
    assert absent["paper_net_profit_for_quoted_part"] is None


def test_stale_data_never_claims_an_actionable_quote():
    row=validate_sanitized_combo_snapshot(sample())[0]
    out=quote_depth_research(row,side="YES",contracts="2",
        entry_total_including_fees=".4",exit_fee_per_contract="0",now=NOW+timedelta(minutes=3))
    assert out["status"]=="ARCHIVAL_SNAPSHOT_ONLY"
    assert not out["executed"]


def test_first_post_entry_snapshot_not_best_hindsight_or_postsettlement():
    points=[
        sample(t=NOW-timedelta(minutes=5),
               yes=[{"bid_dollars":".90","size_fp":"10"}]),
        sample(t=NOW+timedelta(minutes=1),
               yes=[{"bid_dollars":".10","size_fp":"10"}]),
        sample(t=NOW+timedelta(minutes=2),
               yes=[{"bid_dollars":".60","size_fp":"10"}]),
        sample(t=NOW+timedelta(minutes=6),
               yes=[{"bid_dollars":".98","size_fp":"10"}]),
    ]
    rows=[validate_sanitized_combo_snapshot(x)[0] for x in points]
    result=first_observed_paper_target(rows,ticker=TICK,side="YES",contracts="10",
              cost="2",exit_fee=".01",entry_utc=NOW.isoformat(),
              cutoff_utc=(NOW+timedelta(minutes=5)).isoformat(),target_roi="0.5")
    assert result["status"]=="FIRST_OBSERVED_FULL_DEPTH_PAPER_TARGET"
    assert result["at_utc"]==(NOW+timedelta(minutes=2)).isoformat()
    assert not result["executed"]
    result=first_observed_paper_target(rows,ticker=TICK,side="YES",contracts="10",
              cost="2",exit_fee=".01",entry_utc=NOW.isoformat(),
              cutoff_utc=(NOW+timedelta(minutes=2)).isoformat(),target_roi="0.5")
    assert result["status"]=="NO_OBSERVED_FULL_DEPTH_PAPER_TARGET"


@pytest.mark.parametrize("change",[
    lambda d:d.update(schema="OTHER"),
    lambda d:d.update(no_API_credentials=False),
    lambda d:d.update(no_account_history=False),
    lambda d:d.update(snapshot_count=2),
    lambda d:d["markets"][0].update(ticker="../portfolio/orders"),
    lambda d:d["markets"][0].update(fills_verified=True),
    lambda d:d["markets"][0].update(exact_combo_yes_bid_levels=[{"bid_dollars":"nan","size_fp":"2"}]),
    lambda d:d["markets"][0].update(exact_combo_yes_bid_levels=[{"bid_dollars":".35","size_fp":"2"},{"bid_dollars":".50","size_fp":"1"}]),
    lambda d:d["markets"][0].update(observed_at_utc=(NOW+timedelta(hours=1)).isoformat()),
])
def test_invalid_or_sensitive_payload_fails_closed(change):
    d=sample()
    change(d)
    with pytest.raises(ValueError):
        validate_sanitized_combo_snapshot(d)


def test_unsupported_quantity_price_or_fee_never_become_return():
    r=validate_sanitized_combo_snapshot(sample())[0]
    for kwargs in [
       dict(contracts="NaN",entry_total_including_fees=".3",exit_fee_per_contract=".01"),
       dict(contracts="1",entry_total_including_fees="0",exit_fee_per_contract=".01"),
       dict(contracts="1",entry_total_including_fees=".3",exit_fee_per_contract=None),
       dict(contracts="1",entry_total_including_fees=".3",exit_fee_per_contract="1.1"),
    ]:
        with pytest.raises(ValueError):
            quote_depth_research(r,side="YES",now=NOW,**kwargs)
