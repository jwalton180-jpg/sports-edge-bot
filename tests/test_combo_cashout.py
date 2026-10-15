from datetime import datetime, timedelta, timezone
from decimal import Decimal
import pytest

from sports_edge.models.combo_cashout import (
    assess_combo_cashout_quote,
    calculate_user_reported_cashout,
    first_observed_cashout_target,
)

NOW = datetime(2026,10,15,3,tzinfo=timezone.utc)


def quote(**changes):
    kwargs=dict(
        combo_ticker="COMBO-ABC",
        contracts_held=10,
        total_entry_cost="2.00",
        combo_yes_bid=".50",
        available_bid_contracts=10,
        estimated_exit_fee_per_contract=".01",
        quote_observed_at=NOW-timedelta(seconds=5),
        now=NOW,
    )
    kwargs.update(changes)
    return assess_combo_cashout_quote(**kwargs)


def test_profit_before_settlement_does_not_require_all_legs_to_win():
    mark=quote()
    assert mark.status=="FULL_SIZE_BID_OBSERVED_NOT_FILLED"
    assert mark.estimated_net_sale_proceeds==Decimal("4.90")
    assert mark.estimated_mark_profit==Decimal("2.90")
    assert mark.estimated_mark_roi==Decimal("1.45")
    assert mark.target_bid_for_25pct_roi==Decimal(".26")
    assert mark.target_bid_for_50pct_roi==Decimal(".31")
    assert mark.target_bid_for_100pct_roi==Decimal(".41")
    assert not mark.qualified_as_filled
    assert "no verified fill" in mark.caveats[0].lower()


def test_per_leg_prices_are_not_an_executable_combo_quote():
    mark=quote(combo_yes_bid=None)
    assert mark.status=="MISSING_COMBO_BID"
    assert mark.estimated_mark_profit is None
    assert not mark.qualified_as_filled
    mark=quote(available_bid_contracts=None)
    assert mark.status=="MISSING_COMBO_LIQUIDITY"
    assert mark.estimated_mark_profit is None


def test_partial_liquidity_does_not_count_unsold_legs_as_realized():
    mark=quote(available_bid_contracts=3)
    assert mark.status=="PARTIAL_SIZE_BID_ONLY"
    assert mark.quoted_sellable_contracts==3
    assert mark.unpriced_remaining_contracts==7
    assert mark.prorated_original_cost==Decimal("0.60")
    assert mark.estimated_net_sale_proceeds==Decimal("1.47")
    assert mark.estimated_mark_profit==Decimal(".87")
    assert not mark.qualified_as_filled


def test_stale_quote_and_missing_fees_do_not_produce_an_eligible_paper_mark():
    for changed,expected in (
        ({"quote_observed_at":NOW-timedelta(seconds=61)},"STALE_OR_UNDATED_COMBO_QUOTE"),
        ({"quote_observed_at":None},"STALE_OR_UNDATED_COMBO_QUOTE"),
        ({"quote_observed_at":NOW+timedelta(seconds=10)},"STALE_OR_UNDATED_COMBO_QUOTE"),
        ({"estimated_exit_fee_per_contract":None},"EXIT_FEES_UNVERIFIED"),
        ({"available_bid_contracts":0},"MISSING_COMBO_LIQUIDITY"),
    ):
        m=quote(**changed)
        assert m.status==expected
        assert m.estimated_mark_profit is None


@pytest.mark.parametrize("changes",[
    {"contracts_held":0},{"contracts_held":1.5},{"total_entry_cost":"nan"},
    {"combo_yes_bid":0},{"combo_yes_bid":1},{"combo_yes_bid":"NaN"},
    {"available_bid_contracts":-1},{"available_bid_contracts":1.5},
    {"estimated_exit_fee_per_contract":-1},{"estimated_exit_fee_per_contract":1},
    {"quote_observed_at":"2026-10-14T00:00:00"},
    {"max_quote_age_s":999},
])
def test_malformed_quote_and_unsafe_assumptions_fail_closed(changes):
    with pytest.raises(ValueError):
        quote(**changes)


def test_user_reported_completed_cashout_is_not_terminal_result_win():
    report=calculate_user_reported_cashout(
        combo_ticker="COMBO-NFL",
        contracts_held_at_entry=10,
        contracts_sold=10,
        total_entry_cost_including_fees="3.50",
        net_sale_receipts_after_fees="50.50",
    )
    assert report.reported_profit==Decimal("47.00")
    assert report.reported_roi>Decimal("13")
    assert report.contracts_still_open==0
    assert report.verification=="USER_REPORTED_FILL_NOT_EXCHANGE_AUTHENTICATED"
    assert report.outcome_grade_required_for_realized_sale is False


def test_user_reported_partial_exit_is_not_entire_position_profit():
    report=calculate_user_reported_cashout(
        combo_ticker="COMBO-TENNIS",
        contracts_held_at_entry=10,
        contracts_sold=4,
        total_entry_cost_including_fees="3.00",
        net_sale_receipts_after_fees="2.00",
    )
    assert report.original_cost_allocated_to_sold==Decimal("1.20")
    assert report.reported_profit==Decimal(".80")
    assert report.contracts_still_open==6
    with pytest.raises(ValueError):
        calculate_user_reported_cashout(
            combo_ticker="X",contracts_held_at_entry=10,contracts_sold=11,
            total_entry_cost_including_fees=3,net_sale_receipts_after_fees=1)


def test_earliest_profitable_bid_is_causal_and_never_marked_as_filled():
    q=[
        {"observed_at":(NOW+timedelta(minutes=i)).isoformat(),
         "combo_yes_bid":bid,"available_bid_contracts":depth}
        for i,(bid,depth) in enumerate(((.21,10),(.36,5),(.32,10),(.95,10)))
    ]
    result=first_observed_cashout_target(
        combo_ticker="COMBO-ABC",contracts=10,cost=2,
        quotes=q,required_roi=.50,fee_per_contract=".01")
    assert result["status"]=="FIRST_OBSERVED_PROFITABLE_COMBO_QUOTE_ONLY"
    assert result["observed_at"]==(NOW+timedelta(minutes=2)).isoformat()
    assert not result["actual_fill_verified"]
    no_full=first_observed_cashout_target(
        combo_ticker="COMBO-ABC",contracts=10,cost=2,
        quotes=q[:2],required_roi=.5,fee_per_contract=".01")
    assert no_full["status"]=="NO_OBSERVED_FULL_SIZE_PROFIT_TARGET"


def test_no_quote_path_hindsight_wins_or_reorder():
    assert first_observed_cashout_target(
        combo_ticker="C",contracts=1,cost=".3",quotes=[],
        required_roi=.25,fee_per_contract=".01")["actual_fill_verified"] is False
    with pytest.raises(ValueError,match="chronological"):
        first_observed_cashout_target(
            combo_ticker="C",contracts=1,cost=".3",
            quotes=[
                {"observed_at":NOW.isoformat(),"combo_yes_bid":.4,"available_bid_contracts":1},
                {"observed_at":NOW.isoformat(),"combo_yes_bid":.7,"available_bid_contracts":1},
            ],required_roi=.25,fee_per_contract=".01")
