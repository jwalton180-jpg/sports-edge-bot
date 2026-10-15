"""Cash-out-first Kalshi combo research, READ-ONLY and no inferred fill.

A Kalshi combo is its own independent RFQ/order-book market. Independent
component-ask multiplication is NOT a combo bid or an executable cash-out.
The profit opportunity is mark-to-market; only filled sales realize cash.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

MONEY = Decimal("0.01")


def _d(value: Any, field: str) -> Decimal:
    try:
        v = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError(f"{field}_invalid") from None
    if not v.is_finite():
        raise ValueError(f"{field}_invalid")
    return v


def _time(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("quote_time_invalid") from None
    if dt.tzinfo is None:
        raise ValueError("timezone_required")
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class CashoutQuoteResearch:
    combo_ticker: str
    status: str
    contracts_held: int
    quoted_sellable_contracts: int
    unpriced_remaining_contracts: int
    combo_bid_per_contract: Decimal | None
    estimated_net_sale_proceeds: Decimal | None
    prorated_original_cost: Decimal | None
    estimated_mark_profit: Decimal | None
    estimated_mark_roi: Decimal | None
    quote_age_seconds: float | None
    target_bid_for_25pct_roi: Decimal | None
    target_bid_for_50pct_roi: Decimal | None
    target_bid_for_100pct_roi: Decimal | None
    qualified_as_filled: bool
    caveats: tuple[str, ...]


def assess_combo_cashout_quote(
    *, combo_ticker: str, contracts_held: int, total_entry_cost: Any,
    combo_yes_bid: Any = None, available_bid_contracts: int | None = None,
    estimated_exit_fee_per_contract: Any = None,
    quote_observed_at: datetime | str | None = None,
    now: datetime | None = None, max_quote_age_s: float = 60.0,
) -> CashoutQuoteResearch:
    """Compute observed-quote MARKS, never cash proceeds or executed P&L.

    total_entry_cost includes original transaction fees. Exits require actual
    COMBO YES-side bid and its available depth, NOT the individual leg asks.
    If bids can buy only part of the position, expose just that part. No
    continuous quote monitoring or account access is performed.
    """
    if not isinstance(combo_ticker, str) or not combo_ticker.strip():
        raise ValueError("combo_ticker_required")
    if type(contracts_held) is not int or not 1 <= contracts_held <= 1_000_000:
        raise ValueError("invalid_contract_quantity")
    cost = _d(total_entry_cost, "entry_cost")
    if cost <= 0:
        raise ValueError("entry_cost_must_be_positive")
    if not 0 < max_quote_age_s <= 300:
        raise ValueError("bounded_quote_age_required")
    fee = (_d(estimated_exit_fee_per_contract, "exit_fee")
           if estimated_exit_fee_per_contract is not None else None)
    if fee is not None and not Decimal("0") <= fee < Decimal("1"):
        raise ValueError("invalid_exit_fee")
    bid = _d(combo_yes_bid, "combo_bid") if combo_yes_bid is not None else None
    if bid is not None and not Decimal("0") < bid < Decimal("1"):
        raise ValueError("invalid_combo_bid")
    if available_bid_contracts is not None and (
        type(available_bid_contracts) is not int or
        available_bid_contracts < 0 or available_bid_contracts > 10_000_000
    ):
        raise ValueError("invalid_available_combo_size")
    now_time = _time(now) if now is not None else datetime.now(timezone.utc)
    observed = _time(quote_observed_at)
    age = (now_time - observed).total_seconds() if observed is not None else None
    fresh = age is not None and 0 <= age <= max_quote_age_s
    count = min(contracts_held, available_bid_contracts or 0)
    full_size = count == contracts_held
    base = (
        "COMBO quote only: no verified fill, price persistence, depth beyond "
        "the stated level, or realized profit."
    )
    warnings = [
        base,
        "Exit prices must be for the exact Kalshi COMBO contract, not multiplied component prices.",
        "No forward cash-out probability or positive expected cash-out ROI is validated.",
    ]
    if bid is None:
        status = "MISSING_COMBO_BID"
    elif available_bid_contracts is None or count == 0:
        status = "MISSING_COMBO_LIQUIDITY"
    elif fee is None:
        status = "EXIT_FEES_UNVERIFIED"
    elif not fresh:
        status = "STALE_OR_UNDATED_COMBO_QUOTE"
    elif not full_size:
        status = "PARTIAL_SIZE_BID_ONLY"
    else:
        status = "FULL_SIZE_BID_OBSERVED_NOT_FILLED"
    proceeds = original_cost = pnl = roi = None
    if bid is not None and count and fee is not None and fresh:
        # Only the first bid level is evaluated; a partial sale leaves an open
        # residual that may lose the rest of its entry cost.
        proceeds = (bid - fee) * count
        original_cost = cost * Decimal(count) / Decimal(contracts_held)
        pnl = proceeds - original_cost
        roi = pnl / original_cost
        if not full_size:
            warnings.append(
                "The quoted size cannot liquidate the entire combo; unsold "
                "contracts remain exposed to loss."
            )
    def target(gain: str) -> Decimal | None:
        if fee is None:
            return None
        px = (cost * (Decimal("1") + Decimal(gain)) /
              Decimal(contracts_held)) + fee
        return px if px < Decimal("1") else None
    return CashoutQuoteResearch(
        combo_ticker=combo_ticker.strip(),
        status=status,
        contracts_held=contracts_held,
        quoted_sellable_contracts=count,
        unpriced_remaining_contracts=contracts_held-count,
        combo_bid_per_contract=bid,
        estimated_net_sale_proceeds=proceeds,
        prorated_original_cost=original_cost,
        estimated_mark_profit=pnl,
        estimated_mark_roi=roi,
        quote_age_seconds=age,
        target_bid_for_25pct_roi=target("0.25"),
        target_bid_for_50pct_roi=target("0.50"),
        target_bid_for_100pct_roi=target("1.00"),
        qualified_as_filled=False,
        caveats=tuple(warnings),
    )


@dataclass(frozen=True)
class UserReportedCashout:
    combo_ticker: str
    contracts_held_at_entry: int
    contracts_sold: int
    original_cost_allocated_to_sold: Decimal
    reported_net_sale_receipts: Decimal
    reported_profit: Decimal
    reported_roi: Decimal
    contracts_still_open: int
    verification: str
    outcome_grade_required_for_realized_sale: bool


def calculate_user_reported_cashout(
    *, combo_ticker: str, contracts_held_at_entry: int,
    contracts_sold: int, total_entry_cost_including_fees: Any,
    net_sale_receipts_after_fees: Any,
) -> UserReportedCashout:
    """Journal a user-reported FILLED cash-out, separate from game outcomes.

    Receipts must already be NET of exit fees. This avoids double-deducting
    fees and marks the record as user reported until exchange fills verify it.
    """
    if not isinstance(combo_ticker, str) or not combo_ticker.strip():
        raise ValueError("combo_ticker_required")
    if (type(contracts_held_at_entry) is not int or
        type(contracts_sold) is not int or
        not 1 <= contracts_sold <= contracts_held_at_entry <= 1_000_000):
        raise ValueError("invalid_filled_quantity")
    cost=_d(total_entry_cost_including_fees,"total_entry_cost")
    proceeds=_d(net_sale_receipts_after_fees,"net_receipts")
    if cost<=0 or proceeds<0:
        raise ValueError("invalid_cost_or_proceeds")
    allocated=cost*Decimal(contracts_sold)/Decimal(contracts_held_at_entry)
    profit=proceeds-allocated
    return UserReportedCashout(
        combo_ticker.strip(),contracts_held_at_entry,contracts_sold,
        allocated,proceeds,profit,profit/allocated,
        contracts_held_at_entry-contracts_sold,
        "USER_REPORTED_FILL_NOT_EXCHANGE_AUTHENTICATED",
        False,
    )


def first_observed_cashout_target(
    *, combo_ticker: str, contracts: int, cost: Any,
    quotes: list[dict], required_roi: float, fee_per_contract: Any,
) -> dict:
    """Forward-replay price observation only; NEVER call a simulated fill.

    Quotes are causally ordered complete COMBO bid snapshots. At each
    timestamp, decide from that quote alone. No hindsight max is the trade.
    """
    if not 0 < required_roi <= 10:
        raise ValueError("invalid_profit_threshold")
    if len(quotes)>100000:
        raise ValueError("bounded_quote_path_required")
    seen = None
    for q in quotes:
        t=_time(q.get("observed_at"))
        if t is None or (seen is not None and t<=seen):
            raise ValueError("quote_path_not_chronological")
        seen=t
        evidence=assess_combo_cashout_quote(
            combo_ticker=combo_ticker,contracts_held=contracts,
            total_entry_cost=cost,combo_yes_bid=q.get("combo_yes_bid"),
            available_bid_contracts=q.get("available_bid_contracts"),
            estimated_exit_fee_per_contract=fee_per_contract,
            quote_observed_at=t,now=t,
        )
        if (evidence.status=="FULL_SIZE_BID_OBSERVED_NOT_FILLED" and
            evidence.estimated_mark_roi is not None and
            evidence.estimated_mark_roi >= _d(required_roi,"required_roi")):
            return {"status":"FIRST_OBSERVED_PROFITABLE_COMBO_QUOTE_ONLY",
                    "observed_at":t.isoformat(),
                    "mark_roi":float(evidence.estimated_mark_roi),
                    "mark_profit":float(evidence.estimated_mark_profit),
                    "actual_fill_verified":False}
    return {"status":"NO_OBSERVED_FULL_SIZE_PROFIT_TARGET",
            "actual_fill_verified":False}
