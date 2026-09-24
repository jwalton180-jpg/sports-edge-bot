from __future__ import annotations

from dataclasses import dataclass

from sports_edge.models.live_board import LiveSignal


@dataclass(frozen=True)
class KalshiTicket:
    title: str
    text: str


def _price_cents(signal: LiveSignal) -> int:
    return max(1, min(99, int(round(signal.market_probability * 100))))


def format_signal_ticket(signal: LiveSignal) -> KalshiTicket:
    """Return a copy-ready manual ticket. No order is submitted."""
    cents = _price_cents(signal)
    text = (
        f"KALSHI MANUAL TICKET\n"
        f"Ticker: {signal.ticker}\n"
        f"Selection: {signal.selection}\n"
        f"Side: YES\n"
        f"Observed ask: {cents}c\n"
        f"Model fair: {signal.fair_probability:.1%}\n"
        f"Edge: {signal.edge_points:+.1f} pp\n"
        f"Max price: {cents}c  # refresh before submitting\n"
        f"Contracts: [enter manually]\n"
        f"Status: {signal.status}\n"
    )
    return KalshiTicket(title=f"{signal.selection} · {signal.ticker}", text=text)


def format_combo_ticket(signals: list[LiveSignal], *, label: str = "Sports Edge Combo") -> KalshiTicket:
    """Return a copy-ready Kalshi Combo Builder leg list.

    Kalshi decides which legs are Combo-eligible. This function does not imply
    eligibility and never submits an RFQ or order.
    """
    lines = [
        "KALSHI COMBO BUILDER — COPY LIST",
        f"Name: {label}",
        "Paste/search each leg in Kalshi Combo Builder:",
    ]
    for i, signal in enumerate(signals, start=1):
        cents = _price_cents(signal)
        lines.append(
            f"{i}. YES | {signal.ticker} | {signal.selection} | observed {cents}c | fair {signal.fair_probability:.1%}"
        )
    lines.extend(
        [
            "",
            "Before submitting: refresh every price, verify each contract rule, and confirm Combo eligibility.",
            "Sports Edge does not submit the order.",
        ]
    )
    return KalshiTicket(title=label, text="\n".join(lines))
