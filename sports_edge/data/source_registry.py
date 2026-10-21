"""Honest inventory of sports-data sources available to SportEdge.

A source is marked integrated only when this repository has a client for it.
Websites that do not expose a verified authorized API remain research references;
the app must not imply that their data is being ingested automatically.
"""
from __future__ import annotations

import os
from typing import TypedDict


class SourceStatus(TypedDict):
    name: str
    role: str
    access: str
    status: str
    detail: str


def get_source_status() -> list[SourceStatus]:
    """Return deterministic source/configuration status without network calls."""
    odds_key = bool(os.getenv("THE_ODDS_API_KEY", "").strip())
    return [
        {
            "name": "Kalshi public API",
            "role": "Prediction-market catalog, quotes, candlesticks and live data",
            "access": "Public read-only API",
            "status": "integrated",
            "detail": "Existing KalshiPublicClient; no order-submission path is used by this registry.",
        },
        {
            "name": "ESPN public feeds",
            "role": "Live scoreboards and event context",
            "access": "Public feeds; availability may vary",
            "status": "integrated",
            "detail": "Existing NFL and tennis feed clients; stale/conflicting inputs should fail closed.",
        },
        {
            "name": "MLB Stats API",
            "role": "Baseball schedules, game logs and player/team context",
            "access": "Public API",
            "status": "integrated",
            "detail": "Existing MLB data client; verify individual fields and freshness at runtime.",
        },
        {
            "name": "The Odds API",
            "role": "Sportsbook odds and price comparison",
            "access": "API key; free-tier quota/coverage subject to provider",
            "status": "configured" if odds_key else "needs_key",
            "detail": "Existing OddsClient. " + (
                "THE_ODDS_API_KEY is present." if odds_key
                else "Set THE_ODDS_API_KEY in deployment secrets to enable requests."
            ),
        },
        {
            "name": "Betwatch",
            "role": "Odds comparison and line-movement research",
            "access": "Vendor access/API availability must be confirmed",
            "status": "research_only",
            "detail": "Not represented as an automated feed until an authorized API and terms are verified.",
        },
        {
            "name": "BettingPros",
            "role": "Player-prop research and comparison",
            "access": "Website; automated data access not verified",
            "status": "research_only",
            "detail": "No automated ingestion is claimed; use only an authorized data interface if offered.",
        },
        {
            "name": "ScoresAndOdds",
            "role": "Odds and prop-price cross-checks",
            "access": "Website; automated data access not verified",
            "status": "research_only",
            "detail": "No automated ingestion is claimed; use only an authorized data interface if offered.",
        },
        {
            "name": "StatMuse",
            "role": "Historical player/team statistics research",
            "access": "Website; supported API access not verified",
            "status": "research_only",
            "detail": "Not scraped by this app; model inputs should come from reproducible data feeds.",
        },
        {
            "name": "Official NHL data",
            "role": "Goal-scorer context, shots, rosters and game state",
            "access": "Public endpoints may change; confirm terms and schema",
            "status": "partial",
            "detail": "NHL Scorer Lab should not label a pick validated unless its required lineup, usage and goalie inputs are present.",
        },
        {
            "name": "College Football Data",
            "role": "NCAAF team/player statistics",
            "access": "API key and current quota may apply",
            "status": "research_only",
            "detail": "Not configured as a production source in this registry.",
        },
    ]


def source_status_summary() -> dict[str, int]:
    """Count source states for lightweight UI and smoke tests."""
    summary: dict[str, int] = {}
    for source in get_source_status():
        summary[source["status"]] = summary.get(source["status"], 0) + 1
    return summary
