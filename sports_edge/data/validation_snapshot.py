"""Reproducible historical research snapshot, NOT live validated probabilities.

Source: https://github.com/jwalton180-jpg/sports-edge-bot/actions/runs/38553755658
Run date 2026-10-14. Frozen test data follows an earlier training period.
"""
from __future__ import annotations

HISTORICAL_RESEARCH_SNAPSHOT = {
    "NHL": {
        "market": "player 2+ goals",
        "scope": "2024-25 full 920-skater NHL regular season",
        "historical_predictions": 31686,
        "holdout_predictions": 8042,
        "holdout_brier": 0.01708560744255579,
        "holdout_train_baseline_brier": 0.017466280842006243,
        "holdout_ece": 0.0007182551173215567,
        "holdout_accuracy": 0.9822183536433723,
        "holdout_always_no_accuracy": 0.9822183536433723,
        "historical_yes_calls": 0,
        "status": "RESEARCH_ONLY",
        "interpretation": "98.2% accuracy matches always-NO; no demonstrated ability to identify 2+ goal YES sleepers.",
    },
    "NCAAF": {
        "market": "home team winner",
        "scope": "2024-25 NCAA football ESPN scores, no injury/lineup adjustments",
        "historical_predictions": 1286,
        "holdout_predictions": 324,
        "holdout_brier": 0.19374571606430316,
        "holdout_train_baseline_brier": 0.2498971731473464,
        "holdout_ece": 0.06651381965885807,
        "holdout_accuracy": 0.6944444444444444,
        "status": "RESEARCH_ONLY",
        "interpretation": "Some historical signal, but calibration misses 0.06 ECE gate and pregame upset vulnerability inputs are missing.",
    },
    "Soccer": {
        "market": "EPL match total 2+ goals",
        "scope": "English Premier League 2023-25 ESPN scores only",
        "historical_predictions": 501,
        "holdout_predictions": 127,
        "holdout_brier": 0.19443301760306128,
        "holdout_train_baseline_brier": 0.1972991038406796,
        "holdout_ece": 0.09315883998450054,
        "holdout_accuracy": 0.7559055118110236,
        "status": "RESEARCH_ONLY",
        "interpretation": "Platt-calibrated holdout ECE fails; cannot generalize EPL data to MLS, Champions League or other soccer leagues.",
    },
}
