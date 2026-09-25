from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TennisSourcePlan:
    name: str
    role: str
    notes: str

SOURCES = [
    TennisSourcePlan("Kalshi", "market + live milestone context", "Preferred for Kalshi-linked markets and supported live stats."),
    TennisSourcePlan("ITF official", "ITF schedule/results/live-score links", "Use official/authorized endpoints or links only; do not scrape restricted official data."),
    TennisSourcePlan("Jeff Sackmann archive", "historical ATP/WTA modeling", "CC BY-NC-SA; non-commercial research use and attribution required."),
    TennisSourcePlan("Match Charting Project", "point-level research", "Coverage is sparse/selective and license is non-commercial."),
]

LEVEL_WEIGHTS = {"ATP": 1.00, "WTA": 1.00, "CHALLENGER": 0.84, "ITF": 0.68}

def competition_level_weight(level: str) -> float:
    return LEVEL_WEIGHTS.get(str(level).upper(), 0.60)
