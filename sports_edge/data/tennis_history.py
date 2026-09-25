from __future__ import annotations

import csv
import io
from functools import lru_cache
from typing import Iterable

import requests


ARCHIVE_BASE = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"


def _urls(gender: str, years: Iterable[int]) -> list[str]:
    out: list[str] = []
    for year in years:
        if gender == "men":
            out.extend([
                f"{ARCHIVE_BASE}/atp/atp_matches_{year}.csv",
                f"{ARCHIVE_BASE}/atp/atp_matches_qual_chall_{year}.csv",
                f"{ARCHIVE_BASE}/atp/atp_matches_futures_{year}.csv",
            ])
        else:
            out.extend([
                f"{ARCHIVE_BASE}/wta/wta_matches_{year}.csv",
                f"{ARCHIVE_BASE}/wta/wta_matches_qual_itf_{year}.csv",
            ])
    return out


@lru_cache(maxsize=8)
def load_recent_tennis_rows(gender: str, start_year: int, end_year: int) -> tuple[dict, ...]:
    """Load recent public match-history rows for model research.

    Files that are not yet populated for a season are skipped rather than
    treated as zero-history evidence.
    """
    if gender not in {"men", "women"}:
        raise ValueError("gender must be men or women")

    rows: list[dict] = []
    session = requests.Session()
    session.headers.update({
        "User-Agent": "SportsEdgeReadOnly/1.0",
        "Accept": "text/csv,text/plain,*/*",
    })

    for url in _urls(gender, range(start_year, end_year + 1)):
        try:
            response = session.get(url, timeout=12)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            text = response.text.strip()
            if not text or "," not in text:
                continue
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                if not row:
                    continue
                if not row.get("winner_name") or not row.get("loser_name"):
                    continue
                rows.append(dict(row))
        except requests.RequestException:
            continue

    return tuple(rows)
