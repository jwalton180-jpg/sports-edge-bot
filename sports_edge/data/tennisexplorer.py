from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from html import unescape
import re
from urllib.parse import urljoin

import requests


TENNIS_EXPLORER_BASE_URL = "https://www.tennisexplorer.com"
TENNIS_EXPLORER_HOME_URL = TENNIS_EXPLORER_BASE_URL + "/"


@dataclass(frozen=True)
class TennisExplorerPlayerContext:
    player: str
    profile_url: str
    rank: int | None
    highest_rank: int | None
    year_wins: int
    year_losses: int
    recent_wins: int
    recent_losses: int
    clay_wins: int
    clay_losses: int
    hard_wins: int
    hard_losses: int
    indoor_wins: int
    indoor_losses: int
    grass_wins: int
    grass_losses: int
    h2h_wins: int
    h2h_losses: int

    @property
    def recent_matches(self) -> int:
        return self.recent_wins + self.recent_losses

    @property
    def h2h_matches(self) -> int:
        return self.h2h_wins + self.h2h_losses


@dataclass(frozen=True)
class TennisExplorerPairContext:
    player_a: TennisExplorerPlayerContext
    player_b: TennisExplorerPlayerContext
    source: str = "TennisExplorer"


def _clean(value: str) -> str:
    value = re.sub(r"<script\b.*?</script>", " ", value or "", flags=re.I | re.S)
    value = re.sub(r"<style\b.*?</style>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<sup\b.*?</sup>", "", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(unescape(value).replace("\xa0", " ").split())


def _norm(value: str) -> str:
    value = unescape(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _tokens(value: str) -> list[str]:
    return [x for x in _norm(value).split() if x]


def _parse_record(value: str) -> tuple[int, int]:
    match = re.search(r"\b(\d+)\s*/\s*(\d+)\b", value or "")
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _profile_name(page: str) -> str:
    match = re.search(r"<h1[^>]*>(.*?)\s*-\s*profile\s*</h1>", page or "", flags=re.I | re.S)
    return _clean(match.group(1)) if match else ""


def _same_person(full_name: str, profile_name: str) -> bool:
    a, b = set(_tokens(full_name)), set(_tokens(profile_name))
    if not a or not b:
        return False
    overlap = len(a & b) / max(len(a), len(b))
    return overlap >= 0.66 and bool(a & b)


def _abbr_match_score(full_name: str, display: str, href: str = "") -> int:
    full = _tokens(full_name)
    shown = _tokens(display)
    slug = _tokens(href.replace("/player/", " "))
    if not full or not shown:
        return -999

    score = 0
    shown_words = [x for x in shown if len(x) > 1]
    shown_initials = [x for x in shown if len(x) == 1]
    for token in shown_words:
        if token in full:
            score += 4
        else:
            score -= 3
    if full[-1] in shown_words:
        score += 5
    if shown_initials and any(x == full[0][0] for x in shown_initials):
        score += 3
    for token in slug:
        if len(token) > 1 and token in full:
            score += 1
    if set(shown_words) and set(shown_words).issubset(set(full)):
        score += 3
    return score


def parse_tennisexplorer_directory(page: str) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for href, label in re.findall(
        r'<a\b[^>]*href=["\']([^"\']*/player/[^"\']+)["\'][^>]*>(.*?)</a>',
        page or "",
        flags=re.I | re.S,
    ):
        url = urljoin(TENNIS_EXPLORER_BASE_URL, href)
        if url in seen:
            continue
        text = _clean(label)
        if not text:
            continue
        seen.add(url)
        out.append((text, url))
    return tuple(out)


def resolve_tennisexplorer_profile(page: str, player: str) -> str | None:
    ranked = sorted(
        (
            (_abbr_match_score(player, label, url), label, url)
            for label, url in parse_tennisexplorer_directory(page)
        ),
        reverse=True,
    )
    if not ranked or ranked[0][0] < 7:
        return None
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0] and ranked[0][2] != ranked[1][2]:
        return None
    return ranked[0][2]


def _player_record_table(page: str) -> str:
    marker = re.search(r"Player(?:'|’)?s\s+record", page or "", flags=re.I)
    if not marker:
        return ""
    start = page.find("<table", marker.end())
    if start < 0:
        return ""
    end = page.find("</table>", start)
    return page[start:end + 8] if end >= 0 else ""


def _year_record(page: str, year: int) -> dict[str, tuple[int, int]]:
    table = _player_record_table(page)
    if not table:
        return {}
    for row in re.findall(r"<tr\b[^>]*>.*?</tr>", table, flags=re.I | re.S):
        cells = [_clean(x) for x in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, flags=re.I | re.S)]
        if not cells or cells[0] != str(year):
            continue
        values = [(_parse_record(x) if "/" in x else (0, 0)) for x in cells[1:7]]
        while len(values) < 6:
            values.append((0, 0))
        return {
            "summary": values[0],
            "clay": values[1],
            "hard": values[2],
            "indoor": values[3],
            "grass": values[4],
            "not_set": values[5],
        }
    return {}


def _set_winner_from_score(score_html: str) -> int | None:
    score = _clean(score_html)
    left_sets = right_sets = 0
    for a, b in re.findall(r"(?<!\d)(\d+)\s*-\s*(\d+)(?!\d)", score):
        ia, ib = int(a), int(b)
        if ia > ib:
            left_sets += 1
        elif ib > ia:
            right_sets += 1
    if left_sets >= 2 and left_sets > right_sets:
        return 0
    if right_sets >= 2 and right_sets > left_sets:
        return 1
    return None


def _display_matches_full(display: str, full_name: str) -> bool:
    return _abbr_match_score(full_name, display) >= 7


def _recent_and_h2h(
    page: str,
    *,
    player: str,
    opponent: str | None,
    as_of: date,
    recent_limit: int = 10,
) -> tuple[int, int, int, int]:
    marker = re.search(r"Player(?:'|’)?s\s+record", page or "", flags=re.I)
    tail = page[marker.start():] if marker else page
    outcomes: list[bool] = []
    h2h_wins = h2h_losses = 0
    today = f"{as_of.day:02d}.{as_of.month:02d}."

    for row in re.findall(r"<tr\b[^>]*>.*?</tr>", tail, flags=re.I | re.S):
        cells_html = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, flags=re.I | re.S)
        cells = [_clean(x) for x in cells_html]
        date_index = next((i for i, x in enumerate(cells) if re.fullmatch(r"\d{1,2}\.\d{1,2}\.", x)), None)
        if date_index is None:
            continue
        match_index = next((i for i, x in enumerate(cells) if " - " in x and "<strong" in cells_html[i].lower()), None)
        if match_index is None:
            continue
        match_text = cells[match_index]
        parts = [x.strip() for x in match_text.split(" - ", 1)]
        if len(parts) != 2:
            continue
        strong = re.search(r"<strong\b[^>]*>(.*?)</strong>", cells_html[match_index], flags=re.I | re.S)
        strong_text = _clean(strong.group(1)) if strong else ""
        if _display_matches_full(strong_text, parts[0]):
            player_side = 0
        elif _display_matches_full(strong_text, parts[1]):
            player_side = 1
        else:
            continue
        score_index = next(
            (
                i
                for i in range(match_index + 1, len(cells_html))
                if re.search(r"\d+\s*-\s*\d+", _clean(cells_html[i]))
            ),
            None,
        )
        if score_index is None:
            continue
        winner = _set_winner_from_score(cells_html[score_index])
        if winner is None:
            continue

        other_display = parts[1 - player_side]
        is_h2h = bool(opponent and _display_matches_full(other_display, opponent))
        if is_h2h and cells[date_index] == today:
            continue

        player_won = winner == player_side
        if len(outcomes) < recent_limit:
            outcomes.append(player_won)
        if is_h2h:
            if player_won:
                h2h_wins += 1
            else:
                h2h_losses += 1

    return sum(outcomes), len(outcomes) - sum(outcomes), h2h_wins, h2h_losses


def parse_tennisexplorer_player_context(
    page: str,
    *,
    player: str,
    opponent: str | None = None,
    profile_url: str = "",
    as_of: date | None = None,
) -> TennisExplorerPlayerContext | None:
    as_of = as_of or date.today()
    profile_name = _profile_name(page)
    if not profile_name or not _same_person(player, profile_name):
        return None

    plain = _clean(page)
    rank_match = re.search(
        r"Current/Highest\s+rank\s*-\s*singles:\s*(\d+)\.?\s*/\s*(\d+)\.?",
        plain,
        flags=re.I,
    )
    rank = int(rank_match.group(1)) if rank_match else None
    highest_rank = int(rank_match.group(2)) if rank_match else None
    records = _year_record(page, as_of.year)
    summary = records.get("summary", (0, 0))
    clay = records.get("clay", (0, 0))
    hard = records.get("hard", (0, 0))
    indoor = records.get("indoor", (0, 0))
    grass = records.get("grass", (0, 0))
    rw, rl, hw, hl = _recent_and_h2h(
        page, player=player, opponent=opponent, as_of=as_of
    )
    return TennisExplorerPlayerContext(
        player=player,
        profile_url=profile_url,
        rank=rank,
        highest_rank=highest_rank,
        year_wins=summary[0],
        year_losses=summary[1],
        recent_wins=rw,
        recent_losses=rl,
        clay_wins=clay[0],
        clay_losses=clay[1],
        hard_wins=hard[0],
        hard_losses=hard[1],
        indoor_wins=indoor[0],
        indoor_losses=indoor[1],
        grass_wins=grass[0],
        grass_losses=grass[1],
        h2h_wins=hw,
        h2h_losses=hl,
    )


def _get(url: str, *, timeout: float) -> str | None:
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "SportsEdgeReadOnly/1.0",
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return None


@lru_cache(maxsize=16)
def _directory_for_day(day_iso: str, timeout: float) -> str | None:
    return _get(TENNIS_EXPLORER_HOME_URL, timeout=timeout)


@lru_cache(maxsize=256)
def _pair_for_day(
    player_a: str,
    player_b: str,
    day_iso: str,
    timeout: float,
) -> TennisExplorerPairContext | None:
    as_of = date.fromisoformat(day_iso)
    directory = _directory_for_day(day_iso, timeout)
    if not directory:
        return None
    url_a = resolve_tennisexplorer_profile(directory, player_a)
    url_b = resolve_tennisexplorer_profile(directory, player_b)
    if not url_a or not url_b or url_a == url_b:
        return None
    page_a = _get(url_a, timeout=timeout)
    page_b = _get(url_b, timeout=timeout)
    if not page_a or not page_b:
        return None
    ctx_a = parse_tennisexplorer_player_context(
        page_a, player=player_a, opponent=player_b, profile_url=url_a, as_of=as_of
    )
    ctx_b = parse_tennisexplorer_player_context(
        page_b, player=player_b, opponent=player_a, profile_url=url_b, as_of=as_of
    )
    if ctx_a is None or ctx_b is None:
        return None
    return TennisExplorerPairContext(ctx_a, ctx_b)


def fetch_tennisexplorer_pair_context(
    player_a: str,
    player_b: str,
    *,
    as_of: date | None = None,
    timeout: float = 12.0,
) -> TennisExplorerPairContext | None:
    day = as_of or date.today()
    return _pair_for_day(player_a, player_b, day.isoformat(), float(timeout))
