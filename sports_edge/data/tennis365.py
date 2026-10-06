from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
import re

import requests


TENNIS365_LIVE_URL = "https://livescore.tennis365.com/"
TENNIS365_BASE_URL = "https://livescore.tennis365.com"


@dataclass(frozen=True)
class Tennis365LiveMatch:
    match_id: str
    home: str
    away: str
    tour: str
    period: int
    home_sets: int
    away_sets: int
    home_games: int
    away_games: int
    completed_sets: tuple[tuple[int, int], ...]
    home_serving: bool | None
    away_serving: bool | None
    home_point: str | None
    away_point: str | None
    start_at: datetime | None
    source_url: str


@dataclass(frozen=True)
class Tennis365PlayerContext:
    player: str
    rank: int | None
    ranking_points: int | None
    recent_wins: int
    recent_losses: int

    @property
    def recent_matches(self) -> int:
        return self.recent_wins + self.recent_losses


def _clean_html_text(value: str) -> str:
    value = re.sub(r"<script\b.*?</script>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<style\b.*?</style>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(unescape(value).replace("\xa0", " ").split())


def _span_text(block: str, *, element_id: str | None = None, class_name: str | None = None) -> str | None:
    if element_id:
        pattern = rf'<span\b[^>]*\bid=["\']{re.escape(element_id)}["\'][^>]*>(.*?)</span>'
    elif class_name:
        pattern = rf'<span\b[^>]*\bclass=["\'][^"\']*\b{re.escape(class_name)}\b[^"\']*["\'][^>]*>(.*?)</span>'
    else:
        return None
    match = re.search(pattern, block, flags=re.I | re.S)
    if not match:
        return None
    text = _clean_html_text(match.group(1))
    return text or None


def _int_text(value: str | None) -> int | None:
    try:
        if value is None or value.strip() in {"", "-"}:
            return None
        return int(float(value.strip()))
    except (TypeError, ValueError):
        return None


def _match_blocks(page: str) -> list[tuple[str, str]]:
    starts = list(re.finditer(
        r'<div\s+id=["\'](?P<id>\d+)["\']\s+class=["\'][^"\']*\btennis_score_grid\b[^"\']*["\'][^>]*>',
        page or "",
        flags=re.I,
    ))
    out: list[tuple[str, str]] = []
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(page)
        out.append((match.group("id"), page[match.start():end]))
    return out


def _set_scores(block: str, class_name: str, needed: int) -> tuple[int, ...]:
    wrapper = re.search(
        rf'<span\b[^>]*class=["\'][^"\']*\b{re.escape(class_name)}\b[^"\']*["\'][^>]*>(.*?)</span>\s*</span>',
        block,
        flags=re.I | re.S,
    )
    text = wrapper.group(1) if wrapper else block
    values = [
        _int_text(_clean_html_text(raw))
        for raw in re.findall(
            r'<span\b[^>]*class=["\'][^"\']*\btennis_(?:home|away)_set_design\b[^"\']*["\'][^>]*>(.*?)</span>',
            text,
            flags=re.I | re.S,
        )
    ]
    return tuple(value for value in values if value is not None)[:needed]


def _point_text(block: str, side: str) -> str | None:
    for class_name in (
        f"tennis_{side}_point_score",
        f"tennis_{side}_points",
        f"{side}_point_score",
        f"{side}_score_point",
    ):
        value = _span_text(block, class_name=class_name)
        if value and value != "-":
            return value
    return None


def _parse_start(block: str) -> datetime | None:
    match = re.search(r'<span\b[^>]*class=["\'][^"\']*\bgmtdatetime\b[^"\']*["\'][^>]*>(.*?)</span>', block, flags=re.I | re.S)
    if not match:
        return None
    raw = _clean_html_text(match.group(1))
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M %p"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _tour_from_href(href: str) -> str:
    low = href.lower()
    if "-challenger-" in low:
        return "CHALLENGER"
    if "-itf-" in low:
        return "ITF"
    if "-wta-" in low:
        return "WTA"
    return "ATP"


def parse_tennis365_live_matches(page: str) -> tuple[Tennis365LiveMatch, ...]:
    """Parse only genuinely live singles blocks from Tennis365 server HTML."""
    out: list[Tennis365LiveMatch] = []
    for match_id, block in _match_blocks(page):
        if "live_icon" not in block:
            continue
        status = re.search(r'<strong>\s*(?:Set\s*)?(\d+)\s*</strong>', block, flags=re.I)
        if not status:
            continue
        period = int(status.group(1))
        if period < 1 or period > 5:
            continue

        href_match = re.search(r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*class=["\'][^"\']*\bmatches_grid_anchor\b', block, flags=re.I)
        href = href_match.group(1) if href_match else ""
        tour = _tour_from_href(href)
        # Lower-tour expansion is the purpose of this source; ATP/WTA remain
        # on ESPN unless a future explicit integration chooses otherwise.
        if tour not in {"CHALLENGER", "ITF"}:
            continue

        home = _span_text(block, element_id=f"hn-{match_id}")
        away = _span_text(block, element_id=f"an-{match_id}")
        home_sets = _int_text(_span_text(block, element_id=f"hs-{match_id}"))
        away_sets = _int_text(_span_text(block, element_id=f"as-{match_id}"))
        home_games = _int_text(_span_text(block, element_id=f"csh-{match_id}"))
        away_games = _int_text(_span_text(block, element_id=f"csa-{match_id}"))
        if not home or not away or "/" in home or "/" in away:
            continue
        if None in (home_sets, away_sets, home_games, away_games):
            continue

        needed = max(0, period - 1)
        home_prior = _set_scores(block, "tennis_home_set_score", needed)
        away_prior = _set_scores(block, "tennis_away_set_score", needed)
        completed_sets = tuple(zip(home_prior, away_prior))
        # If markup does not expose all completed set games, sets-won totals
        # alone are not enough to claim a turnaround. Fail closed on the row.
        if len(completed_sets) != needed:
            continue

        home_ball = re.search(r'<span\b[^>]*class=["\'][^"\']*\btennis_home_ball\b[^"\']*["\'][^>]*>(.*?)</span>', block, flags=re.I | re.S)
        away_ball = re.search(r'<span\b[^>]*class=["\'][^"\']*\btennis_away_ball\b[^"\']*["\'][^>]*>(.*?)</span>', block, flags=re.I | re.S)
        home_has_ball = bool(home_ball and re.search(r'<img\b', home_ball.group(1), flags=re.I))
        away_has_ball = bool(away_ball and re.search(r'<img\b', away_ball.group(1), flags=re.I))
        if home_has_ball == away_has_ball:
            home_serving = away_serving = None
        else:
            home_serving, away_serving = home_has_ball, away_has_ball

        out.append(Tennis365LiveMatch(
            match_id=match_id,
            home=home,
            away=away,
            tour=tour,
            period=period,
            home_sets=home_sets,
            away_sets=away_sets,
            home_games=home_games,
            away_games=away_games,
            completed_sets=completed_sets,
            home_serving=home_serving,
            away_serving=away_serving,
            home_point=_point_text(block, "home"),
            away_point=_point_text(block, "away"),
            start_at=_parse_start(block),
            source_url=(TENNIS365_BASE_URL + href) if href.startswith("/") else href,
        ))
    return tuple(out)


def fetch_tennis365_live_matches(*, timeout: float = 12.0) -> tuple[Tennis365LiveMatch, ...]:
    try:
        response = requests.get(
            TENNIS365_LIVE_URL,
            headers={
                "User-Agent": "SportsEdgeReadOnly/1.0",
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return parse_tennis365_live_matches(response.text)
    except requests.RequestException:
        return ()


def _normalize_name(value: str) -> str:
    value = unescape(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _name_matches(a: str, b: str) -> bool:
    na, nb = _normalize_name(a), _normalize_name(b)
    if na == nb:
        return True
    ta, tb = na.split(), nb.split()
    return len(ta) >= 2 and len(tb) >= 2 and ta[0] == tb[0] and ta[-1] == tb[-1]


def parse_tennis365_player_context(
    page: str,
    *,
    player_a: str,
    player_b: str,
) -> tuple[Tennis365PlayerContext, Tennis365PlayerContext]:
    """Parse static ranking plus prior-form evidence from a match detail page.

    Recent form intentionally ignores the current match block: only dated rows
    in each player's Latest Games section are counted.
    """
    text = page or ""
    ranking_anchor = re.search(r'(?:ATP|WTA)\s+World\s+Rankings', text, flags=re.I)
    ranking_rows: dict[str, tuple[int, int]] = {}
    if ranking_anchor:
        tail = text[ranking_anchor.start():ranking_anchor.start() + 50000]
        table_end = tail.find("</table>")
        if table_end >= 0:
            tail = tail[:table_end + 8]
        for row in re.findall(r'<tr\b[^>]*>.*?</tr>', tail, flags=re.I | re.S):
            cells = [_clean_html_text(x) for x in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>', row, flags=re.I | re.S)]
            if len(cells) < 3:
                continue
            nums = [int(x.replace(",", "")) for x in cells if re.fullmatch(r"[\d,]+", x)]
            if len(nums) < 2:
                continue
            rank, points = nums[0], nums[-1]
            name = next((x for x in cells if not re.fullmatch(r"[\d,]+", x) and len(x) > 2 and len(x) < 80), "")
            if not name:
                continue
            key = _normalize_name(name)
            old = ranking_rows.get(key)
            if old is None or points > old[1] or (points == old[1] and rank < old[0]):
                ranking_rows[key] = (rank, points)

    def ranking_for(player: str) -> tuple[int | None, int | None]:
        hits = [value for key, value in ranking_rows.items() if _name_matches(key, player)]
        if not hits:
            return None, None
        return sorted(hits, key=lambda x: (-x[1], x[0]))[0]

    def form_for(player: str, other_player: str) -> tuple[int, int]:
        # Locate the player's own Latest Games header. The section ends at the
        # opponent's Latest Games header or the ranking table.
        forms = [player, player.replace(",", " ")]
        starts = [text.lower().find(f"{form.lower()} latest games") for form in forms]
        starts = [x for x in starts if x >= 0]
        if not starts:
            return 0, 0
        start = min(starts)
        ends = []
        other_forms = [other_player, other_player.replace(",", " ")]
        for form in other_forms:
            pos = text.lower().find(f"{form.lower()} latest games", start + 10)
            if pos >= 0:
                ends.append(pos)
        if ranking_anchor and ranking_anchor.start() > start:
            ends.append(ranking_anchor.start())
        section = text[start:min(ends) if ends else min(len(text), start + 120000)]
        rows = re.findall(r'<tr\b[^>]*>.*?</tr>', section, flags=re.I | re.S)
        wins = losses = 0
        for row in rows:
            cleaned = _clean_html_text(row)
            # A prior match row must contain a calendar date and the target.
            if not re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", cleaned):
                continue
            if not any(_name_matches(piece, player) for piece in re.findall(r"[A-Za-z][A-Za-z ,.'-]{2,60}", cleaned)):
                continue
            outcomes = re.findall(r"\b([WL])\b", cleaned)
            if not outcomes:
                continue
            # Latest-game sections place the displayed player outcome first.
            if outcomes[0] == "W":
                wins += 1
            else:
                losses += 1
            if wins + losses >= 10:
                break
        return wins, losses

    a_rank, a_points = ranking_for(player_a)
    b_rank, b_points = ranking_for(player_b)
    a_wins, a_losses = form_for(player_a, player_b)
    b_wins, b_losses = form_for(player_b, player_a)
    return (
        Tennis365PlayerContext(player_a, a_rank, a_points, a_wins, a_losses),
        Tennis365PlayerContext(player_b, b_rank, b_points, b_wins, b_losses),
    )


def fetch_tennis365_player_context(
    source_url: str,
    *,
    player_a: str,
    player_b: str,
    timeout: float = 12.0,
) -> tuple[Tennis365PlayerContext, Tennis365PlayerContext] | None:
    if not source_url.startswith(TENNIS365_BASE_URL):
        return None
    try:
        response = requests.get(
            source_url,
            headers={"User-Agent": "SportsEdgeReadOnly/1.0", "Accept": "text/html"},
            timeout=timeout,
        )
        response.raise_for_status()
        return parse_tennis365_player_context(
            response.text, player_a=player_a, player_b=player_b
        )
    except requests.RequestException:
        return None
