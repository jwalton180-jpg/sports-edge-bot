from __future__ import annotations

from datetime import date, datetime, timezone
import re

from sports_edge.models.game_scope import GameEvent, normalize


_MLB_ALIASES: dict[str, tuple[str, ...]] = {
    "ARI": ("Arizona Diamondbacks", "Diamondbacks", "Arizona", "AZ"),
    "ATH": ("Athletics", "Oakland Athletics", "A's", "Oakland"),
    "ATL": ("Atlanta Braves", "Braves", "Atlanta"),
    "BAL": ("Baltimore Orioles", "Orioles", "Baltimore"),
    "BOS": ("Boston Red Sox", "Red Sox", "Boston"),
    "CHC": ("Chicago Cubs", "Cubs"),
    "CWS": ("Chicago White Sox", "White Sox", "CHW"),
    "CIN": ("Cincinnati Reds", "Reds", "Cincinnati"),
    "CLE": ("Cleveland Guardians", "Guardians", "Cleveland"),
    "COL": ("Colorado Rockies", "Rockies", "Colorado"),
    "DET": ("Detroit Tigers", "Tigers", "Detroit"),
    "HOU": ("Houston Astros", "Astros", "Houston"),
    "KC": ("Kansas City Royals", "Royals", "Kansas City", "KCR"),
    "LAA": ("Los Angeles Angels", "Angels", "LA Angels"),
    "LAD": ("Los Angeles Dodgers", "Dodgers", "LA Dodgers"),
    "MIA": ("Miami Marlins", "Marlins", "Miami"),
    "MIL": ("Milwaukee Brewers", "Brewers", "Milwaukee"),
    "MIN": ("Minnesota Twins", "Twins", "Minnesota"),
    "NYM": ("New York Mets", "Mets"),
    "NYY": ("New York Yankees", "Yankees"),
    "PHI": ("Philadelphia Phillies", "Phillies", "Philadelphia"),
    "PIT": ("Pittsburgh Pirates", "Pirates", "Pittsburgh"),
    "SD": ("San Diego Padres", "Padres", "San Diego", "SDP"),
    "SF": ("San Francisco Giants", "Giants", "San Francisco", "SFG"),
    "SEA": ("Seattle Mariners", "Mariners", "Seattle"),
    "STL": ("St. Louis Cardinals", "St Louis Cardinals", "Cardinals", "St. Louis", "St Louis"),
    "TB": ("Tampa Bay Rays", "Rays", "Tampa Bay", "TBR"),
    "TEX": ("Texas Rangers", "Rangers", "Texas"),
    "TOR": ("Toronto Blue Jays", "Blue Jays", "Toronto"),
    "WSH": ("Washington Nationals", "Nationals", "Washington", "WAS"),
}


_NFL_KALSHI_CODES = {
    "JAX": "JAC",
    "LAR": "LA",
    "WAS": "WSH",
}


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return datetime.now(timezone.utc).date()


def _alias_code(name: str, aliases: dict[str, tuple[str, ...]]) -> str | None:
    q = normalize(name)
    if not q:
        return None
    hits: list[str] = []
    for code, values in aliases.items():
        candidates = {normalize(code), *(normalize(v) for v in values)}
        if q in candidates:
            hits.append(code)
    return hits[0] if len(hits) == 1 else None


def canonical_participant(sport: str, name: str) -> str:
    q = normalize(name)
    sport_key = str(sport or "").upper()

    if sport_key == "NFL":
        # Imported lazily to keep the identity helper independent of the
        # public-team model import graph.
        from sports_edge.data.public_team_data import NFL_TEAM_ALIASES

        code = _alias_code(name, NFL_TEAM_ALIASES)
        if code:
            return _NFL_KALSHI_CODES.get(code, code).lower()

    if sport_key == "MLB":
        code = _alias_code(name, _MLB_ALIASES)
        if code:
            return code.lower()

    return q


def canonical_event_id(
    sport: str,
    participant_a: str,
    participant_b: str,
    event_date: date | datetime | str,
) -> str:
    a = canonical_participant(sport, participant_a)
    b = canonical_participant(sport, participant_b)
    participants = sorted(x for x in (a, b) if x)
    if len(participants) != 2 or participants[0] == participants[1]:
        raw = "|".join(sorted({normalize(participant_a), normalize(participant_b)}))
        participants = [raw or "unknown"]
    return f"{str(sport or '').upper()}:{_as_date(event_date).isoformat()}:{'|'.join(participants)}"


def canonical_event_id_from_title(
    sport: str,
    game_title: str,
    event_date: date | datetime | str,
) -> str:
    raw = str(game_title or "").strip()
    parts = [
        part.strip()
        for part in re.split(r"\s+(?:vs\.?|@|at)\s+", raw, maxsplit=1, flags=re.I)
        if part.strip()
    ]
    if len(parts) == 2:
        return canonical_event_id(sport, parts[0], parts[1], event_date)
    return f"{str(sport or '').upper()}:{_as_date(event_date).isoformat()}:{normalize(raw) or 'unknown'}"


def canonical_event_id_from_game(game: GameEvent) -> str:
    return canonical_event_id(
        game.sport,
        game.away_team,
        game.home_team,
        game.commence_time,
    )
