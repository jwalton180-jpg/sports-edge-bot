from __future__ import annotations

import math
import re

from sports_edge.core.math import clamp
from sports_edge.models.edge import grade_edge, should_surface
from sports_edge.models.game_scope import GameEvent, market_context, market_matches_game, normalize
from sports_edge.models.live_board import LiveSignal, market_yes_probability
from sports_edge.models.props import PropConsensus


_KEYWORDS: dict[str, tuple[str, ...]] = {
    "batter_hits": ("hit", "hits"),
    "batter_hits_alternate": ("hit", "hits"),
    "batter_home_runs": ("home run", "homer"),
    "batter_home_runs_alternate": ("home run", "homer"),
    "pitcher_strikeouts": ("strikeout", "strikeouts"),
    "pitcher_strikeouts_alternate": ("strikeout", "strikeouts"),
    "batter_total_bases": ("total base", "total bases"),
    "batter_total_bases_alternate": ("total base", "total bases"),
    "batter_rbis": ("rbi", "rbis"),
    "batter_rbis_alternate": ("rbi", "rbis"),
    "player_pass_yds": ("passing yard", "pass yards"),
    "player_pass_tds": ("passing touchdown", "pass touchdown"),
    "player_pass_attempts": ("pass attempt", "passing attempt"),
    "player_pass_completions": ("completion", "completions"),
    "player_rush_yds": ("rushing yard", "rush yards"),
    "player_rush_attempts": ("rush attempt", "rushing attempt", "carries"),
    "player_rush_tds": ("rushing touchdown", "rush touchdown"),
    "player_receptions": ("reception", "receptions"),
    "player_reception_yds": ("receiving yard", "reception yards"),
    "player_reception_tds": ("receiving touchdown", "reception touchdown"),
    "player_anytime_td": ("touchdown", "score a td"),
    "player_tds": ("touchdown", "touchdowns"),
    "player_1st_td": ("first touchdown", "1st touchdown"),
}


def _has_player(text: str, player: str) -> bool:
    player_n = normalize(player)
    text_n = normalize(text)
    return bool(player_n) and re.search(rf"\b{re.escape(player_n)}\b", text_n) is not None


def _has_family(text: str, market_key: str) -> bool:
    raw = text.lower()

    # Prevent semantically adjacent prop families from colliding.
    if market_key in {"batter_hits", "batter_hits_alternate"}:
        if any(x in raw for x in ("home run", "homer", "total base", "rbi", "strikeout")):
            return False
    if market_key in {"batter_home_runs", "batter_home_runs_alternate"}:
        return any(x in raw for x in ("home run", "homer"))
    if market_key in {"batter_total_bases", "batter_total_bases_alternate"}:
        return "total base" in raw
    if market_key in {"batter_rbis", "batter_rbis_alternate"}:
        return "rbi" in raw
    if market_key in {"pitcher_strikeouts", "pitcher_strikeouts_alternate"}:
        return "strikeout" in raw

    if market_key == "player_pass_yds":
        return any(x in raw for x in ("passing yard", "pass yards"))
    if market_key == "player_pass_tds":
        return any(x in raw for x in ("passing touchdown", "pass touchdown"))
    if market_key == "player_rush_yds":
        return any(x in raw for x in ("rushing yard", "rush yards"))
    if market_key == "player_reception_yds":
        return any(x in raw for x in ("receiving yard", "reception yards"))
    if market_key == "player_receptions":
        return "reception" in raw and "yard" not in raw and "touchdown" not in raw
    if market_key == "player_anytime_td":
        return any(x in raw for x in ("touchdown", "score a td")) and "first touchdown" not in raw and "1st touchdown" not in raw
    if market_key == "player_1st_td":
        return any(x in raw for x in ("first touchdown", "1st touchdown"))

    return any(k in raw for k in _KEYWORDS.get(market_key, ()))


def _over_threshold_matches(text: str, quote: PropConsensus) -> bool:
    if quote.line is None:
        return True
    if quote.side.lower() != "over":
        return False

    line = float(quote.line)
    # Strictly map standard x.5 sportsbook overs to Kalshi X+ milestones.
    frac = abs(line - math.floor(line))
    if abs(frac - 0.5) > 1e-6:
        return False

    target = math.floor(line) + 1
    raw = text.lower()
    explicit = (
        f"{target}+" in raw
        or f"{target} +" in raw
        or f"at least {target}" in raw
        or f"{target} or more" in raw
    )
    if explicit:
        return True

    # Natural one-or-more wording.
    if target == 1 and quote.market_key in {
        "batter_hits",
        "batter_hits_alternate",
        "batter_home_runs",
        "batter_home_runs_alternate",
        "player_anytime_td",
    }:
        return any(x in raw for x in ("record a hit", "hit a home run", "homer", "score a touchdown", "score a td"))

    return False


def prop_quote_matches_market(market: dict, game: GameEvent, quote: PropConsensus) -> bool:
    if not market_matches_game(market, game):
        return False
    text = market_context(market)
    if not _has_player(text, quote.player):
        return False
    if not _has_family(text, quote.market_key):
        return False

    side = quote.side.lower()
    if side == "over":
        return _over_threshold_matches(text, quote)
    if side == "yes":
        return quote.line is None
    return False


def build_prop_signals(
    markets: list[dict],
    game: GameEvent,
    quotes: list[PropConsensus],
    *,
    min_edge_points: float = 4.0,
    min_confidence: float = 0.60,
    min_data_quality: float = 0.72,
) -> list[LiveSignal]:
    signals: list[LiveSignal] = []
    for quote in quotes:
        if quote.warnings:
            continue

        for market in markets:
            if not prop_quote_matches_market(market, game, quote):
                continue
            market_p = market_yes_probability(market)
            if market_p is None:
                continue

            freshness = clamp(1.0 - quote.median_age_s / 180.0)
            book_factor = min(1.0, quote.book_count / 4.0)
            data_quality = clamp(0.65 * book_factor + 0.35 * freshness)
            confidence = clamp(0.70 * book_factor + 0.30 * freshness)

            card = grade_edge(
                key=str(market.get("ticker") or ""),
                sport=game.sport,
                market=str(market.get("title") or market.get("subtitle") or market.get("ticker") or ""),
                selection=f"{quote.player} {quote.side} {quote.line if quote.line is not None else ''} {quote.market_label}".strip(),
                fair_p=quote.fair_probability,
                market_p=market_p,
                model_confidence=confidence,
                data_quality=data_quality,
                reasons=[
                    f"Exact game identity: {game.away_team} @ {game.home_team}",
                    f"Exact player/prop family match: {quote.player} · {quote.market_label}",
                    f"No-vig/consensus from {quote.book_count} fresh sportsbook source(s)",
                ],
                warnings=[],
                min_edge_points=min_edge_points,
            )
            surfaced = should_surface(
                card,
                min_edge_points=min_edge_points,
                min_confidence=min_confidence,
                min_data_quality=min_data_quality,
            )
            status = "QUALIFIED" if surfaced else ("WATCH" if card.edge_points > 0 else "PASS")
            signals.append(
                LiveSignal(
                    ticker=card.key,
                    sport=game.sport,
                    event_id=game.event_id,
                    event_title=f"{game.away_team} @ {game.home_team}",
                    market=card.market,
                    side="YES",
                    selection=card.selection,
                    market_probability=card.market_probability,
                    fair_probability=card.fair_probability,
                    edge_points=card.edge_points,
                    ev_per_contract=card.ev_per_dollar,
                    confidence=card.confidence,
                    data_quality=card.data_quality,
                    book_count=quote.book_count,
                    source_age_s=quote.median_age_s,
                    status=status,
                    tier="PROP",
                    reasons=tuple(card.reasons),
                    warnings=tuple(card.warnings),
                )
            )

    dedup: dict[tuple[str, str], LiveSignal] = {}
    for s in signals:
        key = (s.ticker, s.selection)
        if key not in dedup or s.confidence > dedup[key].confidence:
            dedup[key] = s
    return sorted(dedup.values(), key=lambda s: (s.status == "QUALIFIED", s.edge_points, s.confidence), reverse=True)
