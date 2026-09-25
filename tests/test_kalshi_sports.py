from sports_edge.models.kalshi_sports import (
    choose_kalshi_ticket,
    classify_kalshi_market,
    group_kalshi_sports,
    side_candidates,
)


def market(ticker, title, yes=0.55, no=0.45, event="EV1", volume=1000):
    return {
        "ticker": ticker,
        "series_ticker": ticker.split("-", 1)[0],
        "event_ticker": event,
        "event_title": title,
        "title": title,
        "yes_sub_title": "YES selection",
        "no_sub_title": "NO selection",
        "yes_ask_dollars": yes,
        "no_ask_dollars": no,
        "volume": volume,
    }


def test_kalshi_sport_classification_by_series():
    assert classify_kalshi_market(market("KXMLBGAME-26SEP24-STLPIT-STL", "St. Louis vs Pittsburgh")) == ("MLB", "Moneyline")
    assert classify_kalshi_market(market("KXNFLPASSYDS-26SEP24-QB-250", "QB passing yards")) == ("NFL", "Passing Yards")
    assert classify_kalshi_market(market("KXATPMATCH-26SEP24-AB", "Player A vs Player B")) == ("Tennis", "Match Winner")


def test_futures_are_excluded_even_with_sport_prefix():
    future = market("KXMLBGAME-CHAMP", "Will Los Angeles win the championship before 2030?")
    assert classify_kalshi_market(future) is None


def test_grouping_keeps_sports_separate_and_props_visible():
    rows = [
        market("KXMLBHIT-26SEP24-A", "Player A hits", event="MLB1"),
        market("KXNFLTD-26SEP24-B", "Player B touchdown", event="NFL1"),
        market("KXATPSETWINNER-26SEP24-C", "Player C vs Player D set 1", event="TEN1"),
    ]
    grouped = group_kalshi_sports(rows)
    assert len(grouped["MLB"]) == 1
    assert grouped["MLB"][0].family == "Hits"
    assert grouped["NFL"][0].family == "Player Touchdowns"
    assert grouped["Tennis"][0].family == "Set Winner"


def test_longshot_and_best_use_distinct_price_profiles():
    rows = []
    for i, price in enumerate((0.12, 0.18, 0.28, 0.52, 0.61, 0.72), 1):
        rows.append(
            market(
                f"KXMLBGAME-26SEP{i:02d}-A{i}",
                f"Game {i}",
                yes=price,
                no=1 - price,
                event=f"E{i}",
                volume=1000 + i,
            )
        )
    grouped = group_kalshi_sports(rows)
    candidates = side_candidates(grouped["MLB"])

    longshot = choose_kalshi_ticket(candidates, mode="longshot", target_legs=3)
    best = choose_kalshi_ticket(candidates, mode="best", target_legs=3)

    assert longshot
    assert best
    assert all(0.07 <= leg.price <= 0.35 for leg in longshot)
    assert all(0.40 <= leg.price <= 0.82 for leg in best)
    assert {leg.ticker for leg in longshot} != {leg.ticker for leg in best}
