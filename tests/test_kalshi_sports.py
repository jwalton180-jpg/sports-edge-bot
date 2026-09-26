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


def test_mlb_total_bases_long_series_is_not_misclassified_as_game_total():
    row = market(
        "KXMLBTOTALBASE-26SEP25-PLAYER-2",
        "Player A: 2+ total bases?",
        event="MLBTB1",
    )
    row["series_ticker"] = "KXMLBTOTALBASE"
    assert classify_kalshi_market(row) == ("MLB", "Total Bases")


def test_nfl_live_rushing_yards_series_is_classified():
    row = market(
        "KXNFLRSHYDS-26SEP27LACBUF-PLAYER-50",
        "Player A: 50+ rushing yards",
        event="NFLRUSH1",
    )
    row["series_ticker"] = "KXNFLRSHYDS"
    assert classify_kalshi_market(row) == ("NFL", "Rushing Yards")


def test_nfl_live_receptions_series_is_classified():
    row = market(
        "KXNFLREC-26SEP27LACBUF-PLAYER-5",
        "Player A: 5+ receptions",
        event="NFLREC1",
    )
    row["series_ticker"] = "KXNFLREC"
    assert classify_kalshi_market(row) == ("NFL", "Receptions")


def test_nfl_volume_series_are_classified():
    cases = [
        ("KXNFLPASSATT", "QB: 33+ passing attempts", "Pass Attempts"),
        ("KXNFLPASSCOMP", "QB: 22+ passing completions", "Pass Completions"),
        ("KXNFLPASSINT", "QB: 1+ passing interceptions", "Pass Interceptions"),
        ("KXNFLRSHATT", "RB: 18+ rushing attempts", "Rush Attempts"),
        ("KXNFLRRYDS", "RB: 100+ rushing and receiving yards combined", "Rushing + Receiving Yards"),
    ]
    for idx, (series, title, family) in enumerate(cases):
        row = market(f"{series}-26SEP27LACBUF-PLAYER-{idx}", title, event=f"NFV{idx}")
        row["series_ticker"] = series
        assert classify_kalshi_market(row) == ("NFL", family)


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
    assert all(0.05 <= leg.price <= 0.35 for leg in longshot)
    assert all(0.40 <= leg.price <= 0.82 for leg in best)
    assert {leg.ticker for leg in longshot} != {leg.ticker for leg in best}



def test_all_current_direct_tennis_series_classify_as_tennis():
    current = {
        "KXATPCHALLENGERMATCH": "Match Winner",
        "KXATPDOUBLES": "Doubles Match Winner",
        "KXATPGTOTAL": "Games Total",
        "KXATPMATCH": "Match Winner",
        "KXATPSETWINNER": "Set Winner",
        "KXITFDOUBLES": "Doubles Match Winner",
        "KXITFMATCH": "Match Winner",
        "KXITFWDOUBLES": "Doubles Match Winner",
        "KXITFWMATCH": "Match Winner",
        "KXWTACHALLENGERMATCH": "Match Winner",
        "KXWTADOUBLES": "Doubles Match Winner",
        "KXWTAMATCH": "Match Winner",
        "KXWTASETWINNER": "Set Winner",
    }
    for series, family in current.items():
        row = market(
            f"{series}-26SEP25-TEST",
            "Player A vs Player B",
            event=f"{series}-26SEP25AB",
        )
        row["series_ticker"] = series
        assert classify_kalshi_market(row) == ("Tennis", family)


def test_new_direct_tennis_series_fails_open_to_other_tennis_not_out_of_app():
    row = market(
        "KXATPBREAKPOINTS-26SEP25-TEST",
        "Player A break points",
        event="TEN-NEW",
    )
    row["series_ticker"] = "KXATPBREAKPOINTS"
    assert classify_kalshi_market(row) == ("Tennis", "Other Tennis")



def test_nba_and_wnba_are_first_class_and_unknown_families_stay_visible():
    nba = market("KXNBAPTS-26SEP25-PLAYER", "Player points", event="NBA1")
    wnba = market("KXWNBAREB-26SEP25-PLAYER", "Player rebounds", event="WNBA1")
    new_nba = market("KXNBANEWSTAT-26SEP25-PLAYER", "New NBA stat", event="NBA2")

    assert classify_kalshi_market(nba) == ("NBA", "Points")
    assert classify_kalshi_market(wnba) == ("WNBA", "Rebounds")
    assert classify_kalshi_market(new_nba) == ("NBA", "Other NBA")

    grouped = group_kalshi_sports([nba, wnba, new_nba])
    assert len(grouped["NBA"]) == 2
    assert len(grouped["WNBA"]) == 1


def test_challenger_itf_and_new_tennis_families_never_drop_out():
    rows = [
        market("KXATPCHALLENGERMATCH-26SEP25-A", "A vs B", event="T1"),
        market("KXITFWSETWINNER-26SEP25-B", "C vs D set", event="T2"),
        market("KXWTABREAKPOINTS-26SEP25-C", "Break points", event="T3"),
    ]
    classified = [classify_kalshi_market(row) for row in rows]
    assert classified[0] == ("Tennis", "Match Winner")
    assert classified[1] == ("Tennis", "Set Winner")
    assert classified[2] == ("Tennis", "Other Tennis")



def test_short_horizon_tennis_tournament_future_is_excluded():
    row = market(
        "KXWTATOURNWIN-26SEP25-PLAYER",
        "Will Player A win tournament?",
        event="TFUT",
    )
    assert classify_kalshi_market(row) is None


def test_table_tennis_never_enters_wta_tennis_catalog():
    row = market(
        "KXWTABLETENNISMATCH-26SEP25-AB",
        "Women's Table Tennis Match",
        event="TT1",
    )
    assert classify_kalshi_market(row) is None
