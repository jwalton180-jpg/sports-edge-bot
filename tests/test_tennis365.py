from datetime import datetime, timezone

from sports_edge.data.tennis365 import parse_tennis365_live_matches
from sports_edge.data.tennis_live import _tennis365_states_from_matches
from sports_edge.models.event_identity import canonical_event_id


NOW = datetime(2026, 10, 6, 21, 0, tzinfo=timezone.utc)


def _match_html(
    *,
    match_id: str,
    href: str,
    home: str,
    away: str,
    set1: tuple[int, int],
    set2: tuple[int, int],
    current: tuple[int, int],
    home_point: str = "0",
    away_point: str = "0",
    home_serving: bool = False,
) -> str:
    home_ball = '<img src="ball.svg" />' if home_serving else ''
    away_ball = '' if home_serving else '<img src="ball.svg" />'
    return f'''
    <div id="{match_id}" class="tennis_score_grid active">
      <strong>Set 3</strong>
      <a href="{href}" class="matches_grid_anchor">match</a>
      <span class="gmtdatetime">10/06/2026 11:00:00 AM</span>
      <span id="hn-{match_id}">{home}</span>
      <span id="an-{match_id}">{away}</span>
      <span id="hs-{match_id}">1</span>
      <span id="as-{match_id}">1</span>
      <span id="csh-{match_id}">{current[0]}</span>
      <span id="csa-{match_id}">{current[1]}</span>
      <span class="tennis_home_set_design">{set1[0]}</span>
      <span class="tennis_home_set_design">{set2[0]}</span>
      <span class="tennis_away_set_design">{set1[1]}</span>
      <span class="tennis_away_set_design">{set2[1]}</span>
      <span class="tennis_home_ball">{home_ball}</span>
      <span class="tennis_away_ball">{away_ball}</span>
      <span class="tennis_home_point_score">{home_point}</span>
      <span class="tennis_away_point_score">{away_point}</span>
      <span class="live_icon">LIVE</span>
    </div>
    '''


def test_macias_de_dios_itf_reversal_fixture_is_parsed_and_stateful():
    page = _match_html(
        match_id="17264658",
        href="/scores/m15-itf-quito/felipe-de-dios-vs-darwin-macias/",
        home="Felipe De Dios",
        away="Darwin Andres Macias Elizalde",
        set1=(7, 6),
        set2=(5, 7),
        current=(1, 3),
        home_serving=False,
    )
    matches = parse_tennis365_live_matches(page)
    assert len(matches) == 1
    match = matches[0]
    assert match.tour == "ITF"
    assert match.completed_sets == ((7, 6), (5, 7))
    assert match.home_games == 1 and match.away_games == 3

    states = _tennis365_states_from_matches(matches, fetched_at=NOW)
    macias = next(row for row in states if row.player == "Darwin Andres Macias Elizalde")
    assert macias.event_id == canonical_event_id(
        "Tennis", "Felipe De Dios", "Darwin Andres Macias Elizalde", "2026-10-06"
    )
    assert macias.player_sets == 1 and macias.opponent_sets == 1
    assert macias.lost_first_set and macias.won_latest_completed_set
    assert macias.turnaround and macias.deciding_set
    assert macias.player_games == 3 and macias.opponent_games == 1
    assert macias.current_set_lead == 2
    assert macias.serving is True
    assert macias.net_break_advantage == 1
    assert macias.score_sources == ("Tennis365",)
    assert not macias.score_conflict


def test_justo_comesana_challenger_reversal_fixture_is_parsed_and_stateful():
    page = _match_html(
        match_id="1902834",
        href="/scores/atp-challenger-antofagasta/guido-justo-vs-francisco-comesana/",
        home="Guido Ivan Justo",
        away="Francisco Comesana",
        set1=(6, 7),
        set2=(7, 6),
        current=(4, 1),
        home_point="0",
        away_point="30",
        home_serving=False,
    )
    matches = parse_tennis365_live_matches(page)
    assert len(matches) == 1
    match = matches[0]
    assert match.tour == "CHALLENGER"
    assert match.completed_sets == ((6, 7), (7, 6))
    assert match.home_games == 4 and match.away_games == 1

    states = _tennis365_states_from_matches(matches, fetched_at=NOW)
    justo = next(row for row in states if row.player == "Guido Ivan Justo")
    assert justo.player_sets == 1 and justo.opponent_sets == 1
    assert justo.lost_first_set and justo.won_latest_completed_set
    assert justo.turnaround and justo.deciding_set
    assert justo.current_set_lead == 3
    assert justo.serving is False
    assert justo.net_break_advantage == 1
    assert justo.point_score == "0-30"
    assert justo.score_sources == ("Tennis365",)
    assert not justo.score_conflict


def test_main_tour_live_rows_are_parsed_but_non_live_rows_are_ignored():
    live_atp = _match_html(
        match_id="99",
        href="/scores/atp-shanghai/a-vs-b/",
        home="Alpha One",
        away="Beta Two",
        set1=(6, 4),
        set2=(3, 6),
        current=(1, 1),
    )
    not_live = _match_html(
        match_id="100",
        href="/scores/m15-itf-quito/c-vs-d/",
        home="Gamma Three",
        away="Delta Four",
        set1=(6, 4),
        set2=(3, 6),
        current=(1, 1),
    ).replace('<span class="live_icon">LIVE</span>', '')
    rows = parse_tennis365_live_matches(live_atp + not_live)
    assert len(rows) == 1
    assert rows[0].tour == "ATP"
    assert rows[0].home == "Alpha One"
    assert rows[0].away == "Beta Two"
    assert rows[0].source_url.endswith("/scores/atp-shanghai/a-vs-b/")
