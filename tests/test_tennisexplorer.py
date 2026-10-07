from datetime import date

from sports_edge.data.tennisexplorer import (
    TENNIS_EXPLORER_BASE_URL,
    parse_tennisexplorer_directory,
    parse_tennisexplorer_player_context,
    resolve_tennisexplorer_profile,
)


DIRECTORY = """
<html><body>
<table>
<tr><td class="t-name"><a href="/player/justo/">Justo G.</a></td></tr>
<tr><td class="t-name"><a href="/player/comesana-c17e0/">Comesana F.</a></td></tr>
<tr><td class="t-name"><a href="/player/macias-elizalde/">Macias Elizalde D.</a></td></tr>
<tr><td class="t-name"><a href="/player/de-dios/">De Dios F.</a></td></tr>
</table>
</body></html>
"""


JUSTO = """
<html><body>
<h1>Justo Guido Ivan - profile</h1>
<p>Country: Argentina</p>
<p>Current/Highest rank - singles: 188. / 188.</p>
<h2>Player's record</h2>
<ul><li>W/L - singles</li></ul>
<table>
<thead><tr><th>Year</th><th>Summary</th><th>Clay</th><th>Hard</th><th>Indoors</th><th>Grass</th><th>Not set</th></tr></thead>
<tbody>
<tr><td>2026</td><td>48/25</td><td>48/24</td><td>0/1</td><td>-</td><td>-</td><td>-</td></tr>
</tbody>
</table>
<table>
<tr><td>06.10.</td><td></td><td><strong>Justo G.</strong> - Comesana F.</td><td>1R</td><td>6<sup>6</sup>-7, 7-6<sup>4</sup>, 6-1</td><td>2.70</td><td>1.42</td></tr>
<tr><td>04.10.</td><td></td><td><strong>Justo G.</strong> - Heide G.</td><td>F</td><td>6-1, 6-4</td><td>3.92</td><td>1.23</td></tr>
<tr><td>03.10.</td><td></td><td><strong>Justo G.</strong> - Barrios Vera M.</td><td>SF</td><td>4-6, 6-4, 6-3</td></tr>
<tr><td>23.09.</td><td></td><td>Mena F. - <strong>Justo G.</strong></td><td>R16</td><td>6-4, 7-6<sup>0</sup></td></tr>
<tr><td>22.09.</td><td></td><td><strong>Justo G.</strong> - Villanueva G.</td><td>1R</td><td>6-4, 3-6, 6-1</td></tr>
</table>
</body></html>
"""


COMESANA = """
<html><body>
<h1>Comesana Francisco - profile</h1>
<p>Current/Highest rank - singles: 54. / 54.</p>
<h2>Player's record</h2>
<ul><li>W/L - singles</li></ul>
<table>
<thead><tr><th>Year</th><th>Summary</th><th>Clay</th><th>Hard</th><th>Indoors</th><th>Grass</th><th>Not set</th></tr></thead>
<tbody><tr><td>2026</td><td>31/22</td><td>20/12</td><td>10/8</td><td>1/1</td><td>0/1</td><td>-</td></tr></tbody>
</table>
<table>
<tr><td>06.10.</td><td></td><td>Justo G. - <strong>Comesana F.</strong></td><td>1R</td><td>6<sup>6</sup>-7, 7-6<sup>4</sup>, 6-1</td></tr>
<tr><td>02.10.</td><td></td><td><strong>Comesana F.</strong> - Player X.</td><td>QF</td><td>6-3, 6-2</td></tr>
<tr><td>01.10.</td><td></td><td>Player Y. - <strong>Comesana F.</strong></td><td>R16</td><td>7-5, 6-4</td></tr>
</table>
</body></html>
"""


def test_directory_resolves_full_names_from_tennisexplorer_abbreviations():
    rows = parse_tennisexplorer_directory(DIRECTORY)
    assert len(rows) == 4
    assert resolve_tennisexplorer_profile(DIRECTORY, "Guido Ivan Justo") == (
        TENNIS_EXPLORER_BASE_URL + "/player/justo/"
    )
    assert resolve_tennisexplorer_profile(DIRECTORY, "Francisco Comesana") == (
        TENNIS_EXPLORER_BASE_URL + "/player/comesana-c17e0/"
    )
    assert resolve_tennisexplorer_profile(DIRECTORY, "Darwin Andres Macias Elizalde") == (
        TENNIS_EXPLORER_BASE_URL + "/player/macias-elizalde/"
    )
    assert resolve_tennisexplorer_profile(DIRECTORY, "Felipe de Dios") == (
        TENNIS_EXPLORER_BASE_URL + "/player/de-dios/"
    )


def test_profile_parser_extracts_rank_year_surface_and_recent_form():
    ctx = parse_tennisexplorer_player_context(
        JUSTO,
        player="Guido Ivan Justo",
        opponent="Francisco Comesana",
        profile_url=TENNIS_EXPLORER_BASE_URL + "/player/justo/",
        as_of=date(2026, 10, 7),
    )
    assert ctx is not None
    assert ctx.rank == 188
    assert ctx.highest_rank == 188
    assert (ctx.year_wins, ctx.year_losses) == (48, 25)
    assert (ctx.clay_wins, ctx.clay_losses) == (48, 24)
    assert (ctx.hard_wins, ctx.hard_losses) == (0, 1)
    assert (ctx.recent_wins, ctx.recent_losses) == (4, 1)
    assert (ctx.h2h_wins, ctx.h2h_losses) == (1, 0)


def test_profile_parser_excludes_same_day_current_pair_from_prior():
    ctx = parse_tennisexplorer_player_context(
        JUSTO,
        player="Guido Ivan Justo",
        opponent="Francisco Comesana",
        as_of=date(2026, 10, 6),
    )
    assert ctx is not None
    assert (ctx.recent_wins, ctx.recent_losses) == (3, 1)
    assert ctx.h2h_matches == 0


def test_profile_identity_fails_closed_on_wrong_player_page():
    ctx = parse_tennisexplorer_player_context(
        COMESANA,
        player="Guido Ivan Justo",
        opponent="Francisco Comesana",
        as_of=date(2026, 10, 7),
    )
    assert ctx is None
