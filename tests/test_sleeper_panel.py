from streamlit.testing.v1 import AppTest


def test_shared_panel_scan_and_exports_work_without_network(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = '''
from sports_edge.sleeper_panel import render_sleeper_panel
from tests.test_application_sleepers import market, example_report
render_sleeper_panel([market()],coverage_complete=True,scope="NHL",runner=lambda *a,**k:example_report())
'''
    app=AppTest.from_string(source,default_timeout=20).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.dataframe)==1
    assert len(app.get("download_button"))==3
    assert (tmp_path/"research/sleeper_ledger.sqlite").exists()


def test_main_ui_exposes_every_sport_and_nfl_tail_focus():
    from pathlib import Path
    source=Path("sports_edge/app.py").read_text()
    assert '"NHL", "NCAAF", "Soccer"' in source
    assert '"NFL 2+ Touchdowns"' in source
    assert 'render_sleeper_panel(markets' in source
    assert 'return audit_app_candidates(candidates' in source
    assert 'discover_all_series=False' in source
    assert 'overview_only=False' in source
    assert 'c4.metric("Illustrative leg-price payout"' in source
    assert 'Joint success probability", "Unavailable"' in source
    assert 'Actual combined price", "Unavailable"' in source
    assert 'expected value and a model/market value ratio are withheld' in source
