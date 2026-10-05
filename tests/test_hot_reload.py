from types import SimpleNamespace

from sports_edge.core import hot_reload


def test_import_module_fresh_returns_none_when_initial_import_fails(monkeypatch):
    def boom(name):
        raise ImportError(name)

    monkeypatch.setattr(hot_reload.importlib, "import_module", boom)
    assert hot_reload.import_module_fresh("sports_edge.fake") is None


def test_import_module_fresh_keeps_stale_module_when_reload_fails(monkeypatch):
    stale = SimpleNamespace(existing_export=object())
    monkeypatch.setattr(hot_reload.importlib, "import_module", lambda name: stale)

    def boom(module):
        raise ImportError("partial deploy")

    monkeypatch.setattr(hot_reload.importlib, "reload", boom)
    assert hot_reload.import_module_fresh("sports_edge.fake") is stale


def test_import_module_fresh_prefers_reloaded_module(monkeypatch):
    stale = SimpleNamespace(version="old")
    fresh = SimpleNamespace(version="new")
    monkeypatch.setattr(hot_reload.importlib, "import_module", lambda name: stale)
    monkeypatch.setattr(hot_reload.importlib, "reload", lambda module: fresh)
    assert hot_reload.import_module_fresh("sports_edge.fake") is fresh
