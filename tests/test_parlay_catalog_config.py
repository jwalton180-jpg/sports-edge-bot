import ast
from pathlib import Path


def test_parlay_catalog_uses_bounded_production_series_discovery():
    source = Path("sports_edge/app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_parlay_kalshi_markets"
    )
    calls = [
        node for node in ast.walk(target)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fetch_supported_sport_catalog"
    ]
    assert len(calls) == 1
    kwargs = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(kwargs["discover_all_series"], ast.Constant)
    assert kwargs["discover_all_series"].value is False
    assert isinstance(kwargs["overview_only"], ast.Constant)
    assert kwargs["overview_only"].value is False
