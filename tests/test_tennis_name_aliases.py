from collections import defaultdict

from sports_edge.models.tennis_research import PlayerState, TennisResearchModel


def _model_with_names(*names: str) -> TennisResearchModel:
    model = TennisResearchModel.__new__(TennisResearchModel)
    model.gender = "men"
    model.start_year = 2025
    model.end_year = 2026
    model.players = defaultdict(PlayerState)
    for name in names:
        state = model.players[model._name(name)]
        state.matches = 10
    model.alias_index = {}
    model._rebuild_alias_index()
    return model


def test_suffix_and_middle_name_aliases_resolve_when_unique():
    model = _model_with_names("Martin Damm", "Federico Gomez")

    assert model._resolve_name("Martin Damm Jr") == model._name("Martin Damm")
    assert model._resolve_name("Federico Agustin Gomez") == model._name("Federico Gomez")


def test_two_part_reversed_name_resolves_when_unique():
    model = _model_with_names("Bu Yunchaokete")

    assert model._resolve_name("Yunchaokete Bu") == model._name("Bu Yunchaokete")


def test_ambiguous_alias_fails_closed():
    model = _model_with_names("John Smith", "James Smith")

    # Exact names still resolve.
    assert model._resolve_name("John Smith") == model._name("John Smith")

    # A non-exact ambiguous alias must not guess.
    model.alias_index.setdefault("smith j", set()).update(
        {model._name("John Smith"), model._name("James Smith")}
    )
    original = model._alias_forms
    model._alias_forms = lambda value: ("smith j",)  # type: ignore[method-assign]
    assert model._resolve_name("J Smith") is None
    model._alias_forms = original  # type: ignore[method-assign]


def test_alias_forms_do_not_create_single_token_surname_matches():
    forms = TennisResearchModel._alias_forms("Anna Maria Rogers")
    assert "rogers" not in forms
    assert "anna rogers" in forms
