from sports_edge.models.nfl_prop_models import _role_shift


def test_role_shift_detects_increased_recent_usage_but_caps_it():
    mult, reason = _role_shift([4.0, 4.0, 4.0, 8.0, 8.0])
    assert 1.0 < mult <= 1.12
    assert reason is not None and "Role trend up" in reason


def test_role_shift_detects_decreased_recent_usage_but_caps_it():
    mult, reason = _role_shift([12.0, 12.0, 12.0, 5.0, 5.0])
    assert 0.88 <= mult < 1.0
    assert reason is not None and "Role trend down" in reason


def test_role_shift_refuses_two_game_sample():
    mult, reason = _role_shift([4.0, 9.0])
    assert mult == 1.0
    assert reason is None
