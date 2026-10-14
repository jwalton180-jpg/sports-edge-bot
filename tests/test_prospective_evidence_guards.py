import math
import pytest
from sports_edge.models.prospective_evidence import record, settle, evaluate

def sample(**overrides):
    args=dict(sport="NHL",ticker="KXNHL-A",side="YES",outcome="2+ goals",
              model="poisson",probability=.3,ask=.2,source_id="official",
              source_as_of="2026-10-14T09:00:00Z",
              observed_at="2026-10-14T10:00:00Z")
    args.update(overrides)
    return record(**args)

@pytest.mark.parametrize("invalid", [float("nan"),float("inf"),float("-inf"),-0.1,1.1])
def test_invalid_probability_fails_closed(invalid):
    with pytest.raises(ValueError): sample(probability=invalid)

@pytest.mark.parametrize("invalid", [float("nan"),float("inf"),0,1])
def test_invalid_ask_fails_closed(invalid):
    with pytest.raises(ValueError): sample(ask=invalid)

def test_graded_requires_real_settlement_and_boolean():
    row=sample()
    assert evaluate([dict(row,settled_win=True)])["graded"]==0
    with pytest.raises(ValueError): settle(row,settled_at="2026-10-14T11:00:00Z",won=1)
    with pytest.raises(ValueError): settle(row,settled_at="2026-10-14T09:59:00Z",won=True)
    assert evaluate([settle(row,settled_at="2026-10-14T11:00:00Z",won=True)])["graded"]==1
