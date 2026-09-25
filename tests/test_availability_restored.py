from datetime import datetime, timezone, timedelta
from sports_edge.core.availability import AvailabilityLedger,AvailabilityUpdate,RecommendationDependency,PlayerStatus

def test_out_invalidates_dependent_pick():
    l=AvailabilityLedger(); l.register(RecommendationDependency("r1",players={"p1"}))
    now=datetime.now(timezone.utc)
    l.ingest(AvailabilityUpdate("NFL","p1","Player One",PlayerStatus.ACTIVE,"official",now-timedelta(minutes=2)))
    ids=l.ingest(AvailabilityUpdate("NFL","p1","Player One",PlayerStatus.OUT,"official",now))
    assert ids==["r1"] and not l.dependencies["r1"].valid

def test_older_status_cannot_overwrite_newer():
    l=AvailabilityLedger(); now=datetime.now(timezone.utc)
    l.ingest(AvailabilityUpdate("MLB","p","P",PlayerStatus.OUT,"official",now))
    l.ingest(AvailabilityUpdate("MLB","p","P",PlayerStatus.ACTIVE,"stale",now-timedelta(minutes=5)))
    assert l.current("p").status==PlayerStatus.OUT
