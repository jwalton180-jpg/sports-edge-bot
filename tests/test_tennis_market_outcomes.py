"""Actual quoted post-alert opportunities, not fictional trading returns."""
from datetime import datetime, timedelta, timezone
import json

from sports_edge.models.tennis_market_outcomes import (
    market_opportunity_report, sample_followthrough, save_opportunity_report,
)
from sports_edge.models.tennis_prospective_store import ProspectiveResearchStore


T=datetime(2026,10,14,1,0,tzinfo=timezone.utc)


def _store(tmp_path, *, price=.148, at=T, lane="PRICE RECOVERY — TRACKING ONLY"):
    store=ProspectiveResearchStore(tmp_path/"data")
    store.record_push_acceptance(
        alert_id="push-1",ticker="KX-WTA-TEST",lane=lane,
        accepted_at=at,provider_id="ntfy-accepted",
        entry_ask=price
    )
    return store


def _market(*, bid=.80, ask=.82, status="active", ticker="KX-WTA-TEST"):
    return {
        "ticker":ticker,
        "status":status,
        "yes_bid_dollars":str(bid),
        "yes_ask_dollars":str(ask),
    }


def test_post_alert_quote_proves_only_bid_opportunity_not_realized_return(tmp_path):
    store=_store(tmp_path)
    assert sample_followthrough(store,[_market()],now=T+timedelta(seconds=30))["new_quote_samples"]==0
    assert sample_followthrough(store,[_market()],now=T+timedelta(minutes=5))["new_quote_samples"]==1
    report=save_opportunity_report(store)
    assert report["notification_count"]==1
    assert report["with_post_alert_bid_samples"]==1
    assert report["alerts"][0]["alert_ask"]==.148
    assert report["alerts"][0]["best_observed_bid"]==.80
    assert round(report["alerts"][0]["max_observed_gross_return_pct"])==441
    assert report["sampled_opportunities"]==1
    assert "not fills" in report["warning"]
    assert json.loads((store.root/"market_opportunities.json").read_text())==report


def test_missing_or_wide_bid_never_counts_as_observed_exit(tmp_path):
    store=_store(tmp_path)
    for i, market in enumerate((
        {"ticker":"KX-WTA-TEST","status":"active","yes_ask_dollars":".84"},
        _market(bid=.13,ask=.40),
        _market(bid=.90,ask=.88),
        _market(status="closed"),
    )):
        sample_followthrough(store,[market],now=T+timedelta(minutes=i+2))
    report=market_opportunity_report(store)
    assert report["with_post_alert_bid_samples"]==0
    assert report["without_post_alert_samples"]==1


def test_same_minute_dedup_and_restart_are_idempotent(tmp_path):
    store=_store(tmp_path)
    m=[_market()]
    assert sample_followthrough(store,m,now=T+timedelta(minutes=5))["new_quote_samples"]==1
    restarted=ProspectiveResearchStore(store.root)
    assert sample_followthrough(restarted,m,now=T+timedelta(minutes=5,seconds=12))["new_quote_samples"]==0
    assert sample_followthrough(restarted,m,now=T+timedelta(minutes=6))["new_quote_samples"]==1
    assert market_opportunity_report(restarted)["alerts"][0]["post_alert_sample_count"]==2


def test_old_notification_receipt_never_invents_price(tmp_path):
    store=ProspectiveResearchStore(tmp_path/"research")
    store.record_push_acceptance(
        alert_id="old",ticker="KX-WTA-TEST",
        lane="EXTREME DIP — TRACKING ONLY",
        accepted_at=T,provider_id="ack",
    )
    store=ProspectiveResearchStore(store.root)
    assert sample_followthrough(store,[_market()],now=T+timedelta(minutes=3))["new_quote_samples"]==0
    r=market_opportunity_report(store)
    assert r["older_unpriced_receipts"]==1
    assert r["alerts"][0]["status"]=="entry_not_recorded"


def test_only_six_hour_window_and_status_active_qualify(tmp_path):
    store=_store(tmp_path)
    assert sample_followthrough(store,[_market(status="settled")],now=T+timedelta(minutes=3))["new_quote_samples"]==0
    assert sample_followthrough(store,[_market()],now=T+timedelta(hours=6,minutes=1))["new_quote_samples"]==0


def test_legacy_integer_cent_market_bid_is_read_as_dollars(tmp_path):
    store=_store(tmp_path)
    m={"ticker":"KX-WTA-TEST","status":"active","yes_bid":80,"yes_ask":82}
    assert sample_followthrough(store,[m],now=T+timedelta(minutes=3))["new_quote_samples"]==1
    assert market_opportunity_report(store)["alerts"][0]["best_observed_bid"]==.8
