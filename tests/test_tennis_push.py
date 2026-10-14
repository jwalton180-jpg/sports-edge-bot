"""Push-provider tests run offline: no real phone/topic is contacted."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from sports_edge.data.tennis_push import NtfySettings, PushEvent, post_ntfy
from sports_edge.models.tennis_push_dispatch import send_tennis_pushes
from sports_edge.models.tennis_prospective_store import (
    ProspectiveResearchStore, first_signal_record,
)


NOW = datetime(2026, 10, 12, 23, 0, tzinfo=timezone.utc)
TOPIC = "sportsedge-test-very-long-random-topic-93858416"


class FakeResponse:
    status_code = 200

    def json(self):
        return {"id": "ntfy-ack-1"}


class FakeClient:
    def __init__(self, *, status=200):
        self.calls = []
        self.status = status

    def post(self, url, *, json, headers, timeout):
        self.calls.append((url, json, headers, timeout))
        return SimpleNamespace(
            status_code=self.status,
            json=lambda: {"id": "remote-accepted-1"},
        )


def _signal(*, lane="WATCH", price=.15, fair=.29, when=NOW, ticker="KXWTAMATCH-SHI"):
    return first_signal_record(
        event_id="TENNIS:2026-10-12:han shi|peyton stearns",
        ticker=ticker, selection="Han Shi", side="YES", lane=lane,
        price=price, fair=fair, score="Shi vs Stearns 2-6 · 3-5",
        observed_at=when, model_version="research-v1",
        candle_end_ts=int(when.timestamp()), spread_pp=1,
        evidence={"score_sources":["ESPN"], "research_only": True},
    )


def _extreme(*, price=.03):
    return SimpleNamespace(
        lane="EXTREME DIP — TRACKING ONLY",
        current_ask=price, live_fair=.05, ticker="KXWTAMATCH-SHI",
        selection="Han Shi", score="Shi trails 2-6, 3-5",
    )


def test_missing_topic_disables_delivery_and_does_not_invent_phone_connection(tmp_path):
    store=ProspectiveResearchStore(tmp_path/"research")
    store.add_signal(_signal())
    client=FakeClient()
    result=send_tennis_pushes(store, now=NOW, environ={}, session=client)
    assert result["configured"] is False
    assert result["accepted"] == 0
    assert client.calls == []
    assert store.notifications == []


def test_ntfy_enforces_long_url_safe_topic_and_https_server():
    assert NtfySettings.from_environ({"SPORTSEDGE_NTFY_TOPIC":TOPIC}).topic == TOPIC
    for topic in ("simple", "unsafe/with/slash", "a"*65):
        with pytest.raises(ValueError):
            NtfySettings.from_environ({"SPORTSEDGE_NTFY_TOPIC":topic})
    with pytest.raises(ValueError):
        NtfySettings.from_environ({
            "SPORTSEDGE_NTFY_TOPIC":TOPIC,
            "SPORTSEDGE_NTFY_SERVER":"http://ntfy.sh",
        })


def test_three_cent_push_is_tracking_only_never_a_suggested_entry(tmp_path):
    store=ProspectiveResearchStore(tmp_path/"research")
    client=FakeClient()
    env={"SPORTSEDGE_NTFY_TOPIC":TOPIC}
    result=send_tennis_pushes(
        store, now=NOW, extreme_observations=[_extreme()],
        environ=env,session=client,
    )
    assert result["configured"] and result["accepted"] == 1
    payload=client.calls[0][1]
    assert payload["priority"] == 2
    assert payload["topic"] == TOPIC
    assert "TRACK ONLY" in payload["title"]
    assert "3.0¢" in payload["message"]
    assert "not a bet" in payload["message"]
    assert not store.signals
    assert len(store.notifications) == 1
    assert store.notifications[0]["lane"] == "EXTREME DIP — TRACKING ONLY"
    assert store.notifications[0]["entry_ask"] == .03
    assert store.notifications[0]["model_fair"] == .05
    assert TOPIC not in str(store.notifications)


def test_recurring_poll_dedup_and_upgrade_from_extreme_to_watch_then_deep(tmp_path):
    root=tmp_path/"research"
    store=ProspectiveResearchStore(root)
    env={"SPORTSEDGE_NTFY_TOPIC":TOPIC}
    client=FakeClient()
    assert send_tennis_pushes(
        store, now=NOW, extreme_observations=[_extreme()],
        environ=env, session=client,
    )["accepted"] == 1
    assert send_tennis_pushes(
        store, now=NOW+timedelta(minutes=1),
        extreme_observations=[_extreme()],
        environ=env,session=client,
    )["accepted"] == 0
    store.add_signal(_signal(when=NOW+timedelta(minutes=1)))
    assert send_tennis_pushes(
        store, now=NOW+timedelta(minutes=1),
        environ=env, session=client,
    )["accepted"] == 1
    assert send_tennis_pushes(
        ProspectiveResearchStore(root), now=NOW+timedelta(minutes=2),
        environ=env,session=client,
    )["accepted"] == 0
    store.add_signal(_signal(
        lane="DEEP REVERSAL",price=.20,fair=.42,
        when=NOW+timedelta(minutes=2),
    ))
    result=send_tennis_pushes(
        store,now=NOW+timedelta(minutes=2),environ=env,session=client,
    )
    assert result["accepted"] == 1
    assert client.calls[-1][1]["priority"] == 4
    assert len(client.calls) == 3
    assert len(store.notifications) == 3
    assert store.report()["push_accepted_notifications"] == 3


def test_stale_and_negative_edge_watch_are_suppressed(tmp_path):
    store=ProspectiveResearchStore(tmp_path/"research")
    store.add_signal(_signal(when=NOW-timedelta(minutes=20)))
    store.add_signal(_signal(ticker="NO-EDGE",price=.18,fair=.20))
    client=FakeClient()
    result=send_tennis_pushes(store,now=NOW,
                              environ={"SPORTSEDGE_NTFY_TOPIC":TOPIC},session=client)
    assert result["accepted"] == 0
    assert not client.calls


def test_provider_error_is_reported_without_secret_leak_and_retriable(tmp_path):
    store=ProspectiveResearchStore(tmp_path/"research")
    store.add_signal(_signal())
    env={"SPORTSEDGE_NTFY_TOPIC":TOPIC,"SPORTSEDGE_NTFY_TOKEN":"do-not-leak-this-token"}
    reject=FakeClient(status=403)
    failed=send_tennis_pushes(store,now=NOW,environ=env,session=reject)
    assert failed["failed"]==1
    assert failed["accepted"]==0
    assert TOPIC not in str(failed)
    assert env["SPORTSEDGE_NTFY_TOKEN"] not in str(failed)
    assert store.notifications==[]
    accepted=send_tennis_pushes(store,now=NOW+timedelta(seconds=20),
                               environ=env,session=FakeClient())
    assert accepted["accepted"]==1


def test_priority_alert_payload_has_public_market_context_only():
    event=PushEvent(
        alert_id="test",selection="Han Shi",ticker="KXWTAMATCH-SHI",
        lane="REVERSAL SIGNAL",price=.12,fair=.30,
        score="set 2, 5-5",observed_at=NOW.isoformat(),
    )
    client=FakeClient()
    cfg=NtfySettings(topic=TOPIC,token="secret-token")
    assert post_ntfy(event,cfg,session=client)=="remote-accepted-1"
    url,payload,headers,_=client.calls[0]
    assert url == "https://ntfy.sh/"
    assert payload["priority"]==4
    assert payload["click"] == "https://kalshi.com/markets"
    assert "KXWTAMATCH-SHI" in payload["message"]
    assert headers["Authorization"]=="Bearer secret-token"
    assert "secret-token" not in str(payload)



def test_unverified_price_tracking_push_without_score_or_model(tmp_path):
    store=ProspectiveResearchStore(tmp_path/"research")
    client=FakeClient()
    notice=SimpleNamespace(
        lane="PRICE RECOVERY — TRACKING ONLY",
        current_ask=.12,live_fair=None,ticker="KX-W15-ZHANG",
        selection="Ziyan Zhang",
        score="Live score UNVERIFIED — market quote movement only",
    )
    env={"SPORTSEDGE_NTFY_TOPIC":TOPIC}
    result=send_tennis_pushes(
        store,now=NOW,price_observations=[notice],environ=env,session=client
    )
    assert result["configured"] and result["accepted"]==1
    payload=client.calls[0][1]
    assert "TRACK ONLY" in payload["title"]
    assert "not a bet" in payload["message"]
    assert "UNVERIFIED" in payload["message"]
    assert "not available" in payload["message"]
    assert store.signals==[]
    again=send_tennis_pushes(
        ProspectiveResearchStore(tmp_path/"research"),
        now=NOW+timedelta(minutes=1),price_observations=[notice],
        environ=env,session=client,
    )
    assert again["accepted"]==0


def test_price_only_recovery_does_not_block_later_model_confirmed_alert(tmp_path):
    store=ProspectiveResearchStore(tmp_path/"research")
    env={"SPORTSEDGE_NTFY_TOPIC":TOPIC}
    client=FakeClient()
    notice=SimpleNamespace(
        lane="PRICE RECOVERY — TRACKING ONLY",
        current_ask=.12,live_fair=None,ticker="KXWTAMATCH-SHI",
        selection="Han Shi",
        score="Live score UNVERIFIED",
    )
    first=send_tennis_pushes(store,now=NOW,price_observations=[notice],
                              environ=env,session=client)
    assert first["accepted"]==1
    store.add_signal(_signal(lane="REVERSAL SIGNAL",price=.15,fair=.39,
                             when=NOW+timedelta(seconds=30)))
    stronger=send_tennis_pushes(store,now=NOW+timedelta(seconds=30),
                                 environ=env,session=client)
    assert stronger["accepted"]==1
    assert len(store.notifications)==2



def test_tracking_alert_includes_quote_trough_rebound_and_spread():
    event=PushEvent(
        alert_id="market-tracking-proof",selection="Maja Chwalinska",
        ticker="KX-WTA-MATCH",lane="PRICE RECOVERY — TRACKING ONLY",
        price=.148,fair=None,score="Live score UNVERIFIED",
        observed_at=NOW.isoformat(),observed_trough=.03,
        rebound_pp=11.8,quote_spread_pp=1.5,
    )
    message=event.public_message()
    assert "Observed quote trough 3.0¢" in message
    assert "rebound +11.8 percentage points" in message
    assert "bid/ask spread 1.5pp" in message
    assert "TRACKING ONLY" in message
    assert "model live fair not available" in message



def test_push_final_scorer_veto_blocks_late_doomed_and_unknown_matches(tmp_path):
    store=ProspectiveResearchStore(tmp_path/"research")
    client=FakeClient()
    env={"SPORTSEDGE_NTFY_TOPIC":TOPIC}
    observation=SimpleNamespace(
        lane="PRICE RECOVERY — TRACKING ONLY",current_ask=.13,
        live_fair=None,ticker="KXITFW-LATE",selection="ITF underdog",
        score="score not available",
    )
    blocked=send_tennis_pushes(
        store,now=NOW,environ=env,session=client,
        price_observations=[observation],allowed_tickers=set(),
    )
    assert blocked["accepted"]==0
    assert client.calls==[]
    assert store.notifications==[]
    approved=send_tennis_pushes(
        store,now=NOW,environ=env,session=client,
        price_observations=[observation],allowed_tickers={"KXITFW-LATE"},
    )
    assert approved["accepted"]==1


def test_model_watch_also_requires_last_second_approved_score(tmp_path):
    store=ProspectiveResearchStore(tmp_path/"research")
    store.add_signal(_signal())
    env={"SPORTSEDGE_NTFY_TOPIC":TOPIC}
    client=FakeClient()
    blocked=send_tennis_pushes(store,now=NOW,environ=env,session=client,
                               allowed_tickers=set())
    assert blocked["accepted"]==0
    allowed=send_tennis_pushes(store,now=NOW,environ=env,session=client,
                               allowed_tickers={"KXWTAMATCH-SHI"})
    assert allowed["accepted"]==1
