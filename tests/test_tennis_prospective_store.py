from datetime import datetime, timedelta, timezone
import json

import pytest

from sports_edge.models.tennis_prospective_store import (
    ProspectiveResearchStore, first_signal_record, verified_kalshi_settlement,
)


NOW=datetime(2026,10,9,10,0,tzinfo=timezone.utc)


def observation(*, lane="EARLY WATCH — RESEARCH", first_at=NOW, price=.12):
    return first_signal_record(
        event_id="Tennis:2026-10-09:justo|comesana",
        ticker="KXATPCHALLENGERMATCH-26OCT09-X", selection="Justo",
        side="YES", lane=lane, price=price, fair=.31,
        score="set3 2-2", observed_at=first_at,
        model_version="early-v1", candle_end_ts=int(first_at.timestamp())-30,
        spread_pp=2.0, evidence={"recovering":True},
    )


def market(*,ticker="KXATPCHALLENGERMATCH-26OCT09-X", status="settled", result="yes"):
    return {"market":{"ticker":ticker,"status":status,"result":result,
                      "settled_time":"2026-10-09T11:00:00Z"}}


def test_verification_uses_only_same_exact_terminal_kalshi_ticker_and_yes_no():
    row=observation()
    kwargs=dict(ticker=row["ticker"],side="YES",signal_id=row["signal_id"],checked_at=NOW)
    assert verified_kalshi_settlement(market(),**kwargs)["outcome"]=="WIN"
    assert verified_kalshi_settlement(market(result="no"),**kwargs)["outcome"]=="LOSS"
    assert verified_kalshi_settlement(market(),**{**kwargs,"side":"NO"})["outcome"]=="LOSS"
    assert verified_kalshi_settlement(market(status="closed"),**kwargs) is None
    assert verified_kalshi_settlement(market(status="open"),**kwargs) is None
    assert verified_kalshi_settlement(market(result=""),**kwargs) is None
    assert verified_kalshi_settlement(market(result="undetermined"),**kwargs) is None
    assert verified_kalshi_settlement(market(ticker="OTHER"),**kwargs) is None
    assert verified_kalshi_settlement({},**kwargs) is None


def test_git_branch_store_append_only_idempotent_across_process_restarts(tmp_path):
    root=tmp_path/"ledger"
    store=ProspectiveResearchStore(root)
    first=observation()
    assert store.add_signal(first)
    assert not store.add_signal(observation(price=.63))
    assert store.add_snapshots([{"snapshot_id":"s1","price":.10},{"snapshot_id":"s1","price":.75}])==1
    assert store.add_run({"observed_at":NOW.isoformat()},hourly_only=True)
    assert not store.add_run({"observed_at":(NOW+timedelta(minutes=5)).isoformat()},hourly_only=True)
    grade=verified_kalshi_settlement(
        market(),ticker=first["ticker"],side="YES",signal_id=first["signal_id"],
        checked_at=NOW+timedelta(hours=1),
    )
    assert store.add_settlement(grade)
    assert not store.add_settlement({**grade,"outcome":"LOSS"})
    restarted=ProspectiveResearchStore(root)
    assert restarted.signals[0]["entry_ask"]==.12
    assert len(restarted.snapshots)==1
    assert len(restarted.settlements)==1
    assert restarted.report()["by_lane"][first["lane"]]["wins"]==1
    assert restarted.report()["by_lane"][first["lane"]]["hypothetical_gross_dollars_per_contract"]==.88
    assert restarted.save_report()["total_settled"]==1
    assert json.loads((root/"report.json").read_text())["total_observations"]==1


def test_report_cannot_use_later_verified_results_as_prior_train_data(tmp_path):
    store=ProspectiveResearchStore(tmp_path/"ledger")
    for i in range(34):
        observed_at=NOW+timedelta(minutes=i*15)
        obs=observation(
            first_at=observed_at,
            lane="EARLY WATCH — RESEARCH",
        )
        # Distinct physical events; no artificial second signal per match.
        obs={**obs,
             "event_id":f"Tennis:2026-10-09:match{i}|other{i}",
             "ticker":f"KXTEST-{i}",
             "signal_id":f"unique-{i}"}
        store.add_signal(obs)
        grade=verified_kalshi_settlement(
            market(ticker=obs["ticker"]),ticker=obs["ticker"],
            side="YES",signal_id=obs["signal_id"],
            checked_at=NOW+timedelta(days=5),
        )
        store.add_settlement(grade)
    lane=store.report(min_train=3)["by_lane"]["EARLY WATCH — RESEARCH"]
    assert lane["settled"]==34
    assert lane["causal_forward"]["evaluated"]==0
    assert lane["causal_forward"]["model_brier"] is None
    assert not lane["causal_forward"]["trained_on_future_results"]


def test_corrupt_or_forged_orphan_grade_fails_closed(tmp_path):
    root=tmp_path/"ledger"
    root.mkdir()
    (root/"settlements.jsonl").write_text(json.dumps({
        "signal_id":"orphan","outcome":"WIN","source":"fake"})+"\n")
    with pytest.raises(ValueError,match="Orphan"):
        ProspectiveResearchStore(root)
