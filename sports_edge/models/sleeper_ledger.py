"""Append-only on-demand evidence ledger; settlements are separate records.

Local SQLite is not a durable hosted database. UI exposes a full export after
each scan and explicitly reports durability; no autonomous service is started.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


class SleeperLedger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=10)
        self.db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS runtime_state (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
        self.db.execute("CREATE TRIGGER IF NOT EXISTS immutable_update BEFORE UPDATE ON records BEGIN SELECT RAISE(ABORT, 'immutable'); END")
        self.db.execute("CREATE TRIGGER IF NOT EXISTS immutable_delete BEFORE DELETE ON records BEGIN SELECT RAISE(ABORT, 'immutable'); END")
        self.db.commit()

    def append_scan(self, report):
        from sports_edge.models.sleeper_pipeline import export_evidence
        payload = export_evidence(report)
        key = hashlib.sha256(payload.encode()).hexdigest()
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO records VALUES (?,?,?)", (key,"scan",payload))
        return key

    def settle(self, *, scan_id, ticker, result, source_url, settled_at, _verified=False):
        from sports_edge.models.sleeper_counts import utc
        scan = self.db.execute("SELECT payload FROM records WHERE id=? AND kind='scan'",(scan_id,)).fetchone()
        if not scan or result not in {"yes","no","void","fair_price"} or not source_url.startswith("https://"):
            raise ValueError("invalid official settlement")
        report = json.loads(scan[0])
        matching = [r for r in report["records"] if r["ticker"] == ticker]
        if not matching or utc(settled_at) > datetime.now(timezone.utc) or any(utc(settled_at) <= utc(r["observed_at"]) or utc(settled_at) < utc(r["event_time"]) for r in matching):
            raise ValueError("noncausal or unknown settlement")
        key = hashlib.sha256((scan_id+":"+ticker).encode()).hexdigest()
        payload = json.dumps({"scan_id":scan_id,"ticker":ticker,"result":result,
                              "settled_at":settled_at,"source_url":source_url,"settlement_verified":_verified},sort_keys=True)
        with self.db:
            self.db.execute("INSERT INTO records VALUES (?,?,?)",(key,"settlement",payload))
        return key

    def grade_official(self, client, *, limit=40):
        """On-demand read-only final-market grading. Open/ambiguous results wait."""
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("bounded positive grading limit required")
        pending=[]
        for scan_id,payload in self.db.execute("SELECT id,payload FROM records WHERE kind='scan'"):
            report=json.loads(payload)
            for ticker in dict.fromkeys(r["ticker"] for r in report["records"]):
                key=hashlib.sha256((scan_id+":"+ticker).encode()).hexdigest()
                if not self.db.execute("SELECT 1 FROM records WHERE id=?",(key,)).fetchone():
                    pending.append((scan_id,ticker))
        cursor=self.db.execute("SELECT value FROM runtime_state WHERE key='grade_cursor'").fetchone()
        offset=(cursor[0] if cursor else 0)%len(pending) if pending else 0
        rotated=pending[offset:]+pending[:offset]
        cache={}; graded=0; errors=[]
        for scan_id,ticker in rotated[:limit]:
            try:
                if ticker not in cache:
                    cache[ticker]=client.market(ticker).data["market"]
                m=cache[ticker]
                if m.get("ticker")!=ticker or m.get("status") not in {"settled","finalized"} or m.get("result") not in {"yes","no"}:
                    continue
                if float(m.get("settlement_value_dollars", "nan")) != int(m["result"] == "yes"):
                    # Fair-price resolutions and voids are not binary wins.
                    continue
                settled=m.get("settlement_ts") or m.get("settlement_time")
                if not settled:
                    continue
                self.settle(scan_id=scan_id,ticker=ticker,result=m["result"],source_url=client.base+"/markets/"+ticker,
                            settled_at=settled,_verified=True)
                graded+=1
            except (ValueError, KeyError, sqlite3.Error, __import__("requests").RequestException) as exc:
                errors.append(type(exc).__name__)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO runtime_state VALUES ('grade_cursor',?)",(offset+min(limit,len(pending)),))
        return {"graded":graded,"pending_records":len(pending)-graded,"capacity_complete":len(pending)<=limit,"errors":errors}

    def metrics(self):
        scans={}; settlements=[]
        for key,kind,payload in self.db.execute("SELECT id,kind,payload FROM records"):
            if kind=="scan":scans[key]=json.loads(payload)
            else:settlements.append(json.loads(payload))
        groups={};seen=set();duplicates=0
        for final in settlements:
            if final.get("settlement_verified") is not True or final["result"] not in {"yes","no"}:
                continue
            for r in scans.get(final["scan_id"],{}).get("records",[]):
                if r["ticker"]!=final["ticker"] or r["probability"] is None:
                    continue
                identity=(r["ticker"],r["model"],r["side"])
                if identity in seen:
                    duplicates+=1
                    continue
                seen.add(identity)
                band=(int(r["quote"]["ask"]*10) if r["quote"]["ask"] is not None else "unpriced")
                key=(r["sport"],r["family"],r["threshold"],r["model"],r["side"],band)
                groups.setdefault(key,[]).append((r["probability"],int(r["side"].lower()==final["result"])))
        return {"groups":[{"sport":k[0],"family":k[1],"threshold":k[2],"model":k[3],"side":k[4],"price_decile":k[5],
                           "graded":len(v),"brier":sum((p-y)**2 for p,y in v)/len(v),
                           "calibration_gap":sum(p-y for p,y in v)/len(v)} for k,v in groups.items()],
                "duplicate_recommendations_excluded":duplicates,"realized_returns":None,"execution_verified":False,"qualified_for_live":False}

    def export_jsonl(self):
        return "".join(json.dumps({"id":i,"kind":k,"payload":json.loads(p)}, sort_keys=True)+"\n"
                       for i,k,p in self.db.execute("SELECT id,kind,payload FROM records ORDER BY rowid"))

    def prospective_review(self):
        """Prepare a read-only, explicitly unattested study dataset."""
        from sports_edge.models.prospective_review import prospective_review_export
        return prospective_review_export(self.export_jsonl())

    def close(self):
        self.db.close()
