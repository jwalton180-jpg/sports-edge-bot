from __future__ import annotations

import unittest
from unittest.mock import Mock

from sports_edge.data.pick_store import SupabasePickStore, parse_timestamp, validate_prediction


BASE = {
    "sport": "NBA",
    "event": "Team A at Team B",
    "market": "Player Points",
    "selection": "Player A over 24.5",
    "observed_at": "2026-10-21T20:00:00Z",
    "event_start": "2026-10-21T23:00:00Z",
    "source": "SportEdge",
    "record_type": "prop",
    "probability": 0.57,
}


class PickStoreTests(unittest.TestCase):
    def test_prediction_requires_causal_timestamp_and_attribution(self):
        self.assertEqual(validate_prediction(BASE)["probability"], 0.57)
        bad = dict(BASE, observed_at="2026-10-22T00:00:00Z")
        with self.assertRaises(ValueError):
            validate_prediction(bad)
        with self.assertRaises(ValueError):
            validate_prediction(dict(BASE, probability=1.2))

    def test_timezone_is_required(self):
        with self.assertRaises(ValueError):
            parse_timestamp("2026-10-21T20:00:00")

    def test_supabase_append_is_server_side_postgrest_and_append_only(self):
        session = Mock()
        response = Mock()
        response.content = b'[{"id":"id-1","record_type":"prediction","payload":{}}]'
        response.json.return_value = [{"id": "id-1", "record_type": "prediction", "payload": {}}]
        response.raise_for_status.return_value = None
        session.request.return_value = response
        store = SupabasePickStore("https://example.supabase.co", "server-secret", session=session)
        out = store.append("prediction", BASE, record_id="id-1")
        self.assertEqual(out["id"], "id-1")
        args, kwargs = session.request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/rest/v1/sportedge_pick_records", args[1])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer server-secret")
        self.assertNotIn("PATCH", str(session.request.call_args_list))

    def test_missing_configuration_rejected(self):
        with self.assertRaises(ValueError):
            SupabasePickStore("", "")


if __name__ == "__main__":
    unittest.main()
