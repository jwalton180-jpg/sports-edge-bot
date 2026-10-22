from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from sports_edge.data.source_registry import get_source_status, source_status_summary


class SourceRegistryTests(unittest.TestCase):
    def test_core_sources_are_explicitly_marked_integrated(self):
        sources = {item["name"]: item for item in get_source_status()}
        self.assertEqual(sources["Kalshi public API"]["status"], "integrated")
        self.assertEqual(sources["ESPN public feeds"]["status"], "integrated")
        self.assertEqual(sources["MLB Stats API"]["status"], "integrated")

    def test_odds_api_requires_explicit_key(self):
        with patch.dict(os.environ, {"THE_ODDS_API_KEY": ""}):
            sources = {item["name"]: item for item in get_source_status()}
            self.assertEqual(sources["The Odds API"]["status"], "needs_key")
        with patch.dict(os.environ, {"THE_ODDS_API_KEY": "test-key"}):
            sources = {item["name"]: item for item in get_source_status()}
            self.assertEqual(sources["The Odds API"]["status"], "configured")

    def test_unverified_vendor_websites_are_not_claimed_as_integrated(self):
        sources = {item["name"]: item for item in get_source_status()}
        for name in ("Betwatch", "BettingPros", "ScoresAndOdds", "StatMuse"):
            self.assertEqual(sources[name]["status"], "research_only")

    def test_external_provider_access_is_truthful_and_not_faked(self):
        sources = {item["name"]: item for item in get_source_status()}
        self.assertEqual(sources["PropsBot.AI"]["status"], "research_only")
        self.assertEqual(sources["Betstamp Pro API"]["status"], "needs_key")
        self.assertEqual(sources["Juice Reel API"]["status"], "research_only")
        self.assertEqual(sources["Outlier"]["status"], "research_only")
        self.assertIn("not connected", sources["Betstamp Pro API"]["detail"])
        self.assertIn("no user data is accessed", sources["Juice Reel API"]["detail"])

    def test_summary_counts_all_sources_without_network_calls(self):
        summary = source_status_summary()
        self.assertEqual(sum(summary.values()), len(get_source_status()))
        self.assertGreaterEqual(summary.get("integrated", 0), 3)


if __name__ == "__main__":
    unittest.main()
