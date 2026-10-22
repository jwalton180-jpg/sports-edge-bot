from __future__ import annotations

import unittest

from sports_edge.models.prop_performance import (
    american_to_decimal,
    implied_probability,
    net_units,
    summarize_parlays,
    summarize_performance,
)


class PropPerformanceTests(unittest.TestCase):
    def test_american_odds_and_implied_probability(self):
        self.assertAlmostEqual(american_to_decimal(150), 2.5)
        self.assertAlmostEqual(american_to_decimal(-200), 1.5)
        self.assertAlmostEqual(implied_probability(+100), 0.5)
        with self.assertRaises(ValueError):
            american_to_decimal(0)

    def test_net_units_for_win_loss_push_void(self):
        self.assertAlmostEqual(net_units("win", 2.5, 2), 3.0)
        self.assertAlmostEqual(net_units("loss", 2.5, 2), -2.0)
        self.assertEqual(net_units("push", 2.5, 2), 0.0)
        self.assertEqual(net_units("void", 2.5, 2), 0.0)

    def test_performance_denominators_and_calibration_are_explicit(self):
        report = summarize_performance([
            {"result": "win", "odds_american": 100, "stake_units": 1, "probability": .6},
            {"result": "loss", "odds_american": -110, "stake_units": 1, "probability": .6},
            {"result": "push", "odds_american": -110},
            {"result": "void", "odds_american": -110},
            {"result": "pending"},
        ])
        self.assertEqual(report["wins"], 1)
        self.assertEqual(report["losses"], 1)
        self.assertEqual(report["pushes"], 1)
        self.assertEqual(report["voids"], 1)
        self.assertEqual(report["pending"], 1)
        self.assertEqual(report["hit_rate_denominator"], 2)
        self.assertAlmostEqual(report["hit_rate"], .5)
        self.assertEqual(report["risked_units"], 2)
        self.assertAlmostEqual(report["net_units"], 0)
        self.assertAlmostEqual(report["roi"], 0)
        self.assertAlmostEqual(report["brier_score"], ((.6-1)**2 + (.6-0)**2)/2)

    def test_parlays_are_grouped_by_whole_ticket_leg_count(self):
        result = summarize_parlays([
            {"record_type": "parlay", "leg_count": 2, "result": "win", "decimal_odds": 3.0},
            {"record_type": "parlay", "leg_count": 2, "result": "loss", "decimal_odds": 3.0},
            {"record_type": "prop", "leg_count": 1, "result": "win", "decimal_odds": 2.0},
            {"record_type": "parlay", "leg_count": 4, "result": "pending"},
        ])
        self.assertEqual(set(result), {2, 4})
        self.assertEqual(result[2]["total_records"], 2)
        self.assertEqual(result[2]["hit_rate_denominator"], 2)
        self.assertEqual(result[4]["pending"], 1)

    def test_roi_is_unavailable_when_any_settled_pick_lacks_original_odds(self):
        report = summarize_performance([
            {"result": "win", "decimal_odds": 2.0, "probability": .6},
            {"result": "loss", "probability": .6},
        ])
        self.assertIsNone(report["roi"])
        self.assertIsNone(report["net_units"])
        self.assertEqual(report["missing_odds_resolved"], 1)
        self.assertEqual(report["priced_net_units"], 1.0)

    def test_invalid_probabilities_fail_closed(self):
        with self.assertRaises(ValueError):
            summarize_performance([{"result": "win", "decimal_odds": 2, "probability": 1.1}])


if __name__ == "__main__":
    unittest.main()
