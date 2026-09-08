#!/usr/bin/env python3
"""Numerical checks for the distance and inference primitives."""

from __future__ import annotations

import unittest

import numpy as np

from punctlib.inference import benjamini_hochberg, percentile_interval
from punctlib.stats import auc, g_stat_f1


class StatsTests(unittest.TestCase):
    def test_g_stat_f1_matches_manual_likelihood_ratio(self) -> None:
        observed = np.array([30.0, 10.0])
        profile = np.array([0.5, 0.5])
        expected = 2 * (30 * np.log(30 / 20) + 10 * np.log(10 / 20))
        self.assertAlmostEqual(g_stat_f1(observed, profile), expected)

    def test_auc_counts_ties_at_half(self) -> None:
        self.assertAlmostEqual(auc([2.0, 3.0], [1.0, 2.0]), 0.875)

    def test_percentile_interval_uses_declared_confidence(self) -> None:
        low, high = percentile_interval(list(range(101)), confidence=0.80)
        self.assertAlmostEqual(low, 10.0)
        self.assertAlmostEqual(high, 90.0)

    def test_benjamini_hochberg_preserves_order_and_monotonicity(self) -> None:
        adjusted = benjamini_hochberg([0.04, 0.001, 0.03, 0.20])
        self.assertEqual(len(adjusted), 4)
        self.assertAlmostEqual(adjusted[1], 0.004)
        self.assertLessEqual(adjusted[2], adjusted[0])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))


if __name__ == "__main__":
    unittest.main()
