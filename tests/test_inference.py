#!/usr/bin/env python3
"""Author-block inference tests."""

from __future__ import annotations

import unittest

from punctlib.inference import (
    author_accuracy_estimate,
    paired_author_difference,
)


class InferenceTests(unittest.TestCase):
    def test_macro_accuracy_is_author_equal_not_run_weighted(self) -> None:
        result = author_accuracy_estimate(
            {"many_runs": (100, 100), "few_runs": (0, 1)},
            n_boot=100,
            seed=17,
        )
        self.assertAlmostEqual(result["macro_accuracy"], 0.5)
        self.assertAlmostEqual(result["micro_accuracy"], 100 / 101)

    def test_author_bootstrap_is_seed_reproducible(self) -> None:
        tally = {"a": (8, 10), "b": (2, 10), "c": (5, 10)}
        first = author_accuracy_estimate(tally, n_boot=200, seed=2026)
        second = author_accuracy_estimate(tally, n_boot=200, seed=2026)
        self.assertEqual(first, second)

    def test_paired_difference_keeps_models_within_author(self) -> None:
        result = paired_author_difference(
            {"a": 0.8, "b": 0.2},
            {"a": 0.5, "b": 0.4},
            n_boot=100,
            seed=11,
        )
        self.assertAlmostEqual(result["mean_difference"], 0.05)
        self.assertEqual(result["n_authors"], 2)

    def test_paired_difference_uses_only_shared_authors(self) -> None:
        result = paired_author_difference(
            {"a": 0.8, "left_only": 1.0},
            {"a": 0.5, "right_only": 0.0},
            n_boot=20,
            seed=2,
        )
        self.assertEqual(result["n_authors"], 1)
        self.assertAlmostEqual(result["mean_difference"], 0.3)


if __name__ == "__main__":
    unittest.main()
