#!/usr/bin/env python3
"""Leakage and end-to-end checks for author-stratified detection folds."""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from punctlib.inference import crossfit_detection_estimate
from punctlib.splits import split_for_fold, stratified_group_folds


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_inference_grid.json"


class DetectionSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.folds = stratified_group_folds(
            cls.fixture["groups"],
            cls.fixture["strata"],
            n_folds=4,
            seed=42,
        )

    def test_each_author_occurs_in_exactly_one_balanced_test_fold(self) -> None:
        self.assertEqual(set(self.folds), set(self.fixture["groups"]))
        for fold in range(4):
            calibration, test = split_for_fold(self.folds, fold)
            self.assertTrue(set(calibration).isdisjoint(test))
            self.assertEqual(set(calibration) | set(test), set(self.folds))
            strata = Counter(self.fixture["strata"][author] for author in test)
            self.assertEqual(strata, {"existing": 1, "new": 1})

    def test_test_author_cannot_change_its_own_calibration_threshold(self) -> None:
        target = "old_a"
        target_fold = self.folds[target]
        _, original_folds = crossfit_detection_estimate(
            self.fixture["human_scores"],
            self.fixture["llm_scores"],
            self.folds,
            fpr_target=0.05,
            n_boot=20,
            seed=7,
        )
        changed_human = {
            author: list(values)
            for author, values in self.fixture["human_scores"].items()
        }
        changed_human[target] = [1000.0, 2000.0]
        _, changed_folds = crossfit_detection_estimate(
            changed_human,
            self.fixture["llm_scores"],
            self.folds,
            fpr_target=0.05,
            n_boot=20,
            seed=7,
        )
        original = {row["fold"]: row for row in original_folds}[target_fold]
        changed = {row["fold"]: row for row in changed_folds}[target_fold]
        self.assertEqual(original["threshold"], changed["threshold"])

    def test_tiny_grid_runs_end_to_end_and_is_seed_reproducible(self) -> None:
        first, first_folds = crossfit_detection_estimate(
            self.fixture["human_scores"],
            self.fixture["llm_scores"],
            self.folds,
            fpr_target=0.05,
            n_boot=50,
            seed=99,
        )
        second, second_folds = crossfit_detection_estimate(
            self.fixture["human_scores"],
            self.fixture["llm_scores"],
            self.folds,
            fpr_target=0.05,
            n_boot=50,
            seed=99,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_folds, second_folds)
        self.assertEqual(first["auc"], 1.0)
        self.assertEqual(first["tpr"], 1.0)


if __name__ == "__main__":
    unittest.main()
