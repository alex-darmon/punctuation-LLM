#!/usr/bin/env python3
"""Reproducibility and regression checks for the full 20-author v2 outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "campaigns" / "inference_v2.json"
OUTPUT = ROOT / "results" / "author_panel_20" / "inference_v2"
MANIFEST = OUTPUT / "inference_manifest.json"


def csv_rows(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@unittest.skipUnless(MANIFEST.is_file(), "run run_inference_v2.py first")
class InferenceOutputTests(unittest.TestCase):
    def test_primary_estimand_declaration_is_pinned(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            config["primary"],
            {
                "author_set": "all_authors",
                "feature": "f3",
                "chunk_size": 2000,
                "reference_policy": "leave_one_book_out",
                "detection_fpr": 0.05,
            },
        )
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["analysis_config_sha256"],
            hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        )
        self.assertEqual(manifest["cluster_units"]["primary"], "author")

    def test_manifest_hashes_every_declared_input_and_output(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for section in (
            "upstream_result_sha256",
            "source_code_sha256",
            "output_sha256",
        ):
            self.assertTrue(manifest[section], section)
        for relative, expected in manifest["upstream_result_sha256"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                expected,
                relative,
            )
        for relative, expected in manifest["source_code_sha256"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                expected,
                relative,
            )
        for filename, expected in manifest["output_sha256"].items():
            self.assertEqual(
                hashlib.sha256((OUTPUT / filename).read_bytes()).hexdigest(),
                expected,
                filename,
            )

    def test_manifest_records_complete_numerical_environment(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        packages = manifest["environment"]["packages"]
        self.assertEqual(
            set(packages),
            {
                "numpy",
                "scipy",
                "pandas",
                "matplotlib",
                "spacy",
                "en-core-web-sm",
            },
        )
        self.assertTrue(all(packages.values()))

    def test_condition_manifests_pin_assembled_texts(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for condition, expected in manifest["condition_manifest_sha256"].items():
            directory = ROOT / manifest["conditions"][condition]
            path = directory / "condition_manifest.json"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_split_manifest_has_no_calibration_test_overlap(self) -> None:
        rows = csv_rows("split_assignments.csv")
        for fold in {row["fold"] for row in rows}:
            fold_rows = [row for row in rows if row["fold"] == fold]
            calibration = {
                row["author"] for row in fold_rows if row["role"] == "calibration"
            }
            test = {row["author"] for row in fold_rows if row["role"] == "test"}
            self.assertTrue(calibration.isdisjoint(test))
            self.assertEqual(len(test), 4)
            self.assertEqual(
                Counter(
                    row["cohort"] for row in fold_rows if row["role"] == "test"
                ),
                {"existing": 2, "new": 2},
            )

    def test_every_audited_reference_excludes_its_scored_book(self) -> None:
        audit = csv_rows("reference_audit.csv")
        for row in audit:
            included = set(filter(None, row["included_books"].split(";")))
            excluded = set(filter(None, row["excluded_books"].split(";")))
            self.assertTrue(included.isdisjoint(excluded), row)
        audited_exclusions = {
            (row["author"], excluded)
            for row in audit
            for excluded in filter(None, row["excluded_books"].split(";"))
        }
        observations = csv_rows("attribution_observations.csv")
        for row in observations:
            scored = (
                row["book_id"]
                if row["source_type"] == "human"
                else row["prompt_source_book"]
            )
            self.assertIn((row["author"], scored), audited_exclusions)

    def test_all_primary_intervals_declare_their_cluster_unit(self) -> None:
        rows = [
            row
            for row in csv_rows("attribution_clustered.csv")
            if row["author_set"] == "all_authors"
            and row["feature"] == "f3"
            and row["chunk_size"] == "2000"
        ]
        for row in rows:
            expected = (
                "author"
                if row["row_type"] == "summary"
                else ("book_id" if row["source_type"] == "human" else "prompt_source_book")
            )
            self.assertEqual(row["cluster_unit"], expected)
            self.assertNotEqual(row["macro_ci_low"], "")
            self.assertNotEqual(row["macro_ci_high"], "")

    def test_primary_attribution_point_estimates_are_stable(self) -> None:
        rows = [
            row
            for row in csv_rows("attribution_clustered.csv")
            if row["row_type"] == "summary"
            and row["author_set"] == "all_authors"
            and row["feature"] == "f3"
            and row["chunk_size"] == "2000"
        ]
        by_condition = {
            row["condition"] or "human": (int(row["n_correct"]), int(row["n_observations"]))
            for row in rows
        }
        self.assertEqual(by_condition["human"], (372, 485))
        self.assertEqual(by_condition["flash"], (35, 400))
        self.assertEqual(by_condition["pro"], (16, 200))

    def test_primary_attribution_matches_the_full_grid(self) -> None:
        with (
            ROOT / "results" / "author_panel_20" / "full_grid" / "attribution.csv"
        ).open(newline="", encoding="utf-8") as handle:
            grid = list(csv.DictReader(handle))
        v2 = [
            row
            for row in csv_rows("attribution_clustered.csv")
            if row["row_type"] == "summary"
            and row["author_set"] == "all_authors"
            and row["feature"] == "f3"
            and row["chunk_size"] == "2000"
        ]
        for row in v2:
            condition = row["condition"]
            experiment = "human_attribution" if not condition else "llm_attribution"
            unit = "chunk" if not condition else "run"
            summary = next(
                candidate
                for candidate in grid
                if candidate["experiment"] == experiment
                and candidate["author_set"] == "all_authors"
                and candidate["condition"] == condition
                and candidate["policy"] == "leave_one_book_out"
                and candidate["unit"] == unit
                and candidate["chunk_size"] == "2000"
                and candidate["feature"] == "f3"
                and candidate["author"] == "ALL"
            )
            self.assertEqual(int(row["n_correct"]), int(summary["correct"]))
            self.assertEqual(int(row["n_observations"]), int(summary["n"]))
            author_rows = [
                candidate
                for candidate in grid
                if candidate["experiment"] == experiment
                and candidate["author_set"] == "all_authors"
                and candidate["condition"] == condition
                and candidate["policy"] == "leave_one_book_out"
                and candidate["unit"] == unit
                and candidate["chunk_size"] == "2000"
                and candidate["feature"] == "f3"
                and candidate["author"] != "ALL"
            ]
            macro = sum(float(item["accuracy_pct"]) for item in author_rows) / (
                100 * len(author_rows)
            )
            self.assertAlmostEqual(float(row["macro_accuracy"]), macro)

    def test_pooled_profiles_are_labelled_as_leakage_contrasts(self) -> None:
        rows = csv_rows("pooled_leakage_contrasts.csv")
        self.assertTrue(rows)
        self.assertEqual(
            {row["interpretation"] for row in rows},
            {"descriptive_leakage_contrast_only"},
        )
        self.assertEqual({row["policy"] for row in rows}, {"pooled_all_books"})

    def test_legacy_grid_has_no_duplicate_exact_attribution_rows(self) -> None:
        path = ROOT / "results" / "author_panel_20" / "full_grid" / "attribution.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        keys = [tuple(sorted(row.items())) for row in rows]
        self.assertEqual(len(keys), 1116)
        self.assertEqual(len(keys), len(set(keys)))

    @unittest.skipUnless(
        (ROOT / "paper" / "tables" / "inference_v2").is_dir(),
        "render inference tables first",
    )
    def test_rendered_primary_table_contains_pinned_counts(self) -> None:
        table = (
            ROOT / "paper" / "tables" / "inference_v2" / "primary_attribution.tex"
        ).read_text(encoding="utf-8")
        self.assertIn("Human & 20 & 485 & 73.1", table)
        self.assertIn("Flash & 20 & 400 & 8.8", table)
        self.assertIn("Pro & 20 & 200 & 8.0", table)


if __name__ == "__main__":
    unittest.main()
