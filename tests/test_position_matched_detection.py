#!/usr/bin/env python3
"""Chunk-construction and equal-length window checks."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

from punctlib.features import chunks
from run_position_matched_detection import mark_window


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DETECTION = (
    ROOT / "results" / "author_panel_20" / "inference_v2" / "detection_crossfit.csv"
)
POSITION_RESULTS = (
    ROOT / "results" / "author_panel_20" / "position_matched_detection"
)


class MarkWindowTests(unittest.TestCase):
    def test_late_window_is_not_a_prefix(self) -> None:
        seq = [str(i) for i in range(5000)]
        prefix = mark_window(seq, 2000, 0)
        late = mark_window(seq, 2000, 2000)
        self.assertEqual(prefix, seq[:2000])
        self.assertEqual(late, seq[2000:4000])
        self.assertNotEqual(prefix, late)
        self.assertEqual(len(prefix), len(late))

    def test_llm_length_grid_is_nested_prefixes(self) -> None:
        seq = [str(i) for i in range(5000)]
        one = mark_window(seq, 1000, 0)
        two = mark_window(seq, 2000, 0)
        four = mark_window(seq, 4000, 0)
        self.assertEqual(one, two[:1000])
        self.assertEqual(two, four[:2000])

    def test_human_tiles_are_nonoverlapping_and_counts_fall_with_length(self) -> None:
        seq = [str(i) for i in range(5000)]
        tiled = {size: list(chunks(seq, size)) for size in (1000, 2000, 4000)}
        self.assertEqual([len(tiled[size]) for size in (1000, 2000, 4000)], [5, 2, 1])
        self.assertEqual(tiled[2000][0], tiled[1000][0] + tiled[1000][1])
        self.assertEqual(tiled[1000][-1], seq[4000:5000])
        self.assertEqual(tiled[2000][1], seq[2000:4000])

    def test_short_sequences_are_dropped(self) -> None:
        self.assertIsNone(mark_window(["a"] * 1999, 2000, 0))
        self.assertIsNone(mark_window(["a"] * 3999, 2000, 2000))


@unittest.skipUnless(
    (POSITION_RESULTS / "detection_position.csv").is_file()
    and CANONICAL_DETECTION.is_file(),
    "run run_position_matched_detection.py first",
)
class PositionMatchedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (POSITION_RESULTS / "detection_position.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            cls.rows = list(csv.DictReader(handle))
        with CANONICAL_DETECTION.open(newline="", encoding="utf-8") as handle:
            cls.canonical = list(csv.DictReader(handle))

    def _position(
        self, condition: str, chunk_size: str, window_name: str
    ) -> dict[str, str]:
        return next(
            row
            for row in self.rows
            if row["condition"] == condition
            and row["chunk_size"] == chunk_size
            and row["window_name"] == window_name
        )

    def _canonical(self, condition: str, chunk_size: str) -> dict[str, str]:
        return next(
            row
            for row in self.canonical
            if row["method"] == "out_of_author_crossfit"
            and row["condition"] == condition
            and row["feature"] == "f3"
            and row["chunk_size"] == chunk_size
            and row["smoothing_eps"] == "0.5"
        )

    def test_prefix_reproduces_declared_detection_point_estimates(self) -> None:
        for condition, chunk_size, window_name in (
            ("flash", "2000", "prefix"),
            ("pro", "2000", "prefix"),
            ("flash", "1000", "window_1"),
            ("pro", "1000", "window_1"),
        ):
            got = self._position(condition, chunk_size, window_name)
            expected = self._canonical(condition, chunk_size)
            for key in ("n_human", "n_llm", "auc", "tpr", "empirical_fpr"):
                self.assertAlmostEqual(
                    float(got[key]),
                    float(expected[key]),
                    places=12,
                    msg=f"{condition} {chunk_size} {window_name} {key}",
                )

    def test_equal_length_windows_keep_llm_counts_fixed(self) -> None:
        for condition, n_llm in (("flash", 400), ("pro", 200)):
            prefix = self._position(condition, "2000", "prefix")
            late = self._position(condition, "2000", "late")
            self.assertEqual(int(prefix["n_llm"]), n_llm)
            self.assertEqual(int(late["n_llm"]), n_llm)
            self.assertEqual(int(prefix["n_human"]), 485)
            self.assertEqual(int(late["n_human"]), 485)
            self.assertEqual(prefix["empirical_fpr"], late["empirical_fpr"])

    def test_five_windows_are_complete(self) -> None:
        names = {
            row["window_name"]
            for row in self.rows
            if row["chunk_size"] == "1000" and row["condition"] == "flash"
        }
        self.assertEqual(
            names, {f"window_{index}" for index in range(1, 6)}
        )


if __name__ == "__main__":
    unittest.main()
