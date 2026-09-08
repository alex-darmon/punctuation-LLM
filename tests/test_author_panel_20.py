#!/usr/bin/env python3
"""Configuration and provenance checks for the 20-author extension."""

from __future__ import annotations

import hashlib
import json
import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "campaigns" / "author_panel_20.json"
FLASH_PATH = ROOT / "campaigns" / "generation_campaign_new10_flash.json"
PRO_PATH = ROOT / "campaigns" / "generation_campaign_new10_pro.json"
MANIFEST_PATH = ROOT / "results" / "author_panel_20" / "source_manifest.json"
ATTRIBUTION_PATH = (
    ROOT / "results" / "author_panel_20" / "preflight" / "attribution.csv"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AuthorPanel20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = load(PANEL_PATH)
        cls.flash = load(FLASH_PATH)
        cls.pro = load(PRO_PATH)

    def test_panel_is_balanced_and_keys_are_unique(self) -> None:
        design = self.panel["panel_design"]
        authors = self.panel["authors"]
        keys = [author["key"] for author in authors]

        self.assertEqual(len(keys), design["existing_authors"] + design["new_authors"])
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            sum(author["cohort"] == "existing" for author in authors),
            design["existing_authors"],
        )
        self.assertEqual(
            sum(author["cohort"] == "new" for author in authors),
            design["new_authors"],
        )

        selected_paths: set[str] = set()
        for author in authors:
            books = author["panel_books"]
            self.assertEqual(len(books), design["books_per_author"], author["key"])
            panel_paths = {book["path"] for book in books}
            self.assertTrue(panel_paths.issubset(set(author["book_paths"])))
            self.assertTrue(selected_paths.isdisjoint(panel_paths))
            selected_paths.update(panel_paths)

    def test_new_authors_were_screened_above_one(self) -> None:
        new_authors = [
            author for author in self.panel["authors"] if author["cohort"] == "new"
        ]
        self.assertEqual(len(new_authors), 10)
        self.assertTrue(
            all(author["full_corpus_f3_dc"] > 1.0 for author in new_authors)
        )

    def test_generation_configs_match_the_new_cohort(self) -> None:
        panel_new = [
            author for author in self.panel["authors"] if author["cohort"] == "new"
        ]
        panel_by_key = {author["key"]: author for author in panel_new}
        expected_keys = list(panel_by_key)

        for campaign in (self.flash, self.pro):
            self.assertEqual(
                [author["key"] for author in campaign["authors"]], expected_keys
            )
            self.assertEqual(campaign["min_source_books_per_author"], 2)
            self.assertEqual(campaign["dash_policy"], "replace_with_comma")
            self.assertTrue(campaign["save_raw_outputs"])
            for author in campaign["authors"]:
                paths = author["book_paths"]
                selected = {
                    book["path"]
                    for book in panel_by_key[author["key"]]["panel_books"]
                }
                self.assertEqual(len(paths), 2)
                self.assertEqual(len(set(paths)), 2)
                self.assertTrue(set(paths).issubset(selected))

        self.assertEqual(self.flash["default_model"], "gemini-2.5-flash")
        self.assertEqual(self.flash["default_runs"], 20)
        self.assertEqual(self.flash["default_target_marks"], 5000)
        self.assertEqual(self.pro["default_model"], "gemini-2.5-pro")
        self.assertEqual(self.pro["default_runs"], 10)
        self.assertEqual(self.pro["default_target_marks"], 5000)

    def test_source_manifest_pins_every_selected_book(self) -> None:
        manifest = load(MANIFEST_PATH)
        rows = manifest["books"]
        expected = sum(
            (author["panel_books"] for author in self.panel["authors"]), start=[]
        )
        self.assertEqual(len(rows), len(expected))

        expected_paths = {book["path"] for book in expected}
        self.assertEqual({row["path"] for row in rows}, expected_paths)
        for row in rows:
            path = ROOT / row["path"]
            self.assertTrue(path.is_file(), row["path"])
            self.assertEqual(path.stat().st_size, row["bytes"])
            self.assertEqual(digest(path), row["sha256"])

    def test_every_new_author_passes_the_preflight_gate(self) -> None:
        with ATTRIBUTION_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        new_keys = {
            author["key"]
            for author in self.panel["authors"]
            if author["cohort"] == "new"
        }
        document_rows = {
            row["author"]: row
            for row in rows
            if row["experiment"] == "human_attribution"
            and row["author_set"] == "all_authors"
            and row["unit"] == "document"
            and row["feature"] == "f3"
            and row["author"] in new_keys
        }
        self.assertEqual(set(document_rows), new_keys)
        for author, row in document_rows.items():
            self.assertEqual(int(row["n"]), 3, author)
            self.assertGreaterEqual(int(row["correct"]), 2, author)


if __name__ == "__main__":
    unittest.main()
