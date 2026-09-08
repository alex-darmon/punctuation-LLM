#!/usr/bin/env python3
"""Guardrail tests for the frozen pipeline.

These check the properties the pipeline is frozen in order to guarantee: that a
reference cannot be built without stating an exclusion policy, that exclusions
follow content rather than paths, and that a silently-ignored exclusion is
impossible.

Run with: python tests/test_reference_builder.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from punctlib import (  # noqa: E402
    build_reference,
    build_reference_set,
    load_corpus,
    reset_audit_log,
    audit_log,
)
from punctlib.reference import assert_excluded  # noqa: E402

CONFIG = "campaigns/generation_campaign_phaseA_two_samples.json"


class ReferenceBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = load_corpus(CONFIG)
        cls.author = "jane_austen"

    def test_exclude_books_is_required(self) -> None:
        with self.assertRaises(TypeError):
            build_reference(self.corpus, self.author)  # type: ignore[call-arg]

    def test_byte_identical_books_are_merged(self) -> None:
        books = self.corpus.book_ids(self.author)
        aliases = [alias for book in self.corpus.books(self.author) for alias in book.aliases]
        self.assertEqual(len(books), 2)
        self.assertEqual(len(aliases), 1)

    def test_alias_exclusion_removes_canonical_book(self) -> None:
        aliases = [alias for book in self.corpus.books(self.author) for alias in book.aliases]
        alias = aliases[0]
        canonical = self.corpus.resolve(alias)
        full = build_reference(self.corpus, self.author, exclude_books=())
        reduced = build_reference(self.corpus, self.author, exclude_books=[alias])
        self.assertIsNotNone(full)
        self.assertIsNotNone(reduced)
        assert full is not None and reduced is not None
        self.assertNotIn(canonical, reduced.included_books)
        self.assertLess(reduced.n_marks, full.n_marks)

    def test_unresolvable_exclusion_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_reference(
                self.corpus,
                self.author,
                exclude_books=["does/not/exist.txt"],
            )

    def test_assert_excluded_detects_leakage(self) -> None:
        book = self.corpus.book_ids(self.author)[0]
        full = build_reference(self.corpus, self.author, exclude_books=())
        self.assertIsNotNone(full)
        assert full is not None
        with self.assertRaises(AssertionError):
            assert_excluded(full, self.corpus, book)

    def test_assert_excluded_accepts_clean_reference(self) -> None:
        alias = next(
            alias
            for book in self.corpus.books(self.author)
            for alias in book.aliases
        )
        canonical = self.corpus.resolve(alias)
        reduced = build_reference(self.corpus, self.author, exclude_books=[alias])
        self.assertIsNotNone(reduced)
        assert reduced is not None and canonical is not None
        assert_excluded(reduced, self.corpus, canonical)

    def test_global_exclusion_only_touches_owning_author(self) -> None:
        books = self.corpus.book_ids(self.author)
        refs = build_reference_set(self.corpus, exclude_books=[books[0]])
        self.assertEqual(len(refs[self.author].included_books), len(books) - 1)
        for author in self.corpus.authors:
            if author != self.author:
                self.assertEqual(
                    len(refs[author].included_books),
                    len(self.corpus.book_ids(author)),
                )

    def test_builds_are_audited_with_external_caller(self) -> None:
        reset_audit_log()
        line = sys._getframe().f_lineno + 1
        build_reference_set(self.corpus, exclude_books=())
        entries = audit_log()
        self.assertEqual(len(entries), len(self.corpus.authors))
        self.assertTrue(
            all(
                not entry.caller.startswith("punctlib")
                and entry.caller.endswith(f":{line}")
                for entry in entries
            ),
            [entry.caller for entry in entries[:2]],
        )

    def test_excluding_all_books_returns_none(self) -> None:
        author = next(
            candidate
            for candidate in self.corpus.authors
            if len(self.corpus.book_ids(candidate)) == 2
        )
        self.assertIsNone(
            build_reference(
                self.corpus,
                author,
                exclude_books=self.corpus.book_ids(author),
            )
        )


if __name__ == "__main__":
    unittest.main()
