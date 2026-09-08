#!/usr/bin/env python3
"""Small-data checks for the full-corpus D/C implementation."""

from __future__ import annotations

import unittest

import numpy as np

from punctlib.features import features
from punctlib.stats import kl
from tools.analyze_original_gutenberg_dc import (
    PUNCT_INDEX,
    features_from_mark_arrays,
    legacy_kl_matrix,
    mean_ordered_within,
)


def encoded(sequence: list[str]) -> np.ndarray:
    return np.asarray([PUNCT_INDEX[mark] for mark in sequence], dtype=np.uint8)


class OriginalGutenbergDCTests(unittest.TestCase):
    def test_vectorized_kl_matches_vendored_definition_with_zeros(self) -> None:
        distributions = np.asarray(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.2, 0.0, 0.8, 0.0],
                [0.0, 0.3, 0.2, 0.5],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=float,
        )
        observed = legacy_kl_matrix(distributions, distributions, block_size=2)
        expected = np.asarray(
            [[kl(left, right) for right in distributions] for left in distributions]
        )
        np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)

    def test_pooled_features_match_literal_concatenation(self) -> None:
        first = ["!", ",", ".", "?", ";", ".", '"']
        second = [",", ",", ".", ":", "!", "^", "."]
        pooled_f1, pooled_f3 = features_from_mark_arrays(
            (encoded(first), encoded(second))
        )
        literal = features(first + second)
        self.assertIsNotNone(literal)
        np.testing.assert_allclose(pooled_f1, literal.f1, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(pooled_f3, literal.f3, atol=1e-6, rtol=1e-6)

    def test_consistency_averages_both_kl_directions(self) -> None:
        vectors = np.asarray(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.4, 0.2],
                [0.2, 0.3, 0.5],
            ]
        )
        expected = np.mean(
            [
                kl(vectors[i], vectors[j])
                for i in range(len(vectors))
                for j in range(len(vectors))
                if i != j
            ]
        )
        self.assertAlmostEqual(
            mean_ordered_within(vectors, block_size=2),
            expected,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
