"""Punctuation feature extraction.

Two feature spaces are used, matching Darmon et al.:

  f1  mark frequencies, a length-k probability vector.
  f3  the joint distribution over successive mark pairs, k*k entries, obtained by
      scaling each row of the transition matrix by the marginal frequency of that
      row's mark. Flattened for use as a distribution.

Both are computed by the vendored library. Alongside them this module produces
the count arrays and smoothed profiles the likelihood-ratio statistic needs, so
the KL path and the G-statistic path derive from one set of definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from punctlib.vendored import (
    PUNCT_VECTOR,
    get_frequencies,
    normalised_transition_mat,
    transition_mat,
)

K = len(PUNCT_VECTOR)
INDEX = {mark: i for i, mark in enumerate(PUNCT_VECTOR)}
FEATURES = ("f1", "f3")

# Add-eps smoothing keeps reference profiles free of exact zeros, which would
# make the likelihood-ratio statistic infinite for any mark the reference author
# never used.
SMOOTH_EPS = 0.5


@dataclass(frozen=True)
class Features:
    """Feature vectors for one sequence of punctuation marks."""

    n_marks: int
    f1: tuple[float, ...]
    f3: tuple[float, ...]

    def __getitem__(self, name: str) -> tuple[float, ...]:
        if name == "f1":
            return self.f1
        if name == "f3":
            return self.f3
        raise KeyError(name)


def features(seq: Sequence[str]) -> Features | None:
    """f1 and f3 for a mark sequence, or None if the sequence carries no signal."""
    if not seq:
        return None
    seq = list(seq)
    f1 = get_frequencies(seq, vector=PUNCT_VECTOR)
    if f1 is None or sum(f1) == 0:
        return None
    f2 = transition_mat(seq)
    if f2 is None:
        return None
    f3 = normalised_transition_mat(f2, f1)
    if f3 is None:
        return None
    return Features(
        n_marks=len(seq),
        f1=tuple(float(x) for x in f1),
        f3=tuple(float(x) for x in np.asarray(f3).flatten()),
    )


def counts1(seq: Sequence[str]) -> np.ndarray:
    """Observed mark counts, the multinomial observation for the f1 G-statistic."""
    out = np.zeros(K, dtype=float)
    for mark in seq:
        i = INDEX.get(mark)
        if i is not None:
            out[i] += 1
    return out


def counts2(seq: Sequence[str]) -> np.ndarray:
    """Observed successive-pair counts, the observation for the f3 G-statistic.

    A mark outside the punctuation vector breaks the chain rather than being
    skipped over, so no spurious adjacency is created across it.
    """
    out = np.zeros((K, K), dtype=float)
    prev = None
    for mark in seq:
        i = INDEX.get(mark)
        if i is None:
            prev = None
            continue
        if prev is not None:
            out[prev, i] += 1
        prev = i
    return out


def smooth1(counts: np.ndarray, eps: float = SMOOTH_EPS) -> np.ndarray:
    v = np.asarray(counts, dtype=float) + eps
    return v / v.sum()


def smooth2(counts: np.ndarray, eps: float = SMOOTH_EPS) -> np.ndarray:
    m = np.asarray(counts, dtype=float) + eps
    return m / m.sum(axis=1, keepdims=True)


def chunks(seq: Sequence[str], size: int, limit: int | None = None) -> Iterator[list[str]]:
    """Non-overlapping windows of exactly `size` marks; a short tail is dropped."""
    n = len(seq) // size
    if limit is not None:
        n = min(n, limit)
    for i in range(n):
        yield list(seq[i * size : (i + 1) * size])


def middle_chunk(seq: Sequence[str], size: int) -> list[str] | None:
    """The `size` marks centred on the midpoint, as used for figure 1 of the paper."""
    if len(seq) < size:
        return None
    start = (len(seq) - size) // 2
    return list(seq[start : start + size])
