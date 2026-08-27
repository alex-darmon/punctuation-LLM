"""The one way to build an author reference profile.

`exclude_books` is a required keyword argument. There is deliberately no default:
a caller that has not decided which texts must be kept out of the profile has not
decided what its experiment measures. Passing an empty tuple is allowed and means
"pool everything", but it has to be written down.

Every profile built during a run is appended to an audit log, which the driver
writes out alongside the results. That log is what makes a leakage claim
checkable after the fact instead of a matter of reading code.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from punctlib.corpus import Corpus
from punctlib.features import (
    counts1,
    counts2,
    features as extract_features,
    smooth1,
    smooth2,
)


@dataclass(frozen=True)
class Reference:
    """An author profile, in both the KL feature space and the G-statistic space."""

    author: str
    included_books: tuple[str, ...]
    excluded_books: tuple[str, ...]
    n_marks: int

    # KL space: distributions comparable with a chunk's own f1/f3.
    f1: tuple[float, ...]
    f3: tuple[float, ...]

    # G-statistic space: smoothed null profiles for the likelihood-ratio test.
    p1: np.ndarray = field(repr=False)
    p3: np.ndarray = field(repr=False)

    def __getitem__(self, name: str) -> tuple[float, ...]:
        if name == "f1":
            return self.f1
        if name == "f3":
            return self.f3
        raise KeyError(name)

    def profile(self, name: str) -> np.ndarray:
        if name == "f1":
            return self.p1
        if name == "f3":
            return self.p3
        raise KeyError(name)


@dataclass(frozen=True)
class AuditEntry:
    author: str
    included_books: tuple[str, ...]
    excluded_books: tuple[str, ...]
    n_marks: int
    caller: str


_AUDIT: list[AuditEntry] = []
_CACHE: dict[tuple[str, tuple[str, ...]], Reference] = {}


def reset_audit_log() -> None:
    _AUDIT.clear()


def audit_log() -> list[AuditEntry]:
    return list(_AUDIT)


def _caller() -> str:
    """Nearest frame outside punctlib, so the audit names the experiment."""
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if not name.startswith("punctlib"):
            return f"{name}.{frame.f_code.co_name}:{frame.f_lineno}"
        frame = frame.f_back
    return "unknown"


def build_reference(
    corpus: Corpus,
    author: str,
    *,
    exclude_books: tuple[str, ...] | list[str],
) -> Reference | None:
    """Pool an author's books into a profile, omitting `exclude_books`.

    Exclusions are resolved through content identity, so excluding any alias of a
    text excludes every copy of it. An exclusion naming a text that is not in the
    corpus raises, rather than silently failing to exclude anything.

    Returns None when every book of the author was excluded.
    """
    requested: list[str] = []
    excluded_ids: set[str] = set()
    for item in exclude_books:
        requested.append(str(item))
        book_id = corpus.resolve(item)
        if book_id is None:
            raise ValueError(
                f"cannot exclude {item!r}: no book with that path or content is in "
                "the corpus, so the exclusion would be silently ignored"
            )
        excluded_ids.add(book_id)

    included = tuple(b for b in corpus.book_ids(author) if b not in excluded_ids)
    if not included:
        return None

    key = (author, included)
    cached = _CACHE.get(key)
    if cached is None:
        seq = [m for book_id in included for m in corpus.marks(book_id)]
        feats = extract_features(seq)
        if feats is None:
            return None
        cached = Reference(
            author=author,
            included_books=included,
            excluded_books=tuple(sorted(excluded_ids)),
            n_marks=len(seq),
            f1=feats.f1,
            f3=feats.f3,
            p1=smooth1(counts1(seq)),
            p3=smooth2(counts2(seq)),
        )
        _CACHE[key] = cached

    _AUDIT.append(
        AuditEntry(
            author=author,
            included_books=included,
            excluded_books=tuple(sorted(excluded_ids)),
            n_marks=cached.n_marks,
            caller=_caller(),
        )
    )
    return cached


def build_reference_set(
    corpus: Corpus,
    *,
    exclude_books: tuple[str, ...] | list[str],
) -> dict[str, Reference]:
    """Profiles for every author, under one shared exclusion policy.

    Exclusions are global: an excluded book is absent from every profile, not
    just from its own author's. That keeps the policy a property of the run
    rather than of individual comparisons.
    """
    out: dict[str, Reference] = {}
    for author in corpus.authors:
        ref = build_reference(corpus, author, exclude_books=exclude_books)
        if ref is not None:
            out[author] = ref
    return out


def assert_excluded(reference: Reference, corpus: Corpus, book_id: str) -> None:
    """Fail loudly if the text about to be scored is inside its own reference."""
    canonical = corpus.resolve(book_id)
    if canonical is not None and canonical in reference.included_books:
        raise AssertionError(
            f"leakage: {canonical} is in the reference for {reference.author} "
            f"({reference.included_books})"
        )
