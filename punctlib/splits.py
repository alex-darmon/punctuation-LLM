"""Deterministic group splits for leakage-controlled inference.

Splits are assigned above the observation level.  In the primary detection
analysis an author, together with all of that author's books, chunks, prompts,
and generations, belongs to exactly one test fold.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence


def stratified_group_folds(
    groups: Sequence[str],
    strata: Mapping[str, str],
    *,
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    """Assign each unique group to one fold, balanced within each stratum."""
    unique = list(dict.fromkeys(groups))
    if len(unique) != len(groups):
        raise ValueError("groups must be unique")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(unique) < n_folds:
        raise ValueError("number of groups must be at least n_folds")
    if set(unique) != set(strata):
        missing = sorted(set(unique) - set(strata))
        extra = sorted(set(strata) - set(unique))
        raise ValueError(f"strata keys differ from groups: missing={missing}, extra={extra}")

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for group in unique:
        by_stratum[strata[group]].append(group)

    assignment: dict[str, int] = {}
    for offset, stratum in enumerate(sorted(by_stratum)):
        members = sorted(by_stratum[stratum])
        if len(members) < n_folds:
            raise ValueError(
                f"stratum {stratum!r} has {len(members)} groups for {n_folds} folds"
            )
        rng = random.Random(seed + offset)
        rng.shuffle(members)
        for index, group in enumerate(members):
            assignment[group] = index % n_folds

    validate_group_folds(assignment, n_folds=n_folds)
    return assignment


def validate_group_folds(assignment: Mapping[str, int], *, n_folds: int) -> None:
    """Raise when an assignment omits a fold or contains an invalid fold."""
    if not assignment:
        raise ValueError("fold assignment is empty")
    invalid = sorted({fold for fold in assignment.values() if fold not in range(n_folds)})
    if invalid:
        raise ValueError(f"invalid fold identifiers: {invalid}")
    counts = Counter(assignment.values())
    missing = [fold for fold in range(n_folds) if counts[fold] == 0]
    if missing:
        raise ValueError(f"empty folds: {missing}")


def split_for_fold(
    assignment: Mapping[str, int], fold: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return disjoint calibration and test group tuples for one fold."""
    test = tuple(sorted(group for group, value in assignment.items() if value == fold))
    calibration = tuple(
        sorted(group for group, value in assignment.items() if value != fold)
    )
    if not test or not calibration:
        raise ValueError(f"fold {fold} has an empty calibration or test side")
    if set(test) & set(calibration):
        raise AssertionError("a group occurs on both sides of a split")
    return calibration, test
