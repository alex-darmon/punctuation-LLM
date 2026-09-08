"""Cluster-aware summaries for the punctuation evaluation instrument.

The primary population unit is an author.  Books, chunks, prompt-source books,
and repeated generations stay together whenever authors are resampled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable

import numpy as np

from punctlib.stats import auc


def percentile_interval(
    values: Sequence[float], *, confidence: float = 0.95
) -> tuple[float, float]:
    """Central percentile interval, returning NaNs for an empty sample."""
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    low, high = np.quantile(sample, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def author_mean_estimate(
    values_by_author: Mapping[str, float],
    *,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Author-equal mean and author-block bootstrap interval."""
    authors = sorted(values_by_author)
    if not authors:
        raise ValueError("values_by_author is empty")
    values = np.asarray([values_by_author[a] for a in authors], dtype=float)
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean(values[rng.integers(0, len(authors), len(authors))]))
        for _ in range(n_boot)
    ]
    low, high = percentile_interval(draws, confidence=confidence)
    return {
        "n_authors": len(authors),
        "estimate": float(np.mean(values)),
        "ci_low": low,
        "ci_high": high,
    }


def author_accuracy_estimate(
    tally_by_author: Mapping[str, tuple[int, int] | list[int]],
    *,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Micro and author-equal macro accuracy with author-block intervals."""
    clean = {
        author: (int(tally[0]), int(tally[1]))
        for author, tally in tally_by_author.items()
        if int(tally[1]) > 0
    }
    if not clean:
        raise ValueError("tally_by_author contains no observations")
    authors = sorted(clean)
    correct = np.asarray([clean[a][0] for a in authors], dtype=float)
    totals = np.asarray([clean[a][1] for a in authors], dtype=float)
    rates = correct / totals

    rng = np.random.default_rng(seed)
    macro_draws: list[float] = []
    micro_draws: list[float] = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(authors), len(authors))
        macro_draws.append(float(np.mean(rates[indices])))
        micro_draws.append(float(correct[indices].sum() / totals[indices].sum()))

    macro_low, macro_high = percentile_interval(
        macro_draws, confidence=confidence
    )
    micro_low, micro_high = percentile_interval(
        micro_draws, confidence=confidence
    )
    return {
        "n_authors": len(authors),
        "n_observations": int(totals.sum()),
        "n_correct": int(correct.sum()),
        "macro_accuracy": float(np.mean(rates)),
        "macro_ci_low": macro_low,
        "macro_ci_high": macro_high,
        "micro_accuracy": float(correct.sum() / totals.sum()),
        "micro_ci_low": micro_low,
        "micro_ci_high": micro_high,
    }


def paired_author_difference(
    left_by_author: Mapping[str, float],
    right_by_author: Mapping[str, float],
    *,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Paired author-equal mean of ``left - right`` with a bootstrap interval."""
    authors = sorted(set(left_by_author) & set(right_by_author))
    if not authors:
        raise ValueError("paired inputs have no shared authors")
    differences = np.asarray(
        [left_by_author[a] - right_by_author[a] for a in authors], dtype=float
    )
    rng = np.random.default_rng(seed)
    draws = [
        float(
            np.mean(
                differences[rng.integers(0, len(authors), len(authors))]
            )
        )
        for _ in range(n_boot)
    ]
    low, high = percentile_interval(draws, confidence=confidence)
    return {
        "n_authors": len(authors),
        "mean_difference": float(np.mean(differences)),
        "ci_low": low,
        "ci_high": high,
    }


def _repeat_records(
    records: Mapping[str, Sequence[float]], sampled_authors: Sequence[str]
) -> list[float]:
    return [
        float(value)
        for author in sampled_authors
        for value in records.get(author, ())
    ]


def _crossfit_detection_once(
    human_by_author: Mapping[str, Sequence[float]],
    llm_by_author: Mapping[str, Sequence[float]],
    fold_by_author: Mapping[str, int],
    *,
    fpr_target: float,
    sampled_authors: Sequence[str],
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    folds = sorted(set(fold_by_author.values()))
    human_test: list[float] = []
    llm_test: list[float] = []
    human_flags: list[bool] = []
    llm_flags: list[bool] = []
    fold_rows: list[dict[str, float | int]] = []

    for fold in folds:
        calibration_authors = [
            author for author in sampled_authors if fold_by_author[author] != fold
        ]
        test_authors = [
            author for author in sampled_authors if fold_by_author[author] == fold
        ]
        calibration = _repeat_records(human_by_author, calibration_authors)
        test_human = _repeat_records(human_by_author, test_authors)
        test_llm = _repeat_records(llm_by_author, test_authors)
        if not calibration or not test_human or not test_llm:
            continue
        threshold = float(np.percentile(calibration, 100.0 * (1.0 - fpr_target)))
        # Fold thresholds put scores from differently scaled author groups onto
        # one out-of-fold operating-point scale before the pooled ROC/AUC.
        human_test.extend(value / threshold for value in test_human)
        llm_test.extend(value / threshold for value in test_llm)
        human_flags.extend(value > threshold for value in test_human)
        llm_flags.extend(value > threshold for value in test_llm)
        fold_rows.append(
            {
                "fold": fold,
                "n_calibration_authors": len(set(calibration_authors)),
                "n_test_authors": len(set(test_authors)),
                "n_calibration_human": len(calibration),
                "n_test_human": len(test_human),
                "n_test_llm": len(test_llm),
                "threshold": threshold,
                "empirical_fpr": float(np.mean([v > threshold for v in test_human])),
                "tpr": float(np.mean([v > threshold for v in test_llm])),
                "auc": auc(test_llm, test_human),
            }
        )

    if not human_test or not llm_test:
        raise ValueError("cross-fitting produced no test observations")
    metrics = {
        "empirical_fpr": float(np.mean(human_flags)),
        "tpr": float(np.mean(llm_flags)),
        "auc": auc(llm_test, human_test),
    }
    return metrics, fold_rows


def crossfit_detection_estimate(
    human_by_author: Mapping[str, Sequence[float]],
    llm_by_author: Mapping[str, Sequence[float]],
    fold_by_author: Mapping[str, int],
    *,
    fpr_target: float,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Out-of-author calibration with author-block bootstrap uncertainty.

    In each fold, the threshold is estimated from all other authors and applied
    only to human and LLM observations from the held-out authors.  Bootstrap
    draws resample authors and re-estimate every fold threshold.
    """
    authors = sorted(fold_by_author)
    if set(authors) != set(human_by_author) or set(authors) != set(llm_by_author):
        raise ValueError("human, LLM, and fold mappings must cover the same authors")
    point, fold_rows = _crossfit_detection_once(
        human_by_author,
        llm_by_author,
        fold_by_author,
        fpr_target=fpr_target,
        sampled_authors=authors,
    )

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {
        "empirical_fpr": [],
        "tpr": [],
        "auc": [],
    }
    attempts = 0
    while len(draws["auc"]) < n_boot:
        attempts += 1
        if attempts > n_boot * 10:
            raise RuntimeError("too many invalid bootstrap draws")
        sampled = [
            authors[index]
            for index in rng.integers(0, len(authors), len(authors))
        ]
        try:
            metrics, _ = _crossfit_detection_once(
                human_by_author,
                llm_by_author,
                fold_by_author,
                fpr_target=fpr_target,
                sampled_authors=sampled,
            )
        except ValueError:
            continue
        for key in draws:
            draws[key].append(metrics[key])

    out: dict[str, float | int] = {
        "n_authors": len(authors),
        "n_human": sum(len(human_by_author[a]) for a in authors),
        "n_llm": sum(len(llm_by_author[a]) for a in authors),
        "target_fpr": fpr_target,
        **point,
    }
    for key, values in draws.items():
        low, high = percentile_interval(values, confidence=confidence)
        out[f"{key}_ci_low"] = low
        out[f"{key}_ci_high"] = high
    return out, fold_rows


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values in original order."""
    values = np.asarray(pvalues, dtype=float)
    if values.size == 0:
        return []
    order = np.argsort(values)
    adjusted = np.empty(values.size, dtype=float)
    running = 1.0
    for rank_index in range(values.size - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, float(values[original_index]) * values.size / rank)
        adjusted[original_index] = min(1.0, running)
    return [float(value) for value in adjusted]


def bootstrap_statistic(
    records_by_author: Mapping[str, Sequence[float]],
    statistic: Callable[[Sequence[float]], float],
    *,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Generic author-block bootstrap for a scalar statistic."""
    authors = sorted(records_by_author)
    if not authors:
        raise ValueError("records_by_author is empty")
    observed = _repeat_records(records_by_author, authors)
    rng = np.random.default_rng(seed)
    draws = [
        statistic(
            _repeat_records(
                records_by_author,
                [
                    authors[index]
                    for index in rng.integers(0, len(authors), len(authors))
                ],
            )
        )
        for _ in range(n_boot)
    ]
    low, high = percentile_interval(draws, confidence=confidence)
    return {
        "n_authors": len(authors),
        "estimate": float(statistic(observed)),
        "ci_low": low,
        "ci_high": high,
    }
