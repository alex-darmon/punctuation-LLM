#!/usr/bin/env python3
"""Cluster-aware, cross-fitted inference for the 20-author evaluation instrument.

This is intentionally additive.  ``run_frozen_grid.py`` and its canonical
outputs remain unchanged; this driver tightens inference around the already
generated texts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from punctlib import (
    FEATURES,
    PUNCT_VECTOR,
    assert_excluded,
    audit_log,
    build_reference_set,
    chunks,
    counts1,
    counts2,
    features,
    kl,
    load_corpus,
    reset_audit_log,
)
from punctlib.features import smooth1, smooth2
from punctlib.inference import (
    author_accuracy_estimate,
    author_mean_estimate,
    benjamini_hochberg,
    crossfit_detection_estimate,
    paired_author_difference,
    percentile_interval,
)
from punctlib.splits import split_for_fold, stratified_group_folds
from punctlib.stats import auc, g_stat


ROOT = Path(__file__).resolve().parent
ANALYTICAL_OUTPUT_FILES = (
    "attribution_clustered.csv",
    "attribution_observations.csv",
    "dash_sensitivity.csv",
    "detection_crossfit.csv",
    "detection_folds.csv",
    "detection_observations.csv",
    "drift_clustered.csv",
    "headline_summary.json",
    "model_contrasts.csv",
    "pooled_leakage_contrasts.csv",
    "prompt_cluster_attribution.csv",
    "reference_audit.csv",
    "separability_crossfit.csv",
    "separability_crossfit_summary.csv",
    "split_assignments.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="campaigns/inference_v2.json",
        help="Inference declaration JSON.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=None,
        help="Override the declared bootstrap count (useful for smoke tests).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override config output_dir; use this for non-destructive reruns.",
    )
    return parser.parse_args()


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(directory: Path, pattern: str = "*.txt") -> str:
    """Hash relative names and contents for every matching file in a tree."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob(pattern)):
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    """Record every package that can affect numerical or rendered outputs."""
    versions: dict[str, str | None] = {}
    for distribution in (
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "spacy",
        "en-core-web-sm",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def git_metadata(*, excluded_prefixes: tuple[str, ...] = ()) -> dict[str, Any]:
    """Describe the source tree, excluding declared generated output paths."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
        ).splitlines()
        outside_outputs = []
        for line in status:
            changed_path = line[3:].split(" -> ")[-1].replace("\\", "/")
            if not any(
                changed_path == prefix.rstrip("/")
                or changed_path.startswith(prefix.rstrip("/") + "/")
                for prefix in excluded_prefixes
            ):
                outside_outputs.append(line)
        return {
            "commit": commit,
            "branch": branch,
            "dirty_outside_declared_outputs": bool(outside_outputs),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "branch": None,
            "dirty_outside_declared_outputs": None,
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [wrote] {path.relative_to(ROOT)} ({len(rows)} rows)")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"  [wrote] {path.relative_to(ROOT)}")


def _feature_cache_get(
    cache: dict[tuple[str, int], Any],
    key: str,
    sequence: list[str],
    size: int,
):
    cache_key = (key, size)
    if cache_key not in cache:
        cache[cache_key] = features(sequence[:size])
    return cache[cache_key]


def _attribute(feature: str, feats: Any, refs: dict[str, Any]) -> str:
    return min(refs, key=lambda author: kl(feats[feature], refs[author][feature]))


def collect_attribution_observations(
    corpus,
    *,
    author_set: str,
    conditions: dict[str, str],
    feature: str,
    chunk_size: int,
    cohort_by_author: dict[str, str],
    feature_cache: dict[tuple[str, int], Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for author in corpus.authors:
        for book_id in corpus.book_ids(author):
            refs = build_reference_set(corpus, exclude_books=[book_id])
            assert_excluded(refs[author], corpus, book_id)
            for chunk_index, chunk in enumerate(
                chunks(corpus.marks(book_id), chunk_size), 1
            ):
                cache_key = f"{book_id}#chunk{chunk_index}"
                feats = _feature_cache_get(
                    feature_cache, cache_key, chunk, len(chunk)
                )
                if feats is None:
                    continue
                predicted = _attribute(feature, feats, refs)
                rows.append(
                    {
                        "author_set": author_set,
                        "source_type": "human",
                        "condition": "",
                        "feature": feature,
                        "chunk_size": chunk_size,
                        "author": author,
                        "cohort": cohort_by_author[author],
                        "sample_id": cache_key,
                        "book_id": book_id,
                        "prompt_source_book": "",
                        "run_id": "",
                        "predicted_author": predicted,
                        "correct": int(predicted == author),
                    }
                )

    for condition in conditions:
        for author in corpus.authors:
            for run in corpus.runs(condition, author):
                if run.source_book_id is None or run.n_marks < chunk_size:
                    continue
                refs = build_reference_set(
                    corpus, exclude_books=[run.source_book_id]
                )
                assert_excluded(refs[author], corpus, run.source_book_id)
                feats = _feature_cache_get(
                    feature_cache,
                    run.cache_key,
                    corpus.marks(run.cache_key),
                    chunk_size,
                )
                if feats is None:
                    continue
                predicted = _attribute(feature, feats, refs)
                rows.append(
                    {
                        "author_set": author_set,
                        "source_type": "llm",
                        "condition": condition,
                        "feature": feature,
                        "chunk_size": chunk_size,
                        "author": author,
                        "cohort": cohort_by_author[author],
                        "sample_id": run.cache_key,
                        "book_id": "",
                        "prompt_source_book": run.source_book_id,
                        "run_id": run.run_id,
                        "predicted_author": predicted,
                        "correct": int(predicted == author),
                    }
                )
    return rows


def summarize_attribution(
    observations: list[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
    confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        key = (
            row["author_set"],
            row["source_type"],
            row["condition"],
            row["feature"],
            row["chunk_size"],
        )
        grouped[key].append(row)

    for group_index, (key, rows) in enumerate(sorted(grouped.items())):
        author_set, source_type, condition, feature, chunk_size = key
        tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        predictions: Counter[str] = Counter()
        for row in rows:
            tally[row["author"]][0] += int(row["correct"])
            tally[row["author"]][1] += 1
            predictions[row["predicted_author"]] += 1
        estimate = author_accuracy_estimate(
            tally,
            n_boot=n_boot,
            seed=seed + group_index,
            confidence=confidence,
        )
        common = {
            "row_type": "summary",
            "author_set": author_set,
            "source_type": source_type,
            "condition": condition,
            "feature": feature,
            "chunk_size": chunk_size,
        }
        summary.append(
            {
                **common,
                "author": "ALL",
                "cohort": "all",
                **estimate,
                "cluster_unit": "author",
                "n_clusters": estimate["n_authors"],
                "most_predicted": "; ".join(
                    f"{author}:{count}" for author, count in predictions.most_common(3)
                ),
            }
        )
        for author in sorted(tally):
            correct, total = tally[author]
            author_cohorts = {
                row["cohort"] for row in rows if row["author"] == author
            }
            author_observations = [row for row in rows if row["author"] == author]
            cluster_field = (
                "book_id" if source_type == "human" else "prompt_source_book"
            )
            by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in author_observations:
                by_cluster[str(row[cluster_field])].append(row)
            clusters = sorted(by_cluster)
            rng = np.random.default_rng(
                seed + group_index * 1000 + len(summary)
            )
            cluster_draws: list[float] = []
            for _ in range(n_boot):
                sampled = [
                    clusters[index]
                    for index in rng.integers(0, len(clusters), len(clusters))
                ]
                sampled_rows = [
                    row for cluster in sampled for row in by_cluster[cluster]
                ]
                cluster_draws.append(
                    float(np.mean([row["correct"] for row in sampled_rows]))
                )
            author_ci_low, author_ci_high = percentile_interval(
                cluster_draws, confidence=confidence
            )
            summary.append(
                {
                    **common,
                    "row_type": "author",
                    "author": author,
                    "cohort": (
                        next(iter(author_cohorts))
                        if len(author_cohorts) == 1
                        else ""
                    ),
                    "n_authors": 1,
                    "n_observations": total,
                    "n_correct": correct,
                    "macro_accuracy": correct / total,
                    "macro_ci_low": author_ci_low,
                    "macro_ci_high": author_ci_high,
                    "micro_accuracy": correct / total,
                    "micro_ci_low": author_ci_low,
                    "micro_ci_high": author_ci_high,
                    "cluster_unit": cluster_field,
                    "n_clusters": len(clusters),
                    "most_predicted": "",
                }
            )

        if source_type == "llm":
            prompt_groups: dict[tuple[str, str], list[int]] = defaultdict(
                lambda: [0, 0]
            )
            for row in rows:
                cluster = (row["author"], row["prompt_source_book"])
                prompt_groups[cluster][0] += int(row["correct"])
                prompt_groups[cluster][1] += 1
            for (author, source_book), (correct, total) in sorted(
                prompt_groups.items()
            ):
                prompt_rows.append(
                    {
                        "author_set": author_set,
                        "condition": condition,
                        "feature": feature,
                        "chunk_size": chunk_size,
                        "author": author,
                        "prompt_source_book": source_book,
                        "n_runs": total,
                        "n_correct": correct,
                        "accuracy": correct / total,
                    }
                )
    return summary, prompt_rows


def model_contrast(
    observations: list[dict[str, Any]],
    *,
    primary: dict[str, Any],
    n_boot: int,
    seed: int,
    confidence: float,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in observations
        if row["author_set"] == primary["author_set"]
        and row["source_type"] == "llm"
        and row["feature"] == primary["feature"]
        and row["chunk_size"] == primary["chunk_size"]
    ]
    conditions = sorted({row["condition"] for row in selected})
    if len(conditions) != 2:
        return []
    rates: dict[str, dict[str, float]] = {}
    for condition in conditions:
        tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in selected:
            if row["condition"] == condition:
                tally[row["author"]][0] += int(row["correct"])
                tally[row["author"]][1] += 1
        rates[condition] = {
            author: correct / total
            for author, (correct, total) in tally.items()
            if total
        }
    left, right = conditions
    estimate = paired_author_difference(
        rates[left],
        rates[right],
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
    )
    authors = sorted(set(rates[left]) & set(rates[right]))
    differences = [rates[left][a] - rates[right][a] for a in authors]
    try:
        pvalue = float(wilcoxon(differences, alternative="two-sided").pvalue)
    except ValueError:
        pvalue = 1.0
    return [
        {
            "left_condition": left,
            "right_condition": right,
            "feature": primary["feature"],
            "chunk_size": primary["chunk_size"],
            **estimate,
            "paired_wilcoxon_p": pvalue,
        }
    ]


def _included_books(corpus, author: str, excluded: list[str]) -> tuple[str, ...]:
    excluded_ids = {corpus.resolve(item) for item in excluded}
    return tuple(
        book_id
        for book_id in corpus.book_ids(author)
        if book_id not in excluded_ids
    )


def _g_profile(
    corpus,
    author: str,
    excluded: list[str],
    feature: str,
    eps: float,
    cache: dict[tuple[Any, ...], np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...]]:
    included = _included_books(corpus, author, excluded)
    if not included:
        raise ValueError(f"no reference books remain for {author}")
    key = (author, included, feature, eps)
    if key not in cache:
        sequence = [
            mark for book_id in included for mark in corpus.marks(book_id)
        ]
        counts = counts1(sequence) if feature == "f1" else counts2(sequence)
        cache[key] = smooth1(counts, eps) if feature == "f1" else smooth2(counts, eps)
    return cache[key], included


def collect_detection_scores(
    corpus,
    *,
    conditions: dict[str, str],
    feature: str,
    chunk_size: int,
    eps: float,
    profile_cache: dict[tuple[Any, ...], np.ndarray],
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    human: dict[str, list[float]] = {author: [] for author in corpus.authors}
    llm: dict[str, dict[str, list[float]]] = {
        condition: {author: [] for author in corpus.authors}
        for condition in conditions
    }
    for author in corpus.authors:
        for book_id in corpus.book_ids(author):
            profile, included = _g_profile(
                corpus, author, [book_id], feature, eps, profile_cache
            )
            if book_id in included:
                raise AssertionError(f"human detection leakage: {book_id}")
            for chunk in chunks(corpus.marks(book_id), chunk_size):
                obs = counts1(chunk) if feature == "f1" else counts2(chunk)
                human[author].append(g_stat(feature, obs, profile))

    for condition in conditions:
        for author in corpus.authors:
            for run in corpus.runs(condition, author):
                if run.source_book_id is None or run.n_marks < chunk_size:
                    continue
                profile, included = _g_profile(
                    corpus,
                    author,
                    [run.source_book_id],
                    feature,
                    eps,
                    profile_cache,
                )
                if run.source_book_id in included:
                    raise AssertionError(
                        f"LLM detection leakage: {run.source_book_id}"
                    )
                sequence = corpus.marks(run.cache_key)[:chunk_size]
                obs = counts1(sequence) if feature == "f1" else counts2(sequence)
                llm[condition][author].append(g_stat(feature, obs, profile))
    return human, llm


def detection_analysis(
    corpus,
    *,
    conditions: dict[str, str],
    config: dict[str, Any],
    folds: dict[str, int],
    n_boot: int,
    confidence: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    sensitivity = config["sensitivity"]
    primary = config["primary"]
    seed = int(config["inference"]["seed"])
    target_fpr = float(primary["detection_fpr"])
    profile_cache: dict[tuple[Any, ...], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    jobs = [
        (feature, size, 0.5)
        for feature in sensitivity["features"]
        for size in sensitivity["chunk_sizes"]
    ]
    jobs += [
        (primary["feature"], primary["chunk_size"], float(eps))
        for eps in sensitivity["smoothing_eps"]
        if float(eps) != 0.5
    ]

    for job_index, (feature, size, eps) in enumerate(jobs):
        human, llm = collect_detection_scores(
            corpus,
            conditions=conditions,
            feature=feature,
            chunk_size=int(size),
            eps=float(eps),
            profile_cache=profile_cache,
        )
        hvals = [value for author in corpus.authors for value in human[author]]
        for condition_index, condition in enumerate(conditions):
            estimate, per_fold = crossfit_detection_estimate(
                human,
                llm[condition],
                folds,
                fpr_target=target_fpr,
                n_boot=n_boot,
                seed=seed + 1000 + job_index * 20 + condition_index,
                confidence=confidence,
            )
            rows.append(
                {
                    "method": "out_of_author_crossfit",
                    "condition": condition,
                    "feature": feature,
                    "chunk_size": size,
                    "smoothing_eps": eps,
                    **estimate,
                }
            )
            for fold_row in per_fold:
                fold_rows.append(
                    {
                        "condition": condition,
                        "feature": feature,
                        "chunk_size": size,
                        "smoothing_eps": eps,
                        **fold_row,
                    }
                )
            if (
                feature == primary["feature"]
                and int(size) == int(primary["chunk_size"])
                and float(eps) == 0.5
            ):
                threshold_by_fold = {
                    int(row["fold"]): float(row["threshold"]) for row in per_fold
                }
                for label, scores in (
                    ("human", human),
                    ("llm", llm[condition]),
                ):
                    for author, values in scores.items():
                        fold = folds[author]
                        threshold = threshold_by_fold[fold]
                        for observation_index, score in enumerate(values, 1):
                            observations.append(
                                {
                                    "condition": condition,
                                    "label": label,
                                    "author": author,
                                    "fold": fold,
                                    "observation_index": observation_index,
                                    "score": score,
                                    "calibrated_threshold": threshold,
                                    "normalised_score": score / threshold,
                                    "flagged": int(score > threshold),
                                }
                            )

            lvals = [
                value
                for author in corpus.authors
                for value in llm[condition][author]
            ]
            threshold = float(np.percentile(hvals, 100 * (1 - target_fpr)))
            rows.append(
                {
                    "method": "in_sample_calibration_contrast",
                    "condition": condition,
                    "feature": feature,
                    "chunk_size": size,
                    "smoothing_eps": eps,
                    "n_authors": len(corpus.authors),
                    "n_human": len(hvals),
                    "n_llm": len(lvals),
                    "target_fpr": target_fpr,
                    "empirical_fpr": float(np.mean(np.asarray(hvals) > threshold)),
                    "tpr": float(np.mean(np.asarray(lvals) > threshold)),
                    "auc": auc(lvals, hvals),
                    "threshold": threshold,
                }
            )
    return rows, fold_rows, observations


def crossfit_separability(
    corpus,
    *,
    features_to_run: list[str],
    chunk_size: int,
    n_boot: int,
    seed: int,
    confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    for feature in features_to_run:
        for fold in range(3):
            held_out = [corpus.book_ids(author)[fold] for author in corpus.authors]
            refs = build_reference_set(corpus, exclude_books=held_out)
            for author in corpus.authors:
                test_book = corpus.book_ids(author)[fold]
                assert_excluded(refs[author], corpus, test_book)
                included = refs[author].included_books
                if len(included) != 2:
                    raise AssertionError(
                        f"cross-fitted D/C requires two reference books for {author}"
                    )
                book_features = [features(corpus.marks(book_id)) for book_id in included]
                if any(value is None for value in book_features):
                    continue
                first, second = book_features
                consistency = 0.5 * (
                    kl(first[feature], second[feature])
                    + kl(second[feature], first[feature])
                )
                nearest_distance, nearest_author = min(
                    (
                        kl(refs[author][feature], refs[other][feature]),
                        other,
                    )
                    for other in corpus.authors
                    if other != author
                )
                ratio = (
                    nearest_distance / consistency
                    if consistency > 0
                    else float("inf")
                )
                document_features = features(corpus.marks(test_book))
                document_pick = _attribute(feature, document_features, refs)
                chunk_correct = 0
                chunk_total = 0
                for chunk in chunks(corpus.marks(test_book), chunk_size):
                    chunk_features = features(chunk)
                    if chunk_features is None:
                        continue
                    chunk_total += 1
                    chunk_correct += (
                        _attribute(feature, chunk_features, refs) == author
                    )
                detail.append(
                    {
                        "feature": feature,
                        "fold": fold,
                        "author": author,
                        "held_out_book": test_book,
                        "reference_books": ";".join(included),
                        "consistency_C": consistency,
                        "nearest_other_D": nearest_distance,
                        "nearest_other": nearest_author,
                        "margin_D_over_C": ratio,
                        "document_correct": int(document_pick == author),
                        "chunk_size": chunk_size,
                        "chunk_correct": chunk_correct,
                        "chunk_total": chunk_total,
                        "chunk_accuracy": (
                            chunk_correct / chunk_total if chunk_total else float("nan")
                        ),
                    }
                )

    summary: list[dict[str, Any]] = []
    for feature in features_to_run:
        rows = [row for row in detail if row["feature"] == feature]
        by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_author[row["author"]].append(row)
        author_rows: list[dict[str, Any]] = []
        for author, author_detail in sorted(by_author.items()):
            ratio = float(
                np.exp(
                    np.mean(
                        np.log(
                            [
                                max(float(row["margin_D_over_C"]), 1e-12)
                                for row in author_detail
                            ]
                        )
                    )
                )
            )
            document_accuracy = float(
                np.mean([row["document_correct"] for row in author_detail])
            )
            chunk_correct = sum(row["chunk_correct"] for row in author_detail)
            chunk_total = sum(row["chunk_total"] for row in author_detail)
            author_rows.append(
                {
                    "row_type": "author",
                    "feature": feature,
                    "author": author,
                    "geometric_mean_D_over_C": ratio,
                    "document_accuracy": document_accuracy,
                    "chunk_accuracy": chunk_correct / chunk_total,
                    "chunk_correct": chunk_correct,
                    "chunk_total": chunk_total,
                }
            )
        ratios = [row["geometric_mean_D_over_C"] for row in author_rows]
        accuracies = [row["chunk_accuracy"] for row in author_rows]
        rho, pvalue = spearmanr(ratios, accuracies)
        rng = np.random.default_rng(seed + (0 if feature == "f1" else 1))
        rho_draws: list[float] = []
        for _ in range(n_boot):
            indices = rng.integers(0, len(author_rows), len(author_rows))
            sampled_rho = spearmanr(
                [ratios[index] for index in indices],
                [accuracies[index] for index in indices],
            ).statistic
            if np.isfinite(sampled_rho):
                rho_draws.append(float(sampled_rho))
        rho_low, rho_high = percentile_interval(
            rho_draws, confidence=confidence
        )
        summary.append(
            {
                "row_type": "summary",
                "feature": feature,
                "author": "ALL",
                "n_authors": len(author_rows),
                "spearman_rho": float(rho),
                "spearman_p": float(pvalue),
                "spearman_ci_low": rho_low,
                "spearman_ci_high": rho_high,
                "document_macro_accuracy": float(
                    np.mean([row["document_accuracy"] for row in author_rows])
                ),
                "chunk_macro_accuracy": float(np.mean(accuracies)),
            }
        )
        summary.extend(author_rows)
    return detail, summary


def drift_clustered_analysis(
    *,
    source_path: Path,
    n_boot: int,
    seed: int,
    confidence: float,
) -> list[dict[str, Any]]:
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["author_set"] == "all_authors"]
    grouped: dict[tuple[str, str, str, str], dict[int, dict[str, Any]]] = defaultdict(
        dict
    )
    for row in selected:
        key = (
            row["condition"],
            row["feature"],
            row["target_author"],
            row["run_id"],
        )
        grouped[key][int(row["window_index"])] = row

    metrics = {
        "target_kl": lambda row: float(row["target_kl"]),
        "target_margin": lambda row: float(row["target_margin"]),
        "target_rank": lambda row: float(row["target_rank"]),
        "target_hit": lambda row: float(row["target_hit"] == "True"),
    }
    changes: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (condition, feature, author, _run_id), windows in grouped.items():
        if 1 not in windows or 5 not in windows:
            continue
        for metric, getter in metrics.items():
            changes[(condition, feature, author)][metric].append(
                getter(windows[5]) - getter(windows[1])
            )

    out: list[dict[str, Any]] = []
    pvalue_indices: list[int] = []
    pvalues: list[float] = []
    for group_index, condition in enumerate(
        sorted({key[0] for key in changes})
    ):
        for feature in FEATURES:
            authors = sorted(
                key[2]
                for key in changes
                if key[0] == condition and key[1] == feature
            )
            for metric_index, metric in enumerate(metrics):
                author_values = {
                    author: float(
                        np.mean(changes[(condition, feature, author)][metric])
                    )
                    for author in authors
                }
                estimate = author_mean_estimate(
                    author_values,
                    n_boot=n_boot,
                    seed=seed + group_index * 20 + metric_index,
                    confidence=confidence,
                )
                values = list(author_values.values())
                alternative = "less" if metric in {"target_margin", "target_hit"} else "greater"
                try:
                    pvalue = float(
                        wilcoxon(values, alternative=alternative).pvalue
                    )
                except ValueError:
                    pvalue = 1.0
                out.append(
                    {
                        "row_type": "delta",
                        "condition": condition,
                        "feature": feature,
                        "metric": metric,
                        "first_window": 1,
                        "last_window": 5,
                        "alternative": alternative,
                        **estimate,
                        "wilcoxon_p": pvalue,
                    }
                )
                pvalue_indices.append(len(out) - 1)
                pvalues.append(pvalue)
    adjusted = benjamini_hochberg(pvalues)
    for index, adjusted_p in zip(pvalue_indices, adjusted):
        out[index]["benjamini_hochberg_p"] = adjusted_p

    position_values: dict[
        tuple[str, str, int, str], list[float]
    ] = defaultdict(list)
    for row in selected:
        position_values[
            (
                row["condition"],
                row["feature"],
                int(row["window_index"]),
                row["target_author"],
            )
        ].append(float(row["target_kl"]))
    position_groups = sorted(
        {
            (condition, feature, window)
            for condition, feature, window, _author in position_values
        }
    )
    for position_index, (condition, feature, window) in enumerate(position_groups):
        author_values = {
            author: float(
                np.mean(position_values[(condition, feature, window, author)])
            )
            for author in sorted(
                {
                    key[3]
                    for key in position_values
                    if key[:3] == (condition, feature, window)
                }
            )
        }
        estimate = author_mean_estimate(
            author_values,
            n_boot=n_boot,
            seed=seed + 500 + position_index,
            confidence=confidence,
        )
        out.append(
            {
                "row_type": "position",
                "condition": condition,
                "feature": feature,
                "metric": "target_kl",
                "window_index": window,
                **estimate,
            }
        )
    return out


def dash_sensitivity(
    *,
    conditions: dict[str, str],
    panel: dict[str, Any],
) -> list[dict[str, Any]]:
    new_authors = {
        author["key"] for author in panel["authors"] if author["cohort"] == "new"
    }
    rows: list[dict[str, Any]] = []
    standard_marks = set(PUNCT_VECTOR)
    for condition in conditions:
        root = resolve(f"generated_texts_campaign_author20_{condition}_new10")
        raw_files = sorted(root.glob("*/raw/run_*.txt"))
        totals = Counter()
        n_runs = 0
        for raw_path in raw_files:
            author = raw_path.parent.parent.name
            if author not in new_authors:
                continue
            processed_path = raw_path.parent.parent / raw_path.name
            if not processed_path.is_file():
                continue
            raw = raw_path.read_text(encoding="utf-8", errors="ignore")
            processed = processed_path.read_text(encoding="utf-8", errors="ignore")
            n_runs += 1
            totals["raw_commas"] += raw.count(",")
            totals["raw_em_dashes"] += raw.count("—")
            totals["raw_en_dashes"] += raw.count("–")
            totals["raw_standard_marks"] += sum(raw.count(mark) for mark in standard_marks)
            totals["processed_commas"] += processed.count(",")
            totals["processed_standard_marks"] += sum(
                processed.count(mark) for mark in standard_marks
            )
        if not n_runs:
            continue
        raw_dashes = totals["raw_em_dashes"] + totals["raw_en_dashes"]
        expected_added = totals["processed_commas"] - totals["raw_commas"]
        rows.append(
            {
                "condition": condition,
                "cohort": "new",
                "n_runs": n_runs,
                **dict(totals),
                "raw_dashes": raw_dashes,
                "comma_increase_after_policy": expected_added,
                "dash_to_comma_accounting_difference": expected_added - raw_dashes,
                "raw_comma_share_excluding_dashes": (
                    totals["raw_commas"] / totals["raw_standard_marks"]
                ),
                "processed_comma_share": (
                    totals["processed_commas"] / totals["processed_standard_marks"]
                ),
            }
        )
    return rows


def pooled_leakage_contrasts(grid: Path) -> list[dict[str, Any]]:
    """Copy frozen-grid pooled-profile rows with an explicit warning label."""
    out: list[dict[str, Any]] = []
    for filename, endpoint in (
        ("attribution.csv", "attribution"),
        ("detection.csv", "detection"),
    ):
        with (grid / filename).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("policy") != "pooled_all_books":
                    continue
                if endpoint == "attribution" and row.get("author") != "ALL":
                    continue
                out.append(
                    {
                        "interpretation": "descriptive_leakage_contrast_only",
                        "endpoint": endpoint,
                        **row,
                    }
                )
    return out


def reference_audit_rows() -> list[dict[str, Any]]:
    builds = Counter(
        (
            entry.author,
            entry.included_books,
            entry.excluded_books,
            entry.n_marks,
            entry.caller,
        )
        for entry in audit_log()
    )
    return [
        {
            "author": author,
            "included_books": ";".join(included),
            "excluded_books": ";".join(excluded),
            "n_marks": n_marks,
            "caller": caller,
            "build_calls": calls,
        }
        for (author, included, excluded, n_marks, caller), calls in sorted(
            builds.items()
        )
    ]


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    panel_path = resolve(config["authors_config"])
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    output_dir = resolve(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream = config["upstream_results"]
    context_drift_path = resolve(upstream["context_drift"])
    full_grid_dir = resolve(upstream["full_grid_dir"])

    inference = config["inference"]
    n_boot = int(
        args.bootstrap_iterations
        if args.bootstrap_iterations is not None
        else inference["bootstrap_iterations"]
    )
    seed = int(inference["seed"])
    confidence = float(inference["confidence_level"])

    corpus = load_corpus(config["authors_config"], config["cache"])
    conditions = dict(config["conditions"])
    for condition, directory in conditions.items():
        corpus.load_runs(condition, directory)

    cohort_by_author = {
        author["key"]: author["cohort"] for author in panel["authors"]
    }
    author_sets = {
        "all_authors": list(corpus.authors),
        "existing": [
            author for author in corpus.authors if cohort_by_author[author] == "existing"
        ],
        "new": [
            author for author in corpus.authors if cohort_by_author[author] == "new"
        ],
    }
    folds = stratified_group_folds(
        list(corpus.authors),
        cohort_by_author,
        n_folds=int(inference["detection_folds"]),
        seed=seed,
    )
    split_rows: list[dict[str, Any]] = []
    for fold in range(int(inference["detection_folds"])):
        calibration, test = split_for_fold(folds, fold)
        for author in corpus.authors:
            split_rows.append(
                {
                    "fold": fold,
                    "author": author,
                    "cohort": cohort_by_author[author],
                    "role": "test" if author in test else "calibration",
                }
            )

    reset_audit_log()
    print("[1/6] clustered attribution")
    attribution_observations: list[dict[str, Any]] = []
    feature_cache: dict[tuple[str, int], Any] = {}
    for author_set in config["sensitivity"]["author_sets"]:
        view = corpus.subset(author_sets[author_set])
        for feature in config["sensitivity"]["features"]:
            for chunk_size in config["sensitivity"]["chunk_sizes"]:
                attribution_observations.extend(
                    collect_attribution_observations(
                        view,
                        author_set=author_set,
                        conditions=conditions,
                        feature=feature,
                        chunk_size=int(chunk_size),
                        cohort_by_author=cohort_by_author,
                        feature_cache=feature_cache,
                    )
                )
    attribution_summary, prompt_summary = summarize_attribution(
        attribution_observations,
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
    )
    contrasts = model_contrast(
        attribution_observations,
        primary=config["primary"],
        n_boot=n_boot,
        seed=seed + 500,
        confidence=confidence,
    )

    print("[2/6] out-of-author detection calibration")
    detection_rows, detection_fold_rows, detection_observations = detection_analysis(
        corpus,
        conditions=conditions,
        config=config,
        folds=folds,
        n_boot=n_boot,
        confidence=confidence,
    )

    print("[3/6] cross-fitted separability")
    separability_detail, separability_summary = crossfit_separability(
        corpus,
        features_to_run=list(config["sensitivity"]["features"]),
        chunk_size=int(config["primary"]["chunk_size"]),
        n_boot=n_boot,
        seed=seed + 2000,
        confidence=confidence,
    )

    print("[4/6] clustered positional drift")
    drift_rows = drift_clustered_analysis(
        source_path=context_drift_path,
        n_boot=n_boot,
        seed=seed + 3000,
        confidence=confidence,
    )

    print("[5/6] dash-policy sensitivity")
    dash_rows = dash_sensitivity(conditions=conditions, panel=panel)
    leakage_rows = pooled_leakage_contrasts(full_grid_dir)

    print("[6/6] outputs and manifest")
    write_csv(output_dir / "split_assignments.csv", split_rows)
    write_csv(
        output_dir / "attribution_observations.csv", attribution_observations
    )
    write_csv(output_dir / "attribution_clustered.csv", attribution_summary)
    write_csv(output_dir / "prompt_cluster_attribution.csv", prompt_summary)
    write_csv(output_dir / "model_contrasts.csv", contrasts)
    write_csv(output_dir / "detection_crossfit.csv", detection_rows)
    write_csv(output_dir / "detection_folds.csv", detection_fold_rows)
    write_csv(
        output_dir / "detection_observations.csv", detection_observations
    )
    write_csv(output_dir / "separability_crossfit.csv", separability_detail)
    write_csv(
        output_dir / "separability_crossfit_summary.csv", separability_summary
    )
    write_csv(output_dir / "drift_clustered.csv", drift_rows)
    write_csv(output_dir / "dash_sensitivity.csv", dash_rows)
    write_csv(output_dir / "pooled_leakage_contrasts.csv", leakage_rows)
    write_csv(output_dir / "reference_audit.csv", reference_audit_rows())

    primary = config["primary"]
    primary_attribution = [
        row
        for row in attribution_summary
        if row["row_type"] == "summary"
        and row["author_set"] == primary["author_set"]
        and row["feature"] == primary["feature"]
        and row["chunk_size"] == primary["chunk_size"]
    ]
    primary_detection = [
        row
        for row in detection_rows
        if row["method"] == "out_of_author_crossfit"
        and row["feature"] == primary["feature"]
        and row["chunk_size"] == primary["chunk_size"]
        and float(row["smoothing_eps"]) == 0.5
    ]
    primary_separability = [
        row
        for row in separability_summary
        if row["row_type"] == "summary"
        and row["feature"] == primary["feature"]
    ]
    headline = {
        "primary": primary,
        "attribution": primary_attribution,
        "model_contrast": contrasts,
        "detection": primary_detection,
        "separability": primary_separability,
        "drift": [
            row
            for row in drift_rows
            if row["row_type"] == "delta"
            and row["feature"] == primary["feature"]
            and row["metric"] in {"target_kl", "target_hit"}
        ],
        "dash_sensitivity": dash_rows,
    }
    write_json(output_dir / "headline_summary.json", headline)

    source_paths = [ROOT / "run_inference_v2.py", *sorted((ROOT / "punctlib").glob("*.py"))]
    upstream_paths = (
        context_drift_path,
        full_grid_dir / "attribution.csv",
        full_grid_dir / "detection.csv",
    )
    try:
        output_prefixes = (
            output_dir.relative_to(ROOT).as_posix(),
            "results/author_panel_20",
            "paper",
            "attic/paper-legacy",
        )
    except ValueError:
        output_prefixes = (
            "results/author_panel_20",
            "paper",
            "attic/paper-legacy",
        )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_config": str(config_path.relative_to(ROOT)),
        "analysis_config_sha256": sha256(config_path),
        "authors_config": config["authors_config"],
        "authors_config_sha256": sha256(panel_path),
        "cache": config["cache"],
        "cache_sha256": corpus.cache_sha256,
        "conditions": conditions,
        "condition_summary_sha256": {
            condition: sha256(resolve(directory) / "all_runs_summary.json")
            for condition, directory in conditions.items()
        },
        "condition_manifest_sha256": {
            condition: sha256(resolve(directory) / "condition_manifest.json")
            for condition, directory in conditions.items()
        },
        "condition_text_tree_sha256": {
            condition: sha256_tree(resolve(directory))
            for condition, directory in conditions.items()
        },
        "upstream_result_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in upstream_paths
        },
        "source_code_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
        "git": git_metadata(excluded_prefixes=output_prefixes),
        "environment": {
            "python": platform.python_version(),
            "packages": package_versions(),
        },
        "bootstrap_iterations": n_boot,
        "confidence_level": confidence,
        "fold_by_author": folds,
        "cluster_units": {
            "primary": "author",
            "secondary_llm": "prompt_source_book",
        },
        "output_files": [*ANALYTICAL_OUTPUT_FILES, "inference_manifest.json"],
        "output_sha256": {
            filename: sha256(output_dir / filename)
            for filename in ANALYTICAL_OUTPUT_FILES
        },
    }
    write_json(output_dir / "inference_manifest.json", manifest)

    print("\nPrimary clustered estimates")
    for row in primary_attribution:
        label = row["condition"] or "human"
        print(
            f"  attribution {label:6s}: macro={100*row['macro_accuracy']:.1f}% "
            f"[{100*row['macro_ci_low']:.1f}, {100*row['macro_ci_high']:.1f}]"
        )
    for row in primary_detection:
        print(
            f"  detection {row['condition']:6s}: AUC={row['auc']:.3f}, "
            f"FPR={100*row['empirical_fpr']:.1f}%, TPR={100*row['tpr']:.1f}%"
        )


if __name__ == "__main__":
    main()
