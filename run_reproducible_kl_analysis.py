#!/usr/bin/env python3
"""
Reproducible KL analysis for LLM-vs-author punctuation experiments.

This script rebuilds key "future direction" KL tables directly from:
  - generated LLM runs (generated_texts* directories)
  - reference full books (full_books/*_full.txt)

Outputs:
  1) run_level_kl_long.csv
  2) all_author_pairwise_self_vs_other.csv
  3) all_author_pairwise_self_vs_other_summary.csv
  4) austen_vs_wells_validation.csv
  5) austen_nearest_of_all_validation.csv
  6) cross_author_kl_means.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu


DEFAULT_AUTHORS = {
    "jane_austen": "Jane Austen",
    "william_shakespeare": "William Shakespeare",
    "herbert_george_wells": "H.G. Wells",
    "agnes_may_fleming": "Agnes May Fleming",
}

DEFAULT_CONDITION_DIRS = {
    "v1_flash": "generated_texts",
    "v1_pro": "generated_texts_v1_gemini_pro",
    "v2_flash": "generated_texts_v2",
    "v3_flash": "generated_texts_v3",
}

TARGET_FEATURES = ("f1", "f3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate reproducible KL analysis tables."
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITION_DIRS.keys()),
        help="Subset of conditions to run.",
    )
    parser.add_argument(
        "--authors",
        nargs="+",
        default=list(DEFAULT_AUTHORS.keys()),
        help="Subset of authors to include.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Punctuation chunk size for comparisons.",
    )
    parser.add_argument(
        "--full-books-dir",
        default="full_books",
        help="Directory containing *_full.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        default="reproducible_kl_results",
        help="Directory where CSV outputs are written.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=5000,
        help="Bootstrap iterations for confidence intervals.",
    )
    parser.add_argument(
        "--permutation-iters",
        type=int,
        default=100000,
        help="Monte Carlo iterations for paired permutation test when exact is too large.",
    )
    parser.add_argument(
        "--no-significance",
        action="store_true",
        help="Skip significance outputs (Step 2).",
    )
    return parser.parse_args()


ARGS = parse_args()

ROOT = Path(__file__).resolve().parent
FULL_BOOKS_DIR = ROOT / ARGS.full_books_dir
OUTPUT_DIR = ROOT / ARGS.output_dir

# ---------------------------------------------------------------------------
# punctuation-stylometry imports (config-aware)
# ---------------------------------------------------------------------------
_PUNCT_STYLOMETRY_DIR = ROOT / "punctuation-stylometry-master"
sys.path.insert(0, str(_PUNCT_STYLOMETRY_DIR))

_CONFIG_PATH = str(_PUNCT_STYLOMETRY_DIR / "conf" / "punctuation.ini")
sys.argv = [sys.argv[0], "-c", _CONFIG_PATH]

from punctuation.config import options  # noqa: E402
from punctuation.feature_operations.distances import d_KL  # noqa: E402
from punctuation.feature_operations.matrix_operations import (  # noqa: E402
    normalised_transition_mat,
    transition_mat,
)
from punctuation.parser.punctuation_parser import (  # noqa: E402
    get_frequencies,
    get_textinfo,
    seq_pun_only,
)


PUNCTUATION_VECTOR = options.punctuation_vector


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _extract_punctuation(text: str) -> list[str] | None:
    text = text.replace("...", "^")
    return seq_pun_only(get_textinfo(text))


def _compute_features(punctuation_seq: list[str]) -> dict[str, list[float]] | None:
    f1 = get_frequencies(punctuation_seq, vector=PUNCTUATION_VECTOR)
    if f1 is None or sum(f1) == 0:
        return None

    f2 = transition_mat(punctuation_seq)
    if f2 is None:
        return None

    f3 = normalised_transition_mat(f2, f1).flatten().tolist()
    return {"f1": f1, "f3": f3}


def _chunk_sequence(
    punctuation_seq: list[str], chunk_size: int, position: str
) -> list[str] | None:
    if punctuation_seq is None or len(punctuation_seq) < chunk_size:
        return None

    if position == "middle":
        start = (len(punctuation_seq) - chunk_size) // 2
    elif position == "start":
        start = 0
    else:
        raise ValueError(f"Unsupported chunk position: {position}")

    return punctuation_seq[start : start + chunk_size]


def _author_full_book_path(author_key: str) -> Path:
    return FULL_BOOKS_DIR / f"{author_key}_full.txt"


def _load_real_author_features(
    author_keys: list[str], chunk_size: int
) -> dict[str, dict[str, list[float]]]:
    features_by_author: dict[str, dict[str, list[float]]] = {}

    for author_key in author_keys:
        path = _author_full_book_path(author_key)
        if not path.exists():
            print(f"[warn] Missing full book for '{author_key}': {path}")
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        punctuation_seq = _extract_punctuation(text)
        chunk = _chunk_sequence(punctuation_seq, chunk_size, position="middle")
        if chunk is None:
            print(
                f"[warn] Full book too short for chunk size {chunk_size}: "
                f"{author_key}"
            )
            continue

        feats = _compute_features(chunk)
        if feats is None:
            print(f"[warn] Could not compute real features for '{author_key}'")
            continue

        features_by_author[author_key] = feats

    return features_by_author


def _parse_run_id(run_file: Path) -> str:
    # run_01.txt -> 01
    stem = run_file.stem
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


def _load_llm_runs(
    condition_dir: Path, author_key: str, chunk_size: int
) -> list[dict[str, object]]:
    run_dir = condition_dir / author_key
    if not run_dir.exists():
        return []

    run_rows: list[dict[str, object]] = []
    for run_file in sorted(run_dir.glob("run_*.txt")):
        text = run_file.read_text(encoding="utf-8", errors="ignore")
        punctuation_seq = _extract_punctuation(text)
        chunk = _chunk_sequence(punctuation_seq, chunk_size, position="start")
        if chunk is None:
            continue

        feats = _compute_features(chunk)
        if feats is None:
            continue

        run_rows.append(
            {
                "run_id": _parse_run_id(run_file),
                "f1": feats["f1"],
                "f3": feats["f3"],
            }
        )

    return run_rows


def _condition_paths(selected_conditions: list[str]) -> dict[str, Path]:
    paths = {}
    for condition in selected_conditions:
        rel = DEFAULT_CONDITION_DIRS.get(condition)
        if rel is None:
            raise ValueError(f"Unknown condition '{condition}'")
        paths[condition] = ROOT / rel
    return paths


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_run_level_kl_rows(
    condition_paths: dict[str, Path],
    author_keys: list[str],
    real_features: dict[str, dict[str, list[float]]],
    chunk_size: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for condition, condition_dir in condition_paths.items():
        if not condition_dir.exists():
            print(f"[warn] Missing condition directory: {condition_dir}")
            continue

        print(f"[run] {condition} -> {condition_dir}")
        for llm_author in author_keys:
            if llm_author not in real_features:
                continue

            runs = _load_llm_runs(condition_dir, llm_author, chunk_size)
            if not runs:
                continue

            for run in runs:
                for target_author in author_keys:
                    target_feats = real_features.get(target_author)
                    if target_feats is None:
                        continue

                    for feature in TARGET_FEATURES:
                        kl_value = d_KL(run[feature], target_feats[feature])
                        rows.append(
                            {
                                "condition": condition,
                                "feature": feature,
                                "llm_copying": llm_author,
                                "target_author": target_author,
                                "run_id": run["run_id"],
                                "kl_value": float(kl_value),
                                "comparison_type": (
                                    "self"
                                    if target_author == llm_author
                                    else "other_author"
                                ),
                            }
                        )
    return rows


def _index_kl_rows(
    run_level_rows: list[dict[str, object]],
) -> dict[tuple[str, str, str, str], dict[str, float]]:
    """
    Returns mapping:
      (condition, feature, llm_copying, target_author) -> {run_id: kl_value}
    """
    indexed: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in run_level_rows:
        key = (
            str(row["condition"]),
            str(row["feature"]),
            str(row["llm_copying"]),
            str(row["target_author"]),
        )
        indexed[key][str(row["run_id"])] = float(row["kl_value"])
    return indexed


def _build_pairwise_tables(
    run_level_rows: list[dict[str, object]],
    author_keys: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    indexed = _index_kl_rows(run_level_rows)
    pairwise_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    grouped_keys = sorted(
        {(r["condition"], r["feature"], r["llm_copying"]) for r in run_level_rows}
    )

    for condition, feature, llm_copying in grouped_keys:
        own_key = (condition, feature, llm_copying, llm_copying)
        own_by_run = indexed.get(own_key, {})
        if not own_by_run:
            continue

        own_mean = _safe_mean(list(own_by_run.values()))
        per_target_rows = []

        for target_author in author_keys:
            if target_author == llm_copying:
                continue

            target_key = (condition, feature, llm_copying, target_author)
            target_by_run = indexed.get(target_key, {})
            if not target_by_run:
                continue

            run_ids = sorted(set(own_by_run).intersection(target_by_run))
            if not run_ids:
                continue

            own_vals = [own_by_run[rid] for rid in run_ids]
            target_vals = [target_by_run[rid] for rid in run_ids]

            own_mean_kl = _safe_mean(own_vals)
            target_mean_kl = _safe_mean(target_vals)
            own_smaller_mean = (
                own_mean_kl < target_mean_kl
                if own_mean_kl is not None and target_mean_kl is not None
                else False
            )
            own_smaller_runs = sum(
                1 for own_v, target_v in zip(own_vals, target_vals) if own_v < target_v
            )
            n_runs = len(run_ids)
            own_smaller_runs_pct = own_smaller_runs / n_runs if n_runs else 0.0

            row = {
                "condition": condition,
                "feature": feature,
                "llm_copying": llm_copying,
                "compare_target": target_author,
                "n_runs": n_runs,
                "own_mean_kl": own_mean_kl,
                "target_mean_kl": target_mean_kl,
                "own_smaller_mean": own_smaller_mean,
                "own_smaller_runs": own_smaller_runs,
                "own_smaller_runs_pct": own_smaller_runs_pct,
            }
            pairwise_rows.append(row)
            per_target_rows.append(row)

        if not per_target_rows:
            continue

        n_pairs = len(per_target_rows)
        pairwise_mean_win_count = sum(1 for r in per_target_rows if r["own_smaller_mean"])
        pairwise_allruns_win_count = sum(
            1 for r in per_target_rows if r["own_smaller_runs"] == r["n_runs"]
        )
        n_runs_for_summary = max(r["n_runs"] for r in per_target_rows)

        summary_rows.append(
            {
                "condition": condition,
                "feature": feature,
                "llm_copying": llm_copying,
                "pairwise_mean_win_count": pairwise_mean_win_count,
                "pairwise_mean_win_pct": pairwise_mean_win_count / n_pairs,
                "pairwise_allruns_win_count": pairwise_allruns_win_count,
                "pairwise_allruns_win_pct": pairwise_allruns_win_count / n_pairs,
                "n_pairs": n_pairs,
                "n_runs": n_runs_for_summary,
                "own_mean_kl": own_mean,
            }
        )

    return pairwise_rows, summary_rows


def _build_austen_vs_wells_table(
    run_level_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    indexed = _index_kl_rows(run_level_rows)
    rows: list[dict[str, object]] = []

    grouped_keys = sorted(
        {(r["condition"], r["feature"], r["llm_copying"]) for r in run_level_rows}
    )

    for condition, feature, llm_copying in grouped_keys:
        if llm_copying != "jane_austen":
            continue

        austen_key = (condition, feature, llm_copying, "jane_austen")
        wells_key = (condition, feature, llm_copying, "herbert_george_wells")
        austen_by_run = indexed.get(austen_key, {})
        wells_by_run = indexed.get(wells_key, {})
        run_ids = sorted(set(austen_by_run).intersection(wells_by_run))
        if not run_ids:
            continue

        austen_vals = [austen_by_run[rid] for rid in run_ids]
        wells_vals = [wells_by_run[rid] for rid in run_ids]
        n_runs = len(run_ids)
        n_runs_austen_smaller = sum(
            1 for austen_v, wells_v in zip(austen_vals, wells_vals) if austen_v < wells_v
        )
        mean_austen = _safe_mean(austen_vals)
        mean_wells = _safe_mean(wells_vals)

        rows.append(
            {
                "condition": condition,
                "feature": feature,
                "n_runs": n_runs,
                "mean_kl_llmAusten_to_Austen": mean_austen,
                "mean_kl_llmAusten_to_Wells": mean_wells,
                "austen_smaller_mean": (
                    mean_austen < mean_wells
                    if mean_austen is not None and mean_wells is not None
                    else False
                ),
                "n_runs_austen_smaller": n_runs_austen_smaller,
                "pct_runs_austen_smaller": (
                    n_runs_austen_smaller / n_runs if n_runs else 0.0
                ),
            }
        )

    return rows


def _build_austen_nearest_table(
    run_level_rows: list[dict[str, object]], author_keys: list[str]
) -> list[dict[str, object]]:
    indexed = _index_kl_rows(run_level_rows)
    rows: list[dict[str, object]] = []

    grouped_keys = sorted(
        {(r["condition"], r["feature"], r["llm_copying"]) for r in run_level_rows}
    )

    for condition, feature, llm_copying in grouped_keys:
        if llm_copying != "jane_austen":
            continue

        by_target = {
            target: indexed.get((condition, feature, llm_copying, target), {})
            for target in author_keys
        }

        # Keep only targets with data.
        by_target = {k: v for k, v in by_target.items() if v}
        if not by_target:
            continue

        mean_by_target = {
            target: _safe_mean(list(run_map.values()))
            for target, run_map in by_target.items()
        }
        mean_by_target = {
            target: value for target, value in mean_by_target.items() if value is not None
        }
        if not mean_by_target:
            continue

        mean_best_target = min(mean_by_target, key=mean_by_target.get)

        run_ids = sorted(set.intersection(*(set(v.keys()) for v in by_target.values())))
        run_level_austen_best_count = 0
        for rid in run_ids:
            vals = {target: run_map[rid] for target, run_map in by_target.items()}
            best_target = min(vals, key=lambda t: (vals[t], t))
            if best_target == "jane_austen":
                run_level_austen_best_count += 1

        row = {
            "condition": condition,
            "feature": feature,
            "mean_best_target": mean_best_target,
            "mean_austen_is_best": mean_best_target == "jane_austen",
            "run_level_austen_best_count": run_level_austen_best_count,
            "run_level_austen_best_pct": (
                run_level_austen_best_count / len(run_ids) if run_ids else 0.0
            ),
            "mean_to_austen": mean_by_target.get("jane_austen"),
            "mean_to_shakespeare": mean_by_target.get("william_shakespeare"),
            "mean_to_wells": mean_by_target.get("herbert_george_wells"),
            "mean_to_fleming": mean_by_target.get("agnes_may_fleming"),
        }
        rows.append(row)

    return rows


def _build_cross_author_means_table(
    run_level_rows: list[dict[str, object]], author_keys: list[str]
) -> list[dict[str, object]]:
    indexed = _index_kl_rows(run_level_rows)
    rows: list[dict[str, object]] = []

    grouped_keys = sorted(
        {(r["condition"], r["feature"], r["llm_copying"]) for r in run_level_rows}
    )

    for condition, feature, llm_copying in grouped_keys:
        means = {}
        for target in author_keys:
            vals = indexed.get((condition, feature, llm_copying, target), {}).values()
            if vals:
                means[target] = _safe_mean(list(vals))
        means = {k: v for k, v in means.items() if v is not None}
        if not means:
            continue

        best_target = min(means, key=means.get)
        row = {
            "condition": condition,
            "llm_copying": llm_copying,
            "feature": feature,
            "own_author_kl": means.get(llm_copying),
            "best_target": best_target,
            "best_target_kl": means[best_target],
            "own_is_smallest": llm_copying == best_target,
        }
        for target in author_keys:
            row[f"vs_{target}"] = means.get(target)
        rows.append(row)

    return rows


def _format_float_columns(rows: list[dict[str, object]], columns: list[str]) -> None:
    for row in rows:
        for col in columns:
            if col in row:
                row[col] = _fmt_float(row[col]) if row[col] is not None else ""


def _paired_perm_pvalue_one_sided(
    diffs: np.ndarray, monte_carlo_iters: int, rng: np.random.Generator
) -> tuple[float, str]:
    """
    One-sided paired permutation (sign-flip) p-value.
    H1: mean(diffs) > 0, where diffs = target_kl - own_kl.
    """
    n = len(diffs)
    observed = float(np.mean(diffs))
    if n == 0:
        return float("nan"), "none"

    # Exact test is feasible for small n and avoids MC noise.
    if n <= 20:
        ge_count = 0
        total = 0
        for signs in product((-1.0, 1.0), repeat=n):
            stat = float(np.mean(diffs * np.asarray(signs)))
            if stat >= observed:
                ge_count += 1
            total += 1
        pvalue = (ge_count + 1.0) / (total + 1.0)
        return pvalue, "exact"

    signs = rng.choice([-1.0, 1.0], size=(monte_carlo_iters, n))
    stats = (signs * diffs[None, :]).mean(axis=1)
    ge_count = int(np.sum(stats >= observed))
    pvalue = (ge_count + 1.0) / (monte_carlo_iters + 1.0)
    return pvalue, "monte_carlo"


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta = P(X>Y) - P(X<Y).
    """
    gt = 0
    lt = 0
    for xv in x:
        gt += int(np.sum(xv > y))
        lt += int(np.sum(xv < y))
    denom = len(x) * len(y)
    if denom == 0:
        return float("nan")
    return float((gt - lt) / denom)


def _paired_bootstrap_ci(
    diffs: np.ndarray, iters: int, rng: np.random.Generator
) -> tuple[float, float, float, float]:
    """
    Returns:
      mean_ci_low, mean_ci_high, median_ci_low, median_ci_high
    """
    n = len(diffs)
    if n == 0:
        nan = float("nan")
        return nan, nan, nan, nan

    samples_idx = rng.integers(0, n, size=(iters, n))
    sampled = diffs[samples_idx]
    mean_stats = sampled.mean(axis=1)
    median_stats = np.median(sampled, axis=1)

    mean_low, mean_high = np.percentile(mean_stats, [2.5, 97.5])
    med_low, med_high = np.percentile(median_stats, [2.5, 97.5])
    return float(mean_low), float(mean_high), float(med_low), float(med_high)


def _bh_fdr(pvals: list[float | None]) -> list[float | None]:
    """
    Benjamini-Hochberg FDR correction.
    """
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None and np.isfinite(p)]
    m = len(indexed)
    qvals = [None] * len(pvals)
    if m == 0:
        return qvals

    indexed.sort(key=lambda x: x[1])
    adjusted = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx, p = indexed[rank - 1]
        q = min(prev, (p * m) / rank)
        adjusted[rank - 1] = q
        prev = q

    for (ranked_pair, q) in zip(indexed, adjusted):
        idx, _ = ranked_pair
        qvals[idx] = float(q)
    return qvals


def _build_significance_tables(
    run_level_rows: list[dict[str, object]],
    author_keys: list[str],
    bootstrap_iters: int,
    permutation_iters: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """
    Returns:
      - significance_summary_rows
      - significance_delta_rows (run-level paired deltas)
    """
    indexed = _index_kl_rows(run_level_rows)
    grouped_keys = sorted(
        {(r["condition"], r["feature"], r["llm_copying"]) for r in run_level_rows}
    )

    summary_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(42)

    for condition, feature, llm_copying in grouped_keys:
        own_key = (condition, feature, llm_copying, llm_copying)
        own_by_run = indexed.get(own_key, {})
        if not own_by_run:
            continue

        for compare_target in author_keys:
            if compare_target == llm_copying:
                continue

            target_key = (condition, feature, llm_copying, compare_target)
            target_by_run = indexed.get(target_key, {})
            if not target_by_run:
                continue

            run_ids = sorted(set(own_by_run).intersection(target_by_run))
            if not run_ids:
                continue

            own_vals = np.asarray([own_by_run[rid] for rid in run_ids], dtype=float)
            target_vals = np.asarray([target_by_run[rid] for rid in run_ids], dtype=float)
            diffs = target_vals - own_vals

            for rid, own_v, target_v, delta in zip(run_ids, own_vals, target_vals, diffs):
                delta_rows.append(
                    {
                        "condition": condition,
                        "feature": feature,
                        "llm_copying": llm_copying,
                        "compare_target": compare_target,
                        "run_id": rid,
                        "own_kl": float(own_v),
                        "target_kl": float(target_v),
                        "delta_target_minus_own": float(delta),
                        "own_smaller_run": bool(own_v < target_v),
                    }
                )

            perm_p, perm_mode = _paired_perm_pvalue_one_sided(
                diffs=diffs,
                monte_carlo_iters=permutation_iters,
                rng=rng,
            )
            mw_p = float(
                mannwhitneyu(own_vals, target_vals, alternative="less").pvalue
            )
            ks_p = float(ks_2samp(own_vals, target_vals).pvalue)

            mean_delta = float(np.mean(diffs))
            median_delta = float(np.median(diffs))
            cliffs = _cliffs_delta(target_vals, own_vals)
            ci_mean_low, ci_mean_high, ci_median_low, ci_median_high = _paired_bootstrap_ci(
                diffs=diffs,
                iters=bootstrap_iters,
                rng=rng,
            )

            summary_rows.append(
                {
                    "condition": condition,
                    "feature": feature,
                    "llm_copying": llm_copying,
                    "compare_target": compare_target,
                    "n_runs": len(run_ids),
                    "mean_own_kl": float(np.mean(own_vals)),
                    "mean_target_kl": float(np.mean(target_vals)),
                    "mean_delta_target_minus_own": mean_delta,
                    "median_delta_target_minus_own": median_delta,
                    "cliffs_delta_target_vs_own": cliffs,
                    "bootstrap_mean_delta_ci_low": ci_mean_low,
                    "bootstrap_mean_delta_ci_high": ci_mean_high,
                    "bootstrap_median_delta_ci_low": ci_median_low,
                    "bootstrap_median_delta_ci_high": ci_median_high,
                    "perm_p_one_sided": perm_p,
                    "perm_test_mode": perm_mode,
                    "mannwhitney_p_one_sided": mw_p,
                    "ks_p_two_sided": ks_p,
                }
            )

    # Multiple-testing correction (BH) across all pair tests.
    perm_qvals = _bh_fdr([r.get("perm_p_one_sided") for r in summary_rows])
    mw_qvals = _bh_fdr([r.get("mannwhitney_p_one_sided") for r in summary_rows])

    for i, row in enumerate(summary_rows):
        row["perm_p_fdr_bh"] = perm_qvals[i]
        row["mannwhitney_p_fdr_bh"] = mw_qvals[i]
        row["perm_significant_fdr_0_05"] = (
            perm_qvals[i] is not None and perm_qvals[i] <= 0.05
        )
        row["mannwhitney_significant_fdr_0_05"] = (
            mw_qvals[i] is not None and mw_qvals[i] <= 0.05
        )

    return summary_rows, delta_rows


def main() -> None:
    author_keys = [a for a in ARGS.authors if a in DEFAULT_AUTHORS]
    if not author_keys:
        raise ValueError("No valid authors selected.")

    for author in ARGS.authors:
        if author not in DEFAULT_AUTHORS:
            print(f"[warn] Unknown author '{author}' (ignored).")

    condition_paths = _condition_paths(ARGS.conditions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[info] chunk_size={ARGS.chunk_size}")
    print(f"[info] authors={author_keys}")
    print(f"[info] conditions={list(condition_paths.keys())}")
    print(f"[info] output_dir={OUTPUT_DIR}")

    real_features = _load_real_author_features(author_keys, ARGS.chunk_size)
    missing_real = sorted(set(author_keys) - set(real_features.keys()))
    if missing_real:
        print(f"[warn] Missing real features for: {missing_real}")

    run_level_rows = _build_run_level_kl_rows(
        condition_paths=condition_paths,
        author_keys=author_keys,
        real_features=real_features,
        chunk_size=ARGS.chunk_size,
    )

    if not run_level_rows:
        raise RuntimeError("No run-level KL rows generated.")

    # 1) Pairwise self-vs-other tables.
    pairwise_rows, pairwise_summary_rows = _build_pairwise_tables(
        run_level_rows=run_level_rows,
        author_keys=author_keys,
    )
    _format_float_columns(
        pairwise_rows,
        ["own_mean_kl", "target_mean_kl", "own_smaller_runs_pct"],
    )
    pairwise_csv = OUTPUT_DIR / "all_author_pairwise_self_vs_other.csv"
    _write_csv(
        path=pairwise_csv,
        fieldnames=[
            "condition",
            "feature",
            "llm_copying",
            "compare_target",
            "n_runs",
            "own_mean_kl",
            "target_mean_kl",
            "own_smaller_mean",
            "own_smaller_runs",
            "own_smaller_runs_pct",
        ],
        rows=pairwise_rows,
    )

    _format_float_columns(
        pairwise_summary_rows,
        [
            "pairwise_mean_win_pct",
            "pairwise_allruns_win_pct",
            "own_mean_kl",
        ],
    )
    pairwise_summary_csv = OUTPUT_DIR / "all_author_pairwise_self_vs_other_summary.csv"
    _write_csv(
        path=pairwise_summary_csv,
        fieldnames=[
            "condition",
            "feature",
            "llm_copying",
            "pairwise_mean_win_count",
            "pairwise_mean_win_pct",
            "pairwise_allruns_win_count",
            "pairwise_allruns_win_pct",
            "n_pairs",
            "n_runs",
            "own_mean_kl",
        ],
        rows=pairwise_summary_rows,
    )

    # 2) Austen-specific validation tables.
    austen_vs_wells_rows = _build_austen_vs_wells_table(run_level_rows)
    _format_float_columns(
        austen_vs_wells_rows,
        [
            "mean_kl_llmAusten_to_Austen",
            "mean_kl_llmAusten_to_Wells",
            "pct_runs_austen_smaller",
        ],
    )
    austen_vs_wells_csv = OUTPUT_DIR / "austen_vs_wells_validation.csv"
    _write_csv(
        path=austen_vs_wells_csv,
        fieldnames=[
            "condition",
            "feature",
            "n_runs",
            "mean_kl_llmAusten_to_Austen",
            "mean_kl_llmAusten_to_Wells",
            "austen_smaller_mean",
            "n_runs_austen_smaller",
            "pct_runs_austen_smaller",
        ],
        rows=austen_vs_wells_rows,
    )

    austen_nearest_rows = _build_austen_nearest_table(run_level_rows, author_keys)
    _format_float_columns(
        austen_nearest_rows,
        [
            "run_level_austen_best_pct",
            "mean_to_austen",
            "mean_to_shakespeare",
            "mean_to_wells",
            "mean_to_fleming",
        ],
    )
    austen_nearest_csv = OUTPUT_DIR / "austen_nearest_of_all_validation.csv"
    _write_csv(
        path=austen_nearest_csv,
        fieldnames=[
            "condition",
            "feature",
            "mean_best_target",
            "mean_austen_is_best",
            "run_level_austen_best_count",
            "run_level_austen_best_pct",
            "mean_to_austen",
            "mean_to_shakespeare",
            "mean_to_wells",
            "mean_to_fleming",
        ],
        rows=austen_nearest_rows,
    )

    # 3) Cross-author mean table (all conditions/features).
    cross_author_rows = _build_cross_author_means_table(run_level_rows, author_keys)
    _format_float_columns(
        cross_author_rows,
        [
            "own_author_kl",
            "best_target_kl",
            "vs_jane_austen",
            "vs_william_shakespeare",
            "vs_herbert_george_wells",
            "vs_agnes_may_fleming",
        ],
    )
    cross_author_csv = OUTPUT_DIR / "cross_author_kl_means.csv"
    _write_csv(
        path=cross_author_csv,
        fieldnames=[
            "condition",
            "llm_copying",
            "feature",
            "own_author_kl",
            "best_target",
            "best_target_kl",
            "own_is_smallest",
            "vs_jane_austen",
            "vs_william_shakespeare",
            "vs_herbert_george_wells",
            "vs_agnes_may_fleming",
        ],
        rows=cross_author_rows,
    )

    # 4) Long-form run-level KL output.
    run_level_rows_out = [dict(r) for r in run_level_rows]
    _format_float_columns(run_level_rows_out, ["kl_value"])
    run_level_csv = OUTPUT_DIR / "run_level_kl_long.csv"
    _write_csv(
        path=run_level_csv,
        fieldnames=[
            "condition",
            "feature",
            "llm_copying",
            "target_author",
            "run_id",
            "kl_value",
            "comparison_type",
        ],
        rows=run_level_rows_out,
    )

    significance_csv = None
    significance_deltas_csv = None
    if not ARGS.no_significance:
        significance_rows, significance_delta_rows = _build_significance_tables(
            run_level_rows=run_level_rows,
            author_keys=author_keys,
            bootstrap_iters=ARGS.bootstrap_iters,
            permutation_iters=ARGS.permutation_iters,
        )

        _format_float_columns(
            significance_rows,
            [
                "mean_own_kl",
                "mean_target_kl",
                "mean_delta_target_minus_own",
                "median_delta_target_minus_own",
                "cliffs_delta_target_vs_own",
                "bootstrap_mean_delta_ci_low",
                "bootstrap_mean_delta_ci_high",
                "bootstrap_median_delta_ci_low",
                "bootstrap_median_delta_ci_high",
                "perm_p_one_sided",
                "mannwhitney_p_one_sided",
                "ks_p_two_sided",
                "perm_p_fdr_bh",
                "mannwhitney_p_fdr_bh",
            ],
        )
        significance_csv = OUTPUT_DIR / "significance_summary.csv"
        _write_csv(
            path=significance_csv,
            fieldnames=[
                "condition",
                "feature",
                "llm_copying",
                "compare_target",
                "n_runs",
                "mean_own_kl",
                "mean_target_kl",
                "mean_delta_target_minus_own",
                "median_delta_target_minus_own",
                "cliffs_delta_target_vs_own",
                "bootstrap_mean_delta_ci_low",
                "bootstrap_mean_delta_ci_high",
                "bootstrap_median_delta_ci_low",
                "bootstrap_median_delta_ci_high",
                "perm_p_one_sided",
                "perm_test_mode",
                "mannwhitney_p_one_sided",
                "ks_p_two_sided",
                "perm_p_fdr_bh",
                "mannwhitney_p_fdr_bh",
                "perm_significant_fdr_0_05",
                "mannwhitney_significant_fdr_0_05",
            ],
            rows=significance_rows,
        )

        _format_float_columns(
            significance_delta_rows,
            ["own_kl", "target_kl", "delta_target_minus_own"],
        )
        significance_deltas_csv = OUTPUT_DIR / "significance_run_deltas.csv"
        _write_csv(
            path=significance_deltas_csv,
            fieldnames=[
                "condition",
                "feature",
                "llm_copying",
                "compare_target",
                "run_id",
                "own_kl",
                "target_kl",
                "delta_target_minus_own",
                "own_smaller_run",
            ],
            rows=significance_delta_rows,
        )

    print("\nWrote:")
    print(f"  {run_level_csv}")
    print(f"  {pairwise_csv}")
    print(f"  {pairwise_summary_csv}")
    print(f"  {austen_vs_wells_csv}")
    print(f"  {austen_nearest_csv}")
    print(f"  {cross_author_csv}")
    if significance_csv is not None:
        print(f"  {significance_csv}")
    if significance_deltas_csv is not None:
        print(f"  {significance_deltas_csv}")


if __name__ == "__main__":
    main()
