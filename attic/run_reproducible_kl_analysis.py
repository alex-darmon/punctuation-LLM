#!/usr/bin/env python3
"""
Reproducible KL analysis for LLM-vs-author punctuation experiments.

This script rebuilds key "future direction" KL tables directly from:
  - generated LLM runs (generated_texts* directories)
  - reference full books (full_books/*_full.txt)

Outputs (per chunk size):
  1) run_level_kl_long.csv
  2) all_author_pairwise_self_vs_other.csv
  3) all_author_pairwise_self_vs_other_summary.csv
  4) austen_vs_wells_validation.csv
  5) austen_nearest_of_all_validation.csv
  6) cross_author_kl_means.csv
  7) significance_summary.csv
  8) significance_run_deltas.csv
  9) llm_within_author_kl_long.csv
 10) llm_within_author_kl_summary.csv
 11) llm_vs_human_within_variance_summary.csv

Step 3 aggregate outputs:
 12) chunk_size_significance_summary.csv
 13) chunk_size_within_author_baseline.csv
 14) chunk_size_llm_within_author_baseline.csv
 15) chunk_size_llm_vs_human_variance_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu


DEFAULT_AUTHORS = {
    "jane_austen": {"name": "Jane Austen", "form": "prose"},
    "william_shakespeare": {"name": "William Shakespeare", "form": "play"},
    "herbert_george_wells": {"name": "H.G. Wells", "form": "prose"},
    "agnes_may_fleming": {"name": "Agnes May Fleming", "form": "prose"},
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
        "--condition-dir",
        action="append",
        default=[],
        help=(
            "Custom condition directory mapping in the form name=path. "
            "Can be repeated; overrides built-in condition names."
        ),
    )
    parser.add_argument(
        "--authors",
        nargs="+",
        default=None,
        help="Subset of authors to include. Default: all authors from built-in map or --authors-config.",
    )
    parser.add_argument(
        "--authors-config",
        default=None,
        help=(
            "Optional campaign-style authors JSON. If provided, author metadata "
            "and source books are loaded from config['authors'] entries "
            "(supports book_path or book_paths)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Punctuation chunk size for comparisons.",
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Optional chunk-size sweep (overrides --chunk-size).",
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
    parser.add_argument(
        "--prose-only",
        action="store_true",
        help="Keep only prose authors in analysis.",
    )
    parser.add_argument(
        "--within-human-samples",
        type=int,
        default=200,
        help="Number of non-overlapping within-book chunk pairs per author/size.",
    )
    return parser.parse_args()


ARGS = parse_args()

ROOT = Path(__file__).resolve().parent
FULL_BOOKS_DIR = ROOT / ARGS.full_books_dir
OUTPUT_DIR = ROOT / ARGS.output_dir
AUTHOR_BOOK_PATHS: dict[str, list[Path]] = {}


def _resolve_local_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def _load_author_metadata() -> dict[str, dict[str, str]]:
    """
    Returns mapping:
      author_key -> {"name": str, "form": str}

    If --authors-config is provided, it overrides built-in metadata and can
    also define per-author book_path/book_paths values.
    """
    if not ARGS.authors_config:
        return {k: {"name": v["name"], "form": v["form"]} for k, v in DEFAULT_AUTHORS.items()}

    cfg_path = _resolve_local_path(ARGS.authors_config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    author_entries = cfg.get("authors", [])
    if not isinstance(author_entries, list) or not author_entries:
        raise ValueError(
            f"--authors-config must contain a non-empty 'authors' list: {cfg_path}"
        )

    metadata: dict[str, dict[str, str]] = {}
    for entry in author_entries:
        key = entry.get("key")
        if not key:
            raise ValueError(f"Invalid author entry (missing 'key') in {cfg_path}: {entry}")
        form = entry.get("form", "prose")
        if form not in {"prose", "play"}:
            raise ValueError(f"Unsupported form '{form}' for author '{key}' in {cfg_path}")
        metadata[key] = {
            "name": entry.get("name", key),
            "form": form,
        }

        raw_paths: list[str] = []
        if isinstance(entry.get("book_paths"), list):
            raw_paths.extend(entry["book_paths"])
        if entry.get("book_path"):
            raw_paths.append(entry["book_path"])
        if raw_paths:
            AUTHOR_BOOK_PATHS[key] = [_resolve_local_path(p) for p in raw_paths]

    return metadata


AUTHOR_METADATA = _load_author_metadata()

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


def _author_book_paths(author_key: str) -> list[Path]:
    if author_key in AUTHOR_BOOK_PATHS:
        return AUTHOR_BOOK_PATHS[author_key]
    return [FULL_BOOKS_DIR / f"{author_key}_full.txt"]


def _load_real_author_features(
    author_keys: list[str], chunk_size: int
) -> dict[str, dict[str, list[float]]]:
    features_by_author: dict[str, dict[str, list[float]]] = {}

    for author_key in author_keys:
        f1_rows: list[list[float]] = []
        f3_rows: list[list[float]] = []
        used_paths = 0

        for path in _author_book_paths(author_key):
            if not path.exists():
                print(f"[warn] Missing source book for '{author_key}': {path}")
                continue

            text = path.read_text(encoding="utf-8", errors="ignore")
            punctuation_seq = _extract_punctuation(text)
            chunk = _chunk_sequence(punctuation_seq, chunk_size, position="middle")
            if chunk is None:
                print(
                    f"[warn] Source text too short for chunk size {chunk_size}: "
                    f"{author_key} ({path})"
                )
                continue

            feats = _compute_features(chunk)
            if feats is None:
                print(f"[warn] Could not compute real features for '{author_key}' from {path}")
                continue

            f1_rows.append(feats["f1"])
            f3_rows.append(feats["f3"])
            used_paths += 1

        if used_paths == 0:
            continue

        # Average reference profile across source books for the author.
        features_by_author[author_key] = {
            "f1": np.mean(np.asarray(f1_rows, dtype=float), axis=0).tolist(),
            "f3": np.mean(np.asarray(f3_rows, dtype=float), axis=0).tolist(),
        }

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
    condition_dir_map = dict(DEFAULT_CONDITION_DIRS)
    for mapping in ARGS.condition_dir:
        if "=" not in mapping:
            raise ValueError(
                f"Invalid --condition-dir '{mapping}'. Expected format name=path."
            )
        name, rel_path = mapping.split("=", 1)
        name = name.strip()
        rel_path = rel_path.strip()
        if not name or not rel_path:
            raise ValueError(
                f"Invalid --condition-dir '{mapping}'. Expected format name=path."
            )
        condition_dir_map[name] = rel_path

    paths = {}
    for condition in selected_conditions:
        rel = condition_dir_map.get(condition)
        if rel is None:
            raise ValueError(f"Unknown condition '{condition}'")
        path = Path(rel)
        paths[condition] = path if path.is_absolute() else ROOT / rel
    return paths


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_author_keys() -> list[str]:
    requested_authors = ARGS.authors if ARGS.authors is not None else list(AUTHOR_METADATA.keys())
    author_keys: list[str] = []
    for author in requested_authors:
        meta = AUTHOR_METADATA.get(author)
        if meta is None:
            print(f"[warn] Unknown author '{author}' (ignored).")
            continue
        if ARGS.prose_only and meta["form"] != "prose":
            print(f"[info] Skipping non-prose author '{author}' in --prose-only mode.")
            continue
        author_keys.append(author)

    if not author_keys:
        raise ValueError("No valid authors selected after filtering.")
    return author_keys


def _resolve_chunk_sizes() -> list[int]:
    if ARGS.chunk_sizes:
        chunk_sizes = sorted(set(int(s) for s in ARGS.chunk_sizes))
    else:
        chunk_sizes = [int(ARGS.chunk_size)]

    if any(s <= 0 for s in chunk_sizes):
        raise ValueError(f"Chunk sizes must be positive integers. Got: {chunk_sizes}")
    return chunk_sizes


def _sample_non_overlapping_chunk_pairs(
    seq_len: int, chunk_size: int, n_samples: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """
    Sample non-overlapping chunk start pairs (start_a, start_b).
    """
    max_start = seq_len - chunk_size
    if max_start < 0 or n_samples <= 0:
        return []

    pairs: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = max(200, n_samples * 40)

    while len(pairs) < n_samples and attempts < max_attempts:
        attempts += 1
        start_a = int(rng.integers(0, max_start + 1))

        # Ensure no overlap.
        candidate_ranges: list[tuple[int, int]] = []
        left_end = start_a - chunk_size
        right_start = start_a + chunk_size
        if left_end >= 0:
            candidate_ranges.append((0, left_end))
        if right_start <= max_start:
            candidate_ranges.append((right_start, max_start))
        if not candidate_ranges:
            continue

        widths = np.asarray([b - a + 1 for a, b in candidate_ranges], dtype=float)
        probs = widths / widths.sum()
        idx = int(rng.choice(len(candidate_ranges), p=probs))
        lo, hi = candidate_ranges[idx]
        start_b = int(rng.integers(lo, hi + 1))
        pairs.append((start_a, start_b))

    return pairs


def _build_within_author_baseline_rows(
    author_keys: list[str],
    chunk_size: int,
    n_samples: int,
    seed: int = 42,
) -> list[dict[str, object]]:
    """
    Build within-author (human-vs-human) KL baseline summary rows for one chunk size.
    """
    rng = np.random.default_rng(seed + chunk_size)
    summary_rows: list[dict[str, object]] = []
    pooled_values: dict[str, list[float]] = {feature: [] for feature in TARGET_FEATURES}

    for author_key in author_keys:
        values_by_feature: dict[str, list[float]] = {feature: [] for feature in TARGET_FEATURES}

        for path in _author_book_paths(author_key):
            if not path.exists():
                print(f"[warn] Missing source book for baseline: {path}")
                continue

            text = path.read_text(encoding="utf-8", errors="ignore")
            seq = _extract_punctuation(text)
            if seq is None or len(seq) < 2 * chunk_size:
                print(
                    f"[warn] Not enough punctuation for within-author baseline "
                    f"({author_key}, chunk={chunk_size}, path={path})"
                )
                continue

            sampled_pairs = _sample_non_overlapping_chunk_pairs(
                seq_len=len(seq),
                chunk_size=chunk_size,
                n_samples=n_samples,
                rng=rng,
            )
            if not sampled_pairs:
                continue

            for start_a, start_b in sampled_pairs:
                chunk_a = seq[start_a : start_a + chunk_size]
                chunk_b = seq[start_b : start_b + chunk_size]
                feats_a = _compute_features(chunk_a)
                feats_b = _compute_features(chunk_b)
                if feats_a is None or feats_b is None:
                    continue

                for feature in TARGET_FEATURES:
                    values_by_feature[feature].append(
                        float(d_KL(feats_a[feature], feats_b[feature]))
                    )

        for feature in TARGET_FEATURES:
            vals = np.asarray(values_by_feature[feature], dtype=float)
            if vals.size == 0:
                continue
            pooled_values[feature].extend(vals.tolist())
            summary_rows.append(
                {
                    "chunk_size": chunk_size,
                    "scope": "per_author",
                    "author_key": author_key,
                    "feature": feature,
                    "n_pairs": int(vals.size),
                    "mean_kl": float(np.mean(vals)),
                    "std_kl": float(np.std(vals)),
                    "median_kl": float(np.median(vals)),
                    "p25_kl": float(np.percentile(vals, 25)),
                    "p75_kl": float(np.percentile(vals, 75)),
                }
            )

    for feature in TARGET_FEATURES:
        vals = np.asarray(pooled_values[feature], dtype=float)
        if vals.size == 0:
            continue
        summary_rows.append(
            {
                "chunk_size": chunk_size,
                "scope": "pooled",
                "author_key": "all_authors",
                "feature": feature,
                "n_pairs": int(vals.size),
                "mean_kl": float(np.mean(vals)),
                "std_kl": float(np.std(vals)),
                "median_kl": float(np.median(vals)),
                "p25_kl": float(np.percentile(vals, 25)),
                "p75_kl": float(np.percentile(vals, 75)),
            }
        )

    return summary_rows


def _build_llm_within_author_kl_tables(
    condition_paths: dict[str, Path],
    author_keys: list[str],
    chunk_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """
    Build LLM-vs-LLM within-author KL tables for one chunk size.

    Returns:
      - long_rows: one row per (condition, author, feature, run_a, run_b)
      - summary_rows: per-author and pooled summaries by condition+feature
    """
    long_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    pooled_values: dict[tuple[str, str], list[float]] = defaultdict(list)

    for condition, condition_dir in condition_paths.items():
        if not condition_dir.exists():
            continue

        for author_key in author_keys:
            runs = _load_llm_runs(condition_dir, author_key, chunk_size)
            if len(runs) < 2:
                continue

            for feature in TARGET_FEATURES:
                pair_vals: list[float] = []
                for i in range(len(runs) - 1):
                    run_a = runs[i]
                    for j in range(i + 1, len(runs)):
                        run_b = runs[j]
                        kl_value = float(d_KL(run_a[feature], run_b[feature]))
                        pair_vals.append(kl_value)
                        long_rows.append(
                            {
                                "condition": condition,
                                "author_key": author_key,
                                "feature": feature,
                                "run_id_a": run_a["run_id"],
                                "run_id_b": run_b["run_id"],
                                "kl_value": kl_value,
                            }
                        )

                vals = np.asarray(pair_vals, dtype=float)
                if vals.size == 0:
                    continue
                pooled_values[(condition, feature)].extend(vals.tolist())
                summary_rows.append(
                    {
                        "condition": condition,
                        "scope": "per_author",
                        "author_key": author_key,
                        "feature": feature,
                        "n_pairs": int(vals.size),
                        "mean_kl": float(np.mean(vals)),
                        "std_kl": float(np.std(vals)),
                        "median_kl": float(np.median(vals)),
                        "p25_kl": float(np.percentile(vals, 25)),
                        "p75_kl": float(np.percentile(vals, 75)),
                    }
                )

    for (condition, feature), vals_list in pooled_values.items():
        vals = np.asarray(vals_list, dtype=float)
        if vals.size == 0:
            continue
        summary_rows.append(
            {
                "condition": condition,
                "scope": "pooled",
                "author_key": "all_authors",
                "feature": feature,
                "n_pairs": int(vals.size),
                "mean_kl": float(np.mean(vals)),
                "std_kl": float(np.std(vals)),
                "median_kl": float(np.median(vals)),
                "p25_kl": float(np.percentile(vals, 25)),
                "p75_kl": float(np.percentile(vals, 75)),
            }
        )

    return long_rows, summary_rows


def _build_llm_vs_human_variance_rows(
    chunk_size: int,
    llm_within_summary_rows: list[dict[str, object]],
    within_human_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Build chunk-level comparison rows between LLM-within and human-within KL.
    """
    human_pooled_by_feature: dict[str, dict[str, object]] = {}
    for row in within_human_rows:
        if row.get("scope") == "pooled" and row.get("author_key") == "all_authors":
            human_pooled_by_feature[str(row["feature"])] = row

    rows: list[dict[str, object]] = []
    for row in llm_within_summary_rows:
        if row.get("scope") != "pooled" or row.get("author_key") != "all_authors":
            continue
        feature = str(row["feature"])
        human_row = human_pooled_by_feature.get(feature)
        if human_row is None:
            continue

        llm_mean = float(row["mean_kl"])
        human_mean = float(human_row["mean_kl"])
        ratio = (llm_mean / human_mean) if human_mean > 0 else None

        rows.append(
            {
                "chunk_size": chunk_size,
                "condition": str(row["condition"]),
                "feature": feature,
                "llm_within_mean_kl": llm_mean,
                "human_within_mean_kl": human_mean,
                "llm_minus_human_mean_kl": llm_mean - human_mean,
                "llm_to_human_mean_kl_ratio": ratio,
                "llm_n_pairs": int(row["n_pairs"]),
                "human_n_pairs": int(human_row["n_pairs"]),
            }
        )

    rows.sort(key=lambda r: (r["condition"], r["feature"]))
    return rows


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


def _run_single_chunk_analysis(
    chunk_size: int,
    author_keys: list[str],
    output_dir: Path,
) -> dict[str, list[dict[str, object]]]:
    condition_paths = _condition_paths(ARGS.conditions)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[chunk] {chunk_size}")
    print(f"[info] authors={author_keys}")
    print(f"[info] conditions={list(condition_paths.keys())}")
    print(f"[info] output_dir={output_dir}")

    real_features = _load_real_author_features(author_keys, chunk_size)
    missing_real = sorted(set(author_keys) - set(real_features.keys()))
    if missing_real:
        print(f"[warn] Missing real features for: {missing_real}")

    run_level_rows = _build_run_level_kl_rows(
        condition_paths=condition_paths,
        author_keys=author_keys,
        real_features=real_features,
        chunk_size=chunk_size,
    )

    if not run_level_rows:
        raise RuntimeError(f"No run-level KL rows generated for chunk size {chunk_size}.")

    llm_within_long_rows, llm_within_summary_rows = _build_llm_within_author_kl_tables(
        condition_paths=condition_paths,
        author_keys=author_keys,
        chunk_size=chunk_size,
    )

    # 1) Pairwise self-vs-other tables.
    pairwise_rows, pairwise_summary_rows = _build_pairwise_tables(
        run_level_rows=run_level_rows,
        author_keys=author_keys,
    )
    _format_float_columns(
        pairwise_rows,
        ["own_mean_kl", "target_mean_kl", "own_smaller_runs_pct"],
    )
    pairwise_csv = output_dir / "all_author_pairwise_self_vs_other.csv"
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
    pairwise_summary_csv = output_dir / "all_author_pairwise_self_vs_other_summary.csv"
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
    austen_vs_wells_csv = output_dir / "austen_vs_wells_validation.csv"
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
    austen_nearest_csv = output_dir / "austen_nearest_of_all_validation.csv"
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
    dynamic_vs_cols = [f"vs_{target}" for target in author_keys]
    _format_float_columns(
        cross_author_rows,
        ["own_author_kl", "best_target_kl"] + dynamic_vs_cols,
    )
    cross_author_csv = output_dir / "cross_author_kl_means.csv"
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
        ]
        + dynamic_vs_cols,
        rows=cross_author_rows,
    )

    # 4) Long-form run-level KL output.
    run_level_rows_out = [dict(r) for r in run_level_rows]
    _format_float_columns(run_level_rows_out, ["kl_value"])
    run_level_csv = output_dir / "run_level_kl_long.csv"
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

    # 5) LLM-vs-LLM within-author KL tables.
    llm_within_long_rows_out = [dict(r) for r in llm_within_long_rows]
    _format_float_columns(llm_within_long_rows_out, ["kl_value"])
    llm_within_long_csv = output_dir / "llm_within_author_kl_long.csv"
    _write_csv(
        path=llm_within_long_csv,
        fieldnames=[
            "condition",
            "author_key",
            "feature",
            "run_id_a",
            "run_id_b",
            "kl_value",
        ],
        rows=llm_within_long_rows_out,
    )

    llm_within_summary_rows_out = [dict(r) for r in llm_within_summary_rows]
    _format_float_columns(
        llm_within_summary_rows_out,
        ["mean_kl", "std_kl", "median_kl", "p25_kl", "p75_kl"],
    )
    llm_within_summary_csv = output_dir / "llm_within_author_kl_summary.csv"
    _write_csv(
        path=llm_within_summary_csv,
        fieldnames=[
            "condition",
            "scope",
            "author_key",
            "feature",
            "n_pairs",
            "mean_kl",
            "std_kl",
            "median_kl",
            "p25_kl",
            "p75_kl",
        ],
        rows=llm_within_summary_rows_out,
    )

    significance_rows: list[dict[str, object]] = []
    significance_delta_rows: list[dict[str, object]] = []
    significance_csv = None
    significance_deltas_csv = None
    if not ARGS.no_significance:
        significance_rows, significance_delta_rows = _build_significance_tables(
            run_level_rows=run_level_rows,
            author_keys=author_keys,
            bootstrap_iters=ARGS.bootstrap_iters,
            permutation_iters=ARGS.permutation_iters,
        )

        significance_rows_out = [dict(r) for r in significance_rows]
        _format_float_columns(
            significance_rows_out,
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
        significance_csv = output_dir / "significance_summary.csv"
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
            rows=significance_rows_out,
        )

        significance_delta_rows_out = [dict(r) for r in significance_delta_rows]
        _format_float_columns(
            significance_delta_rows_out,
            ["own_kl", "target_kl", "delta_target_minus_own"],
        )
        significance_deltas_csv = output_dir / "significance_run_deltas.csv"
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
            rows=significance_delta_rows_out,
        )

    print("[wrote]")
    print(f"  {run_level_csv}")
    print(f"  {pairwise_csv}")
    print(f"  {pairwise_summary_csv}")
    print(f"  {austen_vs_wells_csv}")
    print(f"  {austen_nearest_csv}")
    print(f"  {cross_author_csv}")
    print(f"  {llm_within_long_csv}")
    print(f"  {llm_within_summary_csv}")
    if significance_csv is not None:
        print(f"  {significance_csv}")
    if significance_deltas_csv is not None:
        print(f"  {significance_deltas_csv}")

    return {
        "run_level_rows": run_level_rows,
        "significance_rows": significance_rows,
        "llm_within_summary_rows": llm_within_summary_rows,
    }


def main() -> None:
    author_keys = _resolve_author_keys()
    chunk_sizes = _resolve_chunk_sizes()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[info] prose_only={ARGS.prose_only}")
    print(f"[info] chunk_sizes={chunk_sizes}")
    print(f"[info] output_dir={OUTPUT_DIR}")

    aggregate_significance_rows: list[dict[str, object]] = []
    aggregate_within_human_rows: list[dict[str, object]] = []
    aggregate_within_llm_rows: list[dict[str, object]] = []
    aggregate_llm_vs_human_rows: list[dict[str, object]] = []

    for chunk_size in chunk_sizes:
        chunk_output_dir = (
            OUTPUT_DIR if len(chunk_sizes) == 1 else OUTPUT_DIR / f"chunk_{chunk_size:04d}"
        )
        chunk_result = _run_single_chunk_analysis(
            chunk_size=chunk_size,
            author_keys=author_keys,
            output_dir=chunk_output_dir,
        )

        if not ARGS.no_significance:
            for row in chunk_result["significance_rows"]:
                row_out = dict(row)
                row_out["chunk_size"] = chunk_size
                aggregate_significance_rows.append(row_out)

        baseline_rows = _build_within_author_baseline_rows(
            author_keys=author_keys,
            chunk_size=chunk_size,
            n_samples=ARGS.within_human_samples,
            seed=42,
        )
        aggregate_within_human_rows.extend(baseline_rows)

        llm_within_rows = chunk_result["llm_within_summary_rows"]
        for row in llm_within_rows:
            row_out = dict(row)
            row_out["chunk_size"] = chunk_size
            aggregate_within_llm_rows.append(row_out)

        llm_vs_human_rows = _build_llm_vs_human_variance_rows(
            chunk_size=chunk_size,
            llm_within_summary_rows=llm_within_rows,
            within_human_rows=baseline_rows,
        )
        if llm_vs_human_rows:
            llm_vs_human_rows_out = [dict(r) for r in llm_vs_human_rows]
            _format_float_columns(
                llm_vs_human_rows_out,
                [
                    "llm_within_mean_kl",
                    "human_within_mean_kl",
                    "llm_minus_human_mean_kl",
                    "llm_to_human_mean_kl_ratio",
                ],
            )
            llm_vs_human_csv = chunk_output_dir / "llm_vs_human_within_variance_summary.csv"
            _write_csv(
                path=llm_vs_human_csv,
                fieldnames=[
                    "chunk_size",
                    "condition",
                    "feature",
                    "llm_within_mean_kl",
                    "human_within_mean_kl",
                    "llm_minus_human_mean_kl",
                    "llm_to_human_mean_kl_ratio",
                    "llm_n_pairs",
                    "human_n_pairs",
                ],
                rows=llm_vs_human_rows_out,
            )
            print(f"[wrote] {llm_vs_human_csv}")
            aggregate_llm_vs_human_rows.extend(llm_vs_human_rows)

    if aggregate_significance_rows:
        significance_chunk_csv = OUTPUT_DIR / "chunk_size_significance_summary.csv"
        significance_chunk_rows_out = [dict(r) for r in aggregate_significance_rows]
        _format_float_columns(
            significance_chunk_rows_out,
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
        _write_csv(
            path=significance_chunk_csv,
            fieldnames=[
                "chunk_size",
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
            rows=significance_chunk_rows_out,
        )
        print(f"[wrote] {significance_chunk_csv}")

    if aggregate_within_human_rows:
        within_human_csv = OUTPUT_DIR / "chunk_size_within_author_baseline.csv"
        within_human_rows_out = [dict(r) for r in aggregate_within_human_rows]
        _format_float_columns(
            within_human_rows_out,
            ["mean_kl", "std_kl", "median_kl", "p25_kl", "p75_kl"],
        )
        _write_csv(
            path=within_human_csv,
            fieldnames=[
                "chunk_size",
                "scope",
                "author_key",
                "feature",
                "n_pairs",
                "mean_kl",
                "std_kl",
                "median_kl",
                "p25_kl",
                "p75_kl",
            ],
            rows=within_human_rows_out,
        )
        print(f"[wrote] {within_human_csv}")

    if aggregate_within_llm_rows:
        within_llm_csv = OUTPUT_DIR / "chunk_size_llm_within_author_baseline.csv"
        within_llm_rows_out = [dict(r) for r in aggregate_within_llm_rows]
        _format_float_columns(
            within_llm_rows_out,
            ["mean_kl", "std_kl", "median_kl", "p25_kl", "p75_kl"],
        )
        _write_csv(
            path=within_llm_csv,
            fieldnames=[
                "chunk_size",
                "condition",
                "scope",
                "author_key",
                "feature",
                "n_pairs",
                "mean_kl",
                "std_kl",
                "median_kl",
                "p25_kl",
                "p75_kl",
            ],
            rows=within_llm_rows_out,
        )
        print(f"[wrote] {within_llm_csv}")

    if aggregate_llm_vs_human_rows:
        llm_vs_human_chunk_csv = (
            OUTPUT_DIR / "chunk_size_llm_vs_human_variance_summary.csv"
        )
        llm_vs_human_chunk_rows_out = [dict(r) for r in aggregate_llm_vs_human_rows]
        _format_float_columns(
            llm_vs_human_chunk_rows_out,
            [
                "llm_within_mean_kl",
                "human_within_mean_kl",
                "llm_minus_human_mean_kl",
                "llm_to_human_mean_kl_ratio",
            ],
        )
        _write_csv(
            path=llm_vs_human_chunk_csv,
            fieldnames=[
                "chunk_size",
                "condition",
                "feature",
                "llm_within_mean_kl",
                "human_within_mean_kl",
                "llm_minus_human_mean_kl",
                "llm_to_human_mean_kl_ratio",
                "llm_n_pairs",
                "human_n_pairs",
            ],
            rows=llm_vs_human_chunk_rows_out,
        )
        print(f"[wrote] {llm_vs_human_chunk_csv}")


if __name__ == "__main__":
    main()
