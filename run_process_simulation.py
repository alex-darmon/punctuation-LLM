#!/usr/bin/env python3
"""Run the pre-registered punctuation-process simulation study.

The existing frozen and inference-v2 outputs are read but never modified.
Before a full run, create and hand-label the deterministic segmentation sample:

  python run_process_simulation.py --prepare-validation-sample

Then fit and hash-lock parameters before running the frozen 8x8 sweep:

  python run_process_simulation.py --fit-parameters
  # Record the printed artifact hash in the config, freeze and timestamp it.
  python run_process_simulation.py

Small implementation checks may bypass the manual gate, but that fact is
recorded in the manifest:

  python run_process_simulation.py --replicates 2 --tail-replicates 2 \
      --reference-length-cap 250 --engineering-smoke \
      --output-dir results/repro_check/process_simulation_smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import gamma, ks_2samp, skew, wasserstein_distance

from punctlib import PUNCT_VECTOR, chunks, load_corpus
from punctlib.simulation import (
    DIALOGUE,
    STATE_NAMES,
    FittedProcess,
    SegmentedText,
    estimated_profiles,
    fit_process,
    oracle_profiles,
    score_sequence,
    segment_text,
    simulate_hsmm,
    simulate_iid,
    simulate_markov,
    stationary_distribution,
)
from punctlib.stats import dof


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = "campaigns/process_simulation_v1.json"
CANONICAL_OUTPUT_NAMES = (
    "segmentation_summary.csv",
    "fitted_parameter_summary.csv",
    "observations.csv",
    "summaries.csv",
    "slopes.csv",
    "sweep.csv",
    "tail_comparison.csv",
    "rho_drift.csv",
    "manifest.json",
)
ANALYSIS_SOURCE_PATHS = (
    "run_process_simulation.py",
    "punctlib/corpus.py",
    "punctlib/features.py",
    "punctlib/simulation.py",
    "punctlib/stats.py",
    "punctlib/text.py",
    "punctlib/vendored.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--tail-replicates", type=int, default=None)
    parser.add_argument("--reference-length-cap", type=int, default=None)
    parser.add_argument("--prepare-validation-sample", action="store_true")
    parser.add_argument("--fit-parameters", action="store_true")
    parser.add_argument(
        "--engineering-smoke",
        action="store_true",
        help="Bypass preregistration and manual-validation gates; recorded in manifest.",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [wrote] {path.relative_to(ROOT)} ({len(rows)} rows)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_segmented_books(
    config: dict[str, Any],
) -> tuple[Any, dict[str, dict[str, SegmentedText]]]:
    corpus = load_corpus(config["authors_config"], config["cache"])
    policy = config["segmentation"]
    by_author: dict[str, dict[str, SegmentedText]] = {}
    for author in corpus.authors:
        by_author[author] = {}
        for book_id in corpus.book_ids(author):
            path = resolve(book_id)
            segmented = segment_text(
                path.read_text(encoding="utf-8", errors="ignore"),
                strip_gutenberg=bool(policy["strip_gutenberg_boilerplate"]),
                reset_quotes_at_paragraph=True,
            )
            cached = corpus.marks(book_id)
            if list(segmented.marks) != cached:
                mismatch = next(
                    (
                        index
                        for index, (left, right) in enumerate(
                            zip(segmented.marks, cached)
                        )
                        if left != right
                    ),
                    min(len(segmented.marks), len(cached)),
                )
                raise AssertionError(
                    f"segmentation changed the frozen punctuation sequence for "
                    f"{book_id} at mark {mismatch}: segmented={len(segmented.marks)}, "
                    f"cached={len(cached)}"
                )
            by_author[author][book_id] = segmented
    return corpus, by_author


def segmentation_rows(
    by_author: dict[str, dict[str, SegmentedText]],
) -> list[dict[str, Any]]:
    rows = []
    for author, books in by_author.items():
        for book_id, book in books.items():
            state_counts = np.bincount(book.states, minlength=2)
            block_counts = np.bincount(
                [block.state for block in book.blocks], minlength=2
            )
            rows.append(
                {
                    "author": author,
                    "book": book_id,
                    "n_marks": len(book.marks),
                    "narration_marks": int(state_counts[0]),
                    "dialogue_marks": int(state_counts[1]),
                    "dialogue_fraction": float(
                        state_counts[1] / max(state_counts.sum(), 1)
                    ),
                    "narration_blocks": int(block_counts[0]),
                    "dialogue_blocks": int(block_counts[1]),
                    "unmatched_closers": book.unmatched_closers,
                    "paragraph_unclosed_quotes": book.paragraph_unclosed_quotes,
                }
            )
    return rows


def validation_sample_rows(
    by_author: dict[str, dict[str, SegmentedText]],
    validation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct the exact blinded sample declared by seed and corpus."""
    sample_size = int(validation["sample_size"])
    if sample_size <= 0:
        raise ValueError("manual validation sample_size must be positive")
    rng = np.random.default_rng(int(validation["seed"]))
    candidates: dict[int, list[tuple[str, str, int]]] = {0: [], 1: []}
    for author, books in by_author.items():
        for book_id, book in books.items():
            for index, state in enumerate(book.states):
                candidates[state].append((author, book_id, index))

    rows: list[dict[str, Any]] = []
    per_state = [sample_size // 2, sample_size - sample_size // 2]
    for state, wanted in enumerate(per_state):
        if len(candidates[state]) < wanted:
            raise ValueError(f"not enough {STATE_NAMES[state]} marks to sample")
        selected = rng.choice(len(candidates[state]), size=wanted, replace=False)
        for candidate_index in selected:
            author, book_id, mark_index = candidates[state][int(candidate_index)]
            book = by_author[author][book_id]
            offset = book.offsets[mark_index]
            left = " ".join(
                book.text[max(0, offset - 140) : offset].split()
            )
            marked = book.text[offset : offset + 1]
            right = " ".join(
                book.text[offset + 1 : offset + 141].split()
            )
            context = f"{left} \u27e6{marked}\u27e7 {right}".strip()
            rows.append(
                {
                    "sample_id": "",
                    "author": author,
                    "book": book_id,
                    "mark_index": mark_index,
                    "mark": book.marks[mark_index],
                    "context": context,
                    "manual_state": "",
                    "notes": "",
                }
            )
    rng.shuffle(rows)
    for index, row in enumerate(rows, 1):
        row["sample_id"] = f"S{index:03d}"
    return rows


def prepare_validation_sample(
    by_author: dict[str, dict[str, SegmentedText]],
    validation: dict[str, Any],
) -> None:
    output = resolve(validation["sample_path"])
    if output.exists():
        raise FileExistsError(
            f"{output} already exists; refusing to replace a possible hand annotation"
        )
    rows = validation_sample_rows(by_author, validation)
    write_csv(output, rows)
    print(
        "Fill manual_state with narration or dialogue before the full simulation."
    )


def validation_result(
    validation: dict[str, Any],
    by_author: dict[str, dict[str, SegmentedText]],
    *,
    skip_gate: bool,
) -> dict[str, Any]:
    path = resolve(validation["sample_path"])
    labels = set(validation["labels"])
    if not path.exists():
        if skip_gate:
            return {"status": "bypassed_missing", "path": str(path.relative_to(ROOT))}
        raise FileNotFoundError(
            f"manual segmentation sample is missing: {path}. Run with "
            "--prepare-validation-sample first."
        )
    rows = read_csv(path)
    expected = validation_sample_rows(by_author, validation)
    if len(rows) != int(validation["sample_size"]) or len(rows) != len(expected):
        raise ValueError(
            f"manual segmentation sample has {len(rows)} rows; "
            f"expected exactly {validation['sample_size']}"
        )
    immutable_fields = (
        "sample_id",
        "author",
        "book",
        "mark_index",
        "mark",
        "context",
    )
    for position, (actual, declared) in enumerate(zip(rows, expected), 1):
        for field in immutable_fields:
            if str(actual.get(field, "")) != str(declared[field]):
                raise ValueError(
                    f"manual segmentation sample membership changed at row "
                    f"{position}, field {field}"
                )
    ids = [row["sample_id"] for row in rows]
    members = [(row["book"], row["mark_index"]) for row in rows]
    if len(set(ids)) != len(ids) or len(set(members)) != len(members):
        raise ValueError("manual segmentation sample IDs and members must be unique")
    completed = [row for row in rows if row["manual_state"] in labels]
    invalid = [
        row["sample_id"]
        for row in rows
        if row["manual_state"] and row["manual_state"] not in labels
    ]
    if invalid:
        raise ValueError(f"invalid manual labels: {', '.join(invalid[:10])}")
    if len(completed) != len(rows) and not skip_gate:
        raise ValueError(
            f"manual segmentation validation is incomplete: "
            f"{len(completed)}/{len(rows)} labelled"
        )
    errors = 0
    state_labelled = {state: 0 for state in STATE_NAMES}
    state_errors = {state: 0 for state in STATE_NAMES}
    for row in completed:
        automatic = STATE_NAMES[
            by_author[row["author"]][row["book"]].states[int(row["mark_index"])]
        ]
        is_error = row["manual_state"] != automatic
        errors += is_error
        state_labelled[automatic] += 1
        state_errors[automatic] += is_error
    population_counts = np.bincount(
        [
            state
            for books in by_author.values()
            for book in books.values()
            for state in book.states
        ],
        minlength=2,
    )
    state_error_rates = {
        state: (
            state_errors[state] / state_labelled[state]
            if state_labelled[state]
            else None
        )
        for state in STATE_NAMES
    }
    population_weighted_error = (
        sum(
            population_counts[index] * state_error_rates[state]
            for index, state in enumerate(STATE_NAMES)
        )
        / population_counts.sum()
        if completed and all(value is not None for value in state_error_rates.values())
        else None
    )
    balanced_error = errors / len(completed) if completed else None
    max_balanced = float(validation["max_balanced_error_rate"])
    max_per_state = float(validation["max_per_state_error_rate"])
    thresholds_pass = (
        balanced_error is not None
        and balanced_error <= max_balanced
        and all(
            value is not None and value <= max_per_state
            for value in state_error_rates.values()
        )
    )
    if len(completed) == len(rows) and not thresholds_pass and not skip_gate:
        raise ValueError(
            "manual segmentation validation exceeds the pre-registered error "
            f"thresholds (balanced <= {max_balanced}, each state <= {max_per_state})"
        )
    return {
        "status": (
            "complete"
            if len(completed) == len(rows) and thresholds_pass
            else "bypassed_incomplete_or_failed"
        ),
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "n_sampled": len(rows),
        "n_labelled": len(completed),
        "n_errors": errors,
        "balanced_sample_error_rate": balanced_error,
        "state_error_rates": state_error_rates,
        "population_weighted_error_rate": population_weighted_error,
        "max_balanced_error_rate": max_balanced,
        "max_per_state_error_rate": max_per_state,
        "thresholds_pass": thresholds_pass,
    }


def fitted_by_fold(
    by_author: dict[str, dict[str, SegmentedText]],
    fold_by_author: dict[str, int],
    eps: float,
) -> dict[int, FittedProcess]:
    folds = sorted(set(fold_by_author.values()))
    return {
        fold: fit_process(
            (
                book
                for author, books in by_author.items()
                if fold_by_author[author] != fold
                for book in books.values()
            ),
            smoothing_eps=eps,
        )
        for fold in folds
    }


def fitted_parameter_rows(
    fitted: dict[int, FittedProcess],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    full: dict[str, Any] = {}
    for fold, process in fitted.items():
        full[str(fold)] = {
            "marginal": process.marginal.tolist(),
            "transition": process.transition.tolist(),
            "state_marginals": [value.tolist() for value in process.state_marginals],
            "state_transitions": [
                value.tolist() for value in process.state_transitions
            ],
            "initial_state": process.initial_state.tolist(),
            "dwell_lengths": [
                value.tolist() for value in process.dwell_lengths
            ],
            "n_books": process.n_books,
            "n_marks": process.n_marks,
            "state_mark_counts": list(process.state_mark_counts),
            "state_block_counts": list(process.state_block_counts),
        }
        for state in (0, 1):
            dwell = process.dwell_lengths[state]
            summary.append(
                {
                    "fold": fold,
                    "state": STATE_NAMES[state],
                    "n_training_books": process.n_books,
                    "n_training_marks": process.n_marks,
                    "state_marks": process.state_mark_counts[state],
                    "state_mark_fraction": (
                        process.state_mark_counts[state] / process.n_marks
                    ),
                    "n_blocks": process.state_block_counts[state],
                    "mean_dwell_marks": float(np.mean(dwell)),
                    "median_dwell_marks": float(np.median(dwell)),
                    "dwell_q90_marks": float(np.quantile(dwell, 0.9)),
                    "initial_state_probability": float(
                        process.initial_state[state]
                    ),
                }
            )
    return summary, full


def load_fitted_parameters(path: Path) -> dict[int, FittedProcess]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    output: dict[int, FittedProcess] = {}
    for fold, values in artifact["folds"].items():
        output[int(fold)] = FittedProcess(
            marginal=np.asarray(values["marginal"], dtype=float),
            transition=np.asarray(values["transition"], dtype=float),
            state_marginals=tuple(
                np.asarray(value, dtype=float)
                for value in values["state_marginals"]
            ),
            state_transitions=tuple(
                np.asarray(value, dtype=float)
                for value in values["state_transitions"]
            ),
            dwell_lengths=tuple(
                np.asarray(value, dtype=int) for value in values["dwell_lengths"]
            ),
            initial_state=np.asarray(values["initial_state"], dtype=float),
            n_books=int(values["n_books"]),
            n_marks=int(values["n_marks"]),
            state_mark_counts=tuple(
                int(value) for value in values["state_mark_counts"]
            ),
            state_block_counts=tuple(
                int(value) for value in values["state_block_counts"]
            ),
        )
    return output


def input_hashes(
    config: dict[str, Any],
    corpus: Any,
    by_author: dict[str, dict[str, SegmentedText]],
) -> dict[str, Any]:
    return {
        "authors_config": {
            "path": config["authors_config"],
            "sha256": sha256(resolve(config["authors_config"])),
        },
        "cache": {
            "path": config["cache"],
            "sha256": corpus.cache_sha256,
        },
        "human_text_sha256": {
            book_id: sha256(resolve(book_id))
            for books in by_author.values()
            for book_id in books
        },
    }


def source_hashes() -> dict[str, str]:
    return {path: sha256(ROOT / path) for path in ANALYSIS_SOURCE_PATHS}


def write_parameter_artifact(
    config: dict[str, Any],
    config_path: Path,
    corpus: Any,
    by_author: dict[str, dict[str, SegmentedText]],
    fitted: dict[int, FittedProcess],
    validation: dict[str, Any],
) -> Path:
    path = resolve(config["parameter_artifact"]["path"])
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen parameter artifact {path}")
    _, folds = fitted_parameter_rows(fitted)
    artifact = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_config": str(config_path.relative_to(ROOT)),
        "analysis_config_sha256_at_fit": sha256(config_path),
        "inputs": input_hashes(config, corpus, by_author),
        "source_sha256": source_hashes(),
        "manual_validation": validation,
        "segmentation_summary": segmentation_rows(by_author),
        "fold_by_author": config["evaluation"]["fold_by_author"],
        "smoothing_eps": config["simulation"]["smoothing_eps"],
        "punctuation_vector": list(PUNCT_VECTOR),
        "folds": folds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"  [wrote] {path.relative_to(ROOT)}")
    print(f"  parameter_artifact.sha256 = {sha256(path)}")
    return path


def add_scores(
    rows: list[dict[str, Any]],
    sequence: list[str] | tuple[str, ...],
    profiles: tuple[np.ndarray, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    scores = score_sequence(sequence, profiles)
    for feature in ("f1", "f3"):
        rows.append(
            {
                **metadata,
                "feature": feature,
                "g": scores[f"g_{feature}"],
                "raw_delta": scores[f"raw_delta_{feature}"],
                "excess_delta": scores[f"excess_delta_{feature}"],
            }
        )


def observed_rows(
    corpus: Any,
    by_author: dict[str, dict[str, SegmentedText]],
    fold_by_author: dict[str, int],
    chunk_sizes: list[int],
    eps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame_counts = {
        size: sum(
            len(book.marks) // size
            for books in by_author.values()
            for book in books.values()
        )
        for size in chunk_sizes
    }
    for author, books in by_author.items():
        for target_id, target in books.items():
            references = [
                other.marks for book_id, other in books.items() if book_id != target_id
            ]
            profiles = estimated_profiles(references, smoothing_eps=eps)
            for size in chunk_sizes:
                for chunk_index, sequence in enumerate(chunks(target.marks, size), 1):
                    add_scores(
                        rows,
                        sequence,
                        profiles,
                        {
                            "phase": "observed",
                            "model": "human",
                            "reference": "estimated",
                            "replicate": (
                                f"{author}:{Path(target_id).name}:{size}:{chunk_index}"
                            ),
                            "fold": fold_by_author[author],
                            "design_author": author,
                            "design_target_book": target_id,
                            "design_target_chunk": chunk_index,
                            "sampling_frame_chunks": frame_counts[size],
                            "dwell_scale": "",
                            "contrast": "",
                            "chunk_size": size,
                        },
                    )
    return rows


def make_designs(
    corpus: Any,
    fold_by_author: dict[str, int],
) -> list[dict[str, Any]]:
    designs = []
    for author in corpus.authors:
        books = corpus.book_ids(author)
        for target in books:
            references = [book for book in books if book != target]
            if len(references) != 2:
                raise AssertionError(f"{author} does not have a three-book design")
            designs.append(
                {
                    "fold": fold_by_author[author],
                    "author": author,
                    "target": target,
                    "target_length": len(corpus.marks(target)),
                    "reference_lengths": [
                        len(corpus.marks(book)) for book in references
                    ],
                }
            )
    if not designs:
        raise ValueError("no eligible three-book simulation designs")
    return designs


def chunk_sampling_frame(
    designs: list[dict[str, Any]],
    chunk_size: int,
) -> list[tuple[dict[str, Any], int]]:
    """One entry per real human chunk, preserving the pooled-chunk estimand."""
    frame = [
        (design, chunk_index)
        for design in designs
        for chunk_index in range(int(design["target_length"]) // chunk_size)
    ]
    if not frame:
        raise ValueError(f"no books contain a complete {chunk_size}-mark chunk")
    return frame


def generate_sequence(
    model: str,
    n_marks: int,
    fitted: FittedProcess,
    rng: np.random.Generator,
    *,
    dwell_scale: float = 1.0,
    contrast: float = 1.0,
) -> list[str]:
    if model == "iid":
        return simulate_iid(n_marks, fitted.marginal, rng)
    if model == "markov":
        return simulate_markov(
            n_marks,
            fitted.transition,
            stationary_distribution(fitted.transition),
            rng,
        )
    if model == "two_state_hsmm":
        return simulate_hsmm(
            n_marks,
            fitted,
            dwell_scale=dwell_scale,
            contrast=contrast,
            rng=rng,
        )
    raise ValueError(f"unknown model {model!r}")


def simulate_cell(
    *,
    rows: list[dict[str, Any]],
    phase: str,
    model: str,
    references: list[str],
    replicates: int,
    dwell_scale: float,
    contrast: float,
    chunk_sizes: list[int],
    designs: list[dict[str, Any]],
    fitted: dict[int, FittedProcess],
    eps: float,
    base_seed: int,
    reference_length_cap: int | None,
) -> None:
    frames = {
        size: chunk_sampling_frame(designs, size) for size in chunk_sizes
    }
    for replicate in range(replicates):
        for size in chunk_sizes:
            rng = np.random.default_rng(
                stable_seed(
                    base_seed,
                    phase,
                    model,
                    dwell_scale,
                    contrast,
                    size,
                    replicate,
                )
            )
            frame = frames[size]
            design, chunk_index = frame[int(rng.integers(len(frame)))]
            process = fitted[int(design["fold"])]
            target_end = (chunk_index + 1) * size
            target = generate_sequence(
                model,
                target_end,
                process,
                rng,
                dwell_scale=dwell_scale,
                contrast=contrast,
            )
            target_chunk = target[chunk_index * size : target_end]

            profiles_by_reference: dict[
                str, tuple[np.ndarray, np.ndarray]
            ] = {}
            if "oracle" in references:
                profiles_by_reference["oracle"] = oracle_profiles(model, process)
            if "estimated" in references:
                reference_sequences = [
                    generate_sequence(
                        model,
                        min(length, reference_length_cap)
                        if reference_length_cap is not None
                        else length,
                        process,
                        rng,
                        dwell_scale=dwell_scale,
                        contrast=contrast,
                    )
                    for length in design["reference_lengths"]
                ]
                profiles_by_reference["estimated"] = estimated_profiles(
                    reference_sequences, smoothing_eps=eps
                )

            for reference, profiles in profiles_by_reference.items():
                add_scores(
                    rows,
                    target_chunk,
                    profiles,
                    {
                        "phase": phase,
                        "model": model,
                        "reference": reference,
                        "replicate": replicate,
                        "fold": design["fold"],
                        "design_author": design["author"],
                        "design_target_book": design["target"],
                        "design_target_chunk": chunk_index + 1,
                        "sampling_frame_chunks": len(frame),
                        "dwell_scale": dwell_scale if model == "two_state_hsmm" else "",
                        "contrast": contrast if model == "two_state_hsmm" else "",
                        "chunk_size": size,
                    },
                )


SUMMARY_KEYS = (
    "phase",
    "model",
    "reference",
    "dwell_scale",
    "contrast",
    "chunk_size",
    "feature",
)


def raw_delta_gamma_diagnostic(values: np.ndarray) -> dict[str, float]:
    """Descriptive moment-matched gamma fit for positive raw divergence."""

    def ks_statistic(sample: np.ndarray, shape: float, scale: float) -> float:
        ordered = np.sort(sample)
        fitted_cdf = gamma.cdf(ordered, shape, loc=0, scale=scale)
        n_values = len(ordered)
        upper = np.arange(1, n_values + 1) / n_values - fitted_cdf
        lower = fitted_cdf - np.arange(0, n_values) / n_values
        return float(max(np.max(upper), np.max(lower)))

    values = values[np.isfinite(values)]
    if values.size < 8 or np.any(values < 0) or np.mean(values) <= 0:
        return {
            "raw_delta_gamma_shape": float("nan"),
            "raw_delta_gamma_scale": float("nan"),
            "raw_delta_gamma_ks_d": float("nan"),
        }
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    if variance <= 0:
        return {
            "raw_delta_gamma_shape": float("nan"),
            "raw_delta_gamma_scale": float("nan"),
            "raw_delta_gamma_ks_d": float("nan"),
        }
    shape = mean * mean / variance
    scale = variance / mean
    observed_d = ks_statistic(values, shape, scale)
    return {
        "raw_delta_gamma_shape": float(shape),
        "raw_delta_gamma_scale": float(scale),
        "raw_delta_gamma_ks_d": observed_d,
    }


def author_cluster_mean_interval(
    group: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    by_author: dict[str, list[float]] = defaultdict(list)
    for row in group:
        by_author[str(row["design_author"])].append(float(row["g"]))
    authors = sorted(by_author)
    if len(authors) < 2 or iterations <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = rng.choice(authors, size=len(authors), replace=True)
        values = [
            value for author in sampled for value in by_author[str(author)]
        ]
        draws[index] = np.mean(values)
    return (
        float(np.std(draws, ddof=1)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def summarise_observations(
    rows: list[dict[str, Any]],
    *,
    author_bootstrap_iterations: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in SUMMARY_KEYS)].append(row)

    summaries = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        metadata = dict(zip(SUMMARY_KEYS, key))
        g_values = np.asarray([float(row["g"]) for row in group])
        raw_delta_values = np.asarray(
            [float(row["raw_delta"]) for row in group]
        )
        excess_delta_values = np.asarray(
            [float(row["excess_delta"]) for row in group]
        )
        n = len(group)
        mean_g = float(np.mean(g_values))
        independent_se = (
            float(np.std(g_values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        )
        feature = str(metadata["feature"])
        degrees = dof(feature)
        chunk_size = float(metadata["chunk_size"])
        phi = mean_g / degrees
        if metadata["phase"] == "observed":
            se, ci_low, ci_high = author_cluster_mean_interval(
                group,
                iterations=author_bootstrap_iterations,
                seed=stable_seed(base_seed, "author-bootstrap", *key),
            )
            uncertainty_method = "author_cluster_bootstrap"
            cluster_unit = "author"
        else:
            se = independent_se
            ci_low = mean_g - 1.96 * se
            ci_high = mean_g + 1.96 * se
            uncertainty_method = "monte_carlo_standard_error"
            cluster_unit = "independent_simulation_replicate"
        row = {
            **metadata,
            "n_observations": n,
            "mean_g": mean_g,
            "uncertainty_method": uncertainty_method,
            "cluster_unit": cluster_unit,
            "se_mean_g": se,
            "mean_g_ci_low": ci_low,
            "mean_g_ci_high": ci_high,
            "phi_ci_low": ci_low / degrees,
            "phi_ci_high": ci_high / degrees,
            "g_q025": float(np.quantile(g_values, 0.025)),
            "g_q975": float(np.quantile(g_values, 0.975)),
            "degrees_of_freedom": degrees,
            "phi": phi,
            "rho_at_n": (phi - 1.0) / chunk_size,
            "implied_delta_at_n": (mean_g - degrees) / (2.0 * chunk_size),
            "mean_raw_delta": float(np.mean(raw_delta_values)),
            "mean_excess_delta": float(np.mean(excess_delta_values)),
            "excess_delta_cv": (
                float(
                    np.std(excess_delta_values, ddof=1)
                    / np.mean(excess_delta_values)
                )
                if n > 1 and np.mean(excess_delta_values) > 0
                else float("nan")
            ),
            "excess_delta_skew": (
                float(skew(excess_delta_values, bias=False))
                if n > 2
                else float("nan")
            ),
        }
        if metadata["phase"] in {"observed", "tail"}:
            row.update(raw_delta_gamma_diagnostic(raw_delta_values))
        summaries.append(row)
    return summaries


SLOPE_KEYS = ("phase", "model", "reference", "dwell_scale", "contrast", "feature")


def slope_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[tuple(row[key] for key in SLOPE_KEYS)].append(row)
    output = []
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        if len(rows) < 2:
            continue
        sizes = np.asarray([float(row["chunk_size"]) for row in rows])
        phi = np.asarray([float(row["phi"]) for row in rows])
        rho = float(np.dot(sizes, phi - 1.0) / np.dot(sizes, sizes))
        feature = str(rows[0]["feature"])
        output.append(
            {
                **dict(zip(SLOPE_KEYS, key)),
                "n_chunk_sizes": len(rows),
                "fixed_intercept_rho": rho,
                "implied_delta": rho * dof(feature) / 2.0,
            }
        )
    return output


def sweep_rows(
    summaries: list[dict[str, Any]],
    slopes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observed_slope = next(
        row
        for row in slopes
        if row["phase"] == "observed" and row["feature"] == "f3"
    )
    observed_phi = {
        int(row["chunk_size"]): float(row["phi"])
        for row in summaries
        if row["phase"] == "observed" and row["feature"] == "f3"
    }
    output = []
    for slope in slopes:
        if slope["phase"] != "sweep" or slope["feature"] != "f3":
            continue
        simulated_phi = {
            int(row["chunk_size"]): float(row["phi"])
            for row in summaries
            if row["phase"] == "sweep"
            and row["model"] == slope["model"]
            and row["reference"] == slope["reference"]
            and row["dwell_scale"] == slope["dwell_scale"]
            and row["contrast"] == slope["contrast"]
            and row["feature"] == "f3"
        }
        common = sorted(set(observed_phi) & set(simulated_phi))
        rmse = float(
            np.sqrt(
                np.mean(
                    [
                        (simulated_phi[size] - observed_phi[size]) ** 2
                        for size in common
                    ]
                )
            )
        )
        output.append(
            {
                "dwell_scale": slope["dwell_scale"],
                "contrast": slope["contrast"],
                "simulated_rho": slope["fixed_intercept_rho"],
                "observed_rho": observed_slope["fixed_intercept_rho"],
                "rho_difference": (
                    slope["fixed_intercept_rho"]
                    - observed_slope["fixed_intercept_rho"]
                ),
                "simulated_delta": slope["implied_delta"],
                "observed_delta": observed_slope["implied_delta"],
                "phi_rmse": rmse,
            }
        )
    return output


def tail_comparison_rows(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Distribution-free comparison of signed excess divergence tails."""
    output = []
    features = sorted({row["feature"] for row in observations})
    sizes = sorted({int(row["chunk_size"]) for row in observations})
    for feature in features:
        for size in sizes:
            human = np.asarray(
                [
                    float(row["excess_delta"])
                    for row in observations
                    if row["phase"] == "observed"
                    and row["feature"] == feature
                    and int(row["chunk_size"]) == size
                ]
            )
            simulated = np.asarray(
                [
                    float(row["excess_delta"])
                    for row in observations
                    if row["phase"] == "tail"
                    and row["feature"] == feature
                    and int(row["chunk_size"]) == size
                ]
            )
            if not human.size or not simulated.size:
                continue
            output.append(
                {
                    "feature": feature,
                    "chunk_size": size,
                    "n_human": len(human),
                    "n_simulated": len(simulated),
                    "two_sample_ks_d": float(
                        ks_2samp(human, simulated).statistic
                    ),
                    "wasserstein_distance": float(
                        wasserstein_distance(human, simulated)
                    ),
                    "human_negative_fraction": float(np.mean(human < 0)),
                    "simulated_negative_fraction": float(
                        np.mean(simulated < 0)
                    ),
                    "human_q95": float(np.quantile(human, 0.95)),
                    "simulated_q95": float(np.quantile(simulated, 0.95)),
                    "human_q99": float(np.quantile(human, 0.99)),
                    "simulated_q99": float(np.quantile(simulated, 0.99)),
                    "human_cv": (
                        float(np.std(human, ddof=1) / np.mean(human))
                        if np.mean(human) > 0
                        else float("nan")
                    ),
                    "simulated_cv": (
                        float(np.std(simulated, ddof=1) / np.mean(simulated))
                        if np.mean(simulated) > 0
                        else float("nan")
                    ),
                    "human_skew": float(skew(human, bias=False)),
                    "simulated_skew": float(skew(simulated, bias=False)),
                }
            )
    return output


def rho_drift_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for feature in ("f1", "f3"):
        observed = {
            int(row["chunk_size"]): row
            for row in summaries
            if row["phase"] == "observed" and row["feature"] == feature
        }
        fitted = {
            int(row["chunk_size"]): row
            for row in summaries
            if row["phase"] == "tail"
            and row["model"] == "two_state_hsmm"
            and row["feature"] == feature
        }
        for size in sorted(set(observed) & set(fitted)):
            output.append(
                {
                    "feature": feature,
                    "chunk_size": size,
                    "observed_phi": observed[size]["phi"],
                    "simulated_phi": fitted[size]["phi"],
                    "observed_rho_at_n": observed[size]["rho_at_n"],
                    "simulated_rho_at_n": fitted[size]["rho_at_n"],
                    "rho_difference": (
                        fitted[size]["rho_at_n"] - observed[size]["rho_at_n"]
                    ),
                }
            )
    return output


def assert_preregistered_targets(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    targets = config["pre_registration"]["primary_targets"]
    expected = targets["phi"]
    observed = {
        str(row["chunk_size"]): float(row["phi"])
        for row in summaries
        if row["phase"] == "observed" and row["feature"] == "f3"
    }
    for size, target in expected.items():
        if size not in observed or not np.isclose(
            observed[size], float(target), rtol=0, atol=1e-12
        ):
            raise AssertionError(
                f"frozen observed phi({size}) changed: "
                f"expected {target}, got {observed.get(size)}"
            )
    sizes = np.asarray([float(size) for size in sorted(observed, key=int)])
    phi = np.asarray([observed[str(int(size))] for size in sizes])
    rho = float(np.dot(sizes, phi - 1.0) / np.dot(sizes, sizes))
    delta = rho * dof("f3") / 2.0
    declared = {
        "fixed_intercept_rho": (rho, float(targets["fixed_intercept_rho"])),
        "implied_delta_df90": (delta, float(targets["implied_delta_df90"])),
    }
    for label, (actual, target) in declared.items():
        if not np.isclose(actual, target, rtol=0, atol=1e-12):
            raise AssertionError(
                f"frozen observed {label} changed: expected {target}, got {actual}"
            )


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        return {
            "commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "branch", "--show-current"),
            "dirty": bool(run("git", "status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "branch": "unknown", "dirty": None}


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "scipy", "spacy"):
        versions[package] = importlib.metadata.version(package)
    return versions


def is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def validate_execution_mode(
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    if args.prepare_validation_sample and args.fit_parameters:
        raise ValueError("choose either validation-sample preparation or parameter fitting")
    if args.fit_parameters and args.engineering_smoke:
        raise ValueError("parameter artifacts cannot be fitted in engineering-smoke mode")
    overrides = {
        "--output-dir": args.output_dir,
        "--replicates": args.replicates,
        "--tail-replicates": args.tail_replicates,
        "--reference-length-cap": args.reference_length_cap,
    }
    active_overrides = [name for name, value in overrides.items() if value is not None]
    if args.fit_parameters and active_overrides:
        raise ValueError("parameter fitting does not accept simulation/output overrides")
    if args.prepare_validation_sample:
        return
    if args.fit_parameters:
        return
    if args.engineering_smoke:
        smoke_root = ROOT / "results" / "repro_check"
        if args.output_dir is None or not is_within(output_dir, smoke_root):
            raise ValueError(
                "engineering smoke output must be explicitly placed under "
                "results/repro_check/"
            )
        return
    if active_overrides:
        raise ValueError(
            "runtime overrides are allowed only with --engineering-smoke: "
            + ", ".join(active_overrides)
        )
    if output_dir.resolve() != resolve(config["output_dir"]).resolve():
        raise ValueError("inferential runs must use the configured canonical output")
    if config["pre_registration"]["status"] != "frozen_before_simulation":
        raise ValueError(
            "the pre-registration is not frozen: hash-lock the fitted parameter "
            "artifact, set status to frozen_before_simulation, and commit or "
            "externally timestamp both files"
        )


def validate_parameter_artifact(
    path: Path,
    expected_sha256: str | None,
    config: dict[str, Any],
    corpus: Any,
    by_author: dict[str, dict[str, SegmentedText]],
    validation: dict[str, Any],
) -> None:
    if not expected_sha256:
        raise ValueError("parameter_artifact.sha256 must be frozen in the config")
    if not path.is_file():
        raise FileNotFoundError(f"frozen parameter artifact is missing: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"parameter artifact hash mismatch: expected {expected_sha256}, got {actual}"
        )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["inputs"] != input_hashes(config, corpus, by_author):
        raise ValueError("parameter artifact input hashes no longer match the corpus")
    if artifact["source_sha256"] != source_hashes():
        raise ValueError("parameter-fitting source changed after artifact creation")
    if artifact["manual_validation"]["sha256"] != validation["sha256"]:
        raise ValueError("parameter artifact used a different manual validation file")
    if artifact["fold_by_author"] != config["evaluation"]["fold_by_author"]:
        raise ValueError("parameter artifact author folds do not match the config")
    if float(artifact["smoothing_eps"]) != float(
        config["simulation"]["smoothing_eps"]
    ):
        raise ValueError("parameter artifact smoothing differs from the config")


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    simulation = config["simulation"]
    chunk_sizes = [int(value) for value in simulation["chunk_sizes"]]
    eps = float(simulation["smoothing_eps"])
    base_seed = int(simulation["seed"])
    replicates = (
        args.replicates
        if args.replicates is not None
        else int(simulation["replicates_per_cell"])
    )
    tail_replicates = (
        args.tail_replicates
        if args.tail_replicates is not None
        else int(simulation["tail_replicates_at_fitted_point"])
    )
    reference_length_cap = (
        args.reference_length_cap
        if args.reference_length_cap is not None
        else simulation["reference_length_cap"]
    )
    output_dir = resolve(args.output_dir or config["output_dir"])
    validate_execution_mode(args, config, output_dir)
    if (
        not args.engineering_smoke
        and not args.prepare_validation_sample
        and not args.fit_parameters
    ):
        existing = [
            name for name in CANONICAL_OUTPUT_NAMES if (output_dir / name).exists()
        ]
        if existing:
            raise FileExistsError(
                "refusing to overwrite canonical simulation outputs: "
                + ", ".join(existing)
            )

    print("[1/8] segmenting and verifying human books")
    corpus, by_author = load_segmented_books(config)
    if args.prepare_validation_sample:
        prepare_validation_sample(by_author, config["manual_validation"])
        return
    validation = validation_result(
        config["manual_validation"],
        by_author,
        skip_gate=args.engineering_smoke,
    )

    fold_by_author = {
        author: int(fold)
        for author, fold in config["evaluation"]["fold_by_author"].items()
    }
    if set(fold_by_author) != set(corpus.authors):
        raise ValueError("configured author folds do not match the loaded corpus")
    if args.fit_parameters:
        print("[2/8] fitting parameter artifact only")
        fitted = fitted_by_fold(by_author, fold_by_author, eps)
        write_parameter_artifact(
            config,
            config_path,
            corpus,
            by_author,
            fitted,
            validation,
        )
        return

    parameter_path = resolve(config["parameter_artifact"]["path"])
    if args.engineering_smoke:
        print("[2/8] fitting in-memory engineering-smoke parameters")
        fitted = fitted_by_fold(by_author, fold_by_author, eps)
        parameter_sha256 = None
    else:
        print("[2/8] loading hash-locked parameter artifact")
        validate_parameter_artifact(
            parameter_path,
            config["parameter_artifact"]["sha256"],
            config,
            corpus,
            by_author,
            validation,
        )
        fitted = load_fitted_parameters(parameter_path)
        parameter_sha256 = sha256(parameter_path)
    parameter_rows, _ = fitted_parameter_rows(fitted)

    print("[3/8] reproducing frozen observed targets")
    observations = observed_rows(
        corpus, by_author, fold_by_author, chunk_sizes, eps
    )
    observed_summaries = summarise_observations(
        observations,
        author_bootstrap_iterations=0,
        base_seed=base_seed,
    )
    assert_preregistered_targets(observed_summaries, config)

    designs = make_designs(corpus, fold_by_author)
    sampling_frames = {
        str(size): {
            "n_chunks": len(chunk_sampling_frame(designs, size)),
            "n_books": sum(
                int(design["target_length"]) >= size for design in designs
            ),
        }
        for size in chunk_sizes
    }
    print("[4/8] simulating homogeneous baselines")
    for model in ("iid", "markov"):
        references = list(simulation["models"][model]["references"])
        simulate_cell(
            rows=observations,
            phase="baseline",
            model=model,
            references=references,
            replicates=replicates,
            dwell_scale=1.0,
            contrast=1.0,
            chunk_sizes=chunk_sizes,
            designs=designs,
            fitted=fitted,
            eps=eps,
            base_seed=base_seed,
            reference_length_cap=reference_length_cap,
        )

    print("[5/8] simulating two-state dwell/contrast sweep")
    for dwell_scale in simulation["dwell_scales"]:
        for contrast in simulation["contrast_levels"]:
            simulate_cell(
                rows=observations,
                phase="sweep",
                model="two_state_hsmm",
                references=["estimated"],
                replicates=replicates,
                dwell_scale=float(dwell_scale),
                contrast=float(contrast),
                chunk_sizes=chunk_sizes,
                designs=designs,
                fitted=fitted,
                eps=eps,
                base_seed=base_seed,
                reference_length_cap=reference_length_cap,
            )

    print("[6/8] simulating fitted-point tail")
    fitted_point = simulation["fitted_point"]
    simulate_cell(
        rows=observations,
        phase="tail",
        model="two_state_hsmm",
        references=["estimated"],
        replicates=tail_replicates,
        dwell_scale=float(fitted_point["dwell_scale"]),
        contrast=float(fitted_point["contrast"]),
        chunk_sizes=chunk_sizes,
        designs=designs,
        fitted=fitted,
        eps=eps,
        base_seed=base_seed,
        reference_length_cap=reference_length_cap,
    )

    print("[7/8] computing clustered and distribution-free summaries")
    summaries = summarise_observations(
        observations,
        author_bootstrap_iterations=int(
            simulation["human_author_bootstrap_iterations"]
        ),
        base_seed=base_seed,
    )
    assert_preregistered_targets(summaries, config)
    slopes = slope_rows(summaries)
    sweep = sweep_rows(summaries, slopes)
    tail_comparison = tail_comparison_rows(observations)
    rho_drift = rho_drift_rows(summaries)

    print("[8/8] writing provenance-locked outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "segmentation_summary.csv", segmentation_rows(by_author))
    write_csv(output_dir / "fitted_parameter_summary.csv", parameter_rows)
    write_csv(output_dir / "observations.csv", observations)
    write_csv(output_dir / "summaries.csv", summaries)
    write_csv(output_dir / "slopes.csv", slopes)
    write_csv(output_dir / "sweep.csv", sweep)
    write_csv(output_dir / "tail_comparison.csv", tail_comparison)
    write_csv(output_dir / "rho_drift.csv", rho_drift)

    output_files = [output_dir / name for name in CANONICAL_OUTPUT_NAMES[:-1]]
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_config": str(config_path.relative_to(ROOT)),
        "analysis_config_sha256": sha256(config_path),
        "git": git_state(),
        "inputs": input_hashes(config, corpus, by_author),
        "parameter_artifact": {
            "path": str(parameter_path.relative_to(ROOT)),
            "sha256": parameter_sha256,
        },
        "environment": {
            "python": platform.python_version(),
            "packages": package_versions(),
        },
        "manual_validation": validation,
        "engineering_smoke": args.engineering_smoke,
        "pre_registration_status": config["pre_registration"]["status"],
        "effective_settings": {
            "replicates_per_cell": replicates,
            "tail_replicates": tail_replicates,
            "human_author_bootstrap_iterations": int(
                simulation["human_author_bootstrap_iterations"]
            ),
            "reference_length_cap": reference_length_cap,
            "sampling_frame": (
                "one uniformly sampled entry from the complete pooled human "
                "chunk frame, independently at each chunk size"
            ),
            "sampling_frame_counts": sampling_frames,
        },
        "n_authors": len(corpus.authors),
        "n_books": sum(len(books) for books in by_author.values()),
        "punctuation_vector": list(PUNCT_VECTOR),
        "source_sha256": source_hashes(),
        "output_sha256": {
            path.name: sha256(path) for path in output_files
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  [wrote] {(output_dir / 'manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
