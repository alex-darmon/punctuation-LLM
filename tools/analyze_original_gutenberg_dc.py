#!/usr/bin/env python3
"""Compute the D/C author-separability diagnostic on the paper's full corpus.

The exact 651-author, 14,947-document Project Gutenberg snapshot is archived at
https://doi.org/10.5281/zenodo.3605100.  For each author and each of the paper's
f1 and f3 punctuation features, this script computes:

    C = mean directed KL over all ordered pairs of the author's documents
    D = directed KL from the author's pooled profile to its nearest other author

The reported diagnostic is D/C.  Pooling concatenates all of an author's
punctuation sequences, matching ``build_reference_set(..., exclude_books=())``
in the frozen ten-author pipeline.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DOI = "https://doi.org/10.5281/zenodo.3605100"
EXPECTED_MD5 = "095a1e6d65b993e335a1d2998f13452f"
EXPECTED_DOCUMENTS = 14_947
EXPECTED_AUTHORS = 651
PUNCT_VECTOR = ("!", '"', "(", ")", ",", ".", ":", ";", "?", "^")
PUNCT_INDEX = {mark: i for i, mark in enumerate(PUNCT_VECTOR)}
K = len(PUNCT_VECTOR)
FEATURES = ("f1", "f3")


@dataclass(frozen=True)
class Document:
    author: str
    document_id: str
    marks: np.ndarray
    f1: np.ndarray
    f3: np.ndarray

    def feature(self, name: str) -> np.ndarray:
        if name == "f1":
            return self.f1
        if name == "f3":
            return self.f3
        raise KeyError(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="cache/original_paper/punctuation_stylometry.p",
        help="Zenodo pickle path, relative to the repository root by default.",
    )
    parser.add_argument(
        "--out",
        default="results/original_gutenberg_dc",
        help="Output directory, relative to the repository root by default.",
    )
    parser.add_argument(
        "--variant",
        choices=("all", "deduplicated", "both"),
        default="both",
        help=(
            "Use every archived row, remove identical punctuation sequences "
            "within each author, or report both as a sensitivity check."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Pairwise-KL block size (lower this if memory is constrained).",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Do not verify the archive against the Zenodo MD5.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolved(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def file_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pick_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def _mark_indices(sequence: object, row_number: int) -> np.ndarray:
    if sequence is None:
        raise ValueError(f"row {row_number}: missing punctuation sequence")
    try:
        marks = list(sequence)
    except TypeError as exc:
        raise ValueError(
            f"row {row_number}: punctuation sequence is not iterable"
        ) from exc
    unknown = sorted({str(mark) for mark in marks if mark not in PUNCT_INDEX})
    if unknown:
        raise ValueError(
            f"row {row_number}: marks outside the paper's vector: {unknown}"
        )
    if not marks:
        raise ValueError(f"row {row_number}: empty punctuation sequence")
    return np.fromiter(
        (PUNCT_INDEX[mark] for mark in marks),
        dtype=np.uint8,
        count=len(marks),
    )


def features_from_mark_arrays(mark_arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Feature extraction for concatenated documents, including book boundaries."""
    mark_counts = np.zeros(K, dtype=np.int64)
    pair_counts = np.zeros((K, K), dtype=np.int64)
    previous_last: int | None = None

    for marks in mark_arrays:
        if marks.size == 0:
            continue
        mark_counts += np.bincount(marks, minlength=K)
        if previous_last is not None:
            pair_counts[previous_last, int(marks[0])] += 1
        if marks.size > 1:
            encoded = marks[:-1].astype(np.int64) * K + marks[1:]
            pair_counts += np.bincount(encoded, minlength=K * K).reshape(K, K)
        previous_last = int(marks[-1])

    n_marks = int(mark_counts.sum())
    if n_marks == 0:
        raise ValueError("cannot extract features from no punctuation marks")

    f1 = mark_counts.astype(float) / n_marks
    transition = np.zeros((K, K), dtype=float)
    row_totals = pair_counts.sum(axis=1)
    np.divide(
        pair_counts,
        row_totals[:, None],
        out=transition,
        where=row_totals[:, None] > 0,
    )
    f3 = transition * f1[:, None]
    return f1, f3.reshape(-1)


def _feature_array(value: object, size: int, label: str, row_number: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: invalid {label} feature") from exc
    if array.size != size or not np.all(np.isfinite(array)):
        raise ValueError(
            f"row {row_number}: {label} has shape {array.shape}; expected ({size},)"
        )
    return array


def extract_documents(frame: pd.DataFrame) -> dict[str, list[Document]]:
    author_col = _pick_column(frame, ("author", "Author"))
    sequence_col = _pick_column(frame, ("seq_pun", "punctuation_sequence"))
    id_col = _pick_column(frame, ("book_id", "document_id", "title"))
    f1_col = _pick_column(frame, ("freq_pun", "f1"))
    f3_col = _pick_column(frame, ("normalised_tran_mat", "f3"))

    missing = [
        label
        for label, column in (
            ("author", author_col),
            ("punctuation sequence", sequence_col),
            ("f1", f1_col),
            ("f3", f3_col),
        )
        if column is None
    ]
    if missing:
        raise ValueError(
            f"archive is missing required columns {missing}; available columns: "
            f"{list(frame.columns)}"
        )

    authors = frame[author_col].array
    sequences = frame[sequence_col].array
    identifiers = frame[id_col].array if id_col else None
    f1_values = frame[f1_col].array
    f3_values = frame[f3_col].array

    by_author: dict[str, list[Document]] = {}
    for position in range(len(frame)):
        author_value = authors[position]
        if pd.isna(author_value):
            raise ValueError(f"row {position}: missing author")
        author = str(author_value)
        marks = _mark_indices(sequences[position], position)
        f1 = _feature_array(f1_values[position], K, "f1", position)
        f3 = _feature_array(f3_values[position], K * K, "f3", position)

        # Confirm that sequence interpretation and archived features agree.
        derived_f1, derived_f3 = features_from_mark_arrays((marks,))
        if not np.allclose(f1, derived_f1, atol=2e-6, rtol=2e-6):
            raise ValueError(f"row {position}: archived f1 does not match seq_pun")
        if not np.allclose(f3, derived_f3, atol=2e-5, rtol=2e-5):
            raise ValueError(f"row {position}: archived f3 does not match seq_pun")

        document_id = str(identifiers[position]) if identifiers is not None else str(position)
        by_author.setdefault(author, []).append(
            Document(
                author=author,
                document_id=document_id,
                marks=marks,
                f1=f1,
                f3=f3,
            )
        )
        if (position + 1) % 1_000 == 0:
            print(f"  compacted {position + 1:,}/{len(frame):,} documents", flush=True)
    return by_author


def deduplicate_documents(
    by_author: dict[str, list[Document]],
) -> tuple[dict[str, list[Document]], int]:
    output: dict[str, list[Document]] = {}
    removed = 0
    for author, documents in by_author.items():
        seen: set[bytes] = set()
        kept: list[Document] = []
        for document in documents:
            signature = hashlib.sha256(document.marks.tobytes()).digest()
            if signature in seen:
                removed += 1
                continue
            seen.add(signature)
            kept.append(document)
        output[author] = kept
    return output, removed


def legacy_kl_matrix(
    left: np.ndarray, right: np.ndarray, block_size: int = 128
) -> np.ndarray:
    """Pairwise KL with the vendored library's common-support renormalisation."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("left and right must be 2D arrays with equal feature width")
    if np.any(left < 0) or np.any(right < 0):
        raise ValueError("KL inputs cannot contain negative values")

    result = np.zeros((left.shape[0], right.shape[0]), dtype=float)
    log_left = np.zeros_like(left)
    log_right = np.zeros_like(right)
    np.log(left, out=log_left, where=left > 0)
    np.log(right, out=log_right, where=right > 0)

    for i in range(0, left.shape[0], block_size):
        p = left[i : i + block_size, None, :]
        lp = log_left[i : i + block_size, None, :]
        for j in range(0, right.shape[0], block_size):
            q = right[None, j : j + block_size, :]
            lq = log_right[None, j : j + block_size, :]
            common = (p > 0) & (q > 0)
            p_sum = np.sum(p * common, axis=2)
            q_sum = np.sum(q * common, axis=2)
            numerator = np.sum(np.where(common, p * (lp - lq), 0.0), axis=2)
            valid = (p_sum > 0) & (q_sum > 0)
            block = np.zeros_like(p_sum)
            block[valid] = (
                numerator[valid] / p_sum[valid]
                - np.log(p_sum[valid])
                + np.log(q_sum[valid])
            )
            result[
                i : i + p.shape[0],
                j : j + q.shape[1],
            ] = block
    return result


def mean_ordered_within(features: np.ndarray, block_size: int) -> float:
    n_documents = features.shape[0]
    if n_documents < 2:
        return float("nan")
    total = 0.0
    count = 0
    for i in range(0, n_documents, block_size):
        left = features[i : i + block_size]
        for j in range(0, n_documents, block_size):
            right = features[j : j + block_size]
            distances = legacy_kl_matrix(left, right, block_size=block_size)
            if i == j:
                mask = ~np.eye(distances.shape[0], dtype=bool)
                total += float(distances[mask].sum())
                count += int(mask.sum())
            else:
                total += float(distances.sum())
                count += int(distances.size)
    return total / count


def analyse_variant(
    variant: str,
    by_author: dict[str, list[Document]],
    block_size: int,
) -> list[dict[str, object]]:
    authors = sorted(by_author)
    profiles: dict[str, dict[str, np.ndarray]] = {}
    mark_totals: dict[str, int] = {}
    for number, author in enumerate(authors, start=1):
        documents = by_author[author]
        f1, f3 = features_from_mark_arrays(document.marks for document in documents)
        profiles[author] = {"f1": f1, "f3": f3}
        mark_totals[author] = sum(document.marks.size for document in documents)
        if number % 100 == 0:
            print(f"  pooled {number:,}/{len(authors):,} authors", flush=True)

    rows: list[dict[str, object]] = []
    for feature in FEATURES:
        profile_matrix = np.stack([profiles[author][feature] for author in authors])
        profile_distances = legacy_kl_matrix(
            profile_matrix, profile_matrix, block_size=block_size
        )
        directed_between = profile_distances[~np.eye(len(authors), dtype=bool)]
        between_baseline = float(directed_between.mean())
        np.fill_diagonal(profile_distances, np.inf)

        print(f"  computing {feature} within-author consistency", flush=True)
        for number, author in enumerate(authors, start=1):
            documents = by_author[author]
            document_features = np.stack(
                [document.feature(feature) for document in documents]
            )
            consistency = mean_ordered_within(document_features, block_size)
            nearest_index = int(np.argmin(profile_distances[number - 1]))
            nearest_distance = float(profile_distances[number - 1, nearest_index])
            ratio = (
                nearest_distance / consistency
                if np.isfinite(consistency) and consistency > 0
                else float("nan")
            )
            rows.append(
                {
                    "corpus_variant": variant,
                    "feature": feature,
                    "author": author,
                    "n_documents": len(documents),
                    "n_marks": mark_totals[author],
                    "consistency_C": consistency,
                    "nearest_other_D": nearest_distance,
                    "nearest_other": authors[nearest_index],
                    "ratio_D_over_C": ratio,
                    "margin_D_minus_C": nearest_distance - consistency,
                    "between_author_baseline": between_baseline,
                    "C_over_baseline": (
                        consistency / between_baseline
                        if between_baseline > 0
                        else float("nan")
                    ),
                }
            )
            if number % 100 == 0:
                print(f"    {number:,}/{len(authors):,} authors", flush=True)
    return rows


def make_summary(
    rows: list[dict[str, object]],
    variants: dict[str, dict[str, list[Document]]],
    duplicate_rows_removed: int,
    source_md5: str,
) -> dict[str, object]:
    summary_rows: list[dict[str, object]] = []
    for variant in variants:
        variant_rows = [row for row in rows if row["corpus_variant"] == variant]
        for feature in FEATURES:
            selected = [row for row in variant_rows if row["feature"] == feature]
            ratios = np.asarray([row["ratio_D_over_C"] for row in selected], dtype=float)
            finite = ratios[np.isfinite(ratios)]
            summary_rows.append(
                {
                    "corpus_variant": variant,
                    "feature": feature,
                    "n_authors": len(selected),
                    "n_documents": sum(
                        len(documents) for documents in variants[variant].values()
                    ),
                    "n_ratio_gt_1": int(np.sum(finite > 1)),
                    "fraction_ratio_gt_1": float(np.mean(finite > 1)),
                    "ratio_median": float(np.median(finite)),
                    "ratio_q1": float(np.quantile(finite, 0.25)),
                    "ratio_q3": float(np.quantile(finite, 0.75)),
                    "ratio_min": float(np.min(finite)),
                    "ratio_max": float(np.max(finite)),
                }
            )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_doi": SOURCE_DOI,
        "source_md5": source_md5,
        "expected_documents": EXPECTED_DOCUMENTS,
        "expected_authors": EXPECTED_AUTHORS,
        "punctuation_vector": list(PUNCT_VECTOR),
        "definitions": {
            "C": (
                "mean directed legacy KL over all ordered pairs of distinct "
                "documents by the same author"
            ),
            "D": (
                "minimum directed legacy KL from the author's all-document "
                "concatenated profile to another author's corresponding profile"
            ),
            "ratio": "D/C",
            "pooling": "concatenation in archive row order, including book boundaries",
        },
        "duplicate_rows_removed_in_sensitivity_variant": duplicate_rows_removed,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "summaries": summary_rows,
    }


def make_plots(rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for variant in sorted({str(row["corpus_variant"]) for row in rows}):
        figure, axes = plt.subplots(2, 2, figsize=(12, 10))
        for feature_index, feature in enumerate(FEATURES):
            selected = [
                row
                for row in rows
                if row["corpus_variant"] == variant and row["feature"] == feature
            ]
            c_values = np.asarray([row["consistency_C"] for row in selected], dtype=float)
            d_values = np.asarray([row["nearest_other_D"] for row in selected], dtype=float)
            ratios = np.asarray([row["ratio_D_over_C"] for row in selected], dtype=float)
            valid_cd = (c_values > 0) & (d_values > 0)
            valid_ratio = np.isfinite(ratios) & (ratios > 0)

            scatter = axes[feature_index, 0]
            scatter.scatter(c_values[valid_cd], d_values[valid_cd], s=13, alpha=0.55)
            lower = float(min(c_values[valid_cd].min(), d_values[valid_cd].min()))
            upper = float(max(c_values[valid_cd].max(), d_values[valid_cd].max()))
            scatter.plot([lower, upper], [lower, upper], color="black", linestyle="--")
            scatter.set_xscale("log")
            scatter.set_yscale("log")
            scatter.set_xlabel("within-author consistency C")
            scatter.set_ylabel("nearest-other distance D")
            scatter.set_title(f"{feature}: each point is one author")

            histogram = axes[feature_index, 1]
            log_ratios = np.log10(ratios[valid_ratio])
            histogram.hist(log_ratios, bins=35, color="#4c78a8", alpha=0.85)
            histogram.axvline(0, color="black", linestyle="--")
            histogram.set_xlabel(r"$\log_{10}(D/C)$")
            histogram.set_ylabel("authors")
            histogram.set_title(
                f"{feature}: {int(np.sum(ratios[valid_ratio] > 1))}/"
                f"{int(valid_ratio.sum())} authors have D/C > 1"
            )

        figure.suptitle(f"Full Gutenberg D/C diagnostic — {variant.replace('_', ' ')}")
        figure.tight_layout()
        figure.savefig(out_dir / f"dc_diagnostic_{variant}.png", dpi=180)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.block_size < 1:
        raise ValueError("--block-size must be positive")

    input_path = resolved(args.input)
    out_dir = resolved(args.out)
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist; download punctuation_stylometry.p from "
            f"{SOURCE_DOI}"
        )

    print(f"Verifying {input_path} ...", flush=True)
    source_md5 = file_md5(input_path)
    if not args.skip_checksum and source_md5 != EXPECTED_MD5:
        raise ValueError(
            f"archive MD5 is {source_md5}, expected Zenodo MD5 {EXPECTED_MD5}"
        )

    print("Loading the trusted Zenodo pickle (this can use substantial memory) ...", flush=True)
    frame = pd.read_pickle(input_path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(frame).__name__}")
    print(f"Loaded {len(frame):,} rows and {len(frame.columns):,} columns", flush=True)

    by_author = extract_documents(frame)
    del frame
    gc.collect()

    n_documents = sum(len(documents) for documents in by_author.values())
    n_authors = len(by_author)
    if (n_documents, n_authors) != (EXPECTED_DOCUMENTS, EXPECTED_AUTHORS):
        raise ValueError(
            f"archive resolved to {n_documents:,} documents and {n_authors:,} authors; "
            f"expected {EXPECTED_DOCUMENTS:,} and {EXPECTED_AUTHORS:,}"
        )
    if any(len(documents) < 2 for documents in by_author.values()):
        raise ValueError("C is undefined: at least one author has fewer than two documents")

    deduplicated, duplicate_rows_removed = deduplicate_documents(by_author)
    variants: dict[str, dict[str, list[Document]]] = {}
    if args.variant in ("all", "both"):
        variants["all_documents"] = by_author
    if args.variant in ("deduplicated", "both"):
        variants["deduplicated_within_author"] = deduplicated

    rows: list[dict[str, object]] = []
    for variant, documents in variants.items():
        print(f"\nAnalysing {variant} ...", flush=True)
        rows.extend(analyse_variant(variant, documents, args.block_size))

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "dc_by_author.csv", index=False)
    summary = make_summary(
        rows,
        variants,
        duplicate_rows_removed=duplicate_rows_removed,
        source_md5=source_md5,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not args.no_plots:
        make_plots(rows, out_dir)

    print("\nSummary", flush=True)
    for row in summary["summaries"]:
        print(
            f"  {row['corpus_variant']:28s} {row['feature']}: "
            f"{row['n_ratio_gt_1']}/{row['n_authors']} authors have D/C > 1; "
            f"median={row['ratio_median']:.3f}",
            flush=True,
        )
    print(f"Wrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
