#!/usr/bin/env python3
"""Summarize the balanced 20-author human preflight and enforce its gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="campaigns/author_panel_20.json", help="Panel config."
    )
    parser.add_argument(
        "--results",
        default="results/author_panel_20/preflight",
        help="Human-only grid output directory.",
    )
    return parser.parse_args()


def resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(rows: list[dict[str, str]], **wanted: object) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key, "")) == str(value) for key, value in wanted.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {wanted}, found {len(matches)}")
    return matches[0]


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    results_dir = resolve(args.results)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    design = config["panel_design"]
    feature = design["primary_feature"]
    chunk_size = int(design["primary_chunk_size"])
    gate = design["admission_gate"]
    authors = config["authors"]
    new_keys = [author["key"] for author in authors if author["cohort"] == "new"]

    attribution = read_csv(results_dir / "attribution.csv")
    separability = read_csv(results_dir / "separability.csv")
    document_all = one(
        attribution,
        experiment="human_attribution",
        author_set="all_authors",
        unit="document",
        feature=feature,
        author="ALL",
    )
    chunk_all = one(
        attribution,
        experiment="human_attribution",
        author_set="all_authors",
        unit="chunk",
        chunk_size=chunk_size,
        feature=feature,
        author="ALL",
    )
    per_author_chunks = [
        one(
            attribution,
            experiment="human_attribution",
            author_set="all_authors",
            unit="chunk",
            chunk_size=chunk_size,
            feature=feature,
            author=author["key"],
        )
        for author in authors
    ]
    macro_chunk_accuracy = statistics.fmean(
        float(row["accuracy_pct"]) for row in per_author_chunks
    )

    gate_rows: list[dict] = []
    for key in new_keys:
        document = one(
            attribution,
            experiment="human_attribution",
            author_set="all_authors",
            unit="document",
            feature=feature,
            author=key,
        )
        chunk = one(
            attribution,
            experiment="human_attribution",
            author_set="all_authors",
            unit="chunk",
            chunk_size=chunk_size,
            feature=feature,
            author=key,
        )
        dc = one(separability, feature=feature, author=key)
        correct = int(document["correct"])
        gate_rows.append(
            {
                "author": key,
                "document_correct": correct,
                "document_n": int(document["n"]),
                "document_accuracy_pct": float(document["accuracy_pct"]),
                "chunk_size": chunk_size,
                "chunk_correct": int(chunk["correct"]),
                "chunk_n": int(chunk["n"]),
                "chunk_accuracy_pct": float(chunk["accuracy_pct"]),
                "matched_panel_D_over_C": float(dc["margin_D_over_C"]),
                "passes_gate": correct >= int(gate["minimum_correct"]),
            }
        )

    passing = sum(row["passes_gate"] for row in gate_rows)
    gate_passed = passing >= int(gate["required_new_authors_passing"])
    summary = {
        "panel_config": str(config_path.relative_to(ROOT)),
        "results_dir": str(results_dir.relative_to(ROOT)),
        "input_sha256": {
            "panel_config": sha256(config_path),
            "source_manifest": sha256(resolve(config["source_manifest"])),
            "analysis_manifest": sha256(results_dir / "manifest.json"),
            "attribution": sha256(results_dir / "attribution.csv"),
            "separability": sha256(results_dir / "separability.csv"),
        },
        "authors": len(authors),
        "books": len(authors) * int(design["books_per_author"]),
        "feature": feature,
        "chance_accuracy_pct": 100.0 * float(design["chance_accuracy"]),
        "whole_book": {
            "correct": int(document_all["correct"]),
            "n": int(document_all["n"]),
            "accuracy_pct": float(document_all["accuracy_pct"]),
        },
        "chunk": {
            "size": chunk_size,
            "correct": int(chunk_all["correct"]),
            "n": int(chunk_all["n"]),
            "micro_accuracy_pct": float(chunk_all["accuracy_pct"]),
            "macro_author_accuracy_pct": macro_chunk_accuracy,
        },
        "gate": {
            "minimum_correct_per_new_author": int(gate["minimum_correct"]),
            "required_new_authors_passing": int(
                gate["required_new_authors_passing"]
            ),
            "new_authors_passing": passing,
            "passed": gate_passed,
        },
        "new_authors": gate_rows,
    }
    (results_dir / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (results_dir / "new_author_gate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gate_rows[0]))
        writer.writeheader()
        writer.writerows(gate_rows)

    print(
        f"whole-book {feature}: {summary['whole_book']['accuracy_pct']:.1f}% "
        f"({summary['whole_book']['correct']}/{summary['whole_book']['n']}; "
        f"chance {summary['chance_accuracy_pct']:.1f}%)"
    )
    print(
        f"{chunk_size}-mark {feature}: "
        f"micro {summary['chunk']['micro_accuracy_pct']:.1f}%, "
        f"macro-author {summary['chunk']['macro_author_accuracy_pct']:.1f}%"
    )
    print(f"new-author gate: {passing}/{len(new_keys)} pass")
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
