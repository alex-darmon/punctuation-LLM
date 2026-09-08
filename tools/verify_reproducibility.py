#!/usr/bin/env python3
"""Verify the complete no-API evidence chain for the punctuation study.

The verifier does not generate or alter artifacts. It checks the versioned
human sources and model outputs, derived assembly/cache manifests, canonical
results, and optional non-destructive reruns against the committed golden
outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
PANEL_CONFIG = ROOT / "campaigns" / "author_panel_20.json"
SOURCE_MANIFEST = ROOT / "results" / "author_panel_20" / "source_manifest.json"
INFERENCE_CONFIG = ROOT / "campaigns" / "inference_v2.json"
INFERENCE_RESULTS = ROOT / "results" / "author_panel_20" / "inference_v2"
FULL_GRID_RESULTS = ROOT / "results" / "author_panel_20" / "full_grid"
FROZEN_RESULTS = ROOT / "results" / "frozen"
GENERATION_CONFIGS = (
    ROOT / "campaigns" / "generation_campaign_new10_flash.json",
    ROOT / "campaigns" / "generation_campaign_new10_pro.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-inference",
        help="Compare a rerun directory with canonical inference-v2 hashes.",
    )
    parser.add_argument(
        "--candidate-full-grid",
        help="Byte-compare a rerun directory with canonical full-grid CSVs.",
    )
    parser.add_argument(
        "--candidate-frozen",
        help="Byte-compare a rerun directory with canonical frozen CSVs.",
    )
    parser.add_argument(
        "--require-git-tracked",
        action="store_true",
        help="Also require every canonical non-derived artifact to be in Git.",
    )
    return parser.parse_args()


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(files: Iterable[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes = 0

    def require(self, condition: bool, message: str) -> None:
        if condition:
            self.passes += 1
        else:
            self.failures.append(message)

    def same_hash(self, path: Path, expected: str, label: str) -> None:
        self.require(path.is_file(), f"{label} is missing: {path}")
        if path.is_file():
            self.require(sha256(path) == expected, f"{label} hash mismatch: {path}")


def verify_panel(checks: Checks) -> list[Path]:
    panel = read_json(PANEL_CONFIG)
    manifest = read_json(SOURCE_MANIFEST)
    checks.require(len(panel["authors"]) == 20, "panel must contain 20 authors")
    checks.require(
        {entry["cohort"] for entry in panel["authors"]} == {"existing", "new"},
        "panel must retain existing/new cohorts",
    )
    checks.require(
        all(len(entry["panel_books"]) == 3 for entry in panel["authors"]),
        "every panel author must have three books",
    )
    checks.same_hash(
        PANEL_CONFIG,
        manifest["panel_config_sha256"],
        "source-manifest panel config",
    )
    books = manifest["books"]
    checks.require(len(books) == 60, "source manifest must contain 60 books")
    paths: list[Path] = []
    for book in books:
        path = resolve(book["path"])
        paths.append(path)
        checks.same_hash(path, book["sha256"], f"human source {book['path']}")
        if path.is_file():
            checks.require(
                path.stat().st_size == int(book["bytes"]),
                f"human source byte count mismatch: {book['path']}",
            )
    return paths


def verify_generation_sources(checks: Checks) -> list[Path]:
    tracked_candidates: list[Path] = []
    for config_path in GENERATION_CONFIGS:
        config = read_json(config_path)
        output = resolve(config["output_dir"])
        n_runs = int(config["default_runs"])
        expected = {
            (author["key"], run_id)
            for author in config["authors"]
            for run_id in range(1, n_runs + 1)
        }
        observed: set[tuple[str, int]] = set()
        summary_path = output / "all_runs_summary.json"
        metadata_path = output / "campaign_metadata.json"
        checks.require(summary_path.is_file(), f"missing generation summary: {summary_path}")
        checks.require(metadata_path.is_file(), f"missing campaign metadata: {metadata_path}")
        tracked_candidates.extend((config_path, summary_path, metadata_path))
        if summary_path.is_file():
            for row in read_json(summary_path):
                key = (row["author_key"], int(row["run_id"]))
                observed.add(key)
                source = str(row["source_book_path"])
                checks.require(
                    not Path(source).is_absolute(),
                    f"generation summary contains an absolute source path: {source}",
                )
            checks.require(
                observed == expected,
                f"generation summary run inventory mismatch: {output.name}",
            )
        for author, run_id in sorted(expected):
            processed = output / author / f"run_{run_id:02d}.txt"
            tracked_candidates.append(processed)
            checks.require(
                processed.is_file() and processed.stat().st_size > 0,
                f"missing processed generation: {processed}",
            )
            if config.get("save_raw_outputs"):
                raw = output / author / "raw" / f"run_{run_id:02d}.txt"
                tracked_candidates.append(raw)
                checks.require(
                    raw.is_file() and raw.stat().st_size > 0,
                    f"missing raw generation: {raw}",
                )
    return tracked_candidates


def verify_assemblies(checks: Checks) -> None:
    inference = read_json(INFERENCE_CONFIG)
    for condition, directory in inference["conditions"].items():
        assembled = resolve(directory)
        summary_path = assembled / "all_runs_summary.json"
        manifest_path = assembled / "condition_manifest.json"
        checks.require(manifest_path.is_file(), f"missing {condition} condition manifest")
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        checks.same_hash(
            summary_path,
            manifest["all_runs_summary_sha256"],
            f"{condition} assembled summary",
        )
        processed = list(assembled.glob("*/run_*.txt"))
        raw = list(assembled.glob("*/raw/run_*.txt"))
        checks.require(
            len(processed) == int(manifest["processed_runs"]),
            f"{condition} processed run count mismatch",
        )
        checks.require(
            len(raw) == int(manifest["raw_runs"]),
            f"{condition} raw run count mismatch",
        )
        checks.require(
            tree_sha256(processed, assembled) == manifest["processed_text_tree_sha256"],
            f"{condition} processed text-tree hash mismatch",
        )
        checks.require(
            tree_sha256(raw, assembled) == manifest["raw_text_tree_sha256"],
            f"{condition} raw text-tree hash mismatch",
        )


def verify_inference(checks: Checks, candidate: Path | None) -> None:
    config = read_json(INFERENCE_CONFIG)
    manifest_path = INFERENCE_RESULTS / "inference_manifest.json"
    manifest = read_json(manifest_path)
    checks.same_hash(
        INFERENCE_CONFIG,
        manifest["analysis_config_sha256"],
        "inference config",
    )
    checks.same_hash(
        resolve(config["authors_config"]),
        manifest["authors_config_sha256"],
        "inference author config",
    )
    checks.same_hash(
        resolve(config["cache"]),
        manifest["cache_sha256"],
        "inference cache",
    )
    for relative, expected in manifest["upstream_result_sha256"].items():
        checks.same_hash(resolve(relative), expected, f"upstream result {relative}")
    for relative, expected in manifest["source_code_sha256"].items():
        checks.same_hash(resolve(relative), expected, f"source code {relative}")
    for filename, expected in manifest["output_sha256"].items():
        checks.same_hash(INFERENCE_RESULTS / filename, expected, f"inference {filename}")
        if candidate is not None:
            checks.same_hash(candidate / filename, expected, f"candidate {filename}")
    for distribution, expected in manifest["environment"]["packages"].items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        checks.require(
            actual == expected,
            f"package version mismatch for {distribution}: {actual} != {expected}",
        )

    with (INFERENCE_RESULTS / "attribution_clustered.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    primary = {
        (row["condition"] or "human"): (
            int(row["n_correct"]),
            int(row["n_observations"]),
        )
        for row in rows
        if row["row_type"] == "summary"
        and row["author_set"] == "all_authors"
        and row["feature"] == "f3"
        and row["chunk_size"] == "2000"
    }
    checks.require(
        primary == {"human": (372, 485), "flash": (35, 400), "pro": (16, 200)},
        f"primary attribution counts changed: {primary}",
    )


def compare_csv_directories(checks: Checks, canonical: Path, candidate: Path) -> None:
    canonical_files = sorted(canonical.glob("*.csv"))
    checks.require(bool(canonical_files), f"no canonical CSVs found in {canonical}")
    for source in canonical_files:
        other = candidate / source.name
        checks.same_hash(other, sha256(source), f"rerun comparison {source.name}")


def verify_paper(checks: Checks) -> list[Path]:
    main = ROOT / "paper" / "main.tex"
    checks.require(main.is_file(), "tracked v2 paper/main.tex is missing")
    if not main.is_file():
        return []
    content = main.read_text(encoding="utf-8")
    for filename in (
        "primary_attribution",
        "detection",
        "separability",
        "drift",
    ):
        checks.require(
            f"tables/inference_v2/{filename}" in content,
            f"paper does not include v2 table {filename}",
        )
    for filename in (
        "crossfit_separability",
        "attribution_by_author",
        "detection_oof_roc",
        "g_calibration",
        "positional_drift",
    ):
        checks.require(
            f"figures/inference_v2/{filename}" in content,
            f"paper does not include v2 figure {filename}",
        )
    return [main]


def require_git_tracked(checks: Checks, paths: Iterable[Path]) -> None:
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        checks.require(result.returncode == 0, f"canonical artifact is untracked: {path}")


def main() -> None:
    args = parse_args()
    checks = Checks()
    tracked = verify_panel(checks)
    tracked.extend(verify_generation_sources(checks))
    verify_assemblies(checks)
    verify_inference(
        checks,
        resolve(args.candidate_inference) if args.candidate_inference else None,
    )
    tracked.extend(verify_paper(checks))
    tracked.extend(
        [
            PANEL_CONFIG,
            SOURCE_MANIFEST,
            INFERENCE_CONFIG,
            *GENERATION_CONFIGS,
            *INFERENCE_RESULTS.rglob("*"),
            *FULL_GRID_RESULTS.rglob("*"),
        ]
    )
    if args.candidate_full_grid:
        compare_csv_directories(
            checks,
            FULL_GRID_RESULTS,
            resolve(args.candidate_full_grid),
        )
    if args.candidate_frozen:
        compare_csv_directories(
            checks,
            FROZEN_RESULTS,
            resolve(args.candidate_frozen),
        )
    if args.require_git_tracked:
        require_git_tracked(checks, tracked)

    if checks.failures:
        print(f"{checks.passes} checks passed; {len(checks.failures)} failed:")
        for failure in checks.failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print(f"All {checks.passes} reproducibility checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"reproducibility verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
