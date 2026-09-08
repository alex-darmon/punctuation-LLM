#!/usr/bin/env python3
"""Assemble disjoint generation campaigns into one analysis condition.

The frozen ten-author outputs stay untouched. This tool hard-links (or copies,
when hard links are unavailable) their run files together with the new-author
outputs and merges the per-run metadata expected by ``Corpus.load_runs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Source run dir.")
    parser.add_argument("--out", required=True, help="New assembled condition directory.")
    parser.add_argument(
        "--panel-config",
        default="campaigns/author_panel_20.json",
        help="Config whose author keys must be covered exactly once.",
    )
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help=(
            "Verify an existing assembly against its sources and refresh only "
            "portable summaries/manifests; never replace run text."
        ),
    )
    return parser.parse_args()


def resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_metadata_path(path_like: str | Path) -> str:
    """Normalise historical absolute paths into repository-relative paths."""
    path = Path(path_like)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        text = str(path_like).replace("\\", "/")
        marker = "punctuation-LLM/"
        return text.split(marker, 1)[1] if marker in text else text


def tree_sha256(files: list[tuple[Path, Path]], output: Path) -> str:
    digest = hashlib.sha256()
    for source, destination in sorted(files, key=lambda item: str(item[1])):
        relative = destination.relative_to(output).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(source)))
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    args = parse_args()
    sources = [resolve(path) for path in args.input]
    output = resolve(args.out)
    panel_path = resolve(args.panel_config)

    if output.exists() and not args.refresh_metadata:
        raise FileExistsError(
            f"refusing to modify existing output {output}; choose a new --out path"
        )
    if not output.exists() and args.refresh_metadata:
        raise FileNotFoundError(
            f"cannot refresh missing assembly {output}; omit --refresh-metadata"
        )
    missing = [path for path in sources if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing input directories: {missing}")

    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    expected_authors = {author["key"] for author in panel["authors"]}
    owner: dict[str, Path] = {}
    run_files: list[tuple[Path, Path]] = []
    raw_files: list[tuple[Path, Path]] = []
    summaries: list[dict] = []

    for source in sources:
        summary_path = source / "all_runs_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing run summary: {summary_path}")
        summaries.extend(json.loads(summary_path.read_text(encoding="utf-8")))

        for author_dir in sorted(path for path in source.iterdir() if path.is_dir()):
            processed = sorted(author_dir.glob("run_*.txt"))
            if not processed:
                continue
            author = author_dir.name
            if author in owner:
                raise ValueError(
                    f"author {author!r} occurs in both {owner[author]} and {source}"
                )
            owner[author] = source
            run_files.extend(
                (path, output / author / path.name) for path in processed
            )
            raw_dir = author_dir / "raw"
            if raw_dir.is_dir():
                raw_files.extend(
                    (path, output / author / "raw" / path.name)
                    for path in sorted(raw_dir.glob("run_*.txt"))
                )

    observed_authors = set(owner)
    if observed_authors != expected_authors:
        missing_authors = sorted(expected_authors - observed_authors)
        extra_authors = sorted(observed_authors - expected_authors)
        raise ValueError(
            "assembled condition does not match the panel: "
            f"missing={missing_authors}, extra={extra_authors}"
        )

    summary_keys = [
        (row["author_key"], int(row["run_id"])) for row in summaries
    ]
    if len(summary_keys) != len(set(summary_keys)):
        raise ValueError("input summaries contain duplicate author/run identifiers")
    if len(summary_keys) != len(run_files):
        raise ValueError(
            f"summaries describe {len(summary_keys)} runs but {len(run_files)} "
            "processed run files were found"
        )

    link_modes = {"hardlink": 0, "copy": 0, "existing_verified": 0}
    try:
        for source, destination in run_files + raw_files:
            if args.refresh_metadata:
                if not destination.is_file() or sha256(source) != sha256(destination):
                    raise ValueError(
                        f"existing assembly differs from source: {destination}"
                    )
                link_modes["existing_verified"] += 1
            else:
                link_modes[link_or_copy(source, destination)] += 1

        summaries.sort(key=lambda row: (row["author_key"], int(row["run_id"])))
        for row in summaries:
            author = row["author_key"]
            run_id = int(row["run_id"])
            processed_path = output / author / f"run_{run_id:02d}.txt"
            raw_path = output / author / "raw" / f"run_{run_id:02d}.txt"
            row["source_book_path"] = portable_metadata_path(
                row["source_book_path"]
            )
            row["processed_output_path"] = portable_metadata_path(processed_path)
            row["processed_sha256"] = sha256(processed_path)
            row["raw_output_available"] = raw_path.is_file()
            row["raw_output_path"] = (
                portable_metadata_path(raw_path) if raw_path.is_file() else None
            )
            row["raw_sha256"] = sha256(raw_path) if raw_path.is_file() else None
        summary_path = output / "all_runs_summary.json"
        summary_path.write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
        )
        metadata = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "panel_config": str(panel_path.relative_to(ROOT)),
            "sources": [str(path.relative_to(ROOT)) for path in sources],
            "authors": sorted(observed_authors),
            "processed_runs": len(run_files),
            "raw_runs": len(raw_files),
            "file_modes": link_modes,
        }
        (output / "assembly_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        condition_manifest = {
            "schema_version": 1,
            "panel_config": str(panel_path.relative_to(ROOT)),
            "panel_config_sha256": sha256(panel_path),
            "sources": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "summary_sha256": sha256(path / "all_runs_summary.json"),
                }
                for path in sources
            ],
            "authors": sorted(observed_authors),
            "processed_runs": len(run_files),
            "raw_runs": len(raw_files),
            "all_runs_summary_sha256": sha256(summary_path),
            "processed_text_tree_sha256": tree_sha256(run_files, output),
            "raw_text_tree_sha256": tree_sha256(raw_files, output),
        }
        (output / "condition_manifest.json").write_text(
            json.dumps(condition_manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        if not args.refresh_metadata:
            shutil.rmtree(output, ignore_errors=True)
        raise

    print(
        f"[assembled] {output.relative_to(ROOT)}: {len(observed_authors)} authors, "
        f"{len(run_files)} processed runs, {len(raw_files)} raw runs"
    )


if __name__ == "__main__":
    main()
