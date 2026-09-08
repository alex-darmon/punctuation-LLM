#!/usr/bin/env python3
"""
Parse every human source text and LLM run once, and cache the punctuation
sequences to a single JSON file.

Parsing is by far the slowest part of the pipeline (~0.7 s per book), so every
downstream experiment re-paying that cost is what makes sweeps painful. Cache
once, then all analyses read from here.

Usage:
  python tools/build_punct_cache.py --out cache/punct_sequences.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HUMAN_GLOBS = ["gutenberg_texts/*.txt", "full_books/*.txt"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="cache/punct_sequences.json")
    p.add_argument(
        "--human-glob",
        nargs="+",
        default=None,
    )
    p.add_argument(
        "--authors-config",
        default=None,
        help=(
            "Parse only the human book paths listed in this campaign/panel JSON. "
            "This is mutually exclusive with --human-glob."
        ),
    )
    p.add_argument(
        "--source-manifest",
        default=None,
        help=(
            "Parse the exact paths in a JSON manifest's 'paths' list. "
            "Mutually exclusive with --authors-config and --human-glob."
        ),
    )
    p.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Generated-text directory to cache (repeatable).",
    )
    p.add_argument("--no-strip", action="store_true", help="Keep Gutenberg boilerplate.")
    return p.parse_args()


ARGS = parse_args()

# punctuation.config reads sys.argv at import time, so it has to be rewritten
# only after our own arguments have been consumed.
sys.path.insert(0, str(ROOT / "punctuation-stylometry-master"))
_CONFIG = str(ROOT / "punctuation-stylometry-master" / "conf" / "punctuation.ini")
sys.argv = [sys.argv[0], "-c", _CONFIG]

from punctuation.parser.punctuation_parser import get_textinfo, seq_pun_only  # noqa: E402

GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", re.IGNORECASE
)
GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", re.IGNORECASE
)
# Files whose header/footer boilerplate is present without the *** markers.
LEGAL_PREAMBLE = re.compile(
    r"This eBook is for the use of anyone anywhere.*?(?:\n\n|\r\n\r\n)", re.IGNORECASE | re.DOTALL
)


def strip_boilerplate(text: str) -> str:
    start = GUTENBERG_START.search(text)
    if start:
        text = text[start.end() :]
    end = GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    text = LEGAL_PREAMBLE.sub("", text, count=1)
    return text


def punct_sequence(path: Path, strip: bool) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if strip:
        text = strip_boilerplate(text)
    text = text.replace("...", "^")
    seq = seq_pun_only(get_textinfo(text))
    return list(seq) if seq else []


def configured_human_paths(config_path: Path) -> list[Path]:
    """Resolve every human source path named by an author configuration."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    authors = config.get("authors")
    if not isinstance(authors, list) or not authors:
        raise ValueError(f"{config_path} must contain a non-empty 'authors' list")

    paths: list[Path] = []
    for entry in authors:
        raw_paths = entry.get("book_paths")
        if raw_paths is None and "book_path" in entry:
            raw_paths = [entry["book_path"]]
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError(
                f"author {entry.get('key', '<unknown>')!r} has no book paths"
            )
        for raw_path in raw_paths:
            path = Path(raw_path)
            paths.append(path if path.is_absolute() else ROOT / path)
    return paths


def manifested_human_paths(manifest_path: Path) -> list[Path]:
    """Resolve an immutable human-source inventory."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_paths = manifest.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(f"{manifest_path} must contain a non-empty 'paths' list")
    paths = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        paths.append(path if path.is_absolute() else ROOT / path)
    return paths


def main() -> None:
    args = ARGS
    strip = not args.no_strip
    cache: dict[str, list[str]] = {}

    targets: list[Path] = []
    selectors = sum(
        bool(value)
        for value in (args.authors_config, args.source_manifest, args.human_glob)
    )
    if selectors > 1:
        raise ValueError(
            "--authors-config, --source-manifest, and --human-glob are "
            "mutually exclusive"
        )
    if args.authors_config:
        config_path = Path(args.authors_config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        targets.extend(configured_human_paths(config_path))
    elif args.source_manifest:
        manifest_path = Path(args.source_manifest)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        targets.extend(manifested_human_paths(manifest_path))
    else:
        for pattern in args.human_glob or DEFAULT_HUMAN_GLOBS:
            targets.extend(sorted(ROOT.glob(pattern)))
    for run_dir in args.run_dir:
        targets.extend(sorted((ROOT / run_dir).glob("*/run_*.txt")))

    # Configs may intentionally list two aliases for one book. Parse each path
    # once while retaining both cache keys for content-identity deduplication.
    targets = list(dict.fromkeys(targets))
    missing = [path for path in targets if not path.is_file()]
    if missing:
        detail = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"configured source files are missing:\n{detail}")

    for i, path in enumerate(targets, 1):
        try:
            rel = str(path.relative_to(ROOT))
        except ValueError:
            rel = str(path)
        seq = punct_sequence(path, strip=strip)
        cache[rel] = seq
        print(f"[{i}/{len(targets)}] {rel}: {len(seq)} marks")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stripped_boilerplate": strip, "sequences": cache}, f)
    print(f"\n[wrote] {out_path} ({len(cache)} files)")


if __name__ == "__main__":
    main()
