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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="cache/punct_sequences.json")
    p.add_argument(
        "--human-glob",
        nargs="+",
        default=["gutenberg_texts/*.txt", "full_books/*.txt"],
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


def main() -> None:
    args = ARGS
    strip = not args.no_strip
    cache: dict[str, list[str]] = {}

    targets: list[Path] = []
    for pattern in args.human_glob:
        targets.extend(sorted(ROOT.glob(pattern)))
    for run_dir in args.run_dir:
        targets.extend(sorted((ROOT / run_dir).glob("*/run_*.txt")))

    for i, path in enumerate(targets, 1):
        rel = str(path.relative_to(ROOT))
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
