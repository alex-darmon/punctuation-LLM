#!/usr/bin/env python3
"""
Leave-one-book-out author attribution: the true human baseline for the LLM
comparison.

For every book by every author, the book is held out entirely, that author's
reference profile is rebuilt from their remaining books only, and every
non-overlapping chunk of the held-out book is assigned to the nearest of the 10
author references by KL divergence.

This removes the leakage present in every earlier version of the check, where
the text being scored was part of the reference it was scored against. It is the
matched control for the LLM runs, which are scored against references built from
books the model never produced.

Two units of analysis are reported:
  chunk-level    : each `--chunk-sizes` window of the held-out book, one vote each.
  document-level : the whole held-out book as a single document. This is the unit
                   used in table 2 of Darmon et al., where 10-author KL
                   classification scored f1 = 0.69 and f3 = 0.74.

Usage:
  python tools/loo_book_attribution.py \
      --authors-config campaigns/generation_campaign_phaseA_two_samples.json \
      --chunk-sizes 1000 2000 4000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--authors-config", required=True)
    p.add_argument("--cache", default="cache/punct_sequences.json")
    p.add_argument("--chunk-sizes", nargs="+", type=int, default=[2000])
    p.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Author keys to drop (e.g. authors whose reference is unreliable).",
    )
    p.add_argument("--out", default=None)
    return p.parse_args()


ARGS = parse_args()

sys.path.insert(0, str(ROOT / "punctuation-stylometry-master"))
_CONFIG = str(ROOT / "punctuation-stylometry-master" / "conf" / "punctuation.ini")
sys.argv = [sys.argv[0], "-c", _CONFIG]

from punctuation.config import options  # noqa: E402
from punctuation.feature_operations.distances import d_KL  # noqa: E402
from punctuation.feature_operations.matrix_operations import (  # noqa: E402
    normalised_transition_mat,
    transition_mat,
)
from punctuation.parser.punctuation_parser import get_frequencies  # noqa: E402

PUNCT_VECTOR = options.punctuation_vector
FEATURES = ("f1", "f3")


def features(seq: list[str]) -> dict[str, list[float]] | None:
    if not seq:
        return None
    f1 = get_frequencies(seq, vector=PUNCT_VECTOR)
    if f1 is None or sum(f1) == 0:
        return None
    f2 = transition_mat(seq)
    if f2 is None:
        return None
    return {"f1": f1, "f3": normalised_transition_mat(f2, f1).flatten().tolist()}


def dedupe(rels: list[str], cache: dict[str, list[str]]) -> list[str]:
    seen: set[tuple[int, str]] = set()
    out = []
    for rel in rels:
        seq = cache.get(rel, [])
        sig = (len(seq), "".join(seq[:200]))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(rel)
    return out


def main() -> None:
    cache = json.load(open(ROOT / ARGS.cache))["sequences"]
    cfg = json.load(open(ROOT / ARGS.authors_config))

    author_books = {
        e["key"]: dedupe(list(e.get("book_paths") or [e["book_path"]]), cache)
        for e in cfg["authors"]
        if e["key"] not in ARGS.exclude
    }
    authors = list(author_books)
    chance = 100.0 / len(authors)

    # Full-corpus reference per author, and the leave-one-book-out variants.
    ref_full: dict[str, dict[str, list[float]] | None] = {}
    ref_loo: dict[tuple[str, str], dict[str, list[float]] | None] = {}
    for author, rels in author_books.items():
        ref_full[author] = features([m for rel in rels for m in cache[rel]])
        for rel in rels:
            rest = [m for r2 in rels if r2 != rel for m in cache[r2]]
            ref_loo[(author, rel)] = features(rest) if rest else None

    print(f"\nauthors={len(authors)}  chance={chance:.1f}%")
    for author, rels in author_books.items():
        held = [
            f"{Path(r).name}({len(cache[r])})" for r in rels
        ]
        print(f"  {author:24s} {len(rels)} books: {' '.join(held)}")

    csv_rows: list[dict[str, object]] = []

    # ---------------- document level (matches Darmon et al. table 2) ----------
    print(f"\n{'='*80}\ndocument-level leave-one-book-out")
    print("  (Darmon et al. table 2, 10 authors: f1 = 0.69, f3 = 0.74, baseline 0.21)")
    for feature in FEATURES:
        hits = 0
        total = 0
        confusion: dict[str, int] = {}
        for author, rels in author_books.items():
            for rel in rels:
                feats = features(cache[rel])
                if feats is None or ref_loo[(author, rel)] is None:
                    continue
                kl = {}
                for target in authors:
                    ref = ref_loo[(author, rel)] if target == author else ref_full[target]
                    if ref is None:
                        continue
                    kl[target] = float(d_KL(feats[feature], ref[feature]))
                best = min(kl, key=kl.get)
                total += 1
                confusion[best] = confusion.get(best, 0) + 1
                if best == author:
                    hits += 1
        acc = 100 * hits / max(total, 1)
        top = sorted(confusion.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {feature}: {hits}/{total} = {acc:.1f}%   most-predicted: {top}")
        csv_rows.append(
            {
                "unit": "document",
                "chunk_size": "",
                "feature": feature,
                "n": total,
                "correct": hits,
                "accuracy_pct": acc,
                "chance_pct": chance,
            }
        )

    # ---------------- chunk level -------------------------------------------
    for chunk_size in ARGS.chunk_sizes:
        print(f"\n{'='*80}\nchunk-level leave-one-book-out, chunk_size={chunk_size}")
        for feature in FEATURES:
            hits = 0
            total = 0
            per_author: dict[str, list[int]] = {a: [0, 0] for a in authors}
            confusion: dict[str, int] = {}

            for author, rels in author_books.items():
                ref_a = ref_loo
                for rel in rels:
                    if ref_a[(author, rel)] is None:
                        continue
                    seq = cache[rel]
                    n_chunks = len(seq) // chunk_size
                    for i in range(n_chunks):
                        feats = features(seq[i * chunk_size : (i + 1) * chunk_size])
                        if feats is None:
                            continue
                        kl = {}
                        for target in authors:
                            ref = (
                                ref_a[(author, rel)]
                                if target == author
                                else ref_full[target]
                            )
                            if ref is None:
                                continue
                            kl[target] = float(d_KL(feats[feature], ref[feature]))
                        best = min(kl, key=kl.get)
                        total += 1
                        per_author[author][1] += 1
                        confusion[best] = confusion.get(best, 0) + 1
                        if best == author:
                            hits += 1
                            per_author[author][0] += 1

            acc = 100 * hits / max(total, 1)
            print(
                f"\n  {feature}: {hits}/{total} = {acc:.1f}%  (chance {chance:.1f}%)"
            )
            top = sorted(confusion.items(), key=lambda kv: -kv[1])[:3]
            print(f"      most-predicted: {top}")
            for author in authors:
                c, n = per_author[author]
                if n:
                    print(
                        f"        {author:24s} {c:3d}/{n:3d} = {100*c/n:5.1f}%"
                    )
                    csv_rows.append(
                        {
                            "unit": "chunk",
                            "chunk_size": chunk_size,
                            "feature": feature,
                            "author": author,
                            "n": n,
                            "correct": c,
                            "accuracy_pct": 100 * c / n,
                            "chance_pct": chance,
                        }
                    )
            csv_rows.append(
                {
                    "unit": "chunk",
                    "chunk_size": chunk_size,
                    "feature": feature,
                    "author": "ALL",
                    "n": total,
                    "correct": hits,
                    "accuracy_pct": acc,
                    "chance_pct": chance,
                }
            )

    if ARGS.out and csv_rows:
        import csv

        fields: list[str] = []
        for row in csv_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        out_path = ROOT / ARGS.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
