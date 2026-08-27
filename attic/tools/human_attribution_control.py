#!/usr/bin/env python3
"""
Control experiment: can held-out HUMAN chunks be attributed to their own author
with the same nearest-reference rule applied to LLM runs?

This is the missing baseline for the "is LLM-Author_x nearest to Author_x?"
check. Without it, a low LLM attribution rate is uninterpretable: it could mean
the imitation failed, or it could mean the feature space cannot separate these
authors at this chunk size.

Three reference regimes are reported:
  same_book : reference includes the book the held-out chunk came from, chunk
              included. Inflated by leakage; kept only for comparability with
              earlier tables.
  minus_chunk : reference includes the source book but with the held-out chunk
              removed. This is the matched control for the LLM setup, where the
              prompt excerpt comes from a book that is in the reference but the
              generated text itself is not.
  loo_book  : reference excludes the held-out chunk's book entirely.
              Harder, and the honest measure of author-level separability.

Usage:
  python tools/human_attribution_control.py \
      --authors-config campaigns/generation_campaign_phaseA_two_samples.json \
      --chunk-sizes 500 1000 2000 4000
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
    p.add_argument("--chunk-sizes", nargs="+", type=int, default=[2000])
    p.add_argument("--cache", default="cache/punct_sequences.json")
    p.add_argument("--max-chunks-per-book", type=int, default=40)
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
    }
    authors = list(author_books)

    for chunk_size in ARGS.chunk_sizes:
        # Held-out chunks: non-overlapping windows within each book.
        held_out: list[tuple[str, str, int, dict[str, list[float]]]] = []
        for author, rels in author_books.items():
            for rel in rels:
                seq = cache[rel]
                n_chunks = min(len(seq) // chunk_size, ARGS.max_chunks_per_book)
                for i in range(n_chunks):
                    feats = features(seq[i * chunk_size : (i + 1) * chunk_size])
                    if feats:
                        held_out.append((author, rel, i, feats))

        # Reference profiles from pooled full text, and the per-book pieces
        # needed to rebuild a leave-one-book-out reference.
        pooled_seq = {
            a: [m for rel in rels for m in cache[rel]] for a, rels in author_books.items()
        }
        ref_same = {a: features(s) for a, s in pooled_seq.items()}
        ref_loo: dict[tuple[str, str], dict[str, list[float]] | None] = {}
        for author, rels in author_books.items():
            for rel in rels:
                rest = [m for r2 in rels if r2 != rel for m in cache[r2]]
                ref_loo[(author, rel)] = features(rest) if rest else None

        # Own-author reference with just the held-out chunk removed.
        ref_minus: dict[tuple[str, str, int], dict[str, list[float]] | None] = {}
        for author, rel, idx, _ in held_out:
            rest = [
                m
                for r2 in author_books[author]
                for m in (
                    cache[r2][: idx * chunk_size] + cache[r2][(idx + 1) * chunk_size :]
                    if r2 == rel
                    else cache[r2]
                )
            ]
            ref_minus[(author, rel, idx)] = features(rest) if rest else None

        print(f"\n=== chunk_size={chunk_size} | {len(held_out)} held-out human chunks ===")
        for feature in FEATURES:
            for regime in ("same_book", "minus_chunk", "loo_book"):
                hits = 0
                total = 0
                per_author: dict[str, list[int]] = {a: [0, 0] for a in authors}

                for author, rel, idx, feats in held_out:
                    kl = {}
                    for target in authors:
                        if target != author:
                            ref = ref_same[target]
                        elif regime == "same_book":
                            ref = ref_same[target]
                        elif regime == "minus_chunk":
                            ref = ref_minus[(author, rel, idx)]
                        else:
                            ref = ref_loo[(author, rel)]
                        if ref is None:
                            continue
                        kl[target] = float(d_KL(feats[feature], ref[feature]))
                    if not kl:
                        continue
                    best = min(kl, key=kl.get)
                    total += 1
                    per_author[author][1] += 1
                    if best == author:
                        hits += 1
                        per_author[author][0] += 1

                acc = 100 * hits / max(total, 1)
                worst = sorted(
                    ((a, v[0] / v[1]) for a, v in per_author.items() if v[1]),
                    key=lambda kv: kv[1],
                )
                print(
                    f"  {feature} {regime:11s}: {hits}/{total} = {acc:5.1f}%  "
                    f"(chance {100 / len(authors):.0f}%)  "
                    f"worst: " + ", ".join(f"{a}={100*p:.0f}%" for a, p in worst[:3])
                )


if __name__ == "__main__":
    main()
