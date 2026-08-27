#!/usr/bin/env python3
"""
Diagnostic: how much of the LLM-vs-author attribution result depends on how the
human reference profile is estimated?

Compares three reference estimators for the "is LLM-Author_x nearest to
Author_x?" sanity check:

  chunk_mean : one middle chunk of `--chunk-size` marks per book, f1/f3
               averaged across books (what run_reproducible_kl_analysis.py does)
  full_mean  : full mark sequence of each book, f1/f3 averaged across books
  full_pooled: all books concatenated into one sequence, then f1/f3

Usage:
  python tools/diagnose_reference_profile.py \
      --authors-config campaigns/generation_campaign_phaseA_two_samples.json \
      --condition phaseA_flash=generated_texts_campaign_phaseA_two_samples \
      --condition phaseB_pro=generated_texts_campaign_phaseB_two_samples \
      --chunk-size 2000
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
    p.add_argument(
        "--condition",
        action="append",
        default=[],
        help="name=path for a generated-text directory (repeatable).",
    )
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--cache", default="cache/punct_sequences.json")
    p.add_argument(
        "--dedupe-books",
        action="store_true",
        help="Drop byte-identical duplicate source books within an author.",
    )
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


def middle_chunk(seq: list[str], n: int) -> list[str] | None:
    if len(seq) < n:
        return None
    start = (len(seq) - n) // 2
    return seq[start : start + n]


def main() -> None:
    cache = json.load(open(ROOT / ARGS.cache))["sequences"]
    cfg = json.load(open(ROOT / ARGS.authors_config))

    author_books: dict[str, list[str]] = {}
    for entry in cfg["authors"]:
        paths = list(entry.get("book_paths") or [entry["book_path"]])
        if ARGS.dedupe_books:
            seen: set[tuple[int, str]] = set()
            unique = []
            for rel in paths:
                seq = cache.get(rel, [])
                sig = (len(seq), "".join(seq[:200]))
                if sig in seen:
                    print(f"[dedupe] {entry['key']}: dropping duplicate {rel}")
                    continue
                seen.add(sig)
                unique.append(rel)
            paths = unique
        author_books[entry["key"]] = paths

    authors = list(author_books)

    refs: dict[str, dict[str, dict[str, list[float]]]] = {
        "chunk_mean": {},
        "full_mean": {},
        "full_pooled": {},
    }

    for author, rels in author_books.items():
        chunk_rows: dict[str, list[list[float]]] = {f: [] for f in FEATURES}
        full_rows: dict[str, list[list[float]]] = {f: [] for f in FEATURES}
        pooled_seq: list[str] = []

        for rel in rels:
            seq = cache.get(rel, [])
            pooled_seq.extend(seq)

            chunk = middle_chunk(seq, ARGS.chunk_size)
            if chunk is not None:
                feats = features(chunk)
                if feats:
                    for f in FEATURES:
                        chunk_rows[f].append(feats[f])
            else:
                print(
                    f"[warn] {author}: {rel} has {len(seq)} marks, "
                    f"below chunk size {ARGS.chunk_size} -> excluded from chunk_mean"
                )

            feats_full = features(seq)
            if feats_full:
                for f in FEATURES:
                    full_rows[f].append(feats_full[f])

        if all(chunk_rows[f] for f in FEATURES):
            refs["chunk_mean"][author] = {
                f: np.mean(np.asarray(chunk_rows[f]), axis=0).tolist() for f in FEATURES
            }
        refs["full_mean"][author] = {
            f: np.mean(np.asarray(full_rows[f]), axis=0).tolist() for f in FEATURES
        }
        pooled = features(pooled_seq)
        if pooled:
            refs["full_pooled"][author] = pooled

    conditions: dict[str, str] = {}
    for mapping in ARGS.condition:
        name, path = mapping.split("=", 1)
        conditions[name] = path

    print(f"\nchunk_size={ARGS.chunk_size}  authors={len(authors)}")

    for cond_name, cond_dir in conditions.items():
        runs_by_author: dict[str, list[dict[str, list[float]]]] = {}
        for author in authors:
            rows = []
            for rel in sorted(k for k in cache if k.startswith(f"{cond_dir}/{author}/")):
                chunk = cache[rel][: ARGS.chunk_size]
                if len(chunk) < ARGS.chunk_size:
                    continue
                feats = features(chunk)
                if feats:
                    rows.append(feats)
            runs_by_author[author] = rows

        for estimator in ("chunk_mean", "full_mean", "full_pooled"):
            ref = refs[estimator]
            print(f"\n=== {cond_name} | reference={estimator} ===")
            for feature in FEATURES:
                run_level_hits = 0
                run_level_total = 0
                mean_level_hits = 0
                winners: dict[str, int] = {}

                for author in authors:
                    if author not in ref:
                        continue
                    rows = runs_by_author[author]
                    if not rows:
                        continue

                    kl_by_target: dict[str, list[float]] = {}
                    for target in authors:
                        if target not in ref:
                            continue
                        kl_by_target[target] = [
                            float(d_KL(r[feature], ref[target][feature])) for r in rows
                        ]

                    for i in range(len(rows)):
                        best = min(kl_by_target, key=lambda t: kl_by_target[t][i])
                        winners[best] = winners.get(best, 0) + 1
                        run_level_total += 1
                        if best == author:
                            run_level_hits += 1

                    means = {t: float(np.mean(v)) for t, v in kl_by_target.items()}
                    if min(means, key=means.get) == author:
                        mean_level_hits += 1

                n_authors = sum(1 for a in authors if a in ref and runs_by_author[a])
                top = sorted(winners.items(), key=lambda kv: -kv[1])[:3]
                print(
                    f"  {feature}: mean-level correct {mean_level_hits}/{n_authors}"
                    f" | run-level correct {run_level_hits}/{run_level_total}"
                    f" ({100 * run_level_hits / max(run_level_total, 1):.1f}%)"
                    f" | top winners {top}"
                )


if __name__ == "__main__":
    main()
