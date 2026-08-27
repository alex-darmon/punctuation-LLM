#!/usr/bin/env python3
"""Recompute per-author self-consistency C, nearest-other distance D and margin.

C = mean pairwise KL between an author's own documents (full documents).
D = min over other authors of KL(author's pooled profile -> other's pooled profile).
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--authors-config", required=True)
    p.add_argument("--cache", default="cache/punct_sequences.json")
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

V = options.punctuation_vector


def feats(seq: list[str]) -> dict[str, list[float]] | None:
    if not seq:
        return None
    f1 = get_frequencies(seq, vector=V)
    if f1 is None or sum(f1) == 0:
        return None
    f2 = transition_mat(seq)
    if f2 is None:
        return None
    return {"f1": f1, "f3": normalised_transition_mat(f2, f1).flatten().tolist()}


def dedupe(rels: list[str], cache: dict) -> list[str]:
    seen: set = set()
    out = []
    for r in rels:
        s = cache.get(r, [])
        sig = (len(s), "".join(s[:200]))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(r)
    return out


def main() -> None:
    cache = json.load(open(ROOT / ARGS.cache))["sequences"]
    cfg = json.load(open(ROOT / ARGS.authors_config))
    books = {
        e["key"]: dedupe(list(e.get("book_paths") or [e["book_path"]]), cache)
        for e in cfg["authors"]
    }

    doc_f = {a: [feats(cache[r]) for r in rels] for a, rels in books.items()}
    pooled = {a: feats([m for r in rels for m in cache[r]]) for a, rels in books.items()}

    between = {"f1": [], "f3": []}
    for a, b in combinations(books, 2):
        for feature in ("f1", "f3"):
            between[feature].append(float(d_KL(pooled[a][feature], pooled[b][feature])))

    for feature in ("f1", "f3"):
        base = float(np.mean(between[feature]))
        print(f"\n=== {feature}: between-author baseline = {base:.4f} ===")
        rows = []
        for a in books:
            fs = [f for f in doc_f[a] if f]
            pairs = [
                float(d_KL(x[feature], y[feature])) for x, y in combinations(fs, 2)
            ] + [float(d_KL(y[feature], x[feature])) for x, y in combinations(fs, 2)]
            c = float(np.mean(pairs)) if pairs else float("nan")
            d = min(
                float(d_KL(pooled[a][feature], pooled[o][feature]))
                for o in books
                if o != a
            )
            rows.append((a, len(fs), c, d, d / c if c else float("nan"), c / base))
        rows.sort(key=lambda r: -r[2])
        print(
            f"{'author':24s} {'#bk':>3s} {'C':>8s} {'D':>8s} {'margin':>7s} {'C/base':>7s}"
        )
        for a, nb, c, d, mg, cb in rows:
            print(f"{a:24s} {nb:3d} {c:8.4f} {d:8.4f} {mg:7.2f} {cb:7.2f}")


if __name__ == "__main__":
    main()
