#!/usr/bin/env python3
"""
Replicate figs. 8 and 9 of Darmon et al., "Pull out all the stops" (1901.00519v2)
on the 10-author campaign subset.

Protocol, following section 2.2 and the fig. 9 caption:
  - A "document" is a book. Features are computed on the FULL punctuation
    sequence of the document; the paper does not truncate.
  - f1 = mark-frequency vector (eq. 2.1).
  - f3 = flattened joint matrix of successive pairs (eq. 2.3), i.e. P_ij * f1_i,
    which is what normalised_transition_mat() returns.
  - Within-author distribution: KL over all ordered pairs of distinct documents
    by the same author. Author consistency (eq. 3.1) is the per-author mean.
  - Between-author distribution: KL over ordered pairs of documents by distinct
    authors. The paper samples 1000 such pairs; this subset has few enough
    documents to enumerate them all, so both are reported.
  - KS test between the two distributions.

Reference values from the fig. 9 caption (651 authors, 14947 documents):
    f1  within 0.0828   between 0.240
    f3  within 0.167    between 0.433

`--truncate N` reruns the same comparison on a single N-mark chunk taken from
the midpoint of each document, which is what run_reproducible_kl_analysis.py
currently does, to show what that truncation costs.

Usage:
  python tools/replicate_darmon_fig89.py \
      --authors-config campaigns/generation_campaign_phaseA_two_samples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PAPER_REFERENCE = {
    "f1": {"within": 0.0828, "between": 0.240},
    "f3": {"within": 0.167, "between": 0.433},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--authors-config", required=True)
    p.add_argument("--cache", default="cache/punct_sequences.json")
    p.add_argument(
        "--truncate",
        nargs="*",
        type=int,
        default=[2000],
        help="Also report middle-chunk truncated variants at these sizes.",
    )
    p.add_argument(
        "--between-sample",
        type=int,
        default=1000,
        help="Size of the random between-author pair sample (paper uses 1000).",
    )
    p.add_argument("--seed", type=int, default=42)
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
from scipy.stats import ks_2samp  # noqa: E402

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


def middle(seq: list[str], n: int) -> list[str] | None:
    if len(seq) < n:
        return None
    start = (len(seq) - n) // 2
    return seq[start : start + n]


def dedupe(rels: list[str], cache: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    seen: dict[tuple[int, str], str] = {}
    kept: list[str] = []
    dropped: list[str] = []
    for rel in rels:
        seq = cache.get(rel, [])
        sig = (len(seq), "".join(seq[:200]))
        if sig in seen:
            dropped.append(f"{rel} (identical to {seen[sig]})")
            continue
        seen[sig] = rel
        kept.append(rel)
    return kept, dropped


def run_protocol(
    label: str,
    docs: list[tuple[str, str, dict[str, list[float]]]],
    between_sample: int,
    rng: np.random.Generator,
    csv_rows: list[dict[str, object]],
) -> None:
    print(f"\n{'='*80}\n{label}   ({len(docs)} documents)")

    within_idx = [
        (i, j)
        for i, j in permutations(range(len(docs)), 2)
        if docs[i][0] == docs[j][0]
    ]
    between_idx = [
        (i, j)
        for i, j in permutations(range(len(docs)), 2)
        if docs[i][0] != docs[j][0]
    ]
    print(
        f"  ordered pairs: {len(within_idx)} within-author, "
        f"{len(between_idx)} between-author"
    )

    sampled = between_idx
    if between_sample and between_sample < len(between_idx):
        pick = rng.choice(len(between_idx), size=between_sample, replace=False)
        sampled = [between_idx[i] for i in pick]

    for feature in FEATURES:
        within = np.asarray(
            [float(d_KL(docs[i][2][feature], docs[j][2][feature])) for i, j in within_idx],
            dtype=float,
        )
        between_all = np.asarray(
            [float(d_KL(docs[i][2][feature], docs[j][2][feature])) for i, j in between_idx],
            dtype=float,
        )
        between_s = np.asarray(
            [float(d_KL(docs[i][2][feature], docs[j][2][feature])) for i, j in sampled],
            dtype=float,
        )

        ref = PAPER_REFERENCE[feature]
        ks = ks_2samp(within, between_all)
        ratio_w = within.mean() / ref["within"]
        ratio_b = between_all.mean() / ref["between"]

        print(
            f"\n  {feature}: within  mean={within.mean():.4f} "
            f"(paper {ref['within']:.4f}, ratio {ratio_w:.2f}x)  "
            f"median={np.median(within):.4f}  n={within.size}"
        )
        print(
            f"      between mean={between_all.mean():.4f} "
            f"(paper {ref['between']:.4f}, ratio {ratio_b:.2f}x)  "
            f"median={np.median(between_all):.4f}  n={between_all.size}"
        )
        if between_s.size != between_all.size:
            print(f"      between mean on {between_s.size}-pair sample={between_s.mean():.4f}")
        print(
            f"      separation between/within = {between_all.mean()/within.mean():.2f}x "
            f"(paper {ref['between']/ref['within']:.2f}x)   "
            f"KS D={ks.statistic:.3f} p={ks.pvalue:.2e}"
        )

        csv_rows.append(
            {
                "protocol": label,
                "feature": feature,
                "n_within_pairs": int(within.size),
                "n_between_pairs": int(between_all.size),
                "within_mean": float(within.mean()),
                "within_median": float(np.median(within)),
                "between_mean": float(between_all.mean()),
                "between_median": float(np.median(between_all)),
                "paper_within_mean": ref["within"],
                "paper_between_mean": ref["between"],
                "within_ratio_to_paper": float(ratio_w),
                "between_ratio_to_paper": float(ratio_b),
                "separation_ratio": float(between_all.mean() / within.mean()),
                "paper_separation_ratio": ref["between"] / ref["within"],
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )

    # Figure 8 analogue: per-author consistency against the between-author baseline.
    baseline = {
        feature: float(
            np.mean(
                [float(d_KL(docs[i][2][feature], docs[j][2][feature])) for i, j in sampled]
            )
        )
        for feature in FEATURES
    }
    authors = sorted({a for a, _, _ in docs})
    print("\n  author consistency (fig. 8 analogue; lower = more consistent)")
    print(f"    {'author':24s} {'n_docs':>6s} {'C_f1':>8s} {'C_f3':>8s}")
    for author in authors:
        idx = [k for k, (a, _, _) in enumerate(docs) if a == author]
        if len(idx) < 2:
            print(f"    {author:24s} {len(idx):6d} {'--':>8s} {'--':>8s}   (needs >=2 docs)")
            continue
        cells = []
        for feature in FEATURES:
            vals = [
                float(d_KL(docs[i][2][feature], docs[j][2][feature]))
                for i, j in permutations(idx, 2)
            ]
            cells.append(float(np.mean(vals)))
            csv_rows.append(
                {
                    "protocol": label,
                    "feature": feature,
                    "author_consistency_author": author,
                    "author_consistency_n_docs": len(idx),
                    "author_consistency_mean_kl": float(np.mean(vals)),
                    "consistency_baseline": baseline[feature],
                }
            )
        print(f"    {author:24s} {len(idx):6d} {cells[0]:8.4f} {cells[1]:8.4f}")
    print(
        f"    {'BASELINE (between-author)':24s} {'':6s} "
        f"{baseline['f1']:8.4f} {baseline['f3']:8.4f}"
    )


def main() -> None:
    cache = json.load(open(ROOT / ARGS.cache))["sequences"]
    cfg = json.load(open(ROOT / ARGS.authors_config))
    rng = np.random.default_rng(ARGS.seed)

    author_books: dict[str, list[str]] = {}
    for entry in cfg["authors"]:
        rels = list(entry.get("book_paths") or [entry["book_path"]])
        kept, dropped = dedupe(rels, cache)
        for d in dropped:
            print(f"[dedupe] {entry['key']}: dropped {d}")
        author_books[entry["key"]] = kept

    print("\ndocuments per author (after dedupe):")
    for author, rels in author_books.items():
        marks = [len(cache[r]) for r in rels]
        print(f"  {author:24s} {len(rels)} docs, marks={marks}")

    csv_rows: list[dict[str, object]] = []

    full_docs = []
    for author, rels in author_books.items():
        for rel in rels:
            feats = features(cache[rel])
            if feats:
                full_docs.append((author, rel, feats))
    run_protocol(
        "paper protocol: full documents, no truncation",
        full_docs,
        ARGS.between_sample,
        rng,
        csv_rows,
    )

    for n in ARGS.truncate or []:
        trunc_docs = []
        for author, rels in author_books.items():
            for rel in rels:
                chunk = middle(cache[rel], n)
                if chunk is None:
                    print(
                        f"[warn] {author}: {rel} has {len(cache[rel])} marks, "
                        f"below {n} -> excluded from truncated variant"
                    )
                    continue
                feats = features(chunk)
                if feats:
                    trunc_docs.append((author, rel, feats))
        run_protocol(
            f"current pipeline: single middle {n}-mark chunk per document",
            trunc_docs,
            ARGS.between_sample,
            rng,
            csv_rows,
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
