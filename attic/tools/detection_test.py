#!/usr/bin/env python3
"""
Turn the KL descriptor into a hypothesis test, and measure how well it separates
human chunks from LLM chunks.

Two things are computed.

1) A per-chunk goodness-of-fit test against an author profile.

   For the mark-frequency feature f1, the multinomial likelihood-ratio statistic
   against a fixed author profile p0 is

       G = 2 * sum_i O_i * ln(O_i / (n * p0_i)) = 2 * n * KL(p_hat || p0)

   so the KL already in use IS the likelihood-ratio statistic up to the factor
   2n, and G is asymptotically chi-square with k-1 degrees of freedom. This is
   the size/power framing Sam asked for, and it costs nothing extra.

   For the transition feature the same construction applies row-wise:

       G = 2 * sum_i sum_j O_ij * ln(O_ij / (n_i * p0_ij))

   with k*(k-1) degrees of freedom for a first-order Markov chain.

   Calibration is checked empirically on held-out human chunks rather than
   assumed, because punctuation marks are not independent draws.

2) Detection performance: the distribution of the statistic for human chunks
   versus LLM chunks, summarised by AUC and by the true-positive rate at a
   threshold calibrated to a 5% false-positive rate on human text.

Usage:
  python tools/detection_test.py \
      --authors-config campaigns/generation_campaign_phaseA_two_samples.json \
      --condition phaseA_flash=generated_texts_campaign_phaseA_two_samples \
      --condition phaseB_pro=generated_texts_campaign_phaseB_two_samples \
      --chunk-sizes 1000 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--authors-config", required=True)
    p.add_argument("--condition", action="append", default=[])
    p.add_argument("--chunk-sizes", nargs="+", type=int, default=[2000])
    p.add_argument("--cache", default="cache/punct_sequences.json")
    p.add_argument("--max-chunks-per-book", type=int, default=40)
    p.add_argument("--out", default=None, help="Optional CSV output path.")
    return p.parse_args()


ARGS = parse_args()

sys.path.insert(0, str(ROOT / "punctuation-stylometry-master"))
_CONFIG = str(ROOT / "punctuation-stylometry-master" / "conf" / "punctuation.ini")
sys.argv = [sys.argv[0], "-c", _CONFIG]

from punctuation.config import options  # noqa: E402

PUNCT_VECTOR = list(options.punctuation_vector)
K = len(PUNCT_VECTOR)
INDEX = {m: i for i, m in enumerate(PUNCT_VECTOR)}


def counts_f1(seq: list[str]) -> np.ndarray:
    out = np.zeros(K, dtype=float)
    for m in seq:
        i = INDEX.get(m)
        if i is not None:
            out[i] += 1
    return out


def counts_f2(seq: list[str]) -> np.ndarray:
    out = np.zeros((K, K), dtype=float)
    prev = None
    for m in seq:
        i = INDEX.get(m)
        if i is None:
            prev = None
            continue
        if prev is not None:
            out[prev, i] += 1
        prev = i
    return out


def g_stat_f1(obs: np.ndarray, p0: np.ndarray) -> float:
    n = obs.sum()
    if n <= 0:
        return float("nan")
    exp = n * p0
    mask = (obs > 0) & (exp > 0)
    return float(2.0 * np.sum(obs[mask] * np.log(obs[mask] / exp[mask])))


def g_stat_f2(obs: np.ndarray, p0_rows: np.ndarray) -> float:
    row_n = obs.sum(axis=1, keepdims=True)
    exp = row_n * p0_rows
    mask = (obs > 0) & (exp > 0)
    return float(2.0 * np.sum(obs[mask] * np.log(obs[mask] / exp[mask])))


def row_normalise(mat: np.ndarray) -> np.ndarray:
    rows = mat.sum(axis=1, keepdims=True)
    safe = np.where(rows > 0, rows, 1.0)
    out = mat / safe
    # Rows never observed fall back to uniform so the statistic stays finite.
    out[rows[:, 0] == 0] = 1.0 / K
    return out


def smooth(vec: np.ndarray, eps: float = 0.5) -> np.ndarray:
    """Add-eps smoothing so reference profiles never contain exact zeros."""
    v = vec + eps
    return v / v.sum()


def smooth_rows(mat: np.ndarray, eps: float = 0.5) -> np.ndarray:
    m = mat + eps
    return m / m.sum(axis=1, keepdims=True)


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


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg), ties counted as half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def main() -> None:
    cache = json.load(open(ROOT / ARGS.cache))["sequences"]
    cfg = json.load(open(ROOT / ARGS.authors_config))
    author_books = {
        e["key"]: dedupe(list(e.get("book_paths") or [e["book_path"]]), cache)
        for e in cfg["authors"]
    }
    authors = list(author_books)
    conditions = dict(m.split("=", 1) for m in ARGS.condition)

    csv_rows: list[dict[str, object]] = []

    for chunk_size in ARGS.chunk_sizes:
        # Author reference profiles from the pooled full text.
        ref_f1: dict[str, np.ndarray] = {}
        ref_f2: dict[str, np.ndarray] = {}
        for author, rels in author_books.items():
            seq = [m for rel in rels for m in cache[rel]]
            ref_f1[author] = smooth(counts_f1(seq))
            ref_f2[author] = smooth_rows(counts_f2(seq))

        # Held-out human chunks, scored against their own author, with that
        # chunk removed from the reference.
        human: dict[str, list[float]] = {"f1": [], "f3": []}
        for author, rels in author_books.items():
            full_c1 = counts_f1([m for rel in rels for m in cache[rel]])
            full_c2 = counts_f2([m for rel in rels for m in cache[rel]])
            for rel in rels:
                seq = cache[rel]
                n_chunks = min(len(seq) // chunk_size, ARGS.max_chunks_per_book)
                for i in range(n_chunks):
                    chunk = seq[i * chunk_size : (i + 1) * chunk_size]
                    c1 = counts_f1(chunk)
                    c2 = counts_f2(chunk)
                    p0_1 = smooth(np.maximum(full_c1 - c1, 0))
                    p0_2 = smooth_rows(np.maximum(full_c2 - c2, 0))
                    human["f1"].append(g_stat_f1(c1, p0_1))
                    human["f3"].append(g_stat_f2(c2, p0_2))

        h1 = np.asarray(human["f1"], dtype=float)
        h3 = np.asarray(human["f3"], dtype=float)

        df1 = K - 1
        df3 = K * (K - 1)
        print(f"\n{'='*78}\nchunk_size={chunk_size}   human chunks={len(h1)}")
        print(
            f"  f1 calibration: mean G={h1.mean():8.1f}  vs chi2 df={df1} mean={df1}"
            f"   -> overdispersion x{h1.mean()/df1:.1f}"
        )
        print(
            f"  f3 calibration: mean G={h3.mean():8.1f}  vs chi2 df={df3} mean={df3}"
            f"   -> overdispersion x{h3.mean()/df3:.1f}"
        )
        print(
            f"  nominal 5% chi2 rejection rate on human text: "
            f"f1={100*np.mean(h1 > chi2.ppf(0.95, df1)):.0f}%  "
            f"f3={100*np.mean(h3 > chi2.ppf(0.95, df3)):.0f}%"
            "   (should be 5% if the chi2 reference held)"
        )

        for feature, hvals in (("f1", h1), ("f3", h3)):
            # Empirical threshold at 5% false-positive rate on human text.
            thr = float(np.percentile(hvals, 95))
            for cond_name, cond_dir in conditions.items():
                llm_vals: list[float] = []
                for author in authors:
                    for rel in sorted(
                        k for k in cache if k.startswith(f"{cond_dir}/{author}/")
                    ):
                        seq = cache[rel][:chunk_size]
                        if len(seq) < chunk_size:
                            continue
                        if feature == "f1":
                            llm_vals.append(g_stat_f1(counts_f1(seq), ref_f1[author]))
                        else:
                            llm_vals.append(g_stat_f2(counts_f2(seq), ref_f2[author]))

                lv = np.asarray(llm_vals, dtype=float)
                if lv.size == 0:
                    continue
                a = auc(lv, hvals)
                tpr = float(np.mean(lv > thr))
                mw = mannwhitneyu(lv, hvals, alternative="greater").pvalue
                print(
                    f"  {feature} {cond_name:13s}: human median G={np.median(hvals):7.1f}"
                    f"  LLM median G={np.median(lv):7.1f}"
                    f"  AUC={a:.3f}  TPR@5%FPR={100*tpr:5.1f}%  p={mw:.2e}"
                )
                csv_rows.append(
                    {
                        "chunk_size": chunk_size,
                        "feature": feature,
                        "condition": cond_name,
                        "n_human": int(hvals.size),
                        "n_llm": int(lv.size),
                        "human_median_G": float(np.median(hvals)),
                        "llm_median_G": float(np.median(lv)),
                        "auc": a,
                        "threshold_95pct_human": thr,
                        "tpr_at_5pct_fpr": tpr,
                        "mannwhitney_p_llm_greater": float(mw),
                    }
                )

    if ARGS.out and csv_rows:
        import csv

        out_path = ROOT / ARGS.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
