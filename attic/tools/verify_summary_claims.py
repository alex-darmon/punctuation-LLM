#!/usr/bin/env python3
"""
Checks on three claims that were carried over from earlier runs without being
re-tested under the leave-one-book-out correction.

1) Detection AUC under a leave-one-book-out human null. The earlier detection
   numbers scored human chunks against a reference that still contained the rest
   of the same book, which is exactly the leakage the attribution baseline was
   corrected for. If the correction matters for attribution, it should matter
   here too.

2) LLM attribution with the prompt's source book removed from the target
   author's reference. The claim that the human-vs-LLM gap is "conservative"
   rests on the LLM keeping a reference advantage; this measures it instead of
   asserting it.

3) Binomial tests for the above-chance attribution claims.

Usage:
  python tools/verify_summary_claims.py \
      --authors-config campaigns/generation_campaign_phaseA_two_samples.json
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
    p.add_argument("--chunk-size", type=int, default=2000)
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
from scipy.stats import binomtest, mannwhitneyu  # noqa: E402

V = options.punctuation_vector
K = len(V)
IDX = {m: i for i, m in enumerate(V)}
SUBSET5 = [
    "jane_austen",
    "charles_dickens",
    "mark_twain",
    "arthur_conan_doyle",
    "mary_shelley",
]


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


def c1(seq: list[str]) -> np.ndarray:
    o = np.zeros(K)
    for m in seq:
        i = IDX.get(m)
        if i is not None:
            o[i] += 1
    return o


def c2(seq: list[str]) -> np.ndarray:
    o = np.zeros((K, K))
    prev = None
    for m in seq:
        i = IDX.get(m)
        if i is None:
            prev = None
            continue
        if prev is not None:
            o[prev, i] += 1
        prev = i
    return o


def smooth(v: np.ndarray, eps: float = 0.5) -> np.ndarray:
    x = v + eps
    return x / x.sum()


def smooth_rows(m: np.ndarray, eps: float = 0.5) -> np.ndarray:
    x = m + eps
    return x / x.sum(axis=1, keepdims=True)


def g1(obs: np.ndarray, p0: np.ndarray) -> float:
    n = obs.sum()
    if n <= 0:
        return float("nan")
    exp = n * p0
    mask = (obs > 0) & (exp > 0)
    return float(2.0 * np.sum(obs[mask] * np.log(obs[mask] / exp[mask])))


def g2(obs: np.ndarray, p0: np.ndarray) -> float:
    exp = obs.sum(axis=1, keepdims=True) * p0
    mask = (obs > 0) & (exp > 0)
    return float(2.0 * np.sum(obs[mask] * np.log(obs[mask] / exp[mask])))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (pos.size * neg.size))


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
    n = ARGS.chunk_size

    books = {
        e["key"]: dedupe(list(e.get("book_paths") or [e["book_path"]]), cache)
        for e in cfg["authors"]
    }
    authors = list(books)

    conditions = {
        "flash": "generated_texts_campaign_phaseA_two_samples",
        "pro": "generated_texts_campaign_phaseB_two_samples",
    }

    # ---- 1) detection under both human nulls -----------------------------
    print("=" * 78)
    print(f"1) Detection AUC, chunk={n}: same-book null vs leave-one-book-out null")

    human = {"same_book": {"f1": [], "f3": []}, "loo_book": {"f1": [], "f3": []}}
    for a, rels in books.items():
        tot1 = c1([m for r in rels for m in cache[r]])
        tot2 = c2([m for r in rels for m in cache[r]])
        for rel in rels:
            b1 = c1(cache[rel])
            b2 = c2(cache[rel])
            for i in range(len(cache[rel]) // n):
                ch = cache[rel][i * n : (i + 1) * n]
                k1 = c1(ch)
                k2 = c2(ch)
                # same book retained, only this chunk removed
                human["same_book"]["f1"].append(g1(k1, smooth(np.maximum(tot1 - k1, 0))))
                human["same_book"]["f3"].append(g2(k2, smooth_rows(np.maximum(tot2 - k2, 0))))
                # whole source book removed
                r1 = np.maximum(tot1 - b1, 0)
                r2 = np.maximum(tot2 - b2, 0)
                if r1.sum() > 0:
                    human["loo_book"]["f1"].append(g1(k1, smooth(r1)))
                    human["loo_book"]["f3"].append(g2(k2, smooth_rows(r2)))

    ref1 = {a: smooth(c1([m for r in rels for m in cache[r]])) for a, rels in books.items()}
    ref2 = {
        a: smooth_rows(c2([m for r in rels for m in cache[r]])) for a, rels in books.items()
    }

    llm = {cond: {"f1": [], "f3": []} for cond in conditions}
    for cond, d in conditions.items():
        for a in authors:
            for k in sorted(x for x in cache if x.startswith(f"{d}/{a}/")):
                ch = cache[k][:n]
                if len(ch) < n:
                    continue
                llm[cond]["f1"].append(g1(c1(ch), ref1[a]))
                llm[cond]["f3"].append(g2(c2(ch), ref2[a]))

    for feature in ("f1", "f3"):
        print(f"\n  {feature}:")
        for null_name in ("same_book", "loo_book"):
            h = np.asarray(human[null_name][feature], dtype=float)
            thr = float(np.percentile(h, 95))
            line = f"    {null_name:10s} human median G={np.median(h):8.1f} n={h.size:4d}"
            for cond in conditions:
                lv = np.asarray(llm[cond][feature], dtype=float)
                line += (
                    f" | {cond} AUC={auc(lv, h):.3f}"
                    f" TPR@5%FPR={100*np.mean(lv > thr):4.1f}%"
                )
            print(line)

    # ---- 2) LLM attribution with the prompt book removed -----------------
    print("\n" + "=" * 78)
    print("2) LLM attribution: does removing the prompt's source book from the")
    print("   target reference change the result? (f3, chunk=%d)" % n)

    for cond, d in conditions.items():
        summary = json.load(open(ROOT / d / "all_runs_summary.json"))
        src = {(r["author_key"], r["run_id"]): r["source_book_path"] for r in summary}
        for label, subset in (("all 10", authors), ("separable 5", SUBSET5)):
            targets = [a for a in authors if a in subset]
            for mode in ("full_ref", "drop_prompt_book"):
                hits = tot = 0
                for a in targets:
                    for k in sorted(x for x in cache if x.startswith(f"{d}/{a}/")):
                        ch = cache[k][:n]
                        if len(ch) < n:
                            continue
                        f = feats(ch)
                        run_id = int(Path(k).stem.split("_")[-1])
                        kl = {}
                        for t in targets:
                            if mode == "drop_prompt_book" and t == a:
                                p = src.get((a, run_id))
                                rel = (
                                    str(Path(p).relative_to(ROOT))
                                    if p and str(p).startswith(str(ROOT))
                                    else None
                                )
                                rest = [
                                    m for r2 in books[a] if r2 != rel for m in cache[r2]
                                ]
                                rf = feats(rest) if rest else None
                            else:
                                rf = feats([m for r2 in books[t] for m in cache[r2]])
                            if rf is None:
                                continue
                            kl[t] = float(d_KL(f["f3"], rf["f3"]))
                        if not kl:
                            continue
                        tot += 1
                        if min(kl, key=kl.get) == a:
                            hits += 1
                pct = 100 * hits / max(tot, 1)
                print(f"  {cond:6s} {label:12s} {mode:17s}: {hits}/{tot} = {pct:5.1f}%")

    # ---- 3) binomial tests ----------------------------------------------
    print("\n" + "=" * 78)
    print("3) Is LLM attribution significantly above chance? (f3, one-sided)")
    for label, hits, total, chance in (
        ("flash, all 10", 38, 200, 0.10),
        ("pro,   all 10", 14, 100, 0.10),
        ("flash, separable 5", 32, 100, 0.20),
        ("pro,   separable 5", 12, 50, 0.20),
    ):
        r = binomtest(hits, total, chance, alternative="greater")
        verdict = "significant" if r.pvalue < 0.05 else "NOT significant"
        print(
            f"  {label:20s} {hits:3d}/{total:3d} vs {100*chance:.0f}% chance: "
            f"p={r.pvalue:.4f}  {verdict}"
        )


if __name__ == "__main__":
    main()
