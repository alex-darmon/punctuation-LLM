#!/usr/bin/env python3
"""Frozen analysis grid: the single entry point for every number in the write-up.

All nine experiments run from one corpus load, one feature definition and one
reference builder. Author profiles come only from
`punctlib.reference.build_reference_set`, which demands an explicit
`exclude_books` policy, and every profile built is written to
`reference_audit.csv` so the policy can be verified after the run.

Experiments
  1 replication      within- vs between-author KL over documents, and the cost of
                     truncating references to a single middle chunk.
  2 separability     per-author self-consistency C, nearest-other distance D, and
                     the margin D/C. The margin > 1 author subset used by the
                     later experiments is derived here, from human data only, so
                     no subset is selected on the outcome it is used to report.
  3 human_attribution leave-one-book-out attribution: the human baseline.
  4 llm_attribution  the same task for generated text, under both a matched policy
                     (the prompt's source book excluded) and a pooled policy.
  5 detection        human vs LLM as a binary decision, under a leave-one-book-out
                     null and, for contrast, the pooled null that inflated the
                     earlier figures.
  6 calibration      the chi-square reference against observed human G values.
  7 marks            per-mark frequency, human vs each model.
  8 dispersion       run-to-run LLM spread against within-book and across-book
                     human spread.
  9 context_drift    five consecutive windows from each generation, testing
                     whether target-author similarity weakens and predictions
                     converge as position in the generated text increases.

Usage
  python run_frozen_grid.py \
      --authors-config campaigns/generation_campaign_phaseA_two_samples.json \
      --condition flash=generated_texts_campaign_phaseA_two_samples \
      --condition pro=generated_texts_campaign_phaseB_two_samples \
      --chunk-sizes 1000 2000 4000 \
      --out results/frozen
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, chi2, mannwhitneyu, spearmanr, wilcoxon

from punctlib import (
    FEATURES,
    PUNCT_VECTOR,
    Corpus,
    audit_log,
    build_reference_set,
    chunks,
    counts1,
    counts2,
    features,
    kl,
    load_corpus,
    reset_audit_log,
)
from punctlib.features import middle_chunk
from punctlib.reference import assert_excluded
from punctlib.stats import auc, dof, g_stat, sym_kl, tpr_at_fpr

ROOT = Path(__file__).resolve().parent

# Policy names used in the output tables.
POOLED = "pooled_all_books"       # every book in every profile
LOO = "leave_one_book_out"        # the text under test excluded from its profile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--authors-config", required=True)
    p.add_argument("--cache", default="cache/punct_sequences.json")
    p.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Generated-text directory to analyse (repeatable).",
    )
    p.add_argument("--chunk-sizes", nargs="+", type=int, default=[1000, 2000, 4000])
    p.add_argument(
        "--headline-chunk-size",
        type=int,
        default=2000,
        help="Chunk size quoted in the console summary.",
    )
    p.add_argument(
        "--drift-window-size",
        type=int,
        default=1000,
        help="Marks per consecutive window in the context-drift experiment.",
    )
    p.add_argument(
        "--drift-windows",
        type=int,
        default=5,
        help="Number of consecutive windows required from every generated run.",
    )
    p.add_argument("--out", default="results/frozen")
    return p.parse_args()


# --------------------------------------------------------------------------
# experiment 1: replication of Darmon et al. figures 8 and 9
# --------------------------------------------------------------------------
def exp_replication(corpus: Corpus, truncate_to: int) -> list[dict]:
    docs: dict[str, list] = {}
    trunc: dict[str, list] = {}
    for author in corpus.authors:
        docs[author] = []
        trunc[author] = []
        for book_id in corpus.book_ids(author):
            seq = corpus.marks(book_id)
            full = features(seq)
            if full is not None:
                docs[author].append(full)
            mid = middle_chunk(seq, truncate_to)
            if mid is not None:
                small = features(mid)
                if small is not None:
                    trunc[author].append(small)

    rows: list[dict] = []
    for unit, table in (("full_document", docs), (f"middle_{truncate_to}", trunc)):
        for feature in FEATURES:
            within = [
                kl(x[feature], y[feature])
                for fs in table.values()
                for x, y in combinations(fs, 2)
            ] + [
                kl(y[feature], x[feature])
                for fs in table.values()
                for x, y in combinations(fs, 2)
            ]
            between = []
            for a, b in combinations(table, 2):
                for x in table[a]:
                    for y in table[b]:
                        between.append(kl(x[feature], y[feature]))
                        between.append(kl(y[feature], x[feature]))
            if not within or not between:
                continue
            w = float(np.mean(within))
            bt = float(np.mean(between))
            rows.append(
                {
                    "unit": unit,
                    "feature": feature,
                    "n_within_pairs": len(within),
                    "n_between_pairs": len(between),
                    "within_mean": w,
                    "within_median": float(np.median(within)),
                    "between_mean": bt,
                    "between_median": float(np.median(between)),
                    "separation_ratio": bt / w if w else float("nan"),
                    "mannwhitney_p": float(
                        mannwhitneyu(between, within, alternative="greater").pvalue
                    ),
                }
            )
    return rows


# --------------------------------------------------------------------------
# experiment 2: separability, and the derived author subset
# --------------------------------------------------------------------------
def exp_separability(corpus: Corpus) -> tuple[list[dict], list[str]]:
    pooled = build_reference_set(corpus, exclude_books=())
    rows: list[dict] = []
    for feature in FEATURES:
        between = [
            kl(pooled[a][feature], pooled[b][feature])
            for a, b in combinations(corpus.authors, 2)
        ]
        baseline = float(np.mean(between))
        for author in corpus.authors:
            fs = [features(corpus.marks(b)) for b in corpus.book_ids(author)]
            fs = [f for f in fs if f is not None]
            pairs = [kl(x[feature], y[feature]) for x, y in combinations(fs, 2)]
            pairs += [kl(y[feature], x[feature]) for x, y in combinations(fs, 2)]
            consistency = float(np.mean(pairs)) if pairs else float("nan")
            nearest = min(
                (
                    kl(pooled[author][feature], pooled[other][feature]),
                    other,
                )
                for other in corpus.authors
                if other != author
            )
            rows.append(
                {
                    "feature": feature,
                    "author": author,
                    "n_books": len(fs),
                    "n_marks": pooled[author].n_marks,
                    "consistency_C": consistency,
                    "nearest_other_D": nearest[0],
                    "nearest_other": nearest[1],
                    "margin_D_over_C": nearest[0] / consistency if consistency else float("nan"),
                    "between_author_baseline": baseline,
                    "C_over_baseline": consistency / baseline if baseline else float("nan"),
                }
            )

    subset = sorted(
        r["author"]
        for r in rows
        if r["feature"] == "f3" and r["margin_D_over_C"] > 1.0
    )
    return rows, subset


# --------------------------------------------------------------------------
# experiment 3: human leave-one-book-out attribution
# --------------------------------------------------------------------------
def _attribute(feature: str, feats, refs: dict) -> str:
    return min(refs, key=lambda a: kl(feats[feature], refs[a][feature]))


def exp_human_attribution(
    corpus: Corpus, author_set: str, chunk_sizes: list[int]
) -> list[dict]:
    rows: list[dict] = []
    n_authors = len(corpus.authors)
    chance = 1.0 / n_authors

    for feature in FEATURES:
        # document level, the unit used in table 2 of the paper
        tally: dict[str, list[int]] = {a: [0, 0] for a in corpus.authors}
        confusion: dict[str, int] = {}
        for author in corpus.authors:
            for book_id in corpus.book_ids(author):
                refs = build_reference_set(corpus, exclude_books=[book_id])
                if author not in refs:
                    continue
                assert_excluded(refs[author], corpus, book_id)
                feats = features(corpus.marks(book_id))
                if feats is None:
                    continue
                pick = _attribute(feature, feats, refs)
                confusion[pick] = confusion.get(pick, 0) + 1
                tally[author][1] += 1
                tally[author][0] += pick == author
        rows += _accuracy_rows(
            experiment="human_attribution",
            author_set=author_set,
            policy=LOO,
            unit="document",
            chunk_size="",
            feature=feature,
            tally=tally,
            confusion=confusion,
            chance=chance,
        )

        # chunk level
        for size in chunk_sizes:
            tally = {a: [0, 0] for a in corpus.authors}
            confusion = {}
            for author in corpus.authors:
                for book_id in corpus.book_ids(author):
                    refs = build_reference_set(corpus, exclude_books=[book_id])
                    if author not in refs:
                        continue
                    assert_excluded(refs[author], corpus, book_id)
                    for chunk in chunks(corpus.marks(book_id), size):
                        feats = features(chunk)
                        if feats is None:
                            continue
                        pick = _attribute(feature, feats, refs)
                        confusion[pick] = confusion.get(pick, 0) + 1
                        tally[author][1] += 1
                        tally[author][0] += pick == author
            rows += _accuracy_rows(
                experiment="human_attribution",
                author_set=author_set,
                policy=LOO,
                unit="chunk",
                chunk_size=size,
                feature=feature,
                tally=tally,
                confusion=confusion,
                chance=chance,
            )
    return rows


# --------------------------------------------------------------------------
# experiment 4: LLM attribution
# --------------------------------------------------------------------------
def exp_llm_attribution(
    corpus: Corpus, author_set: str, conditions: dict[str, str], chunk_sizes: list[int]
) -> list[dict]:
    rows: list[dict] = []
    chance = 1.0 / len(corpus.authors)

    for condition in conditions:
        for feature in FEATURES:
            for size in chunk_sizes:
                for policy in (POOLED, LOO):
                    tally: dict[str, list[int]] = {a: [0, 0] for a in corpus.authors}
                    confusion: dict[str, int] = {}
                    skipped = 0
                    for author in corpus.authors:
                        for run in corpus.runs(condition, author):
                            seq = corpus.marks(run.cache_key)[:size]
                            if len(seq) < size:
                                skipped += 1
                                continue
                            if policy == LOO:
                                if run.source_book_id is None:
                                    skipped += 1
                                    continue
                                exclude = [run.source_book_id]
                            else:
                                exclude = []
                            refs = build_reference_set(corpus, exclude_books=exclude)
                            if author not in refs:
                                skipped += 1
                                continue
                            feats = features(seq)
                            if feats is None:
                                skipped += 1
                                continue
                            pick = _attribute(feature, feats, refs)
                            confusion[pick] = confusion.get(pick, 0) + 1
                            tally[author][1] += 1
                            tally[author][0] += pick == author
                    rows += _accuracy_rows(
                        experiment="llm_attribution",
                        author_set=author_set,
                        policy=policy,
                        unit="run",
                        chunk_size=size,
                        feature=feature,
                        tally=tally,
                        confusion=confusion,
                        chance=chance,
                        condition=condition,
                        skipped=skipped,
                    )
    return rows


def _accuracy_rows(
    *,
    experiment: str,
    author_set: str,
    policy: str,
    unit: str,
    chunk_size,
    feature: str,
    tally: dict[str, list[int]],
    confusion: dict[str, int],
    chance: float,
    condition: str = "",
    skipped: int = 0,
) -> list[dict]:
    hits = sum(v[0] for v in tally.values())
    total = sum(v[1] for v in tally.values())
    if total == 0:
        return []
    top = sorted(confusion.items(), key=lambda kv: -kv[1])[:3]
    common = {
        "experiment": experiment,
        "author_set": author_set,
        "condition": condition,
        "policy": policy,
        "unit": unit,
        "chunk_size": chunk_size,
        "feature": feature,
    }
    rows = [
        {
            **common,
            "author": "ALL",
            "n": total,
            "correct": hits,
            "accuracy_pct": 100 * hits / total,
            "chance_pct": 100 * chance,
            "binomial_p_greater_than_chance": float(
                binomtest(hits, total, chance, alternative="greater").pvalue
            ),
            "most_predicted": "; ".join(f"{a}:{n}" for a, n in top),
            "skipped": skipped,
        }
    ]
    for author, (correct, n) in tally.items():
        if n:
            rows.append(
                {
                    **common,
                    "author": author,
                    "n": n,
                    "correct": correct,
                    "accuracy_pct": 100 * correct / n,
                    "chance_pct": 100 * chance,
                    "binomial_p_greater_than_chance": float(
                        binomtest(correct, n, chance, alternative="greater").pvalue
                    ),
                    "most_predicted": "",
                    "skipped": 0,
                }
            )
    return rows


# --------------------------------------------------------------------------
# experiments 5 and 6: detection and chi-square calibration
# --------------------------------------------------------------------------
def exp_detection(
    corpus: Corpus, conditions: dict[str, str], chunk_sizes: list[int]
) -> tuple[list[dict], list[dict]]:
    detection: list[dict] = []
    calibration: list[dict] = []

    for size in chunk_sizes:
        # Human G values under both policies.
        human: dict[str, dict[str, list[float]]] = {
            policy: {f: [] for f in FEATURES} for policy in (POOLED, LOO)
        }
        for author in corpus.authors:
            for book_id in corpus.book_ids(author):
                refs_by_policy = {
                    POOLED: build_reference_set(corpus, exclude_books=()),
                    LOO: build_reference_set(corpus, exclude_books=[book_id]),
                }
                for chunk in chunks(corpus.marks(book_id), size):
                    obs = {"f1": counts1(chunk), "f3": counts2(chunk)}
                    for policy, refs in refs_by_policy.items():
                        if author not in refs:
                            continue
                        if policy == LOO:
                            assert_excluded(refs[author], corpus, book_id)
                        for feature in FEATURES:
                            human[policy][feature].append(
                                g_stat(feature, obs[feature], refs[author].profile(feature))
                            )

        for policy in (POOLED, LOO):
            for feature in FEATURES:
                values = np.asarray(human[policy][feature], dtype=float)
                if values.size == 0:
                    continue
                degrees = dof(feature)
                calibration.append(
                    {
                        "chunk_size": size,
                        "policy": policy,
                        "feature": feature,
                        "n_human_chunks": int(values.size),
                        "mean_G": float(values.mean()),
                        "median_G": float(np.median(values)),
                        "chi2_df": degrees,
                        "overdispersion": float(values.mean() / degrees),
                        "nominal_5pct_rejection_rate": float(
                            np.mean(values > chi2.ppf(0.95, degrees))
                        ),
                    }
                )

        # LLM G values, matched to each human policy.
        for condition in conditions:
            for policy in (POOLED, LOO):
                for feature in FEATURES:
                    llm: list[float] = []
                    skipped = 0
                    for author in corpus.authors:
                        for run in corpus.runs(condition, author):
                            seq = corpus.marks(run.cache_key)[:size]
                            if len(seq) < size:
                                skipped += 1
                                continue
                            if policy == LOO:
                                if run.source_book_id is None:
                                    skipped += 1
                                    continue
                                exclude = [run.source_book_id]
                            else:
                                exclude = []
                            refs = build_reference_set(corpus, exclude_books=exclude)
                            if author not in refs:
                                skipped += 1
                                continue
                            obs = counts1(seq) if feature == "f1" else counts2(seq)
                            llm.append(
                                g_stat(feature, obs, refs[author].profile(feature))
                            )
                    hvals = np.asarray(human[policy][feature], dtype=float)
                    lvals = np.asarray(llm, dtype=float)
                    if lvals.size == 0 or hvals.size == 0:
                        continue
                    rate, threshold = tpr_at_fpr(lvals, hvals, 0.05)
                    detection.append(
                        {
                            "chunk_size": size,
                            "condition": condition,
                            "policy": policy,
                            "feature": feature,
                            "n_human": int(hvals.size),
                            "n_llm": int(lvals.size),
                            "human_median_G": float(np.median(hvals)),
                            "llm_median_G": float(np.median(lvals)),
                            "auc": auc(lvals, hvals),
                            "threshold_at_5pct_fpr": threshold,
                            "tpr_at_5pct_fpr": rate,
                            "mannwhitney_p_llm_greater": float(
                                mannwhitneyu(lvals, hvals, alternative="greater").pvalue
                            ),
                            "skipped": skipped,
                        }
                    )
    return detection, calibration


# --------------------------------------------------------------------------
# experiment 7: per-mark frequencies
# --------------------------------------------------------------------------
def exp_marks(corpus: Corpus, conditions: dict[str, str]) -> list[dict]:
    def shares(seqs: list[list[str]]) -> tuple[dict[str, float], int]:
        counts = np.zeros(len(PUNCT_VECTOR))
        for seq in seqs:
            counts += counts1(seq)
        total = counts.sum()
        return (
            {m: 100 * counts[i] / total for i, m in enumerate(PUNCT_VECTOR)},
            int(total),
        )

    human, human_n = shares(
        [corpus.marks(b) for a in corpus.authors for b in corpus.book_ids(a)]
    )
    per_condition = {
        c: shares([corpus.marks(r.cache_key) for r in corpus.runs(c)])
        for c in conditions
    }

    rows: list[dict] = []
    for mark in PUNCT_VECTOR:
        row = {
            "mark": mark,
            "human_pct": human[mark],
            "human_n_marks": human_n,
        }
        for condition, (share, n) in per_condition.items():
            row[f"{condition}_pct"] = share[mark]
            row[f"{condition}_n_marks"] = n
            row[f"{condition}_human_over_model"] = (
                human[mark] / share[mark] if share[mark] > 0 else float("inf")
            )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# experiment 8: dispersion
# --------------------------------------------------------------------------
def exp_dispersion(
    corpus: Corpus, conditions: dict[str, str], chunk_sizes: list[int]
) -> list[dict]:
    rows: list[dict] = []
    for size in chunk_sizes:
        groups: dict[str, dict[str, list[float]]] = {
            "human_same_book": {f: [] for f in FEATURES},
            "human_across_book": {f: [] for f in FEATURES},
        }
        for condition in conditions:
            groups[f"llm_{condition}_run_pairs"] = {f: [] for f in FEATURES}

        for author in corpus.authors:
            per_book: dict[str, list] = {}
            for book_id in corpus.book_ids(author):
                fs = [features(c) for c in chunks(corpus.marks(book_id), size)]
                per_book[book_id] = [f for f in fs if f is not None]
            for fs in per_book.values():
                for x, y in combinations(fs, 2):
                    for feature in FEATURES:
                        groups["human_same_book"][feature].append(
                            sym_kl(x[feature], y[feature])
                        )
            for b1, b2 in combinations(per_book, 2):
                for x in per_book[b1]:
                    for y in per_book[b2]:
                        for feature in FEATURES:
                            groups["human_across_book"][feature].append(
                                sym_kl(x[feature], y[feature])
                            )

        for condition in conditions:
            for author in corpus.authors:
                fs = []
                for run in corpus.runs(condition, author):
                    seq = corpus.marks(run.cache_key)[:size]
                    if len(seq) < size:
                        continue
                    f = features(seq)
                    if f is not None:
                        fs.append(f)
                for x, y in combinations(fs, 2):
                    for feature in FEATURES:
                        groups[f"llm_{condition}_run_pairs"][feature].append(
                            sym_kl(x[feature], y[feature])
                        )

        for feature in FEATURES:
            same = np.asarray(groups["human_same_book"][feature], dtype=float)
            across = np.asarray(groups["human_across_book"][feature], dtype=float)
            for name, values in groups.items():
                v = np.asarray(values[feature], dtype=float)
                if v.size == 0:
                    continue
                rows.append(
                    {
                        "chunk_size": size,
                        "feature": feature,
                        "group": name,
                        "n_pairs": int(v.size),
                        "mean_sym_kl": float(v.mean()),
                        "median_sym_kl": float(np.median(v)),
                        "ratio_to_human_same_book": float(v.mean() / same.mean())
                        if same.size
                        else float("nan"),
                        "ratio_to_human_across_book": float(v.mean() / across.mean())
                        if across.size
                        else float("nan"),
                    }
                )
    return rows


# --------------------------------------------------------------------------
# experiment 9: positional/context drift within generated texts
# --------------------------------------------------------------------------
def _one_sided_wilcoxon(values: list[float], alternative: str) -> float:
    """One-sample signed-rank p-value, returning 1 when every change is zero."""
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.allclose(array, 0):
        return 1.0
    return float(wilcoxon(array, alternative=alternative).pvalue)


def exp_context_drift(
    corpus: Corpus,
    author_set: str,
    conditions: dict[str, str],
    window_size: int,
    n_windows: int,
) -> tuple[list[dict], list[dict]]:
    """Measure positional drift over equal windows from each generated run.

    Human references use the matched leave-one-book-out policy: the source book
    supplied in the prompt is excluded from the target author's profile. The
    primary repeated-measures outcomes are target KL, target rank, and the
    target margin (nearest-other KL minus target KL; positive means the target
    is nearest). Prediction entropy and cross-target dispersion test the
    complementary "house style" prediction: later windows should become more
    concentrated and more alike across requested authors.

    Windows are not treated as independent samples. First-to-last changes are
    paired within run, then averaged within requested author before the
    signed-rank test, making the ten (or four) authors the inferential units.
    """
    detail: list[dict] = []
    summary: list[dict] = []
    required = window_size * n_windows

    for condition in conditions:
        # Extract each run's windows once; both feature spaces use the result.
        prepared: list[tuple[object, dict, list]] = []
        for author in corpus.authors:
            for run in corpus.runs(condition, author):
                seq = corpus.marks(run.cache_key)
                if len(seq) < required or run.source_book_id is None:
                    continue
                refs = build_reference_set(
                    corpus, exclude_books=[run.source_book_id]
                )
                if author not in refs:
                    continue
                assert_excluded(refs[author], corpus, run.source_book_id)
                window_features = [
                    features(seq[i * window_size : (i + 1) * window_size])
                    for i in range(n_windows)
                ]
                if any(value is None for value in window_features):
                    continue
                prepared.append((run, refs, window_features))

        for feature in FEATURES:
            # Keep feature objects by position for the cross-target dispersion.
            position_features: dict[int, list[tuple[str, object]]] = {
                i + 1: [] for i in range(n_windows)
            }
            feature_rows: list[dict] = []
            position_summaries: list[dict] = []

            for run, refs, window_features in prepared:
                for i, feats in enumerate(window_features, start=1):
                    distances = {
                        candidate: kl(feats[feature], reference[feature])
                        for candidate, reference in refs.items()
                    }
                    target_kl = distances[run.author]
                    other_kl = min(
                        value
                        for candidate, value in distances.items()
                        if candidate != run.author
                    )
                    prediction = min(distances, key=distances.get)
                    target_rank = 1 + sum(
                        value < target_kl
                        for candidate, value in distances.items()
                        if candidate != run.author
                    )
                    row = {
                        "condition": condition,
                        "author_set": author_set,
                        "feature": feature,
                        "window_size": window_size,
                        "window_index": i,
                        "start_mark": (i - 1) * window_size,
                        "end_mark": i * window_size,
                        "target_author": run.author,
                        "run_id": run.run_id,
                        "source_book": run.source_book_id,
                        "target_kl": target_kl,
                        "nearest_other_kl": other_kl,
                        "target_margin": other_kl - target_kl,
                        "target_rank": target_rank,
                        "target_hit": prediction == run.author,
                        "predicted_author": prediction,
                    }
                    detail.append(row)
                    feature_rows.append(row)
                    position_features[i].append((run.author, feats))

            for position in range(1, n_windows + 1):
                rows = [r for r in feature_rows if r["window_index"] == position]
                if not rows:
                    continue
                predictions = Counter(r["predicted_author"] for r in rows)
                probabilities = np.asarray(list(predictions.values()), dtype=float)
                probabilities /= probabilities.sum()
                entropy = -float(np.sum(probabilities * np.log(probabilities)))
                normaliser = np.log(len(corpus.authors))
                pairs = [
                    sym_kl(left[feature], right[feature])
                    for (author_left, left), (author_right, right) in combinations(
                        position_features[position], 2
                    )
                    if author_left != author_right
                ]
                top_author, top_count = predictions.most_common(1)[0]
                position_summary = {
                        "row_type": "position",
                        "condition": condition,
                        "author_set": author_set,
                        "feature": feature,
                        "window_size": window_size,
                        "window_index": position,
                        "n_runs": len(rows),
                        "n_authors": len({r["target_author"] for r in rows}),
                        "mean_target_kl": float(
                            np.mean([r["target_kl"] for r in rows])
                        ),
                        "median_target_kl": float(
                            np.median([r["target_kl"] for r in rows])
                        ),
                        "mean_target_margin": float(
                            np.mean([r["target_margin"] for r in rows])
                        ),
                        "mean_target_rank": float(
                            np.mean([r["target_rank"] for r in rows])
                        ),
                        "target_hit_rate": float(
                            np.mean([r["target_hit"] for r in rows])
                        ),
                        "prediction_entropy_normalized": entropy / normaliser
                        if normaliser
                        else 0.0,
                        "most_predicted_author": top_author,
                        "most_predicted_share": top_count / len(rows),
                        "cross_target_mean_sym_kl": float(np.mean(pairs)),
                        "n_cross_target_pairs": len(pairs),
                    }
                summary.append(position_summary)
                position_summaries.append(position_summary)

            # Paired first-to-last changes. Inferential units are author means,
            # not the five correlated windows or the individual runs.
            by_run: dict[tuple[str, int], dict[int, dict]] = {}
            for row in feature_rows:
                key = (row["target_author"], row["run_id"])
                by_run.setdefault(key, {})[row["window_index"]] = row
            changes: list[dict] = []
            for (author, run_id), values in by_run.items():
                if 1 not in values or n_windows not in values:
                    continue
                first, last = values[1], values[n_windows]
                ordered = [values[i] for i in range(1, n_windows + 1)]
                x = np.arange(n_windows, dtype=float)
                changes.append(
                    {
                        "author": author,
                        "run_id": run_id,
                        "delta_target_kl": last["target_kl"] - first["target_kl"],
                        "delta_target_margin": last["target_margin"]
                        - first["target_margin"],
                        "delta_target_rank": last["target_rank"]
                        - first["target_rank"],
                        "delta_target_hit": int(last["target_hit"])
                        - int(first["target_hit"]),
                        "slope_target_kl": float(
                            np.polyfit(x, [r["target_kl"] for r in ordered], 1)[0]
                        ),
                        "slope_target_margin": float(
                            np.polyfit(
                                x, [r["target_margin"] for r in ordered], 1
                            )[0]
                        ),
                    }
                )
            if not changes:
                continue
            first_position = position_summaries[0]
            last_position = position_summaries[-1]
            author_changes = {
                author: {
                    key: float(np.mean([r[key] for r in changes if r["author"] == author]))
                    for key in (
                        "delta_target_kl",
                        "delta_target_margin",
                        "delta_target_rank",
                        "delta_target_hit",
                    )
                }
                for author in corpus.authors
            }
            summary.append(
                {
                    "row_type": "first_vs_last",
                    "condition": condition,
                    "author_set": author_set,
                    "feature": feature,
                    "window_size": window_size,
                    "first_window": 1,
                    "last_window": n_windows,
                    "n_runs": len(changes),
                    "n_authors": len(author_changes),
                    "mean_delta_target_kl": float(
                        np.mean([r["delta_target_kl"] for r in changes])
                    ),
                    "mean_delta_target_margin": float(
                        np.mean([r["delta_target_margin"] for r in changes])
                    ),
                    "mean_delta_target_rank": float(
                        np.mean([r["delta_target_rank"] for r in changes])
                    ),
                    "target_hit_change_percentage_points": 100
                    * float(np.mean([r["delta_target_hit"] for r in changes])),
                    "mean_per_window_slope_target_kl": float(
                        np.mean([r["slope_target_kl"] for r in changes])
                    ),
                    "mean_per_window_slope_target_margin": float(
                        np.mean([r["slope_target_margin"] for r in changes])
                    ),
                    "prediction_entropy_change": last_position[
                        "prediction_entropy_normalized"
                    ]
                    - first_position["prediction_entropy_normalized"],
                    "cross_target_sym_kl_change_pct": 100
                    * (
                        last_position["cross_target_mean_sym_kl"]
                        / first_position["cross_target_mean_sym_kl"]
                        - 1
                    ),
                    "first_most_predicted_author": first_position[
                        "most_predicted_author"
                    ],
                    "first_most_predicted_share": first_position[
                        "most_predicted_share"
                    ],
                    "last_most_predicted_author": last_position[
                        "most_predicted_author"
                    ],
                    "last_most_predicted_share": last_position[
                        "most_predicted_share"
                    ],
                    "author_wilcoxon_p_target_kl_increase": _one_sided_wilcoxon(
                        [v["delta_target_kl"] for v in author_changes.values()],
                        "greater",
                    ),
                    "author_wilcoxon_p_target_margin_decrease": _one_sided_wilcoxon(
                        [v["delta_target_margin"] for v in author_changes.values()],
                        "less",
                    ),
                    "author_wilcoxon_p_target_rank_worsen": _one_sided_wilcoxon(
                        [v["delta_target_rank"] for v in author_changes.values()],
                        "greater",
                    ),
                    "author_wilcoxon_p_target_hit_decrease": _one_sided_wilcoxon(
                        [v["delta_target_hit"] for v in author_changes.values()],
                        "less",
                    ),
                }
            )

    return detail, summary


# --------------------------------------------------------------------------
# derived: does the margin predict which authors are identifiable?
# --------------------------------------------------------------------------
def exp_margin_vs_accuracy(
    separability: list[dict], attribution: list[dict], chunk_size: int
) -> list[dict]:
    """Rank correlation between the margin and per-author LOO accuracy.

    The margin compares within-author spread against distance to the nearest
    other author, which is close to the quantity the classifier thresholds, so
    this is a diagnostic rather than an independent finding. It is reported with
    its exceptions because a clean threshold at margin = 1 would overstate it.
    """
    rows: list[dict] = []
    for feature in FEATURES:
        margin = {
            r["author"]: r["margin_D_over_C"]
            for r in separability
            if r["feature"] == feature
        }
        accuracy = {
            r["author"]: r["accuracy_pct"]
            for r in attribution
            if r["experiment"] == "human_attribution"
            and r["author_set"] == "all_authors"
            and r["unit"] == "chunk"
            and r["chunk_size"] == chunk_size
            and r["feature"] == feature
            and r["author"] != "ALL"
        }
        shared = [a for a in margin if a in accuracy]
        if len(shared) < 3:
            continue
        rho, pvalue = spearmanr(
            [margin[a] for a in shared], [accuracy[a] for a in shared]
        )
        exceptions = [
            a for a in shared if (margin[a] > 1.0) != (accuracy[a] >= 50.0)
        ]
        for author in shared:
            rows.append(
                {
                    "feature": feature,
                    "chunk_size": chunk_size,
                    "author": author,
                    "margin_D_over_C": margin[author],
                    "loo_accuracy_pct": accuracy[author],
                    "margin_gt_1": margin[author] > 1.0,
                    "accuracy_ge_50": accuracy[author] >= 50.0,
                    "threshold_exception": author in exceptions,
                    "spearman_rho": float(rho),
                    "spearman_p": float(pvalue),
                    "n_threshold_exceptions": len(exceptions),
                }
            )
    return rows


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [wrote] {path.relative_to(ROOT)}  ({len(rows)} rows)")


def _audit_rows() -> list[dict]:
    """One row per distinct profile, with the experiments that requested it.

    The raw log has an entry per call, which runs to tens of thousands of
    identical rows once cached profiles are re-requested inside chunk loops.
    Collapsing to distinct (author, included, excluded) triples is what makes the
    exclusion policy of the run readable, which is the point of keeping the log.
    """
    grouped: dict[tuple, dict] = {}
    for entry in audit_log():
        key = (entry.author, entry.included_books, entry.excluded_books)
        row = grouped.get(key)
        if row is None:
            grouped[key] = {
                "author": entry.author,
                "n_marks": entry.n_marks,
                "n_included_books": len(entry.included_books),
                "included_books": "; ".join(entry.included_books),
                "excluded_books": "; ".join(entry.excluded_books),
                "times_requested": 1,
                "callers": {entry.caller.split(":")[0]},
            }
        else:
            row["times_requested"] += 1
            row["callers"].add(entry.caller.split(":")[0])
    rows = []
    for row in grouped.values():
        row["callers"] = "; ".join(sorted(row["callers"]))
        rows.append(row)
    return sorted(rows, key=lambda r: (r["author"], r["excluded_books"]))


def git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                args, cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def main() -> None:
    args = parse_args()
    conditions = dict(item.split("=", 1) for item in args.condition)
    outdir = ROOT / args.out
    reset_audit_log()

    corpus = load_corpus(args.authors_config, args.cache)
    for name, directory in conditions.items():
        corpus.load_runs(name, directory)

    print(f"corpus: {len(corpus.authors)} authors, {len(corpus.books())} books")
    for author in corpus.authors:
        books = corpus.books(author)
        detail = " ".join(f"{Path(b.book_id).name}({b.n_marks})" for b in books)
        aliased = [a for b in books for a in b.aliases]
        note = f"  [duplicate aliases merged: {', '.join(aliased)}]" if aliased else ""
        print(f"  {author:24s} {len(books)} books: {detail}{note}")
    for name in conditions:
        runs = corpus.runs(name)
        unmapped = sum(1 for r in runs if r.source_book_id is None)
        print(f"  condition {name}: {len(runs)} runs, {unmapped} without a source book")

    print("\n[1/9] replication")
    replication = exp_replication(corpus, args.headline_chunk_size)
    print("[2/9] separability")
    separability, subset = exp_separability(corpus)
    print(f"      margin > 1 subset ({len(subset)}): {', '.join(subset)}")

    subsets = {"all_authors": corpus}
    if subset and len(subset) < len(corpus.authors):
        subsets["margin_gt_1"] = corpus.subset(subset)

    print("[3/9] human attribution")
    human_rows: list[dict] = []
    for label, view in subsets.items():
        human_rows += exp_human_attribution(view, label, args.chunk_sizes)

    print("[4/9] llm attribution")
    llm_rows: list[dict] = []
    for label, view in subsets.items():
        llm_rows += exp_llm_attribution(view, label, conditions, args.chunk_sizes)

    print("[5/9] detection  [6/9] calibration")
    detection, calibration = exp_detection(corpus, conditions, args.chunk_sizes)

    print("[7/9] mark frequencies")
    marks = exp_marks(corpus, conditions)

    print("[8/9] dispersion")
    dispersion = exp_dispersion(corpus, conditions, args.chunk_sizes)

    print("[9/9] context drift")
    context_drift: list[dict] = []
    context_drift_summary: list[dict] = []
    for label, view in subsets.items():
        detail, summary = exp_context_drift(
            view,
            label,
            conditions,
            args.drift_window_size,
            args.drift_windows,
        )
        context_drift += detail
        context_drift_summary += summary

    attribution = human_rows + llm_rows
    margin_vs_accuracy = exp_margin_vs_accuracy(
        separability, attribution, args.headline_chunk_size
    )

    print()
    write_csv(outdir / "replication.csv", replication)
    write_csv(outdir / "separability.csv", separability)
    write_csv(outdir / "margin_vs_accuracy.csv", margin_vs_accuracy)
    write_csv(outdir / "attribution.csv", attribution)
    write_csv(outdir / "detection.csv", detection)
    write_csv(outdir / "calibration.csv", calibration)
    write_csv(outdir / "mark_frequencies.csv", marks)
    write_csv(outdir / "dispersion.csv", dispersion)
    write_csv(outdir / "context_drift.csv", context_drift)
    write_csv(outdir / "context_drift_summary.csv", context_drift_summary)
    write_csv(outdir / "reference_audit.csv", _audit_rows())

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "authors_config": args.authors_config,
        "cache_path": str(corpus.cache_path.relative_to(ROOT)),
        "cache_sha256": corpus.cache_sha256,
        "conditions": conditions,
        "chunk_sizes": args.chunk_sizes,
        "headline_chunk_size": args.headline_chunk_size,
        "drift_window_size": args.drift_window_size,
        "drift_windows": args.drift_windows,
        "authors": list(corpus.authors),
        "margin_gt_1_subset": subset,
        "n_reference_profiles_built": len(audit_log()),
        "punctuation_vector": list(PUNCT_VECTOR),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  [wrote] {(outdir / 'manifest.json').relative_to(ROOT)}")

    _summary(
        args,
        subset,
        replication,
        attribution,
        detection,
        calibration,
        marks,
        dispersion,
        margin_vs_accuracy,
        context_drift_summary,
    )


def _summary(
    args,
    subset,
    replication,
    attribution,
    detection,
    calibration,
    marks,
    dispersion,
    margin_vs_accuracy,
    context_drift_summary,
) -> None:
    size = args.headline_chunk_size
    print(f"\n{'='*78}\nheadline numbers (f3, chunk_size={size})\n{'='*78}")

    for row in replication:
        if row["feature"] == "f3":
            print(
                f"  separation ratio, {row['unit']:16s}: {row['separation_ratio']:.2f}"
                f"  (within {row['within_mean']:.4f}, between {row['between_mean']:.4f})"
            )

    def find(rows, **kw):
        return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]

    for label in ("all_authors", "margin_gt_1"):
        got = find(
            attribution,
            experiment="human_attribution",
            author_set=label,
            unit="chunk",
            chunk_size=size,
            feature="f3",
            author="ALL",
        )
        for row in got:
            print(
                f"  human LOO attribution, {label:12s}: {row['accuracy_pct']:5.1f}%"
                f"  (chance {row['chance_pct']:.1f}%, n={row['n']})"
            )
        for row in find(
            attribution,
            experiment="llm_attribution",
            author_set=label,
            unit="run",
            chunk_size=size,
            feature="f3",
            author="ALL",
            policy=LOO,
        ):
            print(
                f"  LLM attribution [{row['condition']}], {label:12s}: "
                f"{row['accuracy_pct']:5.1f}%  (chance {row['chance_pct']:.1f}%, "
                f"n={row['n']}, p={row['binomial_p_greater_than_chance']:.4f})"
            )

    for row in detection:
        if row["chunk_size"] == size and row["feature"] == "f3":
            print(
                f"  detection [{row['condition']}] {row['policy']:18s}: "
                f"AUC={row['auc']:.3f}  TPR@5%FPR={100*row['tpr_at_5pct_fpr']:5.1f}%"
            )

    for row in calibration:
        if row["chunk_size"] == size and row["policy"] == LOO:
            print(
                f"  calibration {row['feature']}: overdispersion x{row['overdispersion']:.1f}"
                f"  nominal-5% rejection {100*row['nominal_5pct_rejection_rate']:.0f}%"
            )

    semis = [r for r in marks if r["mark"] == ";"]
    if semis:
        row = semis[0]
        parts = [f"human {row['human_pct']:.2f}%"]
        for key in row:
            if key.endswith("_pct") and key != "human_pct":
                parts.append(f"{key[:-4]} {row[key]:.2f}%")
        print(f"  semicolon share: {', '.join(parts)}")

    for row in dispersion:
        if row["chunk_size"] == size and row["feature"] == "f3" and row["group"].startswith("llm_"):
            print(
                f"  dispersion {row['group']:24s}: "
                f"{row['ratio_to_human_same_book']:.2f}x same-book human, "
                f"{row['ratio_to_human_across_book']:.2f}x across-book human"
            )

    f3_margin = [r for r in margin_vs_accuracy if r["feature"] == "f3"]
    if f3_margin:
        row = f3_margin[0]
        exceptions = [r["author"] for r in f3_margin if r["threshold_exception"]]
        print(
            f"  margin vs accuracy: Spearman rho={row['spearman_rho']:.3f}"
            f" p={row['spearman_p']:.4f}, "
            f"{row['n_threshold_exceptions']} exception(s) to a margin=1 threshold"
            + (f": {', '.join(exceptions)}" if exceptions else "")
        )

    for row in context_drift_summary:
        if (
            row["row_type"] == "first_vs_last"
            and row["author_set"] == "all_authors"
            and row["feature"] == "f3"
        ):
            print(
                f"  context drift [{row['condition']}], first→last "
                f"{row['window_size']}-mark window: "
                f"ΔKL={row['mean_delta_target_kl']:+.4f}, "
                f"Δmargin={row['mean_delta_target_margin']:+.4f}, "
                f"Δrank={row['mean_delta_target_rank']:+.2f}, "
                f"Δhit={row['target_hit_change_percentage_points']:+.1f}pp "
                f"cross-target={row['cross_target_sym_kl_change_pct']:+.1f}%, "
                f"entropy={row['prediction_entropy_change']:+.3f} "
                f"(author-level p[KL↑]="
                f"{row['author_wilcoxon_p_target_kl_increase']:.4f})"
            )

    print(f"\n  margin > 1 subset: {', '.join(subset)}")
    print(f"  results: {args.out}")


if __name__ == "__main__":
    main()
