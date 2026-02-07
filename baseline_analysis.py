#!/usr/bin/env python3
"""
Baseline analysis: compare LLM-generated text against random authors.

Q1: Does the LLM move toward the target author?
    KL(LLM-Austen, Real-Austen) vs KL(LLM-Austen, Random_i) for 47 random authors
    If the prompt works, LLM-Austen should be CLOSER to Real-Austen than to randoms.

Q2: Is LLM-Austen as close to Austen as random authors are?
    KL(LLM-Austen, Real-Austen) vs KL(Random_i, Real-Austen)
    If they're similar, the LLM is no better at being Austen than any random author.

Both questions use Option A (empirical random authors: 47 Gutenberg texts).

Usage:
    python baseline_analysis.py
"""

import sys
import json
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# punctuation-stylometry-master imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PUNCT_STYLOMETRY_DIR = _SCRIPT_DIR / 'punctuation-stylometry-master'
sys.path.insert(0, str(_PUNCT_STYLOMETRY_DIR))

_config_path = str(_PUNCT_STYLOMETRY_DIR / 'conf' / 'punctuation.ini')
sys.argv = [sys.argv[0], '-c', _config_path]

from punctuation.config import options
from punctuation.parser.punctuation_parser import (
    get_textinfo, seq_pun_only, get_frequencies,
)
from punctuation.feature_operations.matrix_operations import (
    transition_mat, normalised_transition_mat,
)
from punctuation.feature_operations.distances import d_KL

PUNCTUATION_VECTOR = options.punctuation_vector
CHUNK = 2000
MIN_MARKS = 2000

AUTHORS = {
    'jane_austen': 'Jane Austen',
    'william_shakespeare': 'William Shakespeare',
    'herbert_george_wells': 'H.G. Wells',
    'agnes_may_fleming': 'Agnes May Fleming',
}

FULL_BOOKS_DIR = Path('full_books')
GUTENBERG_DIR = Path('gutenberg_texts')
GENERATED_DIR = Path('generated_texts')
OUTPUT_DIR = Path('baseline_results')


# =============================================================================
# HELPERS
# =============================================================================

def extract_punc(text):
    text = text.replace('...', '^')
    return seq_pun_only(get_textinfo(text))


def compute_features(seq):
    f1 = get_frequencies(seq, vector=PUNCTUATION_VECTOR)
    if f1 is None or sum(f1) == 0:
        return None
    f2 = transition_mat(seq)
    if f2 is None:
        return None
    f3 = normalised_transition_mat(f2, f1)
    # Flatten f3 to a 100-element vector (joint probability distribution),
    # consistent with the paper & library: dKL is applied to flattened f3.
    f3_flat = f3.flatten().tolist()
    return {'f1': f1, 'f3': f3_flat}


def get_chunk(seq, position='middle'):
    """Extract a CHUNK-sized piece from a punctuation sequence."""
    if len(seq) < CHUNK:
        return None
    if position == 'middle':
        start = (len(seq) - CHUNK) // 2
    else:
        start = 0
    return seq[start:start + CHUNK]


# =============================================================================
# LOAD DATA
# =============================================================================

def load_random_authors():
    """Load 2000-mark chunks and features from all usable Gutenberg texts."""
    random_authors = []
    for f in sorted(GUTENBERG_DIR.glob('*.txt')):
        text = f.read_text(encoding='utf-8', errors='ignore')
        seq = extract_punc(text)
        if seq is None or len(seq) < MIN_MARKS:
            continue
        chunk = get_chunk(seq, position='middle')
        feats = compute_features(chunk)
        if feats is not None:
            random_authors.append({
                'file': f.name,
                'f1': feats['f1'],
                'f3': feats['f3'],
            })
    return random_authors


def load_real_author(author_key):
    """Load features from a real author's full book (middle chunk)."""
    text = (FULL_BOOKS_DIR / f"{author_key}_full.txt").read_text()
    seq = extract_punc(text)
    chunk = get_chunk(seq, position='middle')
    return compute_features(chunk)


def load_llm_runs(author_key):
    """Load features from all LLM-generated runs for an author."""
    runs = []
    run_dir = GENERATED_DIR / author_key
    for f in sorted(run_dir.glob('run_*.txt')):
        text = f.read_text()
        seq = extract_punc(text)
        if seq is None or len(seq) < CHUNK:
            continue
        chunk = get_chunk(seq, position='start')
        feats = compute_features(chunk)
        if feats is not None:
            runs.append(feats)
    return runs


# =============================================================================
# ANALYSIS
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading 47 random Gutenberg authors...")
    random_authors = load_random_authors()
    print(f"  Loaded {len(random_authors)} random authors\n")

    all_results = {}

    for author_key, author_name in AUTHORS.items():
        print(f"{'='*60}")
        print(f"  {author_name}")
        print(f"{'='*60}")

        real_feats = load_real_author(author_key)
        llm_runs = load_llm_runs(author_key)
        print(f"  LLM runs loaded: {len(llm_runs)}")

        # ------------------------------------------------------------------
        # Q1: Does the LLM move toward the target?
        #     KL(LLM, Real) vs KL(LLM, Random_i)
        # ------------------------------------------------------------------
        # KL(LLM, Real) — one value per run
        llm_vs_real_f1 = []
        llm_vs_real_f3 = []
        for run in llm_runs:
            llm_vs_real_f1.append(d_KL(run['f1'], real_feats['f1']))
            llm_vs_real_f3.append(d_KL(run['f3'], real_feats['f3']))

        # KL(LLM, Random_i) — for each run x each random author
        llm_vs_random_f1 = []
        llm_vs_random_f3 = []
        for run in llm_runs:
            for ra in random_authors:
                llm_vs_random_f1.append(d_KL(run['f1'], ra['f1']))
                llm_vs_random_f3.append(d_KL(run['f3'], ra['f3']))

        print(f"\n  Q1: Does the LLM move toward {author_name}?")
        print(f"    KL(LLM, Real-{author_name[:8]})  f1: {np.mean(llm_vs_real_f1):.4f} ± {np.std(llm_vs_real_f1):.4f}")
        print(f"    KL(LLM, Random)            f1: {np.mean(llm_vs_random_f1):.4f} ± {np.std(llm_vs_random_f1):.4f}")
        print(f"    KL(LLM, Real-{author_name[:8]})  f3: {np.mean(llm_vs_real_f3):.4f} ± {np.std(llm_vs_real_f3):.4f}")
        print(f"    KL(LLM, Random)            f3: {np.mean(llm_vs_random_f3):.4f} ± {np.std(llm_vs_random_f3):.4f}")

        closer_f1 = np.mean(llm_vs_real_f1) < np.mean(llm_vs_random_f1)
        closer_f3 = np.mean(llm_vs_real_f3) < np.mean(llm_vs_random_f3)
        print(f"    => LLM closer to target than random? f1: {closer_f1}, f3: {closer_f3}")

        # ------------------------------------------------------------------
        # Q2: Is LLM-Author as close to Author as random authors are?
        #     KL(LLM, Real) vs KL(Random_i, Real)
        # ------------------------------------------------------------------
        random_vs_real_f1 = []
        random_vs_real_f3 = []
        for ra in random_authors:
            random_vs_real_f1.append(d_KL(ra['f1'], real_feats['f1']))
            random_vs_real_f3.append(d_KL(ra['f3'], real_feats['f3']))

        print(f"\n  Q2: Is LLM-{author_name[:8]} closer to {author_name[:8]} than random authors are?")
        print(f"    KL(LLM, Real-{author_name[:8]})     f1: {np.mean(llm_vs_real_f1):.4f} ± {np.std(llm_vs_real_f1):.4f}")
        print(f"    KL(Random, Real-{author_name[:8]})   f1: {np.mean(random_vs_real_f1):.4f} ± {np.std(random_vs_real_f1):.4f}")
        print(f"    KL(LLM, Real-{author_name[:8]})     f3: {np.mean(llm_vs_real_f3):.4f} ± {np.std(llm_vs_real_f3):.4f}")
        print(f"    KL(Random, Real-{author_name[:8]})   f3: {np.mean(random_vs_real_f3):.4f} ± {np.std(random_vs_real_f3):.4f}")

        llm_better_f1 = np.mean(llm_vs_real_f1) < np.mean(random_vs_real_f1)
        llm_better_f3 = np.mean(llm_vs_real_f3) < np.mean(random_vs_real_f3)
        print(f"    => LLM better than random at matching target? f1: {llm_better_f1}, f3: {llm_better_f3}")

        print()

        all_results[author_key] = {
            'author_name': author_name,
            'n_llm_runs': len(llm_runs),
            'n_random_authors': len(random_authors),
            'Q1': {
                'llm_vs_real_f1': {'mean': float(np.mean(llm_vs_real_f1)), 'std': float(np.std(llm_vs_real_f1)),
                                    'values': [float(v) for v in llm_vs_real_f1]},
                'llm_vs_real_f3': {'mean': float(np.mean(llm_vs_real_f3)), 'std': float(np.std(llm_vs_real_f3)),
                                    'values': [float(v) for v in llm_vs_real_f3]},
                'llm_vs_random_f1': {'mean': float(np.mean(llm_vs_random_f1)), 'std': float(np.std(llm_vs_random_f1))},
                'llm_vs_random_f3': {'mean': float(np.mean(llm_vs_random_f3)), 'std': float(np.std(llm_vs_random_f3))},
            },
            'Q2': {
                'llm_vs_real_f1': {'mean': float(np.mean(llm_vs_real_f1)), 'std': float(np.std(llm_vs_real_f1))},
                'llm_vs_real_f3': {'mean': float(np.mean(llm_vs_real_f3)), 'std': float(np.std(llm_vs_real_f3))},
                'random_vs_real_f1': {'mean': float(np.mean(random_vs_real_f1)), 'std': float(np.std(random_vs_real_f1)),
                                       'values': [float(v) for v in random_vs_real_f1]},
                'random_vs_real_f3': {'mean': float(np.mean(random_vs_real_f3)), 'std': float(np.std(random_vs_real_f3)),
                                       'values': [float(v) for v in random_vs_real_f3]},
            },
        }

    # Save results
    with open(OUTPUT_DIR / 'baseline_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ------------------------------------------------------------------
    # FINAL SUMMARY TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nQ1: Does the LLM move toward the target? (lower = closer)")
    print(f"{'Author':<20} {'KL(LLM,Target) f1':>20} {'KL(LLM,Random) f1':>20} {'Closer?':>8}")
    print("-" * 72)
    for key, name in AUTHORS.items():
        r = all_results[key]['Q1']
        closer = "YES" if r['llm_vs_real_f1']['mean'] < r['llm_vs_random_f1']['mean'] else "NO"
        print(f"{name:<20} {r['llm_vs_real_f1']['mean']:>17.4f} {r['llm_vs_random_f1']['mean']:>20.4f} {closer:>8}")

    print(f"\n{'Author':<20} {'KL(LLM,Target) f3':>20} {'KL(LLM,Random) f3':>20} {'Closer?':>8}")
    print("-" * 72)
    for key, name in AUTHORS.items():
        r = all_results[key]['Q1']
        closer = "YES" if r['llm_vs_real_f3']['mean'] < r['llm_vs_random_f3']['mean'] else "NO"
        print(f"{name:<20} {r['llm_vs_real_f3']['mean']:>17.4f} {r['llm_vs_random_f3']['mean']:>20.4f} {closer:>8}")

    print("\nQ2: Is LLM closer to target than random authors are? (lower = better)")
    print(f"{'Author':<20} {'KL(LLM,Target) f1':>20} {'KL(Rand,Target) f1':>20} {'Better?':>8}")
    print("-" * 72)
    for key, name in AUTHORS.items():
        r = all_results[key]
        llm = r['Q1']['llm_vs_real_f1']['mean']
        rand = r['Q2']['random_vs_real_f1']['mean']
        better = "YES" if llm < rand else "NO"
        print(f"{name:<20} {llm:>17.4f} {rand:>20.4f} {better:>8}")

    print(f"\n{'Author':<20} {'KL(LLM,Target) f3':>20} {'KL(Rand,Target) f3':>20} {'Better?':>8}")
    print("-" * 72)
    for key, name in AUTHORS.items():
        r = all_results[key]
        llm = r['Q1']['llm_vs_real_f3']['mean']
        rand = r['Q2']['random_vs_real_f3']['mean']
        better = "YES" if llm < rand else "NO"
        print(f"{name:<20} {llm:>17.4f} {rand:>20.4f} {better:>8}")

    print(f"\nResults saved to {OUTPUT_DIR / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
