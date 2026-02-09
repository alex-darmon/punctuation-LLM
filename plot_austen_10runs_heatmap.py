#!/usr/bin/env python3
"""
Heatmap: Real (2000 marks) vs concatenated first 200 marks from each of 10 v1 runs.

Produces a combined figure with all authors (one row per author, 2 columns: real vs concat).

Usage:
    python plot_austen_10runs_heatmap.py
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PUNCT_STYLOMETRY_DIR = _SCRIPT_DIR / 'punctuation-stylometry-master'
sys.path.insert(0, str(_PUNCT_STYLOMETRY_DIR))

_config_path = str(_PUNCT_STYLOMETRY_DIR / 'conf' / 'punctuation.ini')
if '-c' not in sys.argv and '--config' not in sys.argv:
    sys.argv = [sys.argv[0], '-c', _config_path] + sys.argv[1:]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import gridspec

from punctuation.config import options
from punctuation.parser.punctuation_parser import get_textinfo, seq_pun_only

PUNCTUATION_VECTOR = options.punctuation_vector

PUNCTUATION_COLORS = {
    '!': '#FF0000', '"': '#0000FF', '(': '#00FF00', ')': '#FF00FF',
    ',': '#1E90FF', '.': '#FFD700', ':': '#00CED1', ';': '#8B008B',
    '?': '#32CD32', '^': '#000000',
}
COLORS = [PUNCTUATION_COLORS[p] for p in PUNCTUATION_VECTOR]
CMAP = ListedColormap(COLORS)

PUNC_NAMES = {
    '!': 'Excl', '"': 'Quote', '(': 'L.Par', ')': 'R.Par',
    ',': 'Comma', '.': 'Period', ':': 'Colon', ';': 'Semi',
    '?': 'Quest', '^': 'Otr',
}

FULL_BOOKS_DIR = Path('full_books')
GENERATED_DIR = Path('generated_texts')
OUTPUT_DIR = Path('version_comparison_plots')


def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def extract_punctuation(text):
    if text is None:
        return None
    text = text.replace('...', '^')
    text_info = get_textinfo(text)
    return seq_pun_only(text_info)


def punctuation_to_numeric(punc_seq):
    return np.array([PUNCTUATION_VECTOR.index(p) for p in punc_seq])


def plot_heatmap(ax, punc_seq, width=50):
    numeric_seq = punctuation_to_numeric(punc_seq)
    height = max(1, len(numeric_seq) // width)

    if height * width < len(numeric_seq):
        padded = np.full(width * (height + 1), np.nan)
        padded[:len(numeric_seq)] = numeric_seq
        numeric_seq = padded
        height += 1

    grid = numeric_seq[:height * width].reshape(height, width)
    ax.imshow(grid, cmap=CMAP, aspect='auto',
              vmin=-0.5, vmax=len(PUNCTUATION_VECTOR) - 0.5,
              interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])


AUTHORS = {
    'jane_austen': 'Jane Austen',
    'william_shakespeare': 'William Shakespeare',
    'herbert_george_wells': 'H.G. Wells',
    'agnes_may_fleming': 'Agnes May Fleming',
}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    n_authors = len(AUTHORS)
    fig, axes = plt.subplots(n_authors, 3, figsize=(18, 3.5 * n_authors + 1.5))

    for row, (author_key, author_name) in enumerate(AUTHORS.items()):
        # Real text (2000-mark middle chunk)
        real_text = load_text(FULL_BOOKS_DIR / f'{author_key}_full.txt')
        real_seq = extract_punctuation(real_text)
        real_mid = (len(real_seq) - 2000) // 2
        real_chunk = real_seq[real_mid:real_mid + 2000]

        # Concatenate first 200 marks from each of 10 v1 runs
        concat_seq = []
        run_dir = GENERATED_DIR / author_key
        for i in range(1, 11):
            run_path = run_dir / f'run_{i:02d}.txt'
            if run_path.exists():
                gen_text = load_text(run_path)
                gen_seq = extract_punctuation(gen_text)
                if gen_seq and len(gen_seq) >= 200:
                    concat_seq.extend(gen_seq[:200])
                else:
                    print(f"  Warning: {author_key} run {i} has only "
                          f"{len(gen_seq) if gen_seq else 0} marks")

        print(f"  {author_name}: concatenated {len(concat_seq)} marks "
              f"from {len(concat_seq)//200} runs x 200")

        # Single full run (run_01, first 2000 marks)
        run1_path = run_dir / 'run_01.txt'
        single_seq = []
        if run1_path.exists():
            gen_text = load_text(run1_path)
            gen_seq = extract_punctuation(gen_text)
            if gen_seq and len(gen_seq) >= 2000:
                single_seq = gen_seq[:2000]
            elif gen_seq:
                single_seq = gen_seq

        # Plot
        ax_real = axes[row, 0]
        ax_concat = axes[row, 1]
        ax_single = axes[row, 2]

        plot_heatmap(ax_real, real_chunk, width=50)
        plot_heatmap(ax_concat, concat_seq, width=50)
        if single_seq:
            plot_heatmap(ax_single, single_seq, width=50)
        else:
            ax_single.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax_single.transAxes, fontsize=10, color='gray')
            ax_single.set_xticks([])
            ax_single.set_yticks([])

        ax_real.set_ylabel(author_name, fontsize=10, fontweight='bold',
                           rotation=90, labelpad=10)

    # Column headers
    axes[0, 0].set_title('Real (2000 marks, middle of book)',
                         fontsize=11, fontweight='bold')
    axes[0, 1].set_title('LLM v1: 10 runs concatenated\n(first 200 marks each = 2000 total)',
                         fontsize=11, fontweight='bold')
    axes[0, 2].set_title('LLM v1: single run\n(first 2000 marks from run 1)',
                         fontsize=11, fontweight='bold')

    fig.suptitle(
        'Real vs LLM-Generated Punctuation (2000 marks each)',
        fontsize=14, fontweight='bold', y=0.98,
    )

    # Legend
    patches = [mpatches.Patch(color=PUNCTUATION_COLORS[m], label=PUNC_NAMES[m])
               for m in PUNCTUATION_VECTOR]
    fig.legend(handles=patches, loc='lower center',
               ncol=len(PUNCTUATION_VECTOR), fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, 0.0))

    fig.subplots_adjust(bottom=0.05, top=0.92, left=0.10, right=0.98,
                        wspace=0.05, hspace=0.15)

    out_path = OUTPUT_DIR / 'all_authors_concat_200marks_heatmap.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
