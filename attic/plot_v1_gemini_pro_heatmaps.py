#!/usr/bin/env python3
"""
Heatmap: Real vs v1 Gemini Pro — for Jane Austen (and other authors as available).

Columns: Real (2000 marks) | v1 Flash concat (200x10) | v1 Pro concat (200x10)
         | v1 Pro single run (2000 marks)

Usage:
    python plot_v1_gemini_pro_heatmaps.py
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
FLASH_DIR = Path('generated_texts')         # v1 Flash
PRO_DIR = Path('generated_texts_v1_gemini_pro')  # v1 Pro
OUTPUT_DIR = Path('version_comparison_plots')

AUTHORS = {
    'jane_austen': 'Jane Austen',
    'william_shakespeare': 'William Shakespeare',
    'herbert_george_wells': 'H.G. Wells',
    'agnes_may_fleming': 'Agnes May Fleming',
}


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


def concat_runs(run_dir, n_runs=10, marks_per_run=200):
    """Concatenate first `marks_per_run` marks from each run."""
    concat_seq = []
    for i in range(1, n_runs + 1):
        run_path = run_dir / f'run_{i:02d}.txt'
        if run_path.exists():
            gen_text = load_text(run_path)
            gen_seq = extract_punctuation(gen_text)
            if gen_seq and len(gen_seq) >= marks_per_run:
                concat_seq.extend(gen_seq[:marks_per_run])
            elif gen_seq:
                print(f"  Warning: {run_path.name} has only {len(gen_seq)} marks")
                concat_seq.extend(gen_seq)
    return concat_seq


def single_run(run_dir, marks=2000):
    """Get first `marks` from run_01."""
    run_path = run_dir / 'run_01.txt'
    if run_path.exists():
        gen_text = load_text(run_path)
        gen_seq = extract_punctuation(gen_text)
        if gen_seq and len(gen_seq) >= marks:
            return gen_seq[:marks]
        elif gen_seq:
            return gen_seq
    return []


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Only include authors that have Pro results
    authors_with_pro = {}
    for key, name in AUTHORS.items():
        pro_dir = PRO_DIR / key
        if pro_dir.exists() and any(pro_dir.glob('run_*.txt')):
            authors_with_pro[key] = name

    if not authors_with_pro:
        print("No v1 Gemini Pro results found!")
        return

    n_authors = len(authors_with_pro)
    n_cols = 4  # Real | Flash concat | Pro concat | Pro single
    fig, axes = plt.subplots(n_authors, n_cols,
                             figsize=(22, 3.5 * n_authors + 1.5))

    # Handle single-author case (axes is 1D)
    if n_authors == 1:
        axes = axes.reshape(1, -1)

    for row, (author_key, author_name) in enumerate(authors_with_pro.items()):
        # Real text (2000-mark middle chunk)
        real_text = load_text(FULL_BOOKS_DIR / f'{author_key}_full.txt')
        real_seq = extract_punctuation(real_text)
        real_mid = (len(real_seq) - 2000) // 2
        real_chunk = real_seq[real_mid:real_mid + 2000]

        # v1 Flash concat
        flash_concat = concat_runs(FLASH_DIR / author_key)
        # v1 Pro concat
        pro_concat = concat_runs(PRO_DIR / author_key)
        # v1 Pro single
        pro_single = single_run(PRO_DIR / author_key)

        print(f"  {author_name}:")
        print(f"    Real: {len(real_chunk)} marks")
        print(f"    Flash concat: {len(flash_concat)} marks")
        print(f"    Pro concat: {len(pro_concat)} marks")
        print(f"    Pro single: {len(pro_single)} marks")

        # Plot
        plot_heatmap(axes[row, 0], real_chunk, width=50)

        if flash_concat:
            plot_heatmap(axes[row, 1], flash_concat, width=50)
        else:
            axes[row, 1].text(0.5, 0.5, 'No data', ha='center', va='center',
                              transform=axes[row, 1].transAxes, fontsize=10, color='gray')
            axes[row, 1].set_xticks([])
            axes[row, 1].set_yticks([])

        if pro_concat:
            plot_heatmap(axes[row, 2], pro_concat, width=50)
        else:
            axes[row, 2].text(0.5, 0.5, 'No data', ha='center', va='center',
                              transform=axes[row, 2].transAxes, fontsize=10, color='gray')
            axes[row, 2].set_xticks([])
            axes[row, 2].set_yticks([])

        if pro_single:
            plot_heatmap(axes[row, 3], pro_single, width=50)
        else:
            axes[row, 3].text(0.5, 0.5, 'No data', ha='center', va='center',
                              transform=axes[row, 3].transAxes, fontsize=10, color='gray')
            axes[row, 3].set_xticks([])
            axes[row, 3].set_yticks([])

        axes[row, 0].set_ylabel(author_name, fontsize=10, fontweight='bold',
                                rotation=90, labelpad=10)

    # Column headers
    axes[0, 0].set_title('Real\n(2000 marks, middle of book)',
                         fontsize=11, fontweight='bold')
    axes[0, 1].set_title('v1 Flash (concat)\n(200 marks x 10 runs)',
                         fontsize=11, fontweight='bold')
    axes[0, 2].set_title('v1 Pro (concat)\n(200 marks x 10 runs)',
                         fontsize=11, fontweight='bold')
    axes[0, 3].set_title('v1 Pro (single run)\n(first 2000 marks, run 1)',
                         fontsize=11, fontweight='bold')

    fig.suptitle(
        'Punctuation Heatmaps: Real vs v1 Flash vs v1 Gemini 2.5 Pro',
        fontsize=14, fontweight='bold', y=0.98,
    )

    # Legend
    patches = [mpatches.Patch(color=PUNCTUATION_COLORS[m], label=PUNC_NAMES[m])
               for m in PUNCTUATION_VECTOR]
    fig.legend(handles=patches, loc='lower center',
               ncol=len(PUNCTUATION_VECTOR), fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, 0.0))

    fig.subplots_adjust(bottom=0.05, top=0.90, left=0.10, right=0.98,
                        wspace=0.05, hspace=0.15)

    out_path = OUTPUT_DIR / 'v1_flash_vs_pro_heatmaps.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
