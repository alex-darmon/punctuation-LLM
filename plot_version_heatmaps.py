#!/usr/bin/env python3
"""
Generate heatmap comparison across all prompt versions (v1, v2, v3) and real text.

Produces one figure per author with 4 columns: Real, v1 (excerpt), v2 (excerpt+targets), v3 (name only).
Also produces a combined figure with all authors.

Usage:
    python plot_version_heatmaps.py
"""

import sys
from pathlib import Path

# Add punctuation-stylometry-master to path
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

# =============================================================================
# CONFIG
# =============================================================================

AUTHORS = {
    'jane_austen': 'Jane Austen',
    'william_shakespeare': 'William Shakespeare',
    'herbert_george_wells': 'H.G. Wells',
    'agnes_may_fleming': 'Agnes May Fleming',
}

FULL_BOOKS_DIR = Path('full_books')
CHUNK_SIZE = 2000

VERSIONS = {
    'v1': {'dir': Path('generated_texts'), 'label': 'v1 (excerpt)'},
    'v2': {'dir': Path('generated_texts_v2'), 'label': 'v2 (excerpt + targets)'},
    'v3': {'dir': Path('generated_texts_v3'), 'label': 'v3 (name only)'},
}

OUTPUT_DIR = Path('version_comparison_plots')

# Colour mapping (same as chunk_size_analysis.py)
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


# =============================================================================
# HELPERS
# =============================================================================

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


def get_chunk(seq, chunk_size, position='middle'):
    if len(seq) < chunk_size:
        return seq
    if position == 'middle':
        start = (len(seq) - chunk_size) // 2
    else:
        start = 0
    return seq[start:start + chunk_size]


def plot_heatmap(ax, punc_seq, title=None):
    numeric_seq = punctuation_to_numeric(punc_seq)

    width = 50
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
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')


def add_legend(fig):
    patches = []
    for mark in PUNCTUATION_VECTOR:
        patches.append(mpatches.Patch(
            color=PUNCTUATION_COLORS[mark],
            label=PUNC_NAMES[mark],
        ))
    fig.legend(
        handles=patches, loc='lower center', ncol=len(PUNCTUATION_VECTOR),
        fontsize=7, frameon=True, bbox_to_anchor=(0.5, 0.0),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load all data
    data = {}
    for author_key, author_name in AUTHORS.items():
        data[author_key] = {'name': author_name}

        # Real text (middle chunk)
        real_text = load_text(FULL_BOOKS_DIR / f"{author_key}_full.txt")
        real_seq = extract_punctuation(real_text)
        data[author_key]['real'] = get_chunk(real_seq, CHUNK_SIZE, 'middle')

        # Each version (first run, start chunk)
        for vkey, vinfo in VERSIONS.items():
            run_path = vinfo['dir'] / author_key / 'run_01.txt'
            if run_path.exists():
                gen_text = load_text(run_path)
                gen_seq = extract_punctuation(gen_text)
                if gen_seq and len(gen_seq) >= CHUNK_SIZE:
                    data[author_key][vkey] = get_chunk(gen_seq, CHUNK_SIZE, 'start')
                else:
                    data[author_key][vkey] = gen_seq if gen_seq else None
            else:
                data[author_key][vkey] = None

    # ---- Combined figure: all authors x all versions ----
    n_authors = len(AUTHORS)
    n_cols = 4  # Real, v1, v2, v3
    fig, axes = plt.subplots(
        n_authors, n_cols, figsize=(14, 3.2 * n_authors + 1.5),
    )

    col_labels = ['Real', VERSIONS['v1']['label'],
                  VERSIONS['v2']['label'], VERSIONS['v3']['label']]

    for row, (author_key, author_name) in enumerate(AUTHORS.items()):
        sources = ['real', 'v1', 'v2', 'v3']
        for col, (src, label) in enumerate(zip(sources, col_labels)):
            ax = axes[row, col]
            seq = data[author_key].get(src)
            if seq is not None:
                title = f"{author_name}\n({label})" if row == 0 else label
                if row == 0:
                    plot_heatmap(ax, seq, title=title)
                else:
                    plot_heatmap(ax, seq)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])

            # Author name on left edge
            if col == 0:
                ax.set_ylabel(author_name, fontsize=10, fontweight='bold',
                              rotation=90, labelpad=10)

    # Column headers
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=10, fontweight='bold')

    fig.suptitle(
        'Punctuation Heatmaps: Real vs Prompt Versions (2000 marks each)',
        fontsize=14, fontweight='bold', y=0.98,
    )
    fig.subplots_adjust(
        hspace=0.15, wspace=0.05, bottom=0.06, top=0.93, left=0.10, right=0.98,
    )
    add_legend(fig)

    out_path = OUTPUT_DIR / 'all_versions_heatmaps.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ---- Per-author figures ----
    for author_key, author_name in AUTHORS.items():
        fig, axes = plt.subplots(1, n_cols, figsize=(14, 4))

        sources = ['real', 'v1', 'v2', 'v3']
        for col, (src, label) in enumerate(zip(sources, col_labels)):
            ax = axes[col]
            seq = data[author_key].get(src)
            if seq is not None:
                plot_heatmap(ax, seq, title=label)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(label, fontsize=10, fontweight='bold')

        fig.suptitle(
            f'{author_name}: Punctuation Heatmaps by Prompt Version (2000 marks)',
            fontsize=13, fontweight='bold',
        )
        fig.subplots_adjust(
            wspace=0.05, bottom=0.10, top=0.85, left=0.03, right=0.97,
        )
        add_legend(fig)

        out_path = OUTPUT_DIR / f'{author_key}_versions_heatmap.png'
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
