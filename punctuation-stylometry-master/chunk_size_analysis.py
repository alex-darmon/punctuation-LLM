#!/usr/bin/env python3
"""
Separate plots for visual and statistical analysis - V2.

Plot 1: Heatmaps showing FIRST HALF vs SECOND HALF at each chunk size
Plot 2: KL mean ± std dev, one subplot per author

Uses the punctuation-stylometry codebase for analysis.
Run from this directory: python chunk_size_analysis.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Imports from punctuation-stylometry
from punctuation.config import options
from punctuation.parser.punctuation_parser import (
    get_textinfo,
    seq_pun_only,
    get_frequencies,
)
from punctuation.feature_operations.matrix_operations import (
    transition_mat,
    normalised_transition_mat,
)
from punctuation.feature_operations.distances import d_KL, d_KL_mat

# Punctuation vector from original codebase
PUNCTUATION_VECTOR = options.punctuation_vector

# Paths relative to project root (parent of punctuation-stylometry-master)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
FULL_BOOKS_DIR = _PROJECT_ROOT / 'full_books'
OUTPUT_DIR = _PROJECT_ROOT / 'separate_plots_comparison'


def load_text(filepath):
    """Load text from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def extract_punctuation(text):
    """Extract punctuation sequence from text. Handles ellipsis '...' -> '^'."""
    if text is None:
        return None
    text = text.replace('...', '^')  # match paper's ellipsis representation
    text_info = get_textinfo(text)
    return seq_pun_only(text_info)


def compute_all_features(punctuation_seq):
    """
    Compute all features (f1, f2, f3) from a punctuation sequence.
    Returns dict with 'f1' (frequency), 'f2' (conditional), 'f3' (joint).
    """
    f1 = get_frequencies(punctuation_seq, vector=PUNCTUATION_VECTOR)
    if f1 is None or sum(f1) == 0:
        return None
    f2 = transition_mat(punctuation_seq)
    if f2 is None:
        return None
    f3 = normalised_transition_mat(f2, f1)
    return {'f1': f1, 'f2': f2, 'f3': f3}


# =============================================================================
# CONFIGURATION
# =============================================================================

AUTHORS = {
    'jane_austen': 'Jane Austen',
    'william_shakespeare': 'William Shakespeare',
    'herbert_george_wells': 'H.G. Wells',
    'agnes_may_fleming': 'Agnes May Fleming'
}

# Chunk sizes
CHUNK_SIZES = [500, 1000, 1500, 2000, 3000]

# Number of samples per chunk size
N_SAMPLES = 30

# Color mapping
PUNCTUATION_COLORS = {
    '!': '#FF0000', '"': '#0000FF', '(': '#00FF00', ')': '#FF00FF',
    ',': '#1E90FF', '.': '#FFD700', ':': '#00CED1', ';': '#8B008B',
    '?': '#32CD32', '^': '#000000',
}

COLORS = [PUNCTUATION_COLORS[p] for p in PUNCTUATION_VECTOR]
CMAP = ListedColormap(COLORS)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def punctuation_to_numeric(punc_seq):
    return np.array([PUNCTUATION_VECTOR.index(p) for p in punc_seq])


def get_midpoint_chunks(full_seq, chunk_size):
    """Get chunks from midpoint of first half and midpoint of second half."""
    half = len(full_seq) // 2

    # Midpoint of first half
    first_mid = half // 2
    first_start = first_mid - chunk_size // 2
    chunk1 = full_seq[first_start:first_start + chunk_size]

    # Midpoint of second half
    second_mid = half + half // 2
    second_start = second_mid - chunk_size // 2
    chunk2 = full_seq[second_start:second_start + chunk_size]

    return chunk1, chunk2


def sample_chunk_pairs(full_seq, chunk_size, n_samples):
    pairs = []
    max_start = len(full_seq) - 2 * chunk_size

    if max_start < 0:
        return pairs

    for _ in range(n_samples):
        start1 = np.random.randint(0, max_start)
        chunk1 = full_seq[start1:start1 + chunk_size]

        valid_starts = list(range(0, start1 - chunk_size + 1)) + \
                       list(range(start1 + chunk_size, len(full_seq) - chunk_size + 1))

        if valid_starts:
            start2 = np.random.choice(valid_starts)
            chunk2 = full_seq[start2:start2 + chunk_size]
            pairs.append((chunk1, chunk2))

    return pairs


def compute_kl_for_pairs(pairs):
    kl_f1_values = []
    kl_f3_values = []

    for chunk1, chunk2 in pairs:
        f1 = compute_all_features(chunk1)
        f2 = compute_all_features(chunk2)

        if f1 is None or f2 is None:
            continue

        kl_f1 = d_KL(f1['f1'], f2['f1'])
        kl_f3 = d_KL_mat(f1['f3'], f2['f3'])

        kl_f1_values.append(kl_f1)
        kl_f3_values.append(kl_f3)

    return kl_f1_values, kl_f3_values


def plot_heatmap(ax, punc_seq):
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


def create_color_legend(output_path):
    fig, ax = plt.subplots(figsize=(12, 1.2))

    punc_names = {
        '!': '! (Excl.)', '"': '" (Quote)', '(': '( (L.Par)',
        ')': ') (R.Par)', ',': ', (Comma)', '.': '. (Period)',
        ':': ': (Colon)', ';': '; (Semi)', '?': '? (Quest)', '^': '... (Ellip)'
    }

    patches = [mpatches.Patch(color=PUNCTUATION_COLORS[p], label=punc_names[p])
               for p in PUNCTUATION_VECTOR]

    ax.legend(handles=patches, loc='center', ncol=10, fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Generating Separate Plots - V2 (with comparison)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)
    np.random.seed(42)

    # Load books
    print("\nLoading books...")
    all_sequences = {}
    for author_key, author_name in AUTHORS.items():
        filepath = FULL_BOOKS_DIR / f"{author_key}_full.txt"
        text = load_text(filepath)
        full_seq = extract_punctuation(text)
        all_sequences[author_name] = full_seq
        print(f"  {author_name}: {len(full_seq):,} marks")

    # Collect statistics and chunks
    print(f"\nSampling {N_SAMPLES} pairs per chunk size...")
    all_stats = {}

    for author_name, full_seq in all_sequences.items():
        print(f"\n  {author_name}:")
        all_stats[author_name] = {
            'sizes': [],
            'f1_mean': [], 'f1_std': [],
            'f3_mean': [], 'f3_std': [],
            'chunk_pairs': []  # (first_half, second_half) for each size
        }

        for chunk_size in CHUNK_SIZES:
            if chunk_size * 2 > len(full_seq):
                print(f"    {chunk_size}: skipped")
                continue

            # Get the comparison chunks (first half vs second half midpoints)
            chunk1, chunk2 = get_midpoint_chunks(full_seq, chunk_size)

            # Sample random pairs for variance estimation
            pairs = sample_chunk_pairs(full_seq, chunk_size, N_SAMPLES)
            if len(pairs) < 5:
                continue

            kl_f1, kl_f3 = compute_kl_for_pairs(pairs)

            f1_mean, f1_std = np.mean(kl_f1), np.std(kl_f1)
            f3_mean, f3_std = np.mean(kl_f3), np.std(kl_f3)

            all_stats[author_name]['sizes'].append(chunk_size)
            all_stats[author_name]['f1_mean'].append(f1_mean)
            all_stats[author_name]['f1_std'].append(f1_std)
            all_stats[author_name]['f3_mean'].append(f3_mean)
            all_stats[author_name]['f3_std'].append(f3_std)
            all_stats[author_name]['chunk_pairs'].append((chunk1, chunk2))

            print(f"    {chunk_size}: f1={f1_mean:.4f}±{f1_std:.4f}, f3={f3_mean:.4f}±{f3_std:.4f}")

    # =========================================================================
    # PLOT 1: Heatmaps with COMPARISON (First Half vs Second Half)
    # =========================================================================
    print("\nGenerating Plot 1: Heatmaps with comparison...")

    n_authors = len(all_stats)
    n_sizes = len(CHUNK_SIZES)

    # Each chunk size gets 2 columns (first half, second half)
    fig, axes = plt.subplots(n_authors, n_sizes * 2, figsize=(3 * n_sizes, 3 * n_authors))

    for i, (author_name, stats) in enumerate(all_stats.items()):
        for j, target_size in enumerate(CHUNK_SIZES):
            ax1 = axes[i, j * 2]      # First half
            ax2 = axes[i, j * 2 + 1]  # Second half

            if target_size in stats['sizes']:
                idx = stats['sizes'].index(target_size)
                chunk1, chunk2 = stats['chunk_pairs'][idx]

                plot_heatmap(ax1, chunk1)
                plot_heatmap(ax2, chunk2)

                # Add subtle border to show pairing
                for ax in [ax1, ax2]:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#666666')
                        spine.set_linewidth(1)
            else:
                for ax in [ax1, ax2]:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                           transform=ax.transAxes, fontsize=10)
                    ax.set_facecolor('#f0f0f0')

            # Column headers (only for first row)
            if i == 0:
                ax1.set_title(f'{target_size}\n1st half', fontsize=10, fontweight='bold')
                ax2.set_title(f'{target_size}\n2nd half', fontsize=10, fontweight='bold')

            # Row labels (only for first column)
            if j == 0:
                ax1.set_ylabel(author_name, fontsize=11, fontweight='bold')

    plt.suptitle('Within-Book Comparison: First Half Midpoint vs Second Half Midpoint\n'
                 '(Each pair shows chunks from same book at different positions)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot1_heatmaps_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # PLOT 2: KL Mean ± Std Dev, one subplot per author
    # =========================================================================
    print("Generating Plot 2: KL with variance...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    author_colors = {'f1': '#2E86AB', 'f3': '#E94F37'}

    for idx, (author_name, stats) in enumerate(all_stats.items()):
        ax = axes[idx]

        sizes = np.array(stats['sizes'])

        # f1 with error band
        f1_mean = np.array(stats['f1_mean'])
        f1_std = np.array(stats['f1_std'])
        ax.plot(sizes, f1_mean, 'o-', color=author_colors['f1'],
                linewidth=2.5, markersize=8, label='f1 (frequency)')
        ax.fill_between(sizes, f1_mean - f1_std, f1_mean + f1_std,
                        color=author_colors['f1'], alpha=0.2)

        # f3 with error band
        f3_mean = np.array(stats['f3_mean'])
        f3_std = np.array(stats['f3_std'])
        ax.plot(sizes, f3_mean, 's-', color=author_colors['f3'],
                linewidth=2.5, markersize=8, label='f3 (joint prob)')
        ax.fill_between(sizes, f3_mean - f3_std, f3_mean + f3_std,
                        color=author_colors['f3'], alpha=0.2)

        ax.set_xlabel('Chunk Size (punctuation marks)', fontsize=11)
        ax.set_ylabel('KL Divergence', fontsize=11)
        ax.set_title(author_name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)
        ax.set_xlim(400, 3100)

    plt.suptitle('KL Divergence (Mean ± Std Dev) by Chunk Size\n'
                 f'Within-book comparisons ({N_SAMPLES} random chunk pairs per size)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot2_kl_variance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Create legend
    create_color_legend(OUTPUT_DIR / 'color_legend.png')

    print(f"\n✓ Results saved to {OUTPUT_DIR}/")
    print("  - plot1_heatmaps_comparison.png (First Half vs Second Half)")
    print("  - plot2_kl_variance.png")
    print("  - color_legend.png")


if __name__ == "__main__":
    main()
