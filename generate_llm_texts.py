#!/usr/bin/env python3
"""
Pipeline to generate LLM-authored text in the style of classic authors,
then compare punctuation fingerprints using the punctuation-stylometry codebase.

Usage:
    export GEMINI_API_KEY="your-key-here"
    python generate_llm_texts.py                       # all authors, 10 runs each
    python generate_llm_texts.py --author jane_austen   # one author only
    python generate_llm_texts.py --runs 5               # custom number of runs
    python generate_llm_texts.py --skip-existing        # resume, reuse completed runs
"""

import sys
import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ---------------------------------------------------------------------------
# Extract our own CLI args BEFORE configargparse (from punctuation.config)
# sees sys.argv — it would reject unknown flags like --author.
# ---------------------------------------------------------------------------
_author_filter = None
_num_runs = 10
_parallel = 5  # concurrent runs
_skip_existing = False
_clean_argv = [sys.argv[0]]
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i] == '--author' and _i + 1 < len(sys.argv):
        _author_filter = sys.argv[_i + 1]
        _i += 2
    elif sys.argv[_i] == '--runs' and _i + 1 < len(sys.argv):
        _num_runs = int(sys.argv[_i + 1])
        _i += 2
    elif sys.argv[_i] == '--parallel' and _i + 1 < len(sys.argv):
        _parallel = int(sys.argv[_i + 1])
        _i += 2
    elif sys.argv[_i] == '--skip-existing':
        _skip_existing = True
        _i += 1
    else:
        _clean_argv.append(sys.argv[_i])
        _i += 1
sys.argv = _clean_argv

# Thread-safe print
_print_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)

# ---------------------------------------------------------------------------
# punctuation-stylometry-master imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PUNCT_STYLOMETRY_DIR = _SCRIPT_DIR / 'punctuation-stylometry-master'
sys.path.insert(0, str(_PUNCT_STYLOMETRY_DIR))

_config_path = str(_PUNCT_STYLOMETRY_DIR / 'conf' / 'punctuation.ini')
if '-c' not in sys.argv and '--config' not in sys.argv:
    sys.argv = [sys.argv[0], '-c', _config_path] + sys.argv[1:]

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
from punctuation.feature_operations.distances import d_KL

import numpy as np

# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
import google.generativeai as genai

PUNCTUATION_VECTOR = options.punctuation_vector

# =============================================================================
# CONFIGURATION
# =============================================================================

AUTHORS = {
    'jane_austen': {
        'name': 'Jane Austen',
        'form': 'prose',
    },
    'william_shakespeare': {
        'name': 'William Shakespeare',
        'form': 'play',
    },
    'herbert_george_wells': {
        'name': 'H.G. Wells',
        'form': 'prose',
    },
    'agnes_may_fleming': {
        'name': 'Agnes May Fleming',
        'form': 'prose',
    },
}

FULL_BOOKS_DIR = Path('full_books')
OUTPUT_DIR = Path('generated_texts')

MODEL = 'gemini-2.5-flash'

# Target punctuation marks for generated text
TARGET_MARKS = 2000

# How many words to include as a style excerpt in the prompt
EXCERPT_WORDS = 1500

# Max generation attempts (continuation rounds) per run
MAX_ROUNDS = 30

# Gemini config
TEMPERATURE = 1.0        # creative but not wild
MAX_OUTPUT_TOKENS = 8192  # per call


# =============================================================================
# PUNCTUATION HELPERS (thin wrappers around stylometry-master)
# =============================================================================

def extract_punctuation(text):
    """Extract punctuation sequence from text."""
    if text is None:
        return None
    text = text.replace('...', '^')
    text_info = get_textinfo(text)
    return seq_pun_only(text_info)


def count_marks(text):
    """Count punctuation marks in text."""
    seq = extract_punctuation(text)
    return len(seq) if seq else 0


def compute_all_features(punctuation_seq):
    """Compute f1, f2, f3 from a punctuation sequence."""
    f1 = get_frequencies(punctuation_seq, vector=PUNCTUATION_VECTOR)
    if f1 is None or sum(f1) == 0:
        return None
    f2 = transition_mat(punctuation_seq)
    if f2 is None:
        return None
    f3 = normalised_transition_mat(f2, f1)
    # Flatten f3 to a 100-element vector (joint probability distribution),
    # consistent with the paper & library: dKL is applied to flattened f3.
    f3_flat = f3.flatten().tolist()
    return {'f1': f1, 'f2': f2, 'f3': f3_flat}


# =============================================================================
# TEXT HELPERS
# =============================================================================

def load_text(filepath):
    """Load text from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def get_excerpt(text, n_words=EXCERPT_WORDS):
    """
    Extract a representative excerpt from the middle of the text.
    Taking from the middle avoids title pages, preambles, and
    end-matter / appendices.
    """
    words = text.split()
    mid = len(words) // 2
    half = n_words // 2
    start = max(0, mid - half)
    excerpt_words = words[start:start + n_words]
    return ' '.join(excerpt_words)


def clean_em_dashes(text):
    """Remove em-dashes from generated text (LLMs overuse them)."""
    return text.replace('—', ', ').replace('–', ', ').replace(' ,', ',')


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROSE_PROMPT = """Here is a passage by {author_name}:

---
{excerpt}
---

Write an original piece of fiction in the same writing style as the passage above. Create your own characters and story — do not retell or reference any of {author_name}'s existing works.

Important:
- Do NOT use em-dashes or en-dashes (— or –) anywhere in the text.
- Output plain prose text only, with no markdown formatting whatsoever (no #, **, *, etc.).
- The text should read exactly as it would appear in a printed book."""

PLAY_PROMPT = """Here is a passage from a play by {author_name}:

---
{excerpt}
---

Write an original scene of a play in the same writing style as the passage above. Create your own characters and story — do not retell or reference any of {author_name}'s existing works.

Important:
- Do NOT use em-dashes or en-dashes (— or –) anywhere in the text.
- Format the play exactly as in the passage: character name in CAPITALS on its own line, then dialogue on the next line. No colons after character names.
- Output plain text only, with no markdown formatting whatsoever (no #, **, *, etc.).
- The text should read exactly as it would appear in a printed play."""

CONTINUATION_PROMPT = """Continue the story below from where it left off. Maintain the same writing style consistently. Write at least 2000 words.

Important:
- Do NOT use em-dashes or en-dashes (— or –) anywhere in the text.
- Output plain prose text only, with no markdown formatting whatsoever.

Story so far (last section):
---
{last_section}
---

Continue:"""

PLAY_CONTINUATION_PROMPT = """Continue the play below from where it left off. Maintain the same writing style and play format consistently. Write at least 2000 words.

Important:
- Do NOT use em-dashes or en-dashes (— or –) anywhere in the text.
- Format: character name in CAPITALS on its own line, then dialogue on the next line. No colons after character names.
- Output plain text only, with no markdown formatting.

Play so far (last section):
---
{last_section}
---

Continue:"""


# =============================================================================
# GENERATION PIPELINE
# =============================================================================

def generate_text(model, author_key, author_info, run_id):
    """
    Generate text in the style of an author until we reach TARGET_MARKS
    punctuation marks. Each run is independent (fresh story).
    """
    author_name = author_info['name']
    form = author_info['form']
    tag = f"[{author_name[:12]:>12} run {run_id:02d}]"

    # Load the real book and extract an excerpt
    book_path = FULL_BOOKS_DIR / f"{author_key}_full.txt"
    book_text = load_text(book_path)
    excerpt = get_excerpt(book_text)

    # Pick the right prompt template
    if form == 'play':
        initial_prompt = PLAY_PROMPT.format(author_name=author_name, excerpt=excerpt)
        cont_prompt_template = PLAY_CONTINUATION_PROMPT
    else:
        initial_prompt = PROSE_PROMPT.format(author_name=author_name, excerpt=excerpt)
        cont_prompt_template = CONTINUATION_PROMPT

    accumulated_text = ""
    current_marks = 0

    for round_num in range(1, MAX_ROUNDS + 1):
        if round_num == 1:
            prompt = initial_prompt
        else:
            # Use last ~500 words as context for continuation
            last_words = accumulated_text.split()[-500:]
            last_section = ' '.join(last_words)
            prompt = cont_prompt_template.format(last_section=last_section)

        safe_print(f"  {tag} round {round_num}: {current_marks}/{TARGET_MARKS} marks...")

        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
            new_text = response.text
        except Exception as e:
            err_str = str(e)
            if 'per_day' in err_str or 'per_model_per_day' in err_str:
                safe_print(f"  {tag} DAILY QUOTA EXHAUSTED — stopping this run.")
                break
            safe_print(f"  {tag} ERROR: {e} — retrying in 30s...")
            time.sleep(30)
            continue

        # Clean em-dashes from output
        new_text = clean_em_dashes(new_text)

        # Accumulate
        if accumulated_text:
            accumulated_text += "\n\n" + new_text
        else:
            accumulated_text = new_text

        new_marks = count_marks(new_text)
        current_marks = count_marks(accumulated_text)

        if current_marks >= TARGET_MARKS:
            safe_print(f"  {tag} DONE — {current_marks} marks")
            break

        # Rate limiting
        time.sleep(2)

    return accumulated_text


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_comparison(author_key, real_text, generated_text):
    """
    Compare punctuation features between real and generated text.
    Uses a chunk of TARGET_MARKS from each.
    """
    real_seq = extract_punctuation(real_text)
    gen_seq = extract_punctuation(generated_text)

    if real_seq is None or gen_seq is None:
        return None

    # Truncate both to TARGET_MARKS for fair comparison
    chunk_size = min(TARGET_MARKS, len(real_seq), len(gen_seq))
    # Take from middle of real text
    real_start = (len(real_seq) - chunk_size) // 2
    real_chunk = real_seq[real_start:real_start + chunk_size]
    gen_chunk = gen_seq[:chunk_size]

    real_features = compute_all_features(real_chunk)
    gen_features = compute_all_features(gen_chunk)

    if real_features is None or gen_features is None:
        return None

    f1_kl = d_KL(real_features['f1'], gen_features['f1'])
    f3_kl = d_KL(real_features['f3'], gen_features['f3'])

    return {
        'author': author_key,
        'real_marks': len(real_seq),
        'gen_marks': len(gen_seq),
        'chunk_size': chunk_size,
        'f1_kl': float(f1_kl),
        'f3_kl': float(f3_kl),
        'real_f1': real_features['f1'],
        'gen_f1': gen_features['f1'],
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Check for API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable.")
        print("  export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Filter to single author if requested (parsed before configargparse ran)
    if _author_filter:
        if _author_filter not in AUTHORS:
            print(f"ERROR: Unknown author '{_author_filter}'")
            print(f"  Choose from: {', '.join(AUTHORS.keys())}")
            sys.exit(1)
        authors_to_run = {_author_filter: AUTHORS[_author_filter]}
    else:
        authors_to_run = AUTHORS

    num_runs = _num_runs
    max_workers = min(_parallel, num_runs)

    print("=" * 60)
    print("LLM Text Generation Pipeline")
    print(f"Model: {MODEL}")
    print(f"Target: {TARGET_MARKS} punctuation marks per run")
    print(f"Runs per author: {num_runs}")
    print(f"Parallel workers: {max_workers}")
    print(f"Skip existing: {_skip_existing}")
    print(f"Authors: {', '.join(a['name'] for a in authors_to_run.values())}")
    print("=" * 60)

    all_results = []

    for author_key, author_info in authors_to_run.items():
        author_name = author_info['name']
        real_text = load_text(FULL_BOOKS_DIR / f"{author_key}_full.txt")

        # Create author subdirectory
        author_dir = OUTPUT_DIR / author_key
        author_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {author_name} ({num_runs} runs, {max_workers} parallel)")
        print(f"{'='*60}")

        author_results = []

        # Determine which runs need generating vs which already exist
        runs_to_generate = []
        runs_to_reanalyze = []
        for rid in range(1, num_runs + 1):
            out_path = author_dir / f"run_{rid:02d}.txt"
            if _skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                runs_to_reanalyze.append(rid)
            else:
                runs_to_generate.append(rid)

        if runs_to_reanalyze:
            safe_print(f"  Reusing {len(runs_to_reanalyze)} existing runs: "
                        f"{runs_to_reanalyze}")

        # Re-analyze existing runs
        for rid in runs_to_reanalyze:
            out_path = author_dir / f"run_{rid:02d}.txt"
            generated = load_text(out_path)
            comparison = analyze_comparison(author_key, real_text, generated)
            if comparison:
                comparison['run_id'] = rid
                author_results.append(comparison)
                safe_print(f"  [cached]  run {rid:02d}: "
                           f"f1 KL={comparison['f1_kl']:.4f}  "
                           f"f3 KL={comparison['f3_kl']:.4f}  "
                           f"marks={comparison['gen_marks']}")

        # Generate remaining runs
        def run_one(run_id):
            """Generate + analyze a single run. Returns (run_id, comparison)."""
            generated = generate_text(model, author_key, author_info, run_id)

            # Save generated text
            out_path = author_dir / f"run_{run_id:02d}.txt"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(generated)

            comparison = analyze_comparison(author_key, real_text, generated)
            if comparison:
                comparison['run_id'] = run_id
            return run_id, comparison

        if runs_to_generate:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(run_one, rid): rid
                           for rid in runs_to_generate}

                for future in as_completed(futures):
                    run_id, comparison = future.result()
                    if comparison:
                        author_results.append(comparison)
                        safe_print(f"  [result] run {run_id:02d}: "
                                   f"f1 KL={comparison['f1_kl']:.4f}  "
                                   f"f3 KL={comparison['f3_kl']:.4f}  "
                                   f"marks={comparison['gen_marks']}")

        # Sort by run id for clean output
        author_results.sort(key=lambda r: r['run_id'])

        # Summary stats for this author
        if author_results:
            f1_vals = [r['f1_kl'] for r in author_results]
            f3_vals = [r['f3_kl'] for r in author_results]
            print(f"\n  {author_name} — {len(author_results)} runs:")
            print(f"    f1 KL: {np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}")
            print(f"    f3 KL: {np.mean(f3_vals):.4f} ± {np.std(f3_vals):.4f}")

        all_results.extend(author_results)

    # Save all results
    summary_path = OUTPUT_DIR / 'all_runs_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {summary_path}")

    # Final summary table
    if all_results:
        print("\n" + "=" * 70)
        print("SUMMARY: Real vs LLM-Generated Punctuation (mean ± std)")
        print("=" * 70)
        print(f"{'Author':<25} {'Runs':>5} {'f1 KL':>16} {'f3 KL':>16}")
        print("-" * 65)
        for author_key in authors_to_run:
            runs = [r for r in all_results if r['author'] == author_key]
            if runs:
                f1 = [r['f1_kl'] for r in runs]
                f3 = [r['f3_kl'] for r in runs]
                name = AUTHORS[author_key]['name']
                print(f"{name:<25} {len(runs):>5} "
                      f"{np.mean(f1):7.4f}±{np.std(f1):.4f} "
                      f"{np.mean(f3):7.4f}±{np.std(f3):.4f}")


if __name__ == "__main__":
    main()
