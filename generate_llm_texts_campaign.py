#!/usr/bin/env python3
"""
Config-driven LLM text generation campaign for punctuation stylometry.

This script is intended for larger generation campaigns (e.g., 10 authors),
and avoids hard-coding authors/prompts in multiple files.

Example:
  export GEMINI_API_KEY="your-key"
  python generate_llm_texts_campaign.py --campaign-config campaigns/generation_campaign_phaseA.json --dry-run
  python generate_llm_texts_campaign.py --campaign-config campaigns/generation_campaign_phaseA.json --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run config-driven generation campaign for punctuation stylometry."
    )
    parser.add_argument(
        "--campaign-config",
        required=True,
        help="Path to campaign JSON config.",
    )
    parser.add_argument(
        "--author",
        action="append",
        default=[],
        help="Author key to run (repeatable). If omitted, run all authors in config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from campaign config.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override runs per author from campaign config.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Parallel workers for run generation.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model from campaign config.",
    )
    parser.add_argument(
        "--target-marks",
        type=int,
        default=None,
        help="Override target punctuation marks from campaign config.",
    )
    parser.add_argument(
        "--min-book-samples-per-author",
        type=int,
        default=None,
        help=(
            "Require at least this many distinct source books per author. "
            "Overrides campaign config."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing non-empty run files and only re-analyze.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print execution plan without generation.",
    )
    parser.add_argument(
        "--auth-mode",
        choices=("auto", "api_key", "adc"),
        default="auto",
        help=(
            "Authentication mode for Gemini calls. "
            "'auto' prefers API key and falls back to ADC."
        ),
    )
    return parser.parse_args()


ARGS = parse_args()

PRINT_LOCK = threading.Lock()


def safe_print(*args, **kwargs):
    with PRINT_LOCK:
        print(*args, **kwargs)


ROOT = Path(__file__).resolve().parent
CAMPAIGN_CONFIG_PATH = Path(ARGS.campaign_config)
if not CAMPAIGN_CONFIG_PATH.is_absolute():
    CAMPAIGN_CONFIG_PATH = ROOT / CAMPAIGN_CONFIG_PATH


def load_campaign_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "authors" not in cfg or not isinstance(cfg["authors"], list) or not cfg["authors"]:
        raise ValueError("Campaign config must contain a non-empty 'authors' list.")

    required_author_fields = {"key", "name", "form"}
    for author in cfg["authors"]:
        missing = required_author_fields - set(author.keys())
        if missing:
            raise ValueError(f"Author entry missing fields {sorted(missing)}: {author}")
        if author["form"] not in {"prose", "play"}:
            raise ValueError(
                f"Unsupported form '{author['form']}' for author '{author['key']}'."
            )
        has_single = "book_path" in author
        has_multi = "book_paths" in author and isinstance(author["book_paths"], list)
        if not has_single and not has_multi:
            raise ValueError(
                f"Author '{author['key']}' must define 'book_path' or 'book_paths'."
            )
        if has_multi and len(author["book_paths"]) == 0:
            raise ValueError(
                f"Author '{author['key']}' has empty 'book_paths'."
            )

    return cfg


CAMPAIGN = load_campaign_config(CAMPAIGN_CONFIG_PATH)


def resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p


DEFAULT_MODEL = CAMPAIGN.get("default_model", "gemini-2.5-flash")
DEFAULT_RUNS = int(CAMPAIGN.get("default_runs", 10))
DEFAULT_TARGET_MARKS = int(CAMPAIGN.get("default_target_marks", 2000))
EXCERPT_WORDS = int(CAMPAIGN.get("excerpt_words", 1500))
MAX_ROUNDS = int(CAMPAIGN.get("max_rounds", 30))
TEMPERATURE = float(CAMPAIGN.get("temperature", 1.0))
MAX_OUTPUT_TOKENS = int(CAMPAIGN.get("max_output_tokens", 8192))
MIN_BOOK_SAMPLES = (
    ARGS.min_book_samples_per_author
    if ARGS.min_book_samples_per_author is not None
    else int(CAMPAIGN.get("min_source_books_per_author", 1))
)

MODEL = ARGS.model or DEFAULT_MODEL
NUM_RUNS = ARGS.runs if ARGS.runs is not None else DEFAULT_RUNS
TARGET_MARKS = ARGS.target_marks if ARGS.target_marks is not None else DEFAULT_TARGET_MARKS
OUTPUT_DIR = resolve_path(ARGS.output_dir or CAMPAIGN.get("output_dir", "generated_texts_campaign"))

if NUM_RUNS <= 0:
    raise ValueError(f"--runs must be positive, got {NUM_RUNS}")
if TARGET_MARKS <= 0:
    raise ValueError(f"--target-marks must be positive, got {TARGET_MARKS}")
if MIN_BOOK_SAMPLES <= 0:
    raise ValueError(
        f"--min-book-samples-per-author must be positive, got {MIN_BOOK_SAMPLES}"
    )

# ---------------------------------------------------------------------------
# punctuation-stylometry imports
# ---------------------------------------------------------------------------
PUNCT_STYLOMETRY_DIR = ROOT / "punctuation-stylometry-master"
sys.path.insert(0, str(PUNCT_STYLOMETRY_DIR))
CONFIG_PATH = str(PUNCT_STYLOMETRY_DIR / "conf" / "punctuation.ini")
sys.argv = [sys.argv[0], "-c", CONFIG_PATH]

from punctuation.config import options  # noqa: E402
from punctuation.feature_operations.distances import d_KL  # noqa: E402
from punctuation.feature_operations.matrix_operations import (  # noqa: E402
    normalised_transition_mat,
    transition_mat,
)
from punctuation.parser.punctuation_parser import (  # noqa: E402
    get_frequencies,
    get_textinfo,
    seq_pun_only,
)


PUNCTUATION_VECTOR = options.punctuation_vector
GENERATION_CONFIG_FACTORY = None

PROSE_PROMPT = """Here is a passage by {author_name}:

---
{excerpt}
---

Write an original piece of fiction in the same writing style as the passage above.
Create your own characters and story. Do not retell or reference existing works.

Pay close attention to punctuation style: relative use of commas, semicolons, colons,
question marks, exclamation marks, and quotation marks.

Important:
- Do not use em-dashes or en-dashes.
- Output plain prose text only (no markdown).
"""

PLAY_PROMPT = """Here is a passage from a play by {author_name}:

---
{excerpt}
---

Write an original play scene in the same writing style as the passage above.
Create your own characters and story. Do not retell or reference existing works.

Pay close attention to punctuation style and line rhythm.

Important:
- Do not use em-dashes or en-dashes.
- Character name should be uppercase on its own line, then dialogue.
- Output plain text only (no markdown).
"""

PROSE_CONTINUATION_PROMPT = """Continue the story below from where it left off.
Maintain the same writing style and punctuation habits. Write at least 2000 words.

Important:
- Do not use em-dashes or en-dashes.
- Output plain prose text only.

Story so far (last section):
---
{last_section}
---

Continue:
"""

PLAY_CONTINUATION_PROMPT = """Continue the play below from where it left off.
Maintain the same writing style and punctuation habits. Write at least 2000 words.

Important:
- Do not use em-dashes or en-dashes.
- Keep play format: character name on its own line.
- Output plain text only.

Play so far (last section):
---
{last_section}
---

Continue:
"""


def extract_punctuation(text: str):
    if text is None:
        return None
    text = text.replace("...", "^")
    return seq_pun_only(get_textinfo(text))


def count_marks(text: str) -> int:
    seq = extract_punctuation(text)
    return len(seq) if seq else 0


def compute_all_features(punctuation_seq):
    f1 = get_frequencies(punctuation_seq, vector=PUNCTUATION_VECTOR)
    if f1 is None or sum(f1) == 0:
        return None
    f2 = transition_mat(punctuation_seq)
    if f2 is None:
        return None
    f3 = normalised_transition_mat(f2, f1)
    f3_flat = f3.flatten().tolist()
    return {"f1": f1, "f2": f2, "f3": f3_flat}


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def get_excerpt(text: str, n_words: int = EXCERPT_WORDS) -> str:
    words = text.split()
    mid = len(words) // 2
    half = n_words // 2
    start = max(0, mid - half)
    return " ".join(words[start : start + n_words])


def clean_em_dashes(text: str) -> str:
    return text.replace("—", ", ").replace("–", ", ").replace(" ,", ",")


def select_authors():
    author_entries = CAMPAIGN["authors"]
    if not ARGS.author:
        return author_entries

    requested = set(ARGS.author)
    selected = [a for a in author_entries if a["key"] in requested]
    missing = sorted(requested - {a["key"] for a in selected})
    if missing:
        raise ValueError(f"Unknown --author keys: {missing}")
    return selected


def author_book_paths(author: dict) -> list[Path]:
    if "book_paths" in author and isinstance(author["book_paths"], list):
        raw_paths = author["book_paths"]
    else:
        raw_paths = [author["book_path"]]
    return [resolve_path(p) for p in raw_paths]


def validate_author_books(author_entries):
    missing = []
    under_min = []
    for author in author_entries:
        paths = author_book_paths(author)
        for book_path in paths:
            if not book_path.exists():
                missing.append((author["key"], str(book_path)))
        distinct_paths = len({str(p.resolve()) for p in paths})
        if distinct_paths < MIN_BOOK_SAMPLES:
            under_min.append((author["key"], distinct_paths))
    if missing:
        lines = "\n".join(f"  - {k}: {p}" for k, p in missing)
        raise FileNotFoundError(f"Missing author source books:\n{lines}")
    if under_min:
        lines = "\n".join(
            f"  - {k}: {n} source books (min required: {MIN_BOOK_SAMPLES})"
            for k, n in under_min
        )
        raise ValueError(f"Authors below minimum source-book count:\n{lines}")


def analyze_comparison(real_text: str, generated_text: str):
    real_seq = extract_punctuation(real_text)
    gen_seq = extract_punctuation(generated_text)
    if real_seq is None or gen_seq is None:
        return None

    chunk_size = min(TARGET_MARKS, len(real_seq), len(gen_seq))
    if chunk_size <= 0:
        return None

    real_start = (len(real_seq) - chunk_size) // 2
    real_chunk = real_seq[real_start : real_start + chunk_size]
    gen_chunk = gen_seq[:chunk_size]

    real_features = compute_all_features(real_chunk)
    gen_features = compute_all_features(gen_chunk)
    if real_features is None or gen_features is None:
        return None

    f1_kl = d_KL(real_features["f1"], gen_features["f1"])
    f3_kl = d_KL(real_features["f3"], gen_features["f3"])
    return {
        "real_marks": len(real_seq),
        "gen_marks": len(gen_seq),
        "chunk_size": chunk_size,
        "f1_kl": float(f1_kl),
        "f3_kl": float(f3_kl),
    }


def _extract_text_from_vertex_response(response) -> str | None:
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    chunks = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)
    if chunks:
        return "\n".join(chunks)
    return None


class _VertexGenerativeModelAdapter:
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def generate_content(self, prompt: str, generation_config):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=generation_config,
        )
        text = _extract_text_from_vertex_response(response)
        if not text:
            raise RuntimeError("Vertex model returned an empty response.")
        return SimpleNamespace(text=text)


def initialize_generation_backend(model_name: str):
    """
    Initialize model + config factory with either API-key auth or ADC.
    Returns: (model_adapter, generation_config_factory, auth_label)
    """
    auth_mode = ARGS.auth_mode
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if auth_mode in {"auto", "api_key"} and api_key:
        import google.generativeai as genai_module  # noqa: E402

        genai_module.configure(api_key=api_key)
        model = genai_module.GenerativeModel(model_name)

        def config_factory():
            return genai_module.types.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )

        return model, config_factory, "api_key"

    if auth_mode == "api_key":
        raise RuntimeError(
            "Auth mode 'api_key' requested, but GEMINI_API_KEY/GOOGLE_API_KEY is not set."
        )

    if auth_mode in {"auto", "adc"}:
        try:
            import google.auth  # noqa: E402
            from google import genai as genai_sdk  # noqa: E402
            from google.genai import types as genai_types  # noqa: E402
        except ImportError as e:
            if auth_mode == "adc":
                raise RuntimeError(
                    "ADC mode requested, but required packages are missing. "
                    "Install/upgrade google-genai and google-auth."
                ) from e
            raise RuntimeError(
                "No API key found and ADC packages are unavailable. "
                "Either set GEMINI_API_KEY or install google-genai/google-auth."
            ) from e

        _, detected_project = google.auth.default()
        project = (
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
            or detected_project
        )
        location = os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
        if not project:
            raise RuntimeError(
                "ADC authentication selected but project is unknown. "
                "Set GOOGLE_CLOUD_PROJECT (and optionally GOOGLE_CLOUD_LOCATION)."
            )

        client = genai_sdk.Client(
            vertexai=True,
            project=project,
            location=location,
        )
        model = _VertexGenerativeModelAdapter(client=client, model_name=model_name)

        def config_factory():
            return genai_types.GenerateContentConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )

        return model, config_factory, f"adc(project={project}, location={location})"

    raise RuntimeError(
        f"Unsupported auth mode: {auth_mode}. Choose one of: auto, api_key, adc."
    )


def source_book_for_run(run_id: int, book_paths: list[Path], book_texts: list[str]):
    idx = (run_id - 1) % len(book_paths)
    return idx, book_paths[idx], book_texts[idx]


def generate_text(model, author_name: str, form: str, book_text: str, run_id: int) -> str:
    if GENERATION_CONFIG_FACTORY is None:
        raise RuntimeError("Generation backend is not initialized.")

    tag = f"[{author_name[:12]:>12} run {run_id:02d}]"
    excerpt = get_excerpt(book_text)

    if form == "play":
        initial_prompt = PLAY_PROMPT.format(author_name=author_name, excerpt=excerpt)
        continuation_template = PLAY_CONTINUATION_PROMPT
    else:
        initial_prompt = PROSE_PROMPT.format(author_name=author_name, excerpt=excerpt)
        continuation_template = PROSE_CONTINUATION_PROMPT

    accumulated_text = ""
    current_marks = 0

    for round_num in range(1, MAX_ROUNDS + 1):
        if round_num == 1:
            prompt = initial_prompt
        else:
            last_section = " ".join(accumulated_text.split()[-500:])
            prompt = continuation_template.format(last_section=last_section)

        safe_print(f"  {tag} round {round_num}: {current_marks}/{TARGET_MARKS} marks...")

        try:
            response = model.generate_content(
                prompt,
                generation_config=GENERATION_CONFIG_FACTORY(),
            )
            new_text = response.text
        except Exception as e:
            err_str = str(e)
            if "per_day" in err_str or "per_model_per_day" in err_str:
                safe_print(f"  {tag} DAILY QUOTA EXHAUSTED - stopping this run.")
                break
            safe_print(f"  {tag} ERROR: {e} - retrying in 30s...")
            time.sleep(30)
            continue

        new_text = clean_em_dashes(new_text)
        if accumulated_text:
            accumulated_text += "\n\n" + new_text
        else:
            accumulated_text = new_text

        current_marks = count_marks(accumulated_text)
        if current_marks >= TARGET_MARKS:
            safe_print(f"  {tag} DONE - {current_marks} marks")
            break
        time.sleep(2)

    return accumulated_text


def main() -> None:
    global GENERATION_CONFIG_FACTORY
    authors = select_authors()
    validate_author_books(authors)

    safe_print("=" * 70)
    safe_print("Config-driven generation campaign")
    safe_print(f"Campaign: {CAMPAIGN.get('campaign_name', CAMPAIGN_CONFIG_PATH.stem)}")
    safe_print(f"Model: {MODEL}")
    safe_print(f"Target marks: {TARGET_MARKS}")
    safe_print(f"Runs per author: {NUM_RUNS}")
    safe_print(f"Min source books/author: {MIN_BOOK_SAMPLES}")
    safe_print(f"Parallel workers: {max(1, min(ARGS.parallel, NUM_RUNS))}")
    safe_print(f"Skip existing: {ARGS.skip_existing}")
    safe_print(f"Auth mode: {ARGS.auth_mode}")
    safe_print(f"Output dir: {OUTPUT_DIR}")
    safe_print(f"Authors selected ({len(authors)}): {[a['key'] for a in authors]}")
    safe_print("=" * 70)

    if ARGS.dry_run:
        total_runs = len(authors) * NUM_RUNS
        safe_print(f"\n[dry-run] planned runs: {total_runs}")
        for author in authors:
            book_paths = author_book_paths(author)
            safe_print(
                f"  - {author['key']}: {author['name']} | {author['form']} | "
                f"{len(book_paths)} source books"
            )
            for p in book_paths:
                safe_print(f"      * {p}")
        return

    model, GENERATION_CONFIG_FACTORY, auth_label = initialize_generation_backend(MODEL)
    safe_print(f"Authenticated via: {auth_label}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write run metadata for reproducibility.
    metadata_out = {
        "campaign_name": CAMPAIGN.get("campaign_name", CAMPAIGN_CONFIG_PATH.stem),
        "campaign_config_path": str(CAMPAIGN_CONFIG_PATH),
        "model": MODEL,
        "auth_mode_requested": ARGS.auth_mode,
        "auth_mode_used": auth_label,
        "target_marks": TARGET_MARKS,
        "runs_per_author": NUM_RUNS,
        "authors": authors,
    }
    with open(OUTPUT_DIR / "campaign_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, indent=2)

    all_results = []
    max_workers = max(1, min(ARGS.parallel, NUM_RUNS))

    for author in authors:
        author_key = author["key"]
        author_name = author["name"]
        form = author["form"]
        book_paths = author_book_paths(author)
        book_texts = [load_text(path) for path in book_paths]

        author_dir = OUTPUT_DIR / author_key
        author_dir.mkdir(parents=True, exist_ok=True)
        safe_print(
            f"\n{'='*60}\n  {author_name} ({author_key}) | source books: {len(book_paths)}\n{'='*60}"
        )

        author_results = []
        runs_to_generate = []
        runs_to_reanalyze = []

        for run_id in range(1, NUM_RUNS + 1):
            out_path = author_dir / f"run_{run_id:02d}.txt"
            if ARGS.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                runs_to_reanalyze.append(run_id)
            else:
                runs_to_generate.append(run_id)

        if runs_to_reanalyze:
            safe_print(
                f"  Reusing {len(runs_to_reanalyze)} existing runs: {runs_to_reanalyze}"
            )

        for run_id in runs_to_reanalyze:
            out_path = author_dir / f"run_{run_id:02d}.txt"
            generated = load_text(out_path)
            source_idx, source_path, source_text = source_book_for_run(
                run_id=run_id,
                book_paths=book_paths,
                book_texts=book_texts,
            )
            comparison = analyze_comparison(source_text, generated)
            if comparison is None:
                continue
            comparison.update(
                {
                    "author_key": author_key,
                    "author_name": author_name,
                    "run_id": run_id,
                    "source_book_index": source_idx,
                    "source_book_path": str(source_path),
                }
            )
            author_results.append(comparison)
            safe_print(
                f"  [cached] run {run_id:02d}: "
                f"f1 KL={comparison['f1_kl']:.4f} "
                f"f3 KL={comparison['f3_kl']:.4f} "
                f"marks={comparison['gen_marks']}"
            )

        def run_one(run_id: int):
            source_idx, source_path, source_text = source_book_for_run(
                run_id=run_id,
                book_paths=book_paths,
                book_texts=book_texts,
            )
            generated = generate_text(
                model=model,
                author_name=author_name,
                form=form,
                book_text=source_text,
                run_id=run_id,
            )
            out_path = author_dir / f"run_{run_id:02d}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(generated)

            comparison = analyze_comparison(source_text, generated)
            if comparison is not None:
                comparison.update(
                    {
                        "author_key": author_key,
                        "author_name": author_name,
                        "run_id": run_id,
                        "source_book_index": source_idx,
                        "source_book_path": str(source_path),
                    }
                )
            return run_id, comparison

        if runs_to_generate:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(run_one, run_id) for run_id in runs_to_generate]
                for future in as_completed(futures):
                    run_id, comparison = future.result()
                    if comparison is None:
                        continue
                    author_results.append(comparison)
                    safe_print(
                        f"  [result] run {run_id:02d}: "
                        f"f1 KL={comparison['f1_kl']:.4f} "
                        f"f3 KL={comparison['f3_kl']:.4f} "
                        f"marks={comparison['gen_marks']}"
                    )

        author_results.sort(key=lambda r: r["run_id"])
        if author_results:
            f1_vals = [r["f1_kl"] for r in author_results]
            f3_vals = [r["f3_kl"] for r in author_results]
            safe_print(
                f"  Summary ({len(author_results)} runs): "
                f"f1={np.mean(f1_vals):.4f}+-{np.std(f1_vals):.4f}, "
                f"f3={np.mean(f3_vals):.4f}+-{np.std(f3_vals):.4f}"
            )
        all_results.extend(author_results)

    summary_path = OUTPUT_DIR / "all_runs_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    safe_print(f"\nAll results saved to {summary_path}")


if __name__ == "__main__":
    main()
