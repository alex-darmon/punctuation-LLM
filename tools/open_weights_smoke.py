#!/usr/bin/env python3
"""The compliance smoke test open_weights_protocol_v1 requires before freezing.

The campaign file says, of the model it is about to pin: "must pass before the
campaign is frozen ... Do not proceed with a model that fails these checks."
This runs those checks, plus a determinism measurement that the declaration
assumes rather than establishes.

Phase 1 - compliance.  Five condition-c2 runs, one per author, balanced over
cohort and form, at the authors' declared target marks.  Checks:

  1  every run reaches its declared target (and 5,002 marks) within 40 rounds
  2  median response length at least 1,200 words per continuation call
  3  raw em- and en-dash count per run below the Gemini median
  4  no verbatim repetition of a 200-character span within a run
  5  fifth-window text for two runs, extracted for manual reading

Phase 2 - determinism.  The declaration states that "vLLM output depends on
batch composition as well as seed" and that "exact reproduction requires the
same worker count".  Both halves are measured rather than assumed, on short
runs, by generating one job three times:

  D1  alone            D2  alone            D3  inside a pool of five

sha256(D1) == sha256(D2) says a fixed seed reproduces exactly when nothing else
is in flight.  sha256(D1) == sha256(D3) says it survives realistic batching.
The honest claim in the paper is whichever of these holds.

Outputs land in <output_dir>_smoke and are never evidence for any prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_positional_mechanism as gpm  # noqa: E402

N_COMPLIANCE_RUNS = 5
MIN_MEDIAN_CONTINUATION_WORDS = 1200
ABSOLUTE_MARK_FLOOR = 5002
REPEAT_SPAN_CHARS = 200
DETERMINISM_TARGET_MARKS = 800
# A call counts as having left English if this share of its characters is CJK.
# Well above a stray quoted foreign word, well below a call written in Chinese.
CJK_CALL_THRESHOLD = 0.20

# Median raw em+en dash count per run over the 600 Gemini positional-mechanism
# runs.  Recomputed by --dash-baseline-from rather than trusted as a constant.
DEFAULT_GEMINI_DASH_MEDIAN = 3.0


def gemini_dash_median(run_root: Path) -> float | None:
    counts: list[int] = []
    for path in run_root.glob("*/*/summaries/*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        counts.append(int(row.get("raw_em_dash_count", 0)) + int(row.get("raw_en_dash_count", 0)))
    return statistics.median(counts) if counts else None


def gemini_repeat_base_rate(run_root: Path, span: int, sample: int = 60) -> dict[str, Any] | None:
    """How often do the Gemini runs contain a verbatim `span`-character repeat?

    The campaign declares "no verbatim repetition of a 200-character span" as an
    absolute bar, but the 600 Gemini runs it compares against were never held to
    it.  Measuring their base rate says whether a failure here is a fact about
    the candidate model or about the protocol: twenty-odd independent
    continuation calls, each seeing only a 500-word tail, with no repetition
    penalty.  Reported alongside the check rather than folded into its verdict,
    because changing what the check means is a pre-registration decision.
    """
    import random

    files = sorted(run_root.glob("*/*/run_*.txt"))
    if not files:
        return None
    random.Random(0).shuffle(files)
    picked = files[:sample]
    hits = sum(1 for f in picked if longest_repeated_span(f.read_text(encoding="utf-8"), span))
    return {
        "source": str(run_root.name),
        "runs_sampled": len(picked),
        "runs_with_repeat": hits,
        "rate": round(hits / len(picked), 3),
        "sample_seed": 0,
    }


def select_authors(authors: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """A fixed, balanced pick: alternate cohorts, preferring an unused form."""
    by_cohort: dict[str, list[dict[str, Any]]] = {}
    for author in sorted(authors, key=lambda a: a["key"]):
        by_cohort.setdefault(author["cohort"], []).append(author)
    picked: list[dict[str, Any]] = []
    forms_used: set[str] = set()
    cohorts = sorted(by_cohort)
    while len(picked) < n:
        progressed = False
        for cohort in cohorts:
            pool = [a for a in by_cohort[cohort] if a not in picked]
            if not pool or len(picked) >= n:
                continue
            fresh = [a for a in pool if a["form"] not in forms_used]
            choice = (fresh or pool)[0]
            picked.append(choice)
            forms_used.add(choice["form"])
            progressed = True
        if not progressed:
            break
    return picked[:n]


def longest_repeated_span(text: str, span: int) -> str | None:
    """Return a verbatim span of `span` characters that occurs twice, if any."""
    squeezed = re.sub(r"\s+", " ", text)
    if len(squeezed) <= span:
        return None
    seen: dict[str, int] = {}
    for index in range(len(squeezed) - span + 1):
        window = squeezed[index : index + span]
        digest = hashlib.blake2b(window.encode("utf-8"), digest_size=16).digest()
        previous = seen.get(digest)
        if previous is not None and squeezed[previous : previous + span] == window:
            return window
        seen.setdefault(digest, index)
    return None


def cjk_fraction(text: str) -> float:
    """Share of characters in CJK or full-width ranges.

    Qwen2.5 is a Chinese-origin checkpoint and will continue an English prompt
    in Chinese.  This matters more than it looks: the feature vector is
    ['!', '"', '(', ')', ',', '.', ':', ';', '?', '^'] and Chinese uses
    full-width marks, so a run that switches language stops accumulating marks
    almost entirely.  It neither terminates nor measures the thing the campaign
    is about, and none of the declared checks would catch it: the run merely
    looks slow.
    """
    if not text:
        return 0.0
    return sum(
        0x3000 <= ord(c) <= 0x9FFF or 0xFF00 <= ord(c) <= 0xFFEF for c in text
    ) / len(text)


def fifth_window(text: str) -> str:
    words = text.split()
    start = (len(words) * 4) // 5
    return " ".join(words[start : start + 350])


def continuation_word_counts(calls: list[dict[str, Any]]) -> list[int]:
    return [int(c["response_words"]) for c in calls if c["call_index"] > 1]


class Runner:
    """Generates one job through the shared protocol; no protocol code here."""

    def __init__(self, config: dict[str, Any], backend: gpm.Backend, out_dir: Path) -> None:
        self.gen = config["generation"]
        self.backend = backend
        self.out_dir = out_dir
        self.seed_root = int(self.gen["sampling"]["seed"])
        self._books: dict[str, str] = {}
        self._lock = threading.Lock()

    def excerpt_for(self, job: gpm.Job) -> str:
        with self._lock:
            text = self._books.get(job.source_book_path)
            if text is None:
                text = gpm.load_text(ROOT / job.source_book_path)
                self._books[job.source_book_path] = text
        return gpm.get_excerpt(text, int(self.gen["excerpt_words"]))

    def log(self, message: str) -> None:
        with self._lock:
            print(message, flush=True)

    def run(self, job: gpm.Job, tag: str) -> dict[str, Any]:
        def transport(model: str, prompt: str) -> gpm.CallResult:
            return self.backend.generate(
                model, prompt, seed=gpm.derive_seed(self.seed_root, job, gpm.sha256_text(prompt))
            )

        started = datetime.now(timezone.utc)
        base = self.out_dir / tag / job.author_key
        (base / "calls").mkdir(parents=True, exist_ok=True)
        # Checkpoint every call as it happens.  A run that dies on its 30th
        # call is the most informative run in the batch, and without this its
        # text is lost and the cause is unreproducible - which is exactly what
        # happened to the run that overflowed the context window in job 2210361.
        live = (base / "calls" / f"run_{job.run_id:02d}.live.jsonl").open("w", encoding="utf-8")

        def checkpoint(record: dict) -> None:
            with self._lock:
                live.write(json.dumps(record, ensure_ascii=False) + "\n")
                live.flush()

        try:
            processed, raw, calls = gpm.generate_run(
                job,
                gen=self.gen,
                excerpt=self.excerpt_for(job),
                generate=transport,
                log=self.log,
                sleep=lambda _s: None,  # no hosted-API delay against a local server
                checkpoint=checkpoint,
            )
        finally:
            live.close()
        finished = datetime.now(timezone.utc)
        (base / f"run_{job.run_id:02d}.txt").write_text(processed, encoding="utf-8")
        (base / "calls" / f"run_{job.run_id:02d}.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in calls) + "\n", encoding="utf-8"
        )
        (base / f"run_{job.run_id:02d}.raw.txt").write_text(raw, encoding="utf-8")
        marks = int(calls[-1]["marks_after"]) if calls else 0
        return {
            "tag": tag,
            "author_key": job.author_key,
            "cohort": job.cohort,
            "form": job.form,
            "run_id": job.run_id,
            "target_marks": job.target_marks,
            "marks": marks,
            "reached_target": marks >= job.target_marks,
            "reached_absolute_floor": marks >= ABSOLUTE_MARK_FLOOR,
            "n_calls": len(calls),
            "within_max_rounds": len(calls) <= int(self.gen["max_rounds"]),
            "continuation_words": continuation_word_counts(calls),
            "median_continuation_words": (
                statistics.median(continuation_word_counts(calls))
                if len(calls) > 1 else 0
            ),
            "cjk_fraction": cjk_fraction(processed),
            "cjk_calls": [
                c["call_index"] for c in calls
                if cjk_fraction(c["processed_text"]) > CJK_CALL_THRESHOLD
            ],
            "raw_em_dash_count": raw.count("—"),
            "raw_en_dash_count": raw.count("–"),
            "raw_sha256": gpm.sha256_text(raw),
            "processed_sha256": gpm.sha256_text(processed),
            "raw_chars": len(raw),
            "wall_seconds": (finished - started).total_seconds(),
            "repeated_span": longest_repeated_span(processed, REPEAT_SPAN_CHARS),
            "fifth_window": fifth_window(processed),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="campaigns/open_weights_protocol_v1.json")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument(
        "--dash-baseline-from",
        default="generated_texts_positional_mechanism_v1",
        help="Gemini run tree whose median raw dash count is the threshold.",
    )
    parser.add_argument(
        "--repeat-baseline-from",
        default="generated_texts_positional_mechanism_v1",
        help="Gemini run tree whose verbatim-repeat base rate is reported for comparison.",
    )
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = gpm.load_config(config_path)
    gen = config["generation"]
    out_dir = ROOT / (gen["output_dir"] + "_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = gpm.make_backend(gen, args)
    print(f"[backend] {backend.label}")
    runner = Runner(config, backend, out_dir)

    plan = gpm.build_plan(config)
    c2 = {(job.author_key, job.run_id): job for job in plan if job.condition == "c2"}
    chosen = select_authors(gen["authors"], N_COMPLIANCE_RUNS)
    print("[smoke] authors: " + ", ".join(f"{a['key']}({a['cohort']}/{a['form']})" for a in chosen))

    # ---- Phase 1: compliance -------------------------------------------------
    jobs = [c2[(a["key"], 1)] for a in chosen]

    def guarded(job: gpm.Job) -> dict[str, Any] | None:
        try:
            return runner.run(job, "compliance")
        except Exception as error:  # noqa: BLE001
            print(f"[smoke] {job.author_key} FAILED: {type(error).__name__}: {error}", flush=True)
            return {"author_key": job.author_key, "failed": f"{type(error).__name__}: {error}"}

    with ThreadPoolExecutor(max_workers=min(args.parallel, len(jobs))) as pool:
        compliance = list(pool.map(guarded, jobs))
    failed = [r for r in compliance if r.get("failed")]
    compliance = [r for r in compliance if not r.get("failed")]
    if failed:
        print(f"[smoke] {len(failed)} of {len(jobs)} runs failed outright")
    if not compliance:
        raise SystemExit("every smoke run failed; see the log and the .live.jsonl checkpoints")

    dash_median = gemini_dash_median(ROOT / args.dash_baseline_from)
    if dash_median is None:
        dash_median = DEFAULT_GEMINI_DASH_MEDIAN
        dash_source = "default constant (no Gemini summaries found)"
    else:
        dash_source = f"{args.dash_baseline_from} summaries"

    medians = [r["median_continuation_words"] for r in compliance]
    # A run that died never reached its target.  Excluding it from the
    # completion checks would let the two worst runs in the batch turn those
    # checks green, which is the opposite of what a compliance test is for.
    checks = {
        "reaches_target_within_max_rounds": {
            "pass": (
                not failed
                and all(r["reached_target"] and r["within_max_rounds"] for r in compliance)
            ),
            "detail": [
                {"author": r["author_key"], "marks": r["marks"], "target": r["target_marks"],
                 "calls": r["n_calls"]}
                for r in compliance
            ] + [{"author": f["author_key"], "failed": f["failed"]} for f in failed],
        },
        "reaches_absolute_mark_floor": {
            "pass": not failed and all(r["reached_absolute_floor"] for r in compliance),
            "threshold": ABSOLUTE_MARK_FLOOR,
            "runs_that_never_finished": [f["author_key"] for f in failed],
        },
        "median_continuation_words": {
            "pass": all(m >= MIN_MEDIAN_CONTINUATION_WORDS for m in medians),
            "threshold": MIN_MEDIAN_CONTINUATION_WORDS,
            "per_run": {r["author_key"]: r["median_continuation_words"] for r in compliance},
        },
        "dashes_below_gemini_median": {
            "pass": all(
                r["raw_em_dash_count"] + r["raw_en_dash_count"] < dash_median for r in compliance
            ),
            "threshold": dash_median,
            "threshold_source": dash_source,
            "per_run": {
                r["author_key"]: r["raw_em_dash_count"] + r["raw_en_dash_count"]
                for r in compliance
            },
        },
        "stays_in_english": {
            "pass": all(not r["cjk_calls"] for r in compliance),
            "threshold": CJK_CALL_THRESHOLD,
            "note": (
                "not one of the campaign's declared checks; added after a smoke run "
                "continued an English prompt in Chinese from its third call onward. "
                "Chinese uses full-width punctuation, which the feature vector does "
                "not contain, so such a run measures nothing and never terminates."
            ),
            "per_run": {
                r["author_key"]: {
                    "cjk_calls": r["cjk_calls"],
                    "cjk_fraction": round(r["cjk_fraction"], 4),
                }
                for r in compliance
            },
        },
        "no_verbatim_200_char_repeat": {
            "pass": all(r["repeated_span"] is None for r in compliance),
            "gemini_base_rate": gemini_repeat_base_rate(
                ROOT / args.repeat_baseline_from, REPEAT_SPAN_CHARS
            ),
            "base_rate_note": (
                "the Gemini arms were never held to this check. If their base rate is high, "
                "a failure here is a property of the protocol - independent continuation calls "
                "with a 500-word tail and no repetition penalty - rather than of the candidate "
                "model, and the verdict should be read accordingly."
            ),
            "per_run": {
                r["author_key"]: (r["repeated_span"][:120] + "..." if r["repeated_span"] else None)
                for r in compliance
            },
        },
        "fifth_window_coherent": {
            "pass": None,
            "verdict": "manual: read the two excerpts in fifth_window_samples and record a verdict",
        },
    }

    # ---- Phase 2: determinism -----------------------------------------------
    determinism: dict[str, Any] = {"ran": not args.skip_determinism}
    if not args.skip_determinism:
        subject = chosen[0]
        short = gpm.Job(**{**asdict(c2[(subject["key"], 2)]), "target_marks": DETERMINISM_TARGET_MARKS})
        print(f"[smoke] determinism on {subject['key']} at {DETERMINISM_TARGET_MARKS} marks")
        d1 = runner.run(short, "determinism_d1_alone")
        d2 = runner.run(short, "determinism_d2_alone")
        fillers = [
            gpm.Job(**{**asdict(c2[(a["key"], 2)]), "target_marks": DETERMINISM_TARGET_MARKS})
            for a in chosen[1:5]
        ]
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(runner.run, short, "determinism_d3_in_pool")]
            futures += [pool.submit(runner.run, f, "determinism_filler") for f in fillers]
            pooled = [f.result() for f in futures]
        d3 = pooled[0]
        determinism.update(
            {
                "author": subject["key"],
                "target_marks": DETERMINISM_TARGET_MARKS,
                "pool_size_for_d3": 5,
                "d1_raw_sha256": d1["raw_sha256"],
                "d2_raw_sha256": d2["raw_sha256"],
                "d3_raw_sha256": d3["raw_sha256"],
                "exact_when_isolated": d1["raw_sha256"] == d2["raw_sha256"],
                "exact_under_batching": d1["raw_sha256"] == d3["raw_sha256"],
                "reading": (
                    "exact_when_isolated true and exact_under_batching true: byte-exact "
                    "reproduction may be claimed given the worker count. isolated true and "
                    "batched false: claim reproducibility only at a stated worker count. both "
                    "false: claim pinned weights and a recorded seed, and nothing about "
                    "byte-exact reproduction."
                ),
            }
        )

    required = [v["pass"] for k, v in checks.items() if v["pass"] is not None]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_config_sha256": gpm.sha256_file(config_path),
        "generator_script_sha256": gpm.sha256_file(ROOT / "generate_positional_mechanism.py"),
        "prompt_protocol_sha256": gpm.protocol_sha256(),
        "backend_environment": backend.environment(),
        "client_environment": gpm.client_environment(),
        "workers": args.parallel,
        "authors": [a["key"] for a in chosen],
        "failed_runs": failed,
        "checks": checks,
        "determinism": determinism,
        "automated_verdict": "pass" if all(required) else "fail",
        "runs": [{k: v for k, v in r.items() if k != "fifth_window"} for r in compliance],
        "fifth_window_samples": {
            r["author_key"]: r["fifth_window"] for r in compliance[:2]
        },
        "note": "engineering evidence for the freeze decision only; never evidence for P1-P5",
    }
    path = out_dir / "smoke_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== smoke checks ===")
    for name, check in checks.items():
        state = {True: "PASS", False: "FAIL", None: "MANUAL"}[check["pass"]]
        print(f"  {state:6s} {name}")
    if determinism.get("ran"):
        print(f"  {'PASS' if determinism['exact_when_isolated'] else 'FAIL':6s} determinism: exact when isolated")
        print(f"  {'PASS' if determinism['exact_under_batching'] else 'FAIL':6s} determinism: exact under batching")
    print(f"\nautomated verdict: {report['automated_verdict']}")
    print(f"report: {path.relative_to(ROOT)}")
    raise SystemExit(0 if report["automated_verdict"] == "pass" else 1)


if __name__ == "__main__":
    main()
