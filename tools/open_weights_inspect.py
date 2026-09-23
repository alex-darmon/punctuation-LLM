#!/usr/bin/env python3
"""Inspect an open-weights run tree while it is still generating.

The smoke test reports once, at the end.  A 400-run sitting takes the better
part of a day, and the failure that matters most here is silent and late: the
model continues an English prompt in Chinese, several calls in, and from then on
accumulates almost no marks because the feature vector holds no full-width
punctuation.  The run does not error.  It just stops measuring anything and
runs to the round cap.

This reads the per-call records (``*.jsonl``, including the smoke test's
``*.live.jsonl`` checkpoints) and reports length, language, dashes and pace, so
a sitting can be stopped on evidence rather than on its wall clock.

    python tools/open_weights_inspect.py generated_texts_open_weights_v1
    python tools/open_weights_inspect.py <dir> --json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

# Kept in step with tools/open_weights_smoke.py.
CJK_CALL_THRESHOLD = 0.20
MIN_MEDIAN_CONTINUATION_WORDS = 1200
GEMINI_DASH_MEDIAN = 3.0
GEMINI_CONTINUATION_WORDS = 1773


def cjk_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(
        0x3000 <= ord(c) <= 0x9FFF or 0xFF00 <= ord(c) <= 0xFFEF for c in text
    ) / len(text)


def read_calls(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarise(path: Path) -> dict[str, Any] | None:
    calls = read_calls(path)
    if not calls:
        return None
    cont = [c for c in calls if c["call_index"] > 1]
    words = [c["response_words"] for c in cont]
    raw = "".join(c["raw_text"] for c in calls)
    cjk_calls = [c["call_index"] for c in calls if cjk_fraction(c["processed_text"]) > CJK_CALL_THRESHOLD]
    return {
        "run": f"{path.parents[2].name}/{path.parents[1].name}/{path.stem.replace('.live', '')}",
        "author": path.parents[1].name,
        "calls": len(calls),
        "marks": int(calls[-1]["marks_after"]),
        "median_continuation_words": statistics.median(words) if words else 0,
        "cjk_calls": cjk_calls,
        "first_cjk_call": cjk_calls[0] if cjk_calls else None,
        "cjk_fraction": round(cjk_fraction("".join(c["processed_text"] for c in calls)), 4),
        "dashes": raw.count("—") + raw.count("–"),
        "median_latency_s": round(statistics.median([c["latency_s"] for c in calls]), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="A generated_texts_* tree (or any parent of calls/*.jsonl).")
    parser.add_argument("--json", action="store_true", help="Emit the rows as JSON.")
    args = parser.parse_args()

    root = Path(args.run_dir)
    # A finished run has both run_NN.jsonl and the run_NN.live.jsonl checkpoint.
    # Counting both reports every completed run twice and skews every summary,
    # so the final file wins and the checkpoint is used only while it is the
    # only record of a run still in flight.
    by_run: dict[tuple[str, str], Path] = {}
    for path in sorted(p for p in root.rglob("*.jsonl") if p.parent.name == "calls"):
        stem = path.name.removesuffix(".jsonl").removesuffix(".live")
        key = (str(path.parent), stem)
        if key not in by_run or not path.name.endswith(".live.jsonl"):
            by_run[key] = path
    rows = [r for r in (summarise(p) for p in sorted(by_run.values())) if r]
    if not rows:
        raise SystemExit(f"no per-call records under {root}")

    # The determinism phase runs deliberately short jobs.  Pooling them with the
    # compliance runs would let a handful of 800-mark runs decide a check about
    # sustained length, so the headline is computed on compliance only.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["run"].split("/")[0], []).append(row)
    headline = groups.get("compliance", rows)

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print(f"{'run':44}{'calls':>6}{'marks':>7}{'medWords':>9}{'CJK':>5}{'1stCJK':>7}{'dash':>6}{'lat_s':>7}")
    for r in rows:
        print(f"{r['run'][:44]:44}{r['calls']:6d}{r['marks']:7d}"
              f"{r['median_continuation_words']:9.0f}{len(r['cjk_calls']):5d}"
              f"{(r['first_cjk_call'] or 0):7d}{r['dashes']:6d}{r['median_latency_s']:7.1f}")

    words = [r["median_continuation_words"] for r in headline if r["median_continuation_words"]]
    contaminated = [r for r in headline if r["cjk_calls"]]
    total_calls = sum(r["calls"] for r in headline)
    cjk_calls = sum(len(r["cjk_calls"]) for r in headline)
    scope = "compliance" if "compliance" in groups else "all"
    if len(groups) > 1:
        print("\ngroups                : " + ", ".join(
            f"{name} ({len(rs)})" for name, rs in sorted(groups.items())))
    print(f"summary below is over: {scope}")
    print(f"runs                  : {len(headline)}")
    if words:
        med = statistics.median(words)
        print(f"median continuation   : {med:.0f} words  "
              f"(threshold {MIN_MEDIAN_CONTINUATION_WORDS}, Gemini {GEMINI_CONTINUATION_WORDS}) "
              f"-> {'PASS' if med >= MIN_MEDIAN_CONTINUATION_WORDS else 'FAIL'}")
    print(f"runs with a CJK call  : {len(contaminated)}/{len(rows)}"
          f"   calls: {cjk_calls}/{total_calls}")
    if contaminated:
        firsts = [r["first_cjk_call"] for r in contaminated]
        print(f"  first CJK call        : median {statistics.median(firsts):.0f} "
              f"(range {min(firsts)}-{max(firsts)})")
        print("  a run that leaves English accrues almost no marks and will reach the")
        print("  round cap; its later windows are language collapse, not prompt drift.")
    print(f"median dashes per run : {statistics.median([r['dashes'] for r in headline]):.0f} "
          f"(Gemini median {GEMINI_DASH_MEDIAN:.0f} per complete run)")


if __name__ == "__main__":
    main()
