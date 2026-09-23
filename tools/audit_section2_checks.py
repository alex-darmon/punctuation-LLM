#!/usr/bin/env python3
"""One-off audit for Section 2 revisions: dash asymmetry, first-call bound,
preamble contamination. Prints numbers; writes nothing."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

cache = json.loads((ROOT / "cache/author_panel_20_with_runs.json").read_text())
sequences = cache["sequences"]

panel = json.loads((ROOT / "campaigns/author_panel_20.json").read_text())

# ---------------------------------------------------------------- step 5: dashes
print("=" * 72)
print("STEP 5: dash counts in the 60 human panel books")
print("=" * 72)

em, en, dd, total_marks, total_commas = 0, 0, 0, 0, 0
per_book = []
seen_paths = set()
for author in panel["authors"]:
    for book in author["panel_books"]:
        path = book["path"]
        if path in seen_paths:
            continue
        seen_paths.add(path)
        text = (ROOT / path).read_text(encoding="utf-8", errors="ignore")
        seq = sequences.get(path)
        if seq is None:
            print(f"  !! no cached sequence for {path}")
            continue
        n_em = text.count("\u2014")
        n_en = text.count("\u2013")
        n_dd = len(re.findall(r"--+", text))
        marks = len(seq)
        commas = seq.count(",")
        em += n_em; en += n_en; dd += n_dd
        total_marks += marks; total_commas += commas
        per_book.append((path, author["key"], n_em, n_en, n_dd, marks, commas))

n_books = len(per_book)
print(f"books: {n_books}")
print(f"total em dashes: {em}, en dashes: {en}, double-hyphen runs: {dd}")
print(f"total analysed marks: {total_marks}, commas: {total_commas}")
dash_all = em + en + dd
comma_share_now = total_commas / total_marks
comma_share_if = (total_commas + dash_all) / (total_marks + dash_all)
print(f"pooled comma share as parsed: {comma_share_now:.4f}")
print(f"pooled comma share if all dashes were commas: {comma_share_if:.4f}")
print(f"difference: {(comma_share_if - comma_share_now)*100:.3f} percentage points")
print(f"dashes per 1000 analysed marks (pooled): {1000*dash_all/total_marks:.2f}")
per_book.sort(key=lambda r: -(r[2] + r[3] + r[4]) / r[5])
print("top 8 books by dash rate (em, en, --, marks):")
for path, a, n_em, n_en, n_dd, marks, commas in per_book[:8]:
    rate = 1000 * (n_em + n_en + n_dd) / marks
    print(f"  {a:28s} {path:28s} em={n_em:6d} en={n_en:5d} --={n_dd:6d} "
          f"marks={marks:6d} rate={rate:6.1f}/1000")

# ------------------------------------------------------- step 6: first-call bound
print()
print("=" * 72)
print("STEP 6: mark density and first-call bound")
print("=" * 72)

conditions = {
    "flash": "generated_texts_campaign_author20_flash",
    "pro": "generated_texts_campaign_author20_pro",
}
for cond, d in conditions.items():
    densities = []
    marks_at = {}
    for key, seq in sequences.items():
        if not key.startswith(d + "/") or "/raw/" in key:
            continue
        path = ROOT / key
        words = len(path.read_text(encoding="utf-8", errors="ignore").split())
        if words:
            densities.append(len(seq) / words)
    densities.sort()
    n = len(densities)
    med = densities[n // 2]
    lo = densities[int(0.05 * n)]
    hi = densities[int(0.95 * n)]
    print(f"{cond}: n={n} runs, marks/word median={med:.3f} "
          f"(5th pct {lo:.3f}, 95th pct {hi:.3f})")
    for words_cap in (4500, 6000, 6800):
        print(f"   first call at {words_cap} words -> "
              f"~{med*words_cap:.0f} marks (5th pct {lo*words_cap:.0f})")

# ---------------------------------------------------- step 7: preamble contamination
print()
print("=" * 72)
print("STEP 7: preamble / meta-text contamination scan")
print("=" * 72)

OPENERS = re.compile(
    r"^\s*("
    r"here('s| is| are)\b|of course\b|certainly\b|okay\b|sure\b|"
    r"i (cannot|can't|will|would|hope)\b|as an ai\b|"
    r"below is\b|this (story|piece|tale)\b|"
    r"\*\*|##+|#\s|\*{3}|---|___|"
    r"(chapter|part|section)\s+[ivxlc0-9]+\b|"
    r"title\s*:|note\s*:|word count"
    r")",
    re.IGNORECASE,
)
MARKS = set('!"(),.:;?')

def line_marks(line: str) -> int:
    return sum(line.count(c) for c in MARKS) + line.count("...")

affected_runs = 0
total_runs = 0
matched_lines_total = 0
contaminated_marks = 0
total_marks_gen = 0
examples = []
for cond, d in conditions.items():
    for key, seq in sequences.items():
        if not key.startswith(d + "/") or "/raw/" in key:
            continue
        total_runs += 1
        total_marks_gen += len(seq)
        text = (ROOT / key).read_text(encoding="utf-8", errors="ignore")
        hits = [ln for ln in text.splitlines() if ln.strip() and OPENERS.match(ln)]
        if hits:
            affected_runs += 1
            matched_lines_total += len(hits)
            contaminated_marks += sum(line_marks(ln) for ln in hits)
            if len(examples) < 12:
                examples.append((key, hits[0][:90]))

print(f"runs scanned: {total_runs}")
print(f"runs with >=1 matched meta/preamble/heading line: {affected_runs} "
      f"({100*affected_runs/total_runs:.1f}%)")
print(f"matched lines total: {matched_lines_total}")
print(f"marks on matched lines: {contaminated_marks} "
      f"({100*contaminated_marks/total_marks_gen:.4f}% of all generated marks)")
print("examples:")
for key, ln in examples:
    print(f"  {key}: {ln!r}")

# Raw files: what did dash replacement hide? (extension batch only)
print()
print("also raw first-lines sample (extension batch):")
shown = 0
for d in conditions.values():
    for raw in sorted((ROOT / d).glob("*/raw/run_*.txt"))[:3]:
        first = raw.read_text(encoding="utf-8", errors="ignore").lstrip().splitlines()[0][:90]
        print(f"  {raw.relative_to(ROOT)}: {first!r}")
        shown += 1
