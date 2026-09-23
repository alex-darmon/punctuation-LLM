#!/usr/bin/env python3
"""Render the reference-design test (one book of 2N versus two books of N)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STYLE = {
    "one_book": ("One book of 2N marks", "#111827", "o"),
    "two_books": ("Two books of N marks", "#2563eb", "s"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/author_panel_20/reference_design")
    parser.add_argument("--figures", default="paper/figures/reference_design")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"  [wrote] {path.relative_to(ROOT)}")
    plt.close(fig)


def primary_rows(rows: list[dict[str, str]], config: dict) -> list[dict[str, str]]:
    p = config["primary"]
    return [
        r
        for r in rows
        if r["rival_mode"] == p["rival_mode"]
        and r["window_policy"] == p["window_policy"]
        and r["feature"] == p["feature"]
        and int(r["test_chunk_size"]) == int(p["test_chunk_size"])
    ]


def design_figure(summary, contrasts, config, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    ax_acc, ax_kl, ax_gap = axes
    for design, (label, color, marker) in STYLE.items():
        rows = sorted(
            (r for r in summary if r["design"] == design),
            key=lambda r: int(r["budget_n"]),
        )
        n = [2 * int(r["budget_n"]) for r in rows]
        for ax, key in ((ax_acc, "macro_accuracy"), (ax_kl, "macro_kl_true")):
            y = [float(r[key]) for r in rows]
            lo = [y_i - float(r[f"{key}_ci_low"]) for y_i, r in zip(y, rows)]
            hi = [float(r[f"{key}_ci_high"]) - y_i for y_i, r in zip(y, rows)]
            ax.errorbar(
                n, y, yerr=[lo, hi], label=label, color=color, marker=marker,
                capsize=3, linewidth=1.4,
            )
    rows = sorted(contrasts, key=lambda r: int(r["budget_n"]))
    n = [2 * int(r["budget_n"]) for r in rows]
    y = [float(r["accuracy_difference"]) for r in rows]
    lo = [y_i - float(r["accuracy_ci_low"]) for y_i, r in zip(y, rows)]
    hi = [float(r["accuracy_ci_high"]) - y_i for y_i, r in zip(y, rows)]
    ax_gap.errorbar(n, y, yerr=[lo, hi], color="#dc2626", marker="D", capsize=3, linewidth=1.4)
    ax_gap.axhline(0.0, color="#94a3b8", linewidth=1, linestyle=":")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(n)
        ax.set_xticklabels([f"{v:,}" for v in n])
        ax.set_xlabel("Reference budget (marks, both designs)")
    ax_acc.set_ylabel("Author-equal attribution accuracy")
    ax_kl.set_ylabel("Mean KL to true-author reference")
    ax_gap.set_ylabel("Accuracy gap, two books minus one")
    ax_acc.legend(frameon=False, loc="lower right")
    p = config["primary"]
    fig.suptitle(
        f"Reference design test ({p['rival_mode']} rivals, {p['feature']}, "
        f"{p['test_chunk_size']:,}-mark held-out chunks; 95% author bootstrap)",
        fontsize=11,
    )
    save(fig, figures, "design_test")


def collapse_figure(summary, config, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ceiling = config["effective_marks_model"]["predicted_n_eff"]["single_passage_ceiling"]
    for design, (label, color, marker) in STYLE.items():
        rows = sorted(
            (r for r in summary if r["design"] == design),
            key=lambda r: float(r["predicted_n_eff"]),
        )
        x = [float(r["predicted_n_eff"]) for r in rows]
        y = [float(r["macro_accuracy"]) for r in rows]
        lo = [y_i - float(r["macro_accuracy_ci_low"]) for y_i, r in zip(y, rows)]
        hi = [float(r["macro_accuracy_ci_high"]) - y_i for y_i, r in zip(y, rows)]
        ax.errorbar(x, y, yerr=[lo, hi], label=label, color=color, marker=marker, capsize=3, linewidth=1.4)
        offset = (5, 8) if design == "two_books" else (5, -13)
        for r, x_i, y_i in zip(rows, x, y):
            ax.annotate(f"N={int(r['budget_n']):,}", (x_i, y_i), textcoords="offset points", xytext=offset, fontsize=8, color=color)
    ax.axvline(ceiling, color="#94a3b8", linewidth=1, linestyle=":")
    ax.text(ceiling, ax.get_ylim()[1], r" $1/\rho$", va="top", fontsize=9, color="#64748b")
    ax.set_xlabel(r"Predicted effective marks of the reference, $n/(1+\rho n)$ per book")
    ax.set_ylabel("Author-equal attribution accuracy")
    ax.legend(frameon=False, loc="lower right")
    save(fig, figures, "design_collapse")


def main() -> None:
    args = parse_args()
    results = ROOT / args.results
    figures = ROOT / args.figures
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["output_sha256"].items():
        if sha256(results / name) != digest:
            raise ValueError(f"manifest verification failed for {name}")
    config = json.loads((ROOT / manifest["analysis_config"]).read_text(encoding="utf-8"))
    summary = primary_rows(read_csv(results / "design_summary.csv"), config)
    contrasts = primary_rows(read_csv(results / "design_contrasts.csv"), config)
    design_figure(summary, contrasts, config, figures)
    collapse_figure(summary, config, figures)
    figure_files = sorted(p for p in figures.iterdir() if p.suffix in {".pdf", ".png"})
    (figures / "figure_manifest.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "results_manifest_sha256": sha256(results / "manifest.json"),
                "plotter_sha256": sha256(Path(__file__)),
                "matplotlib": importlib.metadata.version("matplotlib"),
                "figures_sha256": {p.name: sha256(p) for p in figure_files},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
