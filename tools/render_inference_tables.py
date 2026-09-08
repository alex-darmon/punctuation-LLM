#!/usr/bin/env python3
"""Render compact LaTeX tables from the pinned inference-v2 CSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "author_panel_20" / "inference_v2"
TABLES = ROOT / "paper" / "tables" / "inference_v2"


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(value: str) -> str:
    return f"{100 * float(value):.1f}"


def interval(low: str, high: str, *, percentage: bool = False) -> str:
    scale = 100 if percentage else 1
    return f"[{scale * float(low):.2f}, {scale * float(high):.2f}]"


def write(name: str, body: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    print(f"  [wrote] {path.relative_to(ROOT)}")


def attribution_table() -> None:
    selected = [
        row
        for row in rows("attribution_clustered.csv")
        if row["row_type"] == "summary"
        and row["author_set"] == "all_authors"
        and row["feature"] == "f3"
        and row["chunk_size"] == "2000"
    ]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Source & Authors & Units & Macro accuracy (\%) & 95\% author CI \\",
        r"\midrule",
    ]
    for row in selected:
        label = (row["condition"] or "Human").capitalize()
        lines.append(
            f"{label} & {row['n_authors']} & {row['n_observations']} & "
            f"{pct(row['macro_accuracy'])} & "
            f"{interval(row['macro_ci_low'], row['macro_ci_high'], percentage=True)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("primary_attribution.tex", "\n".join(lines))


def detection_table() -> None:
    selected = [
        row
        for row in rows("detection_crossfit.csv")
        if row["method"] == "out_of_author_crossfit"
        and row["feature"] == "f3"
        and row["chunk_size"] == "2000"
        and row["smoothing_eps"] == "0.5"
    ]
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & OOF AUC (95\% CI) & Held-out FPR (\%) & TPR (\%) \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['condition'].capitalize()} & "
            f"{float(row['auc']):.3f} {interval(row['auc_ci_low'], row['auc_ci_high'])} & "
            f"{pct(row['empirical_fpr'])} {interval(row['empirical_fpr_ci_low'], row['empirical_fpr_ci_high'], percentage=True)} & "
            f"{pct(row['tpr'])} {interval(row['tpr_ci_low'], row['tpr_ci_high'], percentage=True)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("detection.tex", "\n".join(lines))


def separability_table() -> None:
    selected = [
        row
        for row in rows("separability_crossfit_summary.csv")
        if row["row_type"] == "summary"
    ]
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Feature & Spearman $\rho$ (95\% CI) & Document accuracy (\%) & Chunk accuracy (\%) \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"${row['feature']}$ & {float(row['spearman_rho']):.2f} "
            f"{interval(row['spearman_ci_low'], row['spearman_ci_high'])} & "
            f"{pct(row['document_macro_accuracy'])} & "
            f"{pct(row['chunk_macro_accuracy'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("separability.tex", "\n".join(lines))


def drift_table() -> None:
    selected = [
        row
        for row in rows("drift_clustered.csv")
        if row["row_type"] == "delta"
        and row["feature"] == "f3"
        and row["metric"] in {"target_kl", "target_hit"}
    ]
    lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & First-to-fifth change & Author mean (95\% CI) & BH-adjusted $p$ \\",
        r"\midrule",
    ]
    for row in selected:
        metric = r"Target KL" if row["metric"] == "target_kl" else r"Target hit rate"
        percentage = row["metric"] == "target_hit"
        estimate = (
            f"{100 * float(row['estimate']):.1f} pp"
            if percentage
            else f"{float(row['estimate']):.3f}"
        )
        lines.append(
            f"{row['condition'].capitalize()} & {metric} & {estimate} "
            f"{interval(row['ci_low'], row['ci_high'], percentage=percentage)} & "
            f"{float(row['benjamini_hochberg_p']):.4g} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("drift.tex", "\n".join(lines))


def main() -> None:
    attribution_table()
    detection_table()
    separability_table()
    drift_table()


if __name__ == "__main__":
    main()
