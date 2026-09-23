#!/usr/bin/env python3
"""Render compact LaTeX tables from pinned inference CSV outputs.

The defaults regenerate the v2 tables unchanged.  The repeated-prompt (v3)
tables, which add chunk-size rows, the margin and rank drift rows and the
leakage grid, are rendered with

  python tools/render_inference_tables.py --layout v3 \
      --results results/author_panel_20/inference_v3 \
      --tables paper/tables/inference_v3 \
      --position-matched results/author_panel_20/position_matched_detection_v3 \
      --full-grid results/author_panel_20/full_grid_v3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "author_panel_20" / "inference_v2"
TABLES = ROOT / "paper" / "tables" / "inference_v2"
POSITION_MATCHED = ROOT / "results" / "author_panel_20" / "position_matched_detection"
FULL_GRID = ROOT / "results" / "author_panel_20" / "full_grid"
LAYOUT = "v2"


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


def tex_number(text: str) -> str:
    """Typeset a leading minus sign as mathematics."""
    return f"${text}$" if text.startswith("-") else text


def detection_by_length_table() -> None:
    """v3 layout: one row per model and chunk length."""
    selected = {
        (row["condition"], int(row["chunk_size"])): row
        for row in rows("detection_crossfit.csv")
        if row["method"] == "out_of_author_crossfit"
        and row["feature"] == "f3"
        and row["smoothing_eps"] == "0.5"
    }
    lines = [
        r"\begin{tabular}{lrccc}",
        r"\toprule",
        r"Model & Marks & OOF AUC (95\% CI) & Held-out FPR (\%) & TPR (\%) \\",
        r"\midrule",
    ]
    for index, condition in enumerate(("flash", "pro")):
        if index:
            lines.append(r"\addlinespace")
        for size in sorted(size for c, size in selected if c == condition):
            row = selected[(condition, size)]
            lines.append(
                f"{condition.capitalize()} & {size:,} & "
                f"{float(row['auc']):.3f} {interval(row['auc_ci_low'], row['auc_ci_high'])} & "
                f"{100 * float(row['empirical_fpr']):.2f} & "
                f"{pct(row['tpr'])} [{pct(row['tpr_ci_low'])}, {pct(row['tpr_ci_high'])}] \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("detection.tex", "\n".join(lines))


def leakage_table() -> None:
    """v3 layout: grid-path TPR with the scored book excluded or included."""
    with (FULL_GRID / "detection.csv").open(newline="", encoding="utf-8") as handle:
        grid = {
            (row["feature"], row["policy"], row["condition"], int(row["chunk_size"])): row
            for row in csv.DictReader(handle)
        }
    sizes = sorted({key[3] for key in grid})
    lines = [
        r"\begin{tabular}{l" + "r" * (2 * len(sizes)) + "}",
        r"\toprule",
        rf"& \multicolumn{{{len(sizes)}}}{{c}}{{Flash TPR (\%)}} &",
        rf"  \multicolumn{{{len(sizes)}}}{{c}}{{Pro TPR (\%)}} \\",
        "Feature & " + " & ".join([f"{size // 1000}k" for size in sizes] * 2) + r" \\",
        r"\midrule",
    ]
    names = {"f1": r"\fOne", "f3": r"\fThree"}
    policies = {"leave_one_book_out": "LOBO", "pooled_all_books": "pooled"}
    for feature in ("f1", "f3"):
        for policy, label in policies.items():
            cells = [
                pct(grid[(feature, policy, condition, size)]["tpr_at_5pct_fpr"])
                for condition in ("flash", "pro")
                for size in sizes
            ]
            lines.append(f"{names[feature]}, {label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("leakage_grid.tex", "\n".join(lines))


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
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Feature & Spearman $\rho$ (95\% CI) & Chunk accuracy (\%) \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"${row['feature']}$ & {float(row['spearman_rho']):.2f} "
            f"{interval(row['spearman_ci_low'], row['spearman_ci_high'])} & "
            f"{pct(row['chunk_macro_accuracy'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("separability.tex", "\n".join(lines))


def drift_all_metrics_table() -> None:
    """v3 layout: KL, margin, rank and hit-rate changes per model."""
    labels = {
        "target_kl": "Target KL",
        "target_margin": "Target margin",
        "target_rank": "Target rank",
        "target_hit": "Target hit rate",
    }
    decimals = {"target_kl": 3, "target_margin": 3, "target_rank": 2}
    selected = {
        (row["condition"], row["metric"]): row
        for row in rows("drift_clustered.csv")
        if row["row_type"] == "delta" and row["feature"] == "f3"
    }
    lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & First-to-fifth change & Author mean (95\% CI) & BH-adjusted $p$ \\",
        r"\midrule",
    ]
    for condition in ("flash", "pro"):
        for metric, label in labels.items():
            row = selected[(condition, metric)]
            if metric == "target_hit":
                estimate = tex_number(f"{100 * float(row['estimate']):.1f}") + " pp"
                low, high = (f"{100 * float(row[k]):.2f}" for k in ("ci_low", "ci_high"))
            else:
                places = decimals[metric]
                estimate = tex_number(f"{float(row['estimate']):.{places}f}")
                low, high = (f"{float(row[k]):.{places}f}" for k in ("ci_low", "ci_high"))
            lines.append(
                f"{condition.capitalize()} & {label} & {estimate} "
                f"[{tex_number(low)}, {tex_number(high)}] & "
                f"{float(row['benjamini_hochberg_p']):.4g} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("drift.tex", "\n".join(lines))


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


def position_matched_table() -> None:
    path = POSITION_MATCHED / "detection_position.csv"
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        detection = list(csv.DictReader(handle))
    selected = [
        row
        for row in detection
        if row["chunk_size"] == "2000"
        and row["window_name"] in {"prefix", "late"}
    ]
    order = [("flash", "prefix"), ("flash", "late"), ("pro", "prefix"), ("pro", "late")]
    by_key = {(row["condition"], row["window_name"]): row for row in selected}
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model & Window & OOF AUC (95\% CI) & Held-out FPR (\%) & TPR (\%) \\",
        r"\midrule",
    ]
    labels = {"prefix": "Marks 1--2,000", "late": "Marks 2,001--4,000"}
    for condition, window in order:
        row = by_key[(condition, window)]
        lines.append(
            f"{row['condition'].capitalize()} & {labels[window]} & "
            f"{float(row['auc']):.3f} {interval(row['auc_ci_low'], row['auc_ci_high'])} & "
            f"{pct(row['empirical_fpr'])} & "
            f"{pct(row['tpr'])} {interval(row['tpr_ci_low'], row['tpr_ci_high'], percentage=True)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("position_matched.tex", "\n".join(lines))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_table_manifest() -> None:
    """Tie the rendered tables to the results they were rendered from."""
    sources = {
        "inference_manifest": RESULTS / "inference_manifest.json",
        "position_matched_manifest": POSITION_MATCHED / "manifest.json",
        "full_grid_manifest": FULL_GRID / "manifest.json",
    }
    (TABLES / "table_manifest.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "layout": LAYOUT,
                "results": {
                    name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                    for name, path in sources.items()
                },
                "renderer_sha256": sha256(Path(__file__)),
                "tables_sha256": {
                    path.name: sha256(path) for path in sorted(TABLES.glob("*.tex"))
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(RESULTS.relative_to(ROOT)))
    parser.add_argument("--tables", default=str(TABLES.relative_to(ROOT)))
    parser.add_argument("--position-matched", default=str(POSITION_MATCHED.relative_to(ROOT)))
    parser.add_argument("--full-grid", default=str(FULL_GRID.relative_to(ROOT)))
    parser.add_argument("--layout", choices=("v2", "v3"), default="v2")
    return parser.parse_args()


def main() -> None:
    global RESULTS, TABLES, POSITION_MATCHED, FULL_GRID, LAYOUT
    args = parse_args()
    RESULTS = ROOT / args.results
    TABLES = ROOT / args.tables
    POSITION_MATCHED = ROOT / args.position_matched
    FULL_GRID = ROOT / args.full_grid
    LAYOUT = args.layout
    attribution_table()
    if LAYOUT == "v3":
        separability_table()
        detection_by_length_table()
        drift_all_metrics_table()
        leakage_table()
        position_matched_table()
        write_table_manifest()
        return
    detection_table()
    separability_table()
    drift_table()
    position_matched_table()


if __name__ == "__main__":
    main()
