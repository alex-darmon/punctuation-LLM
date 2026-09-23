#!/usr/bin/env python3
"""Render the positional-mechanism campaign (C1 replicate vs C2 prompt repeated)."""

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
ARM_STYLE = {
    "old": ("Existing runs (C1)", "#6b7280", "o", "-"),
    "replicate": ("C1 replicate", "#111827", "s", "--"),
    "c2": ("C2 prompt repeated", "#2563eb", "D", "-"),
    "human_start": ("Human, book start", "#b45309", "^", ":"),
    "human_interior": ("Human, interior offset", "#d97706", "v", ":"),
}
MODEL_LABEL = {"flash": "Flash", "pro": "Pro"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/author_panel_20/positional_mechanism_v1")
    parser.add_argument("--figures", default="paper/figures/positional_mechanism_v1")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"  [wrote] {path.relative_to(ROOT)}")
    plt.close(fig)


def window_figure(summary: list[dict[str, str]], feature: str, models: list[str], figures: Path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.6), sharey=True)
    axes = list(axes) if len(models) > 1 else [axes]
    human_rows = [r for r in summary if r["row_type"] == "position" and r["model"] == "human" and r["feature"] == feature and r["metric"] == "target_kl"]
    for ax, model in zip(axes, models):
        for arm, (label, colour, marker, style) in ARM_STYLE.items():
            rows = human_rows if arm.startswith("human") else [
                r for r in summary
                if r["row_type"] == "position" and r["model"] == model and r["arm"] == arm and r["feature"] == feature and r["metric"] == "target_kl"
            ]
            rows = [r for r in rows if r["arm"] == arm]
            if not rows:
                continue
            rows.sort(key=lambda r: int(r["window_index"]))
            x = [int(r["window_index"]) for r in rows]
            y = [float(r["estimate"]) for r in rows]
            lo = [float(r["ci_low"]) for r in rows]
            hi = [float(r["ci_high"]) for r in rows]
            ax.plot(x, y, marker=marker, linestyle=style, color=colour, label=label)
            ax.fill_between(x, lo, hi, color=colour, alpha=0.12, linewidth=0)
        ax.set_title(MODEL_LABEL.get(model, model))
        ax.set_xlabel("1,000-mark window")
        ax.set_xticks(range(1, 6))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"Target KL ({feature}), author-equal mean")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, figures, "window_drift")


def contrast_figure(contrasts: list[dict[str, str]], feature: str, models: list[str], figures: Path) -> None:
    rows = [r for r in contrasts if r["feature"] == feature and r["quantity"] == "first_to_fifth_change"]
    if not rows:
        return
    metrics = ["target_kl", "target_margin", "target_rank", "target_hit"]
    names = ["replicate_minus_old", "c2_minus_c1_pooled", "c2_minus_replicate"]
    labels = {"replicate_minus_old": "replicate - old", "c2_minus_c1_pooled": "C2 - C1 pooled", "c2_minus_replicate": "C2 - C1 replicate"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.4))
    for ax, metric in zip(axes, metrics):
        y = 0
        ticks, tick_labels = [], []
        for model in models:
            for name in names:
                row = next((r for r in rows if r["model"] == model and r["contrast"] == name and r["metric"] == metric), None)
                if row is None:
                    continue
                colour = "#2563eb" if name.startswith("c2") else "#6b7280"
                ax.errorbar(
                    float(row["mean_difference"]), y,
                    xerr=[[float(row["mean_difference"]) - float(row["ci_low"])], [float(row["ci_high"]) - float(row["mean_difference"])]],
                    fmt="o", color=colour, capsize=3,
                )
                ticks.append(y)
                tick_labels.append(f"{MODEL_LABEL.get(model, model)}: {labels[name]}")
                y += 1
        ax.axvline(0, color="#111827", linewidth=0.8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_title(f"Δ {metric.replace('target_', '')} (first→fifth)", fontsize=9)
        ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    save(fig, figures, "drift_contrasts")


def detection_figure(detection: list[dict[str, str]], models: list[str], figures: Path) -> None:
    if not detection:
        return
    arms = ["old", "replicate", "c2"]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), sharey=True)
    axes = list(axes) if len(models) > 1 else [axes]
    width = 0.36
    for ax, model in zip(axes, models):
        for i, arm in enumerate(arms):
            for j, window in enumerate(("prefix", "late")):
                row = next((r for r in detection if r["model"] == model and r["arm"] == arm and r["window_name"] == window), None)
                if row is None:
                    continue
                colour = ARM_STYLE[arm][1]
                ax.bar(
                    i + (j - 0.5) * width, 100 * float(row["tpr"]), width,
                    color=colour, alpha=0.55 if window == "prefix" else 1.0,
                    yerr=[[100 * (float(row["tpr"]) - float(row["tpr_ci_low"]))], [100 * (float(row["tpr_ci_high"]) - float(row["tpr"]))]],
                    capsize=3, label=f"{ARM_STYLE[arm][0]}, {window}" if True else None,
                )
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ARM_STYLE[a][0] for a in arms], fontsize=8)
        ax.set_title(MODEL_LABEL.get(model, model))
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("Detection rate at 2,000 marks (%)")
    axes[-1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save(fig, figures, "position_matched_detection")


def share_figure(shares: list[dict[str, str]], models: list[str], figures: Path) -> None:
    rows = [r for r in shares if r["window_index"] == "5" and r["quantity"].startswith("share_")]
    if not rows:
        return
    marks = sorted({r["quantity"] for r in rows})
    arms = ["old", "replicate", "c2"]
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 3.4), sharey=True)
    axes = list(axes) if len(models) > 1 else [axes]
    width = 0.26
    for ax, model in zip(axes, models):
        for i, arm in enumerate(arms):
            xs, ys, err = [], [], [[], []]
            for j, mark in enumerate(marks):
                row = next((r for r in rows if r["model"] == model and r["arm"] == arm and r["quantity"] == mark), None)
                if row is None:
                    continue
                xs.append(j + (i - 1) * width)
                ys.append(100 * float(row["estimate"]))
                err[0].append(100 * (float(row["estimate"]) - float(row["ci_low"])))
                err[1].append(100 * (float(row["ci_high"]) - float(row["estimate"])))
            if xs:
                ax.bar(xs, ys, width, yerr=err, capsize=2, color=ARM_STYLE[arm][1], label=ARM_STYLE[arm][0])
        ax.set_xticks(range(len(marks)))
        ax.set_xticklabels([m.replace("share_", "") for m in marks])
        ax.set_title(f"{MODEL_LABEL.get(model, model)}: window-5 mark shares")
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("Share of marks (%)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, figures, "window5_mark_shares")


def _ci(row: dict[str, str], low: str = "ci_low", high: str = "ci_high", scale: float = 1.0, digits: int = 3) -> str:
    return f"[{scale * float(row[low]):.{digits}f}, {scale * float(row[high]):.{digits}f}]"


def render_tables(results: Path, config: dict, tables: Path) -> list[Path]:
    """Compact LaTeX tables for the manuscript: arms by row, and the C2 - C1 contrasts."""
    feature = config["analysis"]["primary_feature"]
    attractor = config["analysis"]["mark_shares"]["attractor_author"]
    summary = read_csv(results / "window_summary.csv")
    shares = read_csv(results / "mark_shares.csv")
    contrasts = read_csv(results / "contrasts.csv")
    tables.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def delta(model: str, arm: str, metric: str) -> dict[str, str] | None:
        return next((r for r in summary if r["row_type"] == "delta" and r["model"] == model and r["arm"] == arm and r["feature"] == feature and r["metric"] == metric), None)

    def share(model: str, arm: str, quantity: str) -> dict[str, str] | None:
        return next((r for r in shares if r["model"] == model and r["arm"] == arm and r["window_index"] == "5" and r["quantity"] == quantity), None)

    lines = [
        r"\begin{tabular}{llrcccc}",
        r"\toprule",
        r"Model & Arm & Runs & $\Delta$ target KL & $\Delta$ margin & Wells share, w5 & Hit rate, w5 \\",
        r"\midrule",
    ]
    arm_label = {"old": "C1 (existing)", "c2": "C2 (prompt repeated)", "human_start": "book start", "human_interior": "interior offset"}
    for model, arms in (("flash", ("old", "c2")), ("pro", ("old", "c2")), ("human", ("human_start", "human_interior"))):
        for arm in arms:
            kl = delta(model, arm, "target_kl")
            margin = delta(model, arm, "target_margin")
            if kl is None or margin is None:
                continue
            wells = share(model, arm, f"nearest_{attractor}")
            hit = share(model, arm, "target_hit")
            wells_text = f"{100 * float(wells['estimate']):.1f}\\%" if wells else "--"
            hit_text = f"{100 * float(hit['estimate']):.1f}\\%" if hit else "--"
            lines.append(
                f"{MODEL_LABEL.get(model, model.capitalize())} & {arm_label[arm]} & {kl['n_runs']} & "
                f"{float(kl['estimate']):+.3f} {_ci(kl)} & {float(margin['estimate']):+.3f} {_ci(margin)} & {wells_text} & {hit_text} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = tables / "drift_by_arm.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(path)

    lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & First-to-fifth change & C2 $-$ C1, author mean (95\% CI) & BH-adjusted $p$ \\",
        r"\midrule",
    ]
    metric_label = {"target_kl": "Target KL", "target_margin": "Margin", "target_rank": "Rank", "target_hit": "Hit rate"}
    for model in config["generation"]["models"]:
        for metric in ("target_kl", "target_margin", "target_rank", "target_hit"):
            row = next((r for r in contrasts if r["contrast"] == "c2_minus_c1_pooled" and r["model"] == model and r["feature"] == feature and r["metric"] == metric), None)
            if row is None:
                continue
            if metric == "target_hit":
                estimate = f"{100 * float(row['mean_difference']):+.1f} pp {_ci(row, scale=100, digits=1)}"
            elif metric == "target_rank":
                estimate = f"{float(row['mean_difference']):+.2f} {_ci(row, digits=2)}"
            else:
                estimate = f"{float(row['mean_difference']):+.3f} {_ci(row)}"
            lines.append(f"{MODEL_LABEL.get(model, model)} & {metric_label[metric]} & {estimate} & {float(row['benjamini_hochberg_p']):.4g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = tables / "contrasts.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(path)
    for path in written:
        print(f"  [wrote] {path.relative_to(ROOT)}")
    return written


def main() -> None:
    args = parse_args()
    results = ROOT / args.results
    figures = ROOT / args.figures
    tables = ROOT / "paper" / "tables" / "positional_mechanism_v1"
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["output_sha256"].items():
        if sha256(results / name) != digest:
            raise ValueError(f"manifest verification failed for {name}")
    config = json.loads((ROOT / manifest["analysis_config"]).read_text(encoding="utf-8"))
    feature = config["analysis"]["primary_feature"]
    models = list(config["generation"]["models"])
    summary = read_csv(results / "window_summary.csv")
    window_figure(summary, feature, models, figures)
    contrast_figure(read_csv(results / "contrasts.csv"), feature, models, figures)
    detection_figure(read_csv(results / "detection_position.csv"), models, figures)
    share_figure(read_csv(results / "mark_shares.csv"), models, figures)
    table_files = render_tables(results, config, tables) if (results / "contrasts.csv").is_file() else []
    figure_files = sorted(p for p in figures.iterdir() if p.suffix in {".pdf", ".png"})
    (figures / "figure_manifest.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "stage": manifest.get("stage"),
                "results_manifest_sha256": sha256(results / "manifest.json"),
                "plotter_sha256": sha256(Path(__file__)),
                "matplotlib": importlib.metadata.version("matplotlib"),
                "figures_sha256": {p.name: sha256(p) for p in figure_files},
                "tables_sha256": {p.name: sha256(p) for p in table_files},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
