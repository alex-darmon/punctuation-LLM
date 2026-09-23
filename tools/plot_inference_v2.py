#!/usr/bin/env python3
"""Generate the principal displays for the 20-author chapter."""

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
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "author_panel_20" / "inference_v2"
FIGURES = ROOT / "paper" / "figures" / "inference_v2"
POSITION_MATCHED = ROOT / "results" / "author_panel_20" / "position_matched_detection"
COLORS = {"human": "#334155", "flash": "#2563eb", "pro": "#ea580c"}
# Set by --protocol; the v2 defaults keep the committed v2 figures reproducible.
PROTOCOL = "standard"
LABELS = {"human": "Human", "flash": "Flash", "pro": "Pro"}
DRIFT_TITLE = {
    "standard": "Both models drift away from target punctuation profiles",
    "repeated_prompt": "Target divergence under the repeated prompt",
}


def protocol_suffix() -> str:
    return " (repeated prompt)" if PROTOCOL == "repeated_prompt" else ""


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = FIGURES / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"  [wrote] {path.relative_to(ROOT)}")
    plt.close(fig)


def author_names() -> dict[str, str]:
    panel = json.loads(
        (ROOT / "campaigns" / "author_panel_20.json").read_text(encoding="utf-8")
    )
    return {author["key"]: author["name"] for author in panel["authors"]}


def separability_figure(names: dict[str, str]) -> None:
    rows = [
        row
        for row in read_csv("separability_crossfit_summary.csv")
        if row["row_type"] == "author" and row["feature"] == "f3"
    ]
    x = np.asarray([float(row["geometric_mean_D_over_C"]) for row in rows])
    y = np.asarray([100 * float(row["chunk_accuracy"]) for row in rows])
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.scatter(x, y, s=48, color="#2563eb", alpha=0.85)
    for row, x_value, y_value in zip(rows, x, y):
        ax.annotate(
            names[row["author"]].split()[-1],
            (x_value, y_value),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axvline(1.0, color="#94a3b8", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Cross-fitted D/C (geometric mean across three held-out books)")
    ax.set_ylabel("Held-out 2,000-mark attribution accuracy (%)")
    ax.set_title("Prospective separability predicts held-out human attribution")
    summary = next(
        row
        for row in read_csv("separability_crossfit_summary.csv")
        if row["row_type"] == "summary" and row["feature"] == "f3"
    )
    ax.text(
        0.02,
        0.98,
        (
            f"Spearman ρ={float(summary['spearman_rho']):.2f} "
            f"[{float(summary['spearman_ci_low']):.2f}, "
            f"{float(summary['spearman_ci_high']):.2f}]"
        ),
        transform=ax.transAxes,
        va="top",
    )
    ax.grid(alpha=0.18)
    save(fig, "crossfit_separability")


def attribution_figure(names: dict[str, str]) -> None:
    rows = [
        row
        for row in read_csv("attribution_clustered.csv")
        if row["row_type"] == "author"
        and row["author_set"] == "all_authors"
        and row["feature"] == "f3"
        and row["chunk_size"] == "2000"
    ]
    indexed = {
        (row["author"], row["condition"] or "human"): row for row in rows
    }
    authors = sorted(
        {row["author"] for row in rows},
        key=lambda author: float(indexed[(author, "human")]["macro_accuracy"]),
    )
    y = np.arange(len(authors))
    offsets = {"human": -0.24, "flash": 0.0, "pro": 0.24}
    fig, ax = plt.subplots(figsize=(9.0, 8.2))
    for condition in ("human", "flash", "pro"):
        points = np.asarray(
            [100 * float(indexed[(author, condition)]["macro_accuracy"]) for author in authors]
        )
        low = np.asarray(
            [100 * float(indexed[(author, condition)]["macro_ci_low"]) for author in authors]
        )
        high = np.asarray(
            [100 * float(indexed[(author, condition)]["macro_ci_high"]) for author in authors]
        )
        ax.errorbar(
            points,
            y + offsets[condition],
            xerr=np.vstack([points - low, high - points]),
            fmt="o",
            markersize=4,
            capsize=2,
            color=COLORS[condition],
            label=LABELS[condition] + (protocol_suffix() if condition != "human" else ""),
        )
    ax.axvline(5.0, color="#94a3b8", linestyle="--", linewidth=1, label="20-way chance")
    ax.set_yticks(y, [names[author] for author in authors], fontsize=8)
    ax.set_xlim(-3, 105)
    ax.set_xlabel("Target-author attribution accuracy (%)")
    ax.set_title("Human identifiability is strong; LLM imitation is weak and heterogeneous")
    ax.legend(ncol=4, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.18)
    save(fig, "attribution_by_author")


def roc_curve(positive: np.ndarray, negative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[positive, negative]))[::-1], -np.inf]
    fpr = np.asarray([np.mean(negative > value) for value in thresholds])
    tpr = np.asarray([np.mean(positive > value) for value in thresholds])
    return fpr, tpr


def detection_figure() -> None:
    rows = read_csv("detection_observations.csv")
    estimates = {
        row["condition"]: row
        for row in read_csv("detection_crossfit.csv")
        if row["method"] == "out_of_author_crossfit"
        and row["feature"] == "f3"
        and row["chunk_size"] == "2000"
        and row["smoothing_eps"] == "0.5"
    }
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    for condition in ("flash", "pro"):
        condition_rows = [row for row in rows if row["condition"] == condition]
        positive = np.asarray(
            [
                float(row["normalised_score"])
                for row in condition_rows
                if row["label"] == "llm"
            ]
        )
        negative = np.asarray(
            [
                float(row["normalised_score"])
                for row in condition_rows
                if row["label"] == "human"
            ]
        )
        fpr, tpr = roc_curve(positive, negative)
        estimate = estimates[condition]
        ax.plot(
            fpr,
            tpr,
            color=COLORS[condition],
            linewidth=2,
            label=f"{LABELS[condition]}{protocol_suffix()} (AUC {float(estimate['auc']):.2f})",
        )
        ax.scatter(
            [float(estimate["empirical_fpr"])],
            [float(estimate["tpr"])],
            color=COLORS[condition],
            s=45,
            zorder=3,
        )
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1)
    ax.axvline(0.05, color="#64748b", linestyle=":", linewidth=1)
    ax.set_xlabel("Held-out human false-positive rate")
    ax.set_ylabel("LLM true-positive rate")
    ax.set_title("Out-of-author detection and the calibrated 5% operating point")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.18)
    save(fig, "detection_oof_roc")


def calibration_figure() -> None:
    rows = [
        row
        for row in read_csv("../full_grid/calibration.csv")
        if row["chunk_size"] == "2000"
        and row["policy"] == "leave_one_book_out"
    ]
    labels = [row["feature"] for row in rows]
    nominal = [100 * float(row["nominal_5pct_rejection_rate"]) for row in rows]
    detection = [
        row
        for row in read_csv("detection_crossfit.csv")
        if row["method"] == "out_of_author_crossfit"
        and row["condition"] == "flash"
        and row["chunk_size"] == "2000"
        and row["smoothing_eps"] == "0.5"
    ]
    held_out = {
        row["feature"]: 100 * float(row["empirical_fpr"]) for row in detection
    }
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    width = 0.34
    ax.bar(x - width / 2, nominal, width, color="#dc2626", label=r"Nominal $\chi^2$ test")
    ax.bar(
        x + width / 2,
        [held_out[label] for label in labels],
        width,
        color="#2563eb",
        label="Out-of-author empirical calibration",
    )
    ax.axhline(5.0, color="#111827", linestyle="--", linewidth=1, label="Target 5%")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Genuine human texts rejected (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Asymptotic G calibration fails on dependent punctuation")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    save(fig, "g_calibration")


def drift_figure() -> None:
    rows = [
        row
        for row in read_csv("drift_clustered.csv")
        if row["row_type"] == "position"
        and row["feature"] == "f3"
        and row["metric"] == "target_kl"
    ]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for condition in ("flash", "pro"):
        selected = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: int(row["window_index"]),
        )
        x = np.asarray([int(row["window_index"]) for row in selected])
        y = np.asarray([float(row["estimate"]) for row in selected])
        low = np.asarray([float(row["ci_low"]) for row in selected])
        high = np.asarray([float(row["ci_high"]) for row in selected])
        ax.plot(x, y, marker="o", color=COLORS[condition], label=condition.capitalize())
        ax.fill_between(x, low, high, color=COLORS[condition], alpha=0.15)
    ax.set_xticks(range(1, 6), [f"{start}–{start + 1000}" for start in range(0, 5000, 1000)])
    ax.set_xlabel("Consecutive punctuation-mark window")
    ax.set_ylabel("Author-equal mean target KL")
    ax.set_title(DRIFT_TITLE[PROTOCOL])
    ax.legend()
    ax.grid(alpha=0.18)
    save(fig, "positional_drift")


def position_matched_figure() -> None:
    path = POSITION_MATCHED / "detection_position.csv"
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))

    equal = [
        row
        for row in rows
        if row["chunk_size"] == "2000" and row["window_name"] in {"prefix", "late"}
    ]
    ax = axes[0]
    x = np.arange(2)
    width = 0.32
    for offset, window in ((-width / 2, "prefix"), (width / 2, "late")):
        values = []
        for condition in ("flash", "pro"):
            row = next(
                item
                for item in equal
                if item["condition"] == condition and item["window_name"] == window
            )
            values.append(100 * float(row["tpr"]))
        ax.bar(
            x + offset,
            values,
            width,
            color="#1e293b" if window == "prefix" else "#7c3aed",
            label="Marks 1–2,000" if window == "prefix" else "Marks 2,001–4,000",
        )
    ax.set_xticks(x, ["Flash", "Pro"])
    ax.set_ylabel("Out-of-author TPR (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Equal length, different position" + protocol_suffix())
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.18)

    ax = axes[1]
    for condition in ("flash", "pro"):
        selected = sorted(
            (
                row
                for row in rows
                if row["condition"] == condition and row["chunk_size"] == "1000"
            ),
            key=lambda row: int(row["start_mark"]),
        )
        x_vals = np.asarray([int(row["start_mark"]) + 500 for row in selected])
        y_vals = np.asarray([100 * float(row["tpr"]) for row in selected])
        low = np.asarray([100 * float(row["tpr_ci_low"]) for row in selected])
        high = np.asarray([100 * float(row["tpr_ci_high"]) for row in selected])
        ax.plot(
            x_vals,
            y_vals,
            marker="o",
            color=COLORS[condition],
            label=condition.capitalize(),
        )
        ax.fill_between(x_vals, low, high, color=COLORS[condition], alpha=0.15)
    ax.set_xticks(range(500, 5000, 1000), [f"{start}–{start + 1000}" for start in range(0, 5000, 1000)])
    ax.set_xlabel("1,000-mark window")
    ax.set_ylabel("Out-of-author TPR (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Five-window trajectory at fixed length")
    ax.legend()
    ax.grid(alpha=0.18)
    fig.tight_layout()
    save(fig, "position_matched_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS.relative_to(ROOT)))
    parser.add_argument("--figures", default=str(FIGURES.relative_to(ROOT)))
    parser.add_argument("--position-matched", default=str(POSITION_MATCHED.relative_to(ROOT)))
    parser.add_argument(
        "--protocol",
        choices=sorted(DRIFT_TITLE),
        default="standard",
        help="Generation protocol of the plotted runs; labels the generated-text series.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_figure_manifest() -> None:
    """Tie the figures to the results they were drawn from."""
    sources = {
        "inference_manifest": RESULTS / "inference_manifest.json",
        "position_matched_manifest": POSITION_MATCHED / "manifest.json",
    }
    figure_files = sorted(p for p in FIGURES.iterdir() if p.suffix in {".pdf", ".png"})
    (FIGURES / "figure_manifest.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": PROTOCOL,
                "results": {
                    name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                    for name, path in sources.items()
                    if path.is_file()
                },
                "plotter_sha256": sha256(Path(__file__)),
                "matplotlib": importlib.metadata.version("matplotlib"),
                "figures_sha256": {p.name: sha256(p) for p in figure_files},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    global RESULTS, FIGURES, POSITION_MATCHED, PROTOCOL
    args = parse_args()
    RESULTS = ROOT / args.results
    FIGURES = ROOT / args.figures
    POSITION_MATCHED = ROOT / args.position_matched
    PROTOCOL = args.protocol
    names = author_names()
    separability_figure(names)
    attribution_figure(names)
    detection_figure()
    calibration_figure()
    drift_figure()
    position_matched_figure()
    if PROTOCOL != "standard":
        write_figure_manifest()


if __name__ == "__main__":
    main()
