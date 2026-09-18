#!/usr/bin/env python3
"""Render the extended dwell sweep: phi(n) by run length, and rho against run length."""

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
CANONICAL_FIGURES = "paper/figures/process_simulation_v2_extended"
OBSERVED = "#111827"
SIMULATED = "#2563eb"
CROSSING = "#dc2626"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", default="results/author_panel_20/process_simulation_v2_extended"
    )
    parser.add_argument("--figures", default=CANONICAL_FIGURES)
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


def averaged_rows(rho: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        (row for row in rho if row["seed"] == "mean"),
        key=lambda row: float(row["dwell_scale"]),
    )


def phi_curves_figure(rho, targets, figures: Path) -> None:
    sizes = sorted(int(size) for size in targets["phi"])
    rows = averaged_rows(rho)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cmap = plt.get_cmap("viridis")
    for index, row in enumerate(rows):
        color = cmap(index / max(len(rows) - 1, 1))
        phi = [float(row[f"phi_{size}"]) for size in sizes]
        ax.fill_between(
            sizes,
            [float(row[f"phi_{size}_seed_min"]) for size in sizes],
            [float(row[f"phi_{size}_seed_max"]) for size in sizes],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(sizes, phi, color=color, linewidth=1.2, marker=".")
        ax.annotate(
            f"{float(row['dwell_scale']):g}",
            (sizes[-1], phi[-1]),
            textcoords="offset points",
            xytext=(5, -3),
            fontsize=7,
            color=color,
        )
    observed = [targets["phi"][str(size)] for size in sizes]
    ax.errorbar(
        sizes,
        observed,
        yerr=[
            [value - targets["phi_ci"][str(size)][0] for size, value in zip(sizes, observed)],
            [targets["phi_ci"][str(size)][1] - value for size, value in zip(sizes, observed)],
        ],
        color=OBSERVED,
        marker="o",
        linewidth=1.8,
        capsize=3,
        label="Observed (95% author bootstrap)",
        zorder=5,
    )
    ax.plot([], [], color=SIMULATED, linewidth=1.2, label="Simulated, by dwell multiplier (band: seed range)")
    ax.axhline(1.0, color="#94a3b8", linewidth=1, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{size:,}" for size in sizes])
    ax.set_xlabel("Chunk size n (marks)")
    ax.set_ylabel(r"$\phi(n)$, f3")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    save(fig, figures, "extended_phi_curves")


def rho_figure(rho, crossing, targets, mean_dwell: float, figures: Path) -> None:
    rows = averaged_rows(rho)
    run_length = [float(row["dwell_scale"]) * mean_dwell for row in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.fill_between(
        run_length,
        [float(row["fixed_intercept_rho_seed_min"]) for row in rows],
        [float(row["fixed_intercept_rho_seed_max"]) for row in rows],
        color=SIMULATED,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        run_length,
        [float(row["fixed_intercept_rho"]) for row in rows],
        color=SIMULATED,
        marker="o",
        linewidth=1.4,
        label="Simulated, contrast 1.0 (band: seed range)",
    )
    target = float(targets["fixed_intercept_rho"])
    ax.axhline(target, color=OBSERVED, linewidth=1.2, label=r"Observed $\rho$")
    averaged = next(row for row in crossing if row["seed"] == "mean")
    if averaged["crossing_run_length"]:
        at = float(averaged["crossing_run_length"])
        ax.axvline(at, color=CROSSING, linewidth=1, linestyle="--")
        ax.plot([at], [target], color=CROSSING, marker="D", zorder=5)
        ax.annotate(
            f"{at:.0f} marks",
            (at, target),
            textcoords="offset points",
            xytext=(6, -14),
            fontsize=9,
            color=CROSSING,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Mean run length (marks)")
    ax.set_ylabel(r"Fixed-intercept $\rho$, f3")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    save(fig, figures, "extended_rho_vs_dwell")


def main() -> None:
    args = parse_args()
    results = ROOT / args.results
    figures = ROOT / args.figures
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    if manifest["engineering_smoke"] and figures.resolve() == (ROOT / CANONICAL_FIGURES).resolve():
        raise ValueError("refusing to plot an engineering-smoke run into the paper figures")
    for name, digest in manifest["output_sha256"].items():
        if sha256(results / name) != digest:
            raise ValueError(f"manifest verification failed for {name}")
    config = json.loads((ROOT / manifest["analysis_config"]).read_text(encoding="utf-8"))
    targets = config["pre_registration"]["observed_targets"]
    rho = read_csv(results / "extended_sweep_rho.csv")
    crossing = read_csv(results / "extended_sweep_crossing.csv")
    mean_dwell = float(manifest["effective_settings"]["pooled_mean_dwell_marks"]["mean"])
    phi_curves_figure(rho, targets, figures)
    rho_figure(rho, crossing, targets, mean_dwell, figures)
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
