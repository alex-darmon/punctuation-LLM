#!/usr/bin/env python3
"""Render the pre-registered process-simulation displays."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="results/author_panel_20/process_simulation_v1",
    )
    parser.add_argument(
        "--figures",
        default=None,
    )
    parser.add_argument("--allow-engineering-smoke", action="store_true")
    return parser.parse_args()


def read_csv(directory: Path, name: str) -> list[dict[str, str]]:
    with (directory / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def verify_manifest(
    results: Path,
    *,
    allow_engineering_smoke: bool,
) -> dict:
    manifest_path = results / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"simulation manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("engineering_smoke") and not allow_engineering_smoke:
        raise ValueError(
            "refusing to publish engineering-smoke outputs; pass "
            "--allow-engineering-smoke only for visual testing"
        )
    if (
        not manifest.get("engineering_smoke")
        and manifest.get("pre_registration_status") != "frozen_before_simulation"
    ):
        raise ValueError("canonical figures require a frozen pre-registration")
    for relative, expected in manifest.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"analysis source verification failed for {relative}")
    for name in (
        "observations.csv",
        "summaries.csv",
        "sweep.csv",
        "rho_drift.csv",
    ):
        expected = manifest.get("output_sha256", {}).get(name)
        path = results / name
        if not expected or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"manifest verification failed for {name}")
    return manifest


def save(fig: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"  [wrote] {path.relative_to(ROOT)}")
    plt.close(fig)


def phi_curves(summaries: list[dict[str, str]], figures: Path) -> None:
    selected = [
        row
        for row in summaries
        if row["feature"] == "f3"
        and (
            row["phase"] in {"observed", "baseline"}
            or (
                row["phase"] == "tail"
                and float(row["dwell_scale"]) == 1.0
                and float(row["contrast"]) == 1.0
            )
        )
    ]
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in selected:
        key = (row["phase"], row["model"], row["reference"])
        groups.setdefault(key, []).append(row)

    styles = {
        ("observed", "human", "estimated"): ("Human", "#111827", "o", "-"),
        ("baseline", "iid", "oracle"): ("i.i.d., oracle", "#16a34a", "s", "--"),
        ("baseline", "iid", "estimated"): (
            "i.i.d., estimated",
            "#65a30d",
            "s",
            "-",
        ),
        ("baseline", "markov", "oracle"): (
            "Markov, oracle",
            "#2563eb",
            "^",
            "--",
        ),
        ("baseline", "markov", "estimated"): (
            "Markov, estimated",
            "#0284c7",
            "^",
            "-",
        ),
        ("tail", "two_state_hsmm", "estimated"): (
            "Two-state fitted point",
            "#dc2626",
            "D",
            "-",
        ),
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda row: int(row["chunk_size"]))
        label, color, marker, linestyle = styles[key]
        ax.plot(
            [int(row["chunk_size"]) for row in rows],
            [float(row["phi"]) for row in rows],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
        )
    ax.axhline(1.0, color="#94a3b8", linewidth=1, linestyle=":")
    ax.set_xlabel("Chunk length (punctuation marks)")
    ax.set_ylabel(r"Overdispersion $\phi=\overline{G}/df$")
    ax.set_title("Oracle and finite-reference process predictions")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8)
    save(fig, figures, "phi_curves")


def sweep_surface(sweep: list[dict[str, str]], figures: Path) -> None:
    dwell = sorted({float(row["dwell_scale"]) for row in sweep})
    contrast = sorted({float(row["contrast"]) for row in sweep})
    matrix = np.full((len(dwell), len(contrast)), np.nan)
    for row in sweep:
        i = dwell.index(float(row["dwell_scale"]))
        j = contrast.index(float(row["contrast"]))
        matrix[i, j] = float(row["phi_rmse"])

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(contrast)), [f"{value:g}" for value in contrast])
    ax.set_yticks(range(len(dwell)), [f"{value:g}" for value in dwell])
    ax.set_xlabel("Inter-state punctuation contrast")
    ax.set_ylabel("Empirical dwell-length scale")
    ax.set_title(r"Fit to observed $\phi(n)$ (lower RMSE is better)")
    best = np.unravel_index(np.nanargmin(matrix), matrix.shape)
    ax.scatter(best[1], best[0], marker="*", s=150, color="white", edgecolor="black")
    fig.colorbar(image, ax=ax, label=r"RMSE across $\phi(1000,2000,4000)$")
    save(fig, figures, "hsmm_sweep")


def tail_figure(
    observations: list[dict[str, str]],
    figures: Path,
    *,
    chunk_size: int = 2000,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    selections = (
        ("observed", "Human", "#111827"),
        ("tail", "Two-state fitted point", "#dc2626"),
    )
    for phase, label, color in selections:
        values = sorted(
            float(row["excess_delta"])
            for row in observations
            if row["phase"] == phase
            and row["feature"] == "f3"
            and int(row["chunk_size"]) == chunk_size
        )
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", label=label, color=color)
    ax.set_xlabel(r"Excess divergence estimate $(G-df)/(2n)$")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title(f"Tail diagnostic at {chunk_size:,} punctuation marks")
    ax.grid(alpha=0.18)
    ax.legend()
    save(fig, figures, "f3_tail_ecdf")


def rho_drift_figure(rows: list[dict[str, str]], figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharex=True)
    for ax, feature in zip(axes, ("f1", "f3")):
        selected = sorted(
            (row for row in rows if row["feature"] == feature),
            key=lambda row: int(row["chunk_size"]),
        )
        sizes = [int(row["chunk_size"]) for row in selected]
        ax.plot(
            sizes,
            [float(row["observed_rho_at_n"]) for row in selected],
            marker="o",
            color="#111827",
            label="Human",
        )
        ax.plot(
            sizes,
            [float(row["simulated_rho_at_n"]) for row in selected],
            marker="D",
            color="#dc2626",
            label="Two-state fitted point",
        )
        ax.axhline(0.0, color="#94a3b8", linewidth=1, linestyle=":")
        ax.set_title(feature)
        ax.set_xlabel("Chunk length")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel(r"$\rho(n)=[\phi(n)-1]/n$")
    axes[1].legend(fontsize=8)
    fig.suptitle("Length-specific overdispersion drift")
    save(fig, figures, "rho_drift")


def main() -> None:
    args = parse_args()
    results = Path(args.results)
    figures = Path(
        args.figures or "paper/figures/process_simulation_v1"
    )
    if not results.is_absolute():
        results = ROOT / results
    if not figures.is_absolute():
        figures = ROOT / figures
    manifest = verify_manifest(
        results, allow_engineering_smoke=args.allow_engineering_smoke
    )
    if manifest.get("engineering_smoke") and (
        args.figures is None
        or not is_within(figures, ROOT / "results" / "repro_check")
    ):
        raise ValueError(
            "engineering-smoke figures must be explicitly placed under "
            "results/repro_check/"
        )
    summaries = read_csv(results, "summaries.csv")
    sweep = read_csv(results, "sweep.csv")
    observations = read_csv(results, "observations.csv")
    rho_drift = read_csv(results, "rho_drift.csv")
    phi_curves(summaries, figures)
    sweep_surface(sweep, figures)
    tail_figure(observations, figures)
    rho_drift_figure(rho_drift, figures)
    figure_files = sorted(
        path
        for path in figures.iterdir()
        if path.suffix in {".pdf", ".png"}
    )
    figure_manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "results_manifest": str(results / "manifest.json"),
        "results_manifest_sha256": sha256(results / "manifest.json"),
        "plotter_sha256": sha256(Path(__file__)),
        "matplotlib": importlib.metadata.version("matplotlib"),
        "engineering_smoke": manifest.get("engineering_smoke"),
        "figures_sha256": {
            path.name: sha256(path) for path in figure_files
        },
    }
    (figures / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
