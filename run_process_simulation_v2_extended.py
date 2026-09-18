#!/usr/bin/env python3
"""Run the pre-specified extended dwell sweep of the two-state process.

This is a thin extension of `run_process_simulation.py`: the same
`simulate_cell`, sampling frame, author folds and hash-locked v1 parameter
artifact, with a new phase name, a longer dwell grid at the measured contrast
and three seeds.  Nothing is refitted or re-segmented, and no v1 output or
source file is touched (the v1 manifest pins `run_process_simulation.py` by
hash, which is why this lives in its own file).

  python run_process_simulation_v2_extended.py

The parameter-artifact hash is checked in every mode.  Implementation checks
may bypass the freeze gate only, and the manifest records that they did:

  python run_process_simulation_v2_extended.py --engineering-smoke \
      --dwell-scales 1 --replicates 400 \
      --output-dir results/repro_check/process_simulation_v2_smoke
"""

from __future__ import annotations

import argparse
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from punctlib import load_corpus
from punctlib.simulation import FittedProcess
from punctlib.stats import dof
from run_process_simulation import (
    ROOT,
    git_state,
    is_within,
    load_fitted_parameters,
    make_designs,
    package_versions,
    read_csv,
    resolve,
    sha256,
    simulate_cell,
    source_hashes,
    stable_seed,
    write_csv,
)


DEFAULT_CONFIG = "campaigns/process_simulation_v2_extended_sweep.json"
FROZEN_STATUS = "frozen_before_simulation"
FEATURE = "f3"
BOOTSTRAP_DRAWS = 2000
SEED_AVERAGE = "mean"
OUTPUT_NAMES = (
    "extended_sweep_observations.csv",
    "extended_sweep_summary.csv",
    "extended_sweep_rho.csv",
    "extended_sweep_crossing.csv",
    "cap.csv",
    "predictions_check.json",
    "manifest.json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--reference-length-cap", type=int, default=None)
    parser.add_argument(
        "--dwell-scales",
        type=float,
        nargs="+",
        default=None,
        help="Replace the declared dwell grid (engineering smoke only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes for independent cells; results do not depend on it.",
    )
    parser.add_argument(
        "--engineering-smoke",
        action="store_true",
        help="Bypass the freeze gate; recorded in manifest.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Gates


def validate_execution_mode(
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    overrides = {
        "--output-dir": args.output_dir,
        "--replicates": args.replicates,
        "--reference-length-cap": args.reference_length_cap,
        "--dwell-scales": args.dwell_scales,
    }
    active_overrides = [name for name, value in overrides.items() if value is not None]
    if args.engineering_smoke:
        smoke_root = ROOT / "results" / "repro_check"
        if args.output_dir is None or not is_within(output_dir, smoke_root):
            raise ValueError(
                "engineering smoke output must be explicitly placed under "
                "results/repro_check/"
            )
        return
    if active_overrides:
        raise ValueError(
            "runtime overrides are allowed only with --engineering-smoke: "
            + ", ".join(active_overrides)
        )
    if output_dir.resolve() != resolve(config["output_dir"]).resolve():
        raise ValueError("inferential runs must use the configured canonical output")
    if config["pre_registration"]["status"] != FROZEN_STATUS:
        raise ValueError(
            "the pre-registration is not frozen: review the predictions, set "
            f"status to {FROZEN_STATUS}, and commit the config before the run"
        )


def validate_parameter_artifact(path: Path, expected_sha256: str | None) -> str:
    """The v1 artifact is reused unchanged; this is enforced in every mode."""
    if not expected_sha256:
        raise ValueError("parameter_artifact.sha256 must be recorded in the config")
    if not path.is_file():
        raise FileNotFoundError(f"frozen parameter artifact is missing: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"parameter artifact hash mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def validate_parent(
    config: dict[str, Any],
    parent: dict[str, Any],
    artifact: dict[str, Any],
    cache_sha256: str,
) -> None:
    parent_path = resolve(config["parent_config"])
    if sha256(parent_path) != config["parent_config_sha256"]:
        raise ValueError("parent (v1) config hash differs from the recorded one")
    for key in ("authors_config", "cache"):
        if config[key] != parent[key]:
            raise ValueError(f"{key} differs from the parent config")
    if parent["parameter_artifact"] != {
        key: config["parameter_artifact"][key] for key in ("path", "sha256")
    }:
        raise ValueError("parameter artifact differs from the parent config's")
    if artifact["inputs"]["cache"]["sha256"] != cache_sha256:
        raise ValueError("cache differs from the one the parameters were fitted on")
    if artifact["inputs"]["authors_config"]["sha256"] != sha256(
        resolve(config["authors_config"])
    ):
        raise ValueError("author panel differs from the one the parameters were fitted on")
    if artifact["source_sha256"] != source_hashes():
        raise ValueError("simulator source changed after the v1 artifact was created")
    if artifact["fold_by_author"] != parent["evaluation"]["fold_by_author"]:
        raise ValueError("parameter artifact author folds do not match the parent config")
    simulation = config["simulation"]
    if float(artifact["smoothing_eps"]) != float(simulation["smoothing_eps"]):
        raise ValueError("smoothing differs from the parameter artifact")
    if list(simulation["chunk_sizes"]) != list(parent["simulation"]["chunk_sizes"]):
        raise ValueError("chunk sizes differ from the parent config")


def validate_observed_targets(config: dict[str, Any], parent: dict[str, Any]) -> None:
    """The observed targets must be the v1 numbers, not retyped approximations."""
    targets = config["pre_registration"]["observed_targets"]
    primary = parent["pre_registration"]["primary_targets"]
    for size, value in targets["phi"].items():
        if not np.isclose(value, float(primary["phi"][size]), rtol=0, atol=1e-12):
            raise ValueError(f"observed phi({size}) differs from the v1 target")
    if not np.isclose(
        targets["fixed_intercept_rho"],
        float(primary["fixed_intercept_rho"]),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("observed rho differs from the v1 target")
    summaries = resolve(parent["output_dir"]) / "summaries.csv"
    if not summaries.is_file():
        raise FileNotFoundError(f"v1 summaries are missing: {summaries}")
    intervals = {
        row["chunk_size"]: (float(row["phi_ci_low"]), float(row["phi_ci_high"]))
        for row in read_csv(summaries)
        if row["phase"] == "observed" and row["feature"] == FEATURE
    }
    for size, declared in targets["phi_ci"].items():
        if size not in intervals or not np.allclose(
            declared, intervals[size], rtol=0, atol=1e-6
        ):
            raise ValueError(
                f"observed phi({size}) interval differs from v1 summaries.csv"
            )


# --------------------------------------------------------------------------
# Estimands


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    support = p > 0
    return float(np.sum(p[support] * np.log(p[support] / q[support])))


def pure_mode_cap(process: FittedProcess) -> dict[str, float]:
    """Excess divergence if every chunk sat wholly inside one mode.

    f3: share-weighted sum over rows of pi_s(i) KL(T_s(i,.) || T_pool(i,.));
    f1: share-weighted KL(pi_s || pi_pool).  Analytic, no simulation.
    """
    shares = np.asarray(process.state_mark_counts, dtype=float)
    shares = shares / shares.sum()
    cap_f1 = 0.0
    cap_f3 = 0.0
    for share, marginal, transition in zip(
        shares, process.state_marginals, process.state_transitions
    ):
        cap_f1 += share * kl_divergence(marginal, process.marginal)
        cap_f3 += share * sum(
            marginal[row] * kl_divergence(transition[row], process.transition[row])
            for row in range(len(marginal))
        )
    return {"cap_f1": float(cap_f1), "cap_f3": float(cap_f3)}


def cap_rows(
    fitted: dict[int, FittedProcess],
    observed_delta: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, process in sorted(fitted.items()):
        cap = pure_mode_cap(process)
        rows.append({"fold": fold, **cap})
    rows.append(
        {
            "fold": SEED_AVERAGE,
            "cap_f1": float(np.mean([row["cap_f1"] for row in rows])),
            "cap_f3": float(np.mean([row["cap_f3"] for row in rows])),
        }
    )
    for row in rows:
        row["observed_delta_f3"] = observed_delta
        row["cap_f3_over_observed_delta"] = row["cap_f3"] / observed_delta
    return rows


def pooled_mean_dwell(fitted: dict[int, FittedProcess]) -> dict[str, float]:
    """Mean block length over both states per fold; folds averaged equally."""
    by_fold = [
        float(np.mean(np.concatenate(process.dwell_lengths)))
        for _, process in sorted(fitted.items())
    ]
    return {
        "mean": float(np.mean(by_fold)),
        "fold_min": float(np.min(by_fold)),
        "fold_max": float(np.max(by_fold)),
    }


def fixed_intercept_rho(sizes: Sequence[float], phi: Sequence[float]) -> float:
    sizes = np.asarray(sizes, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return float(np.dot(sizes, phi - 1.0) / np.dot(sizes, sizes))


def crossing_multiplier(
    scales: Sequence[float],
    values: Sequence[float],
    target: float,
) -> float | None:
    """First dwell_scale at which the piecewise-linear curve equals target."""
    order = np.argsort(scales)
    x = [float(scales[index]) for index in order]
    y = [float(values[index]) for index in order]
    for index, (scale, value) in enumerate(zip(x, y)):
        if value == target:
            return scale
        if index + 1 == len(x):
            break
        following = y[index + 1]
        if (value - target) * (following - target) < 0:
            return scale + (target - value) * (x[index + 1] - scale) / (
                following - value
            )
    return None


def interpolate(scales: Sequence[float], values: Sequence[float], at: float) -> float:
    order = np.argsort(scales)
    return float(
        np.interp(
            at,
            [float(scales[index]) for index in order],
            [float(values[index]) for index in order],
        )
    )


def observation_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return [
        {
            "dwell_scale": row["dwell_scale"],
            "seed": seed,
            "fold": row["fold"],
            "author": row["design_author"],
            "target_book": row["design_target_book"],
            "chunk_size": row["chunk_size"],
            "chunk_index": row["design_target_chunk"],
            "replicate": row["replicate"],
            "G": row["g"],
            "excess_delta": row["excess_delta"],
        }
        for row in rows
        if row["feature"] == FEATURE
    ]


def summary_rows(
    observations: list[dict[str, Any]],
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int, int], list[float]] = {}
    for row in observations:
        key = (float(row["dwell_scale"]), int(row["seed"]), int(row["chunk_size"]))
        grouped.setdefault(key, []).append(float(row["G"]))
    degrees = dof(FEATURE)
    output = []
    for (dwell_scale, seed, size), values in sorted(grouped.items()):
        g = np.asarray(values)
        rng = np.random.default_rng(
            stable_seed(seed, "chunk-bootstrap", dwell_scale, size)
        )
        draws = g[rng.integers(len(g), size=(bootstrap_draws, len(g)))].mean(axis=1)
        output.append(
            {
                "dwell_scale": dwell_scale,
                "seed": seed,
                "chunk_size": size,
                "n_observations": len(g),
                "mean_G": float(g.mean()),
                "phi": float(g.mean() / degrees),
                "phi_ci_low": float(np.quantile(draws, 0.025) / degrees),
                "phi_ci_high": float(np.quantile(draws, 0.975) / degrees),
            }
        )
    return output


def rho_rows(
    summaries: list[dict[str, Any]],
    observed_phi: dict[int, float],
) -> list[dict[str, Any]]:
    """Per-seed rows, then one seed-averaged row per dwell_scale.

    Seed-averaged estimands are computed from the seed-averaged phi(n); rho is
    linear in phi, so that equals the mean of the per-seed rho.
    """
    sizes = sorted(observed_phi)
    phi: dict[float, dict[int, dict[int, float]]] = {}
    for row in summaries:
        phi.setdefault(row["dwell_scale"], {}).setdefault(row["seed"], {})[
            row["chunk_size"]
        ] = row["phi"]

    def estimands(values: dict[int, float]) -> dict[str, float]:
        return {
            **{f"phi_{size}": values[size] for size in sizes},
            "fixed_intercept_rho": fixed_intercept_rho(
                sizes, [values[size] for size in sizes]
            ),
            "growth_ratio": values[4000] / values[2000],
            "phi_rmse": float(
                np.sqrt(
                    np.mean(
                        [(values[size] - observed_phi[size]) ** 2 for size in sizes]
                    )
                )
            ),
        }

    per_seed = []
    averaged = []
    for dwell_scale in sorted(phi):
        seed_rows = [
            {"dwell_scale": dwell_scale, "seed": seed, **estimands(values)}
            for seed, values in sorted(phi[dwell_scale].items())
        ]
        per_seed.extend(seed_rows)
        mean_phi = {
            size: float(np.mean([row[f"phi_{size}"] for row in seed_rows]))
            for size in sizes
        }
        row = {"dwell_scale": dwell_scale, "seed": SEED_AVERAGE, **estimands(mean_phi)}
        for key in (*(f"phi_{size}" for size in sizes), "fixed_intercept_rho"):
            row[f"{key}_seed_min"] = min(item[key] for item in seed_rows)
            row[f"{key}_seed_max"] = max(item[key] for item in seed_rows)
        averaged.append(row)
    return per_seed + averaged


def crossing_rows(
    rho: list[dict[str, Any]],
    target: float,
    mean_dwell: dict[str, float],
) -> list[dict[str, Any]]:
    output = []
    seeds = sorted({row["seed"] for row in rho}, key=str)
    for seed in seeds:
        rows = [row for row in rho if row["seed"] == seed]
        crossing = crossing_multiplier(
            [row["dwell_scale"] for row in rows],
            [row["fixed_intercept_rho"] for row in rows],
            target,
        )
        output.append(
            {
                "seed": seed,
                "target_rho": target,
                "crossing_multiplier": "" if crossing is None else crossing,
                "crossing_run_length": (
                    "" if crossing is None else crossing * mean_dwell["mean"]
                ),
                "mean_dwell_marks": mean_dwell["mean"],
            }
        )
    seed_level = [
        row["crossing_multiplier"]
        for row in output
        if row["seed"] != SEED_AVERAGE and row["crossing_multiplier"] != ""
    ]
    for row in output:
        if row["seed"] == SEED_AVERAGE:
            row["n_seeds_crossing"] = len(seed_level)
            row["crossing_multiplier_seed_min"] = min(seed_level) if seed_level else ""
            row["crossing_multiplier_seed_max"] = max(seed_level) if seed_level else ""
    return output


def predictions_check(
    config: dict[str, Any],
    cap: list[dict[str, Any]],
    rho: list[dict[str, Any]],
    crossing: list[dict[str, Any]],
) -> dict[str, Any]:
    targets = config["pre_registration"]["observed_targets"]
    observed_phi = {int(size): value for size, value in targets["phi"].items()}
    intervals = {int(size): value for size, value in targets["phi_ci"].items()}
    observed_growth = observed_phi[4000] / observed_phi[2000]
    observed_delta = cap[0]["observed_delta_f3"]

    fold_caps = [row["cap_f3"] for row in cap if row["fold"] != SEED_AVERAGE]
    p1 = {
        "holds": bool(min(fold_caps) > observed_delta),
        "min_fold_cap_f3": min(fold_caps),
        "max_fold_cap_f3": max(fold_caps),
        "observed_delta_f3": observed_delta,
        "threshold_note": "observed delta = fixed_intercept_rho * 90 / 2 (0.028 nats)",
    }

    averaged = next(row for row in crossing if row["seed"] == SEED_AVERAGE)
    multiplier = averaged["crossing_multiplier"]
    crossed = multiplier != ""
    p2 = {
        "holds": bool(crossed and 20 <= multiplier <= 52),
        "crossing_multiplier": multiplier if crossed else None,
        "crossing_run_length": averaged["crossing_run_length"] if crossed else None,
        "crossing_multiplier_seed_range": [
            averaged["crossing_multiplier_seed_min"] or None,
            averaged["crossing_multiplier_seed_max"] or None,
        ],
        "n_seeds_crossing": averaged["n_seeds_crossing"],
        "window": [20, 52],
    }

    if crossed:
        # phi(n) at the seed-averaged crossing multiplier, linearly interpolated
        # in dwell_scale exactly as the crossing itself is.
        at_crossing: dict[str, dict[str, float]] = {}
        for seed in sorted({row["seed"] for row in rho}, key=str):
            rows = [row for row in rho if row["seed"] == seed]
            scales = [row["dwell_scale"] for row in rows]
            values = {
                size: interpolate(
                    scales, [row[f"phi_{size}"] for row in rows], multiplier
                )
                for size in observed_phi
            }
            at_crossing[str(seed)] = {
                **{f"phi_{size}": value for size, value in values.items()},
                "growth_ratio": values[4000] / values[2000],
            }
        sides = [
            (
                values["phi_1000"] > observed_phi[1000],
                values["phi_4000"] < observed_phi[4000],
                values["growth_ratio"] < observed_growth,
            )
            for values in at_crossing.values()
        ]
        p3 = {
            "holds": bool(all(all(side) for side in sides)),
            "at_crossing": at_crossing,
            "observed": {
                "phi_1000": observed_phi[1000],
                "phi_4000": observed_phi[4000],
                "growth_ratio": observed_growth,
            },
        }
    else:
        p3 = {"holds": None, "reason": "seed-averaged rho never crosses the target"}

    matches = []
    for row in rho:
        if row["seed"] != SEED_AVERAGE:
            continue
        inside = {
            str(size): bool(low <= row[f"phi_{size}"] <= high)
            for size, (low, high) in intervals.items()
        }
        matches.append(
            {
                "dwell_scale": row["dwell_scale"],
                "inside_interval": inside,
                "all_inside": all(inside.values()),
            }
        )
    p4 = {
        "holds": not any(item["all_inside"] for item in matches),
        "by_dwell_scale": matches,
    }
    return {
        "P1_cap": p1,
        "P2_level": p2,
        "P3_shape": p3,
        "P4_no_single_match": p4,
    }


# --------------------------------------------------------------------------
# Simulation


def run_cell(task: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    simulate_cell(
        rows=rows,
        phase=task["phase"],
        model=task["model"],
        references=[task["reference"]],
        replicates=task["replicates"],
        dwell_scale=task["dwell_scale"],
        contrast=task["contrast"],
        chunk_sizes=task["chunk_sizes"],
        designs=task["designs"],
        fitted=task["fitted"],
        eps=task["eps"],
        base_seed=task["seed"],
        reference_length_cap=task["reference_length_cap"],
    )
    return observation_rows(rows, task["seed"])


def simulate_grid(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    """Cells are seeded independently, so the output is identical for any workers."""
    observations: list[dict[str, Any]] = []
    if workers <= 1:
        results = map(run_cell, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(run_cell, tasks)
    for done, (task, rows) in enumerate(zip(tasks, results), 1):
        observations.extend(rows)
        print(
            f"  [{done}/{len(tasks)}] dwell_scale={task['dwell_scale']:g} "
            f"seed={task['seed']}",
            flush=True,
        )
    return observations


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    simulation = config["simulation"]
    output_dir = resolve(args.output_dir or config["output_dir"])
    validate_execution_mode(args, config, output_dir)

    parameter_path = resolve(config["parameter_artifact"]["path"])
    parameter_sha256 = validate_parameter_artifact(
        parameter_path, config["parameter_artifact"]["sha256"]
    )
    if not args.engineering_smoke:
        existing = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite canonical simulation outputs: "
                + ", ".join(existing)
            )

    print("[1/4] checking the v1 parent, artifact and observed targets")
    parent = json.loads(resolve(config["parent_config"]).read_text(encoding="utf-8"))
    artifact = json.loads(parameter_path.read_text(encoding="utf-8"))
    corpus = load_corpus(config["authors_config"], config["cache"])
    validate_parent(config, parent, artifact, corpus.cache_sha256)
    validate_observed_targets(config, parent)
    fitted = load_fitted_parameters(parameter_path)
    fold_by_author = {
        author: int(fold) for author, fold in artifact["fold_by_author"].items()
    }
    if set(fold_by_author) != set(corpus.authors):
        raise ValueError("artifact author folds do not match the loaded corpus")
    designs = make_designs(corpus, fold_by_author)

    targets = config["pre_registration"]["observed_targets"]
    observed_phi = {int(size): float(value) for size, value in targets["phi"].items()}
    observed_rho = float(targets["fixed_intercept_rho"])
    observed_delta = observed_rho * dof(FEATURE) / 2.0

    print("[2/4] analytic pure-mode cap")
    cap = cap_rows(fitted, observed_delta)
    mean_dwell = pooled_mean_dwell(fitted)

    replicates = (
        args.replicates
        if args.replicates is not None
        else int(simulation["replicates_per_cell_per_seed"])
    )
    reference_length_cap = (
        args.reference_length_cap
        if args.reference_length_cap is not None
        else simulation["reference_length_cap"]
    )
    dwell_scales = [
        float(value) for value in (args.dwell_scales or simulation["dwell_scales"])
    ]
    seeds = [int(seed) for seed in simulation["seeds"]]
    tasks = [
        {
            "phase": simulation["phase_name"],
            "model": simulation["model"],
            "reference": simulation["reference"],
            "replicates": replicates,
            "dwell_scale": dwell_scale,
            "contrast": float(simulation["contrast"]),
            "chunk_sizes": [int(size) for size in simulation["chunk_sizes"]],
            "designs": designs,
            "fitted": fitted,
            "eps": float(simulation["smoothing_eps"]),
            "seed": seed,
            "reference_length_cap": reference_length_cap,
        }
        for dwell_scale in dwell_scales
        for seed in seeds
    ]
    print(f"[3/4] simulating {len(tasks)} cells x {replicates} replicates")
    observations = simulate_grid(tasks, args.workers)

    print("[4/4] estimands, verdicts and provenance")
    summaries = summary_rows(observations)
    rho = rho_rows(summaries, observed_phi)
    crossing = crossing_rows(rho, observed_rho, mean_dwell)
    verdicts = predictions_check(config, cap, rho, crossing)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "extended_sweep_observations.csv", observations)
    write_csv(output_dir / "extended_sweep_summary.csv", summaries)
    write_csv(output_dir / "extended_sweep_rho.csv", rho)
    write_csv(output_dir / "extended_sweep_crossing.csv", crossing)
    write_csv(output_dir / "cap.csv", cap)
    (output_dir / "predictions_check.json").write_text(
        json.dumps(verdicts, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  [wrote] {(output_dir / 'predictions_check.json').relative_to(ROOT)}")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_config": str(config_path.relative_to(ROOT)),
        "analysis_config_sha256": sha256(config_path),
        "parent_config": config["parent_config"],
        "parent_config_sha256": sha256(resolve(config["parent_config"])),
        "parameter_artifact": {
            "path": str(parameter_path.relative_to(ROOT)),
            "sha256": parameter_sha256,
        },
        "authors_config": config["authors_config"],
        "authors_config_sha256": sha256(resolve(config["authors_config"])),
        "cache": config["cache"],
        "cache_sha256": corpus.cache_sha256,
        "git": git_state(),
        "environment": {
            "python": platform.python_version(),
            "packages": package_versions(),
        },
        "engineering_smoke": args.engineering_smoke,
        "pre_registration_status": config["pre_registration"]["status"],
        "effective_settings": {
            "phase": simulation["phase_name"],
            "contrast": float(simulation["contrast"]),
            "dwell_scales": dwell_scales,
            "seeds": seeds,
            "replicates_per_cell_per_seed": replicates,
            "reference_length_cap": reference_length_cap,
            "chunk_bootstrap_draws": BOOTSTRAP_DRAWS,
            "pooled_mean_dwell_marks": mean_dwell,
        },
        "source_sha256": {
            **source_hashes(),
            Path(__file__).name: sha256(Path(__file__).resolve()),
        },
        "output_sha256": {
            name: sha256(output_dir / name) for name in OUTPUT_NAMES[:-1]
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  [wrote] {(output_dir / 'manifest.json').relative_to(ROOT)}")
    for name, verdict in verdicts.items():
        print(f"  {name}: holds={verdict['holds']}")


if __name__ == "__main__":
    main()
