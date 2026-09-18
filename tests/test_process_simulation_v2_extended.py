#!/usr/bin/env python3
"""Checks for the extended dwell sweep: gates, estimands and provenance."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from punctlib.simulation import FittedProcess
from run_process_simulation_v2_extended import (
    DEFAULT_CONFIG,
    OUTPUT_NAMES,
    ROOT,
    crossing_multiplier,
    main,
    pure_mode_cap,
    sha256,
    validate_execution_mode,
    validate_parameter_artifact,
)


SMOKE_ROOT = ROOT / "results" / "repro_check"
# v1 tail phase at the fitted point, 2000 replicates (summaries.csv).
V1_FITTED_PHI = {1000: 0.756, 2000: 0.880, 4000: 1.016}


def two_state_process(
    marginals: tuple[np.ndarray, np.ndarray],
    transitions: tuple[np.ndarray, np.ndarray],
    counts: tuple[int, int],
) -> FittedProcess:
    shares = np.asarray(counts, dtype=float) / sum(counts)
    return FittedProcess(
        marginal=shares[0] * marginals[0] + shares[1] * marginals[1],
        transition=shares[0] * transitions[0] + shares[1] * transitions[1],
        state_marginals=marginals,
        state_transitions=transitions,
        dwell_lengths=(np.asarray([5]), np.asarray([5])),
        initial_state=shares,
        n_books=1,
        n_marks=sum(counts),
        state_mark_counts=counts,
        state_block_counts=(1, 1),
    )


def namespace(**overrides: object) -> SimpleNamespace:
    values = {
        "engineering_smoke": False,
        "output_dir": None,
        "replicates": None,
        "reference_length_cap": None,
        "dwell_scales": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class GateTests(unittest.TestCase):
    def test_refuses_a_parameter_artifact_with_a_different_hash(self) -> None:
        config = json.loads((ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8"))
        frozen = ROOT / config["parameter_artifact"]["path"]
        self.assertEqual(
            validate_parameter_artifact(frozen, config["parameter_artifact"]["sha256"]),
            sha256(frozen),
        )
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "frozen_parameters.json"
            altered.write_bytes(frozen.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_parameter_artifact(
                    altered, config["parameter_artifact"]["sha256"]
                )
        with self.assertRaises(ValueError):
            validate_parameter_artifact(frozen, None)

    def test_hash_is_enforced_even_in_engineering_smoke(self) -> None:
        config = json.loads((ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8"))
        config["parameter_artifact"]["sha256"] = "0" * 64
        SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(dir=SMOKE_ROOT))
        try:
            altered = directory / "config.json"
            altered.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                main(
                    [
                        "--config",
                        str(altered),
                        "--engineering-smoke",
                        "--output-dir",
                        str(directory / "out"),
                    ]
                )
        finally:
            shutil.rmtree(directory)

    def test_inferential_run_requires_frozen_status(self) -> None:
        config = {
            "output_dir": "results/canonical",
            "pre_registration": {"status": "draft_not_frozen"},
        }
        with self.assertRaisesRegex(ValueError, "not frozen"):
            validate_execution_mode(
                namespace(), config, ROOT / config["output_dir"]
            )
        config["pre_registration"]["status"] = "frozen_before_simulation"
        validate_execution_mode(namespace(), config, ROOT / config["output_dir"])

    def test_overrides_and_smoke_location_are_policed(self) -> None:
        config = {
            "output_dir": "results/canonical",
            "pre_registration": {"status": "frozen_before_simulation"},
        }
        with self.assertRaisesRegex(ValueError, "overrides"):
            validate_execution_mode(
                namespace(dwell_scales=[1.0]), config, ROOT / config["output_dir"]
            )
        with self.assertRaisesRegex(ValueError, "repro_check"):
            validate_execution_mode(
                namespace(engineering_smoke=True, output_dir="results/canonical"),
                config,
                ROOT / config["output_dir"],
            )
        validate_execution_mode(
            namespace(engineering_smoke=True, output_dir="results/repro_check/x"),
            {"output_dir": "results/canonical",
             "pre_registration": {"status": "draft_not_frozen"}},
            SMOKE_ROOT / "x",
        )


class CrossingTests(unittest.TestCase):
    def test_exact_grid_point(self) -> None:
        self.assertEqual(crossing_multiplier([8, 16, 32], [0.1, 0.5, 0.9], 0.5), 16.0)

    def test_linear_interpolation_between_grid_points(self) -> None:
        self.assertAlmostEqual(
            crossing_multiplier([32, 8, 16], [0.9, 0.1, 0.5], 0.7), 24.0
        )

    def test_none_when_the_curve_never_crosses(self) -> None:
        self.assertIsNone(crossing_multiplier([8, 16, 32], [0.1, 0.2, 0.3], 0.5))
        self.assertIsNone(crossing_multiplier([8, 16, 32], [0.6, 0.7, 0.8], 0.5))


class CapTests(unittest.TestCase):
    def test_identical_modes_have_zero_cap(self) -> None:
        marginal = np.asarray([0.2, 0.3, 0.5])
        transition = np.asarray([[0.5, 0.25, 0.25], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])
        cap = pure_mode_cap(
            two_state_process(
                (marginal, marginal.copy()),
                (transition, transition.copy()),
                (300, 700),
            )
        )
        self.assertAlmostEqual(cap["cap_f1"], 0.0, places=12)
        self.assertAlmostEqual(cap["cap_f3"], 0.0, places=12)

    def test_disjoint_modes_give_the_share_entropy(self) -> None:
        # State 0 only ever emits mark 0 and state 1 only mark 1.  The pooled
        # marginal is the share vector w, and the pooled rows reached by each
        # state are w as well, so both caps equal sum_s w_s log(1/w_s).
        marginals = (np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]))
        transitions = (
            np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        )
        shares = np.asarray([0.25, 0.75])
        cap = pure_mode_cap(two_state_process(marginals, transitions, (250, 750)))
        expected = float(-np.sum(shares * np.log(shares)))
        self.assertAlmostEqual(cap["cap_f1"], expected, places=12)
        self.assertAlmostEqual(cap["cap_f3"], expected, places=12)


class EndToEndTests(unittest.TestCase):
    def run_smoke(self, *extra: str) -> Path:
        SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(dir=SMOKE_ROOT))
        self.addCleanup(shutil.rmtree, directory)
        main(["--engineering-smoke", "--output-dir", str(directory), *extra])
        return directory

    def test_every_declared_output_is_written_and_hashed(self) -> None:
        config = json.loads((ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8"))
        self.assertEqual(set(config["outputs"]), set(OUTPUT_NAMES))
        directory = self.run_smoke(
            "--dwell-scales", "8", "12",
            "--replicates", "1",
            "--reference-length-cap", "250",
        )
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["engineering_smoke"])
        self.assertEqual(
            manifest["parameter_artifact"]["sha256"],
            config["parameter_artifact"]["sha256"],
        )
        self.assertEqual(manifest["parent_config_sha256"], config["parent_config_sha256"])
        for name in config["outputs"]:
            self.assertTrue((directory / name).is_file(), name)
            if name != "manifest.json":
                self.assertEqual(manifest["output_sha256"][name], sha256(directory / name))
        verdicts = json.loads(
            (directory / "predictions_check.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(verdicts), set(config["pre_registration"]["predictions"])
        )

    @unittest.skipUnless(
        os.environ.get("PUNCT_SLOW_TESTS"),
        "set PUNCT_SLOW_TESTS=1 (about two minutes); `make process-extended-smoke` "
        "runs the same check at full precision",
    )
    def test_multiplier_one_reproduces_the_v1_fitted_point(self) -> None:
        directory = self.run_smoke(
            "--dwell-scales", "1", "--replicates", "150", "--workers", "3"
        )
        with (directory / "extended_sweep_rho.csv").open(encoding="utf-8") as handle:
            averaged = next(
                row for row in csv.DictReader(handle) if row["seed"] == "mean"
            )
        # 450 chunks per size; the sd of G/90 is about 0.23, so 0.05 is > 4 se.
        for size, expected in V1_FITTED_PHI.items():
            self.assertAlmostEqual(float(averaged[f"phi_{size}"]), expected, delta=0.05)


if __name__ == "__main__":
    unittest.main()
